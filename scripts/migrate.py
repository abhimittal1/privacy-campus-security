#!/usr/bin/env python3
"""
Privacy Campus Security — lightweight multi-database migration runner.

Usage
-----
    python scripts/migrate.py [--db <name>] [--dry-run] [--status]

Each SQLite database managed by the application has its own sub-directory
under ``migrations/<db_name>/`` containing numbered SQL files:

    migrations/
        embeddings/
            0001_initial_schema.sql
            0002_add_index_on_dim.sql
        attendance/
            0001_initial_schema.sql
            0002_add_cameras_meta.sql
        audit/
            0001_initial_schema.sql
        users/
            0001_initial_schema.sql

A ``_schema_migrations`` table is created in **every** database the first time
the runner touches it.  Applied migration IDs are recorded there so the runner
is idempotent — re-running it is always safe.

Environment variables
---------------------
The same env vars the application uses to locate each database are respected:

    PCS_STORE          → embeddings DB   (default: data/embeddings.db)
    PCS_ATTENDANCE_DB  → attendance DB   (default: data/attendance.db)
    PCS_AUDIT_DB       → audit DB        (default: data/audit.db)
    PCS_AUTH_DB        → users DB        (default: data/users.db)

Exit codes
----------
    0  — all migrations applied (or nothing to do)
    1  — at least one migration failed
    2  — bad CLI arguments
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Database registry
# ---------------------------------------------------------------------------

# Each entry:  (db_name, env_var, default_path, migrations_subdir)
_DB_REGISTRY: list[tuple[str, str, str, str]] = [
    ("embeddings", "PCS_STORE",         "data/embeddings.db",  "embeddings"),
    ("attendance", "PCS_ATTENDANCE_DB", "data/attendance.db",  "attendance"),
    ("audit",      "PCS_AUDIT_DB",      "data/audit.db",       "audit"),
    ("users",      "PCS_AUTH_DB",       "data/users.db",       "users"),
]

# ---------------------------------------------------------------------------
# Tracking table DDL (injected into every managed database)
# ---------------------------------------------------------------------------

_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    id          TEXT    PRIMARY KEY,   -- e.g. "0001_initial_schema"
    applied_at  TEXT    NOT NULL,      -- ISO-8601 UTC
    checksum    TEXT                   -- SHA-256 of the SQL file (informational)
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIGRATIONS_ROOT = Path(__file__).parent.parent / "migrations"
_VERSION_RE = re.compile(r"^(\d{4})_.*\.sql$", re.IGNORECASE)


def _resolve_db_path(env_var: str, default: str) -> Path:
    return Path(os.environ.get(env_var, default))


def _open_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_tracking_table(conn: sqlite3.Connection) -> None:
    conn.executescript(_TRACKING_DDL)
    conn.commit()


def _applied_ids(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT id FROM _schema_migrations").fetchall()
    return {r[0] for r in rows}


def _pending_migrations(subdir: str, applied: set[str]) -> list[Path]:
    """Return sorted list of migration files not yet applied."""
    mig_dir = MIGRATIONS_ROOT / subdir
    if not mig_dir.is_dir():
        return []
    files = sorted(
        f for f in mig_dir.glob("*.sql") if _VERSION_RE.match(f.name)
    )
    return [f for f in files if f.stem not in applied]


def _checksum(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_migration(
    conn: sqlite3.Connection,
    sql_file: Path,
    *,
    dry_run: bool = False,
) -> None:
    from datetime import datetime, timezone

    migration_id = sql_file.stem
    sql = sql_file.read_text(encoding="utf-8")
    chk = _checksum(sql_file)
    utcnow = datetime.now(timezone.utc).isoformat()

    if dry_run:
        print(f"  [DRY-RUN]  Would apply: {migration_id}")
        return

    # Run the migration SQL, then record it — both in the same transaction
    # so a partial failure leaves the tracking table clean.
    try:
        conn.executescript(sql)          # executescript commits implicitly
        conn.execute(
            "INSERT INTO _schema_migrations (id, applied_at, checksum) VALUES (?, ?, ?)",
            (migration_id, utcnow, chk),
        )
        conn.commit()
        print(f"  [OK]   {migration_id}")
    except sqlite3.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Migration {migration_id} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Per-database migrate
# ---------------------------------------------------------------------------

def migrate_db(
    db_name: str,
    env_var: str,
    default_path: str,
    mig_subdir: str,
    *,
    dry_run: bool = False,
) -> bool:
    """Apply all pending migrations for one database.  Returns True on success."""
    db_path = _resolve_db_path(env_var, default_path)
    mig_dir = MIGRATIONS_ROOT / mig_subdir

    if not mig_dir.is_dir():
        print(f"\n[{db_name}] No migrations directory at {mig_dir} — skipping.")
        return True

    print(f"\n[{db_name}]  db={db_path}")

    conn = _open_conn(db_path)
    _ensure_tracking_table(conn)

    applied = _applied_ids(conn)
    pending = _pending_migrations(mig_subdir, applied)

    if not pending:
        print("  Nothing to do (all migrations applied).")
        conn.close()
        return True

    ok = True
    for sql_file in pending:
        try:
            _apply_migration(conn, sql_file, dry_run=dry_run)
        except RuntimeError as exc:
            print(f"  [FAIL] {exc}", file=sys.stderr)
            ok = False
            break          # stop at first failure within this DB

    conn.close()
    return ok


# ---------------------------------------------------------------------------
# Status reporter
# ---------------------------------------------------------------------------

def status_db(db_name: str, env_var: str, default_path: str, mig_subdir: str) -> None:
    db_path = _resolve_db_path(env_var, default_path)
    mig_dir = MIGRATIONS_ROOT / mig_subdir

    print(f"\n[{db_name}]  db={db_path}")

    if not db_path.exists():
        print("  Database file does not exist yet.")
    else:
        conn = _open_conn(db_path)
        _ensure_tracking_table(conn)
        applied = _applied_ids(conn)
        if applied:
            for mid in sorted(applied):
                row = conn.execute(
                    "SELECT applied_at FROM _schema_migrations WHERE id = ?", (mid,)
                ).fetchone()
                print(f"  [applied]  {mid}  ({row[0] if row else '?'})")
        else:
            print("  No migrations recorded.")
        conn.close()

    if mig_dir.is_dir():
        all_files = sorted(f for f in mig_dir.glob("*.sql") if _VERSION_RE.match(f.name))
        if not db_path.exists():
            pending_names = [f.stem for f in all_files]
        else:
            conn2 = _open_conn(db_path)
            _ensure_tracking_table(conn2)
            applied2 = _applied_ids(conn2)
            conn2.close()
            pending_names = [f.stem for f in all_files if f.stem not in applied2]
        for name in pending_names:
            print(f"  [pending]  {name}")
    else:
        print(f"  migrations directory not found: {mig_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Apply SQLite schema migrations for Privacy Campus Security.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--db",
        metavar="NAME",
        help="Migrate only this database (embeddings | attendance | audit | users). "
             "Default: migrate all.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be applied but do not modify any database.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print applied / pending status for each database and exit.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Resolve which databases to touch
    if args.db:
        targets = [row for row in _DB_REGISTRY if row[0] == args.db]
        if not targets:
            names = ", ".join(r[0] for r in _DB_REGISTRY)
            print(f"Unknown database '{args.db}'.  Valid names: {names}", file=sys.stderr)
            return 2
    else:
        targets = list(_DB_REGISTRY)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Privacy Campus Security — DB Migration Runner           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.dry_run:
        print("  *** DRY-RUN mode — no changes will be written ***")

    if args.status:
        for name, env_var, default, subdir in targets:
            status_db(name, env_var, default, subdir)
        return 0

    all_ok = True
    for name, env_var, default, subdir in targets:
        ok = migrate_db(name, env_var, default, subdir, dry_run=args.dry_run)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("✓  All migrations complete.")
        return 0
    else:
        print("✗  One or more migrations failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
