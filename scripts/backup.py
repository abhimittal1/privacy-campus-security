#!/usr/bin/env python3
"""
Privacy Campus Security — SQLite backup + rotation script.

Usage
-----
    python scripts/backup.py [options]

Creates a timestamped, optionally compressed backup of every SQLite database
managed by the application.  Backups land in a configurable directory and are
automatically rotated so only the last N days' worth are retained.  When the
optional ``boto3`` package is installed and an S3 bucket is configured, backups
are also uploaded to S3 immediately after creation.

Backup directory layout
-----------------------
    data/backups/
        2026-03-01/
            embeddings_20260301T120000Z.db.gz
            attendance_20260301T120000Z.db.gz
            audit_20260301T120000Z.db.gz
            users_20260301T120000Z.db.gz
        2026-03-02/
            embeddings_20260302T120000Z.db.gz
            ...

S3 key layout (when enabled)
-----------------------------
    <PCS_BACKUP_S3_PREFIX>/<db_name>/<date>/<filename>
    e.g. pcs-backups/embeddings/2026-03-01/embeddings_20260301T120000Z.db.gz

Environment variables
---------------------
    PCS_STORE            Path to embeddings SQLite (default: data/embeddings.db)
    PCS_ATTENDANCE_DB    Path to attendance SQLite (default: data/attendance.db)
    PCS_AUDIT_DB         Path to audit SQLite      (default: data/audit.db)
    PCS_AUTH_DB          Path to users SQLite       (default: data/users.db)
    PCS_BACKUP_DIR       Root backup directory      (default: data/backups)
    PCS_BACKUP_RETAIN    Days of backups to keep    (default: 7)
    PCS_BACKUP_COMPRESS  1 = gzip compress (default: 1)
    PCS_BACKUP_S3_BUCKET S3 bucket name (S3 upload disabled when blank)
    PCS_BACKUP_S3_PREFIX S3 key prefix   (default: pcs-backups)

Exit codes
----------
    0 — all backups created (and uploaded) successfully
    1 — at least one backup failed
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("pcs.backup")

# ---------------------------------------------------------------------------
# Database registry (same as migrate.py)
# ---------------------------------------------------------------------------

_DB_REGISTRY: list[tuple[str, str, str]] = [
    ("embeddings", "PCS_STORE",         "data/embeddings.db"),
    ("attendance", "PCS_ATTENDANCE_DB", "data/attendance.db"),
    ("audit",      "PCS_AUDIT_DB",      "data/audit.db"),
    ("users",      "PCS_AUTH_DB",       "data/users.db"),
]

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key, "1" if default else "0").strip().lower()
    return val in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Backup helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _backup_filename(db_name: str, ts: datetime, compress: bool) -> str:
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    ext = ".db.gz" if compress else ".db"
    return f"{db_name}_{stamp}{ext}"


def _sqlite_backup(src_path: Path, dst_path: Path) -> None:
    """
    Use the SQLite online-backup API to create a consistent hot copy.

    This is equivalent to ``sqlite3 src .backup dst`` but works while the
    source database is being written to — it's safe to run against a live
    server with WAL mode.
    """
    src_conn = sqlite3.connect(str(src_path))
    dst_conn = sqlite3.connect(str(dst_path))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()


def _compress(raw_path: Path, gz_path: Path) -> None:
    """Gzip-compress *raw_path* into *gz_path* then remove the raw file."""
    with raw_path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    raw_path.unlink()


# ---------------------------------------------------------------------------
# S3 upload (optional — requires boto3)
# ---------------------------------------------------------------------------

def _s3_upload(local_path: Path, bucket: str, s3_key: str) -> bool:
    """Upload *local_path* to S3.  Returns True on success, False on error."""
    try:
        import boto3  # type: ignore[import]
    except ImportError:
        log.warning(
            "boto3 is not installed — skipping S3 upload.  "
            "Run: pip install boto3"
        )
        return False

    try:
        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), bucket, s3_key)
        log.info("Uploaded to s3://%s/%s", bucket, s3_key)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("S3 upload failed for %s: %s", local_path.name, exc)
        return False


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def _rotate_backups(backup_root: Path, retain_days: int) -> None:
    """
    Delete date-subdirectories older than *retain_days*.

    Directory names must match ``YYYY-MM-DD`` format; others are left alone.
    """
    if retain_days <= 0:
        return

    cutoff = _utcnow().date() - timedelta(days=retain_days)
    removed = 0

    for d in backup_root.iterdir():
        if not d.is_dir():
            continue
        try:
            date = datetime.strptime(d.name, "%Y-%m-%d").date()
        except ValueError:
            continue

        if date < cutoff:
            shutil.rmtree(d)
            log.info("Rotated old backup directory: %s", d)
            removed += 1

    if removed:
        log.info("Rotation complete — removed %d old backup director%s.",
                 removed, "y" if removed == 1 else "ies")
    else:
        log.debug("Rotation: nothing to remove (retain_days=%d).", retain_days)


# ---------------------------------------------------------------------------
# Per-database backup
# ---------------------------------------------------------------------------

def backup_db(
    db_name: str,
    src_path: Path,
    backup_root: Path,
    ts: datetime,
    *,
    compress: bool = True,
    s3_bucket: str = "",
    s3_prefix: str = "pcs-backups",
) -> bool:
    """
    Back up *src_path* into *backup_root/<date>/<filename>*.

    Returns True on success.
    """
    if not src_path.exists():
        log.warning("[%s] Database not found at %s — skipping.", db_name, src_path)
        return True  # not an error; DB may not have been created yet

    date_str = ts.strftime("%Y-%m-%d")
    day_dir = backup_root / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    filename = _backup_filename(db_name, ts, compress=False)   # raw first
    raw_dst = day_dir / filename
    final_dst = day_dir / (_backup_filename(db_name, ts, compress=compress))

    log.info("[%s] Backing up %s …", db_name, src_path)
    try:
        _sqlite_backup(src_path, raw_dst)
        log.debug("[%s] Hot copy written to %s (%d bytes)",
                  db_name, raw_dst, raw_dst.stat().st_size)
    except Exception as exc:  # noqa: BLE001
        log.error("[%s] Backup failed: %s", db_name, exc)
        return False

    if compress:
        try:
            _compress(raw_dst, final_dst)
            log.info("[%s] Compressed → %s (%d bytes)",
                     db_name, final_dst.name, final_dst.stat().st_size)
        except Exception as exc:  # noqa: BLE001
            log.error("[%s] Compression failed: %s", db_name, exc)
            return False
    else:
        final_dst = raw_dst
        log.info("[%s] Saved → %s", db_name, final_dst.name)

    # Optional S3 upload
    if s3_bucket:
        s3_key = f"{s3_prefix}/{db_name}/{date_str}/{final_dst.name}"
        _s3_upload(final_dst, s3_bucket, s3_key)      # failure is non-fatal

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create timestamped SQLite backups and optionally upload to S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        metavar="NAME",
        help="Back up only this database (embeddings | attendance | audit | users). "
             "Default: all.",
    )
    parser.add_argument(
        "--backup-dir",
        metavar="PATH",
        default=_env("PCS_BACKUP_DIR", "data/backups"),
        help="Root directory for backup files (default: data/backups).",
    )
    parser.add_argument(
        "--retain",
        type=int,
        default=_env_int("PCS_BACKUP_RETAIN", 7),
        help="Days of backups to keep (0 = no rotation).  Default: 7.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        default=not _env_bool("PCS_BACKUP_COMPRESS", True),
        help="Skip gzip compression.",
    )
    parser.add_argument(
        "--s3-bucket",
        metavar="BUCKET",
        default=_env("PCS_BACKUP_S3_BUCKET", ""),
        help="S3 bucket for upload (requires boto3).  Default: disabled.",
    )
    parser.add_argument(
        "--s3-prefix",
        metavar="PREFIX",
        default=_env("PCS_BACKUP_S3_PREFIX", "pcs-backups"),
        help="S3 key prefix.  Default: pcs-backups.",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Skip old-backup rotation even if --retain is set.",
    )
    args = parser.parse_args(argv)

    backup_root = Path(args.backup_dir)
    compress = not args.no_compress
    ts = _utcnow()

    # Resolve databases to back up
    if args.db:
        targets = [(n, e, d) for n, e, d in _DB_REGISTRY if n == args.db]
        if not targets:
            names = ", ".join(n for n, _, _ in _DB_REGISTRY)
            print(f"Unknown database '{args.db}'.  Valid: {names}", file=sys.stderr)
            return 1
    else:
        targets = list(_DB_REGISTRY)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Privacy Campus Security — Backup                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Timestamp : {ts.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"  Directory : {backup_root}")
    print(f"  Compress  : {'yes (gzip)' if compress else 'no'}")
    print(f"  Retain    : {args.retain} days")
    if args.s3_bucket:
        print(f"  S3 Bucket : {args.s3_bucket}/{args.s3_prefix}")
    print()

    all_ok = True
    for db_name, env_var, default_path in targets:
        src_path = Path(os.environ.get(env_var, default_path))
        ok = backup_db(
            db_name,
            src_path,
            backup_root,
            ts,
            compress=compress,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
        )
        all_ok = all_ok and ok

    # Daily rotation
    if not args.no_rotate and args.retain > 0:
        print()
        log.info("Running backup rotation (retain %d days) …", args.retain)
        _rotate_backups(backup_root, args.retain)

    print()
    if all_ok:
        print("✓  Backup complete.")
        return 0
    else:
        print("✗  One or more backups failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
