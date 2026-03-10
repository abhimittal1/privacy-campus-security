"""
camera_store.py – Camera registry: YAML config loader + SQLite-backed state table.

Overview
--------
Two complementary objects are provided:

``CameraRegistry`` (read-only, YAML-backed)
    Loads ``configs/cameras.yaml`` and exposes camera metadata — coordinates,
    door mappings, MQTT topic helpers.  Suitable for any code that only needs to
    *read* camera configuration.

``CameraStore`` (read-write, SQLite-backed)
    Persists the canonical camera table to a SQLite database alongside the
    embedding and attendance tables.  On first open it auto-seeds rows from
    ``configs/cameras.yaml`` when the table is empty.  Subsequent calls to
    ``upsert`` allow the dashboard / MQTT agent to update live state such as
    ``last_seen_utc`` and ``active``.

MQTT Topic Scheme
-----------------
::

    campus/<camera_id>/frame   – JPEG/PNG frame payload (bytes)
    campus/<camera_id>/sensor  – JSON sensor metadata (UTF-8 string)

Coordinates
-----------
All coordinates are stored as WGS-84 decimal-degree floats (lat, lon) so they
can be plotted directly on a Leaflet/OpenLayers campus overview map.

Schema (``cameras`` table)
--------------------------
::

    camera_id       TEXT  PRIMARY KEY
    name            TEXT  NOT NULL
    door_id         TEXT               – nullable
    lat             REAL  NOT NULL
    lon             REAL  NOT NULL
    active          INTEGER NOT NULL DEFAULT 1
    last_seen_utc   TEXT               – ISO-8601 UTC, updated by MQTT agent
    meta_json       TEXT               – arbitrary JSON bag for extensions

Usage
-----
::

    from src.privacy.camera_store import CameraRegistry, CameraStore

    # Read-only YAML view
    reg = CameraRegistry("configs/cameras.yaml")
    cam = reg.get("gate-north")
    print(cam.name, reg.frame_topic("gate-north"))

    # Read-write SQLite store
    store = CameraStore("data/embeddings.db")
    store.upsert_from_registry(reg)            # seed / refresh from YAML
    store.touch("gate-north")                  # update last_seen_utc = now()
    for row in store.list_all():
        print(row)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

# ─── DDL ──────────────────────────────────────────────────────────────────────

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id     TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        door_id       TEXT,
        lat           REAL NOT NULL DEFAULT 0.0,
        lon           REAL NOT NULL DEFAULT 0.0,
        active        INTEGER NOT NULL DEFAULT 1,
        last_seen_utc TEXT,
        meta_json     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cam_active ON cameras (active)",
    "CREATE INDEX IF NOT EXISTS idx_cam_door   ON cameras (door_id)",
]

# ─── Data class ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CameraEntry:
    """Immutable snapshot of one camera's configuration."""

    camera_id: str
    name: str
    door_id: str | None
    lat: float
    lon: float
    active: bool = True
    last_seen_utc: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ── MQTT helpers ──────────────────────────────────────────────────────────

    def frame_topic(self, prefix: str = "campus") -> str:
        """Return the MQTT topic for video frames, e.g. ``campus/gate-north/frame``."""
        return f"{prefix}/{self.camera_id}/frame"

    def sensor_topic(self, prefix: str = "campus") -> str:
        """Return the MQTT topic for sensor events, e.g. ``campus/gate-north/sensor``."""
        return f"{prefix}/{self.camera_id}/sensor"


# ─── YAML-backed registry ─────────────────────────────────────────────────────


class CameraRegistry:
    """
    Read-only view of the camera configuration loaded from ``cameras.yaml``.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file (default ``configs/cameras.yaml``).
    """

    def __init__(self, yaml_path: str | Path = "configs/cameras.yaml") -> None:
        self._path = Path(yaml_path)
        self._cameras: dict[str, CameraEntry] = {}
        self._mqtt_prefix: str = "campus"
        self._broker_host: str = "localhost"
        self._broker_port: int = 1883
        self._load()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for CameraRegistry. "
                "Install with: pip install pyyaml"
            ) from exc

        if not self._path.exists():
            raise FileNotFoundError(
                f"Camera configuration file not found: {self._path}"
            )

        with self._path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        mqtt_cfg = raw.get("mqtt", {})
        self._mqtt_prefix  = mqtt_cfg.get("topic_prefix", "campus")
        self._broker_host  = mqtt_cfg.get("broker_host", "localhost")
        self._broker_port  = int(mqtt_cfg.get("broker_port", 1883))

        for cam_raw in raw.get("cameras", []):
            cid = cam_raw["camera_id"]
            coords = cam_raw.get("coordinates") or {}
            entry = CameraEntry(
                camera_id     = cid,
                name          = cam_raw.get("name", cid),
                door_id       = cam_raw.get("door_id") or None,
                lat           = float(coords.get("lat", 0.0)),
                lon           = float(coords.get("lon", 0.0)),
                active        = bool(cam_raw.get("active", True)),
                meta          = {
                    k: v
                    for k, v in cam_raw.items()
                    if k not in {
                        "camera_id", "name", "door_id", "coordinates", "active"
                    }
                },
            )
            self._cameras[cid] = entry

        log.debug(
            "CameraRegistry loaded %d cameras from %s", len(self._cameras), self._path
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def broker_host(self) -> str:
        """MQTT broker hostname from the YAML ``mqtt.broker_host`` field."""
        return self._broker_host

    @property
    def broker_port(self) -> int:
        """MQTT broker port from the YAML ``mqtt.broker_port`` field."""
        return self._broker_port

    @property
    def topic_prefix(self) -> str:
        """MQTT topic prefix, e.g. ``campus``."""
        return self._mqtt_prefix

    def get(self, camera_id: str) -> CameraEntry:
        """
        Return the :class:`CameraEntry` for *camera_id*.

        Raises
        ------
        KeyError
            If *camera_id* is not present in the registry.
        """
        if camera_id not in self._cameras:
            raise KeyError(f"Camera '{camera_id}' not found in {self._path}")
        return self._cameras[camera_id]

    def list_all(self, active_only: bool = False) -> list[CameraEntry]:
        """Return all cameras, optionally filtering to active-only."""
        cams = list(self._cameras.values())
        if active_only:
            cams = [c for c in cams if c.active]
        return cams

    def __contains__(self, camera_id: object) -> bool:
        return camera_id in self._cameras

    def __len__(self) -> int:
        return len(self._cameras)

    def __iter__(self) -> Iterator[CameraEntry]:
        return iter(self._cameras.values())

    # ── MQTT topic helpers (mirror CameraEntry methods for convenience) ────────

    def frame_topic(self, camera_id: str) -> str:
        """Return ``<prefix>/<camera_id>/frame``."""
        return f"{self._mqtt_prefix}/{camera_id}/frame"

    def sensor_topic(self, camera_id: str) -> str:
        """Return ``<prefix>/<camera_id>/sensor``."""
        return f"{self._mqtt_prefix}/{camera_id}/sensor"


# ─── SQLite-backed store ──────────────────────────────────────────────────────


class CameraStore:
    """
    Persistent camera table stored in a SQLite database.

    The ``cameras`` table is created alongside the ``embeddings`` and
    ``attendance`` tables so all three can live in the same SQLite file without
    interference.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        for stmt in _DDL:
            self._conn.execute(stmt)
        self._conn.commit()

        log.debug("CameraStore initialised at %s", self._path)

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(self, entry: CameraEntry) -> None:
        """
        Insert *or* update a camera row.  Existing ``last_seen_utc`` is
        preserved when the new entry does not provide one.
        """
        self._conn.execute(
            """
            INSERT INTO cameras
                (camera_id, name, door_id, lat, lon, active, last_seen_utc, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                name          = excluded.name,
                door_id       = excluded.door_id,
                lat           = excluded.lat,
                lon           = excluded.lon,
                active        = excluded.active,
                last_seen_utc = COALESCE(excluded.last_seen_utc, cameras.last_seen_utc),
                meta_json     = excluded.meta_json
            """,
            (
                entry.camera_id,
                entry.name,
                entry.door_id,
                entry.lat,
                entry.lon,
                int(entry.active),
                entry.last_seen_utc,
                json.dumps(entry.meta) if entry.meta else None,
            ),
        )
        self._conn.commit()

    def upsert_from_registry(self, registry: CameraRegistry) -> int:
        """
        Seed / refresh the table from a :class:`CameraRegistry`.

        Returns the number of rows upserted.
        """
        for cam in registry:
            self.upsert(cam)
        n = len(registry)
        log.info("CameraStore: upserted %d rows from registry", n)
        return n

    def touch(self, camera_id: str, ts: str | None = None) -> None:
        """
        Update ``last_seen_utc`` for *camera_id* to *ts* (defaults to now).

        Creates the row with minimal data if it does not exist yet.
        """
        now = ts or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO cameras (camera_id, name, lat, lon, last_seen_utc)
            VALUES (?, ?, 0.0, 0.0, ?)
            ON CONFLICT(camera_id) DO UPDATE SET last_seen_utc = ?
            """,
            (camera_id, camera_id, now, now),
        )
        self._conn.commit()

    def set_active(self, camera_id: str, active: bool) -> None:
        """Enable or disable a camera by ``camera_id``."""
        self._conn.execute(
            "UPDATE cameras SET active = ? WHERE camera_id = ?",
            (int(active), camera_id),
        )
        self._conn.commit()

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, camera_id: str) -> CameraEntry | None:
        """Return the :class:`CameraEntry` for *camera_id*, or ``None``."""
        cur = self._conn.execute(
            "SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)
        )
        row = cur.fetchone()
        return _row_to_entry(row) if row else None

    def list_all(self, active_only: bool = False) -> list[CameraEntry]:
        """Return all camera rows, optionally filtered to active cameras."""
        sql = "SELECT * FROM cameras"
        if active_only:
            sql += " WHERE active = 1"
        sql += " ORDER BY camera_id"
        cur = self._conn.execute(sql)
        return [_row_to_entry(r) for r in cur.fetchall()]

    def count(self) -> int:
        """Total number of camera rows."""
        cur = self._conn.execute("SELECT COUNT(*) FROM cameras")
        return int(cur.fetchone()[0])

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the SQLite connection."""
        self._conn.close()

    def __repr__(self) -> str:  # pragma: no cover
        return f"CameraStore(path={self._path!r}, rows={self.count()})"


# ─── Helper ───────────────────────────────────────────────────────────────────


def _row_to_entry(row: sqlite3.Row) -> CameraEntry:
    meta: dict[str, Any] = {}
    raw_meta = row["meta_json"]
    if raw_meta:
        try:
            meta = json.loads(raw_meta)
        except (json.JSONDecodeError, TypeError):
            pass
    return CameraEntry(
        camera_id     = row["camera_id"],
        name          = row["name"],
        door_id       = row["door_id"],
        lat           = float(row["lat"]),
        lon           = float(row["lon"]),
        active        = bool(row["active"]),
        last_seen_utc = row["last_seen_utc"],
        meta          = meta,
    )
