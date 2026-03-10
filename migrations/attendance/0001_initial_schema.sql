-- migrations/attendance/0001_initial_schema.sql
-- Creates all tables stored in the shared attendance database.
--
-- Applied by: scripts/migrate.py
-- DB file:    data/attendance.db  (env: PCS_ATTENDANCE_DB)
--
-- Tables defined here mirror the application's CREATE TABLE IF NOT EXISTS
-- statements and are safe to run against a pre-existing database.

-- ── attendance ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attendance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_key   TEXT    NOT NULL,
    group_id      TEXT    NOT NULL DEFAULT '',
    camera_id     TEXT    NOT NULL DEFAULT 'default',
    timestamp_utc TEXT    NOT NULL,
    similarity    REAL    NOT NULL,
    matched       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_att_subject ON attendance (subject_key);
CREATE INDEX IF NOT EXISTS idx_att_group   ON attendance (group_id);
CREATE INDEX IF NOT EXISTS idx_att_camera  ON attendance (camera_id);
CREATE INDEX IF NOT EXISTS idx_att_ts      ON attendance (timestamp_utc);

-- ── cameras ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cameras (
    camera_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    door_id       TEXT,
    lat           REAL NOT NULL DEFAULT 0.0,
    lon           REAL NOT NULL DEFAULT 0.0,
    active        INTEGER NOT NULL DEFAULT 1,
    last_seen_utc TEXT,
    meta_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_cam_active ON cameras (active);
CREATE INDEX IF NOT EXISTS idx_cam_door   ON cameras (door_id);

-- ── alerts ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type    TEXT    NOT NULL,
    camera_id     TEXT,
    door_id       TEXT,
    subject_key   TEXT,
    confidence    REAL    NOT NULL DEFAULT 1.0,
    timestamp_utc TEXT    NOT NULL,
    meta_json     TEXT,
    acknowledged  INTEGER NOT NULL DEFAULT 0,
    acked_by      TEXT,
    acked_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_alert_type ON alerts (alert_type);
CREATE INDEX IF NOT EXISTS idx_alert_door ON alerts (door_id);
CREATE INDEX IF NOT EXISTS idx_alert_ts   ON alerts (timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_alert_ack  ON alerts (acknowledged);

-- ── notifications ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS notifications (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id      INTEGER,
    alert_type    TEXT    NOT NULL,
    severity      TEXT    NOT NULL,
    channel       TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_attempt  TEXT,
    next_retry    TEXT,
    error_msg     TEXT,
    payload_json  TEXT,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_notif_status   ON notifications (status);
CREATE INDEX IF NOT EXISTS idx_notif_channel  ON notifications (channel);
CREATE INDEX IF NOT EXISTS idx_notif_alert_id ON notifications (alert_id);
CREATE INDEX IF NOT EXISTS idx_notif_created  ON notifications (created_at);

-- ── loitering_events ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS loitering_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id      TEXT    NOT NULL,
    subject_key    TEXT    NOT NULL,
    first_seen_utc TEXT    NOT NULL,
    detected_utc   TEXT    NOT NULL,
    dwell_s        REAL    NOT NULL DEFAULT 0,
    dismissed      INTEGER NOT NULL DEFAULT 0,
    action_by      TEXT,
    action_at      TEXT
);

-- ── left_object_events ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS left_object_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id      TEXT    NOT NULL,
    bbox_json      TEXT,
    first_seen_utc TEXT    NOT NULL,
    detected_utc   TEXT    NOT NULL,
    dwell_s        REAL    NOT NULL DEFAULT 0,
    snapshot_b64   TEXT,
    dismissed      INTEGER NOT NULL DEFAULT 0,
    confirmed      INTEGER NOT NULL DEFAULT 0,
    action_by      TEXT,
    action_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_lo_camera ON left_object_events (camera_id);
CREATE INDEX IF NOT EXISTS idx_lo_dets   ON left_object_events (detected_utc);
CREATE INDEX IF NOT EXISTS idx_lo_dis    ON left_object_events (dismissed);
