-- migrations/audit/0001_initial_schema.sql
-- Creates the immutable audit log table with SQLite-enforced immutability.
--
-- Applied by: scripts/migrate.py
-- DB file:    data/audit.db  (env: PCS_AUDIT_DB)
--
-- IMPORTANT: BEFORE UPDATE and BEFORE DELETE triggers are attached to
-- audit_log so that no application code (or direct SQL access) can mutate
-- or remove rows.  The migration runner uses executescript() which commits
-- per-statement, so both the table and the triggers land atomically.

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    actor         TEXT,
    subject_id    TEXT,
    camera_id     TEXT,
    alert_id      INTEGER,
    detail        TEXT    -- JSON-encoded detail dict
);

CREATE INDEX IF NOT EXISTS idx_audit_ts         ON audit_log (timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS idx_audit_actor      ON audit_log (actor);
CREATE INDEX IF NOT EXISTS idx_audit_alert_id   ON audit_log (alert_id);

-- Prevent any UPDATE on audit rows (immutability enforcement)
CREATE TRIGGER IF NOT EXISTS audit_no_update
    BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log rows are immutable');
END;

-- Prevent any DELETE on audit rows (immutability enforcement)
CREATE TRIGGER IF NOT EXISTS audit_no_delete
    BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log rows are immutable');
END;
