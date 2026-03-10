-- migrations/attendance/0002_alerts_add_camera_index.sql
-- Adds a composite index that speeds up camera-specific alert lookups used
-- by the campus-map status endpoint (GET /api/camera/status).
--
-- Applied by: scripts/migrate.py
-- DB file:    data/attendance.db  (env: PCS_ATTENDANCE_DB)

CREATE INDEX IF NOT EXISTS idx_alert_camera_ts
    ON alerts (camera_id, timestamp_utc);
