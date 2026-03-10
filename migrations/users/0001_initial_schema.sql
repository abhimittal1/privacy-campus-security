-- migrations/users/0001_initial_schema.sql
-- Creates the users table used by the RBAC auth layer.
--
-- Applied by: scripts/migrate.py
-- DB file:    data/users.db  (env: PCS_AUTH_DB)

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,   -- bcrypt hash (cost factor 12)
    role          TEXT    NOT NULL DEFAULT 'viewer',
    created_at    TEXT    NOT NULL,   -- ISO-8601 UTC
    last_login    TEXT                -- ISO-8601 UTC, updated on each login
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_users_role     ON users (role);
