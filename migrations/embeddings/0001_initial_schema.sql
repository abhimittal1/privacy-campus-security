-- migrations/embeddings/0001_initial_schema.sql
-- Creates the core embeddings table used by EmbeddingStore.
--
-- Applied by: scripts/migrate.py
-- DB file:    data/embeddings.db  (env: PCS_STORE)

CREATE TABLE IF NOT EXISTS embeddings (
    hashed_id  TEXT    PRIMARY KEY,   -- HMAC-SHA256(subject_id, salt)
    embedding  BLOB    NOT NULL,       -- float32 bytes, optionally Fernet-encrypted
    dim        INTEGER NOT NULL        -- number of float32 values in the blob
);
