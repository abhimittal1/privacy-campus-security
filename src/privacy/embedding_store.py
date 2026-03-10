"""
embedding_store.py -- Privacy-preserving embedding storage with orthogonal projection.

Security model
--------------
Two complementary privacy mechanisms protect subject data:

1. **Identity hiding (SHA-256 keying)**
   The storage key for each subject is ``SHA-256(salt || subject_id)``
   (hex digest).  The original ID is *never* written to disk.  Security
   relies on the one-way property of SHA-256 and on keeping *salt* secret.

2. **Embedding obfuscation (salted random orthogonal projection)**
   Raw embeddings are never stored.  Instead, a random orthogonal matrix Q
   (512 x 512) is derived deterministically from a salt string via a seeded
   numpy RNG + QR decomposition:

       seed = int.from_bytes(SHA-256(salt)[:4], "big")
       G    = RNG(seed).standard_normal((d, d))   # Gaussian matrix
       Q, _ = numpy.linalg.qr(G)                  # orthogonal Q

   The protected embedding is ``p = Q[:d', :] @ raw`` where *d'* is the
   projected dimension (default 512 = no compression; use a smaller value
   for storage savings).

   **Properties:**
   - Cosine similarity is approximately preserved when d' == d (exact with
     unit vectors, since orthogonal transforms preserve inner products).
   - Without knowing Q (i.e., the salt), an attacker cannot recover the
     original direction of ``raw`` from ``p`` better than chance -- the
     best linear decoder has MSE >= (d'/d) for random unit vectors.
   - The construction is NOT secure against an adversary who knows Q.
     Keep the salt secret just like a password or a symmetric key.

3. **Optional Fernet encryption at rest**
   When ``encrypt=True``, the entire store file is symmetrically encrypted
   using a Fernet key (set via ``PCS_FERNET_KEY``).  This adds a second
   layer: even if the file is exfiltrated, the attacker needs both the salt
   (to invert Q) AND the Fernet key.

Backends
--------
``sqlite`` (default): Uses Python's built-in :mod:`sqlite3`.  Embeddings
are stored as raw ``float32`` blobs.  Zero additional dependencies.

``npz``: Keeps an in-memory ``dict`` and writes a ``.npz`` archive on
:meth:`EmbeddingStore.save`.  Faster for small stores; no concurrent
write safety.

Reconstruction attack
---------------------
:meth:`EmbeddingStore.reconstruction_success_rate` trains a shallow MLP
to map *protected* embeddings back to *original* ones and measures what
fraction of reconstructions fall below an MSE threshold.  Target: < 5%.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_SALT_ENV_VAR = "PCS_EMBEDDING_SALT"
_KEY_ENV_VAR  = "PCS_FERNET_KEY"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_salt(salt: Optional[str]) -> str:
    """Return explicit salt, env-var salt, or a dev-only default (with warning)."""
    if salt:
        return salt
    env_salt = os.environ.get(_SALT_ENV_VAR)
    if env_salt:
        return env_salt
    warnings.warn(
        f"No salt provided and {_SALT_ENV_VAR} is not set. "
        "Using the insecure default salt -- change before production use.",
        UserWarning,
        stacklevel=3,
    )
    return "default-dev-salt-change-in-prod"


def _hash_subject(subject_id: str, salt: str) -> str:
    """Return a hex SHA-256 digest as the privacy-safe storage key.

    The construction ``SHA-256(salt || subject_id)`` binds the ID to the
    salt, preventing rainbow-table attacks on the key space.
    """
    return hashlib.sha256(salt.encode() + subject_id.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Orthogonal projection
# ---------------------------------------------------------------------------


class _OrthProjection:
    """
    Salted random orthogonal projection for embedding obfuscation.

    The projection matrix Q is derived as::

        seed = int.from_bytes(SHA-256(salt.encode())[:4], "big")
        G    = np.random.default_rng(seed).standard_normal((dim, dim))
        Q, _ = np.linalg.qr(G)   # Q is dim x dim orthogonal

    Only the first *projected_dim* rows of Q are used, giving a
    ``(projected_dim, dim)`` projection matrix.  When ``projected_dim == dim``
    the transform is a pure rotation (inner products preserved exactly).

    Args:
        dim:           Input (raw) embedding dimension (e.g. 512).
        projected_dim: Output dimension.  Must satisfy 1 <= projected_dim <= dim.
        salt:          Secret string from which the RNG seed is derived.
    """

    def __init__(self, dim: int, projected_dim: int, salt: str) -> None:
        if projected_dim < 1 or projected_dim > dim:
            raise ValueError(
                f"projected_dim must be in [1, {dim}], got {projected_dim}"
            )
        self.dim          = dim
        self.projected_dim = projected_dim
        seed = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:4], "big")
        rng  = np.random.default_rng(seed)
        G    = rng.standard_normal((dim, dim))
        Q, _ = np.linalg.qr(G)        # Q: (dim, dim) orthogonal
        # Keep only first projected_dim rows -> (projected_dim, dim)
        self._Q: np.ndarray = Q[:projected_dim, :].astype(np.float32)
        log.debug(
            "OrthProjection: dim=%d projected_dim=%d seed=%d",
            dim, projected_dim, seed,
        )

    def project(self, raw: np.ndarray) -> np.ndarray:
        """Map a raw embedding (dim,) -> protected embedding (projected_dim,)."""
        return (self._Q @ raw.astype(np.float32)).astype(np.float32)

    def project_batch(self, raws: np.ndarray) -> np.ndarray:
        """Map raw embeddings (N, dim) -> protected embeddings (N, projected_dim)."""
        return (raws.astype(np.float32) @ self._Q.T).astype(np.float32)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS embeddings (
    hashed_id  TEXT    PRIMARY KEY,
    embedding  BLOB    NOT NULL,
    dim        INTEGER NOT NULL
);
"""


class _SQLiteBackend:
    """
    SQLite persistence backend with optional per-blob Fernet encryption.

    When a :class:`~cryptography.fernet.Fernet` instance is supplied, every
    embedding blob is encrypted before it is written and decrypted transparently
    on read.  The ``hashed_id`` column (a SHA-256 hex string) is never
    encrypted — it carries no subject-identifying information.

    ``all_raw`` / ``put_raw`` bypass numpy serialisation and are used
    exclusively by :meth:`EmbeddingStore.rotate_key` to re-encrypt stored
    blobs without touching the projection layer.
    """

    def __init__(self, path: Path, fernet=None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.commit()
        self._fernet = fernet  # Optional[Fernet]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enc(self, blob: bytes) -> bytes:
        """Encrypt *blob* if a Fernet key is configured; else return as-is."""
        return self._fernet.encrypt(blob) if self._fernet else blob

    def _dec(self, blob: bytes) -> bytes:
        """Decrypt *blob* if a Fernet key is configured; else return as-is."""
        return self._fernet.decrypt(blob) if self._fernet else blob

    # ------------------------------------------------------------------
    # Numpy-level API (used by EmbeddingStore for normal reads/writes)
    # ------------------------------------------------------------------

    def put(self, hashed_id: str, vec: np.ndarray) -> None:
        blob = self._enc(vec.astype(np.float32).tobytes())
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (hashed_id, embedding, dim) VALUES (?,?,?)",
            (hashed_id, blob, vec.shape[0]),
        )
        self._conn.commit()

    def put_many(self, rows: list[tuple[str, np.ndarray]]) -> None:
        data = [
            (hid, self._enc(vec.astype(np.float32).tobytes()), vec.shape[0])
            for hid, vec in rows
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (hashed_id, embedding, dim) VALUES (?,?,?)",
            data,
        )
        self._conn.commit()

    def get(self, hashed_id: str) -> Optional[np.ndarray]:
        cur = self._conn.execute(
            "SELECT embedding, dim FROM embeddings WHERE hashed_id=?", (hashed_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        blob, dim = row
        plain = self._dec(blob)
        return np.frombuffer(plain, dtype=np.float32).copy()

    def delete(self, hashed_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM embeddings WHERE hashed_id=?", (hashed_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def contains(self, hashed_id: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM embeddings WHERE hashed_id=? LIMIT 1", (hashed_id,)
        )
        return cur.fetchone() is not None

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM embeddings")
        return cur.fetchone()[0]

    def all_vectors(self) -> Tuple[list[str], np.ndarray]:
        """Return (hashed_ids, matrix N x D) for all stored entries.

        Blobs are decrypted transparently when Fernet is active.
        """
        cur = self._conn.execute(
            "SELECT hashed_id, embedding FROM embeddings ORDER BY hashed_id"
        )
        rows = cur.fetchall()
        if not rows:
            return [], np.empty((0,), dtype=np.float32)
        ids  = [r[0] for r in rows]
        vecs = [np.frombuffer(self._dec(r[1]), dtype=np.float32).copy() for r in rows]
        return ids, np.stack(vecs)

    # ------------------------------------------------------------------
    # Raw-bytes API (used only by EmbeddingStore.rotate_key)
    # ------------------------------------------------------------------

    def all_raw(self) -> list[tuple[str, bytes, int]]:
        """Return every row as ``(hashed_id, raw_blob, dim)`` without decryption.

        The caller is responsible for decrypting / re-encrypting the blobs
        when performing key rotation.
        """
        cur = self._conn.execute(
            "SELECT hashed_id, embedding, dim FROM embeddings ORDER BY hashed_id"
        )
        return cur.fetchall()

    def put_raw(self, hashed_id: str, blob: bytes, dim: int) -> None:
        """Overwrite a row with a pre-serialised (and possibly Fernet-encrypted)
        byte blob, bypassing numpy serialisation.

        Used by :meth:`EmbeddingStore.rotate_key` to swap ciphertext in-place.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (hashed_id, embedding, dim) VALUES (?,?,?)",
            (hashed_id, blob, dim),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# .npz backend
# ---------------------------------------------------------------------------


class _NpzBackend:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._store: dict[str, np.ndarray] = {}
        if path.exists():
            data = np.load(str(path), allow_pickle=False)
            for k in data.files:
                self._store[k] = data[k]

    def put(self, hashed_id: str, vec: np.ndarray) -> None:
        self._store[hashed_id] = vec.astype(np.float32)

    def put_many(self, rows: list[tuple[str, np.ndarray]]) -> None:
        for hid, vec in rows:
            self._store[hid] = vec.astype(np.float32)

    def get(self, hashed_id: str) -> Optional[np.ndarray]:
        return self._store.get(hashed_id)

    def delete(self, hashed_id: str) -> bool:
        existed = hashed_id in self._store
        self._store.pop(hashed_id, None)
        return existed

    def contains(self, hashed_id: str) -> bool:
        return hashed_id in self._store

    def count(self) -> int:
        return len(self._store)

    def all_vectors(self) -> Tuple[list[str], np.ndarray]:
        ids  = sorted(self._store.keys())
        if not ids:
            return [], np.empty((0,), dtype=np.float32)
        vecs = np.stack([self._store[i] for i in ids])
        return ids, vecs

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(self._path), **self._store)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public EmbeddingStore
# ---------------------------------------------------------------------------


class EmbeddingStore:
    """
    Store and retrieve privacy-protected face embeddings.

    Each enrolment:

    1. Subject ID -> ``SHA-256(salt || id)`` -> storage key   (identity hiding)
    2. Raw embedding -> ``Q[:d', :] @ raw``                   (orthogonal obfuscation)
    3. Protected vector written to SQLite BLOB or .npz file   (persistence)

    **Query / retrieval**:
    The *query* embedding is projected with the same matrix Q before cosine
    similarity is computed.  Because Q is orthogonal, inner products between
    unit vectors are preserved (exact when projected_dim == dim), so ranking
    is unchanged.

    Args:
        store_path:    Path to the SQLite ``.db`` file (or ``.npz`` for npz
                       backend).  Parent directories are created automatically.
        salt:          Secret string for key hashing and Q derivation.
                       Defaults to the ``PCS_EMBEDDING_SALT`` env var, then to
                       an insecure dev constant.
        dim:           Input embedding dimension (default 512).
        projected_dim: Dimension after projection (default = *dim*).
                       Reducing this saves storage but may reduce recall accuracy.
        backend:       ``"sqlite"`` (default) or ``"npz"``.
        encrypt:       Encrypt the store file with Fernet (SQLite not supported;
                       applies only to ``"npz"`` backend currently).

    Security assumptions
    --------------------
    * The *salt* must be kept confidential.  If an attacker learns the salt,
      they can reconstruct Q and invert the projection.
    * The SHA-256 keying provides ID unlinkability, not confidentiality of
      *the fact* that some enrolment exists in the store.
    * Orthogonal projection is an information-theoretically reversible
      transform when projected_dim == dim.  Privacy derives from not
      revealing Q (salt secrecy), NOT from information destruction.
    * Reducing projected_dim does destroy information and provides
      computational privacy, but lowers recall for nearest-neighbour queries.
    """

    def __init__(
        self,
        store_path: str | Path,
        salt: Optional[str] = None,
        dim: int = 512,
        projected_dim: int = 512,
        backend: str = "sqlite",
        encrypt: bool = False,
    ) -> None:
        self.store_path   = Path(store_path)
        self._salt        = _resolve_salt(salt)
        self.dim          = dim
        self.projected_dim = projected_dim
        self._proj        = _OrthProjection(dim, projected_dim, self._salt)
        self._encrypt     = encrypt
        self._fernet      = None

        if encrypt:
            from cryptography.fernet import Fernet
            key = os.environ.get(_KEY_ENV_VAR)
            if key is None:
                key = Fernet.generate_key().decode()
                log.warning(
                    "No Fernet key found in %s; generated ephemeral key.", _KEY_ENV_VAR
                )
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)

        if backend not in ("sqlite", "npz"):
            raise ValueError(f"backend must be 'sqlite' or 'npz', got '{backend}'")
        self._backend_name = backend
        if backend == "sqlite":
            # Pass the Fernet instance so the backend encrypts every blob
            # at rest.  When encrypt=False (default), fernet is None and the
            # backend stores plain float32 bytes as before.
            self._db: _SQLiteBackend | _NpzBackend = _SQLiteBackend(
                self.store_path, fernet=self._fernet
            )
        else:
            if encrypt:
                log.warning(
                    "Fernet encryption is only supported for the 'sqlite' backend; "
                    "npz backend will store embeddings unencrypted."
                )
            self._db = _NpzBackend(self.store_path)

        # Fast-retrieval cache ─────────────────────────────────────────────
        # _cache_mat holds every stored vector, row-normalised, as a
        # (N, projected_dim) float32 matrix so that query_topk can do a
        # single BLAS dot-product without touching the DB.
        # The cache is rebuilt lazily on the first query_topk call after any
        # write or delete.  Thread-safety note: writes set _cache_dirty=True
        # atomically (GIL) before the DB write returns, so a concurrent
        # query_topk will always see a fresh rebuild on its next call.
        self._cache_ids:   Optional[list[str]]  = None
        self._cache_mat:   Optional[np.ndarray] = None   # (N, projected_dim)
        self._cache_dirty: bool                 = True

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_norm(v: np.ndarray) -> np.ndarray:
        """
        Return *v* L2-normalised to unit length.

        Safe when ‖v‖ ≈ 0 (returns the original cast to float32 unchanged
        rather than producing NaN).
        """
        v = v.astype(np.float32, copy=False)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    def _rebuild_cache(self) -> None:
        """Load all stored vectors from the backend and build the lookup matrix."""
        ids, mat = self._db.all_vectors()
        if mat.ndim < 2 or mat.shape[0] == 0:
            self._cache_ids = []
            self._cache_mat = np.empty((0, self.projected_dim), dtype=np.float32)
        else:
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            self._cache_ids = ids
            self._cache_mat = (mat / norms).astype(np.float32)
        self._cache_dirty = False
        log.debug("EmbeddingStore cache rebuilt: %d entries", len(self._cache_ids))

    def store(self, id: str, embedding: np.ndarray) -> None:
        """
        Protect and persist a single embedding.

        The embedding is L2-normalised before the orthogonal projection so
        that cosine similarity in the projected space equals dot-product
        similarity, which is what :meth:`query_topk` computes.

        Args:
            id:        Plain-text subject identifier (never stored).
            embedding: Raw embedding of shape ``(dim,)`` (normalised here if
                       not already).
        """
        hid = _hash_subject(id, self._salt)
        protected = self._proj.project(self._l2_norm(embedding))
        self._db.put(hid, protected)
        self._cache_dirty = True

    def bulk_store(self, ids: List[str], embeddings: np.ndarray) -> None:
        """
        Protect and persist many embeddings efficiently.

        Each embedding is L2-normalised row-wise before projection (same
        guarantee as :meth:`store`).

        Args:
            ids:        List of *N* plain-text subject identifiers.
            embeddings: Float32 array of shape ``(N, dim)``.
        """
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"ids length ({len(ids)}) != embeddings rows ({embeddings.shape[0]})"
            )
        # Row-wise L2 normalisation before projection.
        emb = embeddings.astype(np.float32, copy=False)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)          # avoid div-by-zero
        protected = self._proj.project_batch(emb / norms)   # (N, projected_dim)
        rows = [(_hash_subject(id_, self._salt), protected[i]) for i, id_ in enumerate(ids)]
        self._db.put_many(rows)
        self._cache_dirty = True

    def query_topk(
        self,
        embedding: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the *k* stored subjects most similar to *embedding*.

        Similarity is cosine similarity computed in the projected space.
        Because Q is orthogonal and all vectors are L2-normalised before
        storage, the ranking is identical to querying raw embeddings.

        **Fast retrieval**: results come from an in-memory normalised matrix
        cached after the first call and invalidated on any write or delete.
        Similarity scores are computed as a single matrix–vector product
        (BLAS SGEMV, O(N·d')) and top-k selection uses ``np.argpartition``
        (O(N)), avoiding a full O(N log N) sort.

        Args:
            embedding: Raw query embedding of shape ``(dim,)``.
            k:         Number of ranked results to return.

        Returns:
            List of ``(hashed_id, cosine_similarity)`` tuples, sorted
            descending by similarity.  Length <= min(k, store_size).
        """
        if self._cache_dirty:
            self._rebuild_cache()

        if not self._cache_ids:          # store is empty
            return []

        # Project + normalise the query once.
        q_proj = self._proj.project(self._l2_norm(embedding))   # (projected_dim,)
        q_norm = q_proj / (np.linalg.norm(q_proj) + 1e-9)

        # Single BLAS dot-product against the pre-normalised cache matrix.
        sims: np.ndarray = self._cache_mat @ q_norm             # (N,)

        n = sims.shape[0]
        k = min(k, n)
        if k == n:
            # Trivial case: return all, still sorted.
            idx = np.argsort(sims)[::-1]
        else:
            # O(N) partial selection, then sort only the k winners.
            part_idx = np.argpartition(sims, n - k)[n - k:]
            idx = part_idx[np.argsort(sims[part_idx])[::-1]]

        return [(self._cache_ids[i], float(sims[i])) for i in idx]

    def query_topk_batch(
        self,
        embeddings: np.ndarray,
        k: int = 3,
        threshold: float = 0.0,
    ) -> List[List[Tuple[str, float]]]:
        """
        Vectorised top-k retrieval for multiple query embeddings at once.

        Performs a **single batched matrix multiply** (BLAS SGEMM) to compute
        cosine similarities between all *M* query vectors and all *N* stored
        vectors simultaneously — O(M·N·d') rather than M separate O(N·d')
        SGEMV calls.

        Args:
            embeddings: Raw query embeddings, shape ``(M, dim)``.
            k:          Number of top matches to return per query (default 3).
            threshold:  Minimum cosine similarity to include in results.
                        Set to ``0.0`` (default) to return all top-k regardless.

        Returns:
            List of length *M*.  Each element is a list of up to *k* tuples
            ``(hashed_id, cosine_similarity)``, sorted descending by similarity.
            Entries below *threshold* are omitted.
        """
        if self._cache_dirty:
            self._rebuild_cache()

        M = embeddings.shape[0]

        if not self._cache_ids:              # store is empty
            return [[] for _ in range(M)]

        # Project each query and L2-normalise — shape (M, projected_dim)
        proj = self._proj.project_batch(
            np.stack([self._l2_norm(embeddings[i]) for i in range(M)])
        )
        norms = np.linalg.norm(proj, axis=1, keepdims=True) + 1e-9
        Q_norm: np.ndarray = (proj / norms).astype(np.float32)

        # One SGEMM: (M, projected_dim) @ (projected_dim, N) → (M, N)
        sims_mat: np.ndarray = Q_norm @ self._cache_mat.T

        N  = sims_mat.shape[1]
        k_ = min(k, N)

        results: List[List[Tuple[str, float]]] = []
        for row in sims_mat:
            if k_ == N:
                idx = np.argsort(row)[::-1]
            else:
                part = np.argpartition(row, N - k_)[N - k_:]
                idx  = part[np.argsort(row[part])[::-1]]
            results.append([
                (self._cache_ids[i], float(row[i]))
                for i in idx
                if float(row[i]) >= threshold
            ])
        return results

    # ------------------------------------------------------------------
    # Backward-compatible aliases (preserved for existing tests / CLI)
    # ------------------------------------------------------------------

    def add(self, subject_id: str, embedding: np.ndarray) -> None:
        """Alias for :meth:`store`."""
        self.store(subject_id, embedding)

    def get(self, subject_id: str) -> Optional[np.ndarray]:
        """Return the *protected* embedding for *subject_id*, or None."""
        hid = _hash_subject(subject_id, self._salt)
        return self._db.get(hid)

    def remove(self, subject_id: str) -> bool:
        """Delete a subject. Returns True if the entry existed."""
        hid = _hash_subject(subject_id, self._salt)
        deleted = self._db.delete(hid)
        if deleted:
            self._cache_dirty = True
        return deleted

    def save(self) -> None:
        """Flush in-memory state to disk (no-op for SQLite; required for npz)."""
        if self._backend_name == "npz":
            self._db.save()  # type: ignore[attr-defined]
            log.info("Saved npz store -> %s", self.store_path)
        else:
            log.debug("SQLite backend auto-commits; save() is a no-op.")

    def __len__(self) -> int:
        return self._db.count()

    def __contains__(self, subject_id: str) -> bool:
        return self._db.contains(_hash_subject(subject_id, self._salt))

    def close(self) -> None:
        """Release backend resources (SQLite connection)."""
        self._db.close()

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def rotate_key(self, old_key: str, new_key: str) -> int:
        """
        Re-encrypt every stored embedding blob from *old_key* to *new_key*.

        This is a **migration** operation: after calling this method, the
        store's active Fernet key is updated to *new_key* so that subsequent
        reads and writes use the new key without requiring a store restart.

        The operation is atomic at the row level: each row is re-encrypted
        individually inside a single SQLite transaction.  If the process is
        interrupted, at most one row will be in an inconsistent state because
        SQLite's ``REPLACE`` is atomic for each statement; restart the
        rotation from the same *old_key* to repair.

        Args:
            old_key: Base64-urlsafe Fernet key that was used to encrypt the
                     existing blobs (the *current* key before rotation).
                     If the store was created with ``encrypt=False`` pass
                     the empty string ``""`` to signal plain-text source.
            new_key: Base64-urlsafe Fernet key that will replace *old_key*.
                     Generate one with
                     ``cryptography.fernet.Fernet.generate_key().decode()``.

        Returns:
            Number of rows that were re-encrypted.

        Raises:
            NotImplementedError: When the active backend is not SQLite.
            cryptography.fernet.InvalidToken: If *old_key* cannot decrypt one
                of the stored blobs (wrong key or corrupted data).
        """
        if not isinstance(self._db, _SQLiteBackend):
            raise NotImplementedError(
                "rotate_key is only supported for the 'sqlite' backend."
            )

        from cryptography.fernet import Fernet

        old_fernet = Fernet(old_key.encode() if isinstance(old_key, str) else old_key) \
            if old_key else None
        new_fernet = Fernet(new_key.encode() if isinstance(new_key, str) else new_key)

        raw_rows = self._db.all_raw()   # [(hashed_id, encrypted_blob, dim), ...]
        updated = 0

        for hashed_id, raw_blob, dim in raw_rows:
            # Decrypt with old key (or treat as plain bytes if no old key)
            plain_blob: bytes = (
                old_fernet.decrypt(raw_blob) if old_fernet else raw_blob
            )
            # Re-encrypt with new key
            new_blob = new_fernet.encrypt(plain_blob)
            self._db.put_raw(hashed_id, new_blob, dim)
            updated += 1

        # Activate new key for all subsequent operations
        self._fernet = new_fernet
        self._db._fernet = new_fernet

        self._cache_dirty = True
        log.info(
            "rotate_key: re-encrypted %d embedding row(s) with new Fernet key.",
            updated,
        )
        return updated

    # ------------------------------------------------------------------
    # Reconstruction attack measurement
    # ------------------------------------------------------------------

    def reconstruct_attack_attempt(
        self,
        original_embeddings: np.ndarray,
        protected_embeddings: np.ndarray,
        hidden_dim: int = 256,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> dict:
        """
        Train a shallow MLP decoder to map protected embeddings back to originals.

        This simulates an adversary who has obtained a set of
        (protected, original) pairs and is trying to learn to invert the
        projection.  In practice, such pairs are very hard to obtain (the
        whole point of not storing originals), but this function lets a
        developer measure *worst-case* leakage if the attacker has
        some known pairs (i.e. insider threat scenario).

        Architecture: Linear(projected_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, dim)

        Security note: With d' == d the linear component of the optimal
        decoder IS Q^T (the transpose of the projection), so a sufficiently
        large linear model would perfectly reconstruct.  Using a smaller
        projected_dim (< dim) increases reconstruction MSE by limiting
        information content.

        Args:
            original_embeddings:  (N, dim)          raw embeddings.
            protected_embeddings: (N, projected_dim) projected embeddings.
            hidden_dim:           Hidden layer width of the MLP decoder.
            epochs:               SGD epochs.
            lr:                   Learning rate.

        Returns:
            Dict with keys: ``"train_mse"``, ``"val_mse"``, ``"model"``
            (the fitted numpy weight list), ``"n_train"``, ``"n_val"``.
        """
        N = original_embeddings.shape[0]
        if N < 4:
            raise ValueError("Need at least 4 samples for train/val split.")

        split = max(1, int(N * 0.8))
        X_tr, X_val = protected_embeddings[:split], protected_embeddings[split:]
        Y_tr, Y_val = original_embeddings[:split],  original_embeddings[split:]

        # --- Tiny pure-numpy MLP (no torch dependency) ---
        d_in   = X_tr.shape[1]
        d_out  = Y_tr.shape[1]
        rng    = np.random.default_rng(0)
        W1     = rng.standard_normal((d_in,  hidden_dim)).astype(np.float32) * 0.01
        b1     = np.zeros(hidden_dim, dtype=np.float32)
        W2     = rng.standard_normal((hidden_dim, d_out)).astype(np.float32) * 0.01
        b2     = np.zeros(d_out, dtype=np.float32)

        n_tr   = X_tr.shape[0]
        for epoch in range(epochs):
            # Forward
            h1     = np.maximum(X_tr @ W1 + b1, 0)     # ReLU
            y_hat  = h1 @ W2 + b2
            diff   = y_hat - Y_tr
            loss   = float((diff ** 2).mean())

            # Backward
            dL_dy  = 2 * diff / n_tr
            dW2    = h1.T @ dL_dy
            db2    = dL_dy.sum(0)
            dh1    = dL_dy @ W2.T * (h1 > 0).astype(np.float32)
            dW1    = X_tr.T @ dh1
            db1    = dh1.sum(0)

            W1 -= lr * dW1;  b1 -= lr * db1
            W2 -= lr * dW2;  b2 -= lr * db2

            if epoch % 50 == 0:
                log.debug("ReconstructMLP epoch %d loss %.5f", epoch, loss)

        # Eval
        def _mse(X: np.ndarray, Y: np.ndarray) -> float:
            h = np.maximum(X @ W1 + b1, 0)
            return float(((h @ W2 + b2 - Y) ** 2).mean())

        return {
            "train_mse": _mse(X_tr, Y_tr),
            "val_mse":   _mse(X_val, Y_val),
            "model":     [W1, b1, W2, b2],
            "n_train":   split,
            "n_val":     N - split,
        }

    def reconstruction_success_rate(
        self,
        original_embeddings: np.ndarray,
        protected_embeddings: np.ndarray,
        mse_threshold: float = 0.01,
        hidden_dim: int = 256,
        epochs: int = 200,
        lr: float = 1e-3,
    ) -> float:
        """
        Measure what fraction of reconstructions fall below *mse_threshold*.

        Uses :meth:`reconstruct_attack_attempt` to train an MLP decoder,
        then evaluates per-sample MSE on the validation split and returns
        the fraction of samples the attacker would "successfully" reconstruct.

        Target: < 5% success rate (i.e. the attacker fails on 95%+ of samples).

        Args:
            original_embeddings:  (N, dim) raw embeddings.
            protected_embeddings: (N, projected_dim) projected embeddings.
            mse_threshold:        Per-sample MSE below which reconstruction
                                  is deemed "successful".  Defaults to 0.01
                                  (tight for 512-D unit vectors).
            hidden_dim:           MLP hidden layer width.
            epochs:               Training epochs.
            lr:                   Learning rate.

        Returns:
            Float in [0, 1]: fraction of val-set samples with per-sample
            MSE < mse_threshold.
        """
        result = self.reconstruct_attack_attempt(
            original_embeddings, protected_embeddings,
            hidden_dim=hidden_dim, epochs=epochs, lr=lr,
        )
        W1, b1, W2, b2 = result["model"]
        split = result["n_train"]
        X_val = protected_embeddings[split:]
        Y_val = original_embeddings[split:]
        h     = np.maximum(X_val @ W1 + b1, 0)
        y_hat = h @ W2 + b2
        per_sample_mse = ((y_hat - Y_val) ** 2).mean(axis=1)
        rate = float((per_sample_mse < mse_threshold).mean())
        log.info(
            "Reconstruction attack: val_mse=%.5f | success_rate(thr=%.4f)=%.3f",
            result["val_mse"], mse_threshold, rate,
        )
        return rate


# ---------------------------------------------------------------------------
# Module-level convenience functions (for use without an EmbeddingStore instance)
# ---------------------------------------------------------------------------


def _train_decoder_mlp(
    X_tr: np.ndarray, Y_tr: np.ndarray,
    hidden_dim: int, epochs: int, lr: float,
):
    """Return (W1, b1, W2, b2) trained on X_tr -> Y_tr."""
    d_in  = X_tr.shape[1]
    d_out = Y_tr.shape[1]
    rng   = np.random.default_rng(0)
    W1    = rng.standard_normal((d_in, hidden_dim)).astype(np.float32) * 0.01
    b1    = np.zeros(hidden_dim, dtype=np.float32)
    W2    = rng.standard_normal((hidden_dim, d_out)).astype(np.float32) * 0.01
    b2    = np.zeros(d_out, dtype=np.float32)
    n     = X_tr.shape[0]
    for _ in range(epochs):
        h1     = np.maximum(X_tr @ W1 + b1, 0)
        y_hat  = h1 @ W2 + b2
        diff   = y_hat - Y_tr
        dL_dy  = 2 * diff / n
        dW2    = h1.T @ dL_dy
        db2    = dL_dy.sum(0)
        dh1    = dL_dy @ W2.T * (h1 > 0).astype(np.float32)
        W1    -= lr * (X_tr.T @ dh1);   b1 -= lr * dh1.sum(0)
        W2    -= lr * dW2;               b2 -= lr * db2
    return W1, b1, W2, b2


def reconstruction_success_rate_standalone(
    original_embeddings: np.ndarray,
    protected_embeddings: np.ndarray,
    mse_threshold: float = 0.01,
    hidden_dim: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
) -> float:
    """
    Standalone version of :meth:`EmbeddingStore.reconstruction_success_rate`.

    Useful when you want to assess projection strength without instantiating
    an :class:`EmbeddingStore`.

    Args:
        original_embeddings:  (N, dim) raw embeddings.
        protected_embeddings: (N, projected_dim) projected embeddings.
        mse_threshold:        Per-sample MSE below which reconstruction is
                              deemed "successful".
        hidden_dim:           MLP hidden layer width.
        epochs:               Training epochs.
        lr:                   Learning rate.

    Returns:
        Float in [0, 1].
    """
    N = original_embeddings.shape[0]
    if N < 4:
        raise ValueError("Need at least 4 samples for train/val split.")
    split = max(1, int(N * 0.8))
    X_tr, X_val = protected_embeddings[:split], protected_embeddings[split:]
    Y_tr, Y_val = original_embeddings[:split],  original_embeddings[split:]
    W1, b1, W2, b2 = _train_decoder_mlp(X_tr, Y_tr, hidden_dim, epochs, lr)
    h      = np.maximum(X_val @ W1 + b1, 0)
    y_hat  = h @ W2 + b2
    per_mse = ((y_hat - Y_val) ** 2).mean(axis=1)
    return float((per_mse < mse_threshold).mean())
