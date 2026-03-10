"""
Tests for src/privacy/embedding_store.py.

Coverage targets
----------------
- _OrthProjection: determinism, norm-preservation, salt sensitivity
- EmbeddingStore (SQLite): store/bulk_store/query_topk/remove/len/contains
- EmbeddingStore (npz):    same round-trip via npz backend
- Backward-compat:         add() / get() / remove() match original API
- Privacy invariants:      subject IDs never in plaintext; get() returns
                           protected (not raw) embeddings
- Nearest-neighbour:       cosine ordering in projected space == original space
                           (exact for square orthogonal Q + unit vectors)
- Reconstruction attack:   success_rate is a float in [0, 1]; high threshold
                           trivially yields rate=1; low threshold yields rate=0
                           for a single-sample val set
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import os

from src.privacy.embedding_store import (
    EmbeddingStore,
    _OrthProjection,
    _hash_subject,
    reconstruction_success_rate_standalone,
)

# ---------------------------------------------------------------------------
# Helper that generates a real Fernet key (skips tests when cryptography
# is not installed — the rest of the suite remains unaffected)
# ---------------------------------------------------------------------------
try:
    from cryptography.fernet import Fernet as _Fernet

    def _new_fernet_key() -> str:
        return _Fernet.generate_key().decode()

    FERNET_AVAILABLE = True
except ImportError:
    FERNET_AVAILABLE = False
    _new_fernet_key = None  # type: ignore[assignment]

fernet_required = pytest.mark.skipif(
    not FERNET_AVAILABLE,
    reason="cryptography package not installed",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DIM = 32  # small so tests run in <1 s; tests that verify 512 use dim=512 explicitly

RNG = np.random.default_rng(42)


def _rng_embedding(dim: int = DIM, normalize: bool = True) -> np.ndarray:
    v = RNG.standard_normal(dim).astype(np.float32)
    if normalize:
        v /= np.linalg.norm(v)
    return v


@pytest.fixture
def salt() -> str:
    return "test-salt-abc"


@pytest.fixture
def orth(salt) -> _OrthProjection:
    return _OrthProjection(DIM, DIM, salt)


@pytest.fixture
def store_sqlite(tmp_path, salt) -> EmbeddingStore:
    return EmbeddingStore(
        tmp_path / "store.db",
        salt=salt,
        dim=DIM,
        projected_dim=DIM,
        backend="sqlite",
    )


@pytest.fixture
def store_npz(tmp_path, salt) -> EmbeddingStore:
    return EmbeddingStore(
        tmp_path / "store.npz",
        salt=salt,
        dim=DIM,
        projected_dim=DIM,
        backend="npz",
    )


# ---------------------------------------------------------------------------
# 1. _OrthProjection
# ---------------------------------------------------------------------------


class TestOrthProjection:
    def test_output_shape(self, orth):
        v   = _rng_embedding()
        out = orth.project(v)
        assert out.shape == (DIM,)

    def test_dtype_is_float32(self, orth):
        out = orth.project(_rng_embedding())
        assert out.dtype == np.float32

    def test_deterministic_same_salt(self, salt):
        q1 = _OrthProjection(DIM, DIM, salt)
        q2 = _OrthProjection(DIM, DIM, salt)
        v  = _rng_embedding()
        np.testing.assert_array_equal(q1.project(v), q2.project(v))

    def test_different_salt_different_projection(self, salt):
        q1  = _OrthProjection(DIM, DIM, salt)
        q2  = _OrthProjection(DIM, DIM, salt + "-different")
        v   = _rng_embedding()
        assert not np.allclose(q1.project(v), q2.project(v))

    def test_norm_preserved_unit_vector(self, orth):
        """Square orthogonal Q preserves L2 norm."""
        v   = _rng_embedding(normalize=True)
        out = orth.project(v)
        assert math.isclose(float(np.linalg.norm(out)), 1.0, abs_tol=1e-4)

    def test_inner_product_preserved(self, salt):
        """Square Q: <Q x, Q y> == <x, y> (inner-product preserving)."""
        q  = _OrthProjection(DIM, DIM, salt)
        x  = _rng_embedding(normalize=True)
        y  = _rng_embedding(normalize=True)
        assert math.isclose(
            float(np.dot(q.project(x), q.project(y))),
            float(np.dot(x, y)),
            abs_tol=1e-4,
        )

    def test_project_batch_matches_single(self, orth):
        embs  = RNG.standard_normal((10, DIM)).astype(np.float32)
        batch = orth.project_batch(embs)
        for i in range(10):
            single = orth.project(embs[i])
            np.testing.assert_allclose(batch[i], single, atol=1e-5)

    def test_batch_output_shape(self, orth):
        out = orth.project_batch(RNG.standard_normal((7, DIM)).astype(np.float32))
        assert out.shape == (7, DIM)

    def test_dimensionality_reduction(self, salt):
        """projected_dim < dim gives (projected_dim,) output."""
        q = _OrthProjection(dim=32, projected_dim=16, salt=salt)
        v = _rng_embedding(dim=32)
        assert q.project(v).shape == (16,)

    def test_invalid_projected_dim_raises(self, salt):
        with pytest.raises(ValueError, match="projected_dim"):
            _OrthProjection(dim=32, projected_dim=64, salt=salt)

    def test_invalid_projected_dim_zero_raises(self, salt):
        with pytest.raises(ValueError):
            _OrthProjection(dim=32, projected_dim=0, salt=salt)


# ---------------------------------------------------------------------------
# 2. EmbeddingStore – SQLite backend
# ---------------------------------------------------------------------------


class TestEmbeddingStoreStoreSQLite:
    def test_store_and_get(self, store_sqlite, salt):
        v = _rng_embedding()
        store_sqlite.store("alice", v)
        retrieved = store_sqlite.get("alice")
        assert retrieved is not None
        assert retrieved.shape == (DIM,)

    def test_get_returns_protected_not_raw(self, store_sqlite, salt):
        """get() must NOT return the raw embedding (privacy guarantee)."""
        v = _rng_embedding()
        store_sqlite.store("alice", v)
        retrieved = store_sqlite.get("alice")
        # For a random orthogonal projection the probability that raw==protected is 0.
        assert not np.allclose(retrieved, v, atol=1e-5)

    def test_get_unknown_returns_none(self, store_sqlite):
        assert store_sqlite.get("nobody") is None

    def test_len(self, store_sqlite):
        assert len(store_sqlite) == 0
        store_sqlite.store("a", _rng_embedding())
        assert len(store_sqlite) == 1
        store_sqlite.store("b", _rng_embedding())
        assert len(store_sqlite) == 2

    def test_contains_true(self, store_sqlite):
        store_sqlite.store("alice", _rng_embedding())
        assert "alice" in store_sqlite

    def test_contains_false(self, store_sqlite):
        assert "missing" not in store_sqlite

    def test_remove_existing(self, store_sqlite):
        store_sqlite.store("alice", _rng_embedding())
        assert store_sqlite.remove("alice") is True
        assert "alice" not in store_sqlite

    def test_remove_nonexistent(self, store_sqlite):
        assert store_sqlite.remove("ghost") is False

    def test_bulk_store_count(self, store_sqlite):
        ids   = [f"subject_{i}" for i in range(20)]
        embs  = RNG.standard_normal((20, DIM)).astype(np.float32)
        store_sqlite.bulk_store(ids, embs)
        assert len(store_sqlite) == 20

    def test_bulk_store_retrievable(self, store_sqlite):
        ids  = ["x", "y", "z"]
        embs = RNG.standard_normal((3, DIM)).astype(np.float32)
        store_sqlite.bulk_store(ids, embs)
        for id_ in ids:
            assert store_sqlite.get(id_) is not None

    def test_bulk_store_mismatch_raises(self, store_sqlite):
        with pytest.raises(ValueError, match="ids length"):
            store_sqlite.bulk_store(["a", "b"], RNG.standard_normal((5, DIM)).astype(np.float32))

    def test_sqlite_file_created(self, store_sqlite):
        store_sqlite.store("a", _rng_embedding())
        assert store_sqlite.store_path.exists()

    def test_persistence_across_instances(self, tmp_path, salt):
        path = tmp_path / "persist.db"
        s1   = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="sqlite")
        v    = _rng_embedding()
        s1.store("alice", v)

        s2   = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="sqlite")
        assert "alice" in s2
        assert s2.get("alice") is not None

    def test_overwrite_same_id(self, store_sqlite):
        v1 = _rng_embedding()
        v2 = _rng_embedding()
        store_sqlite.store("alice", v1)
        store_sqlite.store("alice", v2)  # overwrite
        assert len(store_sqlite) == 1


# ---------------------------------------------------------------------------
# 3. EmbeddingStore – npz backend
# ---------------------------------------------------------------------------


class TestEmbeddingStoreNpz:
    def test_store_and_get(self, store_npz):
        v = _rng_embedding()
        store_npz.store("bob", v)
        assert store_npz.get("bob") is not None

    def test_persistence(self, tmp_path, salt):
        path = tmp_path / "data.npz"
        s1   = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="npz")
        s1.store("charlie", _rng_embedding())
        s1.save()

        s2 = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="npz")
        assert "charlie" in s2

    def test_save_creates_file(self, store_npz):
        store_npz.store("x", _rng_embedding())
        store_npz.save()
        assert store_npz.store_path.exists()

    def test_len(self, store_npz):
        assert len(store_npz) == 0
        for i in range(5):
            store_npz.store(str(i), _rng_embedding())
        assert len(store_npz) == 5


# ---------------------------------------------------------------------------
# 4. Backward-compatible add() / get() / remove() API
# ---------------------------------------------------------------------------


class TestBackwardCompatAPI:
    def test_add_and_get(self, store_sqlite):
        v = _rng_embedding()
        store_sqlite.add("alice", v)
        result = store_sqlite.get("alice")
        assert result is not None
        assert result.shape == (DIM,)

    def test_get_unknown_returns_none(self, store_sqlite):
        assert store_sqlite.get("unknown") is None

    def test_contains(self, store_sqlite):
        store_sqlite.add("bob", _rng_embedding())
        assert "bob" in store_sqlite

    def test_remove(self, store_sqlite):
        store_sqlite.add("carol", _rng_embedding())
        assert store_sqlite.remove("carol") is True
        assert "carol" not in store_sqlite

    def test_remove_nonexistent(self, store_sqlite):
        assert store_sqlite.remove("nobody") is False

    def test_len(self, store_sqlite):
        for name in ("a", "b", "c"):
            store_sqlite.add(name, _rng_embedding())
        assert len(store_sqlite) == 3

    def test_subject_id_not_in_db_file(self, tmp_path, salt):
        """Subject IDs must not appear in the SQLite file in plaintext."""
        path = tmp_path / "check.db"
        s    = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="sqlite")
        s.add("super-secret-alice", _rng_embedding())
        raw_bytes = path.read_bytes()
        assert b"super-secret-alice" not in raw_bytes

    def test_persistence_round_trip(self, tmp_path, salt):
        path = tmp_path / "rt.db"
        s1   = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="sqlite")
        v    = _rng_embedding()
        s1.add("dave", v)
        s1.save()  # no-op for sqlite; should not raise

        s2   = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="sqlite")
        assert "dave" in s2

    def test_invalid_backend_raises(self, tmp_path, salt):
        with pytest.raises(ValueError, match="backend"):
            EmbeddingStore(tmp_path / "x.db", salt=salt, backend="json")


# ---------------------------------------------------------------------------
# 5. Privacy invariants
# ---------------------------------------------------------------------------


class TestPrivacyInvariants:
    def test_different_salt_same_id_different_hash(self):
        h1 = _hash_subject("alice", "salt-A")
        h2 = _hash_subject("alice", "salt-B")
        assert h1 != h2

    def test_same_id_same_salt_same_hash(self):
        assert _hash_subject("alice", "s") == _hash_subject("alice", "s")

    def test_store_key_not_plaintext(self, tmp_path, salt):
        """Raw SQL in the DB must not contain the original subject ID."""
        path = tmp_path / "priv.db"
        s    = EmbeddingStore(path, salt=salt, dim=DIM, projected_dim=DIM, backend="sqlite")
        s.store("alice-secret-id", _rng_embedding())
        conn = sqlite3.connect(str(path))
        rows = conn.execute("SELECT hashed_id FROM embeddings").fetchall()
        conn.close()
        keys = [r[0] for r in rows]
        assert all("alice" not in k for k in keys)

    def test_wrong_salt_cannot_retrieve(self, tmp_path):
        """A store opened with a different salt must not find Alice's record."""
        path = tmp_path / "ws.db"
        s1 = EmbeddingStore(path, salt="correct-salt", dim=DIM, projected_dim=DIM)
        s1.store("alice", _rng_embedding())
        s2 = EmbeddingStore(path, salt="wrong-salt", dim=DIM, projected_dim=DIM)
        assert "alice" not in s2

    def test_raw_embedding_not_stored(self, store_sqlite):
        raw = _rng_embedding()
        store_sqlite.store("alice", raw)
        protected = store_sqlite.get("alice")
        # Protected != raw (different salt means different Q)
        assert not np.allclose(raw, protected, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. query_topk
# ---------------------------------------------------------------------------


class TestQueryTopK:
    def test_empty_store_returns_empty(self, store_sqlite):
        assert store_sqlite.query_topk(_rng_embedding()) == []

    def test_top1_matches_only_entry(self, store_sqlite):
        v = _rng_embedding()
        store_sqlite.store("alice", v)
        results = store_sqlite.query_topk(v, k=1)
        assert len(results) == 1
        _, score = results[0]
        assert score > 0.99  # should be cosine ≈ 1

    def test_returns_at_most_k(self, store_sqlite):
        for i in range(10):
            store_sqlite.store(str(i), _rng_embedding())
        results = store_sqlite.query_topk(_rng_embedding(), k=3)
        assert len(results) <= 3

    def test_scores_are_sorted_descending(self, store_sqlite):
        for i in range(8):
            store_sqlite.store(str(i), _rng_embedding())
        results = store_sqlite.query_topk(_rng_embedding(), k=8)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_nearest_neighbour_order_preserved(self, salt):
        """
        Orthogonal projection preserves cosine similarity order for unit vectors.
        The top-1 in raw space must equal top-1 in projected space.
        """
        dim   = 64
        q_proj = _OrthProjection(dim, dim, salt)
        store = EmbeddingStore(
            Path("dummy_not_on_disk"),  # npz in-memory
            salt=salt, dim=dim, projected_dim=dim, backend="npz",
        )

        query = RNG.standard_normal(dim).astype(np.float32)
        query /= np.linalg.norm(query)

        candidates = []
        for i in range(10):
            v = RNG.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v)
            candidates.append(v)
            store.store(str(i), v)

        # Top-1 in raw space (cosine similarity)
        raw_sims  = [float(np.dot(c, query)) for c in candidates]
        best_raw_idx = int(np.argmax(raw_sims))

        # Top-1 from store
        results = store.query_topk(query, k=1)
        assert len(results) == 1
        best_hashed, _ = results[0]

        # Build expected hashed_id for best raw candidate
        expected_hash = _hash_subject(str(best_raw_idx), salt)
        assert best_hashed == expected_hash

    def test_returns_tuples_hashed_id_score(self, store_sqlite):
        store_sqlite.store("x", _rng_embedding())
        res = store_sqlite.query_topk(_rng_embedding(), k=1)
        hid, score = res[0]
        assert isinstance(hid, str) and len(hid) == 64  # sha256 hex
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 7. Reconstruction attack
# ---------------------------------------------------------------------------


class TestReconstructionAttack:
    def _make_data(self, n: int = 20, dim: int = 16):
        raw   = RNG.standard_normal((n, dim)).astype(np.float32)
        proj  = _OrthProjection(dim, dim, "attack-test-salt")
        prot  = proj.project_batch(raw)
        return raw, prot

    def test_returns_float(self, store_sqlite):
        raw, prot = self._make_data()
        rate = store_sqlite.reconstruction_success_rate(raw, prot, mse_threshold=0.5, epochs=5)
        assert isinstance(rate, float)

    def test_rate_in_unit_interval(self, store_sqlite):
        raw, prot = self._make_data()
        rate = store_sqlite.reconstruction_success_rate(raw, prot, mse_threshold=0.5, epochs=5)
        assert 0.0 <= rate <= 1.0

    def test_high_threshold_gives_rate_one(self, store_sqlite):
        """A threshold much larger than any MSE should yield success_rate == 1.0."""
        raw, prot = self._make_data()
        rate = store_sqlite.reconstruction_success_rate(
            raw, prot, mse_threshold=1e9, epochs=5
        )
        assert rate == pytest.approx(1.0)

    def test_zero_threshold_gives_rate_zero(self, store_sqlite):
        """A threshold of 0 (impossible) should yield rate == 0.0."""
        raw, prot = self._make_data()
        rate = store_sqlite.reconstruction_success_rate(
            raw, prot, mse_threshold=0.0, epochs=5
        )
        assert rate == pytest.approx(0.0)

    def test_reconstruct_attack_attempt_keys(self, store_sqlite):
        raw, prot = self._make_data()
        result = store_sqlite.reconstruct_attack_attempt(raw, prot, epochs=5)
        assert "train_mse" in result and "val_mse" in result
        assert "model"     in result
        assert "n_train"   in result and "n_val" in result

    def test_reconstruct_too_few_samples_raises(self, store_sqlite):
        raw  = RNG.standard_normal((3, 16)).astype(np.float32)
        prot = RNG.standard_normal((3, 16)).astype(np.float32)
        with pytest.raises(ValueError, match="least 4"):
            store_sqlite.reconstruct_attack_attempt(raw, prot, epochs=1)

    def test_standalone_function(self):
        raw, prot = self._make_data()
        rate = reconstruction_success_rate_standalone(raw, prot, mse_threshold=0.5, epochs=5)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# 8. Fernet at-rest encryption (SQLite backend)
# ---------------------------------------------------------------------------


class TestFernetEncryption:
    """Per-blob Fernet encryption in the SQLite backend.

    All tests are automatically skipped when *cryptography* is absent so the
    vanilla test suite continues to pass on minimal installs.
    """

    @fernet_required
    def test_encrypted_store_file_not_readable_as_plaintext(self, tmp_path, salt):
        """SQLite binary data must not be recoverable as float32 without the key."""
        key = _new_fernet_key()
        os.environ["PCS_FERNET_KEY"] = key
        try:
            path = tmp_path / "enc.db"
            store = EmbeddingStore(
                path, salt=salt, dim=DIM, projected_dim=DIM,
                backend="sqlite", encrypt=True,
            )
            raw = _rng_embedding()
            store.store("alice", raw)

            # Read the raw file bytes and check that the projected float32
            # blob does NOT appear verbatim — Fernet adds a header + HMAC.
            raw_bytes = path.read_bytes()
            projected_bytes = (store._db._fernet.decrypt(
                store._db._enc(  # re-encrypt to get database bytes
                    store._proj.project(store._l2_norm(raw)).tobytes()
                )
            ))
            # The *plain* float32 bytes must not appear in the file.
            plain_blob = store._proj.project(store._l2_norm(raw)).tobytes()
            assert plain_blob not in raw_bytes, (
                "Plain float32 blob found verbatim in the encrypted store file"
            )
        finally:
            os.environ.pop("PCS_FERNET_KEY", None)

    @fernet_required
    def test_encrypted_round_trip(self, tmp_path, salt):
        """store() + get() works transparently with Fernet enabled."""
        key = _new_fernet_key()
        store = EmbeddingStore(
            tmp_path / "enc_rt.db",
            salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        store._fernet = store._db._fernet  # already set; confirm wired
        raw = _rng_embedding()
        store.store("bob", raw)
        retrieved = store.get("bob")
        assert retrieved is not None
        assert retrieved.shape == (DIM,)
        # Retrieved must be the protected embedding, not the raw one
        assert not np.allclose(retrieved, raw, atol=1e-5)

    @fernet_required
    def test_query_topk_works_with_encryption(self, tmp_path, salt):
        """query_topk cache rebuild decrypts blobs correctly."""
        key = _new_fernet_key()
        store = EmbeddingStore(
            tmp_path / "enc_topk.db",
            salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        vecs = [_rng_embedding() for _ in range(5)]
        for i, v in enumerate(vecs):
            store.store(f"s{i}", v)
        results = store.query_topk(vecs[0], k=3)
        assert len(results) == 3
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    @fernet_required
    def test_encrypted_persistence_across_instances(self, tmp_path, salt):
        """A newly opened store with the same key can read encrypted data."""
        key = _new_fernet_key()
        path = tmp_path / "enc_persist.db"

        s1 = EmbeddingStore(
            path, salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        # Inject the key so both instances use the same one
        from cryptography.fernet import Fernet as _F
        f = _F(key.encode())
        s1._fernet = f
        s1._db._fernet = f
        s1.store("carol", _rng_embedding())

        # Second instance with the same key
        os.environ["PCS_FERNET_KEY"] = key
        try:
            s2 = EmbeddingStore(
                path, salt=salt, dim=DIM, projected_dim=DIM,
                backend="sqlite", encrypt=True,
            )
            assert "carol" in s2
            assert s2.get("carol") is not None
        finally:
            os.environ.pop("PCS_FERNET_KEY", None)

    @fernet_required
    def test_bulk_store_encrypted(self, tmp_path, salt):
        """bulk_store() encrypts every row; count and retrieval are correct."""
        key = _new_fernet_key()
        store = EmbeddingStore(
            tmp_path / "enc_bulk.db",
            salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        from cryptography.fernet import Fernet as _F
        store._fernet = _F(key.encode())
        store._db._fernet = store._fernet
        ids = [f"u{i}" for i in range(10)]
        embs = RNG.standard_normal((10, DIM)).astype(np.float32)
        store.bulk_store(ids, embs)
        assert len(store) == 10
        for id_ in ids:
            assert store.get(id_) is not None


# ---------------------------------------------------------------------------
# 9. Key rotation
# ---------------------------------------------------------------------------


class TestRotateKey:
    """EmbeddingStore.rotate_key() re-encrypts blobs in-place."""

    @fernet_required
    def test_rotate_key_returns_count(self, tmp_path, salt):
        key_a = _new_fernet_key()
        key_b = _new_fernet_key()
        from cryptography.fernet import Fernet as _F
        store = EmbeddingStore(
            tmp_path / "rot.db", salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        store._fernet = _F(key_a.encode())
        store._db._fernet = store._fernet
        for name in ("a", "b", "c"):
            store.store(name, _rng_embedding())
        n = store.rotate_key(key_a, key_b)
        assert n == 3

    @fernet_required
    def test_rotated_store_readable_with_new_key(self, tmp_path, salt):
        """After rotation, the store is queryable with new key immediately."""
        key_a = _new_fernet_key()
        key_b = _new_fernet_key()
        from cryptography.fernet import Fernet as _F
        store = EmbeddingStore(
            tmp_path / "rot2.db", salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        store._fernet = _F(key_a.encode())
        store._db._fernet = store._fernet
        v = _rng_embedding()
        store.store("dave", v)
        store.rotate_key(key_a, key_b)
        # Should still be retrievable via the now-active new key
        result = store.get("dave")
        assert result is not None
        assert result.shape == (DIM,)

    @fernet_required
    def test_rotated_store_topk_still_works(self, tmp_path, salt):
        """Cache is invalidated after rotation and rebuilt with new key."""
        key_a = _new_fernet_key()
        key_b = _new_fernet_key()
        from cryptography.fernet import Fernet as _F
        store = EmbeddingStore(
            tmp_path / "rot3.db", salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        store._fernet = _F(key_a.encode())
        store._db._fernet = store._fernet
        vecs = [_rng_embedding() for _ in range(5)]
        for i, v in enumerate(vecs):
            store.store(f"p{i}", v)
        store.rotate_key(key_a, key_b)
        results = store.query_topk(vecs[0], k=3)
        assert len(results) == 3

    @fernet_required
    def test_rotate_from_plain_to_encrypted(self, tmp_path, salt):
        """Plain (unencrypted) store can be migrated to Fernet-encrypted."""
        key_b = _new_fernet_key()
        # Create an unencrypted store (encrypt=False, no fernet)
        store = EmbeddingStore(
            tmp_path / "plain_to_enc.db", salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=False,
        )
        v = _rng_embedding()
        store.store("eve", v)
        # Rotate: old_key="" signals "no previous encryption"
        store.rotate_key("", key_b)
        # Now the store uses key_b — verify readable
        result = store.get("eve")
        assert result is not None

    @fernet_required
    def test_rotate_key_wrong_old_key_raises(self, tmp_path, salt):
        """Supplying the wrong old key raises InvalidToken."""
        from cryptography.fernet import Fernet as _F, InvalidToken
        key_a = _new_fernet_key()
        key_b = _new_fernet_key()
        key_wrong = _new_fernet_key()
        store = EmbeddingStore(
            tmp_path / "rot_err.db", salt=salt, dim=DIM, projected_dim=DIM,
            backend="sqlite", encrypt=True,
        )
        store._fernet = _F(key_a.encode())
        store._db._fernet = store._fernet
        store.store("frank", _rng_embedding())
        with pytest.raises(InvalidToken):
            store.rotate_key(key_wrong, key_b)

    @fernet_required
    def test_rotate_key_not_implemented_for_npz(self, tmp_path, salt):
        """rotate_key raises NotImplementedError for the npz backend."""
        key_a = _new_fernet_key()
        key_b = _new_fernet_key()
        store = EmbeddingStore(
            tmp_path / "ro.npz", salt=salt, dim=DIM, projected_dim=DIM,
            backend="npz",
        )
        store.store("x", _rng_embedding())
        with pytest.raises(NotImplementedError):
            store.rotate_key(key_a, key_b)
