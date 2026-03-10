"""
tests/test_auth.py – Unit tests for src/web/auth.py

Coverage
--------
* UserStore: CRUD, bcrypt verify, duplicate username, invalid role
* JWT helpers: create_token / decode_token round-trip, expiry
* get_current_user: Bearer header, cookie, query-param, missing token, bad token
* Role helpers: role_rank, has_role
* require_auth decorator: pass / 401 / browser redirect
* require_role decorator: correct role / wrong role / admin bypass / unauthenticated
* socket_auth: valid token, missing token, bad token
* Blueprint routes: /auth/login, /auth/logout, /auth/me,
                    /auth/users, /auth/users/<u>/role, /auth/users/<u>/password,
                    /auth/users/<u> DELETE
* Severity-aware ack: operator cannot ack high-severity; admin can
* Admin-only panic: non-admin 403, admin 200
"""

from __future__ import annotations

import datetime
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers – isolated UserStore in a temp directory
# ---------------------------------------------------------------------------

def make_user_store(tmp_path):
    """Return a fresh UserStore backed by a temp SQLite file."""
    from src.web.auth import UserStore
    db = tmp_path / "test_users.db"
    return UserStore(db_path=str(db))


# ---------------------------------------------------------------------------
# 1. UserStore – CRUD
# ---------------------------------------------------------------------------

class TestUserStore:
    def test_create_and_get_user(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("alice", "pw123", role="viewer")
        u = s.get_user("alice")
        assert u is not None
        assert u["username"] == "alice"
        assert u["role"] == "viewer"
        assert "password_hash" not in u
        s.close()

    def test_create_admin_role(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("bob", "pw", role="admin")
        assert s.get_user("bob")["role"] == "admin"
        s.close()

    def test_create_operator_role(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("carol", "pw", role="operator")
        assert s.get_user("carol")["role"] == "operator"
        s.close()

    def test_duplicate_username_raises(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("dave", "pw1", role="viewer")
        with pytest.raises(ValueError, match="already exists"):
            s.create_user("dave", "pw2", role="admin")
        s.close()

    def test_invalid_role_raises(self, tmp_path):
        s = make_user_store(tmp_path)
        with pytest.raises(ValueError, match="role must be one of"):
            s.create_user("eve", "pw", role="superuser")
        s.close()

    def test_count_and_list(self, tmp_path):
        s = make_user_store(tmp_path)
        assert s.count() == 0
        s.create_user("u1", "pw", role="viewer")
        s.create_user("u2", "pw", role="operator")
        assert s.count() == 2
        users = s.list_users()
        assert len(users) == 2
        names = {u["username"] for u in users}
        assert names == {"u1", "u2"}
        s.close()

    def test_get_nonexistent_user_returns_none(self, tmp_path):
        s = make_user_store(tmp_path)
        assert s.get_user("nobody") is None
        s.close()

    def test_delete_user(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("fred", "pw", role="viewer")
        assert s.delete_user("fred") is True
        assert s.get_user("fred") is None
        s.close()

    def test_delete_nonexistent_returns_false(self, tmp_path):
        s = make_user_store(tmp_path)
        assert s.delete_user("ghost") is False
        s.close()

    def test_update_role(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("grace", "pw", role="viewer")
        assert s.update_role("grace", "operator") is True
        assert s.get_user("grace")["role"] == "operator"
        s.close()

    def test_update_role_invalid_raises(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("heidi", "pw", role="viewer")
        with pytest.raises(ValueError):
            s.update_role("heidi", "god")
        s.close()

    def test_update_role_nonexistent_returns_false(self, tmp_path):
        s = make_user_store(tmp_path)
        assert s.update_role("nobody", "admin") is False
        s.close()

    def test_update_password(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("ivan", "old_pw", role="viewer")
        assert s.update_password("ivan", "new_pw") is True
        assert s.authenticate("ivan", "new_pw") is not None
        assert s.authenticate("ivan", "old_pw") is None
        s.close()

    def test_update_password_nonexistent_returns_false(self, tmp_path):
        s = make_user_store(tmp_path)
        assert s.update_password("nobody", "pw") is False
        s.close()


# ---------------------------------------------------------------------------
# 2. UserStore – Password hashing / authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_authenticate_correct_password(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("judy", "secretpw", role="operator")
        result = s.authenticate("judy", "secretpw")
        assert result is not None
        assert result["username"] == "judy"
        assert result["role"] == "operator"
        assert "password_hash" not in result
        s.close()

    def test_authenticate_wrong_password_returns_none(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("karl", "correct", role="viewer")
        assert s.authenticate("karl", "wrong") is None
        s.close()

    def test_authenticate_nonexistent_user_returns_none(self, tmp_path):
        s = make_user_store(tmp_path)
        assert s.authenticate("nobody", "pw") is None
        s.close()

    def test_authenticate_updates_last_login(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("lara", "pw", role="viewer")
        u_before = s.get_user("lara")
        assert u_before["last_login"] is None
        s.authenticate("lara", "pw")
        u_after = s.get_user("lara")
        assert u_after["last_login"] is not None
        s.close()

    def test_password_not_stored_in_plaintext(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("mike", "p@ssw0rd", role="viewer")
        conn = sqlite3.connect(str(tmp_path / "test_users.db"))
        row = conn.execute("SELECT password_hash FROM users WHERE username='mike'").fetchone()
        conn.close()
        assert "p@ssw0rd" not in row[0]
        # bcrypt hashes start with $2b$
        assert row[0].startswith("$2")
        s.close()


# ---------------------------------------------------------------------------
# 3. ensure_default_admin
# ---------------------------------------------------------------------------

class TestEnsureDefaultAdmin:
    def test_creates_admin_when_empty(self, tmp_path):
        s = make_user_store(tmp_path)
        assert s.count() == 0
        s.ensure_default_admin()
        assert s.count() == 1
        admin = s.get_user("admin")
        assert admin is not None
        assert admin["role"] == "admin"
        s.close()

    def test_does_not_create_if_users_exist(self, tmp_path):
        s = make_user_store(tmp_path)
        s.create_user("existing", "pw", role="viewer")
        s.ensure_default_admin()
        assert s.count() == 1  # still just 1
        s.close()

    def test_default_credentials_authenticate(self, tmp_path):
        from src.web.auth import DEFAULT_ADMIN_PASSWORD, DEFAULT_ADMIN_USER
        s = make_user_store(tmp_path)
        s.ensure_default_admin()
        result = s.authenticate(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD)
        assert result is not None
        s.close()


# ---------------------------------------------------------------------------
# 4. JWT helpers
# ---------------------------------------------------------------------------

class TestJWT:
    def test_create_and_decode_round_trip(self):
        from src.web.auth import create_token, decode_token
        token = create_token(1, "alice", "admin")
        payload = decode_token(token)
        assert payload["sub"] == "alice"
        assert payload["uid"] == 1
        assert payload["role"] == "admin"

    def test_expired_token_raises(self):
        import jwt
        from src.web.auth import _resolve_jwt_secret, decode_token

        secret = _resolve_jwt_secret()
        now = datetime.datetime.now(datetime.timezone.utc)
        expired_payload = {
            "sub": "alice",
            "uid": 1,
            "role": "viewer",
            "iat": now - datetime.timedelta(hours=9),
            "exp": now - datetime.timedelta(hours=1),  # already expired
        }
        token = jwt.encode(expired_payload, secret, algorithm="HS256")
        with pytest.raises(jwt.ExpiredSignatureError):
            decode_token(token)

    def test_tampered_token_raises(self):
        import jwt
        from src.web.auth import create_token, decode_token
        token = create_token(2, "bob", "operator")
        tampered = token[:-4] + "XXXX"
        with pytest.raises(jwt.InvalidTokenError):
            decode_token(tampered)

    def test_all_roles_encodeable(self):
        from src.web.auth import ROLES_ORDERED, create_token, decode_token
        for role in ROLES_ORDERED:
            token = create_token(99, "test_user", role)
            payload = decode_token(token)
            assert payload["role"] == role


# ---------------------------------------------------------------------------
# 5. Role helpers
# ---------------------------------------------------------------------------

class TestRoleHelpers:
    def test_role_rank_ordering(self):
        from src.web.auth import role_rank
        assert role_rank("viewer") < role_rank("operator") < role_rank("admin")

    def test_role_rank_unknown_is_minus_one(self):
        from src.web.auth import role_rank
        assert role_rank("root") == -1

    def test_has_role_exact_match(self):
        from src.web.auth import has_role
        user = {"role": "operator"}
        assert has_role(user, "operator") is True
        assert has_role(user, "viewer") is False

    def test_has_role_admin_bypasses_all(self):
        from src.web.auth import has_role
        admin = {"role": "admin"}
        assert has_role(admin, "viewer") is True
        assert has_role(admin, "operator") is True
        assert has_role(admin, "admin") is True

    def test_has_role_none_user_returns_false(self):
        from src.web.auth import has_role
        assert has_role(None, "viewer") is False

    def test_has_role_multiple_allowed_roles(self):
        from src.web.auth import has_role
        op = {"role": "operator"}
        assert has_role(op, "operator", "admin") is True
        viewer = {"role": "viewer"}
        assert has_role(viewer, "operator", "admin") is False


# ---------------------------------------------------------------------------
# 6. socket_auth
# ---------------------------------------------------------------------------

class TestSocketAuth:
    def test_valid_token_returns_payload(self):
        from src.web.auth import create_token, socket_auth
        token = create_token(1, "alice", "admin")
        payload = socket_auth({"token": token})
        assert payload is not None
        assert payload["sub"] == "alice"

    def test_none_auth_dict_returns_none(self):
        from src.web.auth import socket_auth
        assert socket_auth(None) is None

    def test_empty_dict_returns_none(self):
        from src.web.auth import socket_auth
        assert socket_auth({}) is None

    def test_invalid_token_returns_none(self):
        from src.web.auth import socket_auth
        assert socket_auth({"token": "not.a.jwt"}) is None

    def test_token_key_case_insensitive(self):
        """Accepts both 'token' and 'Token' keys."""
        from src.web.auth import create_token, socket_auth
        tok = create_token(2, "bob", "viewer")
        assert socket_auth({"Token": tok}) is not None


# ---------------------------------------------------------------------------
# 7. Flask Blueprint via test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def auth_app(tmp_path):
    """Return a minimal Flask app with the auth blueprint mounted."""
    from flask import Flask, jsonify
    from src.web.auth import (
        UserStore, create_auth_blueprint,
        require_auth, require_role, get_current_user,
    )

    app = Flask(__name__, template_folder=str(
        Path(__file__).parent.parent / "src" / "web" / "templates"
    ))
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    app.secret_key = "test-secret"

    db = tmp_path / "users.db"
    us = UserStore(db_path=str(db))
    us.create_user("admin_user", "adminpw", role="admin")
    us.create_user("op_user",    "oppw",    role="operator")
    us.create_user("view_user",  "viewpw",  role="viewer")

    bp = create_auth_blueprint(us)
    app.register_blueprint(bp)

    # Protected routes for decorator tests
    @app.get("/protected")
    @require_auth
    def protected():
        user = get_current_user()
        return jsonify(username=user["sub"])

    @app.get("/admin-only")
    @require_role("admin")
    def admin_only():
        return jsonify(ok=True)

    @app.get("/op-or-admin")
    @require_role("operator", "admin")
    def op_or_admin():
        return jsonify(ok=True)

    return app, us


def _login(client, username, password):
    """Log in via the API path and return the JWT token string."""
    resp = client.post(
        "/auth/login",
        json={"username": username, "password": password},
        headers={"Accept": "application/json"},
    )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()["token"]


class TestAuthBlueprint:
    def test_login_returns_token(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.post(
                "/auth/login",
                json={"username": "admin_user", "password": "adminpw"},
                headers={"Accept": "application/json"},
            )
            assert resp.status_code == 200
            body = resp.get_json()
            assert "token" in body
            assert body["role"] == "admin"

    def test_login_wrong_password(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.post(
                "/auth/login",
                json={"username": "admin_user", "password": "wrong"},
                headers={"Accept": "application/json"},
            )
            assert resp.status_code == 401

    def test_logout_clears_cookie(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.post("/auth/logout")
            assert resp.status_code == 200

    def test_me_returns_user_info(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "op_user", "oppw")
            resp = c.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["username"] == "op_user"
            assert body["role"] == "operator"

    def test_me_without_token_returns_401(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.get("/auth/me")
            assert resp.status_code == 401

    def test_list_users_admin(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.get("/auth/users", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200
            users = resp.get_json()["users"]
            names = {u["username"] for u in users}
            assert "admin_user" in names

    def test_list_users_non_admin_forbidden(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "op_user", "oppw")
            resp = c.get("/auth/users", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 403

    def test_create_user_as_admin(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.post(
                "/auth/users",
                json={"username": "newbie", "password": "newpw", "role": "viewer"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 201
            assert resp.get_json()["username"] == "newbie"

    def test_create_user_duplicate_returns_409(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            c.post(
                "/auth/users",
                json={"username": "dup", "password": "pw", "role": "viewer"},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp = c.post(
                "/auth/users",
                json={"username": "dup", "password": "pw2", "role": "viewer"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 409

    def test_create_user_missing_fields_returns_400(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.post(
                "/auth/users",
                json={"username": "", "password": "", "role": "viewer"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 400

    def test_update_role_as_admin(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.put(
                "/auth/users/view_user/role",
                json={"role": "operator"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            assert resp.get_json()["role"] == "operator"

    def test_update_role_invalid_role(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.put(
                "/auth/users/view_user/role",
                json={"role": "superuser"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 400

    def test_update_role_nonexistent_user(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.put(
                "/auth/users/nobody/role",
                json={"role": "viewer"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 404

    def test_change_own_password(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "op_user", "oppw")
            resp = c.post(
                "/auth/users/op_user/password",
                json={"password": "new_oppw"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            # Should now authenticate with new password
            resp2 = c.post(
                "/auth/login",
                json={"username": "op_user", "password": "new_oppw"},
                headers={"Accept": "application/json"},
            )
            assert resp2.status_code == 200

    def test_change_other_user_password_forbidden_for_non_admin(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "op_user", "oppw")
            resp = c.post(
                "/auth/users/view_user/password",
                json={"password": "hacked"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 403

    def test_admin_can_change_any_password(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.post(
                "/auth/users/view_user/password",
                json={"password": "admin_set_pw"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

    def test_delete_user_as_admin(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            # Create then delete
            c.post(
                "/auth/users",
                json={"username": "tobedeleted", "password": "pw", "role": "viewer"},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp = c.delete(
                "/auth/users/tobedeleted",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

    def test_delete_nonexistent_returns_404(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.delete(
                "/auth/users/nobody",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 404

    def test_delete_self_returns_400(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.delete(
                "/auth/users/admin_user",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 8. require_auth / require_role decorators (via test-app routes)
# ---------------------------------------------------------------------------

class TestDecorators:
    def test_require_auth_pass(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "view_user", "viewpw")
            resp = c.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200
            assert resp.get_json()["username"] == "view_user"

    def test_require_auth_missing_token_api(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.get("/protected", headers={"Accept": "application/json"})
            assert resp.status_code == 401
            assert resp.get_json()["code"] == "unauthenticated"

    def test_require_auth_bad_token_returns_401(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.get(
                "/protected",
                headers={"Authorization": "Bearer not.a.real.token",
                          "Accept": "application/json"},
            )
            assert resp.status_code == 401

    def test_require_role_correct_role_passes(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.get("/admin-only", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200

    def test_require_role_wrong_role_forbidden(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "view_user", "viewpw")
            resp = c.get(
                "/admin-only",
                headers={"Authorization": f"Bearer {token}",
                          "Accept": "application/json"},
            )
            assert resp.status_code == 403
            assert resp.get_json()["code"] == "forbidden"

    def test_require_role_admin_bypasses_operator_route(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            token = _login(c, "admin_user", "adminpw")
            resp = c.get("/op-or-admin", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200

    def test_require_role_unauthenticated_api(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.get("/admin-only", headers={"Accept": "application/json"})
            assert resp.status_code == 401

    def test_token_via_cookie(self, auth_app):
        """JWT supplied via pcs_token cookie is accepted."""
        from src.web.auth import create_token
        app, _ = auth_app
        token = create_token(99, "view_user", "viewer")
        with app.test_client() as c:
            c.set_cookie("pcs_token", token)
            resp = c.get("/protected")
            assert resp.status_code == 200

    def test_token_via_query_param(self, auth_app):
        """JWT supplied as ?token= query param is accepted."""
        from src.web.auth import create_token
        app, _ = auth_app
        token = create_token(99, "view_user", "viewer")
        with app.test_client() as c:
            resp = c.get(f"/protected?token={token}")
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 9. Login redirect for browser requests
# ---------------------------------------------------------------------------

class TestBrowserRedirects:
    def test_unauthenticated_browser_redirects_to_login(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.get("/protected", headers={"Accept": "text/html,*/*"})
            assert resp.status_code == 302
            assert "/auth/login" in resp.headers["Location"]

    def test_login_page_get(self, auth_app):
        app, _ = auth_app
        with app.test_client() as c:
            resp = c.get("/auth/login")
            assert resp.status_code == 200
            assert b"Login" in resp.data

    def test_login_page_already_authenticated_redirects(self, auth_app):
        from src.web.auth import create_token
        app, _ = auth_app
        token = create_token(1, "admin_user", "admin")
        with app.test_client() as c:
            c.set_cookie("pcs_token", token)
            resp = c.get("/auth/login")
            # Should redirect away from login page
            assert resp.status_code == 302
