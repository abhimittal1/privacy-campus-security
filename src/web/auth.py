"""
auth.py – RBAC authentication layer for Privacy Campus Security.

Security model
--------------
* **Users table** – SQLite table beside the main stores.  Each row holds:
  ``username``, ``bcrypt``-hashed password, ``role`` (viewer / operator / admin),
  ``created_at``, ``last_login``.

* **Roles** (ordered, lowest to highest):
  - ``viewer``   – read-only: dashboards, alerts list, metrics
  - ``operator`` – viewer + enroll/verify + ack *non-high-severity* alerts
  - ``admin``    – full access: ack any alert, trigger panic, manage users

* **JWT tokens** – signed with ``PCS_JWT_SECRET`` (HS256).  Default expiry is
  8 hours (``PCS_JWT_EXPIRY_H``).  Tokens are accepted as:
  1. ``Authorization: Bearer <token>`` header  (API / curl-friendly)
  2. ``pcs_token`` cookie  (browser-friendly, set at login)

* **Default admin** – If the users table is empty, a default account is
  created automatically:
    username: ``admin``   password: ``admin123``
  Change it immediately via ``POST /auth/users/<username>/password``.

* **Socket.IO** – Pass ``{auth: {token: "<jwt>"}}`` when establishing the
  socket connection.  The server validates the token on ``connect`` and
  stores the role in the session so write events (e.g. arm_enroll) can be
  gated by role.

Environment variables
---------------------
PCS_JWT_SECRET   Secret key for JWT signing.  A default is generated and
                 logged at WARNING level if unset (insecure in production).
PCS_JWT_EXPIRY_H Token lifetime in hours (default 8).
PCS_AUTH_DB      Path to the SQLite users database (default: same dir as
                 PCS_STORE, file ``users.db``).
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import sqlite3
import warnings
from functools import wraps
from pathlib import Path
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / env helpers
# ---------------------------------------------------------------------------

ROLES_ORDERED: List[str] = ["viewer", "operator", "admin"]

_JWT_SECRET_ENV  = "PCS_JWT_SECRET"
_JWT_EXPIRY_ENV  = "PCS_JWT_EXPIRY_H"
_AUTH_DB_ENV     = "PCS_AUTH_DB"

DEFAULT_ADMIN_USER     = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"


def _resolve_jwt_secret() -> str:
    secret = os.environ.get(_JWT_SECRET_ENV, "").strip()
    if not secret:
        # Derive a *reproducible* insecure default from a fixed seed so tests
        # work without env vars; warn loudly in non-test contexts.
        import secrets as _secrets
        secret = _secrets.token_hex(32)
        warnings.warn(
            f"{_JWT_SECRET_ENV} is not set.  A random ephemeral key is being used — "
            "tokens will NOT survive a server restart and this is INSECURE in production.  "
            "Set PCS_JWT_SECRET to a long random string.",
            UserWarning,
            stacklevel=3,
        )
        # Cache in env so the same secret is reused within one process
        os.environ[_JWT_SECRET_ENV] = secret
    return secret


def _jwt_expiry_hours() -> int:
    try:
        return int(os.environ.get(_JWT_EXPIRY_ENV, "8"))
    except ValueError:
        return 8


# ---------------------------------------------------------------------------
# UserStore – SQLite-backed users table
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT    NOT NULL UNIQUE,
    password_hash TEXT   NOT NULL,
    role         TEXT    NOT NULL DEFAULT 'viewer',
    created_at   TEXT    NOT NULL,
    last_login   TEXT
);
"""


class UserStore:
    """
    Thin SQLite wrapper for the ``users`` table.

    bcrypt is used for password hashing (cost factor 12).  The raw password
    is never stored or logged.

    Args:
        db_path: Path to the SQLite file.  Parent dirs are created
                 automatically.  Defaults to ``PCS_AUTH_DB`` env var or
                 ``data/users.db``.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            db_path = os.environ.get(_AUTH_DB_ENV, "data/users.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_DDL)
        self._conn.commit()
        # Lazy-import bcrypt so the module can be imported without it installed
        # (tests that don't exercise password hashing still import fine).
        try:
            import bcrypt as _bcrypt
            self._bcrypt = _bcrypt
        except ImportError:  # pragma: no cover
            self._bcrypt = None  # type: ignore[assignment]
            warnings.warn(
                "bcrypt is not installed; password hashing is unavailable.",
                ImportWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Password helpers
    # ------------------------------------------------------------------

    def _hash(self, password: str) -> str:
        """Return a bcrypt hash string for *password*."""
        if self._bcrypt is None:  # pragma: no cover
            raise RuntimeError("bcrypt is not installed")
        raw = password.encode("utf-8")
        return self._bcrypt.hashpw(raw, self._bcrypt.gensalt(rounds=12)).decode("utf-8")

    def _verify_pw(self, password: str, pw_hash: str) -> bool:
        """Return True if *password* matches the stored *pw_hash*."""
        if self._bcrypt is None:  # pragma: no cover
            raise RuntimeError("bcrypt is not installed")
        return self._bcrypt.checkpw(
            password.encode("utf-8"),
            pw_hash.encode("utf-8"),
        )

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    def _now_iso(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def create_user(self, username: str, password: str, role: str = "viewer") -> dict:
        """
        Create a new user.

        Args:
            username: Must be unique.
            password: Plain-text password (hashed before storage).
            role:     One of ``viewer``, ``operator``, ``admin``.

        Returns:
            The newly created user record (without password hash).

        Raises:
            ValueError: If the role is unknown or the username already exists.
        """
        if role not in ROLES_ORDERED:
            raise ValueError(f"role must be one of {ROLES_ORDERED!r}, got {role!r}")
        pw_hash  = self._hash(password)
        now      = self._now_iso()
        try:
            self._conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (username, pw_hash, role, now),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Username {username!r} already exists")
        log.info("UserStore: created user %r with role %r", username, role)
        return {"username": username, "role": role, "created_at": now}

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """
        Verify *password* for *username*.

        Returns the user dict (id, username, role, last_login) on success, or
        ``None`` on failure.  Updates ``last_login`` on success.
        """
        row = self._conn.execute(
            "SELECT id, username, password_hash, role FROM users WHERE username=?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        if not self._verify_pw(password, row["password_hash"]):
            return None
        now = self._now_iso()
        self._conn.execute(
            "UPDATE users SET last_login=? WHERE id=?", (now, row["id"])
        )
        self._conn.commit()
        return {"id": row["id"], "username": row["username"],
                "role": row["role"], "last_login": now}

    def get_user(self, username: str) -> Optional[dict]:
        """Return user info (without hash) or None."""
        row = self._conn.execute(
            "SELECT id, username, role, created_at, last_login FROM users WHERE username=?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_users(self) -> list[dict]:
        """Return all users (without password hashes)."""
        rows = self._conn.execute(
            "SELECT id, username, role, created_at, last_login FROM users ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_role(self, username: str, new_role: str) -> bool:
        """Change the role for *username*. Returns False if user not found."""
        if new_role not in ROLES_ORDERED:
            raise ValueError(f"Unknown role {new_role!r}")
        cur = self._conn.execute(
            "UPDATE users SET role=? WHERE username=?", (new_role, username)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_password(self, username: str, new_password: str) -> bool:
        """Replace the password hash for *username*. Returns False if not found."""
        pw_hash = self._hash(new_password)
        cur = self._conn.execute(
            "UPDATE users SET password_hash=? WHERE username=?", (pw_hash, username)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_user(self, username: str) -> bool:
        """Remove *username*. Returns False if not found."""
        cur = self._conn.execute("DELETE FROM users WHERE username=?", (username,))
        self._conn.commit()
        return cur.rowcount > 0

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def ensure_default_admin(self) -> None:
        """
        Create the default ``admin`` account if the users table is empty.

        Logs a prominent warning so operators immediately know to change the
        password.
        """
        if self.count() > 0:
            return
        self.create_user(DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASSWORD, role="admin")
        log.warning(
            "⚠  No users found — created default admin account.  "
            "Username: %r  Password: %r  "
            "CHANGE THIS IMMEDIATELY via POST /auth/users/admin/password",
            DEFAULT_ADMIN_USER,
            DEFAULT_ADMIN_PASSWORD,
        )

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_token(user_id: int, username: str, role: str) -> str:
    """
    Issue a signed JWT for *username* with the given *role*.

    Claims:
        sub   – username
        uid   – user DB id
        role  – one of viewer / operator / admin
        iat   – issued-at (UTC unix timestamp)
        exp   – expiry (iat + PCS_JWT_EXPIRY_H hours)
    """
    import jwt

    now    = datetime.datetime.now(datetime.timezone.utc)
    expiry = now + datetime.timedelta(hours=_jwt_expiry_hours())
    payload = {
        "sub":  username,
        "uid":  user_id,
        "role": role,
        "iat":  now,
        "exp":  expiry,
    }
    return jwt.encode(payload, _resolve_jwt_secret(), algorithm="HS256")


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT.

    Returns the payload dict on success.

    Raises:
        jwt.ExpiredSignatureError  – token has expired
        jwt.InvalidTokenError      – any other JWT error (bad sig, malformed…)
    """
    import jwt

    return jwt.decode(
        token,
        _resolve_jwt_secret(),
        algorithms=["HS256"],
    )


# ---------------------------------------------------------------------------
# Request-level helpers for Flask routes
# ---------------------------------------------------------------------------

def get_token_from_request() -> Optional[str]:
    """
    Extract the JWT from the current Flask request.

    Priority:
    1. ``Authorization: Bearer <token>`` header
    2. ``pcs_token`` cookie
    3. ``token`` query-string parameter (WebSocket handshake fallback)
    """
    from flask import request as _req

    # 1. Bearer header
    auth_header = _req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    # 2. Cookie
    cookie = _req.cookies.get("pcs_token", "")
    if cookie:
        return cookie.strip()

    # 3. Query param (Socket.IO handshake passes token here)
    token_qs = _req.args.get("token", "")
    if token_qs:
        return token_qs.strip()

    return None


def get_current_user() -> Optional[dict]:
    """
    Decode the JWT from the current request and return the payload.

    Returns ``None`` when no token is present or the token is invalid / expired.
    The returned dict has keys: ``sub`` (username), ``uid``, ``role``.
    """
    token = get_token_from_request()
    if not token:
        return None
    try:
        return decode_token(token)
    except Exception:
        return None


def role_rank(role: str) -> int:
    """Return integer rank of *role* (higher = more privileged)."""
    try:
        return ROLES_ORDERED.index(role)
    except ValueError:
        return -1


def has_role(user: dict, *required_roles: str) -> bool:
    """
    Return True if *user*'s role is in *required_roles* **or** the user is an
    admin (admins bypass every role check).
    """
    if user is None:
        return False
    user_role = user.get("role", "")
    return user_role in required_roles or user_role == "admin"


# ---------------------------------------------------------------------------
# Flask route decorators
# ---------------------------------------------------------------------------

def _is_browser_request() -> bool:
    """Heuristic: True when the client looks like a browser (HTML Accept header)."""
    from flask import request as _req
    accept = _req.headers.get("Accept", "")
    return "text/html" in accept


def require_auth(f: Callable) -> Callable:
    """Decorator: require any valid JWT (any role).

    Returns 401 JSON for API calls; redirects to ``/auth/login`` for browsers.
    """
    @wraps(f)
    def _inner(*args, **kwargs):
        from flask import jsonify, redirect, url_for
        user = get_current_user()
        if user is None:
            if _is_browser_request():
                from flask import request as _req
                return redirect(f"/auth/login?next={_req.path}")
            return jsonify(error="Authentication required", code="unauthenticated"), 401
        return f(*args, **kwargs)
    return _inner


def require_role(*roles: str) -> Callable:
    """
    Decorator factory: require the caller to have one of *roles* (or admin).

    Usage::

        @app.post("/api/alerts/<int:id>/ack")
        @require_role("operator", "admin")
        def ack(id):
            ...
    """
    def _decorator(f: Callable) -> Callable:
        @wraps(f)
        def _inner(*args, **kwargs):
            from flask import jsonify, redirect
            user = get_current_user()
            if user is None:
                if _is_browser_request():
                    from flask import request as _req
                    return redirect(f"/auth/login?next={_req.path}")
                return jsonify(error="Authentication required", code="unauthenticated"), 401
            if not has_role(user, *roles):
                return jsonify(
                    error=f"Insufficient privilege.  Required: {list(roles)}  "
                           f"Your role: {user.get('role')}",
                    code="forbidden",
                ), 403
            return f(*args, **kwargs)
        return _inner
    return _decorator


def socket_auth(auth_dict: Optional[dict]) -> Optional[dict]:
    """
    Validate a Socket.IO connection's auth payload.

    Call from the ``connect`` handler::

        @socketio.on("connect")
        def on_connect(auth):
            user = socket_auth(auth)
            if user is None:
                return False   # reject connection

    Args:
        auth_dict: The ``auth`` argument passed by the Socket.IO client,
                   expected to be ``{"token": "<jwt>"}``.

    Returns:
        Decoded payload dict or ``None`` when the token is absent / invalid.
    """
    if not auth_dict:
        return None
    token = auth_dict.get("token") or auth_dict.get("Token") or ""
    if not token:
        return None
    try:
        return decode_token(token)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Flask Blueprint with auth routes
# ---------------------------------------------------------------------------

def create_auth_blueprint(user_store: "UserStore"):
    """
    Return a Flask Blueprint that registers:

    GET  /auth/login                  – Login page (HTML)
    POST /auth/login                  – Authenticate and issue JWT cookie
    POST /auth/logout                 – Clear the JWT cookie
    GET  /auth/me                     – Return current user info (JSON)
    GET  /auth/users                  – List users (admin only)
    POST /auth/users                  – Create user (admin only)
    PUT  /auth/users/<username>/role     – Change role (admin only)
    POST /auth/users/<username>/password – Change password (admin or self)
    DELETE /auth/users/<username>        – Delete user (admin only)
    """
    from flask import (
        Blueprint, jsonify, make_response, redirect,
        render_template, request, url_for,
    )

    bp = Blueprint("auth", __name__, url_prefix="/auth")

    # ------------------------------------------------------------------
    # Login page + action
    # ------------------------------------------------------------------

    @bp.get("/login")
    def login_page():
        """Serve the login HTML page."""
        # If already authenticated, go straight to dashboard
        if get_current_user() is not None:
            return redirect("/")
        error = request.args.get("error", "")
        next_url = request.args.get("next", "/")
        return render_template("login.html", error=error, next_url=next_url)

    @bp.post("/login")
    def login_action():
        """
        Authenticate via form-post or JSON.

        Accepts:
          * ``application/x-www-form-urlencoded`` fields: ``username``, ``password``
          * ``application/json`` body: ``{"username": ..., "password": ...}``

        On success: sets ``pcs_token`` cookie + returns JSON with token.
        On failure: 401 JSON (API) or redirect back to login page (browser).
        """
        ct = request.content_type or ""
        if "application/json" in ct:
            data = request.get_json(silent=True) or {}
            username = data.get("username", "")
            password = data.get("password", "")
            next_url = data.get("next", "/")
        else:
            username = request.form.get("username", "")
            password = request.form.get("password", "")
            next_url = request.form.get("next", "/")

        username = username.strip()
        password = password.strip()

        user = user_store.authenticate(username, password)
        if user is None:
            log.warning("auth: failed login attempt for %r", username)
            if _is_browser_request():
                return redirect(f"/auth/login?error=Invalid+credentials&next={next_url}")
            return jsonify(error="Invalid credentials"), 401

        token = create_token(user["id"], user["username"], user["role"])
        log.info("auth: user %r (%s) logged in", username, user["role"])

        if _is_browser_request():
            resp = make_response(redirect(next_url or "/"))
        else:
            resp = make_response(
                jsonify(
                    token    = token,
                    username = user["username"],
                    role     = user["role"],
                )
            )
        # Secure, httponly cookie valid for JWT_EXPIRY hours
        max_age = _jwt_expiry_hours() * 3600
        resp.set_cookie(
            "pcs_token",
            token,
            max_age   = max_age,
            httponly  = True,
            samesite  = "Lax",
        )
        return resp

    @bp.post("/logout")
    def logout():
        """Clear the auth cookie."""
        resp = make_response(jsonify(ok=True, message="Logged out"))
        resp.delete_cookie("pcs_token")
        return resp

    # ------------------------------------------------------------------
    # Current-user info
    # ------------------------------------------------------------------

    @bp.get("/me")
    @require_auth
    def me():
        """Return the decoded JWT payload for the current caller."""
        user = get_current_user()
        return jsonify(
            username = user["sub"],
            role     = user["role"],
            uid      = user["uid"],
        )

    # ------------------------------------------------------------------
    # User management (admin only)
    # ------------------------------------------------------------------

    @bp.get("/users")
    @require_role("admin")
    def list_users():
        """Return all users without password hashes."""
        return jsonify(users=user_store.list_users())

    @bp.post("/users")
    @require_role("admin")
    def create_user():
        """
        Create a new user.

        Body JSON: ``{username, password, role}``
        """
        body = request.get_json(silent=True) or {}
        username = (body.get("username") or "").strip()
        password = (body.get("password") or "").strip()
        role     = (body.get("role")     or "viewer").strip()
        if not username or not password:
            return jsonify(error="username and password are required"), 400
        try:
            user = user_store.create_user(username, password, role)
        except ValueError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(user), 201

    @bp.put("/users/<username>/role")
    @require_role("admin")
    def update_role(username: str):
        """Change a user's role.  Body JSON: ``{role}``"""
        body = request.get_json(silent=True) or {}
        new_role = (body.get("role") or "").strip()
        try:
            ok = user_store.update_role(username, new_role)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if not ok:
            return jsonify(error=f"User {username!r} not found"), 404
        return jsonify(ok=True, username=username, role=new_role)

    @bp.post("/users/<username>/password")
    def update_password(username: str):
        """
        Change a user's password.

        An admin can change any password.  A non-admin user can only change
        their own.  Body JSON: ``{password}``
        """
        caller = get_current_user()
        if caller is None:
            return jsonify(error="Authentication required"), 401
        if caller["role"] != "admin" and caller["sub"] != username:
            return jsonify(error="Cannot change another user's password"), 403

        body = request.get_json(silent=True) or {}
        new_pw = (body.get("password") or "").strip()
        if not new_pw:
            return jsonify(error="password is required"), 400
        ok = user_store.update_password(username, new_pw)
        if not ok:
            return jsonify(error=f"User {username!r} not found"), 404
        log.info("auth: password updated for %r by %r", username, caller["sub"])
        return jsonify(ok=True, username=username)

    @bp.delete("/users/<username>")
    @require_role("admin")
    def delete_user(username: str):
        """Delete a user account."""
        caller = get_current_user()
        if caller and caller["sub"] == username:
            return jsonify(error="Cannot delete your own account"), 400
        ok = user_store.delete_user(username)
        if not ok:
            return jsonify(error=f"User {username!r} not found"), 404
        return jsonify(ok=True, deleted=username)

    return bp
