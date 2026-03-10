"""
flask_app.py – Production-grade Flask dashboard for Privacy Campus Security.

Routes
------
GET  /                        – Full dashboard with summary cards, charts, events feed
GET  /enroll/screen           – Single-screen camera-grid enrollment UI
GET  /admin/campus            – Campus overview map (Leaflet.js with live camera status)
GET  /alerts/mobile           – Mobile-friendly on-call admin view (Ack + Navigate + Panic)
GET  /api/health              – Health-check JSON  { status, enrolled, uptime_s }
GET  /api/events              – Last N events JSON  (query param: ?n=50)
GET  /api/metrics             – Evaluation metrics JSON from experiments/ CSVs & JSONs
GET  /api/metrics/reload      – Force-reload metrics cache then return updated data
GET  /api/cameras             – Camera registry JSON (id, name, door, topics, coords)
GET  /api/camera/status       – Live camera status dict {camera_id: {status, last_event, last_seen}}
GET  /api/alerts              – Recent security alerts (forced_entry, tailgate, safety_alert)
POST /api/alerts/<id>/ack     – Acknowledge an alert (body: {acked_by?}); stores audit trail
POST /api/alerts/panic        – PANIC escalation: bypass throttle, dispatch SMS immediately
GET  /api/camera/<id>/snapshot – Latest JPEG frame for a camera (from MQTT feed)
GET  /api/left-objects         – List left-object detection events (query: n, camera_id, dismissed, confirmed)
POST /api/left-objects/<id>/remove  – Dismiss (object removed); body {action_by?}
POST /api/left-objects/<id>/confirm – Confirm as threat; body {action_by?}
POST /api/left-objects/trigger      – Manually inject a test left_object event (JSON: {camera_id})
GET  /api/loitering               – List loitering events (query: n, camera_id, subject_key, dismissed)
POST /api/loitering/<id>/dismiss  – Dismiss loitering event; body {action_by?}
POST /api/loitering/trigger       – Manually inject a test loitering event (JSON: {camera_id, subject_key?, dwell_s?})
GET  /api/notifications           – List notification delivery records (query: n, channel, status, alert_id, alert_type)
POST /api/enroll              – Enroll a subject (multipart: subject_id, image)
POST /api/enroll/batch        – Detect all faces in image, enroll each anonymously (multipart: image, group_id)
POST /api/verify              – Verify a subject (multipart: subject_id, image)
POST /api/verify/frame        – Detect all faces, return top-k matches per face (vectorised; multipart: image, k, thresh, camera_id?, door_id?)
POST /api/attendance/mark     – Record one or many attendance marks (JSON body); session TTL deduplication
GET  /api/attendance          – Aggregated attendance per subject & per camera (query: from, to, group_id)

Access Control Engine
---------------------
Door sensor events arrive via MQTT on ``campus/<door_id>/event`` (JSON: ``{type, door_id, timestamp}``).
These are correlated with face-verify results to produce two alert types:

* **forced_entry** – door ``open`` event and a failed verify within ±2 s of the same door.
* **tailgate** – ≥ 2 persons at the same door within 5 s and at least one is unmatched.

In addition, safety signals produce:

* **safety_alert** – high-severity smoke/fire indication from either
    MQTT sensor telemetry (``campus/<camera_id>/sensor`` with ``{smoke:true}``)
    or a lightweight frame heuristic (orange/red blob + motion + persistence).

* **loitering** – a recognised face (``subject_key``) is continuously present in
    a camera's field of view for longer than ``PCS_LOITER_DWELL_S`` seconds.
    Detection runs on MQTT frame bytes, throttled to once per ``PCS_LOITER_FRAME_INTERVAL_S``
    seconds per camera to keep CPU load low.

Alerts are persisted to the ``alerts`` SQLite table (same DB as attendance) and broadcast
over Socket.IO as ``alert`` events.

React / SPA upgrade path
------------------------
Set ``REACT_BUILD_DIR`` env var to your Vite/CRA ``dist/`` folder and this
app will serve it as a static SPA.  All ``/api/*`` routes stay intact and
become the backend the React app consumes.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit

try:
    import structlog as _structlog
except ImportError:  # pragma: no cover
    _structlog = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

STORE_PATH        = os.environ.get("PCS_STORE",          "data/embeddings.json")
ENCRYPT           = os.environ.get("PCS_ENCRYPT",        "0") == "1"
CORS_ORIGINS      = os.environ.get("CORS_ORIGINS",        "")   # comma-separated origins
REACT_BUILD       = os.environ.get("REACT_BUILD_DIR",     "")
# Attendance DB — defaults to attendance.db beside the main store
_default_att_path = str(Path(STORE_PATH).parent / "attendance.db")
ATTENDANCE_PATH   = os.environ.get("PCS_ATTENDANCE_DB", _default_att_path)
# Auth DB — defaults to users.db beside the main store
_default_auth_path = str(Path(STORE_PATH).parent / "users.db")
AUTH_DB_PATH       = os.environ.get("PCS_AUTH_DB", _default_auth_path)
# Audit DB — defaults to audit.db beside the main store
_default_audit_path = str(Path(STORE_PATH).parent / "audit.db")
AUDIT_DB_PATH       = os.environ.get("PCS_AUDIT_DB", _default_audit_path)

_START_TIME  = time.monotonic()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """
    Flask application factory.

    Returns a fully-configured :class:`Flask` app.  Heavy ML dependencies
    (PyTorch, backbone, embedding store) are imported lazily inside route
    handlers so the module can be imported in test environments without a GPU.

    Extending this app
    ------------------
    * **New ML model** – add a loader call after the ``model = ...`` line and
      inject it into any route via closure.
    * **MQTT** – start a ``paho.mqtt.client`` in a daemon thread here, calling
      ``data_provider.record_event()`` on each message.
    * **WebSocket** – replace the ``/api/events`` polling endpoint with
      ``flask-socketio``; the JS client in ``dashboard.html`` has a labelled
      comment showing where to swap in ``socket.on('event', handler)``.
    """
    # ── Auth: import RBAC helpers ─────────────────────────────────────────────
    from src.web.auth import (
        UserStore, create_auth_blueprint,
        require_auth, require_role, get_current_user,
        has_role,
    )

    # Initialise UserStore and seed default admin if DB is empty
    _user_store = UserStore(AUTH_DB_PATH)
    _user_store.ensure_default_admin()

    # ── Structured logging + Prometheus metrics ───────────────────────────────
    from src.logging_config import configure_logging
    configure_logging()

    # Structured logger (emits JSON with named fields when structlog is installed)
    _slog = (
        _structlog.get_logger(__name__)
        if _structlog is not None
        else log  # graceful fallback to stdlib
    )

    from src.web.metrics import (
        VERIFY_REQUESTS, VERIFY_LATENCY,
        ALERTS_TOTAL, ATTENDANCE_MARKS,
        metrics_response,
    )

    # Template folder lives at src/web/templates/
    template_dir = Path(__file__).parent / "templates"
    static_dir   = Path(__file__).parent / "static"

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir) if static_dir.exists() else None,
    )
    # Secret key required by Flask session (used by cookie‐based auth redirect)
    app.secret_key = os.environ.get("PCS_FLASK_SECRET",
                                     os.environ.get("PCS_JWT_SECRET", "pcs-dev-secret"))

    # Register the /auth/* blueprint (login, logout, user management)
    auth_bp = create_auth_blueprint(_user_store)
    app.register_blueprint(auth_bp)

    # ── SocketIO ──────────────────────────────────────────────────────────────
    # async_mode="threading" works on all platforms with no extra deps.
    # Switch to "eventlet" or "gevent" for higher concurrency in production.
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )
    app.extensions["socketio"] = socketio   # expose for __main__ / tests

    # Optional CORS headers for React dev-server proxy ────────────────────────
    if CORS_ORIGINS:
        try:
            from flask_cors import CORS  # type: ignore[import]
            CORS(app, origins=CORS_ORIGINS.split(","))
            log.info("CORS enabled for origins: %s", CORS_ORIGINS)
        except ImportError:
            log.warning("flask-cors not installed; CORS headers not set.")

    # Lazy-load ML dependencies ───────────────────────────────────────────────
    import numpy as _np
    import torch
    from PIL import Image

    from src.models.backbone import BackboneEmbedder
    from src.privacy.embedding_store import EmbeddingStore
    from src.web import data_provider

    from src.privacy.attendance_store import AttendanceStore

    store            = EmbeddingStore(STORE_PATH, encrypt=ENCRYPT)
    attendance_store = AttendanceStore(ATTENDANCE_PATH)
    model            = BackboneEmbedder().eval()

    # Alert store — sits in the same SQLite file as attendance
    from src.privacy.access_control import AlertStore, AccessControlEngine
    from src.privacy.audit_store import AuditStore
    _alert_store = AlertStore(ATTENDANCE_PATH)
    _audit_store = AuditStore(AUDIT_DB_PATH)

    # Wrap AlertStore.add so every new alert gets an immutable audit row
    _orig_alert_add = _alert_store.add
    def _audited_alert_add(*_args, **_kwargs):
        alert = _orig_alert_add(*_args, **_kwargs)
        if alert:
            _audit_store.append(
                "alert",
                actor      = "system",
                camera_id  = alert.get("camera_id"),
                alert_id   = alert.get("id"),
                detail     = {
                    "alert_type": alert.get("alert_type"),
                    "door_id":    alert.get("door_id"),
                    "subject_key": alert.get("subject_key"),
                    "confidence": alert.get("confidence"),
                },
            )
        return alert
    _alert_store.add = _audited_alert_add  # type: ignore[method-assign]

    # ── Notification dispatcher ────────────────────────────────────────────────
    from src.notifications.store      import NotificationStore
    from src.notifications.policy     import EscalationPolicy
    from src.notifications.drivers    import (
        WebSocketDriver, SmtpDriver, NullSmsDriver, load_sms_driver,
    )
    from src.notifications.dispatcher import NotificationDispatcher

    _notify_store  = NotificationStore(ATTENDANCE_PATH)
    _notify_policy = EscalationPolicy()

    # Build driver list; optionally replace NullSmsDriver with a custom adapter
    _notify_drivers: list = [
        WebSocketDriver(socketio),
        SmtpDriver(),
        NullSmsDriver(),
    ]
    _sms_driver_path = os.environ.get("PCS_SMS_DRIVER", "").strip()
    if _sms_driver_path:
        try:
            _custom_sms = load_sms_driver(_sms_driver_path)
            _notify_drivers = [d for d in _notify_drivers if d.name != "sms"]
            _notify_drivers.append(_custom_sms)
            log.info("[notify] loaded custom SMS driver: %s", _sms_driver_path)
        except Exception as _sms_exc:
            log.warning("[notify] could not load PCS_SMS_DRIVER=%r: %s",
                        _sms_driver_path, _sms_exc)

    _dispatcher = NotificationDispatcher(
        store    = _notify_store,
        policy   = _notify_policy,
        drivers  = _notify_drivers,
    )
    _dispatcher.start()
    log.info("[notify] dispatcher started with drivers: %s",
             [d.name for d in _notify_drivers])

    # Camera registry — loaded from configs/cameras.yaml; gracefully absent
    from src.privacy.camera_store import CameraRegistry
    _cam_yaml = Path(os.environ.get("PCS_CAMERAS_YAML", "configs/cameras.yaml"))
    try:
        _cam_registry = CameraRegistry(_cam_yaml)
        log.info("CameraRegistry: %d cameras loaded from %s", len(_cam_registry), _cam_yaml)
    except Exception as _cam_exc:
        _cam_registry = None
        log.warning("CameraRegistry unavailable (%s); /enroll/screen will show empty grid.", _cam_exc)

    # ── NumPy-2.x-safe preprocessing ─────────────────────────────────────────
    # torchvision.transforms.ToTensor() internally calls torch.from_numpy().
    # With NumPy 2.x the numpy C-API bridge is unavailable, causing a
    # RuntimeError at inference time.  Using torch.tensor() instead copies
    # the data without touching the bridge and works on any NumPy version.
    _IMG_SIZE = 112   # matches DEFAULT_TRANSFORM in lfw_loader.py

    def _pil_to_tensor(pil_img: Image.Image) -> "torch.Tensor":
        """
        Resize -> float32 [0,1] -> normalise [-1,1] -> (C,H,W) tensor.

        Completely numpy-free path: PIL.tobytes() -> bytearray ->
        torch.frombuffer() avoids every numpy C-API bridge call, which
        fails with NumPy >= 2.0 when torch.from_numpy / torch.tensor is used.
        """
        pil_img = pil_img.resize((_IMG_SIZE, _IMG_SIZE), Image.BILINEAR)
        raw     = bytearray(pil_img.tobytes())           # H*W*3  uint8 bytes
        t       = torch.frombuffer(raw, dtype=torch.uint8).clone()  # 1-D
        t       = t.reshape(_IMG_SIZE, _IMG_SIZE, 3)     # (H,W,C) uint8
        t       = t.float() / 255.0                      # [0,1]
        t       = (t - 0.5) / 0.5                        # [-1,1]
        return  t.permute(2, 0, 1)                       # (C,H,W)

    def _embed(file_storage) -> "_np.ndarray":
        """Return a 1-D float32 numpy embedding, bypassing the broken torch↔numpy bridge."""
        img    = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
        tensor = _pil_to_tensor(img).unsqueeze(0)                  # (1,C,H,W)
        with torch.no_grad():
            t = model(tensor).squeeze(0)                           # torch Tensor
        # t.tolist() → plain Python floats → np.array() never touches the C bridge
        return _np.array(t.tolist(), dtype=_np.float32)

    # ── JSON error handler (prevents HTML 500 pages reaching the JS client) ───
    @app.errorhandler(Exception)
    def _json_error(exc):
        log.exception("Unhandled exception in route")
        return jsonify(error=str(exc)), 500

    # ── Dashboard home ────────────────────────────────────────────────────────

    @app.get("/")
    @require_auth
    def index():
        """Render the full Bootstrap 5 dashboard."""
        caller          = get_current_user()
        metrics         = data_provider.get_metrics()
        k_shot_rows     = metrics["k_shot_rows"]
        pairs_metrics   = metrics["pairs_metrics"]
        roc_image_uri   = metrics["roc_image_uri"]
        enrolled        = len(store)
        recent_events   = data_provider.get_events(n=10)
        top1_best       = max(
            (float(r.get("mean_accuracy", 0)) for r in k_shot_rows),
            default=None,
        )
        return render_template(
            "dashboard.html",
            enrolled      = enrolled,
            recent_events = recent_events,
            k_shot_rows   = k_shot_rows,
            pairs_metrics = pairs_metrics,
            roc_image_uri = roc_image_uri,
            top1_best     = round(top1_best * 100, 1) if top1_best is not None else None,
            source_dir    = metrics["source_dir"],
            current_user  = caller,
        )

    # ── Single-screen enrollment UI ──────────────────────────────────────────

    @app.get("/enroll/screen")
    @require_role("operator", "admin")
    def enroll_screen():
        """
        Render the single-screen enroll UI showing a grid of campus cameras.

        Each camera card has an **Enroll** button.  Clicking it arms the camera
        via the ``arm_enroll`` Socket.IO event; the next frame that arrives on
        ``enroll_frame`` with a detected face triggers a batch enroll and emits
        ``enroll_result`` back to the admin, opening a metadata modal.
        """
        cameras = []
        if _cam_registry:
            cameras = [
                {
                    "camera_id": c.camera_id,
                    "name":      c.name,
                    "door_id":   c.door_id,
                    "lat":       c.lat,
                    "lon":       c.lon,
                    "active":    c.active,
                }
                for c in _cam_registry.list_all()
            ]
        return render_template(
            "enroll_screen.html",
            cameras        = cameras,
            total_enrolled = len(store),
            current_user   = get_current_user(),
        )

    @app.get("/api/cameras")
    def api_cameras():
        """Return the camera registry as JSON."""
        if not _cam_registry:
            return jsonify(cameras=[], source=str(_cam_yaml))
        cameras = [
            {
                "camera_id": c.camera_id,
                "name":      c.name,
                "door_id":   c.door_id,
                "lat":       c.lat,
                "lon":       c.lon,
                "active":    c.active,
                "frame_topic":  _cam_registry.frame_topic(c.camera_id),
                "sensor_topic": _cam_registry.sensor_topic(c.camera_id),
            }
            for c in _cam_registry.list_all()
        ]
        return jsonify(cameras=cameras, source=str(_cam_yaml))

    # ── Health check ──────────────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        """Return system health JSON."""
        return jsonify(
            status   = "ok",
            enrolled = len(store),
            uptime_s = round(time.monotonic() - _START_TIME, 1),
        )

    @app.get("/metrics")
    def prometheus_metrics():
        """
        Prometheus-format metrics scrape endpoint.

        Exposes four metric families:

        * ``verify_requests_total{status}``   – requests to POST /api/verify/frame
        * ``alerts_total{alert_type}``        – security alerts by type
        * ``attendance_marks_total``          – deduplicated attendance inserts
        * ``verify_latency_seconds``          – full-pipeline latency histogram

        Prometheus scrape config (prometheus.yml)
        -----------------------------------------
        .. code-block:: yaml

            scrape_configs:
              - job_name: privacy_campus_security
                static_configs:
                  - targets: ['localhost:5000']
                metrics_path: /metrics
        """
        from flask import make_response
        body, status, headers = metrics_response()
        resp = make_response(body, status)
        for k, v in headers.items():
            resp.headers[k] = v
        return resp

    # ── Events feed (polled by JS every 3 s) ──────────────────────────────────

    @app.get("/api/events")
    def events():
        """
        Return recent system events as JSON.

        Query params
        ------------
        n : int  Maximum number of events to return (default 50, max 500).

        WebSocket upgrade
        -----------------
        Replace this route with a ``flask-socketio`` namespace and emit
        events from ``data_provider.record_event()``.  The JS polling loop
        in dashboard.html has a ``/* WebSocket hook */`` comment block showing
        exactly where to call ``socket.on('pcs:event', handler)``.
        """
        try:
            n = min(int(request.args.get("n", 50)), 500)
        except ValueError:
            n = 50
        return jsonify(events=data_provider.get_events(n=n))

    # ── Metrics endpoint ──────────────────────────────────────────────────────

    @app.get("/api/metrics")
    def metrics_route():
        """
        Return all cached evaluation metrics as JSON.

        Response keys
        -------------
        k_shot_rows    : list[dict]  – few-shot sweep rows
        pairs_metrics  : dict        – pairs verification metrics
        episode_series : list[dict]  – Chart.js-ready accuracy series
        anomaly_scores : list[dict]  – anomaly score timeline
        roc_available  : bool        – whether a ROC PNG exists
        source_dir     : str         – experiments/ subfolder used
        """
        m = data_provider.get_metrics()
        return jsonify(
            k_shot_rows    = m["k_shot_rows"],
            pairs_metrics  = m["pairs_metrics"],
            episode_series = data_provider.get_episode_accuracy_series(),
            anomaly_scores = data_provider.get_anomaly_timeline(n=200),
            roc_available  = m["roc_image_uri"] is not None,
            source_dir     = m["source_dir"],
        )

    @app.get("/api/metrics/reload")
    def metrics_reload():
        """Force a cache refresh and return updated metrics."""
        m = data_provider.get_metrics(force_reload=True)
        return jsonify(
            k_shot_rows    = m["k_shot_rows"],
            pairs_metrics  = m["pairs_metrics"],
            episode_series = data_provider.get_episode_accuracy_series(),
            roc_available  = m["roc_image_uri"] is not None,
            source_dir     = m["source_dir"],
            reloaded       = True,
        )

    # ── Enroll ────────────────────────────────────────────────────────────────

    @app.post("/api/enroll")
    @require_role("operator", "admin")
    def enroll():
        """Enroll a subject; adds an event to the live feed."""
        subject_id = request.form.get("subject_id", "").strip()
        image_file = request.files.get("image")
        if not subject_id or not image_file:
            return jsonify(error="subject_id and image are required"), 400
        embedding = _embed(image_file)
        store.add(subject_id, embedding)
        store.save()
        data_provider.record_event("enroll", subject_id=subject_id)
        caller = get_current_user()
        _audit_store.append(
            "enroll",
            actor      = caller["sub"] if caller else "unknown",
            subject_id = subject_id,
            detail     = {"total_enrolled": len(store)},
        )
        log.info("Enrolled subject '%s'. Total enrolled: %d", subject_id, len(store))
        return jsonify(status="enrolled", subject_id=subject_id, total_enrolled=len(store))

    # ── Batch enroll (multi-face) ─────────────────────────────────────────────

    @app.post("/api/enroll/batch")
    @require_role("operator", "admin")
    def enroll_batch():
        """
        Detect every face in the uploaded image and enroll each as a separate,
        privacy-safe anonymous record.

        For every detected face a unique opaque subject key is generated::

            grp:<group_id>:f<face_idx>:<secrets.token_hex(16)>

        Only the ``SHA-256(salt || key)`` hash is written to the embedding store;
        the plaintext key is never persisted.  The caller receives the list of
        plaintext keys (to use for future ``/api/verify`` calls) along with the
        count of enrolled subjects.

        Form fields
        -----------
        image    : file  (required) — JPEG / PNG image possibly containing multiple faces
        group_id : str   (optional, default ``"default"``) — group tag embedded in each key

        Response JSON
        -------------
        {
          "n_enrolled":     int,          -- number of faces successfully enrolled
          "keys":           list[str],    -- generated plaintext subject keys
          "group_id":       str,
          "total_enrolled": int           -- total subjects now in the store
        }
        """
        import secrets

        from src.web.face_utils import generate_embedding_from_image

        group_id   = (request.form.get("group_id") or "default").strip()
        image_file = request.files.get("image")
        if not image_file:
            return jsonify(error="image is required"), 400

        pil_img = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        results = generate_embedding_from_image(pil_img, model=model, device="cpu")

        enrolled_keys: list[str] = []
        for face in results:
            token       = secrets.token_hex(16)
            subject_key = f"grp:{group_id}:f{face['face_id_local']}:{token}"
            store.add(subject_key, face["embedding"])
            enrolled_keys.append(subject_key)

        store.save()
        data_provider.record_event(
            "enroll_batch",
            meta={"n_enrolled": len(enrolled_keys), "group_id": group_id},
        )
        caller = get_current_user()
        _audit_store.append(
            "enroll_batch",
            actor  = caller["sub"] if caller else "unknown",
            detail = {
                "n_enrolled":     len(enrolled_keys),
                "group_id":       group_id,
                "total_enrolled": len(store),
            },
        )
        log.info(
            "Batch enrolled %d face(s), group='%s'. Total: %d",
            len(enrolled_keys), group_id, len(store),
        )
        return jsonify(
            n_enrolled     = len(enrolled_keys),
            keys           = enrolled_keys,
            group_id       = group_id,
            total_enrolled = len(store),
        )

    # ── Verify ────────────────────────────────────────────────────────────────

    @app.post("/api/verify")
    def verify():
        """Verify a subject against the embedding store; adds an event."""
        import numpy as np

        subject_id = request.form.get("subject_id", "").strip()
        image_file = request.files.get("image")
        if not subject_id or not image_file:
            return jsonify(error="subject_id and image are required"), 400
        if subject_id not in store:
            return jsonify(error="Subject not enrolled"), 404
        query_emb  = _embed(image_file)
        stored_emb = store.get(subject_id)
        similarity = float(np.dot(query_emb, stored_emb))
        match      = similarity >= 0.0
        data_provider.record_event(
            "verify",
            subject_id=subject_id,
            meta={"similarity": round(similarity, 4), "match": match},
        )
        _audit_store.append(
            "verify",
            actor      = "unauthenticated",
            subject_id = subject_id,
            detail     = {"similarity": round(similarity, 4), "match": match},
        )
        log.info("Verify '%s' | sim=%.4f | match=%s", subject_id, similarity, match)
        return jsonify(
            subject_id = subject_id,
            similarity = round(similarity, 4),
            match      = match,
        )

    # ── Live camera enroll (base64 JSON) ───────────────────────────────────────

    @app.post("/api/enroll/frame")
    @require_role("operator", "admin")
    def enroll_frame():
        """
        Enroll via a webcam snapshot.

        Body (JSON)
        -----------
        subject_id : str
        image_b64  : str  base64 JPEG/PNG, with or without data-URI prefix
        """
        import base64

        payload    = request.get_json(force=True, silent=True) or {}
        subject_id = (payload.get("subject_id") or "").strip()
        image_b64  = payload.get("image_b64", "")
        if not subject_id or not image_b64:
            return jsonify(error="subject_id and image_b64 are required"), 400

        raw = base64.b64decode(image_b64.split(",")[-1])   # strip data-URI prefix if present

        class _Buf:                                         # duck-type FileStorage.read()
            def read(self_):
                return raw

        embedding = _embed(_Buf())
        store.add(subject_id, embedding)
        store.save()
        data_provider.record_event("enroll", subject_id=subject_id,
                                   meta={"source": "camera"})
        caller = get_current_user()
        _audit_store.append(
            "enroll",
            actor      = caller["sub"] if caller else "unknown",
            subject_id = subject_id,
            detail     = {"source": "camera", "total_enrolled": len(store)},
        )
        log.info("[camera] Enrolled '%s'. Total: %d", subject_id, len(store))
        return jsonify(status="enrolled", subject_id=subject_id,
                       total_enrolled=len(store))

    # ── Live per-frame verification (base64 JSON, silent) ────────────────────

    @app.post("/api/verify/live")
    def verify_live():
        """
        Per-frame live verification for the webcam verify loop.

        Silent: does NOT write to the event log (avoids flooding the feed
        at 4 fps).  The existing /api/verify (file upload) still logs.

        Body (JSON)
        -----------
        subject_id : str
        image_b64  : str  base64 JPEG/PNG, with or without data-URI prefix

        Returns
        -------
        {similarity, match, latency_ms}
        """
        import base64
        import time

        import numpy as np

        t0         = time.monotonic()
        payload    = request.get_json(force=True, silent=True) or {}
        subject_id = (payload.get("subject_id") or "").strip()
        image_b64  = payload.get("image_b64", "")
        if not subject_id or not image_b64:
            return jsonify(error="subject_id and image_b64 are required"), 400
        if subject_id not in store:
            return jsonify(error="Subject not enrolled"), 404

        raw = base64.b64decode(image_b64.split(",")[-1])

        class _Buf:
            def read(self_):
                return raw

        query_emb  = _embed(_Buf())
        stored_emb = store.get(subject_id)
        similarity = float(np.dot(query_emb, stored_emb))
        match      = similarity >= 0.0
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        return jsonify(similarity=round(similarity, 4), match=match,
                       latency_ms=latency_ms)

    # ── Multi-face frame verify ───────────────────────────────────────────────

    @app.post("/api/verify/frame")
    def verify_frame():
        """
        Detect every face in the uploaded image and return per-face top-k matches
        from the embedding store using a single vectorised matrix multiply.

        Form fields
        -----------
        image  : file  (required) — JPEG / PNG image (may contain multiple faces)
        k      : int   (optional, default 3)  — top-k matches per face
        thresh : float (optional, default 0.0) — minimum cosine similarity to include

        Response JSON
        -------------
        {
          "n_faces":    int,
          "latency_ms": float,
          "faces": [
            {
              "face_id":     int,
              "bbox":        [x, y, w, h],
              "top_matches": [
                {"subject_key": str,   -- hashed store key (opaque)
                 "similarity":  float,
                 "match":       bool},  -- True when sim >= thresh (>= 0 if unset)
                ...
              ]
            }, ...
          ]
        }
        """
        import time as _time

        from src.web.face_utils import generate_embedding_from_image

        t0         = _time.monotonic()
        image_file = request.files.get("image")
        if not image_file:
            VERIFY_REQUESTS.labels(status="error").inc()
            return jsonify(error="image is required"), 400

        try:
            k      = int(request.form.get("k",      3))
            thresh = float(request.form.get("thresh", 0.0))
        except ValueError:
            VERIFY_REQUESTS.labels(status="error").inc()
            return jsonify(error="k must be int, thresh must be float"), 400

        # Optional access-control context fields
        _req_camera_id = (request.form.get("camera_id") or "").strip() or None
        _req_door_id   = (request.form.get("door_id")   or "").strip() or None
        # If camera_id known but door_id not given, resolve from registry
        if _req_camera_id and not _req_door_id and _cam_registry:
            _c = _cam_registry.get(_req_camera_id)
            if _c:
                _req_door_id = _c.door_id

        pil_img  = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        results  = generate_embedding_from_image(pil_img, model=model, device="cpu")

        if not results:
            return jsonify(n_faces=0, latency_ms=0.0, faces=[])

        # Stack all embeddings for a single batched matmul
        import numpy as _np
        emb_matrix = _np.stack([r["embedding"] for r in results])   # (M, 512)
        batch_matches = store.query_topk_batch(emb_matrix, k=k, threshold=thresh)

        import datetime as _dt_vf
        _ts_vf = _dt_vf.datetime.utcnow().isoformat() + "Z"

        faces_out = []
        for face, top_matches in zip(results, batch_matches):
            faces_out.append({
                "face_id":     face["face_id_local"],
                "bbox":        list(face["bbox"]),
                "top_matches": [
                    {
                        "subject_key": hid,
                        "similarity":  round(sim, 4),
                        "match":       sim >= max(thresh, 0.0),
                    }
                    for hid, sim in top_matches
                ],
            })
            # Feed the best match (or no-match) into the access-control engine
            if top_matches:
                _best_key, _best_sim = top_matches[0]
                _best_matched = _best_sim >= max(thresh, 0.0)
            else:
                _best_key, _best_sim, _best_matched = None, 0.0, False
            _ace.on_verify_result(
                camera_id=_req_camera_id,
                door_id=_req_door_id,
                subject_key=_best_key,
                matched=_best_matched,
                similarity=float(_best_sim),
                timestamp_utc=_ts_vf,
            )

        latency_ms = round((_time.monotonic() - t0) * 1000, 1)
        data_provider.record_event(
            "verify",
            meta={"n_faces": len(faces_out), "source": "frame", "latency_ms": latency_ms},
        )
        VERIFY_LATENCY.observe(latency_ms / 1000.0)
        VERIFY_REQUESTS.labels(status="success").inc()
        n_matched = sum(
            1 for f in faces_out
            if f["top_matches"] and f["top_matches"][0]["match"]
        )
        _audit_store.append(
            "verify_frame",
            actor     = "unauthenticated",
            camera_id = _req_camera_id,
            detail    = {
                "n_faces":    len(faces_out),
                "n_matched":  n_matched,
                "latency_ms": latency_ms,
            },
        )
        _slog.info(
            "verify_frame",
            event_type  = "verify",
            camera_id   = _req_camera_id,
            n_faces     = len(faces_out),
            latency_ms  = latency_ms,
        )

        # ── Camera Monitor: update per-camera event trail ─────────────────────
        if _req_camera_id and faces_out:
            cs = _camera_status.setdefault(_req_camera_id, {})
            # Best similarity across all detected faces in this frame
            best_sims = [
                f["top_matches"][0]["similarity"]
                for f in faces_out if f["top_matches"]
            ]
            if best_sims:
                cs["last_similarity"] = round(float(max(best_sims)), 4)
                cs["last_matched"] = cs["last_similarity"] >= max(float(request.form.get("thresh", 0.0)), 0.0)
            # Only upgrade status to "activity" — never downgrade from "alert"
            if cs.get("status", "idle") != "alert":
                cs["status"] = "activity"
                cs["last_event"] = "verify"
                cs["last_seen"] = _ts_vf
            # Append compact event entries, keep last 3 per camera
            ev_q = cs.setdefault("recent_events", [])
            for face in faces_out:
                tm = face["top_matches"][0] if face["top_matches"] else {}
                ev_q.append({
                    "ts":         _ts_vf,
                    "type":       "verify",
                    "matched":    bool(tm.get("match", False)),
                    "similarity": round(float(tm.get("similarity", 0.0)), 3),
                })
            cs["recent_events"] = ev_q[-3:]

        return jsonify(n_faces=len(faces_out), latency_ms=latency_ms, faces=faces_out)

    # ── Attendance endpoints ─────────────────────────────────────────────────

    @app.post("/api/attendance/mark")
    def attendance_mark():
        """
        Record one or many attendance marks.

        The request body must be JSON.  It may be either a **single record**
        (dict) or a **list of records** (array).  Each record accepts:

        Fields
        ------
        subject_key   : str   (required) – opaque subject identifier
        similarity    : float (required) – cosine similarity score
        matched       : bool  (required) – verification result
        group_id      : str   (optional, default "")
        camera_id     : str   (optional, default "default")
        timestamp_utc : str   (optional) – ISO-8601; defaults to *now*

        Session TTL
        -----------
        If the same ``(subject_key, camera_id)`` already has a row within the
        last ``session_ttl_s`` seconds (default 300), the duplicate is silently
        skipped and counted in ``n_skipped``.

        Response JSON
        -------------
        {n_marked, n_skipped, total_attendance, session_ttl_s}
        """
        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify(error="JSON body required"), 400

        records: list[dict] = body if isinstance(body, list) else [body]
        if not records:
            return jsonify(error="No records provided"), 400

        for i, rec in enumerate(records):
            if "subject_key" not in rec:
                return jsonify(error=f"Record {i}: subject_key is required"), 400
            if "similarity" not in rec:
                return jsonify(error=f"Record {i}: similarity is required"), 400
            if "matched" not in rec:
                return jsonify(error=f"Record {i}: matched is required"), 400

        n_marked, n_skipped = attendance_store.mark_many(records)
        data_provider.record_event(
            "attendance",
            meta={"n_marked": n_marked, "n_skipped": n_skipped},
        )
        ATTENDANCE_MARKS.inc(n_marked)
        _slog.info(
            "attendance_mark",
            event_type = "attendance",
            n_marked   = n_marked,
            n_skipped  = n_skipped,
            ttl_s      = attendance_store._ttl,
        )
        # Push fresh summary to all connected dashboard clients.
        if n_marked > 0:
            try:
                socketio.emit("attendance_update",
                              attendance_store.query_aggregated())
            except Exception:   # pragma: no cover
                pass
        return jsonify(
            n_marked         = n_marked,
            n_skipped        = n_skipped,
            total_attendance = attendance_store.count(),
            session_ttl_s    = attendance_store._ttl,
        )

    @app.get("/api/attendance")
    def attendance_query():
        """
        Return aggregated attendance statistics.

        Query parameters (all optional)
        --------------------------------
        from     : str  – ISO-8601 lower bound (inclusive)
        to       : str  – ISO-8601 upper bound (inclusive)
        group_id : str  – filter to one group

        Response JSON
        -------------
        {
          "total_marks":   int,
          "matched_marks": int,
          "session_ttl_s": int,
          "filters":       {from, to, group_id},
          "per_subject":   [{subject_key, total, matched, last_seen}, ...],
          "per_camera":    [{camera_id, total, matched, unique_subjects}, ...],
        }
        """
        from_ts  = request.args.get("from")     or None
        to_ts    = request.args.get("to")       or None
        group_id = request.args.get("group_id") or None

        result = attendance_store.query_aggregated(
            from_ts=from_ts, to_ts=to_ts, group_id=group_id
        )
        return jsonify(result)

    # ── SocketIO events ──────────────────────────────────────────────────
    #
    # Per-client session state:
    #   sid → {mode: "enroll"|"verify"|"verify_frame"|None,  subject_id: str|None}
    # Allows a single "frame" event to behave differently depending on whether
    # the client called "start_enroll" or is doing a live verify loop.
    _sessions: dict[str, dict] = {}

    # Armed cameras for single-screen enrollment:
    #   camera_id → sid that armed it
    # When a frame arrives for that camera_id via "enroll_frame", the server
    # runs face detection, batch-enrolls, and emits "enroll_result" back.
    _armed_cameras: dict[str, str] = {}

    # Live camera status for the campus map.
    #   camera_id → {status, last_event, last_seen}
    # "status" is one of: "idle", "activity", "alert"
    _camera_status: dict[str, dict] = {}
    # Latest JPEG bytes per camera (keyed by camera_id); updated by MQTT frame subscriber
    _camera_frames: dict[str, bytes] = {}
    # Lightweight smoke/fire detector state per camera (used by frame subscriber)
    _smoke_state: dict[str, dict] = {}
    # Per-camera safety alert cooldown to avoid spamming
    _last_safety_alert_s: dict[str, float] = {}

    # ── Left-object detector ──────────────────────────────────────────────────
    from src.privacy.left_object_store    import LeftObjectStore
    from src.privacy.left_object_detector import LeftObjectDetector
    _lo_store    = LeftObjectStore(ATTENDANCE_PATH)
    _lo_detector = LeftObjectDetector(
        dwell_s       = float(os.environ.get("PCS_LEFT_OBJ_DWELL_S",   "30")),
        fg_thresh     = float(os.environ.get("PCS_LEFT_OBJ_FG_THRESH",  "22")),
        min_blob_frac = float(os.environ.get("PCS_LEFT_OBJ_MIN_BLOB",   "0.008")),
        stale_s       = float(os.environ.get("PCS_LEFT_OBJ_STALE_S",    "8")),
        cooldown_s    = float(os.environ.get("PCS_LEFT_OBJ_COOLDOWN_S", "60")),
    )

    # ── Loitering detector ────────────────────────────────────────────────────
    from src.privacy.loitering_store    import LoiteringStore
    from src.privacy.loitering_detector import LoiteringDetector
    _loiter_store    = LoiteringStore(ATTENDANCE_PATH)
    _loiter_detector = LoiteringDetector(
        dwell_s    = float(os.environ.get("PCS_LOITER_DWELL_S",    "60")),
        stale_s    = float(os.environ.get("PCS_LOITER_STALE_S",    "10")),
        cooldown_s = float(os.environ.get("PCS_LOITER_COOLDOWN_S", "120")),
    )
    # Throttle: run face recognition on MQTT frames at most once per N seconds
    _LOITER_FRAME_INTERVAL_S = float(os.environ.get("PCS_LOITER_FRAME_INTERVAL_S", "2.0"))
    _last_loiter_frame_s: dict[str, float] = {}   # camera_id → last run time

    _SAFETY_ALERT_COOLDOWN_S = float(os.environ.get("PCS_SAFETY_ALERT_COOLDOWN_S", "20"))
    _SMOKE_BLOB_MIN = float(os.environ.get("PCS_SMOKE_BLOB_MIN", "0.015"))
    _SMOKE_MOTION_MIN = float(os.environ.get("PCS_SMOKE_MOTION_MIN", "0.45"))
    _SMOKE_PERSIST_MIN = float(os.environ.get("PCS_SMOKE_PERSIST_MIN", "0.010"))
    _SMOKE_HITS_TO_TRIGGER = int(os.environ.get("PCS_SMOKE_HITS_TO_TRIGGER", "3"))

    # ── Access control engine + on_alert broadcast ─────────────────────────────

    def _on_alert(alert: dict) -> None:
        """
        Callback invoked by AccessControlEngine whenever a new alert is created.
        Broadcasts an ``alert`` Socket.IO event to all connected admin clients
        and updates the camera-status map so the campus-map icon turns red.
        """
        import datetime as _dt
        _now = _dt.datetime.utcnow().isoformat() + "Z"
        cid = alert.get("camera_id")
        if cid:
            _prev = _camera_status.get(cid, {})
            # Append alert to the per-camera event trail, preserving existing entries
            ev_q = list(_prev.get("recent_events", []))
            ev_q.append({
                "ts":         _now,
                "type":       "alert",
                "alert_type": alert.get("alert_type", "alert"),
            })
            _camera_status[cid] = {
                "status":          "alert",
                "last_event":      alert.get("alert_type", "alert"),
                "last_seen":       _now,
                "last_similarity": _prev.get("last_similarity"),
                "last_matched":    _prev.get("last_matched"),
                "recent_events":   ev_q[-3:],
            }
            socketio.emit("camera_update", {
                "camera_id":  cid,
                "status":     "alert",
                "last_event": alert.get("alert_type", "alert"),
                "last_seen":  _now,
            })
        # Route through the notification dispatcher:
        # WebSocketDriver emits "alert" synchronously; SMTP/SMS run async.
        _dispatcher.dispatch(alert)
        data_provider.record_event(
            alert.get("alert_type", "alert"),
            meta={"door_id": alert.get("door_id"), "camera_id": cid,
                  "subject_key": alert.get("subject_key")},
        )
        ALERTS_TOTAL.labels(alert_type=alert.get("alert_type", "unknown")).inc()
        _slog.warning(
            "access_control_alert",
            event_type  = "alert",
            alert_type  = alert.get("alert_type"),
            door_id     = alert.get("door_id"),
            camera_id   = cid,
            subject_key = alert.get("subject_key"),
            alert_id    = alert.get("id"),
        )

    _ace = AccessControlEngine(alert_store=_alert_store, on_alert=_on_alert)

    def _door_for_camera(camera_id: str | None) -> str | None:
        if not camera_id or not _cam_registry:
            return None
        for c in _cam_registry.list_all():
            if c.camera_id == camera_id:
                return c.door_id
        return None

    def _emit_safety_alert(
        camera_id: str,
        *,
        confidence: float,
        detector: str,
        meta: dict | None = None,
    ) -> dict | None:
        now_s = time.monotonic()
        last = _last_safety_alert_s.get(camera_id, 0.0)
        if now_s - last < _SAFETY_ALERT_COOLDOWN_S:
            return None

        alert_meta = {
            "severity": "high",
            "detector": detector,
        }
        if meta:
            alert_meta.update(meta)

        alert = _alert_store.add(
            "safety_alert",
            camera_id=camera_id,
            door_id=_door_for_camera(camera_id),
            confidence=max(0.0, min(1.0, float(confidence))),
            meta=alert_meta,
        )
        _last_safety_alert_s[camera_id] = now_s
        _on_alert(alert)
        return alert

    def _process_smoke_fire_frame(camera_id: str, frame_bytes: bytes) -> None:
        """
        Lightweight smoke/fire heuristic:
        - reddish/orange blob ratio in HSV space,
        - dense optical flow motion (Farnebäck),
        - temporal persistence heatmap (EMA).
        """
        try:
            import cv2  # type: ignore[import]
        except Exception:
            return

        try:
            arr = _np.frombuffer(frame_bytes, dtype=_np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return

            frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            lower_orange = _np.array([8, 90, 90], dtype=_np.uint8)
            upper_orange = _np.array([32, 255, 255], dtype=_np.uint8)
            lower_red1 = _np.array([0, 100, 90], dtype=_np.uint8)
            upper_red1 = _np.array([8, 255, 255], dtype=_np.uint8)
            lower_red2 = _np.array([165, 100, 90], dtype=_np.uint8)
            upper_red2 = _np.array([180, 255, 255], dtype=_np.uint8)

            orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
            red1_mask = cv2.inRange(hsv, lower_red1, upper_red1)
            red2_mask = cv2.inRange(hsv, lower_red2, upper_red2)
            color_mask = cv2.bitwise_or(orange_mask, cv2.bitwise_or(red1_mask, red2_mask))
            color_mask = cv2.GaussianBlur(color_mask, (5, 5), 0)
            color_bin = (color_mask > 0).astype(_np.float32)
            blob_ratio = float(color_bin.mean())

            state = _smoke_state.get(camera_id)
            if state is None:
                _smoke_state[camera_id] = {
                    "prev_gray": gray,
                    "persist": color_bin,
                    "hits": 0,
                }
                return

            prev_gray = state["prev_gray"]
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 2, 13, 2, 5, 1.1, 0,
            )
            mag = _np.sqrt(flow[..., 0] * flow[..., 0] + flow[..., 1] * flow[..., 1])
            motion_score = float(mag.mean())

            persist = (0.86 * state["persist"]) + (0.14 * color_bin)
            persistence_ratio = float((persist > 0.5).mean())

            state["prev_gray"] = gray
            state["persist"] = persist

            is_candidate = (
                blob_ratio >= _SMOKE_BLOB_MIN
                and motion_score >= _SMOKE_MOTION_MIN
                and persistence_ratio >= _SMOKE_PERSIST_MIN
            )

            hits = int(state.get("hits", 0))
            hits = hits + 1 if is_candidate else max(0, hits - 1)
            state["hits"] = hits

            if hits >= _SMOKE_HITS_TO_TRIGGER:
                norm_blob = min(1.0, blob_ratio / max(_SMOKE_BLOB_MIN * 2.5, 1e-6))
                norm_motion = min(1.0, motion_score / max(_SMOKE_MOTION_MIN * 2.5, 1e-6))
                norm_persist = min(1.0, persistence_ratio / max(_SMOKE_PERSIST_MIN * 2.5, 1e-6))
                score = (0.45 * norm_blob) + (0.30 * norm_motion) + (0.25 * norm_persist)
                confidence = max(0.70, min(0.99, score))
                _emit_safety_alert(
                    camera_id,
                    confidence=confidence,
                    detector="image_heuristic",
                    meta={
                        "blob_ratio": round(blob_ratio, 4),
                        "motion_score": round(motion_score, 4),
                        "persistence_ratio": round(persistence_ratio, 4),
                    },
                )
                state["hits"] = 0
        except Exception as exc:
            log.debug("[smoke-heuristic] camera=%s failed: %s", camera_id, exc)

    def _process_left_object_frame(camera_id: str, frame_bytes: bytes) -> None:
        """
        Run the left-object background-subtraction detector on one frame.
        On detection: persist to LeftObjectStore, emit Socket.IO ``left_object`` event,
        and update camera status to "alert".
        """
        import datetime as _dt
        detections = _lo_detector.process_frame(camera_id, frame_bytes)
        for det in detections:
            ev = _lo_store.add(
                camera_id,
                bbox           = det.get("bbox"),
                first_seen_utc = det.get("first_seen_utc"),
                dwell_s        = det.get("dwell_s", 0.0),
                snapshot_b64   = det.get("snapshot_b64") or None,
            )
            _now = _dt.datetime.utcnow().isoformat() + "Z"
            _camera_status[camera_id] = {
                "status":     "alert",
                "last_event": "left_object",
                "last_seen":  _now,
            }
            socketio.emit("camera_update", {
                "camera_id":  camera_id,
                "status":     "alert",
                "last_event": "left_object",
                "last_seen":  _now,
            })
            # Route through dispatcher: WebSocket emit + optional SMTP/SMS
            _dispatcher.dispatch({**ev, "alert_type": "left_object"})
            data_provider.record_event(
                "left_object",
                meta={"camera_id": camera_id, "bbox": det.get("bbox"),
                      "dwell_s": det.get("dwell_s")},
            )
            log.warning(
                "[LOD] left_object emitted camera=%s bbox=%s dwell=%.1fs id=%s",
                camera_id, det.get("bbox"), det.get("dwell_s"), ev.get("id"),
            )

    def _process_loitering_frame(camera_id: str, frame_bytes: bytes) -> None:
        """
        Run face recognition on an MQTT frame and feed identified subjects into
        the loitering tracker.  Heavy (PyTorch inference), so throttled to at
        most once per ``PCS_LOITER_FRAME_INTERVAL_S`` seconds per camera.
        """
        import datetime as _dt

        now_s = time.monotonic()
        last_s = _last_loiter_frame_s.get(camera_id, 0.0)
        if now_s - last_s < _LOITER_FRAME_INTERVAL_S:
            return
        _last_loiter_frame_s[camera_id] = now_s

        try:
            from src.web.face_utils import generate_embedding_from_image
            pil_img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            results = generate_embedding_from_image(pil_img, model=model, device="cpu")
            if not results:
                # No faces detected — all subjects in this camera tick are absent
                _loiter_detector.purge_stale(camera_id, active_keys=set(), now_s=now_s)
                return

            import numpy as _np_l
            emb_matrix = _np_l.stack([r["embedding"] for r in results])
            batch_matches = store.query_topk_batch(emb_matrix, k=1, threshold=0.35)

            active_keys: set[str] = set()
            for match_list in batch_matches:
                if match_list:
                    subject_key, similarity = match_list[0]
                    active_keys.add(subject_key)
                    detections = _loiter_detector.observe(camera_id, subject_key, now_s=now_s)
                    for det in detections:
                        ev = _loiter_store.add(
                            camera_id,
                            subject_key,
                            first_seen_utc=det.get("first_seen_utc"),
                            dwell_s=det.get("dwell_s", 0.0),
                        )
                        _now_str = _dt.datetime.utcnow().isoformat() + "Z"
                        _camera_status[camera_id] = {
                            "status":     "alert",
                            "last_event": "loitering",
                            "last_seen":  _now_str,
                        }
                        socketio.emit("camera_update", {
                            "camera_id":  camera_id,
                            "status":     "alert",
                            "last_event": "loitering",
                            "last_seen":  _now_str,
                        })
                        # Route through dispatcher: WebSocket emit + optional SMTP/SMS
                        _dispatcher.dispatch({**ev, "alert_type": "loitering"})
                        data_provider.record_event(
                            "loitering",
                            meta={"camera_id": camera_id,
                                  "subject_key": subject_key,
                                  "dwell_s": det.get("dwell_s")},
                        )
                        log.warning(
                            "[LOITER] loitering emitted camera=%s subject=%s dwell=%.1fs id=%s",
                            camera_id, subject_key, det.get("dwell_s", 0), ev.get("id"),
                        )

            # Purge subjects absent from this frame for longer than stale_s
            _loiter_detector.purge_stale(camera_id, active_keys=active_keys, now_s=now_s)

        except Exception as exc:
            log.debug("[loitering] camera=%s frame processing failed: %s", camera_id, exc)

    # ── MQTT door-event subscriber (daemon thread) ─────────────────────────────
    # Subscribes to campus/+/event; feeds door-open/close events to the engine.
    # Only started when paho-mqtt is installed AND a broker is configured.

    def _start_door_subscriber() -> None:
        broker_host = "localhost"
        broker_port = 1883
        topic_prefix = "campus"
        if _cam_registry:
            broker_host  = _cam_registry.broker_host
            broker_port  = _cam_registry.broker_port
            topic_prefix = _cam_registry.topic_prefix

        import json as _json

        try:
            import paho.mqtt.client as _mqtt  # type: ignore[import]
        except ImportError:
            log.info("[door-mqtt] paho-mqtt not installed; door-event subscriber not started.")
            return

        door_topic = f"{topic_prefix}/+/event"

        def _on_message(client, userdata, msg):  # type: ignore[no-untyped-def]
            try:
                payload = _json.loads(msg.payload.decode())
            except Exception:
                return
            # Extract door_id from topic: campus/<door_id>/event
            parts = msg.topic.split("/")
            door_id = parts[1] if len(parts) >= 3 else payload.get("door_id", "")
            event_type   = payload.get("type", "").lower()
            timestamp_utc = payload.get("timestamp", "")
            if not door_id or event_type not in ("open", "close"):
                return
            # Resolve camera_id for this door from registry
            camera_id: str | None = None
            if _cam_registry:
                for c in _cam_registry.list_all():
                    if c.door_id == door_id:
                        camera_id = c.camera_id
                        break
            log.info("[door-mqtt] door=%s type=%s camera=%s", door_id, event_type, camera_id)
            _ace.on_door_event(door_id, event_type, timestamp_utc, camera_id=camera_id)

        try:
            client = _mqtt.Client(
                _mqtt.CallbackAPIVersion.VERSION2,
                client_id="pcs-door-subscriber",
                clean_session=True,
            )
        except AttributeError:
            client = _mqtt.Client(client_id="pcs-door-subscriber", clean_session=True)  # type: ignore[call-arg]

        def _on_connect(client, userdata, flags, reason_code, properties=None):  # type: ignore[no-untyped-def]
            log.info("[door-mqtt] connected to %s:%d", broker_host, broker_port)
            client.subscribe(door_topic)

        client.on_connect = _on_connect  # type: ignore[method-assign]
        client.on_message = _on_message  # type: ignore[method-assign]

        try:
            client.connect(broker_host, broker_port, keepalive=60)
            client.loop_forever()
        except Exception as exc:
            log.info("[door-mqtt] subscriber exited: %s", exc)

    import threading as _threading
    _door_thread = _threading.Thread(
        target=_start_door_subscriber, daemon=True, name="pcs-door-mqtt"
    )
    _door_thread.start()

    # ── MQTT camera-frame subscriber (daemon thread) ───────────────────────────
    # Subscribes to campus/+/frame; keeps the latest JPEG per camera in memory
    # so /api/camera/<id>/snapshot can serve it without hitting disk.

    def _start_frame_subscriber() -> None:
        broker_host  = "localhost"
        broker_port  = 1883
        topic_prefix = "campus"
        if _cam_registry:
            broker_host  = _cam_registry.broker_host
            broker_port  = _cam_registry.broker_port
            topic_prefix = _cam_registry.topic_prefix

        try:
            import paho.mqtt.client as _mqttf  # type: ignore[import]
        except ImportError:
            return

        frame_topic = f"{topic_prefix}/+/frame"

        def _on_frame_msg(client, userdata, msg):  # type: ignore[no-untyped-def]
            # topic: campus/<camera_id>/frame
            parts = msg.topic.split("/")
            if len(parts) < 3:
                return
            cam_id = parts[1]
            frame_bytes = bytes(msg.payload)
            _camera_frames[cam_id] = frame_bytes
            _process_smoke_fire_frame(cam_id, frame_bytes)
            _process_left_object_frame(cam_id, frame_bytes)
            _process_loitering_frame(cam_id, frame_bytes)

        try:
            fc = _mqttf.Client(
                _mqttf.CallbackAPIVersion.VERSION2,
                client_id="pcs-frame-subscriber",
                clean_session=True,
            )
        except AttributeError:
            fc = _mqttf.Client(client_id="pcs-frame-subscriber", clean_session=True)  # type: ignore[call-arg]

        def _on_frame_connect(client, userdata, flags, reason_code, properties=None):  # type: ignore[no-untyped-def]
            client.subscribe(frame_topic)

        fc.on_connect = _on_frame_connect  # type: ignore[method-assign]
        fc.on_message = _on_frame_msg      # type: ignore[method-assign]
        try:
            fc.connect(broker_host, broker_port, keepalive=60)
            fc.loop_forever()
        except Exception as exc:
            log.debug("[frame-mqtt] subscriber exited: %s", exc)

    _frame_thread = _threading.Thread(
        target=_start_frame_subscriber, daemon=True, name="pcs-frame-mqtt"
    )
    _frame_thread.start()

    # ── MQTT camera-sensor subscriber (daemon thread) ─────────────────────────
    # Subscribes to campus/+/sensor; if payload contains {smoke:true} then
    # emits a high-severity safety_alert for that camera.

    def _start_sensor_subscriber() -> None:
        broker_host  = "localhost"
        broker_port  = 1883
        topic_prefix = "campus"
        if _cam_registry:
            broker_host  = _cam_registry.broker_host
            broker_port  = _cam_registry.broker_port
            topic_prefix = _cam_registry.topic_prefix

        import json as _json

        try:
            import paho.mqtt.client as _mqtts  # type: ignore[import]
        except ImportError:
            return

        sensor_topic = f"{topic_prefix}/+/sensor"

        def _on_sensor_msg(client, userdata, msg):  # type: ignore[no-untyped-def]
            parts = msg.topic.split("/")
            if len(parts) < 3:
                return
            cam_id = parts[1]
            try:
                payload = _json.loads(msg.payload.decode("utf-8"))
            except Exception:
                return

            if payload.get("smoke") is True:
                _emit_safety_alert(
                    cam_id,
                    confidence=0.99,
                    detector="mqtt_sensor",
                    meta={
                        "sensor": "smoke",
                        "smoke": True,
                    },
                )

        try:
            sc = _mqtts.Client(
                _mqtts.CallbackAPIVersion.VERSION2,
                client_id="pcs-sensor-subscriber",
                clean_session=True,
            )
        except AttributeError:
            sc = _mqtts.Client(client_id="pcs-sensor-subscriber", clean_session=True)  # type: ignore[call-arg]

        def _on_sensor_connect(client, userdata, flags, reason_code, properties=None):  # type: ignore[no-untyped-def]
            client.subscribe(sensor_topic)

        sc.on_connect = _on_sensor_connect  # type: ignore[method-assign]
        sc.on_message = _on_sensor_msg      # type: ignore[method-assign]

        try:
            sc.connect(broker_host, broker_port, keepalive=60)
            sc.loop_forever()
        except Exception as exc:
            log.debug("[sensor-mqtt] subscriber exited: %s", exc)

    _sensor_thread = _threading.Thread(
        target=_start_sensor_subscriber, daemon=True, name="pcs-sensor-mqtt"
    )
    _sensor_thread.start()

    @socketio.on("connect")
    def on_connect():
        """
        Emitted automatically when a client opens a WebSocket connection.

        Emits back
        ----------
        connected : {sid, enrolled}  – server session-ID and current store size.
        """
        _sessions[request.sid] = {"mode": None, "subject_id": None}
        emit("connected", {"sid": request.sid, "enrolled": len(store)})
        log.info("[ws] connect  sid=%s", request.sid)

    @socketio.on("disconnect")
    def on_disconnect():
        """Clean up per-client state on disconnect."""
        _sessions.pop(request.sid, None)
        log.info("[ws] disconnect  sid=%s", request.sid)

    @socketio.on("get_attendance")
    def on_get_attendance(data):
        """
        Client requests the current attendance summary.

        Client sends (all optional)
        ---------------------------
        {from_ts: str, to_ts: str, group_id: str}

        Emits back to caller
        --------------------
        attendance_update: full :meth:`AttendanceStore.query_aggregated` dict.
        """
        payload  = data or {}
        from_ts  = payload.get("from_ts")  or None
        to_ts    = payload.get("to_ts")    or None
        group_id = payload.get("group_id") or None
        result   = attendance_store.query_aggregated(
            from_ts=from_ts, to_ts=to_ts, group_id=group_id
        )
        emit("attendance_update", result)
        log.debug("[ws] attendance_update -> sid=%s", request.sid)

    @socketio.on("start_enroll")
    def on_start_enroll(data):
        """
        Signal that the next "frame" event(s) should be treated as enrollment
        for the given subject.

        Client sends
        ------------
        {subject_id: str}

        Emits back
        ----------
        enroll_ready : {subject_id}  – confirmation that session is armed.
        error        : {message}     – if subject_id is missing.
        """
        subject_id = (data.get("subject_id") or "").strip()
        if not subject_id:
            emit("error", {"message": "subject_id is required"})
            return
        _sessions[request.sid] = {"mode": "enroll", "subject_id": subject_id}
        emit("enroll_ready", {"subject_id": subject_id})
        log.info("[ws] start_enroll  subject=%s  sid=%s", subject_id, request.sid)

    @socketio.on("frame")
    def on_frame(data):
        """
        Process a single camera frame.  Behaviour depends on the current
        session mode (set by "start_enroll").

        Client sends
        ------------
        {
          image_b64  : str            base64 JPEG/PNG (data-URI prefix optional)
          subject_id : str (optional) overrides session subject_id
          mode       : str (optional) "enroll" | "verify" – overrides session mode
        }

        Emits back
        ----------
        enroll_result : {status, subject_id, total_enrolled}  – mode=="enroll"
        verify_result : {subject_id, similarity, match, latency_ms}  – mode=="verify"
        error         : {message}  – on failure
        """
        import base64
        import time

        import numpy as _np
        import numpy as np

        session    = _sessions.get(request.sid, {})
        mode       = data.get("mode")       or session.get("mode")
        subject_id = (data.get("subject_id") or session.get("subject_id") or "").strip()
        image_b64  = data.get("image_b64", "")

        if not image_b64:
            emit("error", {"message": "image_b64 is required"})
            return

        raw = base64.b64decode(image_b64.split(",")[-1])

        class _Buf:
            def read(self_): return raw

        t0 = time.monotonic()

        if mode == "enroll":
            embedding = _embed(_Buf())
            if not subject_id:
                emit("error", {"message": "subject_id required for enroll"})
                return
            store.add(subject_id, embedding)
            store.save()
            data_provider.record_event("enroll", subject_id=subject_id,
                                       meta={"source": "websocket"})
            _sessions[request.sid] = {"mode": None, "subject_id": None}
            emit("enroll_result", {"status": "enrolled", "subject_id": subject_id,
                                   "total_enrolled": len(store)})
            log.info("[ws] enrolled '%s'  sid=%s", subject_id, request.sid)

        elif mode == "verify":
            embedding = _embed(_Buf())
            if not subject_id:
                emit("error", {"message": "subject_id required for verify"})
                return
            if subject_id not in store:
                emit("error", {"message": f"Subject '{subject_id}' is not enrolled"})
                return
            stored_emb = store.get(subject_id)
            similarity = float(np.dot(embedding, stored_emb))
            match      = similarity >= 0.0
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            emit("verify_result", {"subject_id": subject_id,
                                   "similarity":  round(similarity, 4),
                                   "match":       match,
                                   "latency_ms":  latency_ms})

        elif mode == "verify_frame":
            from src.web.face_utils import generate_embedding_from_image
            from PIL import Image as _PILImage

            k      = int(data.get("k",      3))
            thresh = float(data.get("thresh", 0.0))
            pil    = _PILImage.open(io.BytesIO(raw)).convert("RGB")
            faces  = generate_embedding_from_image(pil, model=model, device="cpu")

            if not faces:
                emit("verify_frame_result",
                     {"n_faces": 0, "faces": [],
                      "latency_ms": round((time.monotonic()-t0)*1000, 1)})
                return

            emb_matrix = _np.stack([f["embedding"] for f in faces])
            batch_hits = store.query_topk_batch(emb_matrix, k=k, threshold=thresh)

            faces_out = []
            for face, hits in zip(faces, batch_hits):
                faces_out.append({
                    "face_id":     face["face_id_local"],
                    "bbox":        list(face["bbox"]),
                    "top_matches": [
                        {"subject_key": hid,
                         "similarity":  round(sim, 4),
                         "match":       sim >= max(thresh, 0.0)}
                        for hid, sim in hits
                    ],
                })
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            emit("verify_frame_result",
                 {"n_faces": len(faces_out), "faces": faces_out,
                  "latency_ms": latency_ms})

        else:
            emit("error", {"message": (
                "Unknown mode. Call 'start_enroll' first, or pass "
                "mode='enroll'|'verify'|'verify_frame' in the frame payload."
            )})

    @socketio.on("verify")
    def on_verify(data):
        """
        One-shot explicit verify event — same as sending a frame with
        mode="verify" but self-contained; does not require a prior
        "start_enroll" call.  Logs the result to the event feed.

        Client sends
        ------------
        {subject_id: str, image_b64: str}

        Emits back
        ----------
        verify_result : {subject_id, similarity, match, latency_ms}
        error         : {message}
        """
        import base64
        import time

        import numpy as np

        subject_id = (data.get("subject_id") or "").strip()
        image_b64  = data.get("image_b64", "")

        if not subject_id or not image_b64:
            emit("error", {"message": "subject_id and image_b64 are required"})
            return
        if subject_id not in store:
            emit("error", {"message": f"Subject '{subject_id}' is not enrolled"})
            return

        raw = base64.b64decode(image_b64.split(",")[-1])

        class _Buf:
            def read(self_): return raw

        t0         = time.monotonic()
        query_emb  = _embed(_Buf())
        stored_emb = store.get(subject_id)
        similarity = float(np.dot(query_emb, stored_emb))
        match      = similarity >= 0.0
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        data_provider.record_event(
            "verify", subject_id=subject_id,
            meta={"similarity": round(similarity, 4), "match": match,
                  "source": "websocket"},
        )
        emit("verify_result", {"subject_id": subject_id,
                               "similarity":  round(similarity, 4),
                               "match":       match,
                               "latency_ms":  latency_ms})
        log.info("[ws] verify '%s'  sim=%.4f  match=%s", subject_id, similarity, match)

    # ── Single-screen enrollment Socket.IO events ──────────────────────────────

    @socketio.on("arm_enroll")
    def on_arm_enroll(data):
        """
        Admin arms a camera for single-frame enrollment capture.

        Client sends
        ------------
        {camera_id: str}

        Emits back (to caller)
        ----------------------
        arm_ack  : {camera_id, armed: true}  – camera is now armed.
        error    : {message}                 – validation failure.

        Emits (broadcast, room=camera_id)
        ----------------------------------
        camera_status : {camera_id, armed: true, armed_by: sid}
        """
        camera_id = (data.get("camera_id") or "").strip()
        if not camera_id:
            emit("error", {"message": "camera_id is required"})
            return

        # Disarm any camera this session was previously holding
        for cid, sid in list(_armed_cameras.items()):
            if sid == request.sid:
                del _armed_cameras[cid]
                socketio.emit("camera_status", {"camera_id": cid, "armed": False})

        _armed_cameras[camera_id] = request.sid

        import datetime as _dt
        _now = _dt.datetime.utcnow().isoformat() + "Z"
        _camera_status[camera_id] = {
            "status":     "activity",
            "last_event": "arm_enroll",
            "last_seen":  _now,
        }

        emit("arm_ack", {"camera_id": camera_id, "armed": True})
        socketio.emit("camera_status", {
            "camera_id": camera_id,
            "armed":     True,
            "armed_by":  request.sid,
        })
        socketio.emit("camera_update", {
            "camera_id":  camera_id,
            "status":     "activity",
            "last_event": "arm_enroll",
            "last_seen":  _now,
        })
        log.info("[ws] arm_enroll camera=%s sid=%s", camera_id, request.sid)

    @socketio.on("disarm_enroll")
    def on_disarm_enroll(data):
        """
        Admin disarms a camera (cancel without capturing).

        Client sends
        ------------
        {camera_id: str}  or {}  (disarms whichever camera this session holds)

        Emits back (to caller)
        ----------------------
        arm_ack  : {camera_id, armed: false}
        """
        camera_id = (data.get("camera_id") or "").strip()

        disarmed: list[str] = []
        if camera_id:
            if _armed_cameras.get(camera_id) == request.sid:
                del _armed_cameras[camera_id]
                disarmed.append(camera_id)
        else:
            for cid, sid in list(_armed_cameras.items()):
                if sid == request.sid:
                    del _armed_cameras[cid]
                    disarmed.append(cid)

        for cid in disarmed:
            import datetime as _dt
            _now = _dt.datetime.utcnow().isoformat() + "Z"
            _camera_status[cid] = {
                "status":     "idle",
                "last_event": "disarm_enroll",
                "last_seen":  _now,
            }
            socketio.emit("camera_status", {"camera_id": cid, "armed": False})
            socketio.emit("camera_update", {
                "camera_id":  cid,
                "status":     "idle",
                "last_event": "disarm_enroll",
                "last_seen":  _now,
            })
            emit("arm_ack", {"camera_id": cid, "armed": False})
        log.info("[ws] disarm_enroll camera=%s sid=%s", disarmed, request.sid)

    @socketio.on("enroll_frame")
    def on_enroll_frame(data):
        """
        Process a single frame from an armed camera for enrollment.

        The admin's browser sends webcam frames for the currently armed camera.
        On first frame with at least one detected face the server:
          1. Runs multi-face detection + embedding (``generate_embedding_from_image``).
          2. Batch-enrolls every detected face with a privacy-preserving opaque key.
          3. Immediately disarms the camera so further frames are not processed.
          4. Emits ``enroll_result`` to the caller with face crops (base64 JPEG),
             generated keys, and metadata for the modal to display.

        Client sends
        ------------
        {
          camera_id : str
          image_b64 : str   base64 JPEG/PNG with or without data-URI prefix
          group_id  : str   (optional, default "default")
        }

        Emits back (to caller)
        ----------------------
        enroll_result : {
          status        : "enrolled" | "no_face" | "not_armed" | "error"
          camera_id     : str
          group_id      : str
          n_enrolled    : int
          total_enrolled: int
          faces         : [
            {
              face_id  : int
              key      : str        opaque subject key (caller must retain)
              crop_b64 : str        base64 JPEG of the face crop (for modal preview)
              bbox     : [x,y,w,h]
            }, ...
          ]
          timestamp     : str   ISO-8601 UTC
        }
        error : {message}  on hard failure
        """
        import base64
        import secrets
        import io as _io
        from datetime import datetime as _dt, timezone as _tz

        from src.web.face_utils import generate_embedding_from_image
        from PIL import Image as _PILImage

        camera_id = (data.get("camera_id") or "").strip()
        image_b64 = data.get("image_b64", "")
        group_id  = (data.get("group_id")  or "default").strip()

        if not camera_id or not image_b64:
            emit("error", {"message": "camera_id and image_b64 are required"})
            return

        # ── Check armed ───────────────────────────────────────────────────────
        armed_sid = _armed_cameras.get(camera_id)
        if armed_sid != request.sid:
            emit("enroll_result", {
                "status":    "not_armed",
                "camera_id": camera_id,
                "message":   "Camera is not armed for this session",
            })
            return

        # ── Decode image ──────────────────────────────────────────────────────
        try:
            raw = base64.b64decode(image_b64.split(",")[-1])
            pil = _PILImage.open(_io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            emit("error", {"message": f"Image decode failed: {exc}"})
            return

        # ── Face detection + embedding ────────────────────────────────────────
        try:
            faces = generate_embedding_from_image(pil, model=model, device="cpu")
        except Exception as exc:
            emit("error", {"message": f"Face detection failed: {exc}"})
            return

        if not faces:
            # No face — keep camera armed so admin can retry
            emit("enroll_result", {
                "status":    "no_face",
                "camera_id": camera_id,
                "n_enrolled": 0,
                "faces":     [],
                "timestamp": _dt.now(_tz.utc).isoformat(),
            })
            return

        # ── Batch enroll each face ────────────────────────────────────────────
        # Disarm immediately so only the first face-containing frame enrolls
        del _armed_cameras[camera_id]
        socketio.emit("camera_status", {"camera_id": camera_id, "armed": False})

        import datetime as _dt_m
        _now_m = _dt_m.datetime.utcnow().isoformat() + "Z"
        _camera_status[camera_id] = {
            "status":     "activity",
            "last_event": "enroll_batch",
            "last_seen":  _now_m,
        }
        socketio.emit("camera_update", {
            "camera_id":  camera_id,
            "status":     "activity",
            "last_event": "enroll_batch",
            "last_seen":  _now_m,
        })

        enrolled_faces: list[dict] = []
        for face in faces:
            token       = secrets.token_hex(16)
            subject_key = f"grp:{group_id}:f{face['face_id_local']}:{token}"
            store.add(subject_key, face["embedding"])

            # ── Crop face from original image for modal preview ───────────────
            crop_b64 = ""
            try:
                x, y, w, h = (int(v) for v in face["bbox"])
                pad  = max(10, int(min(w, h) * 0.15))
                x1   = max(0, x - pad);  y1 = max(0, y - pad)
                x2   = min(pil.width,  x + w + pad)
                y2   = min(pil.height, y + h + pad)
                crop = pil.crop((x1, y1, x2, y2)).resize((160, 160), _PILImage.BILINEAR)
                buf  = _io.BytesIO()
                crop.save(buf, format="JPEG", quality=80)
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                pass  # non-fatal; modal preview just won't show

            enrolled_faces.append({
                "face_id": face["face_id_local"],
                "key":     subject_key,
                "crop_b64": crop_b64,
                "bbox":    list(face["bbox"]),
            })

        store.save()
        data_provider.record_event(
            "enroll_batch",
            meta={"n_enrolled": len(enrolled_faces), "group_id": group_id, "camera_id": camera_id},
        )
        log.info(
            "[ws] enroll_frame camera=%s group=%s faces=%d  sid=%s",
            camera_id, group_id, len(enrolled_faces), request.sid,
        )

        emit("enroll_result", {
            "status":         "enrolled",
            "camera_id":      camera_id,
            "group_id":       group_id,
            "n_enrolled":     len(enrolled_faces),
            "total_enrolled": len(store),
            "faces":          enrolled_faces,
            "timestamp":      _dt.now(_tz.utc).isoformat(),
        })

    # ── Alerts REST endpoints ─────────────────────────────────────────────────

    @app.get("/api/alerts")
    def api_alerts():
        """
        Return recent security alerts.

        Query parameters (all optional)
        --------------------------------
        n           : int   Max alerts to return (default 50).
        alert_type  : str   Filter to ``forced_entry``, ``tailgate``, or ``safety_alert``.
        door_id     : str   Filter to a specific door.
        unacked_only: 1     Only return unacknowledged alerts.

        Response JSON
        -------------
        {
          "alerts":        [{id, alert_type, camera_id, door_id, subject_key,
                             confidence, timestamp_utc, meta, acknowledged}, ...],
          "total_unacked": int
        }
        """
        n           = min(int(request.args.get("n", 50)), 500)
        alert_type  = request.args.get("alert_type") or None
        door_id     = request.args.get("door_id")    or None
        unacked_only = request.args.get("unacked_only") in ("1", "true", "yes")
        alerts = _alert_store.list_recent(
            n=n,
            alert_type=alert_type,
            door_id=door_id,
            unacked_only=unacked_only,
        )
        return jsonify(
            alerts        = alerts,
            total_unacked = _alert_store.count_unacked(),
        )

    @app.post("/api/alerts/<int:alert_id>/ack")
    @require_role("operator", "admin")
    def api_alert_ack(alert_id: int):
        """
        Acknowledge an alert (operator+ for low/medium; admin-only for high severity).

        Request JSON (optional)
        -----------------------
        {"acked_by": "operator-name-or-id"}

        Response JSON
        -------------
        {acknowledged: true, id: int, acked_by: str|null, acked_at: str}  or 404.
        """
        caller = get_current_user()
        alert  = _alert_store.get(alert_id)
        if alert is None:
            return jsonify(error=f"Alert {alert_id} not found"), 404

        # Determine severity from alert type.
        # High-severity types require admin role.
        _HIGH_SEVERITY_TYPES = {"safety_alert", "forced_entry", "tailgate"}
        alert_type = alert.get("alert_type", "")
        if alert_type in _HIGH_SEVERITY_TYPES and (caller is None or caller.get("role") != "admin"):
            return jsonify(
                error="Acknowledging high-severity alerts requires the admin role.",
                alert_type=alert_type,
                required_role="admin",
                code="forbidden",
            ), 403

        body     = request.get_json(silent=True) or {}
        acked_by = body.get("acked_by") or (caller["sub"] if caller else None)
        ok = _alert_store.acknowledge(alert_id, acked_by=acked_by)
        if not ok:
            return jsonify(error=f"Alert {alert_id} ack failed"), 500
        alert = _alert_store.get(alert_id)  # includes acked_by + acked_at
        _audit_store.append(
            "ack",
            actor     = caller["sub"] if caller else acked_by,
            camera_id = alert.get("camera_id"),
            alert_id  = alert_id,
            detail    = {
                "alert_type": alert.get("alert_type"),
                "acked_by":   alert.get("acked_by"),
                "acked_at":   alert.get("acked_at"),
            },
        )
        socketio.emit("alert_ack", alert)
        return jsonify(acknowledged=True, id=alert_id,
                       acked_by=alert.get("acked_by"),
                       acked_at=alert.get("acked_at"))

    @app.post("/api/alerts/panic")
    @require_role("admin")
    def api_panic():
        """
        Trigger an immediate out-of-band PANIC escalation.

        Sends an urgent SMS (bypassing the normal throttle) to every number
        listed in ``PCS_PANIC_SMS_TO`` (comma-separated) or falls back to
        ``PCS_SMTP_TO``.  Also broadcasts a ``panic`` Socket.IO event to all
        connected dashboard clients.

        Request JSON (optional)
        -----------------------
        {"triggered_by": "operator-name", "message": "custom note"}

        Response JSON
        -------------
        {"ok": true, "sms_count": int, "triggered_by": str, "timestamp": str}
        """
        import datetime as _dt, smtplib as _smtplib, threading as _threading

        body         = request.get_json(silent=True) or {}
        triggered_by = body.get("triggered_by") or "mobile-admin"
        message      = body.get("message")      or "PANIC — immediate escalation requested"
        now_iso      = _dt.datetime.now(_dt.timezone.utc).isoformat()
        _panic_caller = get_current_user()
        _audit_store.append(
            "panic",
            actor     = _panic_caller["sub"] if _panic_caller else triggered_by,
            camera_id = body.get("camera_id"),
            detail    = {"triggered_by": triggered_by, "message": message},
        )

        # Gather on-call numbers (PCS_PANIC_SMS_TO preferred, fallback PCS_SMTP_TO)
        raw_numbers  = (
            os.environ.get("PCS_PANIC_SMS_TO", "")
            or os.environ.get("PCS_SMTP_TO",   "")
        ).strip()
        phone_numbers = [n.strip() for n in raw_numbers.split(",") if n.strip()]

        panic_payload = {
            "event":        "panic",
            "triggered_by": triggered_by,
            "message":      message,
            "timestamp":    now_iso,
            "camera_id":    body.get("camera_id"),
        }

        # Broadcast to all dashboard clients immediately
        socketio.emit("panic", panic_payload)
        log.warning("[PANIC] triggered_by=%s message=%r", triggered_by, message)

        sms_count = 0

        # Load active SMS driver and call it directly (bypasses throttle)
        active_sms_driver = next(
            (d for d in _notify_drivers if d.name == "sms" and d.is_configured()),
            None,
        )

        if active_sms_driver:
            # Build a synthetic notification record for the driver
            panic_notification = {
                "alert_type": "panic",
                "severity":   "high",
                "channel":    "sms",
                "payload":    panic_payload,
            }
            for number in phone_numbers:
                def _send_sms(driver=active_sms_driver, notif=panic_notification, num=number):
                    try:
                        # Inject per-number target into env temporarily is not thread-safe;
                        # call driver.send() and let the driver read PCS_PANIC_SMS_TO itself.
                        driver.send(notif)
                        log.info("[PANIC] SMS dispatched to %s", num)
                    except Exception as exc:  # noqa: BLE001
                        log.error("[PANIC] SMS to %s failed: %s", num, exc)
                _threading.Thread(target=_send_sms, daemon=True).start()
                sms_count += 1
        elif phone_numbers:
            # No SMS driver configured — fall back to SMTP as emergency channel
            smtp_host = os.environ.get("PCS_SMTP_HOST", "").strip()
            if smtp_host:
                def _send_panic_email():
                    try:
                        from email.mime.text import MIMEText as _MIMEText
                        from_addr = os.environ.get("PCS_SMTP_FROM", "pcs@localhost")
                        to_list   = raw_numbers or from_addr
                        msg = _MIMEText(
                            f"PANIC ESCALATION\n\n"
                            f"Triggered by : {triggered_by}\n"
                            f"Time         : {now_iso}\n"
                            f"Message      : {message}\n",
                            "plain", "utf-8",
                        )
                        msg["Subject"] = "[PCS/PANIC] Immediate escalation — action required"
                        msg["From"]    = from_addr
                        msg["To"]      = to_list
                        port     = int(os.environ.get("PCS_SMTP_PORT", "587"))
                        use_tls  = os.environ.get("PCS_SMTP_TLS", "1") == "1"
                        user     = os.environ.get("PCS_SMTP_USER",     "").strip() or None
                        password = os.environ.get("PCS_SMTP_PASSWORD", "").strip() or None
                        with _smtplib.SMTP(smtp_host, port, timeout=10) as srv:
                            srv.ehlo()
                            if use_tls:
                                srv.starttls(); srv.ehlo()
                            if user and password:
                                srv.login(user, password)
                            srv.sendmail(from_addr, to_list.split(","), msg.as_string())
                        log.info("[PANIC] Fallback email sent to %s", to_list)
                    except Exception as exc:  # noqa: BLE001
                        log.error("[PANIC] Fallback email failed: %s", exc)
                _threading.Thread(target=_send_panic_email, daemon=True).start()
                sms_count = len(phone_numbers) or 1  # treat email recipients as count

        return jsonify(
            ok           = True,
            sms_count    = sms_count,
            triggered_by = triggered_by,
            timestamp    = now_iso,
            note         = "panic broadcast sent" if sms_count == 0 else None,
        )

    # ── Audit log endpoint ────────────────────────────────────────────────────

    @app.get("/api/audit")
    @require_role("admin")
    def api_audit():
        """
        Return audit log rows, newest-first.

        Query parameters
        ----------------
        limit      : int  (default 100, max 1000)
        event_type : str  filter by event type (enroll, verify, ack, panic, alert, …)
        actor      : str  filter by actor username
        camera_id  : str  filter by camera
        alert_id   : int  filter by alert row id
        since      : str  ISO-8601 timestamp — only rows after this time

        Response JSON
        -------------
        {
          "audit": [
            {"id", "timestamp_utc", "event_type", "actor",
             "subject_id", "camera_id", "alert_id", "detail"},
            …
          ],
          "total": int
        }
        """
        try:
            limit = int(request.args.get("limit", 100))
        except ValueError:
            return jsonify(error="limit must be an integer"), 400

        rows = _audit_store.query(
            limit      = limit,
            event_type = request.args.get("event_type") or None,
            actor      = request.args.get("actor")      or None,
            camera_id  = request.args.get("camera_id") or None,
            alert_id   = int(request.args["alert_id"]) if request.args.get("alert_id") else None,
            since      = request.args.get("since")     or None,
        )
        return jsonify(audit=rows, total=_audit_store.count())

    # ── Campus overview map ───────────────────────────────────────────────────

    @app.get("/api/camera/<camera_id>/snapshot")
    def api_camera_snapshot(camera_id: str):
        """
        Return the latest JPEG frame received via MQTT for a camera.

        Responds with 200 + image/jpeg, or 404 if no frame has been received yet.
        Used by the Alerts panel to show camera thumbnails alongside each alert.
        """
        frame = _camera_frames.get(camera_id)
        if frame is None:
            return jsonify(error="No frame available"), 404
        return app.response_class(
            response=frame,
            status=200,
            mimetype="image/jpeg",
        )

    @app.get("/admin/campus")
    @require_auth
    def admin_campus():
        """
        Render the campus overview map page.

        Passes every camera from configs/cameras.yaml to the template together
        with live-status snapshots so the Jinja layer can pre-render the
        initial icon colours without waiting for the first Socket.IO tick.
        """
        cameras: list[dict] = []
        if _cam_registry:
            for c in _cam_registry.list_all():
                snap = _camera_status.get(c.camera_id, {})
                cameras.append({
                    "camera_id":   c.camera_id,
                    "name":        c.name,
                    "door_id":     c.door_id,
                    "lat":         c.lat,
                    "lon":         c.lon,
                    "active":      c.active,
                    "status":      snap.get("status", "idle"),
                    "last_seen":   snap.get("last_seen"),
                    "frame_topic": _cam_registry.frame_topic(c.camera_id),
                    "sensor_topic": _cam_registry.sensor_topic(c.camera_id),
                })
        return render_template(
            "campus_map.html",
            cameras      = cameras,
            enrolled     = len(store),
            current_user = get_current_user(),
        )

    @app.get("/alerts/mobile")
    @require_role("operator", "admin")
    def alerts_mobile():
        """
        Render the mobile-friendly on-call admin alert view.

        A stripped-down, touch-optimised page designed for small screens that
        gives on-call security staff:

        * Live alert feed via Socket.IO (no polling)
        * One-tap Acknowledge per alert
        * Navigate-to-camera shortcut (opens campus map anchored to the camera)
        * Panic button – POST /api/alerts/panic for immediate out-of-band SMS
        """
        return render_template(
            "mobile_alerts.html",
            enrolled     = len(store),
            current_user = get_current_user(),
        )

    @app.get("/api/camera/status")
    def api_camera_status_route():
        """
        Return the last-known status for every camera as a JSON dict.

        Response body
        -------------
        {
          "<camera_id>": {
            "status":     "idle" | "activity" | "alert",
            "last_event": str,
            "last_seen":  str (ISO-8601) | null
          },
          ...
        }
        """
        return jsonify({
            cid: {
                "status":          snap.get("status", "idle"),
                "last_event":      snap.get("last_event", ""),
                "last_seen":       snap.get("last_seen"),
                "last_similarity": snap.get("last_similarity"),
                "last_matched":    snap.get("last_matched"),
                "recent_events":   snap.get("recent_events", []),
            }
            for cid, snap in _camera_status.items()
        })

    # ── Left-object REST endpoints ───────────────────────────────────────────

    @app.get("/api/left-objects")
    def api_left_objects():
        """
        Return recent left-object detection events.

        Query parameters (all optional)
        --------------------------------
        n          : int  Max events (default 50, cap 500).
        camera_id  : str  Filter to a specific camera.
        dismissed  : 0|1  Filter by dismissed state (omit → no filter).
        confirmed  : 0|1  Filter by confirmed state (omit → no filter).

        Response JSON
        -------------
        {events: [{id, camera_id, bbox, first_seen_utc, detected_utc, dwell_s,
                   snapshot_b64, dismissed, confirmed, action_by, action_at}],
         total_active: int}
        """
        n          = min(int(request.args.get("n", 50)), 500)
        camera_id  = request.args.get("camera_id") or None
        def _boolparam(key):
            v = request.args.get(key)
            return None if v is None else (v in ("1", "true", "yes"))
        dismissed  = _boolparam("dismissed")
        confirmed  = _boolparam("confirmed")
        events = _lo_store.list_recent(
            n=n, camera_id=camera_id,
            dismissed=dismissed, confirmed=confirmed,
        )
        # Strip heavy snapshot from list view (caller can fetch /api/left-objects/<id> for full)
        slim = [{k: v for k, v in ev.items() if k != "snapshot_b64"} for ev in events]
        return jsonify(events=slim, total_active=_lo_store.count_active())

    @app.get("/api/left-objects/<int:event_id>")
    def api_left_object_get(event_id: int):
        """Return a single left-object event including the snapshot_b64 field."""
        ev = _lo_store.get(event_id)
        if ev is None:
            return jsonify(error=f"Event {event_id} not found"), 404
        return jsonify(ev)

    @app.post("/api/left-objects/<int:event_id>/remove")
    @require_role("operator", "admin")
    def api_left_object_remove(event_id: int):
        """
        Mark a left-object event as dismissed (object removed from scene).

        Request JSON (optional): ``{action_by: str}``
        """
        body       = request.get_json(silent=True) or {}
        action_by  = body.get("action_by") or None
        ok = _lo_store.dismiss(event_id, action_by=action_by)
        if not ok:
            return jsonify(error=f"Event {event_id} not found"), 404
        ev = _lo_store.get(event_id)
        socketio.emit("left_object_update", ev)
        return jsonify(dismissed=True, id=event_id)

    @app.post("/api/left-objects/<int:event_id>/confirm")
    @require_role("operator", "admin")
    def api_left_object_confirm(event_id: int):
        """
        Confirm a left-object event as a security threat.

        Also creates a ``safety_alert`` in the alert store so the Alerts
        panel reflects the threat, and broadcasts ``left_object_update``.

        Request JSON (optional): ``{action_by: str}``
        """
        body      = request.get_json(silent=True) or {}
        action_by = body.get("action_by") or None
        ok = _lo_store.confirm(event_id, action_by=action_by)
        if not ok:
            return jsonify(error=f"Event {event_id} not found"), 404
        ev = _lo_store.get(event_id)
        # Escalate to a full security alert
        alert = _alert_store.add(
            "safety_alert",
            camera_id  = ev.get("camera_id"),
            door_id    = _door_for_camera(ev.get("camera_id")),
            confidence = 0.95,
            meta       = {"severity": "high", "source": "left_object",
                          "left_object_id": event_id,
                          "bbox": ev.get("bbox")},
        )
        _on_alert(alert)
        socketio.emit("left_object_update", ev)
        return jsonify(confirmed=True, id=event_id, alert_id=alert.get("id"))

    @app.post("/api/left-objects/trigger")
    def api_left_object_trigger():
        """
        Manually inject a test ``left_object`` event for a given camera.

        Request JSON: ``{camera_id: str, bbox?: [x,y,w,h], dwell_s?: float}``

        Useful for integration tests and demo runs without a live MQTT stream.
        """
        body      = request.get_json(silent=True) or {}
        camera_id = (body.get("camera_id") or "").strip()
        if not camera_id:
            return jsonify(error="camera_id is required"), 400
        bbox    = body.get("bbox") or [0, 0, 80, 80]
        dwell_s = float(body.get("dwell_s") or 30.0)
        ev = _lo_store.add(
            camera_id,
            bbox      = bbox,
            dwell_s   = dwell_s,
        )
        import datetime as _dt
        _now = _dt.datetime.utcnow().isoformat() + "Z"
        _camera_status[camera_id] = {"status": "alert", "last_event": "left_object", "last_seen": _now}
        socketio.emit("camera_update", {"camera_id": camera_id, "status": "alert",
                                         "last_event": "left_object", "last_seen": _now})
        _dispatcher.dispatch({**ev, "alert_type": "left_object"})
        data_provider.record_event("left_object", meta={"camera_id": camera_id,
                                                          "bbox": bbox, "dwell_s": dwell_s})
        return jsonify(ev)

    # ── Loitering detection endpoints ─────────────────────────────────────────

    @app.get("/api/loitering")
    def api_loitering_list():
        """
        Return a list of loitering events, newest-first.

        Query params
        ------------
        n           : int    (default 50) — max events to return
        camera_id   : str    (optional)   — restrict to one camera
        subject_key : str    (optional)   — restrict to one subject
        dismissed   : "true"|"false" (optional) — filter by dismiss status
        """
        n           = int(request.args.get("n", 50))
        camera_id   = request.args.get("camera_id")   or None
        subject_key = request.args.get("subject_key") or None
        dismissed_q = request.args.get("dismissed")
        dismissed: bool | None = None
        if dismissed_q in ("true",  "1"): dismissed = True
        if dismissed_q in ("false", "0"): dismissed = False
        events = _loiter_store.list_recent(
            n, camera_id=camera_id, subject_key=subject_key, dismissed=dismissed
        )
        return jsonify(events=events, total_active=_loiter_store.count_active())

    @app.post("/api/loitering/<int:event_id>/dismiss")
    @require_role("operator", "admin")
    def api_loitering_dismiss(event_id: int):
        """
        Dismiss a loitering event.

        Request JSON (optional): ``{action_by: str}``
        """
        body      = request.get_json(silent=True) or {}
        action_by = body.get("action_by") or None
        ok = _loiter_store.dismiss(event_id, action_by=action_by)
        if not ok:
            return jsonify(error=f"Loitering event {event_id} not found"), 404
        ev = _loiter_store.get(event_id)
        socketio.emit("loitering_update", ev)
        return jsonify(dismissed=True, id=event_id)

    @app.post("/api/loitering/trigger")
    def api_loitering_trigger():
        """
        Manually inject a test ``loitering`` event for a given camera.

        Request JSON: ``{camera_id: str, subject_key?: str, dwell_s?: float}``

        Useful for integration tests and demo runs without a live MQTT/face stream.
        """
        body        = request.get_json(silent=True) or {}
        camera_id   = (body.get("camera_id")   or "").strip()
        if not camera_id:
            return jsonify(error="camera_id is required"), 400
        subject_key = (body.get("subject_key") or "").strip() or "test-subject-001"
        dwell_s     = float(body.get("dwell_s") or 65.0)
        ev = _loiter_store.add(camera_id, subject_key, dwell_s=dwell_s)
        import datetime as _dt
        _now = _dt.datetime.utcnow().isoformat() + "Z"
        _camera_status[camera_id] = {
            "status": "alert", "last_event": "loitering", "last_seen": _now
        }
        socketio.emit("camera_update", {
            "camera_id": camera_id, "status": "alert",
            "last_event": "loitering", "last_seen": _now,
        })
        _dispatcher.dispatch({**ev, "alert_type": "loitering"})
        data_provider.record_event(
            "loitering",
            meta={"camera_id": camera_id, "subject_key": subject_key, "dwell_s": dwell_s},
        )
        return jsonify(ev)

    # ── Notifications REST endpoint ───────────────────────────────────────────

    @app.get("/api/notifications")
    def api_notifications():
        """
        Return recent notification delivery records.

        Each record corresponds to one delivery attempt on one channel for one
        alert or event.  Multiple records may exist per alert (one per channel).

        Query parameters (all optional)
        --------------------------------
        n          : int   Max records to return (default 50, cap 500).
        channel    : str   Filter to ``websocket``, ``smtp``, or ``sms``.
        status     : str   Filter to ``pending``, ``sent``, ``failed``, or ``throttled``.
        alert_id   : int   Filter to notifications for a specific alert ID.
        alert_type : str   Filter to a specific alert type.

        Response JSON
        -------------
        {
          "notifications":  [{id, alert_id, alert_type, severity, channel,
                              status, attempts, last_attempt, next_retry,
                              error_msg, created_at}, ...],
          "total_sent":     int,
          "total_failed":   int,
          "total_throttled": int,
        }
        """
        try:
            n = min(int(request.args.get("n", 50)), 500)
        except (ValueError, TypeError):
            n = 50

        channel    = request.args.get("channel")    or None
        status     = request.args.get("status")     or None
        alert_type = request.args.get("alert_type") or None
        try:
            alert_id = int(request.args.get("alert_id")) if request.args.get("alert_id") else None
        except (ValueError, TypeError):
            alert_id = None

        records = _notify_store.list_recent(
            n,
            channel    = channel,
            status     = status,
            alert_id   = alert_id,
            alert_type = alert_type,
        )
        # Strip large payload content from list view to keep response compact
        slim = [{k: v for k, v in r.items() if k != "payload"} for r in records]

        return jsonify(
            notifications   = slim,
            total_sent      = _notify_store.count_by_status("sent"),
            total_failed    = _notify_store.count_by_status("failed"),
            total_throttled = _notify_store.count_by_status("throttled"),
        )

    # ── Optional: serve React SPA build ──────────────────────────────────────
    # Uncomment and set REACT_BUILD_DIR env var pointing to your dist/ folder.
    #
    # @app.get("/<path:path>")
    # def spa(path: str):
    #     build_dir = Path(REACT_BUILD)
    #     target = build_dir / path
    #     if target.is_file():
    #         return send_from_directory(str(build_dir), path)
    #     return send_from_directory(str(build_dir), "index.html")

    return app


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
    _app = create_app()
    _sio = _app.extensions["socketio"]
    # Use socketio.run() instead of app.run() so SocketIO's transport layer
    # (long-polling fallback + WebSocket upgrade) is active.
    _sio.run(
        _app,
        host=os.environ.get("FLASK_HOST", "0.0.0.0"),
        port=int(os.environ.get("FLASK_PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1",
        use_reloader=False,   # reloader forks the process; disable with SocketIO
        allow_unsafe_werkzeug=True,
    )
