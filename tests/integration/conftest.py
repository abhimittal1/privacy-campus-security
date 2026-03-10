"""
tests/integration/conftest.py
─────────────────────────────────────────────────────────────────────────────
Shared pytest fixtures for the integration test suite.

Fixture graph
─────────────────────────────────────────────────────────────────────────────
  mosquitto_broker   ← subprocess Mosquitto on a free port
        │                 (skip if `mosquitto` not found on PATH)
        ▼
  cameras_yaml       ← temp cameras.yaml pointing at the broker
        │
        ▼
  live_server        ← Flask/SocketIO app in a daemon thread
        │                 env vars patched; BackboneEmbedder stubbed
        ▼
  base_url           ← "http://127.0.0.1:<port>"

  mqtt_pub           ← paho publisher client already connected to the broker
  fake_jpeg          ← minimal valid JPEG bytes for frame tests
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Generator, NamedTuple
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return a random free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_tcp(host: str, port: int, timeout: float = 10.0) -> bool:
    """Poll until a TCP connection is accepted or *timeout* seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _wait_http(url: str, timeout: float = 30.0) -> bool:
    """Poll *url* with requests until a 200 response or *timeout* elapses."""
    import requests  # noqa: PLC0415 — lazy to avoid import at collection time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# Named-tuple returned by the mosquitto_broker fixture
# ---------------------------------------------------------------------------

class BrokerInfo(NamedTuple):
    host: str
    port: int
    proc: subprocess.Popen  # type: ignore[type-arg]


# ---------------------------------------------------------------------------
# mosquitto_broker fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mosquitto_broker(tmp_path_factory: pytest.TempPathFactory) -> Generator[BrokerInfo, None, None]:
    """
    Start a Mosquitto broker process on a random free port.

    The fixture is **session-scoped** so only one broker process is created
    for the entire integration suite.

    The test session is **skipped** (not failed) when:
    * ``mosquitto`` is not found on PATH.
    * The broker process fails to bind within 5 seconds.
    """
    if not shutil.which("mosquitto"):
        pytest.skip(
            "mosquitto not found on PATH — install Mosquitto to run integration tests. "
            "On Windows: https://mosquitto.org/download/  "
            "On Linux:   sudo apt install mosquitto"
        )

    port     = _free_port()
    tmp_dir  = tmp_path_factory.mktemp("mosquitto")
    conf     = tmp_dir / "mosquitto.conf"
    pid_file = tmp_dir / "mosquitto.pid"

    conf.write_text(
        f"listener {port} 127.0.0.1\n"
        "allow_anonymous true\n"
        f"pid_file {pid_file}\n"
        "log_type error\n"
        "log_type warning\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(  # noqa: S603
        ["mosquitto", "-c", str(conf)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _wait_tcp("127.0.0.1", port, timeout=8.0):
        proc.terminate()
        pytest.skip(f"Mosquitto broker failed to start on port {port}")

    broker = BrokerInfo(host="127.0.0.1", port=port, proc=proc)
    yield broker

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# cameras_yaml fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cameras_yaml(
    mosquitto_broker: BrokerInfo,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """
    Write a minimal cameras.yaml that points at the test broker.
    Returns the path to the file.
    """
    tmp_dir = tmp_path_factory.mktemp("cameras")
    yaml_path = tmp_dir / "cameras.yaml"
    yaml_path.write_text(
        f"""\
cameras:
  - camera_id: gate-north
    name: Main Gate (North)
    door_id: door-01
    lat: 28.7041
    lon: 77.1025
    active: true
  - camera_id: parking-east
    name: Parking Lot East
    door_id: door-06
    lat: 28.7033
    lon: 77.1035
    active: false

mqtt:
  broker_host: {mosquitto_broker.host}
  broker_port: {mosquitto_broker.port}
  topic_prefix: campus
""",
        encoding="utf-8",
    )
    return yaml_path


# ---------------------------------------------------------------------------
# live_server fixture
# ---------------------------------------------------------------------------

class ServerInfo(NamedTuple):
    base_url: str
    port: int


def _stub_backbone() -> object:
    """
    Return a fake BackboneEmbedder that produces deterministic random-ish
    L2-normalised 512-D vectors without loading any PyTorch model weights.

    This lets the integration tests exercise the full enroll/verify/loitering
    pipeline without needing a GPU or pre-trained checkpoint.
    """
    import numpy as np  # noqa: PLC0415

    class _StubEmbedder:
        def eval(self):
            return self

        def __call__(self, tensor):
            """Accept a (N,3,H,W) tensor, return (N,512) numpy array."""
            import torch  # noqa: PLC0415

            batch_size = tensor.shape[0] if hasattr(tensor, "shape") else 1
            rng = np.random.default_rng(seed=42)
            vecs = rng.standard_normal((batch_size, 512)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs /= np.clip(norms, 1e-8, None)
            return torch.from_numpy(vecs)

    return _StubEmbedder()


@pytest.fixture(scope="session")
def live_server(
    mosquitto_broker: BrokerInfo,
    cameras_yaml: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[ServerInfo, None, None]:
    """
    Spin up the Flask/SocketIO app in a daemon thread on a random free port.

    Environment variables are patched so the app:
    * uses isolated temp SQLite databases (not the dev data/ directory)
    * connects to the test Mosquitto broker
    * uses a stub BackboneEmbedder (no real ML weights needed)

    The fixture blocks until ``GET /api/health`` returns 200 or 30 s elapse.
    """
    import requests  # noqa: PLC0415 — checked lazily

    port    = _free_port()
    db_dir  = tmp_path_factory.mktemp("db")
    store   = str(db_dir / "embeddings.db")
    att_db  = str(db_dir / "attendance.db")

    env_patch = {
        "PCS_STORE":           store,
        "PCS_ATTENDANCE_DB":   att_db,
        "PCS_CAMERAS_YAML":    str(cameras_yaml),
        "PCS_EMBEDDING_SALT":  "integration-test-salt-do-not-use-in-prod",
        "PCS_ENCRYPT":         "0",
        "PCS_DEVICE":          "cpu",
        # Speed up loitering / left-object in tests
        "PCS_LOITER_DWELL_S":          "3",
        "PCS_LOITER_STALE_S":          "2",
        "PCS_LOITER_COOLDOWN_S":       "5",
        "PCS_LOITER_FRAME_INTERVAL_S": "0.5",
        "PCS_LEFT_OBJ_DWELL_S":        "3",
        "PCS_LEFT_OBJ_COOLDOWN_S":     "5",
        "PCS_SAFETY_ALERT_COOLDOWN_S": "2",
        "FLASK_DEBUG": "0",
    }

    _server_exc: list[BaseException] = []
    _app_ref:    list = []   # [app, socketio]

    def _run_server():
        # Patch env vars for this thread (os.environ is process-global, so we
        # set them before create_app() reads them, then restore on exit)
        old = {k: os.environ.get(k) for k in env_patch}
        os.environ.update(env_patch)
        try:
            # Patch BackboneEmbedder BEFORE importing flask_app so the stub is
            # used throughout the application lifecycle.
            stub = _stub_backbone()
            with patch("src.models.backbone.BackboneEmbedder", return_value=stub):
                from src.web.flask_app import create_app  # noqa: PLC0415

                os.environ.update(env_patch)  # re-apply after import side-effects
                app = create_app()
                sio = app.extensions["socketio"]
                _app_ref.extend([app, sio])
                sio.run(
                    app,
                    host="127.0.0.1",
                    port=port,
                    debug=False,
                    use_reloader=False,
                    allow_unsafe_werkzeug=True,
                    log_output=False,
                )
        except Exception as exc:  # noqa: BLE001
            _server_exc.append(exc)
        finally:
            # Restore env
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    t = threading.Thread(target=_run_server, daemon=True, name="pcs-integration-server")
    t.start()

    health_url = f"http://127.0.0.1:{port}/api/health"
    ready = _wait_http(health_url, timeout=35.0)
    if not ready:
        if _server_exc:
            raise RuntimeError(f"Integration server crashed: {_server_exc[0]}") from _server_exc[0]
        raise RuntimeError(f"Integration server did not become healthy at {health_url} within 35 s")

    info = ServerInfo(base_url=f"http://127.0.0.1:{port}", port=port)
    yield info

    # Server runs in daemon thread — it will be killed when the test process exits.
    # Optionally send SIGTERM to the socketio loop if the app reference is available.


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def base_url(live_server: ServerInfo) -> str:
    return live_server.base_url


# ---------------------------------------------------------------------------
# mqtt_pub — paho publisher client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mqtt_pub(mosquitto_broker: BrokerInfo):
    """
    A paho MQTT client, already connected to the test broker, for publishing
    test messages.  Connection is kept alive for the whole session.
    """
    try:
        import paho.mqtt.client as mqtt  # type: ignore[import]
    except ImportError:
        pytest.skip("paho-mqtt not installed")

    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="pcs-test-publisher",
            clean_session=True,
        )
    except AttributeError:
        client = mqtt.Client(client_id="pcs-test-publisher", clean_session=True)  # type: ignore[call-arg]

    client.connect(mosquitto_broker.host, mosquitto_broker.port, keepalive=60)
    client.loop_start()

    yield client

    client.loop_stop()
    client.disconnect()


# ---------------------------------------------------------------------------
# fake_jpeg — minimal valid JPEG for frame publishing tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fake_jpeg() -> bytes:
    """
    Return a valid 64×64 RGB JPEG as bytes.  Uses Pillow if available,
    otherwise constructs the smallest possible valid JPEG by hand.
    """
    try:
        from PIL import Image as _PILImage  # noqa: PLC0415

        buf = io.BytesIO()
        img = _PILImage.new("RGB", (64, 64), color=(80, 80, 80))
        img.save(buf, format="JPEG", quality=30)
        return buf.getvalue()
    except ImportError:
        pass

    # Minimal valid JPEG (1×1 grey pixel) — from JFIF spec
    return bytes([
        0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
        0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
        0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
        0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
        0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
        0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
        0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
        0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
        0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
        0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0A,0x0B,0xFF,0xC4,0x00,0xB5,0x10,0x00,0x02,0x01,0x03,
        0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
        0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,
        0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,
        0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,0x33,0x62,0x72,
        0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
        0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,
        0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
        0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,
        0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
        0x8A,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,
        0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,
        0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,
        0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,0xE3,
        0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,
        0xF6,0xF7,0xF8,0xF9,0xFA,0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,
        0x00,0x3F,0x00,0xFB,0xD4,0xFF,0xD9,
    ])
