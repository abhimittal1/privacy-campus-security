"""
data_provider.py – Central data broker for the Flask dashboard.

Reads experiment artefacts (CSV, JSON, PNG) produced by ``src/cli/run.py
eval-lfw`` and ``src/eval/evaluate_lfw.py``, caches them in memory, and
exposes clean Python dicts that the Flask routes serialise to JSON.

In-memory event log
-------------------
Enrollment and verification events are appended to an in-process ring buffer
(``_EVENT_LOG``).  On a real deployment replace the ``record_event()`` call
site with a subscription to Mosquitto/MQTT:

    # ── MQTT hook (future) ─────────────────────────────────────────────────
    # import paho.mqtt.client as mqtt
    # def _on_message(client, userdata, msg):
    #     payload = json.loads(msg.payload.decode())
    #     record_event(payload["type"], payload["subject_id"], payload.get("meta", {}))
    # mqtt_client = mqtt.Client()
    # mqtt_client.on_message = _on_message
    # mqtt_client.connect("localhost", 1883)
    # mqtt_client.subscribe("pcs/events/#")
    # mqtt_client.loop_start()
    # ── end MQTT hook ──────────────────────────────────────────────────────

Anomaly scores
--------------
Anomaly scores are read from ``experiments/anomaly_scores.csv`` when present.
Populate that file from the autoencoder module (src/models/autoencoder.py) or
any external sensor.  Expected columns: ``timestamp``, ``score``, ``label``.

React / SPA upgrade path
-------------------------
All JSON endpoints are CORS-ready (set ``CORS_ORIGINS`` env var).
Replace ``render_template('dashboard.html')`` with a static ``index.html``
from a ``dist/`` folder built by Vite/CRA and every endpoint becomes a
pure REST API consumed by a React frontend.
"""

from __future__ import annotations

import base64
import collections
import csv
import datetime
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Configuration (override via env vars) ─────────────────────────────────────

EXPERIMENTS_DIR = Path(os.environ.get("PCS_EXPERIMENTS_DIR", "experiments"))
_LFW_RESULTS_DIR = EXPERIMENTS_DIR / "lfw_results"
_ANOMALY_CSV = EXPERIMENTS_DIR / "anomaly_scores.csv"

# Maximum events stored in the ring buffer
MAX_EVENTS: int = 500

# Cache TTL in seconds (re-reads files from disk every N seconds)
METRICS_CACHE_TTL: float = float(os.environ.get("PCS_CACHE_TTL", "30"))

# ── In-memory event log ───────────────────────────────────────────────────────

_EVENT_LOG: collections.deque[dict[str, Any]] = collections.deque(maxlen=MAX_EVENTS)
_event_lock = threading.Lock()
_event_counter = 0


def record_event(
    kind: str,
    subject_id: Optional[str] = None,
    meta: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Append an event to the ring buffer.

    Args:
        kind:       Event type string, e.g. ``"enroll"``, ``"verify"``,
                    ``"anomaly"``, ``"alarm"``.
        subject_id: Identity involved (``None`` for system events).
        meta:       Arbitrary extra payload (similarity score, match result, …).

    Returns:
        The newly created event dict.

    Plug-in point
    -------------
    Call this function from any MQTT subscriber, edge-node webhook, or IoT
    gateway handler to stream live events into the dashboard without touching
    the Flask routes.
    """
    global _event_counter
    with _event_lock:
        _event_counter += 1
        event: dict[str, Any] = {
            "id":         _event_counter,
            "kind":       kind,
            "subject_id": subject_id,
            "timestamp":  datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "meta":       meta or {},
        }
        _EVENT_LOG.appendleft(event)
    return event


def get_events(n: int = 50) -> list[dict[str, Any]]:
    """Return the *n* most recent events (newest first)."""
    with _event_lock:
        return list(_EVENT_LOG)[:n]


# ── Metrics cache ─────────────────────────────────────────────────────────────

_metrics_cache: dict[str, Any] = {}
_metrics_cache_ts: float = 0.0


def _load_json(path: Path) -> Optional[dict]:
    """Read a JSON file; return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.debug("Could not load %s: %s", path, exc)
        return None


def _load_csv_as_dicts(path: Path) -> list[dict]:
    """Read a CSV file and return a list of row dicts."""
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:
        log.debug("Could not load %s: %s", path, exc)
        return []


def _png_to_data_uri(path: Path) -> Optional[str]:
    """Encode a PNG file as a base64 data URI for inline embedding."""
    try:
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def get_metrics(force_reload: bool = False) -> dict[str, Any]:
    """
    Load and cache evaluation metrics from the experiments directory.

    Searches the following paths (in order of preference):
    - ``experiments/lfw_results/lfw_few_shot_sweep.csv``   → K-shot table
    - ``experiments/lfw_results/lfw_pairs_metrics.json``   → pairs verification
    - ``experiments/lfw_results/roc_curve.png``            → ROC plot (base64)
    - ``experiments/demo_run/results/…``                   → demo-run fallback

    Returns
    -------
    dict with keys:
        ``k_shot_rows``    – list of dicts from the few-shot sweep CSV
        ``pairs_metrics``  – dict from pairs metrics JSON (or {})
        ``roc_image_uri``  – base64 data URI of the ROC PNG (or None)
        ``anomaly_scores`` – list of dicts from anomaly CSV (or [])
        ``source_dir``     – string path of the directory actually used
    """
    import time

    global _metrics_cache, _metrics_cache_ts
    now = time.monotonic()
    if not force_reload and _metrics_cache and (now - _metrics_cache_ts) < METRICS_CACHE_TTL:
        return _metrics_cache

    # Candidate result directories (most recent / authoritative first)
    candidate_dirs = [
        _LFW_RESULTS_DIR,
        EXPERIMENTS_DIR / "demo_run" / "results",
        EXPERIMENTS_DIR / "sample_runs",
    ]

    result_dir: Optional[Path] = None
    for d in candidate_dirs:
        if d.exists():
            result_dir = d
            break

    payload: dict[str, Any] = {
        "k_shot_rows":    [],
        "pairs_metrics":  {},
        "roc_image_uri":  None,
        "anomaly_scores": [],
        "source_dir":     str(result_dir) if result_dir else "",
    }

    if result_dir is None:
        log.debug("No experiments directory found – returning empty metrics.")
        _metrics_cache = payload
        _metrics_cache_ts = now
        return payload

    # Few-shot sweep CSV
    sweep_csv = result_dir / "lfw_few_shot_sweep.csv"
    if not sweep_csv.exists():
        sweep_csv = result_dir / "metrics_summary.csv"
    payload["k_shot_rows"] = _load_csv_as_dicts(sweep_csv)

    # Pairs metrics JSON
    for name in ("lfw_pairs_metrics.json", "pairs_metrics.json"):
        pj = result_dir / name
        if pj.exists():
            payload["pairs_metrics"] = _load_json(pj) or {}
            break

    # ROC curve PNG
    for name in ("roc_curve.png", "report.png"):
        roc_png = result_dir / name
        if roc_png.exists():
            payload["roc_image_uri"] = _png_to_data_uri(roc_png)
            break

    # Anomaly scores CSV (global location)
    payload["anomaly_scores"] = _load_csv_as_dicts(_ANOMALY_CSV)

    log.info("Metrics reloaded from %s", result_dir)
    _metrics_cache = payload
    _metrics_cache_ts = now
    return payload


# ── Episode accuracy history ──────────────────────────────────────────────────

def get_episode_accuracy_series() -> list[dict[str, Any]]:
    """
    Build a Chart.js-ready dataset from the K-shot sweep CSV.

    Returns a list of ``{"k": int, "accuracy": float, "ci95": float}`` dicts
    sorted by K ascending.
    """
    metrics = get_metrics()
    series = []
    for row in metrics["k_shot_rows"]:
        try:
            series.append({
                "k":        int(float(row.get("k_shot", row.get("K", 1)))),
                "accuracy": round(float(row.get("mean_accuracy", row.get("accuracy", 0))), 4),
                "ci95":     round(float(row.get("ci95", 0)), 4),
            })
        except (ValueError, TypeError):
            continue
    series.sort(key=lambda r: r["k"])
    return series


# ── Anomaly timeline ──────────────────────────────────────────────────────────

def get_anomaly_timeline(n: int = 200) -> list[dict[str, Any]]:
    """
    Return the last *n* rows of the anomaly score CSV as JSON-serialisable dicts.

    Plug-in point
    -------------
    Replace this function body with a database query or MQTT-backed buffer
    once the autoencoder is wired to a live video stream.
    """
    metrics = get_metrics()
    rows = metrics["anomaly_scores"][-n:]
    timeline = []
    for row in rows:
        try:
            timeline.append({
                "timestamp": row.get("timestamp", ""),
                "score":     round(float(row.get("score", 0)), 4),
                "label":     row.get("label", ""),
            })
        except (ValueError, TypeError):
            continue
    return timeline
