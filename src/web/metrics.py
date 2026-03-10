"""
src/web/metrics.py
─────────────────────────────────────────────────────────────────────────────
Prometheus metrics for Privacy Campus Security.

All metric objects are module-level singletons so they persist across
requests and are safe to import from multiple modules.

Exported metrics
----------------
VERIFY_REQUESTS   Counter  verify_requests_total{status}
    Incremented by POST /api/verify/frame.
    ``status`` label: "success" (HTTP 200) | "error" (non-200 or exception)

ALERTS_TOTAL      Counter  alerts_total{alert_type}
    Incremented whenever AccessControlEngine fires a new alert.
    ``alert_type`` label: "forced_entry" | "tailgate" | "safety_alert"

ATTENDANCE_MARKS  Counter  attendance_marks_total
    Incremented by n_marked on each POST /api/attendance/mark call.
    Deduplicated (TTL-skipped) records are NOT counted.

VERIFY_LATENCY    Histogram  verify_latency_seconds
    End-to-end wall-clock latency of POST /api/verify/frame
    (face detection + batched embedding + store lookup).
    Buckets chosen around the 200 ms CPU evaluation target.

Usage in flask_app.py
---------------------
    from src.web.metrics import (
        VERIFY_REQUESTS, VERIFY_LATENCY,
        ALERTS_TOTAL, ATTENDANCE_MARKS,
        metrics_response,
    )

    # In verify_frame():
    with VERIFY_LATENCY.time():
        ...
    VERIFY_REQUESTS.labels(status="success").inc()

    # In _on_alert():
    ALERTS_TOTAL.labels(alert_type=alert.get("alert_type", "unknown")).inc()

    # In attendance_mark():
    ATTENDANCE_MARKS.inc(n_marked)

    # /metrics route:
    @app.get("/metrics")
    def prometheus_metrics():
        return metrics_response()

Prometheus scrape config (prometheus.yml)
-----------------------------------------
    scrape_configs:
      - job_name: privacy_campus_security
        static_configs:
          - targets: ['localhost:5000']
        metrics_path: /metrics
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Tuple

# prometheus_client is an optional dependency; gracefully degrade if absent.
try:
    from prometheus_client import (
        Counter,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


# ── Metric definitions ────────────────────────────────────────────────────────

if _PROM_AVAILABLE:
    VERIFY_REQUESTS: "Counter" = Counter(
        "verify_requests_total",
        "Total calls to POST /api/verify/frame",
        ["status"],   # "success" | "error"
    )

    ALERTS_TOTAL: "Counter" = Counter(
        "alerts_total",
        "Total security alerts created by the access-control engine",
        ["alert_type"],   # forced_entry | tailgate | safety_alert
    )

    ATTENDANCE_MARKS: "Counter" = Counter(
        "attendance_marks_total",
        "Total attendance records inserted (TTL-deduplicated records excluded)",
    )

    VERIFY_LATENCY: "Histogram" = Histogram(
        "verify_latency_seconds",
        "Wall-clock latency of POST /api/verify/frame (face detection + embedding + lookup)",
        # Buckets: fine-grained around the 200 ms evaluation target
        buckets=[
            0.010, 0.025, 0.050, 0.075,
            0.100, 0.125, 0.150, 0.175,
            0.200, 0.250, 0.300, 0.400,
            0.500, 0.750, 1.000, 2.000,
            float("inf"),
        ],
    )

else:
    # Stub objects so the rest of the codebase can import and call .inc() / .labels()
    # without conditionals everywhere — operations become no-ops.

    class _NoOpCounter:                                             # pragma: no cover
        def inc(self, amount: float = 1) -> None: pass
        def labels(self, **_kw) -> "_NoOpCounter": return self

    class _NoOpHistogram:                                          # pragma: no cover
        def observe(self, val: float) -> None: pass
        def time(self):
            import contextlib, time as _t
            @contextlib.contextmanager
            def _ctx():
                yield
            return _ctx()
        def labels(self, **_kw) -> "_NoOpHistogram": return self

    VERIFY_REQUESTS  = _NoOpCounter()   # type: ignore[assignment]
    ALERTS_TOTAL     = _NoOpCounter()   # type: ignore[assignment]
    ATTENDANCE_MARKS = _NoOpCounter()   # type: ignore[assignment]
    VERIFY_LATENCY   = _NoOpHistogram() # type: ignore[assignment]


# ── Response helper ───────────────────────────────────────────────────────────

def metrics_response() -> "Tuple[bytes, int, dict]":
    """
    Return a Flask-compatible (body, status, headers) triple for the
    ``/metrics`` endpoint.

    When prometheus_client is not installed, returns a plain-text
    explanation with HTTP 503 so Prometheus scrapes fail loudly instead
    of silently returning empty data.
    """
    if not _PROM_AVAILABLE:
        body = (
            b"# prometheus_client not installed.\n"
            b"# Install with: pip install prometheus_client>=0.20\n"
        )
        return body, 503, {"Content-Type": "text/plain; charset=utf-8"}

    body = generate_latest(REGISTRY)
    return body, 200, {"Content-Type": CONTENT_TYPE_LATEST}
