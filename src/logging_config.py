"""
src/logging_config.py
─────────────────────────────────────────────────────────────────────────────
Structured JSON logging configuration using structlog.

Calling ``configure_logging()`` rewires **both** stdlib ``logging`` and
structlog so that every log record — whether emitted with
``logging.getLogger(__name__).info(...)`` or ``structlog.get_logger().info(...)``
— is written to stdout as a single JSON object.

Typical JSON line produced by a structlog call:

    {
      "timestamp":  "2026-03-01T12:00:00.123456Z",
      "level":      "info",
      "logger":     "src.web.flask_app",
      "event":      "verify_frame",
      "event_type": "verify",
      "camera_id":  "gate-north",
      "latency_ms": 142.3,
      "n_faces":    2
    }

Stdlib ``logging`` calls (e.g. from Flask itself or third-party libs) are
intercepted via ``structlog.stdlib.ProcessorFormatter`` and rendered in the
same JSON format.

Environment
-----------
PCS_LOG_LEVEL : str  (default: "INFO")
    One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    Also accepts lowercase.

PCS_LOG_FORMAT : "json" | "console"  (default: "json")
    "json"    → compact JSON lines (production / CI / Prometheus scrape logs)
    "console" → coloured human-readable output (use in interactive dev)

Usage
-----
    from src.logging_config import configure_logging
    configure_logging()          # call once, early (e.g. top of create_app())

    import structlog
    _slog = structlog.get_logger(__name__)
    _slog.info("verify_frame", event_type="verify", camera_id="gate-north",
               subject_key="grp:A:f0:abc", similarity=0.87, latency_ms=142.3)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from typing import Any

_configured = False   # guard against double-configuration in tests


def configure_logging(
    level: str = "INFO",
    fmt:   str = "json",
) -> None:
    """
    Configure structlog + stdlib logging to emit structured JSON.

    Safe to call multiple times (idempotent after first call unless
    ``force=True`` is used internally by tests).

    Parameters
    ----------
    level : str
        Default log level when ``PCS_LOG_LEVEL`` is not set.
    fmt : str
        Default output format when ``PCS_LOG_FORMAT`` is not set.
    """
    global _configured
    if _configured:
        return

    try:
        import structlog  # noqa: F401 (presence check — fail early if missing)
    except ImportError as exc:
        # Fall back to plain stdlib logging so the server still works.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)-30s %(levelname)-8s %(message)s",
        )
        logging.getLogger(__name__).warning(
            "structlog not installed (%s); using plain stdlib logging.", exc
        )
        _configured = True
        return

    import structlog

    log_level   = getattr(
        logging,
        os.environ.get("PCS_LOG_LEVEL", level).upper(),
        logging.INFO,
    )
    output_fmt  = os.environ.get("PCS_LOG_FORMAT", fmt).lower()

    # ── Processors applied to every log event ─────────────────────────────
    shared_processors: list[Any] = [
        # Merge context variables bound with structlog.contextvars.bind_contextvars()
        structlog.contextvars.merge_contextvars,
        # Add structlog's own log-level string ("info", "warning", …)
        structlog.stdlib.add_log_level,
        # Add the logger name so we know which module emitted the line
        structlog.stdlib.add_logger_name,
        # Expand positional printf-style args: log.info("x=%s", 1) → {"event": "x=1"}
        structlog.stdlib.PositionalArgumentsFormatter(),
        # ISO-8601 timestamp
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Render exc_info tuples into a "exception" key
        structlog.processors.ExceptionRenderer(),
        # Render stack_info into a "stack" key
        structlog.processors.StackInfoRenderer(),
    ]

    # ── Format choice ──────────────────────────────────────────────────────
    if output_fmt == "console":
        # Human-readable, coloured (uses Rich if available, else plain text)
        final_processor: Any = structlog.dev.ConsoleRenderer(
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        # Compact single-line JSON — easy to ingest with Loki / Elasticsearch
        final_processor = structlog.processors.JSONRenderer()

    # ── structlog configuration ────────────────────────────────────────────
    structlog.configure(
        processors=shared_processors + [
            # Tell structlog to hand events off to the stdlib handler via
            # ProcessorFormatter rather than printing them directly.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory   = structlog.stdlib.LoggerFactory(),
        wrapper_class    = structlog.stdlib.BoundLogger,
        cache_logger_on_first_use = True,
    )

    # ── stdlib formatter that wraps structlog's pipeline ──────────────────
    formatter = structlog.stdlib.ProcessorFormatter(
        # foreign_pre_chain handles records emitted via stdlib (Flask, paho, …)
        foreign_pre_chain = shared_processors,
        processors = [
            # Remove the internal _record/_from_structlog metadata keys
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final_processor,
        ],
    )

    # ── Root handler (stdout) ─────────────────────────────────────────────
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Clear any handlers added by Flask/Werkzeug before we configure
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Suppress noisy third-party loggers
    for noisy in ("werkzeug", "engineio", "socketio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
