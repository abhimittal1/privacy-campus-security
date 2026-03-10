"""
src/notifications — Notification subsystem for Privacy Campus Security.

Public API
----------
NotificationStore      – SQLite-backed persistence for notification records.
EscalationPolicy       – Maps alert severity → list of delivery channels.
WebSocketDriver        – Emits Socket.IO events (always active).
SmtpDriver             – Sends email via smtplib (configurable via env vars).
SmsDriverBase          – Abstract base for SMS; subclass to add a vendor.
NullSmsDriver          – Default no-op SMS driver (no vendor credentials baked in).
NotificationDispatcher – Orchestrates policy, drivers, retry, and throttling.
"""

from .dispatcher import NotificationDispatcher
from .drivers    import NullSmsDriver, SmtpDriver, SmsDriverBase, WebSocketDriver
from .policy     import EscalationPolicy, SEVERITY_MAP, ESCALATION_POLICY
from .store      import NotificationStore

__all__ = [
    "NotificationDispatcher",
    "NotificationStore",
    "EscalationPolicy",
    "SEVERITY_MAP",
    "ESCALATION_POLICY",
    "WebSocketDriver",
    "SmtpDriver",
    "SmsDriverBase",
    "NullSmsDriver",
]
