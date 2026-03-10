"""
notifications/policy.py – Escalation policy for security notifications.

Severity levels
---------------
* **high**   – immediate safety / security threat (safety_alert, forced_entry, tailgate)
* **medium** – elevated concern needing review (loitering, left_object)
* **low**    – informational or ambient events (everything else)

Escalation channels per severity
---------------------------------
* high   → websocket + smtp + sms (sms opt-in; sends if PCS_SMS_OPT_IN=1)
* medium → websocket + smtp
* low    → websocket only

Customising
-----------
Instantiate :class:`EscalationPolicy` with your own ``policy`` and/or
``severity_map`` dicts to override the defaults without touching this module.
"""

from __future__ import annotations

import os

# ── Severity mapping: alert_type → severity ───────────────────────────────────

SEVERITY_MAP: dict[str, str] = {
    "safety_alert": "high",
    "forced_entry": "high",
    "tailgate":     "high",
    "loitering":    "medium",
    "left_object":  "medium",
}

# ── Escalation policy: severity → ordered list of channels ────────────────────

_SMS_OPT_IN = os.environ.get("PCS_SMS_OPT_IN", "0") == "1"

ESCALATION_POLICY: dict[str, list[str]] = {
    "high":   ["websocket", "smtp", "sms"] if _SMS_OPT_IN else ["websocket", "smtp"],
    "medium": ["websocket", "smtp"],
    "low":    ["websocket"],
}


class EscalationPolicy:
    """
    Maps an alert type (or an explicit severity string) to the ordered list of
    delivery channels that should be notified.

    Parameters
    ----------
    policy:
        Override the default :data:`ESCALATION_POLICY` dict.
        Keys are severity strings; values are lists of channel names.
    severity_map:
        Override the default :data:`SEVERITY_MAP` dict.
        Keys are alert-type strings; values are severity strings.
    sms_opt_in:
        Whether the SMS channel should be included for high-severity alerts.
        When *None* (default) the value of the ``PCS_SMS_OPT_IN`` environment
        variable is used.

    Examples
    --------
    ::

        # Default policy
        p = EscalationPolicy()
        p.get_severity("safety_alert")   # → "high"
        p.get_channels("high")           # → ["websocket", "smtp"]  (sms added if opt-in)

        # Override severity map to treat loitering as high
        my_map = {**SEVERITY_MAP, "loitering": "high"}
        p = EscalationPolicy(severity_map=my_map)
    """

    def __init__(
        self,
        policy:       dict[str, list[str]] | None = None,
        severity_map: dict[str, str]        | None = None,
        sms_opt_in:   bool | None                  = None,
    ) -> None:
        # Read env var at instantiation so tests can patch it reliably
        _sms = (
            sms_opt_in
            if sms_opt_in is not None
            else (os.environ.get("PCS_SMS_OPT_IN", "0") == "1")
        )

        if policy is None:
            policy = {
                "high":   ["websocket", "smtp", "sms"] if _sms else ["websocket", "smtp"],
                "medium": ["websocket", "smtp"],
                "low":    ["websocket"],
            }

        self._policy       = policy
        self._severity_map = severity_map or dict(SEVERITY_MAP)

    # ── public API ────────────────────────────────────────────────────────────

    def get_severity(self, alert_type: str) -> str:
        """
        Return the severity level for a given alert type.

        Unrecognised types default to ``"medium"``.
        """
        return self._severity_map.get(alert_type, "medium")

    def get_channels(self, severity: str) -> list[str]:
        """
        Return the ordered list of delivery channel names for a severity level.

        Unrecognised severity levels fall back to ``["websocket"]``.
        """
        return list(self._policy.get(severity, ["websocket"]))

    def channels_for_alert(self, alert_type: str) -> list[str]:
        """Convenience: resolve alert type → severity → channels in one call."""
        return self.get_channels(self.get_severity(alert_type))
