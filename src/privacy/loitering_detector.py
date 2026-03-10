"""
loitering_detector.py – Time-based loitering tracker.

Design
------
The detector is a lightweight, purely time-based tracker.  It does **not**
perform face recognition itself; the caller is responsible for supplying
identified ``subject_key`` strings.

Per-camera, per-subject state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    _states[camera_id][subject_key] = _SubjectState(
        first_seen_s : float   # time.monotonic() when subject was first observed
        last_seen_s  : float   # time.monotonic() of most recent observation
        fired        : bool    # True once a loitering event has been emitted
        cooldown_until_s : float  # time.monotonic() before a repeat event is allowed
    )

Detection logic
~~~~~~~~~~~~~~~
* ``observe(camera_id, subject_key)`` is called each time a face is matched.
  - If the subject is new → create a state entry with ``first_seen_s = now``.
  - Update ``last_seen_s = now``.
  - If ``last_seen_s − first_seen_s ≥ dwell_s`` **and** not fired (or cooldown
    elapsed) → **emit** a loitering detection dict and set ``fired = True``.

* ``purge_stale(camera_id, active_keys, now_s)`` removes subjects who have NOT
  been seen in the current frame tick and whose ``last_seen_s`` is older than
  ``stale_s`` seconds.  Resetting their state means the dwell clock resets if
  the same person re-enters the scene after being absent.

Usage
-----
::

    from src.privacy.loitering_detector import LoiteringDetector

    detector = LoiteringDetector(dwell_s=60.0)

    # called once per recognised face per camera frame:
    events = detector.observe("gate-north", "grp:A:f0:abc123")
    for ev in events:
        # ev = {camera_id, subject_key, first_seen_utc, detected_utc, dwell_s}
        store.add(ev["camera_id"], ev["subject_key"], dwell_s=ev["dwell_s"])

    # called once per frame tick to remove faces that left the scene:
    detector.purge_stale("gate-north", active_keys={"grp:A:f0:abc123"})
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Set


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monotonic_to_utc(mono_s: float) -> str:
    """
    Convert a time.monotonic() value to a UTC ISO-8601 string by computing the
    offset between monotonic time and wall-clock time at this instant.
    """
    wall = datetime.now(timezone.utc)
    mono_now = time.monotonic()
    delta = mono_s - mono_now
    utc = wall.timestamp() + delta
    return datetime.fromtimestamp(utc, tz=timezone.utc).isoformat()


@dataclass
class _SubjectState:
    first_seen_s: float
    last_seen_s: float
    fired: bool = False
    cooldown_until_s: float = 0.0


class LoiteringDetector:
    """
    Thread-safe loitering tracker.

    Each call to :meth:`observe` feeds one ``(camera_id, subject_key)``
    observation.  When the same subject has been continuously present for at
    least ``dwell_s`` seconds the method returns a loitering detection dict.

    Parameters
    ----------
    dwell_s     : float   Minimum continuous presence to trigger loitering (s).
    stale_s     : float   Seconds of absence before a subject's dwell timer resets.
    cooldown_s  : float   Seconds before the **same** subject can trigger again.
    """

    def __init__(
        self,
        dwell_s: float = 60.0,
        stale_s: float = 10.0,
        cooldown_s: float = 120.0,
    ) -> None:
        self.dwell_s    = float(dwell_s)
        self.stale_s    = float(stale_s)
        self.cooldown_s = float(cooldown_s)

        # camera_id → {subject_key → _SubjectState}
        self._states: Dict[str, Dict[str, _SubjectState]] = {}
        self._lock = threading.Lock()

    # ── public API ────────────────────────────────────────────────────────────

    def observe(
        self,
        camera_id: str,
        subject_key: str,
        now_s: float | None = None,
    ) -> list[dict]:
        """
        Record that *subject_key* was seen at *camera_id* right now.

        Returns a list of detection dicts (usually 0 or 1) if the dwell
        threshold is crossed for the first time (or after a cooldown).

        Each detection dict contains::

            {
              "camera_id":      str,
              "subject_key":    str,
              "first_seen_utc": str,   # ISO-8601 UTC
              "detected_utc":   str,   # ISO-8601 UTC (now)
              "dwell_s":        float,
            }
        """
        if now_s is None:
            now_s = time.monotonic()

        detections: list[dict] = []

        with self._lock:
            cam_states = self._states.setdefault(camera_id, {})
            state = cam_states.get(subject_key)

            if state is None:
                # First time we see this subject in this camera
                state = _SubjectState(first_seen_s=now_s, last_seen_s=now_s)
                cam_states[subject_key] = state
            else:
                # Update last-seen timestamp
                state.last_seen_s = now_s

            # Check dwell threshold
            dwell_elapsed = state.last_seen_s - state.first_seen_s
            cooldown_ok   = now_s >= state.cooldown_until_s

            if dwell_elapsed >= self.dwell_s and (not state.fired or cooldown_ok):
                state.fired = True
                state.cooldown_until_s = now_s + self.cooldown_s
                detections.append({
                    "camera_id":      camera_id,
                    "subject_key":    subject_key,
                    "first_seen_utc": _monotonic_to_utc(state.first_seen_s),
                    "detected_utc":   _monotonic_to_utc(now_s),
                    "dwell_s":        round(dwell_elapsed, 2),
                })

        return detections

    def purge_stale(
        self,
        camera_id: str,
        active_keys: Set[str] | None = None,
        now_s: float | None = None,
    ) -> None:
        """
        Remove subjects who have been absent for longer than ``stale_s`` seconds.

        Call this once per frame tick *after* all ``observe()`` calls for that
        frame, passing the set of subject keys that were detected in this frame.
        Subjects not present in *active_keys* and absent longer than ``stale_s``
        will have their dwell state removed — their timer restarts if they
        re-appear.

        Parameters
        ----------
        camera_id   : which camera's state to purge.
        active_keys : set of subject_keys seen in the *current* frame.
                      If None, all subjects for this camera are purged when stale.
        now_s       : current monotonic time; defaults to ``time.monotonic()``.
        """
        if now_s is None:
            now_s = time.monotonic()
        if active_keys is None:
            active_keys = set()

        with self._lock:
            cam_states = self._states.get(camera_id)
            if not cam_states:
                return

            stale_keys = [
                key for key, state in cam_states.items()
                if key not in active_keys
                and (now_s - state.last_seen_s) > self.stale_s
            ]
            for key in stale_keys:
                del cam_states[key]

    def active_subjects(self, camera_id: str) -> list[str]:
        """Return the list of subject_keys currently tracked for *camera_id*."""
        with self._lock:
            return list(self._states.get(camera_id, {}).keys())

    def reset_camera(self, camera_id: str) -> None:
        """Clear all tracked subjects for a camera (e.g. camera offline)."""
        with self._lock:
            self._states.pop(camera_id, None)

    def reset_all(self) -> None:
        """Clear all state across all cameras."""
        with self._lock:
            self._states.clear()
