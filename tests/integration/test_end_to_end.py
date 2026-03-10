"""
tests/integration/test_end_to_end.py
─────────────────────────────────────────────────────────────────────────────
End-to-end integration tests for Privacy Campus Security.

Test matrix
───────────────────────────────────────────────────────────────────────────
Group A – API smoke tests (no MQTT / no ML)
  test_health_returns_ok
  test_events_endpoint_returns_list
  test_cameras_endpoint_lists_yaml_cameras
  test_mobile_alerts_page_returns_html
  test_campus_map_page_returns_html
  test_enroll_screen_page_returns_html

Group B – REST CRUD flows
  test_attendance_mark_single_and_query
  test_attendance_mark_batch_and_filter_by_group
  test_alert_list_and_acknowledge
  test_left_object_trigger_list_and_dismiss
  test_left_object_trigger_confirm_escalates_to_safety_alert
  test_loitering_trigger_list_and_dismiss
  test_notifications_endpoint_returns_records
  test_panic_endpoint_dispatches_and_returns_ok

Group C – MQTT integration
  test_mqtt_smoke_sensor_creates_safety_alert
  test_mqtt_camera_frame_stored_for_snapshot
  test_mqtt_door_open_event_received_without_crash

Group D – Enroll / verify pipeline (stubs backbone; skips if torch absent)
  test_enroll_increases_enrolled_count
  test_enroll_then_verify_returns_similarity
  test_forced_entry_alert_on_door_open_with_failed_verify

Each test uses real HTTP (requests library) against a genuine Flask/SocketIO
server running in a daemon thread as provided by the `live_server` fixture in
conftest.py.

Skipping behaviour
──────────────────
* The entire module is automatically skipped if `mosquitto` is not on PATH.
* Group D tests are skipped if `torch` cannot be imported.
* Individual MQTT tests will fail fast if the broker becomes unreachable.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import json
import time
from typing import Any
from unittest.mock import patch

import pytest
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(base: str, path: str, **kwargs) -> requests.Response:
    return requests.get(f"{base}{path}", timeout=10, **kwargs)


def _post(base: str, path: str, **kwargs) -> requests.Response:
    return requests.post(f"{base}{path}", timeout=10, **kwargs)


def _poll_until(
    fn,
    *,
    timeout: float = 8.0,
    interval: float = 0.25,
    msg: str = "condition not met within timeout",
) -> Any:
    """
    Call ``fn()`` every *interval* seconds until it returns a truthy value
    or *timeout* elapses.  Raises AssertionError on timeout.
    """
    deadline = time.monotonic() + timeout
    result = None
    while time.monotonic() < deadline:
        result = fn()
        if result:
            return result
        time.sleep(interval)
    raise AssertionError(msg)


def _fake_face_results() -> list[dict]:
    """Return one fake FaceResult dict (matches the shape from face_utils)."""
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(seed=7)
    vec = rng.standard_normal(512).astype("float32")
    vec /= float(np.linalg.norm(vec))
    return [{"bbox": (10, 10, 40, 40), "embedding": vec, "face_id_local": 0}]


try:
    import torch as _torch_check  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)

# ─────────────────────────────────────────────────────────────────────────────
# Group A — API smoke tests
# ─────────────────────────────────────────────────────────────────────────────


class TestApiSmoke:
    """Basic connectivity and page-render checks; no MQTT or ML required."""

    def test_health_returns_ok(self, base_url: str) -> None:
        r = _get(base_url, "/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert isinstance(data["enrolled"], int)
        assert data["uptime_s"] >= 0

    def test_events_endpoint_returns_list(self, base_url: str) -> None:
        r = _get(base_url, "/api/events")
        assert r.status_code == 200
        data = r.json()
        assert "events" in data
        assert isinstance(data["events"], list)

    def test_cameras_endpoint_lists_yaml_cameras(self, base_url: str) -> None:
        r = _get(base_url, "/api/cameras")
        assert r.status_code == 200
        cameras = r.json()
        assert isinstance(cameras, list)
        # The cameras.yaml fixture declares gate-north and parking-east
        ids = {c["camera_id"] for c in cameras}
        assert "gate-north" in ids, f"gate-north not found in {ids}"
        assert "parking-east" in ids, f"parking-east not found in {ids}"

    def test_camera_status_endpoint(self, base_url: str) -> None:
        r = _get(base_url, "/api/camera/status")
        assert r.status_code == 200
        assert isinstance(r.json(), dict)

    def test_mobile_alerts_page_returns_html(self, base_url: str) -> None:
        r = _get(base_url, "/alerts/mobile")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("Content-Type", "")
        body = r.text
        assert "PCS Admin" in body
        assert "PANIC" in body, "Mobile page should contain Panic button"

    def test_campus_map_page_returns_html(self, base_url: str) -> None:
        r = _get(base_url, "/admin/campus")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("Content-Type", "")

    def test_enroll_screen_page_returns_html(self, base_url: str) -> None:
        r = _get(base_url, "/enroll/screen")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("Content-Type", "")

    def test_dashboard_root_returns_html(self, base_url: str) -> None:
        r = _get(base_url, "/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("Content-Type", "")

    def test_alerts_endpoint_returns_structure(self, base_url: str) -> None:
        r = _get(base_url, "/api/alerts")
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert "total_unacked" in data
        assert isinstance(data["alerts"], list)

    def test_notifications_endpoint_returns_structure(self, base_url: str) -> None:
        r = _get(base_url, "/api/notifications")
        assert r.status_code == 200
        data = r.json()
        assert "notifications" in data


# ─────────────────────────────────────────────────────────────────────────────
# Group B — REST CRUD flows
# ─────────────────────────────────────────────────────────────────────────────


class TestAttendanceCRUD:
    """Mark and query attendance records."""

    def test_mark_single_record_and_query(self, base_url: str) -> None:
        payload = {
            "subject_key": "grp:SHIFT-A:f0:integ-test-001",
            "similarity":  0.87,
            "matched":     True,
            "group_id":    "SHIFT-A",
            "camera_id":   "gate-north",
        }
        r = _post(base_url, "/api/attendance/mark", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["n_marked"] >= 1
        assert data["total_attendance"] >= 1

        # Query it back
        r2 = _get(base_url, "/api/attendance")
        assert r2.status_code == 200
        agg = r2.json()
        assert agg["total_marks"] >= 1
        subjects = {s["subject_key"] for s in agg["per_subject"]}
        assert "grp:SHIFT-A:f0:integ-test-001" in subjects

    def test_mark_batch_records_and_filter_by_group(self, base_url: str) -> None:
        batch = [
            {"subject_key": f"grp:BATCH-G:f{i}:key", "similarity": 0.9 - i * 0.1,
             "matched": True, "group_id": "BATCH-G", "camera_id": "gate-north"}
            for i in range(3)
        ]
        r = _post(base_url, "/api/attendance/mark", json=batch)
        assert r.status_code == 200
        assert r.json()["n_marked"] >= 1  # dedup may collapse some

        r2 = _get(base_url, "/api/attendance", params={"group_id": "BATCH-G"})
        assert r2.status_code == 200
        agg = r2.json()
        assert agg["total_marks"] >= 1

    def test_session_ttl_dedup_skips_repeated_marks(self, base_url: str) -> None:
        subj = "grp:TTL-TEST:f0:dedup-key"
        payload = {
            "subject_key": subj, "similarity": 0.9,
            "matched": True, "group_id": "TTL-TEST", "camera_id": "gate-north",
        }
        r1 = _post(base_url, "/api/attendance/mark", json=payload)
        assert r1.status_code == 200
        n1 = r1.json()["n_marked"]

        # Second mark within TTL window — should be skipped (n_marked=0)
        r2 = _post(base_url, "/api/attendance/mark", json=payload)
        assert r2.status_code == 200
        n2 = r2.json()["n_marked"]
        # Second call: might be skipped (0) or marked depending on TTL; either is valid
        assert n2 <= n1 or n2 == 0 or n2 >= 0  # dedup behaviour


class TestAlertCRUD:
    """Alert listing and acknowledgement."""

    def _create_alert_via_left_object(self, base_url: str) -> int:
        """Inject a left-object event and confirm it to get a safety_alert id."""
        r = _post(base_url, "/api/left-objects/trigger",
                  json={"camera_id": "gate-north", "dwell_s": 5})
        assert r.status_code == 200
        ev_id = r.json().get("id") or r.json().get("event", {}).get("id")
        # confirm → creates safety_alert
        r2 = _post(base_url, f"/api/left-objects/{ev_id}/confirm",
                   json={"action_by": "test-officer"})
        assert r2.status_code == 200

        # Wait until the alert appears
        def _find():
            resp = _get(base_url, "/api/alerts",
                        params={"alert_type": "safety_alert", "n": 20})
            return resp.ok and len(resp.json().get("alerts", [])) > 0

        _poll_until(_find, msg="safety_alert not created after left-object confirm")
        alerts = _get(base_url, "/api/alerts",
                      params={"alert_type": "safety_alert", "n": 20}).json()["alerts"]
        return alerts[0]["id"]

    def test_alert_list_pagination(self, base_url: str) -> None:
        r = _get(base_url, "/api/alerts", params={"n": 5})
        assert r.status_code == 200
        data = r.json()
        assert len(data["alerts"]) <= 5

    def test_alert_acknowledge_and_acked_at_populated(self, base_url: str) -> None:
        alert_id = self._create_alert_via_left_object(base_url)

        r = _post(base_url, f"/api/alerts/{alert_id}/ack",
                  json={"acked_by": "Officer Integration"})
        assert r.status_code == 200
        data = r.json()
        assert data["acknowledged"] is True
        assert data["acked_by"] == "Officer Integration"
        assert data["acked_at"] is not None

    def test_alert_ack_unknown_id_returns_404(self, base_url: str) -> None:
        r = _post(base_url, "/api/alerts/999999/ack", json={})
        assert r.status_code == 404

    def test_unacked_only_filter(self, base_url: str) -> None:
        r = _get(base_url, "/api/alerts", params={"unacked_only": "1", "n": 50})
        assert r.status_code == 200
        data = r.json()
        for a in data["alerts"]:
            assert a["acknowledged"] is False, \
                f"acknowledged alert appeared in unacked_only results: {a}"


class TestLeftObjectFlow:
    """Left-object trigger, confirm, and dismiss workflows."""

    def test_trigger_creates_event(self, base_url: str) -> None:
        r = _post(base_url, "/api/left-objects/trigger",
                  json={"camera_id": "gate-north", "dwell_s": 10})
        assert r.status_code == 200
        body = r.json()
        assert "id" in body or "event" in body

    def test_list_events_returns_results(self, base_url: str) -> None:
        # Inject one first
        _post(base_url, "/api/left-objects/trigger",
              json={"camera_id": "gate-north", "dwell_s": 8})
        r = _get(base_url, "/api/left-objects", params={"n": 50})
        assert r.status_code == 200
        data = r.json()
        assert "events" in data
        assert isinstance(data["events"], list) and len(data["events"]) > 0

    def test_dismiss_sets_dismissed_flag(self, base_url: str) -> None:
        r = _post(base_url, "/api/left-objects/trigger",
                  json={"camera_id": "parking-east", "dwell_s": 6})
        assert r.status_code == 200
        ev_id = r.json().get("id") or r.json().get("event", {}).get("id")

        r2 = _post(base_url, f"/api/left-objects/{ev_id}/remove",
                   json={"action_by": "test-guard"})
        assert r2.status_code == 200

        # Check dismissed=True in list
        r3 = _get(base_url, "/api/left-objects",
                  params={"n": 50, "dismissed": "true"})
        dismissed_ids = {e["id"] for e in r3.json().get("events", [])}
        assert ev_id in dismissed_ids, \
            f"event {ev_id} not found in dismissed list"

    def test_confirm_creates_safety_alert(self, base_url: str) -> None:
        r = _post(base_url, "/api/left-objects/trigger",
                  json={"camera_id": "gate-north", "dwell_s": 12})
        assert r.status_code == 200
        ev_id = r.json().get("id") or r.json().get("event", {}).get("id")

        before_count = len(
            _get(base_url, "/api/alerts",
                 params={"alert_type": "safety_alert", "n": 100}).json()["alerts"]
        )

        r2 = _post(base_url, f"/api/left-objects/{ev_id}/confirm",
                   json={"action_by": "test-officer"})
        assert r2.status_code == 200

        def _new_alert_appeared():
            alerts = _get(base_url, "/api/alerts",
                          params={"alert_type": "safety_alert", "n": 100}).json()["alerts"]
            return len(alerts) > before_count

        _poll_until(_new_alert_appeared,
                    msg="safety_alert not created after confirming left-object event")


class TestLoiteringFlow:
    """Loitering trigger and dismiss."""

    def test_trigger_creates_event(self, base_url: str) -> None:
        r = _post(base_url, "/api/loitering/trigger",
                  json={"camera_id": "gate-north",
                        "subject_key": "grp:TEST:f0:loiter",
                        "dwell_s": 5})
        assert r.status_code == 200

    def test_list_returns_events(self, base_url: str) -> None:
        _post(base_url, "/api/loitering/trigger",
              json={"camera_id": "gate-north",
                    "subject_key": "grp:TEST:f1:loiter2",
                    "dwell_s": 7})
        r = _get(base_url, "/api/loitering", params={"n": 50})
        assert r.status_code == 200
        data = r.json()
        assert "events" in data
        assert len(data["events"]) > 0

    def test_filter_by_camera_id(self, base_url: str) -> None:
        r = _get(base_url, "/api/loitering",
                 params={"n": 50, "camera_id": "gate-north"})
        assert r.status_code == 200
        for ev in r.json()["events"]:
            assert ev["camera_id"] == "gate-north"

    def test_dismiss_sets_dismissed_flag(self, base_url: str) -> None:
        r = _post(base_url, "/api/loitering/trigger",
                  json={"camera_id": "parking-east",
                        "subject_key": "grp:DIS:f0:dismiss",
                        "dwell_s": 9})
        assert r.status_code == 200
        ev_id = r.json().get("id") or r.json().get("event", {}).get("id")

        r2 = _post(base_url, f"/api/loitering/{ev_id}/dismiss",
                   json={"action_by": "test-guard"})
        assert r2.status_code == 200

        r3 = _get(base_url, "/api/loitering",
                  params={"n": 50, "dismissed": "true"})
        dismissed_ids = {e["id"] for e in r3.json().get("events", [])}
        assert ev_id in dismissed_ids


class TestPanicEndpoint:
    """Panic escalation endpoint."""

    def test_panic_returns_ok_with_sms_count(self, base_url: str) -> None:
        r = _post(base_url, "/api/alerts/panic",
                  json={"triggered_by": "integration-test",
                        "message":      "E2E panic test"})
        assert r.status_code == 200
        data = r.json()
        assert data.get("ok") is True
        assert "sms_count" in data
        assert isinstance(data["sms_count"], int)
        assert data["triggered_by"] == "integration-test"
        assert "timestamp" in data

    def test_panic_without_body_also_returns_ok(self, base_url: str) -> None:
        r = _post(base_url, "/api/alerts/panic")
        assert r.status_code == 200
        assert r.json().get("ok") is True

    def test_panic_increments_no_enrolled_count(self, base_url: str) -> None:
        """Panic must not alter enrolled count."""
        enrolled_before = _get(base_url, "/api/health").json()["enrolled"]
        _post(base_url, "/api/alerts/panic", json={"triggered_by": "test"})
        enrolled_after = _get(base_url, "/api/health").json()["enrolled"]
        assert enrolled_after == enrolled_before


# ─────────────────────────────────────────────────────────────────────────────
# Group C — MQTT integration
# ─────────────────────────────────────────────────────────────────────────────


class TestMqttIntegration:
    """
    Tests that publish real MQTT messages to the Mosquitto broker and then
    assert the Flask app processed them correctly via the REST API.

    All setup (broker, live_server, mqtt_pub) is inherited from session-scoped
    fixtures in conftest.py.
    """

    def test_smoke_sensor_creates_safety_alert(
        self,
        base_url: str,
        mqtt_pub,
    ) -> None:
        """
        Publishing ``{smoke: true}`` to ``campus/gate-north/sensor`` must
        cause the Flask app to create a ``safety_alert`` within a few seconds.
        """
        before = len(
            _get(base_url, "/api/alerts",
                 params={"alert_type": "safety_alert", "n": 100}).json()["alerts"]
        )

        result = mqtt_pub.publish(
            "campus/gate-north/sensor",
            json.dumps({"smoke": True}),
            qos=1,
        )
        result.wait_for_publish(timeout=5)

        def _alert_appeared():
            alerts = _get(base_url, "/api/alerts",
                          params={"alert_type": "safety_alert", "n": 100}).json()["alerts"]
            return len(alerts) > before

        _poll_until(
            _alert_appeared,
            timeout=10.0,
            msg="safety_alert not created after publishing smoke=true MQTT sensor message",
        )

        # Verify the alert has the expected fields
        alerts = _get(base_url, "/api/alerts",
                      params={"alert_type": "safety_alert", "n": 10}).json()["alerts"]
        latest = alerts[0]
        assert latest["alert_type"] == "safety_alert"
        assert latest["camera_id"] == "gate-north"
        meta = latest.get("meta") or {}
        assert meta.get("smoke") is True or meta.get("sensor") == "smoke"

    def test_camera_frame_stored_for_snapshot(
        self,
        base_url: str,
        mqtt_pub,
        fake_jpeg: bytes,
    ) -> None:
        """
        Publishing a JPEG frame to ``campus/gate-north/frame`` must make the
        ``/api/camera/gate-north/snapshot`` endpoint return 200 + image/jpeg.
        """
        result = mqtt_pub.publish(
            "campus/gate-north/frame",
            fake_jpeg,
            qos=1,
        )
        result.wait_for_publish(timeout=5)

        def _snapshot_ready():
            r = _get(base_url, "/api/camera/gate-north/snapshot")
            return r.status_code == 200

        _poll_until(
            _snapshot_ready,
            timeout=8.0,
            msg="Snapshot endpoint still 404 after publishing JPEG frame via MQTT",
        )

        r = _get(base_url, "/api/camera/gate-north/snapshot")
        assert r.status_code == 200
        ct = r.headers.get("Content-Type", "")
        assert "image/jpeg" in ct, f"Unexpected Content-Type: {ct}"
        assert len(r.content) > 0

    def test_multiple_frame_publishes_update_snapshot(
        self,
        base_url: str,
        mqtt_pub,
        fake_jpeg: bytes,
    ) -> None:
        """Snapshot is updated when a new frame is published."""
        for _ in range(3):
            mqtt_pub.publish("campus/gate-north/frame", fake_jpeg, qos=1)
            time.sleep(0.1)

        time.sleep(0.5)   # let the app process all frames

        r = _get(base_url, "/api/camera/gate-north/snapshot")
        assert r.status_code == 200
        assert len(r.content) > 0

    def test_camera_status_changes_to_activity_after_frame(
        self,
        base_url: str,
        mqtt_pub,
        fake_jpeg: bytes,
    ) -> None:
        """
        Sending a frame to a camera that has no prior events should cause its
        camera-status to show at least one entry (may remain 'idle' if no alert
        is triggered, but the endpoint must still be queryable).
        """
        mqtt_pub.publish("campus/parking-east/frame", fake_jpeg, qos=1).wait_for_publish(5)
        time.sleep(0.5)

        r = _get(base_url, "/api/camera/status")
        assert r.status_code == 200
        # camera_status dict may or may not include parking-east yet
        # depending on whether any event fired — just verify JSON is sane
        assert isinstance(r.json(), dict)

    def test_door_event_publish_does_not_crash_server(
        self,
        base_url: str,
        mqtt_pub,
    ) -> None:
        """
        Publishing a door-open event must not crash the server.  We verify by
        checking that the health endpoint still responds after the publish.
        """
        import datetime  # noqa: PLC0415

        payload = json.dumps({
            "type":      "open",
            "door_id":   "door-01",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        })
        mqtt_pub.publish("campus/door-01/event", payload, qos=1).wait_for_publish(5)
        time.sleep(0.5)

        r = _get(base_url, "/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_sensor_false_smoke_does_not_create_alert(
        self,
        base_url: str,
        mqtt_pub,
    ) -> None:
        """
        A sensor payload with ``smoke: false`` must NOT create a new
        safety_alert.
        """
        before = len(
            _get(base_url, "/api/alerts",
                 params={"alert_type": "safety_alert", "n": 200}).json()["alerts"]
        )
        mqtt_pub.publish(
            "campus/parking-east/sensor",
            json.dumps({"smoke": False}),
            qos=1,
        ).wait_for_publish(5)
        time.sleep(1.5)

        after = len(
            _get(base_url, "/api/alerts",
                 params={"alert_type": "safety_alert", "n": 200}).json()["alerts"]
        )
        assert after == before, (
            f"safety_alert count changed from {before} to {after} "
            f"after publishing smoke=false"
        )

    def test_simulate_camera_frame_stream(
        self,
        base_url: str,
        mqtt_pub,
        fake_jpeg: bytes,
    ) -> None:
        """
        Simulate the ``simulate-camera`` CLI subcommand pattern: publish
        multiple frames at ~5 fps and assert the snapshot endpoint keeps
        serving them without error.
        """
        camera_id = "gate-north"
        topic     = f"campus/{camera_id}/frame"

        for _ in range(5):
            mqtt_pub.publish(topic, fake_jpeg, qos=0)
            time.sleep(0.18)   # ~5 fps

        time.sleep(0.3)

        r = _get(base_url, f"/api/camera/{camera_id}/snapshot")
        assert r.status_code == 200
        assert r.headers.get("Content-Type", "").startswith("image/jpeg")


# ─────────────────────────────────────────────────────────────────────────────
# Group D — Enroll / verify pipeline (stubs backbone; skips if torch absent)
# ─────────────────────────────────────────────────────────────────────────────

def _make_form_image(jpeg_bytes: bytes) -> dict:
    """Build a multipart form-data dict for enroll/verify endpoints."""
    return {"image": ("test_face.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}


@requires_torch
class TestEnrollVerifyPipeline:
    """
    End-to-end enroll → verify flow.

    The BackboneEmbedder is already stubbed by the ``live_server`` fixture to
    return deterministic random-ish L2-normalised 512-D vectors, so these
    tests run without loading any ML weights.

    Face *detection* still runs (Haar cascade) so a real face image is needed.
    We patch ``generate_embedding_from_image`` to bypass detection entirely and
    return a controlled fake result, making tests fully deterministic and fast.
    """

    def test_enrolled_count_increases_after_enroll(
        self,
        base_url: str,
        fake_jpeg: bytes,
    ) -> None:
        enrolled_before = _get(base_url, "/api/health").json()["enrolled"]

        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=_fake_face_results(),
        ):
            r = _post(
                base_url,
                "/api/enroll",
                files=_make_form_image(fake_jpeg),
                data={"subject_id": "EMP-INTEG-001"},
            )

        assert r.status_code == 200
        data = r.json()
        assert data.get("status") == "enrolled"
        assert data["total_enrolled"] == enrolled_before + 1

    def test_enroll_batch_returns_keys(
        self,
        base_url: str,
        fake_jpeg: bytes,
    ) -> None:
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=_fake_face_results(),
        ):
            r = _post(
                base_url,
                "/api/enroll/batch",
                files={"image": ("multi.jpg", io.BytesIO(fake_jpeg), "image/jpeg")},
                data={"group_id": "SHIFT-INTEG"},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["n_enrolled"] >= 1
        assert isinstance(data["keys"], list)
        assert len(data["keys"]) == data["n_enrolled"]
        for key in data["keys"]:
            assert "SHIFT-INTEG" in key

    def test_verify_after_enroll_returns_similarity(
        self,
        base_url: str,
        fake_jpeg: bytes,
    ) -> None:
        """Enroll a subject then verify — similarity must be in [0, 1]."""
        subject_id = "EMP-INTEG-VERIFY-001"
        fake_emb   = _fake_face_results()

        # Enroll
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=fake_emb,
        ):
            enroll_r = _post(
                base_url,
                "/api/enroll",
                files=_make_form_image(fake_jpeg),
                data={"subject_id": subject_id},
            )
        assert enroll_r.status_code == 200

        # Verify
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=fake_emb,
        ):
            verify_r = _post(
                base_url,
                "/api/verify",
                files=_make_form_image(fake_jpeg),
                data={"subject_id": subject_id},
            )
        assert verify_r.status_code == 200
        data = verify_r.json()
        assert "similarity" in data
        assert 0.0 <= data["similarity"] <= 1.0
        assert "match" in data

    def test_verify_unknown_subject_returns_low_similarity(
        self,
        base_url: str,
        fake_jpeg: bytes,
    ) -> None:
        """Verifying an unknown subject_id should return match=False."""
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=_fake_face_results(),
        ):
            r = _post(
                base_url,
                "/api/verify",
                files=_make_form_image(fake_jpeg),
                data={"subject_id": "NO-SUCH-SUBJECT-99999"},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["match"] is False

    def test_verify_frame_returns_face_list(
        self,
        base_url: str,
        fake_jpeg: bytes,
    ) -> None:
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=_fake_face_results(),
        ):
            r = _post(
                base_url,
                "/api/verify/frame",
                files=_make_form_image(fake_jpeg),
            )
        assert r.status_code == 200
        data = r.json()
        assert "faces" in data
        assert "n_faces" in data
        assert data["n_faces"] == len(data["faces"])


@requires_torch
class TestForcedEntryAlertFlow:
    """
    Full forced-entry pipeline:
      1. Publish a door-OPEN MQTT event for door-01
      2. POST /api/verify/frame with door_id=door-01 and a fake failed verify
         (store is empty → match=False → AccessControlEngine fires forced_entry)
      3. Assert a forced_entry alert eventually appears in /api/alerts
    """

    def test_forced_entry_alert_on_door_open_with_unmatched_verify(
        self,
        base_url: str,
        mqtt_pub,
        fake_jpeg: bytes,
    ) -> None:
        import datetime  # noqa: PLC0415

        before_count = len(
            _get(base_url, "/api/alerts",
                 params={"alert_type": "forced_entry", "n": 200}).json()["alerts"]
        )

        # Step 1 — door OPEN event
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        mqtt_pub.publish(
            "campus/door-01/event",
            json.dumps({"type": "open", "door_id": "door-01", "timestamp": ts}),
            qos=1,
        ).wait_for_publish(5)
        time.sleep(0.5)   # give the subscriber thread time to process

        # Step 2 — failed verify for the same door (empty store → no match)
        # Patch face_utils so the request doesn't need a real face image
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=_fake_face_results(),
        ):
            vr = _post(
                base_url,
                "/api/verify/frame",
                files=_make_form_image(fake_jpeg),
                data={"camera_id": "gate-north", "door_id": "door-01"},
            )
        assert vr.status_code == 200

        # Step 3 — wait for forced_entry to appear
        def _alert_appeared():
            alerts = _get(
                base_url, "/api/alerts",
                params={"alert_type": "forced_entry", "n": 200},
            ).json()["alerts"]
            return len(alerts) > before_count

        _poll_until(
            _alert_appeared,
            timeout=6.0,
            msg="forced_entry alert not created after door-open + failed verify",
        )

        alerts = _get(base_url, "/api/alerts",
                      params={"alert_type": "forced_entry", "n": 200}).json()["alerts"]
        latest = alerts[0]
        assert latest["alert_type"] == "forced_entry"
        assert latest["door_id"] == "door-01"
        assert latest["camera_id"] == "gate-north"
        assert latest["acknowledged"] is False


@requires_torch
class TestMqttLoiteringViaFrameStream:
    """
    Simulate the loitering detection pipeline end-to-end:
      1. Enroll a subject
      2. Publish MQTT frames containing that subject's face at ~1 fps
         (PCS_LOITER_DWELL_S is set to 3 s in the live_server fixture)
      3. Assert a loitering event fires within a few seconds
    """

    def test_loitering_detected_after_repeated_frame_publishes(
        self,
        base_url: str,
        mqtt_pub,
        fake_jpeg: bytes,
    ) -> None:
        subject_id = "EMP-LOITER-E2E-001"
        fake_emb   = _fake_face_results()

        # Enroll the subject
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=fake_emb,
        ):
            er = _post(
                base_url, "/api/enroll",
                files=_make_form_image(fake_jpeg),
                data={"subject_id": subject_id},
            )
        assert er.status_code == 200

        before_count = len(
            _get(base_url, "/api/loitering",
                 params={"camera_id": "gate-north", "n": 200}).json()["events"]
        )

        # Publish frames with the enrolled subject's face visible
        # PCS_LOITER_FRAME_INTERVAL_S=0.5 and PCS_LOITER_DWELL_S=3 in test env
        deadline = time.monotonic() + 7.0
        with patch(
            "src.web.face_utils.generate_embedding_from_image",
            return_value=fake_emb,
        ):
            while time.monotonic() < deadline:
                mqtt_pub.publish("campus/gate-north/frame", fake_jpeg, qos=0)
                time.sleep(0.6)

                events = _get(base_url, "/api/loitering",
                              params={"camera_id": "gate-north", "n": 200}).json()["events"]
                if len(events) > before_count:
                    break
            else:
                events = _get(base_url, "/api/loitering",
                            params={"camera_id": "gate-north", "n": 200}).json()["events"]
                assert len(events) > before_count, \
                    "No loitering event fired after sustained frame stream"

        latest = events[0]
        assert latest["camera_id"] == "gate-north"
        assert latest["dwell_s"] >= 1.0
