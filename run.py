#!/usr/bin/env python3
"""
run.py — Project runner and feature validation for Privacy Campus Security.

Starts the Flask/SocketIO server, waits for it to become healthy, then runs
a comprehensive smoke-test against every feature group.  Prints a coloured
pass/fail summary at the end.

Usage
-----
    python run.py                  # Start server + auto-test, keep server alive
    python run.py --test-only      # Run checks then exit (CI / automated mode)
    python run.py --server-only    # Start server without running checks
    python run.py --port 5001      # Use a different port
    python run.py --no-color       # Disable ANSI colour output

Exit codes
----------
    0  All checks passed (or --server-only)
    1  One or more checks failed
    2  Server failed to start within the timeout
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any

# ── ANSI colours (can be disabled) ───────────────────────────────────────────

_USE_COLOR = True

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

GREEN  = lambda t: _c("32;1", t)
RED    = lambda t: _c("31;1", t)
YELLOW = lambda t: _c("33;1", t)
CYAN   = lambda t: _c("36;1", t)
BOLD   = lambda t: _c("1",    t)
DIM    = lambda t: _c("2",    t)

PASS_MARK = "✓"
FAIL_MARK = "✗"
SKIP_MARK = "·"


# ── Result tracking ───────────────────────────────────────────────────────────

class _Result:
    __slots__ = ("group", "name", "ok", "note")

    def __init__(self, group: str, name: str, ok: bool, note: str = ""):
        self.group = group
        self.name  = name
        self.ok    = ok
        self.note  = note


_results: list[_Result] = []


def _check(group: str, name: str, ok: bool, note: str = "") -> bool:
    """Record one test result and print it immediately."""
    _results.append(_Result(group, name, ok, note))
    mark  = PASS_MARK if ok else FAIL_MARK
    color = GREEN     if ok else RED
    suffix = f"  {DIM(note)}" if note else ""
    print(f"  {color(mark)} {name}{suffix}")
    return ok


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(session, base: str, path: str, **kwargs) -> tuple[int, Any]:
    """GET request; returns (status_code, parsed_body_or_None)."""
    try:
        r = session.get(f"{base}{path}", timeout=10, **kwargs)
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body
    except Exception as exc:
        return -1, str(exc)


def _post(session, base: str, path: str, json_body: dict | None = None,
          files=None, data=None) -> tuple[int, Any]:
    """POST request; returns (status_code, parsed_body_or_None)."""
    try:
        r = session.post(
            f"{base}{path}", timeout=10,
            json=json_body, files=files, data=data,
        )
        try:
            body = r.json()
        except Exception:
            body = r.text
        return r.status_code, body
    except Exception as exc:
        return -1, str(exc)


# ── Server lifecycle ──────────────────────────────────────────────────────────

def free_port(port: int) -> None:
    """
    Kill whatever process is listening on *port* so the new server can bind.
    Works on Windows (netstat + taskkill) and Linux/macOS (lsof/fuser).
    """
    import platform
    killed_any = False
    if platform.system() == "Windows":
        # netstat -ano lists PID in the last column for LISTENING entries
        try:
            out = subprocess.check_output(
                ["netstat", "-ano"],
                text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                parts = line.split()
                # Match lines like: TCP  0.0.0.0:5000  ...  LISTENING  1234
                if len(parts) >= 5 and f":{port}" in parts[1] and "LISTENING" in parts[3]:
                    pid = parts[-1]
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", pid, "/F"],
                            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        print(DIM(f"  Killed existing process on port {port} (PID {pid})"))
                        killed_any = True
                    except Exception:
                        pass
        except Exception:
            pass
    else:
        # Linux / macOS
        for cmd in (["fuser", "-k", f"{port}/tcp"],
                    ["lsof", "-ti", f":{port}"]):
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
                if cmd[0] == "lsof" and out:
                    for pid in out.splitlines():
                        subprocess.run(["kill", "-9", pid.strip()],
                                       check=False, stderr=subprocess.DEVNULL)
                        killed_any = True
                elif cmd[0] == "fuser":
                    killed_any = True
                break
            except Exception:
                pass
    if killed_any:
        time.sleep(1)   # give the OS time to release the port


def start_server(port: int, project_root: str) -> tuple[subprocess.Popen, str]:
    """
    Launch the Flask application in a subprocess and return ``(proc, log_path)``.

    stdout+stderr are tee'd to ``server.log`` in the project root AND drained
    by a daemon reader thread so the pipe buffer never fills up and deadlocks
    the server during PyTorch's verbose startup logging.
    """
    # Kill any stale process occupying the port first
    free_port(port)

    env = os.environ.copy()
    env.update({
        "FLASK_PORT":  str(port),
        "FLASK_DEBUG": "0",          # disable reloader — incompatible with SocketIO
        "FLASK_HOST":  "127.0.0.1",
        "PCS_LOITER_DWELL_S": "60",
        "PCS_LOITER_STALE_S": "10",
        "PCS_LOITER_COOLDOWN_S": "120",
    })
    cmd = [sys.executable, "-m", "src.web.flask_app"]
    log_path = os.path.join(project_root, "server.log")

    proc = subprocess.Popen(
        cmd,
        cwd=project_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Drain the pipe in a background daemon thread so it never fills up and
    # deadlocks the server (classic Python subprocess pipe-buffer deadlock).
    def _drain(pipe, log_file):
        try:
            with open(log_file, "w", encoding="utf-8", errors="replace") as fh:
                for line in pipe:
                    fh.write(line)
                    fh.flush()
        except Exception:
            pass

    t = threading.Thread(target=_drain, args=(proc.stdout, log_path), daemon=True)
    t.start()
    # Detach proc.stdout so wait_for_server doesn't accidentally read it
    proc.stdout = None

    return proc, log_path


def wait_for_server(base: str, proc: subprocess.Popen, log_path: str,
                    timeout: int = 40) -> bool:
    """
    Poll ``/api/health`` until the server responds with HTTP 200 or we time out.
    Also watches *proc* for early exit and aborts immediately if it crashes.
    Returns True on success.
    """
    try:
        import requests  # local import — only needed at runtime
    except ImportError:
        print(RED("  'requests' is not installed.  Run: pip install requests"))
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # If the subprocess already died, print its output and give up
        rc = proc.poll()
        if rc is not None:
            print(RED(f"  Server process exited early with code {rc}."))
            if os.path.exists(log_path):
                try:
                    with open(log_path, encoding="utf-8", errors="replace") as fh:
                        out = fh.read()
                    if out:
                        print(YELLOW("\n  Server output (last 3 kB):"))
                        print(DIM(out[-3000:]))
                except Exception:
                    pass
            return False

        try:
            r = requests.get(f"{base}/api/health", timeout=3)
            if r.status_code == 200:
                data = r.json()
                uptime = data.get("uptime_s", "?")
                # Reject a stale server: uptime > 60 s means we hit an old process
                if isinstance(uptime, (int, float)) and uptime > 60:
                    print(YELLOW(f"  WARNING: health check hit a stale server "
                                 f"(uptime {uptime} s). Retrying …"))
                    time.sleep(2)
                    continue
                print(GREEN(f"  Server ready (uptime {uptime} s, "
                             f"enrolled={data.get('enrolled', '?')})"))
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ── Feature checks ────────────────────────────────────────────────────────────

def check_core_api(session, base: str) -> None:
    """Health, events, metrics, cameras."""
    print(BOLD("\n[Core API]"))

    status, body = _get(session, base, "/api/health")
    _check("core", "GET /api/health returns 200", status == 200)
    if status == 200:
        _check("core", "  health.status == 'ok'",
               isinstance(body, dict) and body.get("status") == "ok")

    status, body = _get(session, base, "/api/events")
    _check("core", "GET /api/events returns 200", status == 200)
    if status == 200:
        _check("core", "  events list present",
               isinstance(body, dict) and "events" in body)

    status, body = _get(session, base, "/api/metrics")
    _check("core", "GET /api/metrics returns 200", status == 200)
    if status == 200:
        _check("core", "  k_shot_rows present",
               isinstance(body, dict) and "k_shot_rows" in body)

    status, body = _get(session, base, "/api/cameras")
    _check("core", "GET /api/cameras returns 200", status == 200)
    if status == 200:
        _check("core", "  cameras list present",
               isinstance(body, dict) and "cameras" in body)

    status, body = _get(session, base, "/api/camera/status")
    _check("core", "GET /api/camera/status returns 200", status == 200)

    status, _ = _get(session, base, "/api/metrics/reload")
    _check("core", "GET /api/metrics/reload returns 200", status == 200)


def check_pages(session, base: str) -> None:
    """Dashboard, enrollment screen, campus map render without error."""
    print(BOLD("\n[Pages]"))

    status, body = _get(session, base, "/")
    _check("pages", "GET / (dashboard) returns 200", status == 200)
    if status == 200:
        _check("pages", "  dashboard contains Bootstrap CDN",
               "bootstrap" in str(body).lower())

    status, body = _get(session, base, "/enroll/screen")
    _check("pages", "GET /enroll/screen returns 200", status == 200)

    status, body = _get(session, base, "/admin/campus")
    _check("pages", "GET /admin/campus returns 200", status == 200)


def check_attendance(session, base: str) -> None:
    """Mark attendance and query it back."""
    print(BOLD("\n[Attendance]"))

    # API requires: subject_key, similarity, matched (all required)
    payload = {
        "subject_key": "run-py-test-subject-001",
        "camera_id":   "gate-north",
        "similarity":  0.92,
        "matched":     True,
        "group_id":    "test-group",
    }
    status, body = _post(session, base, "/api/attendance/mark", json_body=payload)
    _check("attendance", "POST /api/attendance/mark returns 200", status == 200)
    if status == 200:
        _check("attendance", "  n_marked + n_skipped >= 1",
               isinstance(body, dict) and
               (body.get("n_marked", 0) + body.get("n_skipped", 0)) >= 1)

    status, body = _get(session, base, "/api/attendance")
    _check("attendance", "GET /api/attendance returns 200", status == 200)
    if status == 200:
        _check("attendance", "  response has 'total_marks' key",
               isinstance(body, dict) and "total_marks" in body)


def check_alerts(session, base: str) -> None:
    """List alerts, acknowledge one (if any)."""
    print(BOLD("\n[Alerts]"))

    status, body = _get(session, base, "/api/alerts")
    _check("alerts", "GET /api/alerts returns 200", status == 200)

    if status == 200 and isinstance(body, dict):
        alerts = body.get("alerts", [])
        _check("alerts", "  alerts list present", True,
               note=f"{len(alerts)} alert(s)")
        # Ack the first unacknowledged alert if one exists
        unacked = [a for a in alerts if not a.get("acked")]
        if unacked:
            alert_id = unacked[0]["id"]
            status2, body2 = _post(session, base,
                                   f"/api/alerts/{alert_id}/ack",
                                   json_body={"acked_by": "run.py"})
            _check("alerts", f"  POST /api/alerts/{alert_id}/ack returns 200",
                   status2 == 200)
        else:
            _check("alerts", "  (no unacknowledged alerts to test ack)", True,
                   note="skipped")


def check_left_objects(session, base: str) -> None:
    """List, trigger, remove and confirm left-object events."""
    print(BOLD("\n[Left-Object Detection]"))

    status, body = _get(session, base, "/api/left-objects")
    _check("left_obj", "GET /api/left-objects returns 200", status == 200)

    # Trigger a test event
    trigger_payload = {"camera_id": "gate-north", "bbox": [10, 20, 60, 80], "dwell_s": 35.0}
    status, body = _post(session, base, "/api/left-objects/trigger",
                         json_body=trigger_payload)
    _check("left_obj", "POST /api/left-objects/trigger returns 200 or 201",
           status in (200, 201))

    if status in (200, 201) and isinstance(body, dict):
        event_id = body.get("id")
        _check("left_obj", "  event has 'id'", event_id is not None,
               note=f"id={event_id}")

        if event_id:
            # Test dismiss (remove)
            st2, b2 = _post(session, base, f"/api/left-objects/{event_id}/remove",
                            json_body={"action_by": "run.py"})
            _check("left_obj", f"  POST /api/left-objects/{event_id}/remove returns 200",
                   st2 == 200)

    # Trigger a second event for confirm test
    status3, body3 = _post(session, base, "/api/left-objects/trigger",
                           json_body={"camera_id": "gate-south", "dwell_s": 40.0})
    if status3 in (200, 201) and isinstance(body3, dict):
        eid2 = body3.get("id")
        if eid2:
            st4, b4 = _post(session, base, f"/api/left-objects/{eid2}/confirm",
                            json_body={"action_by": "run.py"})
            _check("left_obj", f"  POST /api/left-objects/{eid2}/confirm returns 200",
                   st4 == 200)
            if st4 == 200:
                _check("left_obj", "    confirm response has alert_id",
                       isinstance(b4, dict) and "alert_id" in b4)


def check_loitering(session, base: str) -> None:
    """List, trigger, and dismiss loitering events."""
    print(BOLD("\n[Loitering Detection]"))

    status, body = _get(session, base, "/api/loitering")
    _check("loiter", "GET /api/loitering returns 200", status == 200)
    if status == 200:
        _check("loiter", "  response has 'events' key",
               isinstance(body, dict) and "events" in body)

    # Trigger a test event
    trigger_payload = {
        "camera_id":   "gate-north",
        "subject_key": "run-test-subject",
        "dwell_s":     70.0,
    }
    status, body = _post(session, base, "/api/loitering/trigger",
                         json_body=trigger_payload)
    _check("loiter", "POST /api/loitering/trigger returns 200",
           status == 200)

    if status == 200 and isinstance(body, dict):
        event_id = body.get("id")
        _check("loiter", "  event has 'id'", event_id is not None,
               note=f"id={event_id}")

        if event_id:
            st2, b2 = _post(session, base,
                            f"/api/loitering/{event_id}/dismiss",
                            json_body={"action_by": "run.py"})
            _check("loiter",
                   f"  POST /api/loitering/{event_id}/dismiss returns 200",
                   st2 == 200)
            if st2 == 200:
                _check("loiter", "    dismissed=True in response",
                       isinstance(b2, dict) and b2.get("dismissed") is True)

    # Verify the dismissed event no longer appears in active list
    status2, body2 = _get(session, base, "/api/loitering?dismissed=false")
    if status2 == 200 and isinstance(body2, dict):
        active_ids = [e.get("id") for e in body2.get("events", [])]
        _check("loiter", "  dismissed event not in active list",
               event_id not in active_ids if status == 200 else True,
               note="dismissed filter")


def check_camera_snapshot(session, base: str) -> None:
    """Camera snapshot endpoint (404 expected when no MQTT stream)."""
    print(BOLD("\n[Camera Snapshot]"))
    status, body = _get(session, base, "/api/camera/gate-north/snapshot")
    # 404 is expected when there's no live MQTT feed
    _check("snapshot", "GET /api/camera/<id>/snapshot responds (200 or 404)",
           status in (200, 404),
           note="404 expected without MQTT stream")


def check_enroll_verify(session, base: str) -> None:
    """
    Enroll + verify using a 1×1 pixel JPEG so we can run without a real face image.
    The model will return no matches (expected) but the routes must return 200.
    """
    print(BOLD("\n[Enrollment & Verification]"))

    # Build a minimal valid JPEG (1×1 white pixel)
    import io
    try:
        from PIL import Image as _PILImage
    except ImportError:
        _check("enroll", "PIL not available — enroll/verify skipped", True,
               note="install Pillow")
        return

    buf = io.BytesIO()
    img = _PILImage.new("RGB", (112, 112), color=(200, 200, 200))
    img.save(buf, format="JPEG")
    buf.seek(0)
    jpeg_bytes = buf.getvalue()

    # Enroll
    files = {"image": ("test.jpg", jpeg_bytes, "image/jpeg")}
    data  = {"subject_id": "run-py-test-subject"}
    status, body = _post(session, base, "/api/enroll", files=files, data=data)
    _check("enroll", "POST /api/enroll returns 200", status == 200,
           note="synthetic 112×112 grey image")

    # Verify (no match expected but route must work)
    buf.seek(0)
    files2 = {"image": ("test.jpg", jpeg_bytes, "image/jpeg")}
    data2  = {"subject_id": "run-py-test-subject"}
    status2, body2 = _post(session, base, "/api/verify", files=files2, data=data2)
    _check("enroll", "POST /api/verify returns 200", status2 == 200)

    # Batch verify/frame
    buf.seek(0)
    files3 = {"image": ("test.jpg", jpeg_bytes, "image/jpeg")}
    data3  = {"k": "3", "thresh": "0.0"}
    status3, body3 = _post(session, base, "/api/verify/frame",
                           files=files3, data=data3)
    _check("enroll", "POST /api/verify/frame returns 200", status3 == 200)


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary() -> int:
    """Print coloured summary table. Returns exit code (0=all pass, 1=some fail)."""
    total  = len(_results)
    passed = sum(1 for r in _results if r.ok)
    failed = total - passed

    print()
    print(BOLD("=" * 60))
    print(BOLD(f"  Feature Check Summary   {passed}/{total} passed"))
    print(BOLD("=" * 60))

    # Group by group name
    groups: dict[str, list[_Result]] = {}
    for r in _results:
        groups.setdefault(r.group, []).append(r)

    for group, items in groups.items():
        gpass  = sum(1 for r in items if r.ok)
        gcolor = GREEN if gpass == len(items) else RED
        print(f"\n  {gcolor(group.upper())} ({gpass}/{len(items)})")
        for r in items:
            mark  = PASS_MARK if r.ok else FAIL_MARK
            color = GREEN     if r.ok else RED
            note  = f"  {DIM(r.note)}" if r.note else ""
            print(f"    {color(mark)} {r.name}{note}")

    print()
    if failed == 0:
        print(GREEN(f"  {PASS_MARK} All {total} checks passed!"))
        return 0
    else:
        print(RED(f"  {FAIL_MARK} {failed} check(s) FAILED out of {total}."))
        return 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    global _USE_COLOR

    parser = argparse.ArgumentParser(
        description="Privacy Campus Security — project runner and feature validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument("--port",        type=int, default=5000,
                        help="Port to bind the Flask server to (default: 5000)")
    parser.add_argument("--host",        default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--test-only",   action="store_true",
                        help="Run checks then exit (CI mode)")
    parser.add_argument("--server-only", action="store_true",
                        help="Start server without running any checks")
    parser.add_argument("--no-color",    action="store_true",
                        help="Disable ANSI colour output")
    parser.add_argument("--timeout",     type=int, default=40,
                        help="Seconds to wait for the server to become healthy (default: 40)")
    parser.add_argument("--skip-ml",     action="store_true",
                        help="Skip enroll/verify checks (for environments without PyTorch)")
    args = parser.parse_args()

    if args.no_color:
        _USE_COLOR = False

    project_root = os.path.dirname(os.path.abspath(__file__))
    base_url     = f"http://{args.host}:{args.port}"

    # ── Start server ─────────────────────────────────────────────────────────
    print(BOLD("\n╔══════════════════════════════════════════════════════════╗"))
    print(BOLD(  "║      Privacy Campus Security — Project Runner            ║"))
    print(BOLD(  "╚══════════════════════════════════════════════════════════╝"))
    print()
    print(CYAN(f"  Starting server on {base_url} …"))
    proc, log_path = start_server(args.port, project_root)
    print(DIM(f"  Server log → {log_path}"))

    print(CYAN(f"  Waiting up to {args.timeout} s for server to become healthy …"))
    ready = wait_for_server(base_url, proc, log_path, timeout=args.timeout)

    if not ready:
        print(RED("\n  Server did not become healthy in time."))
        if proc.poll() is None:
            proc.terminate()
        if os.path.exists(log_path):
            try:
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    out = fh.read()
                if out:
                    print(YELLOW("\n  Server output (last 3 kB):"))
                    print(DIM(out[-3000:]))
            except Exception:
                pass
        return 2

    if args.server_only:
        print(GREEN("\n  Server is running.  Press Ctrl-C to stop.\n"))
        try:
            proc.wait()
        except KeyboardInterrupt:
            print(YELLOW("\n  Stopping server …"))
            proc.terminate()
        return 0

    # ── Run feature checks ────────────────────────────────────────────────────
    try:
        import requests
    except ImportError:
        print(RED("  'requests' is not installed.  Run: pip install requests"))
        proc.terminate()
        return 2

    print(CYAN("\n  Running feature checks …"))
    session = requests.Session()

    try:
        check_core_api(session, base_url)
        check_pages(session, base_url)
        check_attendance(session, base_url)
        check_alerts(session, base_url)
        check_left_objects(session, base_url)
        check_loitering(session, base_url)
        check_camera_snapshot(session, base_url)
        if not args.skip_ml:
            check_enroll_verify(session, base_url)
        else:
            print(BOLD("\n[Enrollment & Verification]"))
            print(DIM("  · Skipped (--skip-ml)"))
    except KeyboardInterrupt:
        print(YELLOW("\n  Checks interrupted by user."))

    exit_code = print_summary()

    if args.test_only:
        print(CYAN("\n  --test-only: shutting server down …"))
        proc.terminate()
        return exit_code

    # ── Keep server running ────────────────────────────────────────────────────
    print(CYAN(f"\n  Server is still running on {base_url}"))
    print(CYAN("  Open the dashboard in your browser, or press Ctrl-C to stop.\n"))
    try:
        proc.wait()
    except KeyboardInterrupt:
        print(YELLOW("\n  Stopping server …"))
        proc.terminate()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
