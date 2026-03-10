"""
scripts/load_test_verify_frame.py
──────────────────────────────────────────────────────────────────────────────
Load test: 6-face synthetic frames × N cameras at M fps for T seconds
against POST /api/verify/frame.

Default configuration matches the project evaluation target
  (6 persons, 10 cameras, 4 fps, 30 s → 1 200 requests, CPU < 200 ms).

Usage
─────
# Minimal – hits localhost:5000 with defaults:
    python scripts/load_test_verify_frame.py

# Custom URL / tuning:
    python scripts/load_test_verify_frame.py \\
        --url http://localhost:5000 \\
        --cameras 10 \\
        --fps 4 \\
        --duration 30 \\
        --faces 6 \\
        --out results/load_test.csv

Output
──────
CSV written to <--out> (default: experiments/load_test_YYYYMMDD_HHMMSS.csv)

CSV columns
───────────
  request_id   : monotonic integer across all threads
  camera_id    : simulated camera identifier  (cam-00 … cam-09)
  worker_id    : thread index (0 … N-1)
  status_code  : HTTP response code (0 = connection error / timeout)
  n_faces      : value of response JSON n_faces (-1 on error)
  latency_ms   : end-to-end round-trip time in milliseconds
  error        : empty string on success; error description on failure

Summary printed to stdout
──────────────────────────
  Total requests, success rate, error rate
  Latency: min, p50, p95, p99, max, mean
  Throughput: requests/s
  Single-query latency assessment vs the 200 ms CPU target
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests


# ─── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_URL      = "http://localhost:5000"
_DEFAULT_CAMERAS  = 10
_DEFAULT_FPS      = 4
_DEFAULT_DURATION = 30      # seconds
_DEFAULT_FACES    = 6
_DEFAULT_OUT_DIR  = "experiments"
_REQUEST_TIMEOUT  = 10      # per-request socket timeout (seconds)

# Single-query CPU latency budget from project evaluation targets
_CPU_BUDGET_MS    = 200.0


# ─── Synthetic frame generator ─────────────────────────────────────────────────

def _build_face_jpeg(n_faces: int = 6, width: int = 640, height: int = 480) -> bytes:
    """
    Build a synthetic JPEG containing ``n_faces`` realistic-looking face ovals.

    The image is generated with Pillow (already a project dependency).  Each
    "face" is a skin-toned filled ellipse with a darker outline, placed at
    evenly-spaced grid positions.  Haar/MTCNN detectors may or may not find
    them – but the endpoint processes real JPEG bytes through the full
    face-detection pipeline regardless, exercising actual CPU code paths.

    Falls back to a minimal 1×1 grey JPEG when Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Minimal JFIF fallback (no Pillow) – still exercises the HTTP path
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
            b"\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03"
            b"\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12"
            b"!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1"
            b"\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJ"
            b"STUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xdc\x8a(\x03"
            b"\xff\xd9"
        )

    # Background: muted indoor scene colour
    bg_colour = (80, 80, 85)
    img  = Image.new("RGB", (width, height), bg_colour)
    draw = ImageDraw.Draw(img)

    # Face skin tones (slight variety)
    skin_tones = [
        (240, 200, 160),
        (220, 175, 130),
        (190, 140, 100),
        (255, 220, 185),
        (200, 160, 120),
        (235, 195, 155),
    ]
    outline_colour = (80, 55, 35)

    # Lay faces out in a 3-column grid
    cols   = 3
    rows   = (n_faces + cols - 1) // cols
    cell_w = width  // cols
    cell_h = height // rows

    for i in range(n_faces):
        col = i % cols
        row = i // cols
        cx  = cell_w * col + cell_w // 2
        cy  = cell_h * row + cell_h // 2
        fw  = int(cell_w * 0.45)    # face oval width
        fh  = int(cell_h * 0.55)    # face oval height (taller than wide)

        tone = skin_tones[i % len(skin_tones)]
        draw.ellipse(
            [cx - fw, cy - fh, cx + fw, cy + fh],
            fill=tone,
            outline=outline_colour,
            width=2,
        )

        # Eye sockets (dark circles) — help face detectors with structure cues
        ew = max(fw // 5, 4)
        eh = max(fh // 8, 3)
        for ex in (cx - fw // 3, cx + fw // 3):
            draw.ellipse([ex - ew, cy - fh // 4 - eh,
                          ex + ew, cy - fh // 4 + eh],
                         fill=(50, 35, 25))

        # Mouth hint
        draw.arc(
            [cx - fw // 3, cy + fh // 5, cx + fw // 3, cy + fh // 2],
            start=0, end=180, fill=outline_colour, width=2,
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ─── Worker thread ─────────────────────────────────────────────────────────────

class _Result:
    """One HTTP result row."""
    __slots__ = (
        "request_id", "camera_id", "worker_id",
        "status_code", "n_faces", "latency_ms", "error",
    )

    def __init__(
        self,
        request_id: int,
        camera_id: str,
        worker_id: int,
        status_code: int,
        n_faces: int,
        latency_ms: float,
        error: str,
    ):
        self.request_id  = request_id
        self.camera_id   = camera_id
        self.worker_id   = worker_id
        self.status_code = status_code
        self.n_faces     = n_faces
        self.latency_ms  = latency_ms
        self.error       = error


def _camera_worker(
    worker_id:   int,
    camera_id:   str,
    base_url:    str,
    jpeg_bytes:  bytes,
    fps:         float,
    duration_s:  float,
    results:     List[_Result],
    results_lock: threading.Lock,
    id_counter:  list,           # mutable [int]
    id_lock:     threading.Lock,
    stop_event:  threading.Event,
) -> None:
    """
    Worker thread: sends ``fps`` requests per second until ``stop_event`` is set
    or ``duration_s`` seconds have elapsed.
    Appends one :class:`_Result` per request to ``results`` (thread-safe).
    """
    url          = f"{base_url}/api/verify/frame"
    interval     = 1.0 / fps
    session      = requests.Session()

    deadline     = time.monotonic() + duration_s

    while not stop_event.is_set() and time.monotonic() < deadline:
        loop_start = time.monotonic()

        # Assign monotonic request ID
        with id_lock:
            req_id = id_counter[0]
            id_counter[0] += 1

        status_code = 0
        n_faces     = -1
        error_msg   = ""

        t0 = time.monotonic()
        try:
            resp = session.post(
                url,
                files={"image": (f"{camera_id}_frame.jpg", jpeg_bytes, "image/jpeg")},
                data={"camera_id": camera_id},
                timeout=_REQUEST_TIMEOUT,
            )
            latency_ms  = (time.monotonic() - t0) * 1000.0
            status_code = resp.status_code
            if resp.ok:
                try:
                    body    = resp.json()
                    n_faces = body.get("n_faces", -1)
                except (ValueError, KeyError):
                    error_msg = "invalid JSON response"
            else:
                error_msg = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except requests.exceptions.Timeout:
            latency_ms = _REQUEST_TIMEOUT * 1000.0
            error_msg  = f"timeout after {_REQUEST_TIMEOUT}s"
        except requests.exceptions.ConnectionError as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            error_msg  = f"connection error: {exc}"
        except Exception as exc:                               # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000.0
            error_msg  = f"unexpected: {exc}"

        row = _Result(
            request_id  = req_id,
            camera_id   = camera_id,
            worker_id   = worker_id,
            status_code = status_code,
            n_faces     = n_faces,
            latency_ms  = round(latency_ms, 2),
            error       = error_msg,
        )
        with results_lock:
            results.append(row)

        # Rate-limit: sleep for the remaining portion of our inter-frame window
        elapsed = time.monotonic() - loop_start
        sleep_s = interval - elapsed
        if sleep_s > 0:
            stop_event.wait(timeout=sleep_s)


# ─── Single-query baseline ─────────────────────────────────────────────────────

def _single_query_latency(base_url: str, jpeg_bytes: bytes) -> Optional[float]:
    """
    Fire ONE request to /api/verify/frame and return its latency in ms.
    This isolates server-side ML inference time with zero concurrency.
    Returns None on failure.
    """
    url = f"{base_url}/api/verify/frame"
    try:
        t0   = time.monotonic()
        resp = requests.post(
            url,
            files={"image": ("warmup.jpg", jpeg_bytes, "image/jpeg")},
            timeout=_REQUEST_TIMEOUT,
        )
        lat_ms = (time.monotonic() - t0) * 1000.0
        if resp.ok:
            return lat_ms
        return None
    except Exception:                                          # noqa: BLE001
        return None


# ─── CSV writer ────────────────────────────────────────────────────────────────

_CSV_FIELDS = (
    "request_id", "camera_id", "worker_id",
    "status_code", "n_faces", "latency_ms", "error",
)


def _write_csv(results: List[_Result], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "request_id":  r.request_id,
                "camera_id":   r.camera_id,
                "worker_id":   r.worker_id,
                "status_code": r.status_code,
                "n_faces":     r.n_faces,
                "latency_ms":  r.latency_ms,
                "error":       r.error,
            })


# ─── Summary printer ──────────────────────────────────────────────────────────

def _percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = (len(sorted_vals) - 1) * p / 100
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _print_summary(
    results:         List[_Result],
    duration_s:      float,
    n_cameras:       int,
    fps:             float,
    n_faces:         int,
    out_path:        Path,
    single_lat_ms:   Optional[float],
) -> None:
    total     = len(results)
    successes = [r for r in results if r.status_code == 200]
    errors    = [r for r in results if r.status_code != 200]
    n_ok      = len(successes)
    n_err     = len(errors)

    latencies = sorted(r.latency_ms for r in successes)

    sep = "─" * 72

    print()
    print(sep)
    print("  Privacy Campus Security — Load Test Summary")
    print(sep)
    print(f"  Config   : {n_cameras} camera(s) × {n_faces} face(s)/frame @ {fps} fps × {duration_s}s")
    print(f"  Endpoint : POST /api/verify/frame")
    print()
    print(f"  Requests sent : {total:>6}")
    print(f"  Success (200) : {n_ok:>6}  ({100*n_ok/max(total,1):.1f}%)")
    print(f"  Errors        : {n_err:>6}  ({100*n_err/max(total,1):.1f}%)")
    print()
    if latencies:
        print(f"  Latency (successful requests only):")
        print(f"    min  : {min(latencies):>8.1f} ms")
        print(f"    p50  : {_percentile(latencies, 50):>8.1f} ms")
        print(f"    p95  : {_percentile(latencies, 95):>8.1f} ms")
        print(f"    p99  : {_percentile(latencies, 99):>8.1f} ms")
        print(f"    max  : {max(latencies):>8.1f} ms")
        print(f"    mean : {statistics.mean(latencies):>8.1f} ms")
    else:
        print("  No successful requests — cannot compute latency stats.")
    print()
    if single_lat_ms is not None:
        verdict = "PASS ✓" if single_lat_ms < _CPU_BUDGET_MS else "FAIL ✗  (consider GPU — see README)"
        print(f"  Single-query baseline : {single_lat_ms:.1f} ms  —  CPU < {_CPU_BUDGET_MS:.0f} ms  [{verdict}]")
    else:
        print(f"  Single-query baseline : could not measure (server unreachable?)")
    print()
    throughput = n_ok / duration_s if duration_s > 0 else 0
    print(f"  Throughput : {throughput:.1f} successful req/s  ({n_ok} ok / {duration_s}s)")
    print(f"  CSV output : {out_path.resolve()}")
    print(sep)

    # Error breakdown (top 5)
    if errors:
        from collections import Counter
        top = Counter(r.error for r in errors).most_common(5)
        print()
        print("  Top error messages:")
        for msg, cnt in top:
            print(f"    [{cnt:>4}×] {msg[:80]}")
        print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load test POST /api/verify/frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url",      default=_DEFAULT_URL,
                   help=f"Base URL of the running server. (default: {_DEFAULT_URL})")
    p.add_argument("--cameras",  type=int, default=_DEFAULT_CAMERAS,
                   help=f"Number of concurrent simulated cameras. (default: {_DEFAULT_CAMERAS})")
    p.add_argument("--fps",      type=float, default=_DEFAULT_FPS,
                   help=f"Frames per second per camera. (default: {_DEFAULT_FPS})")
    p.add_argument("--duration", type=float, default=_DEFAULT_DURATION,
                   help=f"Test duration in seconds. (default: {_DEFAULT_DURATION})")
    p.add_argument("--faces",    type=int, default=_DEFAULT_FACES,
                   help=f"Number of synthetic face ovals per frame. (default: {_DEFAULT_FACES})")
    p.add_argument("--out",      default=None,
                   help="Output CSV path. (default: experiments/load_test_YYYYMMDD_HHMMSS.csv)")
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip the single-query warmup / baseline measurement.")
    return p.parse_args(argv)


def run(argv: Optional[list] = None) -> int:
    """Entry point — returns exit code (0 = all ok, 1 = errors present)."""
    args = _parse_args(argv)

    base_url   = args.url.rstrip("/")
    n_cameras  = args.cameras
    fps        = args.fps
    duration_s = args.duration
    n_faces    = args.faces

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(_DEFAULT_OUT_DIR) / f"load_test_{ts}.csv"

    # ── Check server reachability ─────────────────────────────────────────
    print(f"\nConnecting to {base_url}/api/health …", end=" ", flush=True)
    try:
        r = requests.get(f"{base_url}/api/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"ok  (enrolled={info.get('enrolled', '?')}, uptime={info.get('uptime_s', '?')}s)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"FAILED: {exc}")
        print("Start the server first:  python run.py --server-only")
        return 1

    # ── Generate synthetic JPEG ───────────────────────────────────────────
    print(f"Building synthetic {n_faces}-face JPEG frame …", end=" ", flush=True)
    jpeg_bytes = _build_face_jpeg(n_faces=n_faces)
    print(f"{len(jpeg_bytes):,} bytes")

    # ── Single-query warmup / baseline ────────────────────────────────────
    single_lat_ms: Optional[float] = None
    if not args.no_warmup:
        print("Running single-query baseline (CPU) …", end=" ", flush=True)
        single_lat_ms = _single_query_latency(base_url, jpeg_bytes)
        if single_lat_ms is not None:
            verdict = "< 200 ms ✓" if single_lat_ms < _CPU_BUDGET_MS else f"> {_CPU_BUDGET_MS:.0f} ms (consider GPU)"
            print(f"{single_lat_ms:.1f} ms  [{verdict}]")
        else:
            print("failed (will skip in summary)")

    # ── Spin up camera workers ────────────────────────────────────────────
    results:      List[_Result] = []
    results_lock  = threading.Lock()
    id_counter    = [0]
    id_lock       = threading.Lock()
    stop_event    = threading.Event()

    camera_ids    = [f"cam-{i:02d}" for i in range(n_cameras)]
    threads       = []

    print(
        f"\nStarting load test: {n_cameras} cameras × {fps} fps × {duration_s}s "
        f"(≈{int(n_cameras * fps * duration_s)} requests expected)"
    )
    print("  [", end="", flush=True)

    t_start = time.monotonic()

    for i, cid in enumerate(camera_ids):
        t = threading.Thread(
            target=_camera_worker,
            args=(
                i, cid, base_url, jpeg_bytes,
                fps, duration_s, results,
                results_lock, id_counter, id_lock, stop_event,
            ),
            daemon=True,
            name=f"cam-{cid}",
        )
        t.start()
        threads.append(t)

    # Progress indicator: one dot per 10% of duration
    tick = duration_s / 10
    for _ in range(10):
        time.sleep(tick)
        print(".", end="", flush=True)
    print("]")

    # Signal workers to stop and wait
    stop_event.set()
    for t in threads:
        t.join(timeout=_REQUEST_TIMEOUT + 2)

    actual_duration = time.monotonic() - t_start

    # ── Write CSV ─────────────────────────────────────────────────────────
    results_sorted = sorted(results, key=lambda r: r.request_id)
    _write_csv(results_sorted, out_path)

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(
        results        = results_sorted,
        duration_s     = actual_duration,
        n_cameras      = n_cameras,
        fps            = fps,
        n_faces        = n_faces,
        out_path       = out_path,
        single_lat_ms  = single_lat_ms,
    )

    n_errors = sum(1 for r in results_sorted if r.status_code != 200)
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
