# Privacy-Preserving Campus Security

> Research prototype — software-only, modular, reproducible.  
> Authors: Abhishek Mittal & Smriti Aggarwal · 2026

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-compose-blue?logo=docker)](docker-compose.yml)

A **privacy-aware campus security platform** built on few-shot face recognition, a privacy-preserving embedding store (no raw images, no plaintext IDs ever stored on disk), and a real-time Flask dashboard with live webcam enroll/verify, MQTT camera simulation, alert escalation, and loitering/left-object detection.

Full system design, threat model, and data-flow diagrams: [docs/design.md](docs/design.md)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  PRIVACY CAMPUS SECURITY — SYSTEM                       │
└─────────────────────────────────────────────────────────────────────────┘

  ┌────────────┐  JPEG frames         ┌──────────────────────────────────┐
  │ IP Camera  │ ──────────────────►  │  Eclipse Mosquitto MQTT Broker   │
  │  (or sim)  │  campus/<id>/frame   │  Port 1883 (TCP) / 9001 (WS)     │
  └────────────┘                      └───────────────┬──────────────────┘
                                                      │ subscribe
  ┌────────────┐  door/sensor events                  ▼
  │ Door/Smoke │ ──────────────────►  ┌──────────────────────────────────┐
  │   Sensor   │  campus/<id>/sensor  │      Flask + SocketIO server     │
  └────────────┘                      │      src/web/flask_app.py        │
                                      │                                  │
  ┌────────────┐  Browser WebSocket   │  ┌──────────────────────────┐    │
  │  Dashboard │ ◄────────────────►   │  │ BackboneEmbedder         │    │
  │  Browser   │  Socket.IO events    │  │ ResNet-50 + 512-D L2norm │    │
  └────────────┘                      │  └───────────┬──────────────┘    │
                                      │              │ 512-D vectors      │
  ┌────────────┐  REST API            │  ┌───────────▼──────────────┐    │
  │ curl/Mobile│ ◄────────────────►   │  │ EmbeddingStore (SQLite)  │    │
  └────────────┘  JSON responses      │  │ HMAC keys · projection   │    │
                                      │  │ optional Fernet encrypt  │    │
  ┌────────────┐  Email / SMS         │  └──────────────────────────┘    │
  │  Your      │ ◄──────────────────  │                                  │
  │  Device    │  alert escalation    │  ┌──────────────────────────┐    │
  └────────────┘                      │  │ AccessControlEngine      │    │
                                      │  │ forced-entry · tailgate  │    │
                                      │  │ safety_alert · loitering │    │
                                      │  └──────────────────────────┘    │
                                      │                                  │
                                      │  ┌──────────────────────────┐    │
                                      │  │ NotificationDispatcher   │    │
                                      │  │ WebSocket·SMTP·SMS       │    │
                                      │  │ throttle · retry · log   │    │
                                      │  └──────────────────────────┘    │
                                      │                                  │
                                      │  SQLite files in  data/          │
                                      │  embeddings.db  attendance.db    │
                                      │  alerts.db      notifications.db │
                                      │  audit.db                        │
                                      └──────────────────────────────────┘

Data flow:  frame → face detect → embed → store.query_topk()
                ↓ match?                         ↓
        attendance.mark()           access_control.evaluate()
                                              ↓
                               alert → WebSocket (dashboard)
                                     → SMTP     (email)
                                     → SMS      (phone)
```

### Component map

| Component | Source file | Role |
|---|---|---|
| Web server | `src/web/flask_app.py` | 26 REST routes + 14 Socket.IO events |
| Face backbone | `src/models/backbone.py` | ResNet-50, 512-D L2-norm output |
| ProtoNet | `src/models/prototypical.py` | N-way K-shot few-shot training |
| Embedding store | `src/privacy/embedding_store.py` | HMAC keys, orthogonal projection, SQLite |
| Attendance | `src/privacy/attendance_store.py` | Session-TTL dedup, aggregated queries |
| Access control | `src/privacy/access_control.py` | Forced-entry, tailgate, safety-alert rules |
| Left-object detect | `src/privacy/left_object_detector.py` | Dual-EMA background subtraction + IoU tracker |
| Loitering detect | `src/privacy/loitering_detector.py` | Per-camera dwell timer with cooldown |
| Notifications | `src/notifications/dispatcher.py` | Policy + WebSocket + SMTP + SMS + retry |
| Auth / RBAC | `src/web/auth.py` | bcrypt passwords, JWT tokens, role decorators |
| Audit log | `src/privacy/audit_store.py` | Append-only SQLite with immutability triggers |
| CLI | `src/cli/run.py` | 11 subcommands for ML pipeline and simulation |

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Starting the App & Verifying It Works](#starting-the-app--verifying-it-works)
3. [Where to See Your Data](#where-to-see-your-data)
4. [Getting Alerts & Notifications on Your Device](#getting-alerts--notifications-on-your-device)
5. [How to Simulate Cameras](#how-to-simulate-cameras-without-real-hardware)
6. [Running Tests — How Code Correctness Is Ensured](#running-tests)
7. [Security Notes](#security-notes)
8. [Evaluation Metrics Targets](#evaluation-metrics-targets)
9. [REST API Reference](#rest-api)
10. [CLI Subcommands](#cli-subcommands)
11. [Docker](#docker)
12. [Load Testing](#load-testing)
13. [Environment Variables](#environment-variables-reference)
14. [Project Structure](#project-structure)

---

## Quickstart

### 1 — Create a virtual environment (Python 3.11 required)

```powershell
git clone https://github.com/your-org/privacy-campus-security.git
cd privacy-campus-security

py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate          # Linux / macOS
```

### 2 — Install dependencies

```powershell
pip install -e .                     # runtime (torch, flask, …)
pip install -r requirements-dev.txt  # dev & test tools
```

### 3 — Configure environment (`.env`)

Copy the provided template and fill in **only the essentials** — the five starred lines are enough to get email alerts and the Panic SMS working:

```powershell
Copy-Item .env .env.local   # keep a backup of defaults
# then edit .env:
```

| Variable | What to fill in |
|---|---|
| `PCS_EMBEDDING_SALT` | Any long random string — run `python -c "import secrets; print(secrets.token_hex(32))"`  |
| `PCS_JWT_SECRET` | Long random string used to sign JWTs — generate the same way as the salt above |
| `PCS_SMTP_HOST` | Your SMTP server (e.g. `smtp.gmail.com`) — enables email alerts |
| `PCS_SMTP_USER` | Your email login |
| `PCS_SMTP_PASSWORD` | Your email password / app-password |
| `PCS_SMTP_TO` | Comma-separated recipients for alert emails |
| `PCS_PANIC_SMS_TO` | Comma-separated phone numbers (E.164) for the **Panic** button |

All other variables have sensible defaults and can be tuned later.  Load them before starting the server:

```powershell
# PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=]+)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim())
    }
}
```

### 3.5 — Create your `.env` configuration file

Create a file named `.env` in the project root (the two security keys are the **minimum required**):

```ini
# .env — paste into project root, fill in values

# ★ REQUIRED — generate with: python -c "import secrets; print(secrets.token_hex(32))"
PCS_EMBEDDING_SALT=replace-with-a-random-64-char-hex-string
PCS_JWT_SECRET=replace-with-another-random-64-char-hex-string

# ★ Email alerts (fill in your Gmail / SMTP details to receive real alerts)
PCS_SMTP_HOST=smtp.gmail.com
PCS_SMTP_PORT=587
PCS_SMTP_USER=your.email@gmail.com
PCS_SMTP_PASSWORD=your-16-char-app-password
PCS_SMTP_TO=you@gmail.com

# ★ SMS panic button (E.164 format, e.g. +919876543210)
PCS_PANIC_SMS_TO=+919876543210

# Optional — defaults work fine for local development
PCS_STORE=data/embeddings.db
PCS_ATTENDANCE_DB=data/attendance.db
PCS_DEVICE=cpu
PCS_ENCRYPT=0
PCS_LOG_LEVEL=INFO
```

Load `.env` into your shell before every server start:

```powershell
# PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^#=\s][^=]*)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim())
    }
}
```

```bash
# Linux / macOS
export $(grep -v '^#' .env | xargs)
```

> **Gmail app password:** Google Account → Security → 2-Step Verification → App passwords.  
> Create a new "Mail" app password (16 chars). Use that — NOT your real password.

### 4 — Run the complete project

```powershell
python run.py
```

This single command:
1. Starts the Flask + SocketIO server on `http://localhost:5000`
2. Waits for the server to become healthy (up to 40 s)
3. Automatically smoke-tests **every feature** (core API, attendance, alerts, left-object detection, loitering detection, enrollment, verification, all pages)
4. Prints a colour-coded pass/fail summary
5. Keeps the server running so you can open the dashboard in your browser

```
╔══════════════════════════════════════════════════════════╗
║      Privacy Campus Security — Project Runner            ║
╚══════════════════════════════════════════════════════════╝

  Starting server on http://127.0.0.1:5000 …
  Waiting up to 40 s for server to become healthy …
  Server log → /path/to/server.log
  Server ready (uptime 2.8 s, enrolled=0)

  Running feature checks …

[Core API]
  ✓ GET /api/health returns 200
  ✓   health.status == 'ok'
  ✓ GET /api/events returns 200
  …
[Loitering Detection]
  ✓ GET /api/loitering returns 200
  ✓ POST /api/loitering/trigger returns 200
  ✓   POST /api/loitering/1/dismiss returns 200
  …
════════════════════════════════════════════════
  ✓ All 38 checks passed!
```

#### `run.py` options

| Flag | Effect |
|---|---|
| `python run.py` | Start server → run all checks → keep server alive |
| `python run.py --test-only` | Run checks then exit (CI / automated mode, exit 0 = all pass) |
| `python run.py --server-only` | Start server without running any checks |
| `python run.py --skip-ml` | Skip enroll/verify checks (no GPU/PyTorch needed) |
| `python run.py --port 5001` | Bind to a different port |
| `python run.py --timeout 60` | Allow up to 60 s for the server to start |
| `python run.py --no-color` | Plain text output (for logs/CI) |

> **Note:** `run.py` launches `src/web/flask_app.py` via SocketIO's own server wrapper,
> **not** `flask run`. This is required for WebSocket transport (webcam, live alerts) to work.

---

## Docker

Docker is the quickest way to get a fully isolated, reproducible runtime with no local Python setup required.

### Web app only

```powershell
docker compose up
# Dashboard available at http://localhost:5000
```

The `./data/` directory is bind-mounted into the container at `/app/data`, so the SQLite embedding store persists between restarts.

### Web app + Mosquitto MQTT broker

```powershell
docker compose --profile mqtt up
```

This additionally starts an `eclipse-mosquitto:2.0` broker:

| Port | Protocol | Use |
|---|---|---|
| `1883` | Plain MQTT | IoT sensors / CLI publishers |
| `9001` | MQTT-over-WebSocket | Browser MQTT clients |

Mosquitto configuration lives in [mosquitto/mosquitto.conf](mosquitto/mosquitto.conf). Message data and logs are stored in named Docker volumes (`mosquitto-data`, `mosquitto-log`).

### Environment variables (Docker)

Pass any of these via `docker compose run -e VAR=value web …` or edit the `environment:` block in [docker-compose.yml](docker-compose.yml):

| Variable | Default in container | Description |
|---|---|---|
| `PCS_STORE` | `/app/data/embeddings.json` | Path to SQLite store inside the container |
| `PCS_EMBEDDING_SALT` | `changeme-replace-in-production` | **Change before deploying** |
| `PCS_ENCRYPT` | `0` | `1` → enable Fernet at-rest encryption |
| `PCS_DEVICE` | `cpu` | `cuda` for GPU containers |
| `FLASK_DEBUG` | `0` | `1` → enable auto-reload |

### Build the image manually

```powershell
docker build -t privacy-campus-security:latest .
docker run --rm -p 5000:5000 -v "${PWD}/data:/app/data" privacy-campus-security:latest
```

---

## Project Structure

```
privacy-campus-security/
├─ run.py                      ← ONE-COMMAND project launcher & feature validator
├─ configs/
│   ├─ proto.yaml              ← ProtoNet training hyperparameters
│   └─ cameras.yaml            ← camera registry (id, name, door, GPS coords, MQTT config)
├─ docs/design.md              ← full system design & data-flow diagrams
├─ samples/
│   ├─ gate_frames/            ← 5 sample JPEG frames for simulate-camera
│   └─ gen_gate_frames.py      ← script that generated the sample frames
├─ mosquitto/mosquitto.conf    ← Eclipse Mosquitto broker config
├─ experiments/sample_runs/    ← metrics JSON, checkpoints, plots (git-ignored)
├─ scripts/
│   └─ load_test_verify_frame.py  ← threaded load test: 6-face frames × N cameras → latency CSV
├─ src/
│   ├─ cli/run.py              ← 11 CLI subcommands (pipeline + simulate-camera + simulate-door)
│   ├─ data/lfw_loader.py      ← LFW dataset wrapper + download flag
│   ├─ eval/evaluate_lfw.py    ← LFW pairs + few-shot evaluation
│   ├─ models/
│   │   ├─ backbone.py         ← ResNet-50 + 512-D L2-normalised head
│   │   ├─ prototypical.py     ← N-way K-shot ProtoNet trainer
│   │   └─ embedding_utils.py  ← batch embedding cache helper
│   ├─ privacy/
│   │   ├─ embedding_store.py     ← HMAC-SHA256 keys + orthogonal projection + SQLite
│   │   ├─ attendance_store.py    ← attendance table + session TTL dedup + aggregated queries
│   │   ├─ camera_store.py        ← CameraRegistry (YAML) + CameraStore (SQLite)
│   │   ├─ access_control.py      ← AlertStore (SQLite) + AccessControlEngine (forced-entry + tailgate + safety alert)
│   │   ├─ audit_store.py         ← AuditStore – append-only SQLite audit log with immutability triggers
│   │   ├─ left_object_detector.py ← dual-EMA background subtraction + IoU blob tracker
│   │   ├─ left_object_store.py    ← SQLite store for left-object events
│   │   ├─ loitering_detector.py   ← per-camera dwell tracker with cooldown
│   │   └─ loitering_store.py      ← SQLite store for loitering events
│   ├─ notifications/
│   │   ├─ __init__.py             ← public exports
│   │   ├─ store.py               ← NotificationStore – SQLite delivery log (status, retries)
│   │   ├─ policy.py              ← EscalationPolicy – severity → channels mapping
│   │   ├─ drivers.py             ← WebSocketDriver, SmtpDriver, NullSmsDriver, SmsDriverBase
│   │   └─ dispatcher.py          ← NotificationDispatcher – orchestrates policy + drivers + retry + throttle
│   ├─ logging_config.py          ← structlog JSON logging setup (PCS_LOG_LEVEL / PCS_LOG_FORMAT)
│   └─ web/
│       ├─ flask_app.py        ← Flask + SocketIO server (26 REST endpoints + 14 WS events)
│       ├─ auth.py             ← RBAC layer: UserStore (bcrypt), JWT helpers, decorators, auth Blueprint
│       ├─ face_utils.py       ← multi-face detection + batch embedding
│       ├─ data_provider.py    ← metrics cache + event ring buffer
│       ├─ metrics.py          ← Prometheus counters/histograms + /metrics response helper
│       └─ templates/
│           ├─ login.html         ← Login page (dark theme, Bootstrap 5)
│           ├─ dashboard.html     ← Bootstrap 5, Chart.js 4, live alerts, left-object & loitering panels
│           ├─ enroll_screen.html ← single-screen camera grid enrollment UI
│           ├─ campus_map.html    ← Leaflet.js campus overview with live camera status
│           └─ mobile_alerts.html ← mobile on-call admin view (Ack + Navigate + Panic)
└─ tests/                      ← pytest test suite (483 unit + ~30 integration tests)
    ├─ integration/            ← end-to-end integration tests (requires Mosquitto)
    │   ├─ conftest.py         ← broker, live-server, mqtt-pub, fake-jpeg fixtures
    │   └─ test_end_to_end.py  ← ~30 tests across 4 groups (A: API, B: CRUD, C: MQTT, D: ML)
    └─ test_*.py               ← unit test modules
```

> **`src/cli/run.py`** exposes **11 CLI subcommands** (`prepare-data`, `compute-embeddings`, `train-protonet`, `eval-lfw`, `enroll`, `enroll-batch`, `verify`, `eval`, `serve`, `simulate-camera`, `simulate-door`).
> **`run.py`** (project root) is the **one-command launcher** — starts the server *and* validates all features automatically.

---

## Attendance API

The attendance system tracks when enrolled subjects are verified at a camera. It applies a **session TTL** (default 5 min, configurable via `PCS_SESSION_TTL`) to prevent duplicate marks from continuous video streams.

### Mark attendance (single or batch)

```powershell
# Single record
curl -s -X POST http://localhost:5000/api/attendance/mark \
  -H 'Content-Type: application/json' \
  -d '{"subject_key":"grp:SHIFT-A:f0:abc123","similarity":0.87,"matched":true,
       "group_id":"SHIFT-A","camera_id":"entrance-cam"}'

# Batch: 6 verify results in one call
curl -s -X POST http://localhost:5000/api/attendance/mark \
  -H 'Content-Type: application/json' \
  -d '[{"subject_key":"s1","similarity":0.9,"matched":true,"group_id":"SHIFT-A"},
       {"subject_key":"s2","similarity":0.4,"matched":false,"group_id":"SHIFT-A"},
       ...]'
```

### Query attendance

```powershell
# All attendance, aggregated
curl http://localhost:5000/api/attendance

# Filter by date range and group
curl 'http://localhost:5000/api/attendance?from=2026-02-28T00:00:00Z&to=2026-02-28T23:59:59Z&group_id=SHIFT-A'
```

Response shape:

```json
{
  "total_marks":   6,
  "matched_marks": 5,
  "session_ttl_s": 300,
  "filters":       {"from": "...", "to": "...", "group_id": "..."},
  "per_subject":   [{"subject_key": "...", "total": 2, "matched": 2, "last_seen": "..."}],
  "per_camera":    [{"camera_id": "entrance-cam", "total": 6, "matched": 5, "unique_subjects": 5}]
}
```

### Environment variable

| Variable | Default | Description |
|---|---|---|
| `PCS_ATTENDANCE_DB` | `data/attendance.db` | SQLite file for the attendance log |
| `PCS_SESSION_TTL` | `300` | Session TTL in seconds for dedup |

---

## CLI Subcommands

```powershell
# Download LFW dataset (~200 MB)
python -m src.data.lfw_loader --download --root data/lfw

# Compute and cache backbone embeddings
python -m src.cli.run compute-embeddings --root data/lfw --out experiments/embeddings.pt

# Train ProtoNet (uses configs/proto.yaml by default)
python -m src.cli.run train-protonet --root data/lfw

# Evaluate on LFW
python -m src.cli.run eval-lfw --root data/lfw --mode both

# Enroll / verify via CLI
python -m src.cli.run enroll --subject-id EMP-001 --image face.jpg --store data/store.db
python -m src.cli.run verify --subject-id EMP-001 --image query.jpg --store data/store.db

# Batch-enroll every face detected in a photo (multi-person image)
# Returns a JSON list of opaque, salted subject keys — no plaintext IDs are saved.
python -m src.cli.run enroll-batch --image samples/multi_face.jpg \
    --group-id SHIFT-A --store data/embeddings.db
```

### `enroll-batch` — privacy model

For each detected face, a unique ephemeral key is generated:

```
grp:<group_id>:f<face_idx>:<secrets.token_hex(16)>
```

Only `SHA-256(salt ‖ key)` is written to the SQLite store — the original key
is never persisted.  The caller receives the list of plaintext keys in stdout
(JSON) and must retain them to perform future `verify` calls.

```powershell
# Returned output example
[
  "grp:SHIFT-A:f0:3f8a1c4e...",
  "grp:SHIFT-A:f1:b2d7e90c..."
]
```

---

## Web Pages

| Route | Description |
|---|---|
| `GET /` | Main dashboard — live stats, **Camera Monitor thumbnail grid**, attendance table, event log |
| `GET /enroll/screen` | Single-screen enrollment — camera grid + webcam capture panel |
| `GET /admin/campus` | Campus overview map — Leaflet.js with live camera-status icons |
| `GET /alerts/mobile` | **Mobile on-call admin view** — stripped-down alert feed for small screens with one-tap Acknowledge, Navigate-to-camera, and Panic button |

---

## REST API

### Core endpoints

| Method | Endpoint | Body | Response |
|---|---|---|---|
| GET | `/api/health` | — | `{status, enrolled, uptime_s}` |
| GET | `/api/events?n=50` | — | Last N events (newest-first) |
| GET | `/api/metrics` | — | K-shot table, pairs metrics, anomaly series |
| GET | `/metrics` | — | **Prometheus scrape endpoint** (counters + histogram) |
| POST | `/api/enroll` | form: `subject_id`, `image` | `{status, total_enrolled}` |
| POST | `/api/enroll/batch` | form: `image`, `group_id` (opt.) | `{n_enrolled, keys, total_enrolled}` |
| POST | `/api/verify` | form: `subject_id`, `image` | `{similarity, match}` |
| POST | `/api/verify/frame` | form: `image`, `k` (opt.), `thresh` (opt.) | `{n_faces, faces[{face_id,bbox,top_matches}], latency_ms}` |
| POST | `/api/enroll/frame` | JSON: `{subject_id, image_b64}` | `{status, total_enrolled}` |
| POST | `/api/verify/live` | JSON: `{subject_id, image_b64}` | `{similarity, match, latency_ms}` |
| POST | `/api/attendance/mark` | JSON: record or `[records]` each with `subject_key, similarity, matched` | `{n_marked, n_skipped, total_attendance, session_ttl_s}` |
| GET | `/api/attendance` | query: `from`, `to`, `group_id` | `{total_marks, matched_marks, per_subject[], per_camera[]}` |

### Camera endpoints

| Method | Endpoint | Response |
|---|---|---|
| GET | `/api/cameras` | JSON array of all cameras with `frame_topic`, `sensor_topic` |
| GET | `/api/camera/status` | JSON dict `{camera_id: {status, last_event, last_seen, last_similarity, last_matched, recent_events[]}}` |
| GET | `/api/camera/<id>/snapshot` | Latest JPEG frame from MQTT feed (used by Camera Monitor thumbnails), or 404 |

`status` values: `"idle"` (green), `"activity"` (amber), `"alert"` (red)

`recent_events` is a list of up to 3 recent per-camera events in the form:
```json
[
  {"ts": "2026-03-01T12:00:01Z", "type": "verify", "matched": true,  "similarity": 0.85},
  {"ts": "2026-03-01T12:00:05Z", "type": "alert",  "alert_type": "forced_entry"}
]
```

### Alert endpoints

| Method | Endpoint | Params / Body | Response |
|---|---|---|---|
| GET | `/api/alerts` | `n`, `alert_type`, `door_id`, `unacked_only=1` | `{alerts[], total_unacked}` |
| POST | `/api/alerts/<id>/ack` | JSON body `{acked_by?}` | `{acknowledged, id, acked_by, acked_at}` or 404 |
| POST | `/api/alerts/panic` | JSON body `{triggered_by?, message?, camera_id?}` | `{ok, sms_count, triggered_by, timestamp}` |
| GET | `/api/audit` | query: `limit`, `event_type`, `actor`, `camera_id`, `alert_id`, `since` | `{audit[], total}` (admin only) |
| GET | `/api/camera/<id>/snapshot` | — | Latest JPEG frame from MQTT feed, or 404 |

**Alert types:** `forced_entry`, `tailgate`, `safety_alert`

#### Panic escalation

The **Panic** endpoint sends an **immediate, throttle-bypassing** SMS to every number in `PCS_PANIC_SMS_TO` (falls back to email via `PCS_SMTP_TO` when no SMS driver is configured).  It also broadcasts a `panic` Socket.IO event to all connected dashboard clients so the desktop dashboard lights up.

```powershell
# Trigger from curl (simulates the mobile Panic button):
curl -X POST http://localhost:5000/api/alerts/panic `
     -H "Content-Type: application/json" `
     -d '{"triggered_by":"Officer Smith","message":"Intruder at north gate"}'
```

Each alert object:
```json
{
  "id": 1,
  "alert_type": "forced_entry",
  "camera_id": "gate-north",
  "door_id": "door-01",
  "subject_key": "grp:A:f0:abc…",
  "confidence": 0.90,
  "timestamp_utc": "2026-02-28T10:00:01+00:00",
  "meta": {"trigger": "failed_verify_with_door_open", "time_diff_s": 0.5},
  "acknowledged": true,
  "acked_by": "Officer Smith",
  "acked_at": "2026-02-28T10:05:00+00:00"
}
```

For `safety_alert`, the `meta` payload includes `{"severity":"high", "detector":"mqtt_sensor"|"image_heuristic", ...}`.

### Smoke / Fire simulation

The app supports two lightweight smoke/fire trigger paths:

1) **Sensor simulator (MQTT)**

```powershell
mosquitto_pub -h localhost -p 1883 -t campus/gate-north/sensor -m '{"smoke": true}'
```

2) **Image heuristic (automatic on frame stream)**

- Runs on incoming `campus/<camera_id>/frame` JPEG messages
- Combines reddish/orange blob ratio + dense optical flow + persistence heatmap
- Emits `safety_alert` when confidence is sustained across consecutive frames

Expected result after publishing sensor telemetry:

```powershell
curl http://localhost:5000/api/alerts?alert_type=safety_alert&n=5
```

You should see at least one `safety_alert` with `meta.severity = "high"`.

---

### Left Object Detection

The app includes a per-camera background-subtraction pipeline that raises `left_object` events when
a new static object has been present for longer than a configurable dwell period.

#### Algorithm

| Component | Detail |
|---|---|
| **Slow EMA background** (`alpha=0.002`) | ~500-frame memory; objects appear as foreground vs this model |
| **Fast EMA background** (`alpha=0.05`) | ~20-frame memory; currently-moving pixels appear as foreground |
| **Left-object mask** | `fg_slow AND NOT fg_fast` — object is visible vs long-term bg but no longer moving |
| Morphological clean-up | Open + dilate (cv2 if installed, numpy fallback otherwise) |
| Blob tracking | IoU ≥ 0.25 across consecutive frames; per-candidate `first_seen` timestamp |
| Detection trigger | Blob dwell ≥ `PCS_LEFT_OBJ_DWELL_S` seconds — fired once per cooldown window |

#### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/left-objects` | List events (filters: `n`, `camera_id`, `dismissed`, `confirmed`) — snapshot omitted in list view |
| `GET` | `/api/left-objects/<id>` | Full event including `snapshot_b64` |
| `POST` | `/api/left-objects/<id>/remove` | Dismiss event (`{action_by?}`) |
| `POST` | `/api/left-objects/<id>/confirm` | Confirm as security threat (`{action_by?}`) — also creates a `safety_alert` |
| `POST` | `/api/left-objects/trigger` | Inject a synthetic test event (`{camera_id, bbox?, dwell_s?}`) |

#### Socket.IO Events

| Event | Direction | Payload |
|---|---|---|
| `left_object` | Server → Client | Full detection dict: `{id, camera_id, bbox, first_seen_utc, detected_utc, dwell_s, snapshot_b64, dismissed, confirmed}` |
| `left_object_update` | Server → Client | Same shape, emitted after remove / confirm action |

#### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PCS_LEFT_OBJ_DWELL_S` | `30` | Seconds an object must remain static before a `left_object` event is created |
| `PCS_LEFT_OBJ_FG_THRESH` | `22` | Pixel-difference threshold (0–255) for foreground mask |
| `PCS_LEFT_OBJ_MIN_BLOB` | `0.008` | Minimum blob area as a fraction of the processing frame |
| `PCS_LEFT_OBJ_STALE_S` | `8` | Seconds before an unseen candidate is discarded |
| `PCS_LEFT_OBJ_COOLDOWN_S` | `60` | Minimum seconds between repeat events for the same blob |

#### Quick simulation

```powershell
# Inject a synthetic left-object event for a camera:
curl -X POST http://localhost:5000/api/left-objects/trigger `
     -H "Content-Type: application/json" `
     -d '{"camera_id":"gate-north","dwell_s":45}'
```

Or click the **Simulate** button in the *Left Object Detection* panel on the dashboard.

---

### Loitering Detection

The app tracks face-recognition matches per camera across MQTT frame streams.  When the same
`subject_key` is continuously present in a camera's field of view for longer than a configurable
dwell period, a `loitering` event is created and broadcast.

#### How it works

| Step | Detail |
|---|---|
| **Frame throttle** | Face recognition runs at most once every `PCS_LOITER_FRAME_INTERVAL_S` s per camera (default 2 s) to keep CPU usage low |
| **Face identify** | Each MQTT frame → `generate_embedding_from_image` → `store.query_topk_batch(k=1, threshold=0.35)` |
| **Dwell tracking** | Per-camera, per-`subject_key` timer: `last_seen − first_seen ≥ dwell_s` → fire event |
| **Stale purge** | Subjects absent for `> stale_s` seconds have their dwell timer reset (they "left") |
| **Cooldown** | After firing, the same subject cannot trigger again for `cooldown_s` seconds |

#### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/loitering` | List events (filters: `n`, `camera_id`, `subject_key`, `dismissed`) |
| `POST` | `/api/loitering/<id>/dismiss` | Dismiss event (`{action_by?}`) |
| `POST` | `/api/loitering/trigger` | Inject a synthetic test event (`{camera_id, subject_key?, dwell_s?}`) |
---

## Notification System

All alert and event types are routed through a **notification dispatcher** that applies an escalation policy and persists every delivery attempt to the `notifications` SQLite table.

### Escalation policy

| Severity | Alert types | Channels |
|---|---|---|
| **high** | `safety_alert`, `forced_entry`, `tailgate` | WebSocket + SMTP + SMS* |
| **medium** | `loitering`, `left_object` | WebSocket + SMTP |
| **low** | _anything else_ | WebSocket only |

\* SMS is opt-in: set `PCS_SMS_OPT_IN=1` to enable.  No vendor credentials are baked in — bring your own adapter (see [Pluggable SMS adapter](#pluggable-sms-adapter)).

### Drivers

| Driver | Description |
|---|---|
| `WebSocketDriver` | Emits Socket.IO event to all connected clients (`"alert"`, `"loitering"`, or `"left_object"` depending on type). Always active. |
| `SmtpDriver` | Sends a plain-text email via `smtplib`. Enabled when `PCS_SMTP_HOST` is set. |
| `NullSmsDriver` | Default no-op SMS adapter — logs the would-be SMS at DEBUG level without calling any vendor. |

### Per-alert retry & throttling

| Feature | Detail |
|---|---|
| **Throttling** | At most one notification per `(channel, alert_type)` pair within `PCS_NOTIFY_THROTTLE_S` seconds (default 60 s). Throttled records are persisted with `status="throttled"`. |
| **Retry** | On delivery failure: exponential back-off `delay = min(2^attempt × 5, 300)` s. A daemon thread retries every `PCS_NOTIFY_RETRY_S` seconds (default 10 s). |
| **Max retries** | Configurable via `PCS_NOTIFY_MAX_RETRIES` (default 3). Records that exhaust retries are marked `status="failed"`. |
| **Async delivery** | WebSocket is dispatched synchronously for minimal latency. SMTP and SMS run in background threads so Flask/MQTT threads are not blocked. |

### Notifications REST endpoint

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/notifications` | List delivery records (filters: `n`, `channel`, `status`, `alert_id`, `alert_type`) |

Response shape:

```json
{
  "notifications": [
    {
      "id":          1,
      "alert_id":    3,
      "alert_type":  "safety_alert",
      "severity":    "high",
      "channel":     "websocket",
      "status":      "sent",
      "attempts":    1,
      "last_attempt": "2026-03-01T10:00:01+00:00",
      "next_retry":  null,
      "error_msg":   null,
      "created_at":  "2026-03-01T10:00:00+00:00"
    }
  ],
  "total_sent":      12,
  "total_failed":     1,
  "total_throttled":  3
}
```

### Pluggable SMS adapter

Subclass `SmsDriverBase` from `src.notifications.drivers`, implement `send()`, and point `PCS_SMS_DRIVER` to its fully-qualified class path:

```python
# my_sms.py
from src.notifications.drivers import SmsDriverBase

class TwilioSmsDriver(SmsDriverBase):
    name = "sms"

    def __init__(self):
        from twilio.rest import Client
        self._client = Client(
            os.environ["TWILIO_ACCOUNT_SID"],
            os.environ["TWILIO_AUTH_TOKEN"],
        )

    def send(self, notification):
        payload = notification.get("payload") or {}
        self._client.messages.create(
            body=f"[PCS ALERT] {payload.get('alert_type')} camera={payload.get('camera_id')}",
            from_=os.environ["TWILIO_FROM"],
            to=os.environ["TWILIO_TO"],
        )
```

```powershell
$env:PCS_SMS_DRIVER = "my_sms.TwilioSmsDriver"
$env:PCS_SMS_OPT_IN = "1"
python run.py
```

### Trigger high-severity notification (quick test)

```powershell
# Inject a high-severity safety event (no MQTT needed):
curl -X POST http://localhost:5000/api/left-objects/trigger `
     -H "Content-Type: application/json" `
     -d '{"camera_id":"gate-north","dwell_s":45}'

# Then confirm it as threat (severity=high → triggers WebSocket + SMTP)
curl -X POST http://localhost:5000/api/left-objects/1/confirm `
     -H "Content-Type: application/json" `
     -d '{"action_by":"Officer Smith"}'

# Inspect notification delivery log
curl http://localhost:5000/api/notifications?n=10
```

Expected (with `PCS_SMTP_HOST` set):

- `alert_type="safety_alert"` → one `status="sent"` row for `channel="websocket"` and one for `channel="smtp"`
- No SMS row unless `PCS_SMS_OPT_IN=1` and a custom driver is configured
#### Socket.IO Events

| Event | Direction | Payload |
|---|---|---|
| `loitering` | Server → Client | `{id, camera_id, subject_key, first_seen_utc, detected_utc, dwell_s, dismissed}` |
| `loitering_update` | Server → Client | Same shape, emitted after dismiss action |

#### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PCS_LOITER_DWELL_S` | `60` | Seconds a subject must be continuously present to trigger a loitering alert |
| `PCS_LOITER_STALE_S` | `10` | Seconds of absence before a subject's dwell timer resets |
| `PCS_LOITER_COOLDOWN_S` | `120` | Minimum seconds between repeat events for the same subject/camera |
| `PCS_LOITER_FRAME_INTERVAL_S` | `2.0` | Minimum seconds between face-recognition passes on MQTT frames (throttle) |

#### Quick simulation

```powershell
# Inject a synthetic loitering event (no live camera needed):
curl -X POST http://localhost:5000/api/loitering/trigger `
     -H "Content-Type: application/json" `
     -d '{"camera_id":"gate-north","subject_key":"grp:A:f0:abc","dwell_s":65}'
```

Or click the **Simulate** button in the *Loitering Detection* panel on the dashboard.

---

## Starting the App & Verifying It Works

### Launch modes

| Command | What happens |
|---|---|
| `python run.py` | Server **+** 38 smoke checks **+** keeps server alive |
| `python run.py --server-only` | Server only, no checks |
| `python run.py --test-only` | Checks then exit (CI mode — exit 0 = all pass) |
| `python run.py --skip-ml` | Skip face enroll/verify checks (no GPU required) |
| `python run.py --port 5001` | Different port |
| `python run.py --timeout 60` | Allow 60 s for server startup |
| `python run.py --no-color` | Plain text (logs/CI) |

> Do **not** use `flask run` — it breaks WebSocket transport. Always use `python run.py`.

### Pages to open in your browser

| URL | What you see |
|---|---|
| http://localhost:5000/ | **Main dashboard** — stats, Camera Monitor grid, attendance table, event log |
| http://localhost:5000/enroll/screen | **Enrollment** — webcam capture + camera grid |
| http://localhost:5000/admin/campus | **Campus map** — Leaflet.js with live camera status icons |
| http://localhost:5000/alerts/mobile | **Mobile admin view** — alert feed with Acknowledge, Navigate, Panic |

### Quick smoke test from curl

```powershell
# Health check
curl http://localhost:5000/api/health
# { "status": "ok", "enrolled": 0, "uptime_s": 12.4 }

# Enroll a face (use any JPEG with a visible face)
curl -X POST http://localhost:5000/api/enroll `
     -F "subject_id=EMP-001" -F "image=@face.jpg"
# { "status": "enrolled", "total_enrolled": 1 }

# Verify the same face
curl -X POST http://localhost:5000/api/verify `
     -F "subject_id=EMP-001" -F "image=@face.jpg"
# { "similarity": 0.97, "match": true }
```

---

## Where to See Your Data

All runtime data is stored in SQLite files in the `data/` directory (auto-created on first run). Use [DB Browser for SQLite](https://sqlitebrowser.org/) to open any of these:

| File | What is inside |
|---|---|
| `data/embeddings.db` | Enrolled subjects — HMAC-SHA256 key hashes + projected embedding vectors. **No raw images. No plaintext IDs.** |
| `data/attendance.db` | Attendance records + the `alerts` table (forced-entry, tailgate, safety alerts) |
| `data/notifications.db` | Every alert notification — channel (websocket/smtp/sms), status, retry count, timestamps |
| `data/audit.db` | Append-only audit trail — every API action with actor, camera, alert reference |
| `data/left_objects.db` | Left-object events with bounding box, dwell time, JPEG snapshot (base64) |
| `data/loitering.db` | Loitering events with subject key, camera, dwell time |

### Query via REST API (no SQL needed)

```powershell
# How many faces are enrolled?
curl http://localhost:5000/api/health
# { "enrolled": 5, ... }

# Attendance summary
curl http://localhost:5000/api/attendance

# Unacknowledged security alerts
curl "http://localhost:5000/api/alerts?unacked_only=1&n=20"

# Notification delivery log (did the email actually send?)
curl http://localhost:5000/api/notifications?n=20
# Look for: "channel": "smtp", "status": "sent"

# Audit trail (requires Authorization: Bearer <jwt_token>)
curl -H "Authorization: Bearer <token>" http://localhost:5000/api/audit?limit=20

# Left-object events
curl http://localhost:5000/api/left-objects

# Loitering events
curl http://localhost:5000/api/loitering

# ML evaluation metrics (accuracy, ROC-AUC)
curl http://localhost:5000/api/metrics
```

---

## Getting Alerts & Notifications on Your Device

This section tells you **exactly what file to edit** and what value to change for each channel.

### A. Email alerts — step by step

1. **Gmail:** Enable 2-Step Verification → Google Account → Security → App passwords → create a "Mail" password (16 chars)
2. Edit `.env`:

```ini
PCS_SMTP_HOST=smtp.gmail.com
PCS_SMTP_PORT=587
PCS_SMTP_USER=yourname@gmail.com
PCS_SMTP_PASSWORD=abcdabcdabcdabcd    # 16-char app password, no spaces
PCS_SMTP_TO=yourname@gmail.com,security@yourorg.com
```

3. Reload `.env` and restart the server.
4. Test:

```powershell
# Inject a left-object event
curl -X POST http://localhost:5000/api/left-objects/trigger `
     -H "Content-Type: application/json" `
     -d '{"camera_id":"gate-north","dwell_s":45}'

# Confirm it as a threat → severity HIGH → email fires immediately
curl -X POST http://localhost:5000/api/left-objects/1/confirm `
     -H "Content-Type: application/json" `
     -d '{"action_by":"Officer Smith"}'

# Confirm email was sent:
curl http://localhost:5000/api/notifications?n=5
# Look for: "channel": "smtp", "status": "sent"
```

### B. SMS for the Panic button

The built-in `NullSmsDriver` logs SMS without sending. To get real SMS, subclass `SmsDriverBase`. Twilio example:

```python
# my_sms.py — place in the project root
import os
from src.notifications.drivers import SmsDriverBase

class TwilioSmsDriver(SmsDriverBase):
    name = "sms"

    def send(self, notification: dict) -> None:
        from twilio.rest import Client
        client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
        msg = notification["payload"].get("message", "CAMPUS SECURITY ALERT")
        for number in os.environ.get("PCS_PANIC_SMS_TO", "").split(","):
            if number.strip():
                client.messages.create(
                    to=number.strip(),
                    from_=os.environ["TWILIO_FROM_NUMBER"],
                    body=f"[PCS ALERT] {msg}",
                )
```

Add to `.env`:

```ini
PCS_SMS_DRIVER=my_sms.TwilioSmsDriver
PCS_SMS_OPT_IN=1
PCS_PANIC_SMS_TO=+919876543210,+911234567890
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
```

Test:

```powershell
curl -X POST http://localhost:5000/api/alerts/panic `
     -H "Content-Type: application/json" `
     -d '{"triggered_by":"Officer Smith","message":"Intruder at main gate"}'
# { "ok": true, "sms_count": 1 }
```

### C. Real-time browser alerts (zero config)

The dashboard at http://localhost:5000 receives every alert via WebSocket automatically — no setup needed.

### D. Mobile admin view on your phone

Open **http://\<your-pc-IP\>:5000/alerts/mobile** on your phone (same WiFi network). You get:
- Live alert feed with Acknowledge button
- Navigate button (opens Google Maps to the camera's GPS location)
- **Panic button** — sends immediate SMS + WebSocket broadcast, bypassing all throttle limits

### E. Where to change the alert escalation policy

File: `src/notifications/policy.py` — edit `SEVERITY_MAP` and `CHANNEL_MAP`:

```python
# Which alert types are high/medium/low severity:
SEVERITY_MAP = {
    "safety_alert": "high",   # → websocket + smtp + sms
    "forced_entry": "high",
    "tailgate":     "high",
    "loitering":    "medium", # → websocket + smtp only
    "left_object":  "medium",
}

# Which channels each severity uses:
CHANNEL_MAP = {
    "high":   ["websocket", "smtp", "sms"],
    "medium": ["websocket", "smtp"],
    "low":    ["websocket"],
}
```

### F. Throttle & retry settings

| Variable | Default | Change it when |
|---|---|---|
| `PCS_NOTIFY_THROTTLE_S` | `60` | Set to `0` during testing to receive every alert without throttling |
| `PCS_NOTIFY_RETRY_S` | `10` | How often the retry worker polls for failed deliveries |
| `PCS_NOTIFY_MAX_RETRIES` | `3` | Max retry attempts before marking delivery as "failed" |

---

## Running Tests

```powershell
# Activate the venv first
.venv\Scripts\Activate.ps1

# Run all 483 unit tests
python -m pytest tests/ -v --tb=short

# With line-by-line coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run a single module
python -m pytest tests/test_notifications.py -v
python -m pytest tests/test_embedding_store.py -v
python -m pytest tests/test_attendance.py -v
```

### How the tests work WITHOUT real cameras, faces, or data

This is the key insight: **the tests never need a real camera, a face photo, a running server, or a live MQTT broker.** Each layer uses a specific isolation technique:

#### 1. Fresh temporary SQLite databases per test

Every test that needs a database creates a brand-new temporary file:

```python
# Pattern used in every test file:
import tempfile, os

def _tmp_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return NotificationStore(path)   # empty DB, brand-new, isolated
```

Effect: tests never interfere with each other. No pre-existing data is needed. No cleanup after failures.

#### 2. Mock objects replace real external services

Instead of calling real email servers or SMS APIs, tests use `unittest.mock`:

```python
from unittest.mock import MagicMock, patch

# WebSocket test — no real browser needed:
mock_socketio = MagicMock()
driver = WebSocketDriver(mock_socketio)
driver.send(notification)
mock_socketio.emit.assert_called_once_with("alert", notification["payload"])

# SMTP test — no real email server:
with patch("smtplib.SMTP") as mock_smtp:
    smtp_driver.send(notification)
    mock_smtp.assert_called_with("smtp.gmail.com", 587)
    # ↑ verifies the driver called SMTP correctly, without sending a real email
```

#### 3. Synthetic face images (no real photos)

Backbone and face-utils tests create random pixel arrays:

```python
import numpy as np
from PIL import Image

fake_image = Image.fromarray(
    np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
)
embedding = backbone.embed(fake_image)
assert embedding.shape == (512,)
assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5  # L2-normalised
```

#### 4. Backbone stub in integration tests (no GPU, no model download)

Integration tests patch `BackboneEmbedder` with a function that returns a random unit-norm vector in microseconds:

```python
import numpy as np
from unittest.mock import patch

def _stub_embed(image_bytes):
    v = np.random.randn(512).astype("float32")
    return (v / np.linalg.norm(v)).tolist()

# The integration fixture patches the real backbone:
with patch("src.web.flask_app.generate_embedding_from_image", side_effect=_stub_embed):
    # Enroll/verify tests run in milliseconds with zero ML overhead
    ...
```

#### 5. Programmatically generated JPEG bytes

Tests that need a JPEG file create an 8×8 placeholder:

```python
import io
from PIL import Image

buf = io.BytesIO()
Image.new("RGB", (8, 8), color=(100, 150, 200)).save(buf, format="JPEG")
fake_jpeg = buf.getvalue()   # ~350 bytes, fully valid JPEG
```

#### 6. Live Flask server in a daemon thread (integration tests)

Integration tests start a **real Flask+SocketIO server** in a background thread on a random free port — no separate terminal, no port conflicts, automatic cleanup:

```python
# From tests/integration/conftest.py:
port = _free_port()          # random free port (e.g. 54321)
thread = threading.Thread(target=lambda: socketio.run(app, port=port), daemon=True)
thread.start()
_wait_http(f"http://127.0.0.1:{port}/api/health")  # poll until ready
```

#### 7. Auto-skipping when dependencies are absent

```python
pytest.importorskip("torch")              # skip ML tests if torch not installed
pytest.importorskip("paho.mqtt.client")   # skip MQTT tests if paho not installed

# Integration conftest:
if shutil.which("mosquitto") is None:
    pytest.skip("Mosquitto not installed — skipping integration tests")
```

### What the tests prove

| What | How |
|---|---|
| **Business logic correctness** | Dedup, throttle, HMAC, access-control rules behave exactly as specified |
| **Test isolation** | No test depends on another (fresh DB every time → no order-dependency bugs) |
| **Regression safety** | Breaking any function causes ≥1 test to fail immediately |
| **Privacy guarantee** | Tests verify that the store never writes raw IDs or raw embeddings |
| **Full-stack correctness** | Integration tests prove MQTT → Flask → SQLite → REST works end-to-end |

### Test file summary

| File | Tests | Covers |
|---|---|---|
| `test_access_control.py` | 30 | Forced-entry, tailgate, safety-alert rules; AlertStore CRUD |
| `test_attendance.py` | 19 | Attendance marking, session TTL dedup, date-range queries |
| `test_audit.py` | 26 | Append-only audit log, immutability triggers, `/api/audit` |
| `test_auth.py` | 68 | bcrypt, JWT sign/verify, RBAC decorators, auth routes |
| `test_backbone.py` | 20 | ResNet-50 output shape, L2-norm, device placement |
| `test_embedding_store.py` | 56 | HMAC keys, orthogonal projection, encryption, thread safety |
| `test_face_utils.py` | 39 | Multi-face detection, batch embedding, enroll/verify flow |
| `test_left_object.py` | 26 | Dual-EMA detector, IoU tracker, store CRUD |
| `test_loader.py` | 38 | LFW dataset loader, episode sampler, few-shot collate |
| `test_loitering.py` | 30 | Dwell tracker, cooldown, store CRUD |
| `test_notifications.py` | 69 | Store, policy, all drivers, dispatcher throttle/retry |
| `test_prototypical.py` | 51 | ProtoNet config, training step, episodic accuracy |
| `integration/test_end_to_end.py` | ~30 | MQTT → Flask → face verify → alert → REST |
| **Total** | **≥483** | |

### Integration tests

Integration tests spin up a **real Mosquitto broker** and a **real Flask server**, then simulate MQTT traffic (camera frames, door sensor events, smoke sensors) and assert the REST APIs respond correctly.

**Prerequisites**

| Requirement | Install |
|---|---|
| Mosquitto broker | Windows: https://mosquitto.org/download/ · Linux: `sudo apt install mosquitto` · macOS: `brew install mosquitto` |
| `pytest-asyncio` | already in `requirements-dev.txt` |

```powershell
# Run the full integration suite
pytest tests/integration/ -v --tb=short

# Run only MQTT tests (Group C)
pytest tests/integration/ -v -k "Mqtt"

# Run only ML pipeline tests (Group D — requires torch)
pytest tests/integration/ -v -k "Enroll or Verify or Forced or Loitering"

# Skip ML tests (Groups A–C only)
pytest tests/integration/ -v -k "not torch"

# Run everything: unit + integration
pytest tests/ -v --tb=short
```

The suite **auto-skips** gracefully when Mosquitto is not installed — it will not fail, just print a skip message.

**Test files (~483 total):**

| File | Tests | Covers |
|---|---|---|
| `test_access_control.py` | 30 | AlertStore, AccessControlEngine (forced-entry, tailgate, safety-alert) |
| `test_attendance.py` | 19 | Attendance marking, session TTL dedup, aggregated queries |
| `test_audit.py` | 26 | AuditStore CRUD, immutability triggers, `/api/audit` endpoint, audit rows per route |
| `test_auth.py` | 68 | UserStore, bcrypt, JWT, RBAC decorators, auth Blueprint endpoints |
| `test_backbone.py` | 20 | ResNet-50 backbone, L2 normalisation, device handling |
| `test_embedding_store.py` | 56 | HMAC-SHA256 store, Fernet encryption, thread-safety |
| `test_face_utils.py` | 39 | Multi-face detection, batch embedding, enroll/verify flow |
| `test_left_object.py` | 26 | Left-object store + dual-EMA background detector |
| `test_loader.py` | 38 | LFW dataset wrapper, episode sampler, few-shot collate |
| `test_loitering.py` | 30 | Loitering store + per-camera dwell tracker, cooldown |
| `test_notifications.py` | 69 | NotificationStore, EscalationPolicy, all drivers, dispatcher (throttle, retry, async), Flask integration |
| `test_prototypical.py` | 51 | ProtoNet config, training loop, episodic accuracy |
| `integration/test_end_to_end.py` | ~30 | Mosquitto broker fixture, live Flask server, MQTT frame/sensor/door simulation, enroll/verify pipeline, forced-entry alert, loitering detection |

---

## Load Testing

`scripts/load_test_verify_frame.py` is a self-contained, dependency-free (only `requests` + `Pillow`, both already installed) threaded producer that pushes synthetic **6-face JPEG frames across 10 simulated cameras at 4 fps for 30 seconds** against `POST /api/verify/frame` and writes a latency CSV.

### Quick start

```powershell
# Defaults: 10 cameras × 6 faces × 4 fps × 30 s → ~1 200 requests
python scripts/load_test_verify_frame.py

# Custom config
python scripts/load_test_verify_frame.py `
    --url      http://localhost:5000 `
    --cameras  10 `
    --fps      4 `
    --duration 30 `
    --faces    6 `
    --out      experiments/my_load_test.csv

# Skip the single-query warmup (faster, but no baseline measurement)
python scripts/load_test_verify_frame.py --no-warmup
```

The server must be running before the script starts.  Use `python run.py --server-only` in a separate terminal.

### What it measures

| Measurement | How |
|---|---|
| **Single-query baseline** | One isolated request fires *before* the concurrent load so you get a clean CPU-inference latency with zero contention |
| **End-to-end round-trip** | Wall-clock time from `requests.post()` call to response received, measured per worker thread |
| **p50 / p95 / p99** | Derived from all 200-OK responses in the CSV |
| **Error rate** | Non-200 responses + connection errors + timeouts |
| **Throughput** | Successful requests ÷ actual elapsed seconds |

### CSV columns

| Column | Type | Description |
|---|---|---|
| `request_id` | int | Monotonic across all threads |
| `camera_id` | str | Simulated camera (e.g. `cam-00` … `cam-09`) |
| `worker_id` | int | Thread index |
| `status_code` | int | HTTP response code; `0` = connection/timeout error |
| `n_faces` | int | `n_faces` from response JSON; `-1` on error |
| `latency_ms` | float | End-to-end round-trip in milliseconds |
| `error` | str | Empty on success; brief description on failure |

### Expected output (CPU, no GPU)

```
Connecting to http://localhost:5000/api/health … ok  (enrolled=0, uptime=3s)
Building synthetic 6-face JPEG frame … 28,456 bytes
Running single-query baseline (CPU) … 142.3 ms  [< 200 ms ✓]

Starting load test: 10 cameras × 4.0 fps × 30s (≈1200 requests expected)
  [.........]

────────────────────────────────────────────────────────────────────────
  Privacy Campus Security — Load Test Summary
────────────────────────────────────────────────────────────────────────
  Config   : 10 camera(s) × 6 face(s)/frame @ 4.0 fps × 30s
  Endpoint : POST /api/verify/frame

  Requests sent : 1 197
  Success (200) : 1 193  (99.7%)
  Errors        :     4  (0.3%)

  Latency (successful requests only):
    min  :     98.4 ms
    p50  :    155.1 ms
    p95  :    238.7 ms
    p99  :    291.0 ms
    max  :    412.3 ms
    mean :    162.4 ms

  Single-query baseline : 142.3 ms  —  CPU < 200 ms  [PASS ✓]

  Throughput : 39.8 successful req/s  (1193 ok / 30.0s)
  CSV output : D:\...\experiments\load_test_20260301_120000.csv
────────────────────────────────────────────────────────────────────────
```

### CPU vs GPU path

| Scenario | Expected single-query latency | Notes |
|---|---|---|
| CPU only (default) | 100 – 250 ms | Meets the < 200 ms target for isolated queries; p95 under concurrent 10-camera load may exceed 200 ms |
| CUDA GPU | 20 – 60 ms | Set `PCS_DEVICE=cuda`; rebuild image with a CUDA base if using Docker |
| Stub backbone (integration tests) | < 5 ms | `BackboneEmbedder` replaced with numpy RNG — exercises HTTP path only |

For GPU runs, pass `PCS_DEVICE=cuda` before starting the server and re-run the load test; `single-query baseline` should drop below 60 ms and p95 under 10-camera load should stay well within 200 ms.

---

## Structured Logging & Prometheus Metrics

### Structured JSON logging

All log output is routed through **structlog** (with a stdlib fallback) and written as compact JSON lines, ready to ingest with Loki, Elasticsearch, or any log aggregator.

```json
{"timestamp": "2026-03-01T12:00:01.234Z", "level": "info",  "logger": "src.web.flask_app", "event": "verify_frame",         "event_type": "verify",     "camera_id": "gate-north", "n_faces": 2, "latency_ms": 142.3}
{"timestamp": "2026-03-01T12:00:02.011Z", "level": "info",  "logger": "src.web.flask_app", "event": "attendance_mark",      "event_type": "attendance", "n_marked": 1, "n_skipped": 0, "ttl_s": 300}
{"timestamp": "2026-03-01T12:00:03.500Z", "level": "warning","logger": "src.web.flask_app", "event": "access_control_alert","event_type": "alert",      "alert_type": "forced_entry", "camera_id": "gate-north", "door_id": "door-01", "subject_key": "grp:A:f0:abc"}
```

| Structured field | Present in events | Description |
|---|---|---|
| `event_type` | all | `"verify"`, `"attendance"`, `"alert"` |
| `camera_id` | `verify_frame`, `access_control_alert` | Camera that produced the frame or alert |
| `subject_key` | `access_control_alert` | Opaque hashed subject key (never plaintext) |
| `similarity` | (via verify/frame endpoint response body) | Cosine similarity score |
| `latency_ms` | `verify_frame` | Full-pipeline wall-clock time |
| `alert_type` | `access_control_alert` | `forced_entry`, `tailgate`, `safety_alert` |

Switch to human-readable coloured output for interactive development:

```powershell
$env:PCS_LOG_FORMAT = "console"
python run.py
```

### Prometheus metrics (`/metrics`)

`GET /metrics` returns Prometheus exposition format — scrapeable directly by a Prometheus server.

```powershell
# View raw output
curl http://localhost:5000/metrics

# Or with python
python -c "import requests; print(requests.get('http://localhost:5000/metrics').text[:1200])"
```

**Metric families**

| Metric | Type | Labels | Description |
|---|---|---|---|
| `verify_requests_total` | Counter | `status` (`success`\|`error`) | Total calls to `POST /api/verify/frame` |
| `alerts_total` | Counter | `alert_type` | Total security alerts by type (`forced_entry`, `tailgate`, `safety_alert`) |
| `attendance_marks_total` | Counter | — | Deduplicated attendance rows inserted |
| `verify_latency_seconds` | Histogram | — | Full-pipeline latency; buckets tuned around the 200 ms CPU target |

**Prometheus scrape config (`prometheus.yml`)**

```yaml
scrape_configs:
  - job_name: privacy_campus_security
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: /metrics
    scrape_interval: 15s
```

**Example Grafana panels**

```promql
# Verify request rate (per minute)
rate(verify_requests_total[1m]) * 60

# p95 verify latency
histogram_quantile(0.95, rate(verify_latency_seconds_bucket[5m]))

# Alert rate by type
rate(alerts_total[5m])
```

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Top-1 accuracy (LFW, K=5) | ≥ 85 % |
| ROC-AUC (pairs) | ≥ 0.80 |
| TAR @ FAR = 0.1 % | ≥ 70 % |
| CPU inference latency | < 200 ms |
| Privacy reconstruction attack | < 5 % success |

---

## Socket.IO Events

### Client → Server

| Event | Payload | Description |
|---|---|---|
| `arm_enroll` | `{camera_id}` | Mark a camera as active for enrollment; next `enroll_frame` processes a face |
| `disarm_enroll` | `{camera_id?}` | Cancel enrollment mode for a camera (empty = all cameras for this session) |
| `enroll_frame` | `{camera_id, image_b64, group_id?}` | Submit a webcam frame; server detects faces and batch-enrolls |
| `frame` | `{image_b64, mode?}` | Generic frame for enroll/verify workflows on the main dashboard |
| `verify` | `{subject_id, image_b64}` | Direct 1:1 verify request |
| `get_attendance` | `{}` | Request a fresh attendance snapshot |

### Server → Client (broadcast)

| Event | Payload | Description |
|---|---|---|
| `camera_status` | `{camera_id, armed, armed_by?}` | Arm/disarm notification (all clients) |
| `camera_update` | `{camera_id, status, last_event, last_seen}` | Real-time status colour for campus map |
| `enroll_result` | `{status, camera_id, n_enrolled, faces[], total_enrolled, timestamp}` | Enrollment outcome |
| `verify_result` | `{similarity, match, …}` | Verification outcome |
| `attendance_update` | `{records[]}` | Fresh attendance snapshot |
| `arm_ack` | `{camera_id, armed}` | Acknowledge arm/disarm to the initiating client |
| `alert` | `{id, alert_type, door_id, camera_id, subject_key, confidence, timestamp_utc, meta, acknowledged, acked_by, acked_at}` | New security alert broadcast to all clients |
| `alert_ack` | Full alert object (same shape as `alert`) | Alert acknowledged — full object so clients can render audit trail without an extra fetch |
| `left_object` | `{id, camera_id, bbox, first_seen_utc, detected_utc, dwell_s, snapshot_b64, dismissed, confirmed}` | Left-object event fired when an object dwells past the threshold |
| `left_object_update` | Same shape as `left_object` | Emitted after a remove or confirm action so clients can update status in-place |
| `loitering` | `{id, camera_id, subject_key, first_seen_utc, detected_utc, dwell_s, dismissed}` | Loitering event fired when the same face is continuously present past the dwell threshold |
| `loitering_update` | Same shape as `loitering` | Emitted after a dismiss action |
| `panic` | `{event, triggered_by, message, timestamp, camera_id?}` | Emergency escalation broadcast — emitted by `POST /api/alerts/panic`; the mobile view and desktop dashboard both listen for this |

---

## Camera Registry (`configs/cameras.yaml`)

All physical cameras are declared here.  The server reads this file on startup
(path overridable via `PCS_CAMERAS_YAML`).

```yaml
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
  # … 4 more cameras …

mqtt:
  broker_host: localhost
  broker_port: 1883
  topic_prefix: campus
```

Derived MQTT topics for every camera:

```
campus/<camera_id>/frame    ← raw JPEG bytes
campus/<camera_id>/sensor   ← JSON telemetry (e.g. {"smoke":true})
```

---

## Access Control Engine

`src/privacy/access_control.py` implements a stateful, thread-safe rule engine that correlates door-sensor events with face-recognition results in real time.

### Rules

| Alert Type | Trigger | Window |
|---|---|---|
| `forced_entry` | Door `open` + `matched == false` verify for the same door | ± 2 s |
| `tailgate` | ≥ 2 persons at the same door with at least one `matched == false` | 5 s sliding |
| `safety_alert` | `campus/<camera_id>/sensor` has `{smoke:true}` OR image heuristic detects smoke/fire | cooldown-throttled |

### Data flow

```
MQTT campus/<door_id>/event
          │
          ▼
   door subscriber thread
          │
          ▼
   AccessControlEngine.on_door_event()
          │
          ├── correlate with verify_buf ──→ forced_entry?
          └── ───────────────────────────────────────────

POST /api/verify/frame  (camera_id + door_id optional form fields)
          │
          ▼
   AccessControlEngine.on_verify_result()
          ├── correlate with door_buf   ──→ forced_entry?
          └── scan window               ──→ tailgate?

Either rule fires
          │
          ▼
   AlertStore.add()  (persists to SQLite)
          │
          ▼
   socketio.emit("alert", …)  (broadcasts to all dashboard clients)
   socketio.emit("camera_update", …status=alert…)  (turns icon red on campus map)
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PCS_CAMERAS_YAML` | `configs/cameras.yaml` | Camera registry (also provides broker settings for door subscriber) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PCS_STORE` | `data/embeddings.db` | Path to the SQLite embedding store |
| `PCS_EMBEDDING_SALT` | *(insecure default)* | HMAC salt — **set this in production** |
| `PCS_ENCRYPT` | `0` | Set to `1` to enable Fernet at-rest encryption |
| `PCS_DEVICE` | `cpu` | PyTorch device (`cpu` or `cuda`) |
| `FLASK_DEBUG` | `0` | Set to `1` for auto-reload in development |
| `PCS_CAMERAS_YAML` | `configs/cameras.yaml` | Camera registry YAML path |
| `PCS_ATTENDANCE_DB` | `data/attendance.db` | SQLite file for the attendance log |
| `PCS_SESSION_TTL` | `300` | Attendance session TTL in seconds (dedup window) |
| `PCS_LOITER_DWELL_S` | `60` | Seconds before a loitering alert fires |
| `PCS_LOITER_STALE_S` | `10` | Seconds of absence before dwell timer resets |
| `PCS_LOITER_COOLDOWN_S` | `120` | Minimum seconds between repeat loitering events |
| `PCS_LOITER_FRAME_INTERVAL_S` | `2.0` | Face-recognition throttle per camera (seconds) |
| `PCS_NOTIFY_THROTTLE_S` | `60` | Minimum seconds between duplicate `(channel, alert_type)` notifications |
| `PCS_NOTIFY_MAX_RETRIES` | `3` | Maximum delivery attempts per notification record |
| `PCS_NOTIFY_RETRY_S` | `10` | Seconds between retry-worker sweeps |
| `PCS_SMS_OPT_IN` | `0` | Set to `1` to include SMS in high-severity escalation |
| `PCS_PANIC_SMS_TO` | _(none)_ | Comma-separated phone numbers (E.164) that receive **immediate** SMS when the Panic button is pressed. Falls back to `PCS_SMTP_TO` if blank |
| `PCS_SMS_DRIVER` | _(none)_ | Fully-qualified class path of a custom `SmsDriverBase` subclass |
| `PCS_SMTP_HOST` | _(none)_ | SMTP server hostname (enables email delivery when set) |
| `PCS_SMTP_PORT` | `587` | SMTP port (STARTTLS by default) |
| `PCS_SMTP_USER` | _(none)_ | SMTP login username (optional) |
| `PCS_SMTP_PASSWORD` | _(none)_ | SMTP login password (optional) |
| `PCS_SMTP_FROM` | `pcs@localhost` | Sender address |
| `PCS_SMTP_TO` | _(= FROM)_ | Comma-separated recipient addresses |
| `PCS_SMTP_TLS` | `1` | Set to `0` to disable STARTTLS |
| `PCS_LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `PCS_LOG_FORMAT` | `json` | Log output format: `json` (production) or `console` (dev / interactive) |
| `PCS_AUDIT_DB` | `data/audit.db` | SQLite file for the immutable audit log |
| `PCS_JWT_SECRET` | *(random ephemeral)* | Secret key for JWT signing — **set a long random value in production** |
| `PCS_JWT_EXPIRY_H` | `8` | JWT token lifetime in hours |
| `PCS_AUTH_DB` | `data/users.db` | SQLite file for the users / roles table |
| `PCS_FLASK_SECRET` | *(= JWT secret)* | Flask session secret — defaults to `PCS_JWT_SECRET` |

> **Tip:** All of these variables can be kept in a `.env` file at the project root (see [`.env`](.env)).  
> Copy the template, fill in the five starred fields and the project is ready to run.

---

## Audit Logging

Every sensitive mutation is appended to an **immutable SQLite audit log** (`data/audit.db`, configurable via `PCS_AUDIT_DB`). Rows can never be updated or deleted — SQLite `BEFORE UPDATE` and `BEFORE DELETE` triggers enforce this at the database level.

### Audited event types

| Event type | Triggered by |
|---|---|
| `enroll` | `POST /api/enroll`, `POST /api/enroll/frame` |
| `enroll_batch` | `POST /api/enroll/batch` |
| `verify` | `POST /api/verify` |
| `verify_frame` | `POST /api/verify/frame` (logs n\_faces, n\_matched, latency, camera\_id) |
| `alert` | Any new alert created by AccessControlEngine (forced\_entry, tailgate, safety\_alert) |
| `ack` | `POST /api/alerts/<id>/ack` |
| `panic` | `POST /api/alerts/panic` |

### Audit row schema

```json
{
  "id":            1,
  "timestamp_utc": "2026-03-01T10:00:01+00:00",
  "event_type":    "ack",
  "actor":         "officer-smith",
  "subject_id":    null,
  "camera_id":     "gate-north",
  "alert_id":      7,
  "detail":        {"alert_type": "forced_entry", "acked_by": "officer-smith", "acked_at": "2026-03-01T10:00:01+00:00"}
}
```

### Query the audit log

```powershell
# All audit rows (newest first, default limit 100)
curl -H 'Authorization: Bearer <admin-token>' http://localhost:5000/api/audit

# Only ack events
curl -H 'Authorization: Bearer <admin-token>' 'http://localhost:5000/api/audit?event_type=ack&limit=50'

# All actions by a specific operator
curl -H 'Authorization: Bearer <admin-token>' 'http://localhost:5000/api/audit?actor=officer-smith'

# Events for a specific alert
curl -H 'Authorization: Bearer <admin-token>' 'http://localhost:5000/api/audit?alert_id=7'

# Events after a given point in time
curl -H 'Authorization: Bearer <admin-token>' 'http://localhost:5000/api/audit?since=2026-03-01T09:00:00Z'
```

Only users with the `admin` role can read the audit log (`GET /api/audit`). The audit log itself is write-only from application code — there is no delete or update path.

### Environment variable

| Variable | Default | Description |
|---|---|---|
| `PCS_AUDIT_DB` | `data/audit.db` | SQLite file for the immutable audit log |

---

## Authentication & RBAC

The web interface and REST API are protected by **JWT-based role-based access control (RBAC)**.  
Users are stored in a SQLite table with **bcrypt**-hashed passwords (cost factor 12).

### Roles

| Role | Permissions |
|---|---|
| `viewer` | Read-only: dashboards, alert list, metrics |
| `operator` | viewer + enroll/verify faces + acknowledge **non-high-severity** alerts |
| `admin` | Full access: acknowledge any alert (including high-severity), trigger Panic, manage users |

> **High-severity alert types** (`safety_alert`, `forced_entry`, `tailgate`) require an **admin** token to acknowledge.

### Default credentials

When the users table is empty the server automatically creates:

| Username | Password | Role |
|---|---|---|
| `admin` | `admin123` | `admin` |

A `WARNING` is logged at startup. **Change the password immediately:**

```powershell
curl -s -X POST http://localhost:5000/auth/users/admin/password \
  -H 'Authorization: Bearer <admin-token>' \
  -H 'Content-Type: application/json' \
  -d '{"password": "my-new-secure-password"}'
```

### Obtaining a token

```powershell
# Returns {token, username, role}
curl -s -X POST http://localhost:5000/auth/login \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -d '{"username": "admin", "password": "admin123"}'
```

Pass the token as a **Bearer header** on subsequent API calls:

```powershell
curl -H 'Authorization: Bearer <token>' http://localhost:5000/api/health
```

Browser sessions receive a **`pcs_token` httponly cookie** set automatically after a successful login at `GET /auth/login`.

### Auth endpoints

| Method | Path | Role required | Description |
|---|---|---|---|
| `GET` | `/auth/login` | — | Login page (HTML) or redirect if already authenticated |
| `POST` | `/auth/login` | — | Authenticate; returns JWT + sets cookie |
| `POST` | `/auth/logout` | — | Clear the auth cookie |
| `GET` | `/auth/me` | any | Return current user info (JSON) |
| `GET` | `/auth/users` | `admin` | List all users |
| `POST` | `/auth/users` | `admin` | Create a user `{username, password, role}` |
| `PUT` | `/auth/users/<u>/role` | `admin` | Change role `{role}` |
| `POST` | `/auth/users/<u>/password` | admin or self | Change password `{password}` |
| `DELETE` | `/auth/users/<u>` | `admin` | Delete a user |

### Socket.IO authentication

Pass the JWT in the Socket.IO handshake `auth` option:

```javascript
const socket = io({
  auth: { token: "<your-jwt>" }
});
```

The server validates the token on `connect` and stores the role in the session so write events (e.g. `arm_enroll`) are gated by role.

### Protected routes summary

| Route | Minimum role |
|---|---|
| `GET /` (dashboard) | any authenticated |
| `GET /admin/campus` | any authenticated |
| `GET /enroll/screen` | `operator` |
| `GET /alerts/mobile` | `operator` |
| `POST /api/enroll` | `operator` |
| `POST /api/enroll/batch` | `operator` |
| `POST /api/enroll/frame` | `operator` |
| `POST /api/alerts/<id>/ack` (low/medium severity) | `operator` |
| `POST /api/alerts/<id>/ack` (high severity) | `admin` |
| `POST /api/alerts/panic` | `admin` |
| `POST /api/left-objects/<id>/remove` | `operator` |
| `POST /api/left-objects/<id>/confirm` | `operator` |
| `POST /api/loitering/<id>/dismiss` | `operator` |

---

## DB Migrations

Schema migrations are managed by a lightweight **stdlib-only** custom runner
(`scripts/migrate.py`) that targets each SQLite database independently.
No external Alembic/Flyway process or ORM is required.

### Directory layout

```
migrations/
├─ embeddings/
│   └─ 0001_initial_schema.sql          ← embeddings table
├─ attendance/
│   ├─ 0001_initial_schema.sql          ← attendance / cameras / alerts /
│   │                                       notifications / loitering / left-objects
│   └─ 0002_alerts_add_camera_index.sql ← example incremental migration
├─ audit/
│   └─ 0001_initial_schema.sql          ← audit_log + immutability triggers
└─ users/
    └─ 0001_initial_schema.sql          ← users table
```

Each SQL file is numbered `NNNN_<description>.sql`.  New migrations are added
by creating the next numbered file — the runner processes them in alphabetical
order and skips anything already applied.

### Apply migrations

```powershell
make migrate
# — or —
python scripts/migrate.py
```

Expected output:

```
╔══════════════════════════════════════════════════════════╗
║  Privacy Campus Security — DB Migration Runner           ║
╚══════════════════════════════════════════════════════════╝

[embeddings]  db=data/embeddings.db
  [OK]   0001_initial_schema

[attendance]  db=data/attendance.db
  [OK]   0001_initial_schema
  [OK]   0002_alerts_add_camera_index

[audit]  db=data/audit.db
  [OK]   0001_initial_schema

[users]  db=data/users.db
  [OK]   0001_initial_schema

✓  All migrations complete.
```

### Additional commands

| Command | Effect |
|---|---|
| `make migrate` | Apply all pending migrations across all databases |
| `make migrate-status` | Print applied / pending status (read-only) |
| `make migrate-dry-run` | Preview what would be applied without writing |
| `python scripts/migrate.py --db attendance` | Migrate only the attendance database |

### How it works

| Feature | Detail |
|---|---|
| **Tracking table** | `_schema_migrations (id, applied_at, checksum)` — created automatically in every managed database |
| **Idempotent** | Re-running `make migrate` is always safe; already-applied files are skipped |
| **Checksum recorded** | SHA-256 of each SQL file is stored for auditability |
| **No external deps** | Uses only `sqlite3` from the Python standard library |
| **Immutability-safe** | The `audit_log` migration installs SQLite triggers that prevent any UPDATE/DELETE — consistent with the append-only design |

---

## Database Backups

`scripts/backup.py` creates timestamped, hot (online) backups of every SQLite
database using the **SQLite backup API** — safe while the server is running.

### Quick backup

```powershell
make backup
# — or —
python scripts/backup.py
```

Backups land in `data/backups/<YYYY-MM-DD>/`:

```
data/backups/
└─ 2026-03-01/
    ├─ embeddings_20260301T120000Z.db.gz
    ├─ attendance_20260301T120000Z.db.gz
    ├─ audit_20260301T120000Z.db.gz
    └─ users_20260301T120000Z.db.gz
```

### Backup options

| Command / flag | Effect |
|---|---|
| `make backup` | Back up all databases locally (gzip compressed, keep 7 days) |
| `make backup-s3` | Back up + upload to S3 (requires `PCS_BACKUP_S3_BUCKET` and `boto3`) |
| `--db attendance` | Back up only the attendance database |
| `--retain 14` | Keep 14 days of daily backups (default: 7) |
| `--no-compress` | Skip gzip compression (raw `.db` files) |
| `--no-rotate` | Skip automatic deletion of old backups |

### Scheduled daily backups

Add a scheduled task (cron on Linux/macOS, Task Scheduler on Windows) to run
the backup automatically every day:

```bash
# Linux / macOS — add to crontab (crontab -e)
0 2 * * *  cd /app && python scripts/backup.py >> /var/log/pcs-backup.log 2>&1
```

```powershell
# Windows — PowerShell (run once as admin to register the task)
$action  = New-ScheduledTaskAction -Execute "python" `
           -Argument "scripts\backup.py" `
           -WorkingDirectory "D:\Coding\Major_Project\privacy-campus-security"
$trigger = New-ScheduledTaskTrigger -Daily -At "02:00"
Register-ScheduledTask -Action $action -Trigger $trigger `
    -TaskName "PCS Daily Backup" -RunLevel Highest
```

### S3 upload

Install `boto3` and set the bucket name, then run `make backup-s3`:

```powershell
pip install boto3

$env:PCS_BACKUP_S3_BUCKET = "my-pcs-backups"   # your S3 bucket
$env:AWS_REGION           = "ap-south-1"        # or any region
make backup-s3
```

S3 keys follow the pattern:

```
<PCS_BACKUP_S3_PREFIX>/<db_name>/<YYYY-MM-DD>/<filename>
e.g. pcs-backups/attendance/2026-03-01/attendance_20260301T120000Z.db.gz
```

### Backup environment variables

| Variable | Default | Description |
|---|---|---|
| `PCS_BACKUP_DIR` | `data/backups` | Root directory for local backup files |
| `PCS_BACKUP_RETAIN` | `7` | Days of daily backups to keep (0 = no rotation) |
| `PCS_BACKUP_COMPRESS` | `1` | `1` = gzip compress, `0` = leave as raw `.db` |
| `PCS_BACKUP_S3_BUCKET` | _(disabled)_ | S3 bucket name; leave blank to skip S3 upload |
| `PCS_BACKUP_S3_PREFIX` | `pcs-backups` | S3 key prefix |

### How the backup works

| Step | Detail |
|---|---|
| **Hot copy** | `sqlite3.Connection.backup()` — the SQLite online-backup API; consistent even under concurrent writes |
| **WAL mode** | All databases use WAL journal mode so readers (backup) never block writers |
| **Compression** | Python `gzip` module; typically 10–30× reduction for sparse SQLite files |
| **Rotation** | Backup sub-directories named `YYYY-MM-DD`; directories older than `PCS_BACKUP_RETAIN` days are deleted |
| **S3** | `boto3.client("s3").upload_file()` with standard AWS credential chain (env, `~/.aws`, IAM role) |

---

## CI / CD

Automated pipelines live in [.github/workflows/ci.yml](.github/workflows/ci.yml).

### Workflow overview

| Job | Runs on | When |
|---|---|---|
| **Lint** (`ruff` + `flake8`) | `ubuntu-latest` · Python 3.11 | Every push & PR |
| **Test** (unit suite, 483 tests) | `ubuntu-latest` · Python 3.11 matrix | Every push & PR |
| **Docker build + push** | `ubuntu-latest` | Every push (build); `main` branch & `v*.*.*` tags (push) |

### Trigger rules

```
Any branch push / PR  → lint + test
main branch push       → lint + test + docker build + push to GHCR
git tag  v*.*.*        → lint + test + docker build + push to GHCR (with semver tags)
```

### Container registry

Images are published to **GitHub Container Registry** (`ghcr.io`) and tagged automatically:

| Git event | Image tag(s) |
|---|---|
| Push to `main` | `ghcr.io/your-org/privacy-campus-security:main`, `:sha-<short>` |
| Push tag `v1.2.3` | `ghcr.io/your-org/privacy-campus-security:1.2.3`, `:1.2`, `:sha-<short>` |
| Pull request | `ghcr.io/your-org/privacy-campus-security:pr-<number>` |

No extra secrets are required — the workflow uses the auto-injected `GITHUB_TOKEN`.

### Switching to Docker Hub

Add these **repository secrets** and uncomment the Docker Hub login step in [.github/workflows/ci.yml](.github/workflows/ci.yml#L100):

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (read/write) |

### Optional secrets

| Secret | Purpose |
|---|---|
| `CODECOV_TOKEN` | Upload coverage reports to [codecov.io](https://codecov.io) |
| `DOCKERHUB_USERNAME` | Docker Hub push (see above) |
| `DOCKERHUB_TOKEN` | Docker Hub push (see above) |

### Running CI locally (act)

Use [act](https://github.com/nektos/act) to simulate a push event without opening a PR:

```powershell
# Install act (Windows — Chocolatey)
choco install act-cli

# Dry-run the full CI pipeline locally
act push --container-architecture linux/amd64

# Run only the test job
act push -j test
```

---

## License

MIT © 2026 Abhishek Mittal & Smriti Aggarwal
