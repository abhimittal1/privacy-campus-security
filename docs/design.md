# System Design — Privacy-Preserving Campus Security

> Authors: Abhishek Mittal & Smriti Aggarwal · 2026

---

## 1. Goals

| Goal | Detail |
|---|---|
| **Accuracy** | ≥ 85 % top-1 at K=5 on LFW; ROC-AUC ≥ 0.80 |
| **Privacy** | No raw images, no plaintext IDs stored at rest — ever |
| **Low latency** | Single-query CPU inference < 200 ms |
| **Modularity** | Backbone, head, store, and routes are independently replaceable |
| **Real-time security** | Door-sensor MQTT events correlated with face-verify results in < 100 ms |

---

## 1a. Threat Model

### Assets and adversaries

| Asset | Who must be protected from | Risk without protection |
|---|---|---|
| Subject identity (who is enrolled) | Database thief, insider, rogue admin | Stalking, discrimination, profiling |
| Face biometric (embedding vector) | Algorithmic inversion attacker | Reconstructing a face without consent |
| Attendance history | Employer misuse, data broker | Detailed movement tracking |
| Audit log | Rogue admin | Covering up unauthorized access |
| JWT tokens | Network eavesdropper | Unauthorized API access |

### Threat scenarios and mitigations

| Threat | Attack vector | Mitigation in this system |
|---|---|---|
| **Database theft** | Attacker copies `embeddings.db` | Keys are `SHA-256(salt ‖ subject_id)` — uninvertible without the salt. Embeddings are orthogonally projected — not reversible to original 512-D vector. Optional Fernet AES-128 encryption via `PCS_ENCRYPT=1`. |
| **Embedding inversion** | ML model trained to reconstruct face from stored vector | Orthogonal projection changes the geometric structure; the projected vector is in a different basis. Even if the projection matrix leaks, you need the original HMAC salt to look up who it belongs to. |
| **Man-in-the-middle (network)** | Intercept MQTT frames or REST calls | Deploy NGINX with TLS (HTTPS); enable Mosquitto TLS in `mosquitto/mosquitto.conf`. The system architecture separates camera MQTT from web REST — MQTT can be on a private LAN. |
| **Replay attack** | Replay a captured JPEG to enroll/verify as someone else | JWT tokens have short expiry (`PCS_JWT_EXPIRY_S`, default 3600 s). Production deployments should use TLS to prevent capture. |
| **Insider — rogue admin** | Admin modifies or deletes audit records | `audit.db` has SQLite `BEFORE UPDATE` / `BEFORE DELETE` triggers that raise errors. Rows are immutable once written. |
| **Privilege escalation** | Operator tries to access admin-only endpoints | RBAC decorators (`@require_role("admin")`) on every sensitive route. Roles encoded in JWT payload and verified server-side on every request. |
| **Denial of service** | Flood `/api/verify/frame` | Notification throttle (`PCS_NOTIFY_THROTTLE_S`) prevents alert storms. Load balancer + NGINX rate limiting recommended for production. |
| **Camera spoofing** | Publish fake JPEG to MQTT topic | MQTT authentication required (username/password in `mosquitto.conf`). Without credentials, broker rejects connections. |

### Trust boundary diagram

```
UNTRUSTED ZONE                     TRUSTED ZONE (server process)
──────────────────────────────     ─────────────────────────────────────────
  Browser (user)                     flask_app.py  (JWT-authenticated)
  curl / external API calls          EmbeddingStore (HMAC + projection)
  MQTT publishers (cameras)          AttendanceStore (session-TTL dedup)
  SMTP relay                         AccessControlEngine (rule engine)
  SMS gateway                        AuditStore (append-only, immutable)
                                     NotificationDispatcher (throttle+retry)
──────────────────────────────     ─────────────────────────────────────────
         │ All external input            │ Output only: JSON responses,
         │ validated + sanitised         │ Socket.IO events, email, SMS
         ▼                              ▼
     [Authentication boundary: JWT Bearer token on every /api/* route]
     [RBAC: admin / operator / viewer roles enforced per route]
```

---

## 1b. Data Minimization

Data minimization is the principle of collecting and retaining the **least amount of personal data** necessary for the system to function. This system applies it at every layer.

### What is NEVER stored

| Data | Why it is discarded |
|---|---|
| Raw JPEG / face crop | Pixel data is only needed for `embed()`. After the 512-D vector is produced, the image bytes are GC'd. |
| Plaintext subject ID (e.g. `"EMP-001"`) | Only `SHA-256(salt ‖ subject_id)` is written to the DB. The original ID cannot be recovered from the hash without the salt. |
| Original 512-D embedding | The embedding is projected by a random orthogonal matrix P before writing. `proj = embedding @ P`. The original manifold geometry is not preserved. |
| GPS location of a person | The system records which camera saw a face, not a lat/lng of the person. Camera coordinates are fixed metadata in `configs/cameras.yaml`. |

### What IS stored and why

| Data | Where | Retention | Purpose |
|---|---|---|---|
| Projected embedding (512-D float32) | `embeddings.db` | Until subject is un-enrolled | Required for verify/identify |
| HMAC key hash | `embeddings.db` | Until subject is un-enrolled | Maps hash → embedding row |
| Attendance record (subject_key, camera_id, similarity, matched, timestamp) | `attendance.db` | Configurable (default: forever) | Attendance reporting |
| Alert record (alert_type, camera_id, door_id, confidence, timestamp) | `attendance.db` (alerts table) | Configurable | Security incident log |
| Notification delivery record (channel, status, retry_count, timestamp) | `notifications.db` | Configurable | Operational audit |
| Audit event (event_type, actor, camera_id, alert_id, timestamp) | `audit.db` | **Forever, immutable** | Regulatory / legal accountability |

### Orthogonal projection — technical detail

```
original_embedding:  v ∈ ℝ⁵¹² (unit sphere, L2-norm = 1)
projection matrix:   P ∈ ℝ⁵¹²ˣ⁵¹² (random orthogonal, generated once per store)
stored:              proj = v @ P

Properties:
  ‖proj‖₂ = ‖v‖₂ = 1          (isometric — distances preserved for verify)
  P is stored in clear text    (in the DB alongside the vector)
  v = proj @ Pᵀ                (mathematically invertible IF P is known)

Defence-in-depth rationale:
  1. proj alone, without P: cannot recover v (need P)
  2. proj + P, without salt: can recover v, but cannot link it to a person (HMAC key unknown)
  3. proj + P + salt: can link to person — this is the "master key" scenario
     → Mitigation: store salt in env var (PCS_EMBEDDING_SALT), never in the DB
     → Mitigation: enable Fernet encryption (PCS_ENCRYPT=1) to encrypt proj at rest
     → Mitigation: rotate salt periodically (requires re-enrollment)
```

### Enroll-batch anonymous mode

When `POST /api/enroll/batch` is called, each detected face gets a completely random ephemeral key:

```
key = "grp:<group_id>:f<face_idx>:<secrets.token_hex(16)>"
stored_key = SHA-256(salt ‖ key)
```

The plaintext key is **returned to the caller once** in the JSON response and **never persisted**. If the caller loses it, the enrollment cannot be linked to any person — it becomes an orphaned projection. This is the strongest privacy guarantee the system offers for batch contexts (e.g., crowd monitoring without individual tracking).

---

## 2. End-to-End Request Flow

### 2.1 Enroll (webcam capture or file upload)

```
Browser / curl
    │
    │  POST /api/enroll/frame   { subject_id, image_b64 }   ← webcam
    │     ─ or ─
    │  POST /api/enroll         multipart { subject_id, image }   ← file
    ▼
flask_app.py  _embed()
    │
    │  1. Decode JPEG bytes → PIL.Image (RGB, any resolution)
    │  2. Resize to 112 × 112  [PIL.BILINEAR — no NumPy bridge]
    │  3. tobytes() → bytearray → torch.frombuffer()
    │  4. float / 255 → normalise [-1, 1] → (C, H, W) tensor
    ▼
BackboneEmbedder.forward()                        backbone.py
    │
    │  ResNet-50 body  (ImageNet pretrained, frozen)
    │        ↓
    │  Linear(2048 → 512) + BatchNorm1d            ← trainable head
    │        ↓
    │  F.normalize(·, p=2)                         ← unit-sphere output
    │
    │  Output: 512-D float32 tensor
    ▼
tensor.tolist() → np.array()                      (bridge-free numpy)
    ▼
EmbeddingStore.add(subject_id, embedding)         embedding_store.py
    │
    │  key   = HMAC-SHA256(subject_id, salt)   ← no plaintext ID stored
    │  proj  = embedding @ P                   ← random orthogonal matrix
    │  row   → SQLite BLOB  (optional Fernet encryption)
    ▼
{ status: "enrolled", total_enrolled: N }
```

### 2.2 Verify (live webcam loop or file upload)

```
Browser (250 ms tick) / curl
    │
    │  POST /api/verify/live   { subject_id, image_b64 }   ← webcam
    │     ─ or ─
    │  POST /api/verify        multipart { subject_id, image }   ← file
    ▼
flask_app.py  _embed()       (same 4-step pipeline as enroll)
    ▼
np.dot(query_emb, store.get(subject_id))
    │
    │  Both embeddings are L2-normalised
    │  → dot product = cosine similarity ∈ [-1, 1]
    │  match = similarity ≥ 0.0
    ▼
data_provider.record_event("verify", ...)         data_provider.py
    ▼
{ similarity, match, latency_ms }

Browser updates:
    • green/red overlay on video stream
    • similarity progress bar
    • persistent result strip (stays after camera stops)
```

### 2.3 Batch Frame Verify + Access Control Correlation

```
POST /api/verify/frame   multipart { image, k?, thresh?, camera_id?, door_id? }
    │
    ▼
face_utils.detect_faces()            ← MTCNN or HOG, returns bboxes
    │
    ▼ (for each face)
BackboneEmbedder + EmbeddingStore.query_topk(k)
    │
    ├─ attendance_store.mark()        ← session-TTL dedup (PCS_SESSION_TTL s)
    │
    └─ AccessControlEngine.on_verify_result(camera_id, door_id, subject_key,
                                             matched, similarity, timestamp_utc)
               │
               ├─ correlate with door_buf  →  forced_entry alert?
               └─ scan verify_buf window   →  tailgate alert?

{ n_faces, faces[{face_id, bbox, top_matches}], latency_ms }
```

### 2.4 Door Sensor → Access Control Alert

```
MQTT broker  topic: campus/<door_id>/event
    payload: { "type": "open"|"close", "door_id": "...", "timestamp": "<ISO-8601>" }
    │
    ▼
_start_door_subscriber() (daemon thread, started at app init)
    │
    ├─ resolves camera_id from CameraRegistry (by door_id)
    │
    └─ AccessControlEngine.on_door_event(door_id, event_type, timestamp_utc, camera_id)
               │
               ├─ correlate with verify_buf  →  forced_entry alert?
               └─ prune stale events (TTL 30 s relative to arriving event's timestamp)

alert fired
    │
    ├─ AlertStore.add()               ← persists to SQLite `alerts` table
    ├─ socketio.emit("alert", …)      ← broadcast to all dashboard clients (toast + table row)
    ├─ socketio.emit("camera_update", …status="alert"…)  ← icon turns red on campus map
    └─ data_provider.record_event()   ← appears in dashboard event feed
```

### 2.4b Camera Frame Subscriber

```
MQTT broker  topic: campus/+/frame
    payload: raw JPEG bytes
    │
    ▼
_start_frame_subscriber() (daemon thread, started at app init)
    │
    └─ _camera_frames[camera_id] = bytes(msg.payload)   ← latest JPEG per camera

GET /api/camera/<camera_id>/snapshot
    └─ returns _camera_frames[camera_id] as image/jpeg, or 404
```

### 2.4c Alert Acknowledge (Phase 5)

```
Browser  POST /api/alerts/<id>/ack  {acked_by: "Officer Smith"}
    ▼
_alert_store.acknowledge(alert_id, acked_by=acked_by)
    │  sets acknowledged=1, acked_by="Officer Smith", acked_at=<utcnow ISO-8601>
    │
    └─ socketio.emit("alert_ack", full_alert_dict)   ← broadcast to all admin clients
           │
           └─ Dashboard JS: updates row Status cell to show ✓ + operator + timestamp
```

### 2.5 Dashboard load

```
Browser  GET /
    ▼
flask_app.py  index()
    ▼
data_provider.get_metrics()      (60-s TTL in-memory cache)
    │
    ├─ experiments/lfw_results/kshot_metrics.json  → K-shot accuracy table
    ├─ experiments/lfw_results/pairs_metrics.json  → ROC-AUC, TAR@FAR
    └─ experiments/anomaly_scores.csv              → anomaly timeline
    ▼
render_template("dashboard.html", **metrics)
    ▼
Bootstrap 5 + Chart.js 4 + Socket.IO served
    │
    └─ on page load: _loadAlerts() fetches /api/alerts → populates Security Alerts table
                     socket.on("alert") → toast popup + prepend row + update badge
                     _camMonitor.start() → Camera Monitor thumbnail grid (see §2.6)
```

### 2.6 Camera Monitor thumbnail panel

```
Browser  GET /   (dashboard page load)
    │
    ▼ _camMonitor.start()   (IIFE, auto-starts on DOMContentLoaded)
    │
    ├─ GET /api/cameras       → [{camera_id, name, door_id, …}]   (fetched once)
    ├─ GET /api/camera/status → {camera_id: {status, last_seen,
    │                                        last_similarity, last_matched,
    │                                        recent_events[]}}     (polled every 5 s)
    │
    └─ For each camera, renders a card with:
         • Lazy-loaded JPEG snapshot from GET /api/camera/<id>/snapshot?_t=<epoch>
             (HTTP 200 → displayed; 404 → placeholder icon shown instead)
         • Coloured status stripe: green=idle / amber=activity / red=alert
         • Top-3 recent-event chips overlaid on the thumbnail:
             verify-match chip  (green) – similarity score
             verify-nomatch chip (red)
             alert chip (red)   – alert_type label
         • Similarity progress bar in the card footer:
             width = ((last_similarity + 1) / 2) × 100 %
             colour = green if last_matched else red
         • Bootstrap tooltip on hover: "Last seen: <HH:MM:SS>"
         • Cards update on socket.io "camera_update" event (no full re-render)

Backend update path (flask_app.py):
    POST /api/verify/frame (with camera_id)
        └─ updates _camera_status[camera_id]:
               last_similarity, last_matched, recent_events (max 3, FIFO)
               status = "activity" (unless already "alert")

    _on_alert() callback
        └─ preserves existing recent_events, appends alert chip entry
               status = "alert"
```

---

## 3. Module Map

```
src/
│
├─ cli/run.py              ← entry point — 11 subcommands (§4)
│
├─ data/lfw_loader.py      ← LFW dataset (torchvision.datasets.LFWPairs)
│                             DEFAULT_TRANSFORM: resize 112 + norm [-1,1]
│                             --download flag for data fetching
│
├─ models/
│   ├─ backbone.py         ← BackboneEmbedder
│   │                         ResNet-50 body + Linear(2048→512) + L2-norm
│   ├─ prototypical.py     ← ProtoNetConfig + ProtoNetTrainer
│   │                         N-way K-shot episodic training
│   └─ embedding_utils.py  ← save_embeddings() batch-cache helper
│                             used by compute-embeddings subcommand
│
├─ eval/evaluate_lfw.py    ← LFW pairs + few-shot evaluation
│                             outputs JSON metrics + ROC PNG
│
├─ privacy/
│   ├─ embedding_store.py  ← privacy-preserving store
│   │                         HMAC-SHA256 keys, orthogonal projection
│   │                         SQLite backend, Fernet encryption (opt.)
│   │                         fast query_topk via in-memory np cache
│   ├─ attendance_store.py ← AttendanceStore (SQLite `attendance` table)
│   │                         mark() with session-TTL dedup
│   │                         query_aggregated() → per_subject, per_camera
│   ├─ camera_store.py     ← CameraRegistry (YAML) + CameraStore (SQLite)
│   │                         loads configs/cameras.yaml
│   │                         derives MQTT frame/sensor/event topics
│   └─ access_control.py  ← AlertStore (SQLite `alerts` table)
│                            AccessControlEngine
│                              on_door_event()    → forced_entry?
│                              on_verify_result() → forced_entry? tailgate?
│                            Thread-safe; prune uses event-relative TTL
│
└─ web/
    ├─ flask_app.py        ← Flask + SocketIO server (26 REST routes + 14 WS events)
    ├─ face_utils.py       ← multi-face detection + batch embedding
    ├─ data_provider.py    ← metrics cache (60 s TTL) + event ring buffer
    └─ templates/
        ├─ dashboard.html  ← Bootstrap 5, Chart.js 4, Security Alerts panel,
        │                     Camera Monitor thumbnail grid (lazy-loaded JPEG
        │                     snapshots, top-3 event chips, similarity bar, hover
        │                     tooltip with last-seen timestamp), Socket.IO live
        │                     updates + toast notifications
        ├─ enroll_screen.html ← single-screen camera-grid enrollment UI
        │                       MQTT arm/disarm, webcam capture, Socket.IO
        └─ campus_map.html ← Leaflet.js campus overview
                              CartoDB dark tiles, custom SVG DivIcon markers
                              per-camera status colours (idle/activity/alert)
                              side drawer with event log per camera
```

---

## 4. CLI Subcommands (`src/cli/run.py`)

| Subcommand | What it does |
|---|---|
| `prepare-data` | Download LFW via torchvision |
| `compute-embeddings` | Run backbone over dataset → save `.pt` embedding cache |
| `train-protonet` | Episodic ProtoNet training; saves checkpoint + learning-curve plot |
| `eval-lfw` | Pairs accuracy + F1 + AUC + ROC; few-shot episode accuracy |
| `enroll` | Enroll one subject image into an EmbeddingStore |
| `enroll-batch` | Detect all faces in image, enroll each with anonymous ephemeral key |
| `verify` | Verify one query image against a stored subject |
| `eval` | End-to-end evaluation report (Markdown + PNG + PDF) |
| `serve` | Start the Flask/SocketIO dashboard server |
| `simulate-camera` | Publish JPEG frames to `campus/<camera_id>/frame` over MQTT |
| `simulate-door` | Publish open/close events to `campus/<door_id>/event` over MQTT |

---

## 5. Privacy Model

```
  subject_id  ──HMAC-SHA256(·, salt)──►  key  (stored in DB)
                                           ↑
                                   no plaintext ID on disk

  embedding (512-D)  ──·@ P──►  projected embedding  (stored in DB)
                                           ↑
                               random orthogonal P is never persisted
                               → protects against embedding inversion
```

**Threat model:**

| Threat | Mitigation |
|---|---|
| DB breach exposes identities | HMAC-SHA256 keyed by env-var salt; IDs unrecoverable without salt |
| Embedding inversion recovers faces | Orthogonal projection + L2-norm; reconstruction success target < 5 % |
| Store file intercepted in transit | Optional Fernet encryption (`PCS_ENCRYPT=1`) |
| Salt extracted from source code | Salt lives only in `PCS_EMBEDDING_SALT` env var, never in code |

---

## 6. REST API

### Core

| Method | Endpoint | Body / Params | Response |
|---|---|---|---|
| GET | `/` | — | Full dashboard HTML |
| GET | `/enroll/screen` | — | Camera-grid enrollment UI HTML |
| GET | `/admin/campus` | — | Campus map HTML |
| GET | `/api/health` | — | `{status, enrolled, uptime_s}` |
| GET | `/api/events` | `?n=50` | Last N event objects |
| GET | `/api/metrics` | — | K-shot rows, pairs metrics, anomaly series |
| GET | `/api/metrics/reload` | — | Force cache refresh, returns updated data |
| POST | `/api/enroll` | form: `subject_id`, `image` | `{status, subject_id, total_enrolled}` |
| POST | `/api/enroll/batch` | form: `image`, `group_id?` | `{n_enrolled, keys[], total_enrolled}` |
| POST | `/api/verify` | form: `subject_id`, `image` | `{similarity, match}` |
| POST | `/api/enroll/frame` | JSON: `{subject_id, image_b64}` | `{status, subject_id, total_enrolled}` |
| POST | `/api/verify/live` | JSON: `{subject_id, image_b64}` | `{similarity, match, latency_ms}` |
| POST | `/api/verify/frame` | form: `image`, `k?`, `thresh?`, `camera_id?`, `door_id?` | `{n_faces, faces[], latency_ms}` |

### Attendance

| Method | Endpoint | Body / Params | Response |
|---|---|---|---|
| POST | `/api/attendance/mark` | JSON: record or `[records]` with `subject_key, similarity, matched` | `{n_marked, n_skipped, total_attendance, session_ttl_s}` |
| GET | `/api/attendance` | `?from`, `?to`, `?group_id` | `{total_marks, matched_marks, per_subject[], per_camera[]}` |

### Cameras

| Method | Endpoint | Response |
|---|---|---|
| GET | `/api/cameras` | Array of camera objects with `frame_topic`, `sensor_topic`, coords |
| GET | `/api/camera/status` | `{camera_id: {status, last_event, last_seen, last_similarity, last_matched, recent_events[]}}` |

`status` values: `"idle"` (green), `"activity"` (amber), `"alert"` (red)

`recent_events` contains up to 3 of the most recent per-camera events:
```json
[
  {"ts": "2026-03-01T12:00:01Z", "type": "verify", "matched": true,  "similarity": 0.85},
  {"ts": "2026-03-01T12:00:05Z", "type": "alert",  "alert_type": "forced_entry"}
]
```
These are consumed by the **Camera Monitor** thumbnail panel on the dashboard
to overlay event chips and render the live similarity progress bar.

### Alerts & Camera Snapshots

| Method | Endpoint | Params / Body | Response |
|---|---|---|---|
| GET | `/api/alerts` | `n`, `alert_type`, `door_id`, `unacked_only=1` | `{alerts[], total_unacked}` |
| POST | `/api/alerts/<id>/ack` | JSON body `{acked_by?}` | `{acknowledged, id, acked_by, acked_at}` or 404 |
| GET | `/api/camera/<id>/snapshot` | — | Latest JPEG frame (from `_camera_frames` MQTT store) or 404 |

Alert object shape:
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

---

## 7. WebSocket Events (Socket.IO)

### Client → Server

| Event | Payload | Purpose |
|---|---|---|
| `connect` | — | Session open; server emits back `{sid, enrolled}` |
| `arm_enroll` | `{camera_id}` | Mark camera as active for enrollment |
| `disarm_enroll` | `{camera_id?}` | Cancel enrollment mode (empty = all cameras for this session) |
| `enroll_frame` | `{camera_id, image_b64, group_id?}` | Submit webcam frame; server detects + batch-enrolls |
| `frame` | `{image_b64, mode?}` | Generic frame for enroll/verify workflows on main dashboard |
| `verify` | `{subject_id, image_b64}` | One-shot 1:1 verify request |
| `get_attendance` | `{}` | Request a fresh attendance snapshot |

### Server → Client (broadcast)

| Event | Payload | Purpose |
|---|---|---|
| `camera_status` | `{camera_id, armed, armed_by?}` | Arm/disarm notification to all clients |
| `camera_update` | `{camera_id, status, last_event, last_seen}` | Real-time status colour for campus map |
| `enroll_result` | `{status, camera_id, n_enrolled, faces[], total_enrolled, timestamp}` | Enrollment outcome |
| `verify_result` | `{similarity, match, …}` | Verification outcome |
| `attendance_update` | `{records[]}` | Fresh attendance snapshot |
| `arm_ack` | `{camera_id, armed}` | Acknowledge arm/disarm to the initiating client |
| `alert` | `{id, alert_type, door_id, camera_id, subject_key, confidence, timestamp_utc, meta, acknowledged, acked_by, acked_at}` | New security alert broadcast |
| `alert_ack` | Full alert object (same shape as `alert`) | Alert acknowledged — full object so clients render audit trail without extra fetch |

---

## 8. Access Control Engine

```
src/privacy/access_control.py
```

### Alert rules

| Alert type | Trigger condition | Detection window |
|---|---|---|
| `forced_entry` | Door `open` event + `matched == false` verify at same door | ± 2 s |
| `tailgate` | ≥ 2 persons at same door, ≥ 1 with `matched == false` | 5 s sliding |

### Internal data flow

```
on_door_event(door_id, "open", ts, camera_id)
    │
    ├─ append to door_buf
    ├─ _prune(door_buf, reference_dt=ts)   ← TTL relative to event, not wall clock
    ├─ scan verify_buf for same door within ±2 s with matched==False
    │       → create forced_entry alert if found + deduplicate on (door_id, open_ts_str)
    └─ return alert dict or None

on_verify_result(camera_id, door_id, subject_key, matched, similarity, ts)
    │
    ├─ append to verify_buf
    ├─ _prune(verify_buf, reference_dt=ts)
    ├─ if not matched and door_id given:
    │       scan door_buf for "open" at same door within ±2 s
    │       → forced_entry alert if found
    └─ _check_tailgate(door_id, ts)
           scan verify_buf for same door in [ts - 5s, ts]
           → tailgate if ≥ 2 distinct subjects with ≥ 1 unmatched
```

### Key design decisions

| Decision | Reason |
|---|---|
| `_prune()` uses `reference_dt` (event timestamp), not `datetime.now()` | Correct with replayed/historical events; predictable in unit tests |
| `_used_door_opens: set[tuple[str,str]]` deduplication | Prevents the same door `open` from triggering multiple `forced_entry` alerts |
| `_used_tailgate: set[str]` keyed on frozenset of subjects + door | Prevents duplicate tailgate alerts for the same group crossing |
| Thread-safe with `threading.Lock()` | Door subscriber daemon thread and Flask request threads share state |
| Alert callback (`on_alert`) is exception-safe | A crashing callback does not prevent the alert from being persisted |

---

## 9. SQLite Schema

All tables coexist in four separate SQLite files (configurable via env vars).
Canonical schemas are defined in `migrations/` and applied via `make migrate`.

### `data/embeddings.db` (`PCS_STORE`)
```sql
CREATE TABLE embeddings (
    hashed_id  TEXT    PRIMARY KEY,   -- HMAC-SHA256(subject_id, salt)
    embedding  BLOB    NOT NULL,       -- float32 bytes, optionally Fernet-encrypted
    dim        INTEGER NOT NULL
);
```

### `data/attendance.db` (`PCS_ATTENDANCE_DB`)
All operational tables share this file so foreign-key joins can work in a
single SQLite connection and the server needs only one persistent handle.

```sql
CREATE TABLE attendance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_key   TEXT    NOT NULL,
    group_id      TEXT    NOT NULL DEFAULT '',
    camera_id     TEXT    NOT NULL DEFAULT 'default',
    timestamp_utc TEXT    NOT NULL,
    similarity    REAL    NOT NULL,
    matched       INTEGER NOT NULL
);

CREATE TABLE cameras (
    camera_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    door_id       TEXT,
    lat           REAL NOT NULL DEFAULT 0.0,
    lon           REAL NOT NULL DEFAULT 0.0,
    active        INTEGER NOT NULL DEFAULT 1,
    last_seen_utc TEXT,
    meta_json     TEXT
);

CREATE TABLE alerts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type    TEXT    NOT NULL,   -- forced_entry | tailgate | safety_alert
    camera_id     TEXT,
    door_id       TEXT,
    subject_key   TEXT,
    confidence    REAL    NOT NULL DEFAULT 1.0,
    timestamp_utc TEXT    NOT NULL,
    meta_json     TEXT,
    acknowledged  INTEGER NOT NULL DEFAULT 0,
    acked_by      TEXT,
    acked_at      TEXT
);

CREATE TABLE notifications (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id      INTEGER,
    alert_type    TEXT    NOT NULL,
    severity      TEXT    NOT NULL,
    channel       TEXT    NOT NULL,   -- websocket | smtp | sms
    status        TEXT    NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_attempt  TEXT,
    next_retry    TEXT,
    error_msg     TEXT,
    payload_json  TEXT,
    created_at    TEXT    NOT NULL
);

CREATE TABLE loitering_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id      TEXT    NOT NULL,
    subject_key    TEXT    NOT NULL,
    first_seen_utc TEXT    NOT NULL,
    detected_utc   TEXT    NOT NULL,
    dwell_s        REAL    NOT NULL DEFAULT 0,
    dismissed      INTEGER NOT NULL DEFAULT 0,
    action_by      TEXT,
    action_at      TEXT
);

CREATE TABLE left_object_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id      TEXT    NOT NULL,
    bbox_json      TEXT,
    first_seen_utc TEXT    NOT NULL,
    detected_utc   TEXT    NOT NULL,
    dwell_s        REAL    NOT NULL DEFAULT 0,
    snapshot_b64   TEXT,
    dismissed      INTEGER NOT NULL DEFAULT 0,
    confirmed      INTEGER NOT NULL DEFAULT 0,
    action_by      TEXT,
    action_at      TEXT
);
```

### `data/audit.db` (`PCS_AUDIT_DB`)
```sql
CREATE TABLE audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    actor         TEXT,
    subject_id    TEXT,
    camera_id     TEXT,
    alert_id      INTEGER,
    detail        TEXT    -- JSON-encoded detail dict
);

-- Rows are immutable — enforced by SQLite triggers
CREATE TRIGGER audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log rows are immutable'); END;

CREATE TRIGGER audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log rows are immutable'); END;
```

### `data/users.db` (`PCS_AUTH_DB`)
```sql
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,   -- bcrypt hash, cost factor 12
    role          TEXT    NOT NULL DEFAULT 'viewer',
    created_at    TEXT    NOT NULL,
    last_login    TEXT
);
```

---

## 10. MQTT Topics

| Topic | Direction | Payload | Description |
|---|---|---|---|
| `campus/<camera_id>/frame` | sensor → server | raw JPEG bytes | Video frame from camera |
| `campus/<camera_id>/sensor` | sensor → broker | `{camera_id, frame_no, ts_utc}` | Telemetry from simulate-camera |
| `campus/<door_id>/event` | sensor → server | `{type, door_id, timestamp}` | Door open/close from simulate-door |

Broker configuration is read from `configs/cameras.yaml` under the `mqtt:` key.
All subscriber connections use `paho-mqtt 2.x` `CallbackAPIVersion.VERSION2`.

---

## 11. Key Implementation Decisions

| Decision | Why |
|---|---|
| PIL → `tobytes()` → `torch.frombuffer()` | NumPy 2.x breaks the `torch.from_numpy` C-API bridge; pure-byte path works everywhere |
| `tensor.tolist()` → `np.array()` for output | Same bridge issue on the output side; Python list intermediary avoids it |
| SQLite over flat JSON | Atomic writes, no full-file rewrite on every enroll, scales to millions of rows |
| `np.argpartition` in `query_topk` | O(N) selection vs O(N log N) sort on the full store |
| In-memory embedding matrix cache + dirty flag | Avoids re-reading SQLite on every verify; invalidated automatically on any write |
| `async_mode="threading"` for SocketIO | No eventlet/gevent dependency; works with Flask's built-in dev server |
| Door subscriber as daemon thread | Does not block Flask startup; exits automatically when the process exits |
| Alert TTL relative to event timestamp | Prevents test flakiness and correct behaviour with replayed event streams |

---

## 12. DB Migrations & Backups

### 12.1 Migration runner (`scripts/migrate.py`)

The application uses a **stdlib-only lightweight migration runner** rather than
Alembic or Flyway.  Key design rationale:

| Rationale | Detail |
|---|---|
| **Multiple databases** | The app maintains four separate SQLite files (`embeddings`, `attendance`, `audit`, `users`); Alembic's single-database model would require four separate environments to configure and run |
| **No ORM / metadata** | The schemas are hand-written SQL; a declarative ORM would be a heavier dependency than justified for a research prototype |
| **Simplicity** | The runner is ~200 lines of stdlib Python with zero additional packages |
| **Idempotency** | Applied migration IDs are recorded in a `_schema_migrations` table in every managed database; re-running is always safe |

#### Runner design

```
migrations/
├─ embeddings/       → data/embeddings.db  (PCS_STORE)
│   ├─ 0001_initial_schema.sql
│   └─ NNNN_<description>.sql   ← new migrations added here
├─ attendance/       → data/attendance.db  (PCS_ATTENDANCE_DB)
│   ├─ 0001_initial_schema.sql  ← all tables in shared attendance DB
│   └─ 0002_alerts_add_camera_index.sql
├─ audit/            → data/audit.db       (PCS_AUDIT_DB)
│   └─ 0001_initial_schema.sql
└─ users/            → data/users.db       (PCS_AUTH_DB)
    └─ 0001_initial_schema.sql

scripts/migrate.py   ← runner entry-point
```

Each SQL file is applied inside a transaction then recorded.  The runner stops
at the first failure within a database and exits with code 1.

```
make migrate            # apply all pending
make migrate-status     # print applied / pending (read-only)
make migrate-dry-run    # preview without writing
```

#### `_schema_migrations` table (injected into every DB)

```sql
CREATE TABLE IF NOT EXISTS _schema_migrations (
    id          TEXT PRIMARY KEY,   -- e.g. "0001_initial_schema"
    applied_at  TEXT NOT NULL,      -- ISO-8601 UTC
    checksum    TEXT                -- SHA-256 of the SQL file
);
```

#### Adding a new migration

1. Create `migrations/<db_name>/NNNN_<description>.sql` (next number in sequence).
2. Run `make migrate` — the runner picks it up automatically.
3. Commit both the SQL file and the runner together in one PR.

### 12.2 Backup script (`scripts/backup.py`)

```
make backup      # local gzip backup of all four DBs, 7-day rotation
make backup-s3   # same + S3 upload (requires PCS_BACKUP_S3_BUCKET + boto3)
```

#### Backup flow

```
scripts/backup.py
    │
    ├─ for each database (embeddings / attendance / audit / users):
    │       sqlite3.Connection.backup(dst)   ← online hot copy (WAL-safe)
    │       gzip.open(dst + ".gz")           ← compress (optional)
    │       boto3.upload_file(…)             ← upload to S3 (optional)
    │
    └─ rotate: delete data/backups/<YYYY-MM-DD>/ dirs older than PCS_BACKUP_RETAIN days
```

#### Backup directory layout

```
data/backups/
└─ 2026-03-01/
    ├─ embeddings_20260301T120000Z.db.gz
    ├─ attendance_20260301T120000Z.db.gz
    ├─ audit_20260301T120000Z.db.gz
    └─ users_20260301T120000Z.db.gz
```

#### S3 key layout

```
<PCS_BACKUP_S3_PREFIX>/<db_name>/<YYYY-MM-DD>/<filename>
e.g. pcs-backups/attendance/2026-03-01/attendance_20260301T120000Z.db.gz
```

#### Key design decisions

| Decision | Reason |
|---|---|
| `sqlite3.Connection.backup()` (online backup API) | Safe under concurrent writes; no need to stop/lock the server |
| WAL journal mode on all databases | Readers (backup process) never block writers (Flask threads) |
| Separate `YYYY-MM-DD` sub-directories | One `rm -rf` per expired day — no prefix-scan needed for rotation |
| `boto3` is imported lazily | S3 upload is **optional**; the backup script works without boto3 installed |
| `PCS_BACKUP_RETAIN=0` disables rotation | Useful in CI or ephemeral environments where rotation is unwanted |
