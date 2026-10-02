# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-powered real-time face tracking and visitor management system that detects, tracks, recognizes, and automatically registers visitors while maintaining persistent identities and recording ENTRY/EXIT events.

The system combines **YOLO face detection, ByteTrack tracking, InsightFace recognition, ArcFace embeddings, SQLite persistence, and structured event logging**.

---

## 🚀 Project Overview

The **Intelligent Face Tracker** processes a video stream and identifies unique visitors in real time.

When a person enters the camera view:

1. Their face is detected.
2. ByteTrack assigns a temporary `TRACK_ID`.
3. InsightFace generates a face embedding.
4. The system searches existing registered visitors.
5. If the visitor is known, their existing `FACE_ID` is reused.
6. If the visitor is unknown, the system automatically registers a new `FACE_ID`.
7. An `ENTRY` event is stored with a timestamp and cropped face image.
8. The visitor remains tracked while visible.
9. When the visitor disappears for the configured number of frames, an `EXIT` event is generated.
10. The visitor's history remains available in SQLite.

The system is designed to prevent the same visitor from being counted multiple times when they are re-identified.

---

## 🎯 Problem Statement

Traditional people-counting systems often count detections rather than actual individuals.

For example:

```text
Person enters
      ↓
TRACK_ID = 1
      ↓
Person temporarily leaves camera view
      ↓
TRACK_ID = 8
      ↓
Naive counter → 2 visitors ❌
```

The Intelligent Face Tracker instead maintains a persistent identity:

```text
TRACK_ID = 1 ──┐
               ├──> FACE_0001
TRACK_ID = 8 ──┘
```

Therefore, the same person can be recognized again without creating a duplicate visitor identity.

---

# ✨ Key Features

### Face Detection

* YOLO-based face detection
* Configurable confidence threshold
* Configurable image size

### Face Tracking

* ByteTrack integration
* Persistent short-term `TRACK_ID`
* Missing-track handling
* Configurable missing-frame threshold

### Face Recognition

* InsightFace
* ArcFace-based embeddings
* Normalized face embeddings
* Cosine similarity matching
* Configurable recognition threshold

### Automatic Registration

Unknown faces are automatically registered after multiple recognition attempts.

Current process:

```text
Unknown Face
     ↓
Recognition Attempt 1
     ↓
Recognition Attempt 2
     ↓
Recognition Attempt 3
     ↓
Average Embedding
     ↓
FACE_XXXX Registration
```

### Visitor Counting

* Persistent `FACE_ID`
* Unique visitor tracking
* Re-identification support
* No duplicate count for the same persistent identity

### ENTRY / EXIT Logging

Every visitor appearance generates:

```text
1 ENTRY
1 EXIT
```

for the corresponding tracking session.

Each event includes:

* `FACE_ID`
* `TRACK_ID`
* timestamp
* event type
* confidence
* cropped face image path

### Database

SQLite stores:

* visitor identities
* embeddings
* events
* tracking sessions
* first/last seen timestamps

### Event Log

Human-readable application logs are stored in:

```text
logs/events.log
```

### Image Logging

Face crops are saved for ENTRY and EXIT events.

### RTSP Support

The same pipeline can process:

* local video files
* RTSP camera streams

### Configuration

Important parameters are controlled through:

```text
config.json
```

---

# 🧠 System Architecture

```text
                         VIDEO SOURCE
                              |
                    +---------+---------+
                    |                   |
                    v                   v
              OpenCV Capture       RTSP Camera
                    |                   |
                    +---------+---------+
                              |
                              v
                    YOLO Face Detection
                              |
                              v
                       ByteTrack
                              |
                        TRACK_ID
                              |
                              v
                    InsightFace Analysis
                              |
                       Face Embedding
                              |
                              v
                    Visitor Manager
                              |
                 +------------+------------+
                 |                         |
            Known Visitor            Unknown Visitor
                 |                         |
                 v                         v
          Existing FACE_ID          3 Recognition Attempts
                                           |
                                           v
                                    Average Embedding
                                           |
                                           v
                                      New FACE_ID
                 |                         |
                 +------------+------------+
                              |
                              v
                       ENTRY / EXIT
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
           SQLite        events.log       Face Images
```

---

# 🔑 TRACK_ID vs FACE_ID

The system intentionally uses two different identifiers.

## TRACK_ID

`TRACK_ID` is generated by ByteTrack.

It represents a temporary tracked object.

Example:

```text
TRACK_ID = 7
```

A new tracking ID may be assigned when the same person reappears.

## FACE_ID

`FACE_ID` represents the persistent identity stored in the database.

Example:

```text
FACE_0001
FACE_0002
FACE_0003
```

Example:

```text
TRACK_ID 7
     ↓
FACE_0001

Person disappears

TRACK_ID 14
     ↓
FACE_0001
```

The persistent `FACE_ID` prevents the same visitor from being treated as a new person simply because their tracking ID changed.

---

# 🔍 Face Recognition

InsightFace is used to generate face embeddings.

The embeddings are normalized before comparison.

The system compares a new embedding against stored visitor embeddings using similarity matching.

Current configuration:

```json
"recognition": {
    "similarity_threshold": 0.45,
    "embedding_update_interval": 30
}
```

The current visitor manager uses multiple observations when registering a new face.

---

# 🆕 Automatic Face Registration

When a face cannot be matched against the existing database, the system does not immediately register it.

Instead, it collects multiple recognition attempts.

```text
Attempt 1
   ↓
Attempt 2
   ↓
Attempt 3
   ↓
Embedding Averaging
   ↓
Normalization
   ↓
Database Registration
   ↓
FACE_XXXX
```

This reduces the chance of creating a visitor identity from a single poor-quality observation.

---

# 🚪 ENTRY / EXIT Processing

## ENTRY

When a new tracking session is successfully assigned to a `FACE_ID`, the system:

1. Creates an ENTRY timestamp.
2. Saves a cropped face image.
3. Inserts an ENTRY event into SQLite.
4. Creates an active tracking session.
5. Writes an ENTRY message to `events.log`.

The application prevents repeated ENTRY events for the same active tracking session.

---

## EXIT

When a visitor disappears from the camera view, the application does not immediately generate an EXIT.

Instead, it waits for:

```text
max_missing_frames = 50
```

If the visitor remains missing for the configured period:

1. EXIT timestamp is created.
2. Latest available face crop is saved.
3. EXIT event is inserted into SQLite.
4. Tracking session is closed.
5. EXIT event is written to `events.log`.

This reduces false EXIT events caused by temporary detection loss.

---

# 👥 Unique Visitor Counting

Unique visitors are identified using persistent `FACE_ID` values.

Example:

```text
TRACK_ID 1  → FACE_0001
TRACK_ID 5  → FACE_0002
TRACK_ID 8  → FACE_0001
TRACK_ID 12 → FACE_0003
```

Unique visitors:

```text
FACE_0001
FACE_0002
FACE_0003
```

Total:

```text
3 unique visitors
```

The reappearance of `FACE_0001` does not create another unique visitor.

---

# 🗄️ Database

The application uses SQLite.

Database location:

```text
database/visitors.db
```

## `persons`

Stores persistent visitor identities.

```text
id
face_id
first_seen
last_seen
embedding
created_at
```

## `events`

Stores ENTRY and EXIT events.

```text
id
face_id
track_id
event_type
timestamp
image_path
confidence
```

## `tracks`

Stores tracking sessions.

```text
id
face_id
track_id
start_time
end_time
status
```

---

# 📁 Project Structure

```text
intelligent-face-tracker/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── detector.py
│   ├── tracker.py
│   ├── recognizer.py
│   ├── visitor_manager.py
│   ├── database.py
│   ├── event_logger.py
│   └── utils.py
│
├── models/
│   └── yolov11n-face.pt
│
├── input/
│   └── sample.mp4
│
├── output/
│   └── annotated/
│
├── logs/
│   ├── entries/
│   ├── exits/
│   └── events.log
│
├── database/
│   └── visitors.db
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_events.py
│   └── test_visitor_manager.py
│
├── docs/
│   ├── architecture.md
│   ├── AI_PLANNING.md
│   └── COMPUTE.md
│
├── config.json
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

| Component        | Technology                 |
| ---------------- | -------------------------- |
| Language         | Python                     |
| Face Detection   | YOLO                       |
| Object Tracking  | ByteTrack                  |
| Face Recognition | InsightFace                |
| Face Embeddings  | ArcFace                    |
| Computer Vision  | OpenCV                     |
| Database         | SQLite                     |
| Configuration    | JSON                       |
| Testing          | Pytest                     |
| Input            | MP4 / RTSP                 |
| Logging          | Python logging / file logs |

---

# ⚙️ Configuration

Configuration is controlled through `config.json`.

Example:

```json
{
    "video": {
        "source": "input/sample.mp4",
        "output": "output/annotated/output.mp4"
    },

    "detection": {
        "model": "models/yolov11n-face.pt",
        "confidence": 0.5,
        "iou": 0.5,
        "image_size": 1280,
        "frame_skip": 3
    },

    "recognition": {
        "similarity_threshold": 0.45,
        "embedding_update_interval": 30
    },

    "tracking": {
        "tracker": "bytetrack.yaml",
        "max_missing_frames": 50
    },

    "hardware": {
        "device": "cpu"
    },

    "database": {
        "path": "database/visitors.db"
    },

    "logging": {
        "log_file": "logs/events.log",
        "save_entry_images": true,
        "save_exit_images": true
    },

    "camera": {
        "type": "file",
        "rtsp_url": ""
    }
}
```

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd intelligent-face-tracker
```

## 2. Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# 📦 Install Dependencies

```powershell
pip install -r requirements.txt
```

The project uses the major dependencies required for:

* Ultralytics
* OpenCV
* InsightFace
* ONNX Runtime
* NumPy
* Pytest

---

# ▶️ Run the Application

Make sure the sample video exists:

```text
input/sample.mp4
```

Make sure the YOLO model exists:

```text
models/yolov11n-face.pt
```

Run:

```powershell
python -m app.main
```

The application processes the video and produces an annotated output.

---

# 🎥 RTSP Camera

To use an RTSP camera, modify:

```json
"camera": {
    "type": "rtsp",
    "rtsp_url": "rtsp://YOUR_CAMERA_URL"
}
```

Then run:

```powershell
python -m app.main
```

The same detection, tracking, recognition, visitor management, database, and logging pipeline is used.

---

# 📊 Output

The application produces:

### Annotated Video

```text
output/annotated/
```

### ENTRY Images

```text
logs/entries/
```

### EXIT Images

```text
logs/exits/
```

### Event Log

```text
logs/events.log
```

### Database

```text
database/visitors.db
```

---

# 🧪 Testing

The project contains automated tests for the major components.

Run:

```powershell
pytest -v
```

Final validation:

```text
21 passed
```

The tests cover areas including:

* configuration
* database operations
* visitor registration
* event creation
* duplicate ENTRY prevention
* duplicate EXIT prevention
* track updates
* missing tracks
* visitor management

---

# ✅ Validation Results

A sample video run was validated against the SQLite database.

Result:

```text
DATABASE VALIDATION
================================
ENTRY events      : 12
EXIT events       : 12
Unique visitors   : 9
```

This demonstrates that the tested run produced:

* 12 ENTRY events
* 12 EXIT events
* 9 persistent visitor identities

The application also passed the automated test suite:

```text
21 passed
```

---

# ⚡ Performance

The current validated configuration uses CPU inference.

```text
YOLO Device              : CPU
InsightFace Provider     : CPUExecutionProvider
Frame Skip               : 3
YOLO Image Size          : 1280
YOLO Confidence          : 0.5
```

Frame skipping is used to reduce the frequency of expensive computer-vision processing.

The application also includes processing-frequency instrumentation for performance evaluation.

Exact FPS, RAM consumption, and CPU utilization are intentionally not hard-coded into the documentation because they depend on the machine and test conditions.

More details are available in:

```text
docs/COMPUTE.md
```

---

# 🤖 AI-Assisted Development

AI tools were used during development for:

* requirement decomposition
* architecture planning
* modular code generation
* debugging assistance
* automated test generation
* configuration design
* documentation
* interview preparation

AI-generated code was not treated as automatically correct.

The development process was:

```text
Requirement
    ↓
AI-Assisted Planning
    ↓
Code Generation
    ↓
Manual Review
    ↓
Application Execution
    ↓
Debugging
    ↓
Automated Testing
    ↓
Manual Validation
    ↓
Final Implementation
```

Development details are documented in:

```text
docs/AI_PLANNING.md
```

---

# 📐 Architecture Documentation

Detailed architecture:

```text
docs/architecture.md
```

Compute and performance information:

```text
docs/COMPUTE.md
```

AI-assisted development process:

```text
docs/AI_PLANNING.md
```

---

# 🔐 Design Decisions

### Why YOLO?

YOLO provides fast object detection suitable for real-time video processing.

### Why ByteTrack?

ByteTrack provides temporary tracking identities across video frames.

### Why InsightFace?

InsightFace provides face analysis and high-quality face embeddings suitable for identity matching.

### Why SQLite?

SQLite is lightweight, local, and sufficient for the current single-application architecture.

### Why separate TRACK_ID and FACE_ID?

Tracking IDs are temporary, while visitor identities need to persist.

### Why multiple registration attempts?

Multiple observations provide more reliable information than immediately registering an unknown face from a single frame.

### Why frame skipping?

It reduces expensive processing frequency and allows the application to operate more efficiently on CPU hardware.

---

# ⚠️ Limitations

The current implementation has several limitations.

### Lighting

Poor lighting can reduce face detection and recognition quality.

### Occlusion

Heavy face occlusion can make recognition unreliable.

### Extreme Angles

Very large head rotations can reduce embedding quality.

### CPU Performance

CPU-only inference limits throughput compared with GPU-based processing.

### Single-Application Storage

SQLite is suitable for the current architecture but is not intended as the final database for a large distributed deployment.

### Camera Placement

Camera angle, distance, resolution, and field of view can significantly affect detection and recognition performance.

---

# 🔮 Future Improvements

Potential future improvements include:

* GPU-accelerated inference
* multi-camera support
* centralized database
* REST API
* web dashboard
* visitor analytics
* camera-specific identities
* improved face-quality filtering
* stronger re-identification strategies
* log rotation
* configurable image retention
* cloud/object storage
* Docker deployment
* production monitoring
* authentication and access control

---

# 📈 Scalability

The current architecture can be extended from:

```text
Single Camera
      ↓
Single Application
      ↓
SQLite
```

to:

```text
Multiple Cameras
       ↓
Processing Services
       ↓
Central API
       ↓
Central Database
       ↓
Dashboard / Analytics
```

The modular separation of detection, tracking, recognition, visitor management, and persistence makes this extension easier.

---

# 🎬 Demo

### Demo Video

Add your Loom or YouTube demonstration link here:

```text
DEMO_LINK_HERE
```

The recommended demo should show:

1. Application startup
2. Face detection
3. TRACK_ID assignment
4. FACE_ID assignment
5. New visitor registration
6. Existing visitor recognition
7. ENTRY event
8. Visitor tracking
9. EXIT event
10. SQLite validation
11. Event log
12. Unique visitor count

---

# 🗃️ Sample Evidence

Recommended GitHub evidence:

```text
sample_output/
├── entry_images/
├── exit_images/
├── events.log
└── database_validation.txt
```

This allows reviewers to inspect actual system output without needing to reproduce the complete environment immediately.

---

# 📋 Hackathon Submission Checklist

Before submission:

* [ ] GitHub repository is public
* [ ] `README.md` is complete
* [ ] `requirements.txt` is included
* [ ] `config.json` is included
* [ ] Source code is organized
* [ ] Tests are included
* [ ] `21 tests passed`
* [ ] Sample output is included
* [ ] Event log sample is included
* [ ] Database evidence is included
* [ ] Architecture documentation is included
* [ ] AI planning documentation is included
* [ ] Compute documentation is included
* [ ] Demo video link is added
* [ ] No secrets/API keys are committed
* [ ] `.gitignore` is configured
* [ ] Repository contains only relevant project files
* [ ] Final README has the required hackathon statement

---

# 👨‍💻 Project Status

```text
Phase 1 — Core Face Tracking       ✅ Complete
Phase 2 — RTSP Support              ✅ Complete
Phase 3 — Automated Testing         ✅ Complete
Phase 4 — Performance               ✅ Complete
Phase 5 — Documentation             ✅ Complete
Phase 6 — README                    ✅ Complete
Phase 7 — GitHub Cleanup            ⏳
Phase 8 — Final Demo & Interview    ⏳
```

---

# 📌 Final Summary

The Intelligent Face Tracker provides an end-to-end visitor identification pipeline:

```text
Video / RTSP
     ↓
YOLO Face Detection
     ↓
ByteTrack Tracking
     ↓
InsightFace Recognition
     ↓
Persistent FACE_ID
     ↓
Automatic Registration
     ↓
ENTRY Logging
     ↓
Visitor Tracking
     ↓
EXIT Logging
     ↓
SQLite + Images + Event Log
```

Validated results:

```text
21 automated tests passed

12 ENTRY events
12 EXIT events
9 unique visitors
```

The system demonstrates persistent visitor identification rather than simple frame-by-frame face counting.

---

This project is a part of a hackathon run by https://katomaran.com
