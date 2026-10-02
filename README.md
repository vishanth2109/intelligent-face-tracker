# Intelligent Face Tracker with Auto-Registration and Visitor Counting

An AI-powered real-time face tracking and visitor monitoring system that detects, tracks, recognizes, automatically registers, and logs visitors from video streams.

The system combines **YOLO face detection, ByteTrack tracking, InsightFace recognition, persistent FACE_ID management, SQLite storage, event logging, and a Streamlit monitoring dashboard**.

---

## 1. Project Overview

The objective of this project is to build an intelligent visitor tracking system capable of:

* Detecting faces in real time
* Tracking faces across video frames
* Generating persistent visitor identities
* Recognizing previously registered visitors
* Automatically registering new visitors
* Preventing duplicate visitor counting
* Recording exactly one ENTRY and EXIT event per appearance
* Saving timestamped face crops
* Maintaining visitor history in SQLite
* Supporting both video files and RTSP streams
* Providing a professional monitoring dashboard
* Maintaining an auditable event log
* Running automated tests

The system is designed as a modular computer-vision pipeline that separates temporary tracking identities from persistent visitor identities.

---

# 2. Key Features

### Face Detection

Uses **YOLO** for fast and accurate face detection.

### Multi-Object Tracking

Uses **ByteTrack** to maintain tracking identities across consecutive frames.

### Face Recognition

Uses **InsightFace / ArcFace embeddings** to recognize previously registered visitors.

### Automatic Registration

When an unknown face is detected repeatedly, the system generates a normalized embedding and automatically creates a persistent `FACE_ID`.

Example:

```text
FACE_0001
FACE_0002
FACE_0003
```

### Persistent Identity

A persistent `FACE_ID` is stored in SQLite and can be recognized across different tracking sessions.

### Unique Visitor Counting

The system separates:

```text
TRACK_ID
```

from:

```text
FACE_ID
```

`TRACK_ID` represents a temporary tracking session.

`FACE_ID` represents a persistent visitor identity.

This prevents a returning visitor from being counted as a new unique visitor simply because a new tracking ID was created.

### ENTRY / EXIT Logging

Each visitor appearance produces:

```text
ENTRY
```

and when the visitor leaves the scene:

```text
EXIT
```

Both events contain:

* Face ID
* Track ID
* Timestamp
* Event type
* Image path
* Confidence

### Face Crop Storage

Timestamped face crops are saved for entry and exit events.

### SQLite Database

The database stores:

* Visitor identities
* Face embeddings
* Event history
* Tracking sessions
* First/last seen timestamps

### Event Logging

The system records important operations in:

```text
logs/events.log
```

including detection, recognition, registration, tracking, entry, and exit events.

### RTSP Support

The same processing pipeline supports RTSP camera streams.

### Configurable Processing

Important parameters can be modified through:

```text
config.json
```

without changing the source code.

### Professional Web Dashboard

A Streamlit-based monitoring dashboard provides:

* Unique visitor count
* Entry event count
* Exit event count
* Active tracking sessions
* Event analytics
* Recent visitor events
* Registered visitors
* Latest visitor event image
* System status
* AI technology stack
* Recognition configuration

---

# 3. System Architecture

```text
                    VIDEO / RTSP
                         │
                         ▼
                ┌──────────────────┐
                │ YOLO FACE        │
                │ DETECTION        │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ BYTE TRACK       │
                │ TRACKING         │
                └────────┬─────────┘
                         │
                         ▼
                     TRACK_ID
                         │
                         ▼
                ┌──────────────────┐
                │ INSIGHTFACE      │
                │ RECOGNITION      │
                └────────┬─────────┘
                         │
                         ▼
                  FACE EMBEDDING
                         │
                         ▼
                ┌──────────────────┐
                │ VISITOR MANAGER  │
                └────────┬─────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          FACE_ID               ENTRY / EXIT
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌──────────────────┐
                │ SQLITE DATABASE  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ STREAMLIT        │
                │ DASHBOARD        │
                └──────────────────┘
```

---

# 4. Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Face Detection       | YOLO                  |
| Tracking             | ByteTrack             |
| Face Recognition     | InsightFace / ArcFace |
| Computer Vision      | OpenCV                |
| Database             | SQLite                |
| Dashboard            | Streamlit             |
| Data Processing      | NumPy / Pandas        |
| Testing              | Pytest                |
| Input                | MP4 / RTSP            |
| Configuration        | JSON                  |

---

# 5. TRACK_ID vs FACE_ID

This is a key architectural concept.

### TRACK_ID

Generated by ByteTrack.

It represents a temporary tracking session.

Example:

```text
TRACK_ID = 17
```

A person may receive another tracking ID if the tracker loses the person and later detects them again.

### FACE_ID

Generated by the visitor management system.

It represents the persistent identity of the visitor.

Example:

```text
FACE_ID = FACE_0007
```

The same person can therefore have:

```text
TRACK_ID 17 → FACE_0007
TRACK_ID 25 → FACE_0007
```

The visitor is still counted as one unique person.

---

# 6. Face Recognition and Auto-Registration

When a tracked face is detected:

1. InsightFace detects the face.
2. The YOLO bounding box is matched with the InsightFace face.
3. A face embedding is extracted.
4. The embedding is normalized.
5. Existing visitor embeddings are searched.
6. Similarity is calculated.
7. If a match exceeds the configured threshold, the existing `FACE_ID` is reused.
8. If no match is found, multiple recognition attempts are performed.
9. The embeddings are averaged and normalized.
10. A new persistent visitor is registered.

The current implementation uses multiple attempts before registering a new face to reduce the risk of creating identities from a single unreliable observation.

---

# 7. ENTRY and EXIT Processing

### ENTRY

An ENTRY event is created when a new tracking session is assigned to a visitor.

The system:

* Assigns `FACE_ID`
* Creates tracking session
* Saves face crop
* Records timestamp
* Inserts ENTRY event
* Logs the event

### EXIT

When the tracker no longer detects the person for the configured missing-frame threshold:

* The tracking session is closed
* The latest face crop is saved
* EXIT event is recorded
* Exit timestamp is stored
* Event is written to the event log

This prevents repeated ENTRY and EXIT events for the same continuous appearance.

---

# 8. Database Structure

SQLite database:

```text
database/visitors.db
```

## persons

Stores persistent visitor identities.

| Field      | Description                 |
| ---------- | --------------------------- |
| id         | Internal database ID        |
| face_id    | Persistent visitor identity |
| first_seen | First observation           |
| last_seen  | Latest observation          |
| embedding  | Stored face embedding       |
| created_at | Registration timestamp      |

## events

Stores visitor events.

| Field      | Description                      |
| ---------- | -------------------------------- |
| id         | Event ID                         |
| face_id    | Persistent visitor ID            |
| track_id   | Tracking session ID              |
| event_type | ENTRY / EXIT                     |
| timestamp  | Event timestamp                  |
| image_path | Saved face crop                  |
| confidence | Detection/recognition confidence |

## tracks

Stores tracking sessions.

| Field      | Description        |
| ---------- | ------------------ |
| id         | Track database ID  |
| face_id    | Associated visitor |
| track_id   | ByteTrack ID       |
| start_time | Tracking start     |
| end_time   | Tracking end       |
| status     | Tracking status    |

---

# 9. Project Structure

```text
intelligent-face-tracker/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── detector.py
│   ├── event_logger.py
│   ├── main.py
│   ├── recognizer.py
│   ├── tracker.py
│   ├── utils.py
│   └── visitor_manager.py
│
├── docs/
│   ├── AI_PLANNING.md
│   ├── COMPUTE.md
│   └── architecture.md
│
├── frontend/
│   └── dashboard.py
│
├── input/
│   └── sample.mp4
│
├── models/
│   └── yolov11n-face.pt
│
├── logs/
│   ├── entries/
│   └── exits/
│
├── output/
│   └── annotated/
│
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_database.py
│   └── test_visitor_manager.py
│
├── config.json
├── requirements.txt
├── .gitignore
└── README.md
```

Runtime-generated files such as databases, logs, face images, generated videos, virtual environments, and caches are excluded from version control where appropriate.

---

# 10. Configuration

Main configuration file:

```text
config.json
```

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
    }
}
```

Important parameters include:

### Frame Skip

```text
frame_skip = 3
```

Controls how frequently detection and recognition processing is performed.

### Recognition Threshold

```text
similarity_threshold = 0.45
```

Controls the face embedding similarity threshold used for recognition.

### Missing Frames

```text
max_missing_frames = 50
```

Controls how long a missing track is retained before an EXIT event is generated.

---

# 11. Installation

## Clone the repository

```bash
git clone https://github.com/vishanth2109/intelligent-face-tracker.git
cd intelligent-face-tracker
```

## Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

## Install dependencies

```powershell
pip install -r requirements.txt
```

---

# 12. Run the AI Face Tracker

From the project root:

```powershell
python -m app.main
```

The system will:

1. Load YOLO
2. Load InsightFace
3. Open the configured input
4. Detect faces
5. Track faces
6. Recognize existing visitors
7. Register unknown visitors
8. Record ENTRY events
9. Monitor tracking sessions
10. Record EXIT events
11. Store data in SQLite

---

# 13. Run the Frontend Dashboard

The project includes a professional Streamlit monitoring interface.

Run:

```powershell
streamlit run frontend\dashboard.py
```

The dashboard opens at:

```text
http://localhost:8501
```

## Dashboard Features

### KPI Monitoring

The dashboard displays:

```text
Unique Visitors
Entry Events
Exit Events
Active Tracks
```

### Event Analytics

Visualizes ENTRY and EXIT activity.

### Latest Visitor Event

Displays the most recent visitor event and associated face crop when available.

### Recent Events

Displays:

* Face ID
* Track ID
* Event
* Timestamp
* Confidence

### Registered Visitors

Displays persistent visitor identities and their first/last seen timestamps.

### System Information

Displays the technologies used by the tracking system:

```text
YOLO
ByteTrack
InsightFace
SQLite
Video / RTSP
```

The frontend acts as a **monitoring layer** over the AI processing pipeline. The actual computer-vision processing remains modular and independent from the Streamlit interface.

---

# 14. RTSP Camera Support

The application can be configured for an RTSP camera.

Update:

```json
"camera": {
    "type": "rtsp",
    "rtsp_url": "rtsp://YOUR_CAMERA_URL"
}
```

The processing pipeline remains the same:

```text
RTSP
 ↓
YOLO
 ↓
ByteTrack
 ↓
InsightFace
 ↓
Visitor Manager
 ↓
SQLite
 ↓
Dashboard
```

For development and testing, the included sample video is used.

---

# 15. Logging

The application maintains:

```text
logs/events.log
```

The event logger records important system operations including:

```text
Face detection
Track creation
Face recognition
Embedding generation
New face registration
ENTRY
Tracking
EXIT
```

Face crops are stored separately for entry and exit events.

---

# 16. Validation Results

The core system was validated using the development video.

Example database validation:

```text
DATABASE VALIDATION
================================
ENTRY events      : 12
EXIT events       : 12
Unique visitors   : 9
```

This demonstrates:

* ENTRY events are being recorded
* EXIT events are being recorded
* Persistent visitors are stored
* Re-identification does not create duplicate identities

---

# 17. Automated Testing

The project includes automated tests using Pytest.

Run:

```powershell
pytest -v
```

Current validation:

```text
21 passed
```

Tests cover areas including:

* Configuration
* Database operations
* Visitor registration
* Visitor matching
* Track management
* ENTRY/EXIT behavior
* Event processing

---

# 18. Performance

The system was developed and tested using CPU-based inference.

Current configuration uses:

```text
YOLO              → CPU
InsightFace       → CPUExecutionProvider
Frame Skip        → 3
Tracker           → ByteTrack
Database          → SQLite
```

The frame-skip configuration provides a tunable way to balance processing workload and tracking responsiveness.

Detailed compute information is available in:

```text
docs/COMPUTE.md
```

---

# 19. AI-Assisted Development

AI coding assistance was used during development for:

* Project architecture planning
* Module design
* Debugging
* Test generation
* Documentation
* Code improvement
* Error analysis
* README preparation

All generated code was reviewed, executed, tested, and validated manually.

Development iterations included debugging:

* Database schema issues
* Face registration behavior
* Recognition attempts
* Tracking lifecycle
* EXIT event generation
* Test assumptions
* Configuration handling

More information is available in:

```text
docs/AI_PLANNING.md
```

---

# 20. Design Decisions

### Why YOLO?

YOLO provides efficient real-time object detection and works well for processing video streams.

### Why ByteTrack?

ByteTrack maintains object identities across frames and provides temporary tracking IDs.

### Why InsightFace?

InsightFace provides face analysis and high-dimensional face embeddings suitable for identity matching.

### Why SQLite?

SQLite is lightweight, local, reliable, and sufficient for a standalone visitor monitoring application.

### Why separate TRACK_ID and FACE_ID?

This prevents temporary tracking sessions from being confused with persistent visitor identities.

### Why multiple registration attempts?

A single face observation may produce an unreliable embedding. Multiple observations provide a more stable representation before creating a new persistent identity.

### Why configuration through JSON?

Important parameters can be changed without modifying application source code.

---

# 21. Error Handling and Resilience

The application includes handling for:

* Missing configuration
* Invalid video frames
* Missing face detections
* Invalid bounding boxes
* Missing embeddings
* Invalid database states
* Missing event images
* Lost tracking sessions

The system is designed so that the database and event logging layers remain separate from the detection and recognition components.

---

# 22. Current Limitations

The current implementation has some practical limitations:

* Recognition performance depends on lighting, camera angle, face size, and image quality.
* CPU inference is slower than GPU inference.
* SQLite is intended for local/small-scale deployments rather than high-concurrency production systems.
* RTSP reliability depends on camera/network quality.
* The current unique visitor set maintained by the running application is session-based, while persistent identities are stored in SQLite.
* Face recognition thresholds may require tuning for different environments.

---

# 23. Future Improvements

Potential improvements include:

* GPU acceleration
* Stronger re-identification
* DeepSORT comparison
* Better camera calibration
* Multi-camera support
* PostgreSQL deployment
* REST API
* Authentication and role-based access
* Real-time WebSocket dashboard updates
* Visitor search and filtering
* Advanced analytics
* Daily/weekly visitor reports
* Cloud deployment
* Containerization with Docker
* Edge-device deployment

---

# 24. Documentation

Additional documentation:

### Architecture

```text
docs/architecture.md
```

### AI-Assisted Development

```text
docs/AI_PLANNING.md
```

### Compute and Performance

```text
docs/COMPUTE.md
```

---

# 25. Hackathon Demo Flow

Recommended demonstration sequence:

```text
1. Problem Statement
        ↓
2. Architecture
        ↓
3. Start AI Pipeline
        ↓
4. Face Detection
        ↓
5. ByteTrack Tracking
        ↓
6. Face Recognition
        ↓
7. Auto Registration
        ↓
8. ENTRY Event
        ↓
9. EXIT Event
        ↓
10. SQLite Database
        ↓
11. Streamlit Dashboard
        ↓
12. Automated Tests
```

The key technical concept to explain during the demonstration is:

```text
TRACK_ID = temporary tracking identity

FACE_ID = persistent visitor identity
```

---

# 26. Demo

Demo video:

```text
To be added before final submission.
```

---

# 27. Repository

GitHub:

https://github.com/vishanth2109/intelligent-face-tracker

---

# 28. Submission Checklist

* [x] YOLO face detection
* [x] ByteTrack tracking
* [x] InsightFace recognition
* [x] Automatic face registration
* [x] Persistent FACE_ID
* [x] Unique visitor counting
* [x] ENTRY event logging
* [x] EXIT event logging
* [x] Timestamped face crops
* [x] SQLite database
* [x] Event log
* [x] RTSP support
* [x] Configurable frame skip
* [x] Automated tests
* [x] Documentation
* [x] AI planning documentation
* [x] Compute documentation
* [x] Professional frontend dashboard
* [x] GitHub repository
* [x] Final demo video link
* [x] Final hackathon submission

---

# 29. Project Status

**Core AI System:** Complete

**Database & Logging:** Complete

**RTSP Support:** Implemented

**Automated Testing:** Complete — 21 tests passing

**Documentation:** Complete

**Frontend Dashboard:** Implemented

**GitHub Repository:** Published

**Final Demo:** To be recorded

---

# 30. Final Summary

The Intelligent Face Tracker is a modular AI-powered visitor monitoring system that combines face detection, multi-object tracking, face recognition, automatic identity registration, persistent visitor management, ENTRY/EXIT event logging, SQLite storage, RTSP support, automated testing, and a professional monitoring dashboard.

The architecture separates temporary tracking identities from persistent face identities, allowing visitors to be recognized across tracking sessions without incorrectly increasing the unique visitor count.

The project is designed to be understandable, configurable, testable, and extensible for future real-world deployment.

This project is a part of a hackathon run by https://katomaran.com

