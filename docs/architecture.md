\# Intelligent Face Tracker — System Architecture



\## 1. Overview



The Intelligent Face Tracker is an AI-based visitor monitoring system that detects, tracks, recognizes, and counts unique faces from a video stream.



The system supports:



\- Local video input

\- RTSP camera input

\- Real-time face detection

\- Multi-object face tracking

\- Face recognition

\- Automatic registration of new visitors

\- Persistent visitor identities

\- ENTRY and EXIT event logging

\- Cropped entry and exit images

\- SQLite-based storage

\- Unique visitor counting

\- Re-identification of returning visitors



\---



\## 2. High-Level Architecture



```text

&#x20;                        VIDEO SOURCE

&#x20;                    ┌──────────────────┐

&#x20;                    │                  │

&#x20;                    │ Sample Video     │

&#x20;                    │       OR         │

&#x20;                    │ RTSP Camera      │

&#x20;                    │                  │

&#x20;                    └────────┬─────────┘

&#x20;                             │

&#x20;                             ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │     OpenCV       │

&#x20;                   │  Frame Capture   │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │       YOLO       │

&#x20;                   │  Face Detection  │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │    ByteTrack     │

&#x20;                   │ Face Tracking    │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │   InsightFace    │

&#x20;                   │ ArcFace Embedding│

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                            ▼

&#x20;                 ┌────────────────────────┐

&#x20;                 │    Face Matching       │

&#x20;                 │ Cosine Similarity      │

&#x20;                 └────────────┬───────────┘

&#x20;                              │

&#x20;                ┌─────────────┴─────────────┐

&#x20;                │                           │

&#x20;             MATCH                       NO MATCH

&#x20;                │                           │

&#x20;                ▼                           ▼

&#x20;       Existing FACE\_ID              New Face

&#x20;                │                           │

&#x20;                │                           ▼

&#x20;                │                    Register FACE\_ID

&#x20;                │                           │

&#x20;                └─────────────┬─────────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌──────────────────┐

&#x20;                   │ Visitor Manager  │

&#x20;                   └────────┬─────────┘

&#x20;                            │

&#x20;                ┌───────────┴───────────┐

&#x20;                │                       │

&#x20;                ▼                       ▼

&#x20;             ENTRY                    EXIT

&#x20;                │                       │

&#x20;                └───────────┬───────────┘

&#x20;                            │

&#x20;              ┌─────────────┼─────────────┐

&#x20;              │             │             │

&#x20;              ▼             ▼             ▼

&#x20;         SQLite DB     Cropped Image   events.log

