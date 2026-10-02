# Compute & Performance

## 1. Overview

The Intelligent Face Tracker is designed to run locally on a standard Windows development machine.

The current implementation was developed and validated using CPU-based inference. The architecture keeps the detection, tracking, recognition, database, and logging components modular so that hardware can be upgraded later without redesigning the complete application.

---

## 2. Development Environment

### Operating System

```text
Windows
```

### Programming Language

```text
Python 3.11.9
```

### Testing Framework

```text
pytest 9.1.1
```

### Database

```text
SQLite
```

### Computer Vision

```text
OpenCV
```

### Object Detection and Tracking

```text
Ultralytics YOLO
ByteTrack
```

### Face Recognition

```text
InsightFace
ArcFace-based face embeddings
```

---

## 3. Current Inference Configuration

The current hardware configuration is defined in `config.json`.

```json
"hardware": {
    "device": "cpu"
}
```

The YOLO + ByteTrack pipeline therefore runs using CPU inference.

The current InsightFace implementation explicitly uses:

```text
CPUExecutionProvider
```

for ONNX Runtime.

Therefore, the current documented configuration should be considered a **CPU-based implementation**.

---

## 4. AI Models

### YOLO Face Detector

The project uses a YOLO face detection model:

```text
models/yolov11n-face.pt
```

The model is responsible for detecting faces and providing bounding boxes for the tracking pipeline.

Current configuration:

```json
"detection": {
    "confidence": 0.5,
    "iou": 0.5,
    "image_size": 1280,
    "frame_skip": 3
}
```

### InsightFace

InsightFace is used for:

* face detection
* face feature extraction
* face embeddings
* face recognition

The configured InsightFace model package is:

```text
buffalo_l
```

The recognition pipeline uses normalized embeddings and cosine-similarity-based matching.

---

## 5. CPU Execution

The application is capable of running without a dedicated NVIDIA GPU.

This makes the project easier to demonstrate on standard development systems.

The main CPU-intensive operations are:

1. YOLO inference
2. InsightFace processing
3. Embedding generation
4. Image processing
5. Video encoding when annotated output is enabled

SQLite and event logging contribute comparatively less computational overhead during normal processing.

---

## 6. Frame Skipping

Frame skipping is used as the primary configurable performance optimization.

Current configuration:

```json
"frame_skip": 3
```

This means the expensive detection and recognition processing is not required on every video frame.

The application processes selected frames and maintains the latest tracking information between processing cycles.

Conceptually:

```text
Frame 1  -> Process
Frame 2  -> Reuse tracking state
Frame 3  -> Reuse tracking state
Frame 4  -> Process
Frame 5  -> Reuse tracking state
Frame 6  -> Reuse tracking state
Frame 7  -> Process
```

This reduces the frequency of expensive computer-vision operations.

---

## 7. Why Frame Skipping Is Used

Running face detection and face recognition on every frame can increase CPU utilization significantly.

Frame skipping provides a configurable trade-off between:

* processing cost
* responsiveness
* tracking continuity

A smaller `frame_skip` value increases processing frequency.

A larger value reduces processing frequency but may reduce responsiveness, particularly when people move quickly through the camera view.

The value can therefore be adjusted according to the available hardware and camera frame rate.

---

## 8. Recognition Processing

Recognition is more computationally expensive than simple bounding-box tracking because face embeddings need to be generated.

The application therefore separates:

```text
Detection
Tracking
Recognition
Identity Management
```

rather than repeatedly performing all operations unnecessarily.

For a newly detected visitor, multiple recognition attempts are used before automatic registration.

The current registration process requires three attempts:

```text
Recognition Attempt 1
        |
Recognition Attempt 2
        |
Recognition Attempt 3
        |
        v
Average Embedding
        |
        v
New FACE_ID
```

This also helps reduce the possibility of registering a visitor based on a single poor-quality observation.

---

## 9. Tracking Efficiency

ByteTrack maintains temporary object identities between processed frames.

The system distinguishes:

```text
TRACK_ID
```

from:

```text
FACE_ID
```

`TRACK_ID` is used for short-term tracking, while `FACE_ID` represents persistent visitor identity.

This allows the application to maintain a visitor's identity even when the tracking ID changes.

---

## 10. Memory Usage

The application stores only the information required for the active processing pipeline in memory.

Examples include:

* current video frame
* current tracking results
* active track information
* recent embeddings for recognition attempts
* recent face crops
* persistent visitor mappings

Persistent visitor information is stored in SQLite rather than requiring the entire visitor database to remain in application memory.

The project does not currently claim a fixed RAM requirement because memory usage depends on factors such as:

* video resolution
* number of simultaneous faces
* InsightFace model configuration
* output video settings
* operating-system background processes

Actual memory consumption should therefore be measured on the target deployment machine if a production resource limit is required.

---

## 11. GPU Considerations

The current validated configuration uses CPU inference.

A GPU can potentially improve processing throughput, particularly for higher-resolution video or multiple simultaneous faces.

However, the current `recognizer.py` explicitly configures:

```text
CPUExecutionProvider
```

Therefore, GPU acceleration should not be considered part of the currently validated configuration.

A future GPU implementation could provide:

```text
YOLO -> CUDA
InsightFace / ONNX Runtime -> CUDA
```

subject to compatible NVIDIA drivers, CUDA libraries, ONNX Runtime configuration, and model support.

---

## 12. Performance Measurement

Performance testing was performed after the core functionality was working.

The application includes processing-frequency measurement and recognition-call monitoring to help evaluate the effect of frame skipping.

Performance should be reported using measurements from the actual machine used for demonstration.

Recommended metrics are:

| Metric               | Measurement                     |
| -------------------- | ------------------------------- |
| Video resolution     | Record actual input resolution  |
| Input FPS            | Record source FPS               |
| Processing FPS       | Record measured application FPS |
| CPU usage            | Measure during processing       |
| RAM usage            | Measure during processing       |
| Number of faces      | Record test scenario            |
| Frame skip           | `3`                             |
| Device               | CPU                             |
| Recognition provider | CPUExecutionProvider            |

No fixed FPS, RAM, or CPU-utilization number is claimed in this document because these values depend on the execution environment and test conditions.

---

## 13. Performance vs Accuracy Trade-Off

The system exposes several parameters that affect performance and recognition behavior.

### `frame_skip`

Controls how frequently expensive processing is performed.

```text
Lower value -> More processing -> Higher computational cost
Higher value -> Less processing -> Lower computational cost
```

### `confidence`

Controls the minimum YOLO detection confidence.

```text
"detection": {
    "confidence": 0.5
}
```

### `image_size`

Controls the YOLO inference image size.

```text
"image_size": 1280
```

Higher image sizes can improve detection of smaller faces but require more computation.

### `similarity_threshold`

Controls the recognition matching threshold.

```text
"similarity_threshold": 0.45
```

This parameter affects the balance between matching existing visitors and registering unknown visitors.

### `max_missing_frames`

Controls how long a missing track remains active.

```text
"max_missing_frames": 50
```

A larger value can reduce premature EXIT events caused by temporary detection loss, while increasing the amount of time a disappeared track remains active.

---

## 14. Recommended Deployment Scaling

### Single Camera

The current architecture is suitable for a single video file or RTSP camera.

```text
Camera
   |
   v
Face Tracker
   |
   v
SQLite
```

### Multiple Cameras

For multiple cameras, the architecture can be extended by assigning a camera identifier to tracking and event records.

For example:

```text
Camera 1 -> Tracker 1
Camera 2 -> Tracker 2
Camera 3 -> Tracker 3
        |
        v
Central Visitor Database
```

A production deployment could then use a server-based database and centralized processing.

---

## 15. RTSP Deployment

The application supports RTSP input through configuration.

Example:

```json
"camera": {
    "type": "rtsp",
    "rtsp_url": "rtsp://YOUR_CAMERA_URL"
}
```

The processing pipeline remains the same:

```text
RTSP Camera
     |
     v
OpenCV VideoCapture
     |
     v
YOLO
     |
     v
ByteTrack
     |
     v
InsightFace
     |
     v
Visitor Manager
```

The main difference between development and live deployment is the video source.

---

## 16. Storage Requirements

The application generates persistent data in several locations.

```text
database/
└── visitors.db

logs/
├── entries/
├── exits/
└── events.log

output/
└── annotated/
```

Storage requirements increase with:

* number of visitors
* number of ENTRY/EXIT events
* number of saved face images
* number of generated output videos
* duration of logging

For long-term production deployment, log rotation and image retention policies should be added.

---

## 17. Reliability Considerations

The application separates runtime processing from persistent storage.

Important information is stored in SQLite and event logs rather than relying only on in-memory state.

This means that persistent visitor identities and historical events can be inspected after the application has stopped.

The application also performs cleanup of active tracks when video processing ends so that active sessions can be closed and EXIT events can be generated.

---

## 18. Current Compute Architecture

The current validated architecture can be summarized as:

```text
                    CPU
                     |
        +------------+-------------+
        |                          |
        v                          v
   YOLO + ByteTrack          InsightFace
        |                          |
        +------------+-------------+
                     |
                     v
             Visitor Manager
                     |
          +----------+----------+
          |          |          |
          v          v          v
       SQLite     Event Log   Images
```

---

## 19. Hardware Upgrade Path

The system can be improved for higher workloads by upgrading compute resources.

### Current

```text
CPU-based inference
Single video source
SQLite
Local image storage
```

### Future

```text
GPU inference
Multiple camera streams
Centralized database
Object storage
Distributed processing
```

The modular architecture makes these changes possible without replacing the entire application.

---

## 20. Compute Summary

| Component               | Current Configuration |
| ----------------------- | --------------------- |
| Operating System        | Windows               |
| Python                  | 3.11.9                |
| Detection               | YOLO                  |
| Tracking                | ByteTrack             |
| Recognition             | InsightFace           |
| Embedding               | ArcFace-based         |
| YOLO Device             | CPU                   |
| InsightFace Provider    | CPUExecutionProvider  |
| Database                | SQLite                |
| Video Processing        | OpenCV                |
| Frame Skip              | 3                     |
| YOLO Confidence         | 0.5                   |
| YOLO Image Size         | 1280                  |
| Recognition Threshold   | 0.45                  |
| Missing Track Threshold | 50 frames             |
| Automated Tests         | 21 passed             |

---

## 21. Performance Validation Summary

The performance phase focused on reducing unnecessary expensive processing while preserving the already-validated visitor tracking behavior.

The primary optimization currently enabled is:

```text
frame_skip = 3
```

The application also includes instrumentation for monitoring recognition processing frequency.

The final demo should report actual measured FPS and resource usage from the demonstration machine rather than using estimated values.

This keeps the performance claims reproducible and verifiable.

---

## 22. Conclusion

The Intelligent Face Tracker is currently configured as a CPU-based local computer-vision application.

The combination of YOLO, ByteTrack, InsightFace, configurable frame skipping, SQLite persistence, and modular visitor management allows the system to run without requiring a dedicated GPU.

For larger deployments, the next optimization steps would be GPU inference, multi-camera processing, centralized storage, and distributed processing.

The current implementation prioritizes correctness, persistent identity management, reliable ENTRY/EXIT logging, and configurable performance over claiming hardware-independent benchmark numbers.
