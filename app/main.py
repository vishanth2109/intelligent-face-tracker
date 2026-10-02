import os
import time
import cv2

from app.config import Config
from app.recognizer import FaceRecognizer
from app.database import Database
from app.event_logger import EventLogger
from app.visitor_manager import VisitorManager
from app.tracker import FaceTracker


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    directories = [
        "database",
        "logs",
        "logs/entries",
        "logs/exits",
        "output",
        "output/annotated",
        "output/reports"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# ============================================================
# VIDEO / RTSP OPENING
# ============================================================

def open_video(config):

    camera_type = config.get(
        "camera",
        "type",
        default="file"
    )

    camera_type = str(camera_type).strip().lower()

    # --------------------------------------------------------
    # RTSP CAMERA
    # --------------------------------------------------------

    if camera_type == "rtsp":

        rtsp_url = config.get(
            "camera",
            "rtsp_url",
            default=""
        )

        if not rtsp_url:
            raise ValueError(
                "RTSP camera selected but rtsp_url is empty."
            )

        print("\nOpening RTSP stream...")

        # Try FFmpeg backend first
        cap = cv2.VideoCapture(
            rtsp_url,
            cv2.CAP_FFMPEG
        )

        # Fallback to default OpenCV backend
        if not cap.isOpened():

            print(
                "FFmpeg backend failed. "
                "Trying default OpenCV backend..."
            )

            cap.release()

            cap = cv2.VideoCapture(
                rtsp_url
            )

        if not cap.isOpened():

            raise RuntimeError(
                "Unable to open RTSP stream.\n"
                "Check:\n"
                "1. RTSP URL\n"
                "2. Camera availability\n"
                "3. Network connection\n"
                "4. Camera credentials\n"
                "5. OpenCV/FFmpeg support"
            )

        print("RTSP stream connected successfully!")

        return cap

    # --------------------------------------------------------
    # LOCAL VIDEO FILE
    # --------------------------------------------------------

    source = config.get(
        "video",
        "source",
        default="input/sample.mp4"
    )

    if not os.path.exists(source):

        raise FileNotFoundError(
            f"Video source not found: {source}"
        )

    print(f"\nOpening video file: {source}")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():

        raise RuntimeError(
            f"Unable to open video file: {source}"
        )

    print("Video file opened successfully!")

    return cap


# ============================================================
# DISPLAY HELPERS
# ============================================================

def crop_for_display(frame, bbox):

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def draw_track(
    frame,
    track,
    face_id=None
):

    bbox = track["bbox"]
    track_id = track["track_id"]
    confidence = track["confidence"]

    x1, y1, x2, y2 = bbox

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    if face_id is not None:

        label = (
            f"FACE_ID: {face_id} "
            f"| TRACK_ID: {track_id}"
        )

    else:

        label = (
            f"TRACK_ID: {track_id}"
        )

    label += f" | {confidence:.2f}"

    # --------------------------------------------------------
    # Label background
    # --------------------------------------------------------

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1
    )

    cv2.rectangle(
        frame,
        (x1, max(0, y1 - text_height - baseline - 5)),
        (
            x1 + text_width + 5,
            y1
        ),
        (0, 255, 0),
        -1
    )

    # --------------------------------------------------------
    # Label text
    # --------------------------------------------------------

    cv2.putText(
        frame,
        label,
        (
            x1 + 2,
            max(
                text_height + 2,
                y1 - 3
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )


# ============================================================
# INFORMATION PANEL
# ============================================================

def draw_panel(
    frame,
    unique_visitors,
    active_tracks,
    frame_number,
    processing_fps,
    camera_type
):

    panel_height = 125

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            frame.shape[1],
            panel_height
        ),
        (0, 0, 0),
        -1
    )

    # Slight transparency
    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "INTELLIGENT FACE TRACKER",
        (20, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Unique visitors
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Unique Visitors: {unique_visitors}",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Active tracks
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Active Tracks: {active_tracks}",
        (20, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Frame
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Processing FPS: {processing_fps:.2f}",
        (260, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source_text = (
        "RTSP LIVE"
        if camera_type == "rtsp"
        else "VIDEO FILE"
    )

    cv2.putText(
        frame,
        f"Source: {source_text}",
        (260, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Quit instruction
    # --------------------------------------------------------

    if camera_type == "rtsp":

        cv2.putText(
            frame,
            "Press Q to stop",
            (260, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print("=" * 60)
    print("INTELLIGENT FACE TRACKER")
    print("=" * 60)

    start_time = time.time()

    # --------------------------------------------------------
    # Create required directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    print("\nLoading configuration...")

    config = Config()

    # --------------------------------------------------------
    # Camera configuration
    # --------------------------------------------------------

    camera_type = config.get(
        "camera",
        "type",
        default="file"
    )

    camera_type = str(
        camera_type
    ).strip().lower()

    video_source = config.get(
        "video",
        "source",
        default="input/sample.mp4"
    )

    output_path = config.get(
        "video",
        "output",
        default="output/annotated/output.mp4"
    )

    # --------------------------------------------------------
    # Detection configuration
    # --------------------------------------------------------

    model_path = config.get(
        "detection",
        "model",
        default="models/yolov11n-face.pt"
    )

    confidence = config.get(
        "detection",
        "confidence",
        default=0.5
    )

    image_size = config.get(
        "detection",
        "image_size",
        default=1280
    )

    frame_skip = config.get(
        "detection",
        "frame_skip",
        default=1
    )

    frame_skip = max(
        1,
        int(frame_skip)
    )

    # --------------------------------------------------------
    # Recognition configuration
    # --------------------------------------------------------

    similarity_threshold = config.get(
        "recognition",
        "similarity_threshold",
        default=0.45
    )

    # --------------------------------------------------------
    # Tracking configuration
    # --------------------------------------------------------

    max_missing_frames = config.get(
        "tracking",
        "max_missing_frames",
        default=50
    )

    tracker_config = config.get(
        "tracking",
        "tracker",
        default="bytetrack.yaml"
    )

    # --------------------------------------------------------
    # Hardware configuration
    # --------------------------------------------------------

    device = config.get(
        "hardware",
        "device",
        default="cpu"
    )

    device = str(
        device
    ).strip().lower()

    # --------------------------------------------------------
    # Database configuration
    # --------------------------------------------------------

    database_path = config.get(
        "database",
        "path",
        default="database/visitors.db"
    )

    # --------------------------------------------------------
    # Logging configuration
    # --------------------------------------------------------

    log_file = config.get(
        "logging",
        "log_file",
        default="logs/events.log"
    )

    save_entry_images = config.get(
        "logging",
        "save_entry_images",
        default=True
    )

    save_exit_images = config.get(
        "logging",
        "save_exit_images",
        default=True
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print("\nCONFIGURATION")
    print("-" * 60)

    print(f"Camera Type       : {camera_type.upper()}")

    if camera_type == "file":
        print(f"Video Source      : {video_source}")
    else:
        print("Video Source      : RTSP stream")

    print(f"YOLO Model        : {model_path}")
    print(f"Confidence        : {confidence}")
    print(f"Image Size        : {image_size}")
    print(f"Frame Skip        : {frame_skip}")
    print(f"Similarity        : {similarity_threshold}")
    print(f"Max Missing       : {max_missing_frames}")
    print(f"Tracker            : {tracker_config}")
    print(f"Device             : {device}")
    print(f"Database           : {database_path}")
    print(f"Log File           : {log_file}")

    print("-" * 60)

    # --------------------------------------------------------
    # Validate model
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    # --------------------------------------------------------
    # Create required output directories
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path)
        or ".",
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(database_path)
        or ".",
        exist_ok=True
    )

    os.makedirs(
        os.path.dirname(log_file)
        or ".",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Initialize Face Tracker
    # --------------------------------------------------------

    print("\nInitializing Face Tracker...")

    tracker = FaceTracker(
        model_path=model_path,
        confidence=confidence,
        image_size=image_size,
        tracker=tracker_config,
        device=device
    )

    # --------------------------------------------------------
    # Initialize InsightFace
    # --------------------------------------------------------

    print("\nInitializing Face Recognizer...")

    recognizer = FaceRecognizer()

    # --------------------------------------------------------
    # Initialize Database
    # --------------------------------------------------------

    print("\nInitializing database...")

    database = Database(
        database_path
    )

    database.create_tables()

    # --------------------------------------------------------
    # Initialize Event Logger
    # --------------------------------------------------------

    print("\nInitializing event logger...")

    logger = EventLogger(
        log_file
    )

    # --------------------------------------------------------
    # Initialize Visitor Manager
    # --------------------------------------------------------

    print("\nInitializing Visitor Manager...")

    visitor_manager = VisitorManager(
        database=database,
        logger=logger,
        recognizer=recognizer,
        similarity_threshold=similarity_threshold,
        max_missing_frames=max_missing_frames,
        save_entry_images=save_entry_images,
        save_exit_images=save_exit_images
    )

    # --------------------------------------------------------
    # Open video / RTSP
    # --------------------------------------------------------

    cap = open_video(config)

    # --------------------------------------------------------
    # Get video properties
    # --------------------------------------------------------

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if source_fps <= 0:
        source_fps = 25.0

    # --------------------------------------------------------
    # Print video information
    # --------------------------------------------------------

    print("\nVIDEO INFORMATION")
    print("-" * 60)

    print(f"Resolution        : {width} x {height}")
    print(f"Source FPS        : {source_fps:.2f}")

    if camera_type == "rtsp":

        print("Total Frames      : LIVE STREAM")

    else:

        print(
            f"Total Frames      : {total_frames}"
        )

    print("-" * 60)

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    writer = None

    if camera_type == "file":

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            source_fps,
            (
                width,
                height
            )
        )

        if not writer.isOpened():

            cap.release()

            raise RuntimeError(
                f"Unable to create output video: "
                f"{output_path}"
            )

        print(
            f"\nOutput video: {output_path}"
        )

    else:

        print(
            "\nRTSP mode: live display enabled."
        )

        print(
            "Press Q in the video window to stop."
        )

    # --------------------------------------------------------
    # Processing variables
    # --------------------------------------------------------

    frame_number = 0

    detection_cycles = 0

    processed_frames = 0

    last_tracks = []

    processing_start = time.time()

    last_status_time = processing_start

    # --------------------------------------------------------
    # Main processing loop
    # --------------------------------------------------------

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                if camera_type == "rtsp":

                    print(
                        "\nRTSP stream ended or connection lost."
                    )

                else:

                    print(
                        "\nVideo processing completed."
                    )

                break

            frame_number += 1

            # ------------------------------------------------
            # Frame skipping
            # ------------------------------------------------

            should_detect = (
                frame_number == 1
                or frame_number % frame_skip == 0
            )

            if should_detect:

                detection_cycles += 1

                # --------------------------------------------
                # YOLO + ByteTrack
                # --------------------------------------------

                tracks = tracker.track(
                    frame
                )

                last_tracks = tracks

                # --------------------------------------------
                # InsightFace
                # --------------------------------------------

                insight_faces = recognizer.get_faces(
                    frame
                )

                current_track_ids = set()

                # --------------------------------------------
                # Process tracks
                # --------------------------------------------

                for track in tracks:

                    track_id = track[
                        "track_id"
                    ]

                    bbox = track[
                        "bbox"
                    ]

                    current_track_ids.add(
                        track_id
                    )

                    # ----------------------------------------
                    # New Track
                    # ----------------------------------------

                    if (
                        track_id
                        not in visitor_manager.track_to_face
                    ):

                        visitor_manager.process_track(
                            frame,
                            track,
                            insight_faces
                        )

                    # ----------------------------------------
                    # Existing Track
                    # ----------------------------------------

                    else:

                        face_id = (
                            visitor_manager.track_to_face[
                                track_id
                            ]
                        )

                        visitor_manager.update_track(
                            track_id,
                            frame,
                            bbox
                        )

                # --------------------------------------------
                # Detect tracks that disappeared
                # --------------------------------------------

                visitor_manager.update_missing_tracks(
                    current_track_ids
                )

            else:

                # ------------------------------------------------
                # Reuse previous tracks between detection cycles
                # ------------------------------------------------

                tracks = last_tracks

            # ------------------------------------------------
            # Draw tracks
            # ------------------------------------------------

            for track in tracks:

                track_id = track[
                    "track_id"
                ]

                face_id = (
                    visitor_manager.track_to_face.get(
                        track_id
                    )
                )

                draw_track(
                    frame,
                    track,
                    face_id
                )

            # ------------------------------------------------
            # Processing FPS
            # ------------------------------------------------

            processed_frames += 1

            elapsed = (
                time.time()
                - processing_start
            )

            processing_fps = (
                processed_frames / elapsed
                if elapsed > 0
                else 0
            )

            # ------------------------------------------------
            # Draw information panel
            # ------------------------------------------------

            unique_visitors = (
                visitor_manager.get_unique_visitor_count()
            )

            active_tracks = len(
                visitor_manager.track_to_face
            )

            draw_panel(
                frame=frame,
                unique_visitors=unique_visitors,
                active_tracks=active_tracks,
                frame_number=frame_number,
                processing_fps=processing_fps,
                camera_type=camera_type
            )

            # ------------------------------------------------
            # Save annotated video
            # ------------------------------------------------

            if writer is not None:

                writer.write(
                    frame
                )

            # ------------------------------------------------
            # RTSP live display
            # ------------------------------------------------

            if camera_type == "rtsp":

                cv2.imshow(
                    "Intelligent Face Tracker - RTSP",
                    frame
                )

                key = cv2.waitKey(
                    1
                ) & 0xFF

                if key == ord("q"):

                    print(
                        "\nStopping RTSP stream..."
                    )

                    break

            # ------------------------------------------------
            # File mode display
            # ------------------------------------------------

            else:

                # Optional display
                # Press Q to stop processing early

                cv2.imshow(
                    "Intelligent Face Tracker",
                    frame
                )

                key = cv2.waitKey(
                    1
                ) & 0xFF

                if key == ord("q"):

                    print(
                        "\nProcessing stopped by user."
                    )

                    break

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            current_time = time.time()

            if (
                current_time
                - last_status_time
                >= 5
            ):

                if camera_type == "rtsp":

                    print(
                        f"[LIVE] "
                        f"Frame={frame_number} | "
                        f"Tracks={active_tracks} | "
                        f"Unique={unique_visitors} | "
                        f"FPS={processing_fps:.2f}"
                    )

                else:

                    if total_frames > 0:

                        progress = (
                            frame_number
                            / total_frames
                        ) * 100

                        print(
                            f"[PROGRESS] "
                            f"{progress:.1f}% | "
                            f"Frame={frame_number}/{total_frames} | "
                            f"Tracks={active_tracks} | "
                            f"Unique={unique_visitors} | "
                            f"FPS={processing_fps:.2f}"
                        )

                    else:

                        print(
                            f"[PROCESSING] "
                            f"Frame={frame_number} | "
                            f"Tracks={active_tracks} | "
                            f"Unique={unique_visitors} | "
                            f"FPS={processing_fps:.2f}"
                        )

                last_status_time = current_time

    # ========================================================
    # KEYBOARD INTERRUPT
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n\nProcessing interrupted by user."
        )

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print(
            "\nERROR DURING PROCESSING"
        )

        print(
            "-" * 60
        )

        print(
            str(error)
        )

        raise

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        print(
            "\nCleaning up..."
        )

        # ----------------------------------------------------
        # Close active tracks
        # ----------------------------------------------------

        try:

            active_track_ids = list(
                visitor_manager.track_to_face.keys()
            )

            for track_id in active_track_ids:

                try:

                    visitor_manager.close_track(
                        track_id
                    )

                except Exception as error:

                    print(
                        f"Warning: "
                        f"Unable to close track "
                        f"{track_id}: {error}"
                    )

        except Exception as error:

            print(
                f"Warning during track cleanup: "
                f"{error}"
            )

        # ----------------------------------------------------
        # Release video capture
        # ----------------------------------------------------

        if cap is not None:

            cap.release()

        # ----------------------------------------------------
        # Release output writer
        # ----------------------------------------------------

        if writer is not None:

            writer.release()

        # ----------------------------------------------------
        # Close OpenCV windows
        # ----------------------------------------------------

        cv2.destroyAllWindows()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_processing_time = (
        time.time()
        - start_time
    )

    final_unique_visitors = (
        visitor_manager.get_unique_visitor_count()
    )

    average_fps = (
        processed_frames
        / total_processing_time
        if total_processing_time > 0
        else 0
    )

    print("\n")
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Input Type        : "
        f"{camera_type.upper()}"
    )

    print(
        f"Frames Processed  : "
        f"{processed_frames}"
    )

    print(
        f"Detection Cycles  : "
        f"{detection_cycles}"
    )

    print(
        f"Processing Time   : "
        f"{total_processing_time:.2f} seconds"
    )

    print(
        f"Average FPS       : "
        f"{average_fps:.2f}"
    )

    print(
        f"Unique Visitors   : "
        f"{final_unique_visitors}"
    )

    print(
        f"Database          : "
        f"{database_path}"
    )

    print(
        f"Event Log         : "
        f"{log_file}"
    )

    if camera_type == "file":

        print(
            f"Output Video      : "
            f"{output_path}"
        )

    else:

        print(
            "Output Video      : "
            "Live RTSP display"
        )

    print("=" * 60)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()



