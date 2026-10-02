import os
from datetime import datetime

import cv2
import numpy as np

from app.utils import find_best_match


class VisitorManager:

    def __init__(
        self,
        database,
        logger,
        recognizer,
        similarity_threshold=0.45,
        max_missing_frames=50,
        save_entry_images=True,
        save_exit_images=True
    ):
        self.database = database
        self.logger = logger
        self.recognizer = recognizer

        self.similarity_threshold = similarity_threshold
        self.max_missing_frames = max_missing_frames

        self.save_entry_images = save_entry_images
        self.save_exit_images = save_exit_images

        # --------------------------------------------------------
        # ByteTrack ID -> Persistent FACE_ID
        # --------------------------------------------------------

        self.track_to_face = {}

        # --------------------------------------------------------
        # Active track state
        # --------------------------------------------------------

        self.active_tracks = {}

        # --------------------------------------------------------
        # Recognition embeddings
        # --------------------------------------------------------

        self.track_embeddings = {}
        self.track_attempts = {}

        # --------------------------------------------------------
        # Latest face crop for EXIT image
        # --------------------------------------------------------

        self.last_crops = {}

        # --------------------------------------------------------
        # Unique visitors detected during this run
        # --------------------------------------------------------

        self.unique_visitors = set()

    # ============================================================
    # SAVE IMAGE
    # ============================================================

    def save_image(
        self,
        image,
        folder,
        face_id,
        event_type
    ):
        """
        Save ENTRY / EXIT face image.
        """

        if image is None:
            return None

        os.makedirs(
            folder,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        filename = (
            f"{face_id}_"
            f"{event_type}_"
            f"{timestamp}.jpg"
        )

        path = os.path.join(
            folder,
            filename
        )

        success = cv2.imwrite(
            path,
            image
        )

        if not success:
            print(
                f"WARNING: Failed to save image: {path}"
            )
            return None

        return path

    # ============================================================
    # PROCESS TRACK
    # ============================================================

    def process_track(
        self,
        frame,
        track,
        insight_faces
    ):
        """
        Process a new/unrecognized ByteTrack track.

        YOLO:
            Provides bounding box and track ID.

        InsightFace:
            Provides face embedding.

        Process:
            1. Match YOLO box with InsightFace face.
            2. Collect multiple embeddings.
            3. Compare against registered faces.
            4. Recognize existing visitor OR register new visitor.
            5. Create tracking session.
        """

        track_id = track["track_id"]
        bbox = track["bbox"]
        confidence = track["confidence"]

        # --------------------------------------------------------
        # Already recognized
        # --------------------------------------------------------

        if track_id in self.track_to_face:

            return self.track_to_face[track_id]

        # --------------------------------------------------------
        # Match YOLO bounding box to InsightFace face
        # --------------------------------------------------------

        embedding = self.recognizer.match_face_to_bbox(
            bbox,
            insight_faces
        )

        if embedding is None:
            return None

        # --------------------------------------------------------
        # Initialize embedding storage
        # --------------------------------------------------------

        if track_id not in self.track_embeddings:

            self.track_embeddings[
                track_id
            ] = []

        if track_id not in self.track_attempts:

            self.track_attempts[
                track_id
            ] = 0

        # --------------------------------------------------------
        # Store embedding
        # --------------------------------------------------------

        self.track_embeddings[
            track_id
        ].append(
            embedding
        )

        # Keep only the latest 5 embeddings
        self.track_embeddings[
            track_id
        ] = self.track_embeddings[
            track_id
        ][-5:]

        self.track_attempts[
            track_id
        ] += 1

        attempt = self.track_attempts[
            track_id
        ]

        print(
            f"Track {track_id}: "
            f"Recognition attempt "
            f"{attempt}/3"
        )

        # --------------------------------------------------------
        # Average embeddings
        # --------------------------------------------------------

        embeddings = self.track_embeddings[
            track_id
        ]

        average_embedding = np.mean(
            embeddings,
            axis=0
        )

        # Normalize embedding
        norm = np.linalg.norm(
            average_embedding
        )

        if norm > 0:

            average_embedding = (
                average_embedding / norm
            )

        # --------------------------------------------------------
        # Get registered faces
        # --------------------------------------------------------

        stored_faces = self.database.get_all_faces()

        # --------------------------------------------------------
        # Find best face match
        # --------------------------------------------------------

        face_id, similarity = find_best_match(
            average_embedding,
            stored_faces,
            threshold=self.similarity_threshold
        )

        # ========================================================
        # EXISTING FACE RECOGNIZED
        # ========================================================

        if face_id is not None:

            print(
                f"Track {track_id}: "
                f"Recognized {face_id} "
                f"(similarity={similarity:.4f})"
            )

            self.logger.face_recognized(
                face_id,
                similarity
            )

            self.assign_track(
                track_id,
                face_id,
                frame,
                bbox,
                confidence
            )

            return face_id

        # ========================================================
        # WAIT FOR MORE EMBEDDINGS
        # ========================================================

        if attempt < 3:

            print(
                f"Track {track_id}: "
                f"No match yet. "
                f"Waiting for more embeddings..."
            )

            return None

        # ========================================================
        # REGISTER NEW FACE
        # ========================================================

        face_id = self.database.register_face(
            average_embedding
        )

        print(
            f"Track {track_id}: "
            f"New face registered -> {face_id}"
        )

        self.logger.new_face_registered(
            face_id
        )

        self.assign_track(
            track_id,
            face_id,
            frame,
            bbox,
            confidence
        )

        return face_id

    # ============================================================
    # ASSIGN TRACK
    # ============================================================

    def assign_track(
        self,
        track_id,
        face_id,
        frame,
        bbox,
        confidence
    ):
        """
        Assign a persistent FACE_ID to a ByteTrack track.

        Creates:
            1. Active track
            2. ENTRY image
            3. ENTRY event
            4. Database tracking session
        """

        # --------------------------------------------------------
        # Prevent duplicate assignment
        # --------------------------------------------------------

        if track_id in self.active_tracks:

            return

        # --------------------------------------------------------
        # Use ONE timestamp for the ENTRY
        # --------------------------------------------------------

        start_time = datetime.now().isoformat()

        # --------------------------------------------------------
        # Map ByteTrack ID -> FACE_ID
        # --------------------------------------------------------

        self.track_to_face[
            track_id
        ] = face_id

        # --------------------------------------------------------
        # Store active track
        # --------------------------------------------------------

        self.active_tracks[
            track_id
        ] = {
            "face_id": face_id,
            "missing_frames": 0,
            "confidence": confidence,
            "start_time": start_time
        }

        # --------------------------------------------------------
        # Add to unique visitor set
        # --------------------------------------------------------

        self.unique_visitors.add(
            face_id
        )

        # --------------------------------------------------------
        # Crop face
        # --------------------------------------------------------

        crop = self.crop_face(
            frame,
            bbox
        )

        self.last_crops[
            track_id
        ] = crop

        # --------------------------------------------------------
        # Save ENTRY image
        # --------------------------------------------------------

        image_path = None

        if self.save_entry_images:

            image_path = self.save_image(
                crop,
                "logs/entries",
                face_id,
                "entry"
            )

        # --------------------------------------------------------
        # Add ENTRY event
        # --------------------------------------------------------

        self.database.add_event(
            face_id=face_id,
            track_id=track_id,
            event_type="ENTRY",
            image_path=image_path,
            confidence=confidence,
            timestamp=start_time
        )

        # --------------------------------------------------------
        # Add tracking session
        #
        # IMPORTANT:
        # start_time is required by Database.add_track()
        # --------------------------------------------------------

        self.database.add_track(
            face_id=face_id,
            track_id=track_id,
            start_time=start_time,
            status="ACTIVE"
        )

        # --------------------------------------------------------
        # Logger
        # --------------------------------------------------------

        self.logger.entry(
            face_id,
            track_id
        )

        print(
            f"ENTRY -> {face_id} "
            f"(Track {track_id})"
        )

    # ============================================================
    # UPDATE TRACK
    # ============================================================

    def update_track(
        self,
        track_id,
        frame,
        bbox
    ):
        """
        Update an already recognized track.

        Resets missing-frame counter and updates:
            - latest face crop
            - visitor last_seen timestamp
        """

        if track_id not in self.active_tracks:

            return

        # --------------------------------------------------------
        # Reset missing counter
        # --------------------------------------------------------

        self.active_tracks[
            track_id
        ]["missing_frames"] = 0

        # --------------------------------------------------------
        # Update latest crop
        # --------------------------------------------------------

        crop = self.crop_face(
            frame,
            bbox
        )

        if crop is not None:

            self.last_crops[
                track_id
            ] = crop

        # --------------------------------------------------------
        # Update visitor last seen
        # --------------------------------------------------------

        face_id = self.track_to_face.get(
            track_id
        )

        if face_id:

            self.database.update_last_seen(
                face_id
            )

    # ============================================================
    # UPDATE MISSING TRACKS
    # ============================================================

    def update_missing_tracks(
        self,
        current_track_ids
    ):
        """
        Detect ByteTrack tracks that disappeared.

        A track is closed after:
            max_missing_frames
        """

        current_track_ids = set(
            current_track_ids
        )

        exited_tracks = []

        # --------------------------------------------------------
        # Check every active track
        # --------------------------------------------------------

        for track_id in list(
            self.active_tracks.keys()
        ):

            # Track is currently visible
            if track_id in current_track_ids:

                continue

            # Track is missing
            self.active_tracks[
                track_id
            ]["missing_frames"] += 1

            missing = self.active_tracks[
                track_id
            ]["missing_frames"]

            # ----------------------------------------------------
            # Close track after maximum missing frames
            # ----------------------------------------------------

            if missing >= self.max_missing_frames:

                self.close_track(
                    track_id
                )

                exited_tracks.append(
                    track_id
                )

        return exited_tracks

    # ============================================================
    # CLOSE TRACK
    # ============================================================

    def close_track(
        self,
        track_id
    ):
        """
        Close an active tracking session.

        Creates:
            1. EXIT image
            2. EXIT event
            3. Database track closure
        """

        # --------------------------------------------------------
        # Track does not exist
        # --------------------------------------------------------

        if track_id not in self.active_tracks:

            return

        state = self.active_tracks[
            track_id
        ]

        face_id = state[
            "face_id"
        ]

        # --------------------------------------------------------
        # ONE timestamp for EXIT
        # --------------------------------------------------------

        end_time = datetime.now().isoformat()

        # --------------------------------------------------------
        # Get latest face crop
        # --------------------------------------------------------

        crop = self.last_crops.get(
            track_id
        )

        # --------------------------------------------------------
        # Save EXIT image
        # --------------------------------------------------------

        image_path = None

        if self.save_exit_images:

            image_path = self.save_image(
                crop,
                "logs/exits",
                face_id,
                "exit"
            )

        # --------------------------------------------------------
        # Add EXIT event
        # --------------------------------------------------------

        self.database.add_event(
            face_id=face_id,
            track_id=track_id,
            event_type="EXIT",
            image_path=image_path,
            confidence=state[
                "confidence"
            ],
            timestamp=end_time
        )

        # --------------------------------------------------------
        # Close database tracking session
        # --------------------------------------------------------

        self.database.close_track(
            track_id,
            end_time=end_time
        )

        # --------------------------------------------------------
        # Logger
        # --------------------------------------------------------

        self.logger.exit(
            face_id,
            track_id
        )

        self.logger.track_exit(
            track_id
        )

        print(
            f"EXIT -> {face_id} "
            f"(Track {track_id})"
        )

        # --------------------------------------------------------
        # Remove active state
        # --------------------------------------------------------

        del self.active_tracks[
            track_id
        ]

        self.track_to_face.pop(
            track_id,
            None
        )

        self.track_embeddings.pop(
            track_id,
            None
        )

        self.track_attempts.pop(
            track_id,
            None
        )

        self.last_crops.pop(
            track_id,
            None
        )

    # ============================================================
    # CROP FACE
    # ============================================================

    def crop_face(
        self,
        frame,
        bbox
    ):
        """
        Crop face with 30% margin.

        Returns:
            OpenCV image crop
            or None if invalid.
        """

        # --------------------------------------------------------
        # Validate input
        # --------------------------------------------------------

        if frame is None:

            return None

        if bbox is None:

            return None

        # --------------------------------------------------------
        # Frame dimensions
        # --------------------------------------------------------

        height, width = frame.shape[:2]

        # --------------------------------------------------------
        # Bounding box
        # --------------------------------------------------------

        x1, y1, x2, y2 = bbox

        # Convert to integer coordinates
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # --------------------------------------------------------
        # Face dimensions
        # --------------------------------------------------------

        face_width = x2 - x1
        face_height = y2 - y1

        if face_width <= 0 or face_height <= 0:

            return None

        # --------------------------------------------------------
        # Add margin
        # --------------------------------------------------------

        margin_x = int(
            face_width * 0.30
        )

        margin_y = int(
            face_height * 0.30
        )

        x1 = max(
            0,
            x1 - margin_x
        )

        y1 = max(
            0,
            y1 - margin_y
        )

        x2 = min(
            width,
            x2 + margin_x
        )

        y2 = min(
            height,
            y2 + margin_y
        )

        # --------------------------------------------------------
        # Validate crop
        # --------------------------------------------------------

        if x2 <= x1 or y2 <= y1:

            return None

        # --------------------------------------------------------
        # Extract crop
        # --------------------------------------------------------

        crop = frame[
            y1:y2,
            x1:x2
        ].copy()

        if crop.size == 0:

            return None

        return crop
