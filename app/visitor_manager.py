import os
import cv2
import numpy as np
from datetime import datetime


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

        # TRACK_ID -> FACE_ID
        self.track_to_face = {}

        # TRACK_ID -> active track information
        self.active_tracks = {}

        # TRACK_ID -> collected embeddings
        self.track_embeddings = {}

        # TRACK_ID -> recognition attempts
        self.track_attempts = {}

        # TRACK_ID -> latest face crop
        self.last_crops = {}

        # Unique FACE_ID values seen during this run
        self.unique_visitors = set()

    # =========================================================
    # FACE CROPPING
    # =========================================================

    def crop_face(self, frame, bbox, margin=0.30):

        if frame is None:
            return None

        if bbox is None or len(bbox) != 4:
            return None

        try:
            height, width = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        face_width = x2 - x1
        face_height = y2 - y1

        margin_x = int(face_width * margin)
        margin_y = int(face_height * margin)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)

        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    # =========================================================
    # SAVE FACE IMAGE
    # =========================================================

    def save_face_image(
        self,
        frame,
        bbox,
        face_id,
        event_type
    ):

        if frame is None:
            return None

        crop = self.crop_face(
            frame,
            bbox
        )

        if crop is None:
            return None

        now = datetime.now()

        date_folder = now.strftime(
            "%Y-%m-%d"
        )

        timestamp = now.strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        event_type = event_type.upper()

        if event_type == "ENTRY":

            base_folder = os.path.join(
                "logs",
                "entries"
            )

        elif event_type == "EXIT":

            base_folder = os.path.join(
                "logs",
                "exits"
            )

        else:

            base_folder = os.path.join(
                "logs",
                event_type.lower()
            )

        folder = os.path.join(
            base_folder,
            date_folder
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

        filename = (
            f"{face_id}_{event_type}_{timestamp}.jpg"
        )

        image_path = os.path.join(
            folder,
            filename
        )

        try:

            success = cv2.imwrite(
                image_path,
                crop
            )

            if not success:
                return None

            return image_path

        except Exception as error:

            print(
                f"Warning: Could not save face image: {error}"
            )

            return None

    # =========================================================
    # NORMALIZE EMBEDDING
    # =========================================================

    def normalize_embedding(
        self,
        embedding
    ):

        if embedding is None:
            return None

        try:

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            )

            norm = np.linalg.norm(
                embedding
            )

            if norm < 1e-8:
                return None

            return embedding / norm

        except Exception:

            return None

    # =========================================================
    # EXTRACT EMBEDDING
    #
    # Supports:
    #
    # 1. InsightFace object
    # 2. Dictionary
    # 3. NumPy array
    # =========================================================

    def extract_embedding(
        self,
        matched_face
    ):

        if matched_face is None:
            return None

        # -----------------------------------------------------
        # InsightFace Face object
        # -----------------------------------------------------

        try:

            if hasattr(
                matched_face,
                "embedding"
            ):

                embedding = (
                    matched_face.embedding
                )

                return self.normalize_embedding(
                    embedding
                )

        except Exception:
            pass

        # -----------------------------------------------------
        # Dictionary
        # -----------------------------------------------------

        if isinstance(
            matched_face,
            dict
        ):

            embedding = (
                matched_face.get(
                    "embedding"
                )
            )

            return self.normalize_embedding(
                embedding
            )

        # -----------------------------------------------------
        # NumPy array / list
        # -----------------------------------------------------

        if isinstance(
            matched_face,
            (
                np.ndarray,
                list,
                tuple
            )
        ):

            return self.normalize_embedding(
                matched_face
            )

        return None

    # =========================================================
    # FIND BEST MATCH
    # =========================================================

    def find_best_match(
        self,
        embedding
    ):

        embedding = self.normalize_embedding(
            embedding
        )

        if embedding is None:
            return None, 0.0

        try:

            registered_faces = (
                self.database.get_all_faces()
            )

        except Exception as error:

            print(
                f"Warning: Could not retrieve "
                f"registered faces: {error}"
            )

            return None, 0.0

        best_face_id = None
        best_similarity = -1.0

        for face in registered_faces:

            try:

                # -------------------------------------------------
                # Dictionary database record
                # -------------------------------------------------

                if isinstance(
                    face,
                    dict
                ):

                    face_id = face.get(
                        "face_id"
                    )

                    stored_embedding = face.get(
                        "embedding"
                    )

                # -------------------------------------------------
                # Tuple/list database record
                # -------------------------------------------------

                elif isinstance(
                    face,
                    (tuple, list)
                ):

                    face_id = face[0]

                    stored_embedding = face[-1]

                else:

                    continue

                stored_embedding = (
                    self.normalize_embedding(
                        stored_embedding
                    )
                )

                if stored_embedding is None:
                    continue

                similarity = float(
                    np.dot(
                        embedding,
                        stored_embedding
                    )
                )

                if similarity > best_similarity:

                    best_similarity = similarity

                    best_face_id = face_id

            except Exception:

                continue

        if (
            best_face_id is not None
            and best_similarity >=
            self.similarity_threshold
        ):

            return (
                best_face_id,
                best_similarity
            )

        return (
            None,
            best_similarity
        )

    # =========================================================
    # REGISTER NEW FACE
    # =========================================================

    def register_new_face(
        self,
        embedding
    ):

        embedding = self.normalize_embedding(
            embedding
        )

        if embedding is None:
            return None

        try:

            face_id = (
                self.database.register_face(
                    embedding
                )
            )

            if face_id:

                self.unique_visitors.add(
                    face_id
                )

                try:

                    self.logger.registration(
                        face_id=face_id
                    )

                except Exception:

                    try:

                        self.logger.new_face_registered(
                            face_id
                        )

                    except Exception:
                        pass

            return face_id

        except Exception as error:

            print(
                f"Error registering new face: {error}"
            )

            return None

    # =========================================================
    # ASSIGN TRACK
    # =========================================================

    def assign_track(
        self,
        frame,
        track,
        face_id,
        confidence=0.0
    ):

        track_id = int(
            track["track_id"]
        )

        bbox = track["bbox"]

        # -----------------------------------------------------
        # Prevent duplicate ENTRY
        # -----------------------------------------------------

        if track_id in self.active_tracks:

            return self.track_to_face.get(
                track_id
            )

        # -----------------------------------------------------
        # Track -> FACE_ID
        # -----------------------------------------------------

        self.track_to_face[
            track_id
        ] = face_id

        self.unique_visitors.add(
            face_id
        )

        start_time = datetime.now()

        image_path = None

        # -----------------------------------------------------
        # ENTRY image
        # -----------------------------------------------------

        if self.save_entry_images:

            image_path = self.save_face_image(
                frame,
                bbox,
                face_id,
                "ENTRY"
            )

        # -----------------------------------------------------
        # ENTRY event
        # -----------------------------------------------------

        try:

            self.database.add_event(
                face_id=face_id,
                track_id=track_id,
                event_type="ENTRY",
                timestamp=start_time,
                image_path=image_path,
                confidence=confidence
            )

        except Exception as error:

            print(
                f"Warning: Could not save "
                f"ENTRY event: {error}"
            )

        # -----------------------------------------------------
        # Track database record
        # -----------------------------------------------------

        try:

            self.database.add_track(
                face_id=face_id,
                track_id=track_id,
                start_time=start_time
            )

        except Exception as error:

            print(
                f"Warning: Could not create "
                f"track record: {error}"
            )

        # -----------------------------------------------------
        # Runtime state
        # -----------------------------------------------------

        self.active_tracks[
            track_id
        ] = {

            "face_id": face_id,

            "start_time": start_time,

            "last_seen": start_time,

            "missing_frames": 0,

            "bbox": bbox,

            "confidence": confidence
        }

        # -----------------------------------------------------
        # Latest crop
        # -----------------------------------------------------

        crop = self.crop_face(
            frame,
            bbox
        )

        if crop is not None:

            self.last_crops[
                track_id
            ] = crop

        # -----------------------------------------------------
        # Logger
        # -----------------------------------------------------

        try:

            self.logger.entry(
                face_id=face_id,
                track_id=track_id
            )

        except Exception:
            pass

        print(
            f"Track {track_id} assigned "
            f"to {face_id} - ENTRY"
        )

        return face_id

    # =========================================================
    # PROCESS TRACK
    # =========================================================

    def process_track(
        self,
        frame,
        track,
        faces
    ):

        track_id = int(
            track["track_id"]
        )

        bbox = track["bbox"]

        confidence = float(
            track.get(
                "confidence",
                0.0
            )
        )

        # -----------------------------------------------------
        # Existing active track
        # -----------------------------------------------------

        if track_id in self.active_tracks:

            return self.update_track(
                track_id,
                frame,
                bbox
            )

        # -----------------------------------------------------
        # Match face to YOLO bbox
        # -----------------------------------------------------

        try:

            matched_face = (
                self.recognizer.match_face_to_bbox(
                    faces,
                    bbox
                )
            )

        except Exception as error:

            print(
                f"Warning: Face matching failed "
                f"for track {track_id}: {error}"
            )

            return None

        if matched_face is None:

            return None

        # -----------------------------------------------------
        # Extract embedding
        #
        # Supports both:
        #
        # InsightFace Face object
        # NumPy embedding
        # -----------------------------------------------------

        embedding = self.extract_embedding(
            matched_face
        )

        if embedding is None:

            return None

        # -----------------------------------------------------
        # Store embeddings
        # -----------------------------------------------------

        if track_id not in self.track_embeddings:

            self.track_embeddings[
                track_id
            ] = []

        embeddings = (
            self.track_embeddings[
                track_id
            ]
        )

        if len(embeddings) < 5:

            embeddings.append(
                embedding
            )

        # -----------------------------------------------------
        # Direct recognition
        # -----------------------------------------------------

        face_id, similarity = (
            self.find_best_match(
                embedding
            )
        )

        if face_id is not None:

            print(
                f"Track {track_id}: "
                f"Recognized {face_id} "
                f"(similarity={similarity:.3f})"
            )

            return self.assign_track(
                frame,
                track,
                face_id,
                confidence
            )

        # -----------------------------------------------------
        # Recognition attempts
        # -----------------------------------------------------

        attempts = self.track_attempts.get(
            track_id,
            0
        )

        attempts += 1

        self.track_attempts[
            track_id
        ] = attempts

        print(
            f"Track {track_id}: "
            f"Recognition attempt "
            f"{attempts}/3"
        )

        if attempts < 3:

            print(
                f"Track {track_id}: "
                f"No match yet. "
                f"Waiting for more embeddings..."
            )

            return None

        # -----------------------------------------------------
        # Make sure embeddings exist
        # -----------------------------------------------------

        if len(embeddings) == 0:

            return None

        # -----------------------------------------------------
        # Average embeddings
        # -----------------------------------------------------

        try:

            averaged_embedding = np.mean(
                np.vstack(
                    embeddings
                ),
                axis=0
            )

            averaged_embedding = (
                self.normalize_embedding(
                    averaged_embedding
                )
            )

        except Exception as error:

            print(
                f"Warning: Could not average "
                f"embeddings for track "
                f"{track_id}: {error}"
            )

            return None

        if averaged_embedding is None:

            return None

        # -----------------------------------------------------
        # Try matching averaged embedding
        # -----------------------------------------------------

        face_id, similarity = (
            self.find_best_match(
                averaged_embedding
            )
        )

        if face_id is not None:

            print(
                f"Track {track_id}: "
                f"Matched existing "
                f"{face_id} "
                f"after embedding aggregation "
                f"(similarity={similarity:.3f})"
            )

            return self.assign_track(
                frame,
                track,
                face_id,
                confidence
            )

        # -----------------------------------------------------
        # New visitor
        # -----------------------------------------------------

        face_id = self.register_new_face(
            averaged_embedding
        )

        if face_id is None:

            print(
                f"Track {track_id}: "
                f"Failed to register new face"
            )

            return None

        print(
            f"Track {track_id}: "
            f"New visitor registered as "
            f"{face_id}"
        )

        return self.assign_track(
            frame,
            track,
            face_id,
            confidence
        )

    # =========================================================
    # UPDATE EXISTING TRACK
    #
    # Matches main.py:
    #
    # update_track(track_id, frame, bbox)
    # =========================================================

    def update_track(
        self,
        track_id,
        frame,
        bbox
    ):

        try:

            track_id = int(
                track_id
            )

        except Exception:

            return None

        if track_id not in self.active_tracks:

            return None

        state = self.active_tracks[
            track_id
        ]

        face_id = state[
            "face_id"
        ]

        now = datetime.now()

        # -----------------------------------------------------
        # Update state
        # -----------------------------------------------------

        state[
            "missing_frames"
        ] = 0

        state[
            "last_seen"
        ] = now

        state[
            "bbox"
        ] = bbox

        # -----------------------------------------------------
        # Update latest crop
        # -----------------------------------------------------

        crop = self.crop_face(
            frame,
            bbox
        )

        if crop is not None:

            self.last_crops[
                track_id
            ] = crop

        # -----------------------------------------------------
        # Database
        # -----------------------------------------------------

        try:

            self.database.update_last_seen(
                face_id
            )

        except Exception as error:

            print(
                f"Warning: Could not update "
                f"last seen for {face_id}: "
                f"{error}"
            )

        return face_id

    # =========================================================
    # UPDATE MISSING TRACKS
    # =========================================================

    def update_missing_tracks(
        self,
        current_track_ids
    ):

        current_track_ids = {
            int(track_id)
            for track_id in current_track_ids
        }

        tracks_to_close = []

        for track_id in list(
            self.active_tracks.keys()
        ):

            if track_id not in current_track_ids:

                state = self.active_tracks[
                    track_id
                ]

                state[
                    "missing_frames"
                ] += 1

                if (
                    state["missing_frames"]
                    >= self.max_missing_frames
                ):

                    tracks_to_close.append(
                        track_id
                    )

        for track_id in tracks_to_close:

            self.close_track(
                track_id
            )

    # =========================================================
    # CLOSE TRACK
    # =========================================================

    def close_track(
        self,
        track_id
    ):

        try:

            track_id = int(
                track_id
            )

        except Exception:

            return

        if track_id not in self.active_tracks:

            return

        state = self.active_tracks[
            track_id
        ]

        face_id = state[
            "face_id"
        ]

        end_time = datetime.now()

        confidence = state.get(
            "confidence",
            0.0
        )

        image_path = None

        # -----------------------------------------------------
        # EXIT image
        # -----------------------------------------------------

        if self.save_exit_images:

            latest_crop = (
                self.last_crops.get(
                    track_id
                )
            )

            if (
                latest_crop is not None
                and isinstance(
                    latest_crop,
                    np.ndarray
                )
            ):

                try:

                    date_folder = (
                        end_time.strftime(
                            "%Y-%m-%d"
                        )
                    )

                    timestamp = (
                        end_time.strftime(
                            "%Y%m%d_%H%M%S_%f"
                        )
                    )

                    folder = os.path.join(
                        "logs",
                        "exits",
                        date_folder
                    )

                    os.makedirs(
                        folder,
                        exist_ok=True
                    )

                    filename = (
                        f"{face_id}_EXIT_"
                        f"{timestamp}.jpg"
                    )

                    image_path = os.path.join(
                        folder,
                        filename
                    )

                    success = cv2.imwrite(
                        image_path,
                        latest_crop
                    )

                    if not success:

                        image_path = None

                except Exception as error:

                    print(
                        f"Warning: Could not save "
                        f"EXIT image: {error}"
                    )

        # -----------------------------------------------------
        # EXIT event
        # -----------------------------------------------------

        try:

            self.database.add_event(
                face_id=face_id,
                track_id=track_id,
                event_type="EXIT",
                timestamp=end_time,
                image_path=image_path,
                confidence=confidence
            )

        except Exception as error:

            print(
                f"Warning: Could not save "
                f"EXIT event: {error}"
            )

        # -----------------------------------------------------
        # Close database track
        # -----------------------------------------------------

        try:

            self.database.close_track(
                track_id=track_id,
                end_time=end_time
            )

        except Exception as error:

            print(
                f"Warning: Could not close "
                f"database track: {error}"
            )

        # -----------------------------------------------------
        # Logger
        # -----------------------------------------------------

        try:

            self.logger.exit(
                face_id=face_id,
                track_id=track_id
            )

        except Exception:
            pass

        try:

            self.logger.track_exit(
                face_id=face_id,
                track_id=track_id
            )

        except Exception:
            pass

        print(
            f"Track {track_id} -> "
            f"{face_id} - EXIT"
        )

        # -----------------------------------------------------
        # Remove runtime state
        # -----------------------------------------------------

        self.active_tracks.pop(
            track_id,
            None
        )

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

    # =========================================================
    # ACTIVE TRACKS
    # =========================================================

    def get_active_tracks(self):

        return self.active_tracks.copy()

    # =========================================================
    # FACE ID
    # =========================================================

    def get_face_id(
        self,
        track_id
    ):

        try:

            track_id = int(
                track_id
            )

        except Exception:

            return None

        return self.track_to_face.get(
            track_id
        )

    # =========================================================
    # UNIQUE VISITOR COUNT
    # =========================================================

    def get_unique_visitor_count(self):

        return len(
            self.unique_visitors
        )

    # =========================================================
    # UNIQUE VISITORS
    # =========================================================

    def get_unique_visitors(self):

        return self.unique_visitors.copy()

    # =========================================================
    # CLOSE ALL TRACKS
    # =========================================================

    def close_all_tracks(self):

        active_track_ids = list(
            self.active_tracks.keys()
        )

        for track_id in active_track_ids:

            try:

                self.close_track(
                    track_id
                )

            except Exception as error:

                print(
                    f"Warning: Could not close "
                    f"track {track_id}: {error}"
                )

