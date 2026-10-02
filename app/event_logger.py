import logging
import os


class EventLogger:

    def __init__(self, log_file="logs/events.log"):

        self.log_file = log_file

        # Create log directory
        log_directory = os.path.dirname(log_file)

        if log_directory:
            os.makedirs(
                log_directory,
                exist_ok=True
            )

        self.logger = logging.getLogger(
            "FaceTracker"
        )

        self.logger.setLevel(
            logging.INFO
        )

        # Avoid duplicate handlers
        if not self.logger.handlers:

            formatter = logging.Formatter(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            file_handler = logging.FileHandler(
                log_file,
                encoding="utf-8"
            )

            file_handler.setFormatter(
                formatter
            )

            self.logger.addHandler(
                file_handler
            )

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def face_detected(
        self,
        track_id
    ):
        self.info(
            f"FACE_DETECTED | "
            f"track={track_id}"
        )

    def embedding_generated(
        self,
        track_id
    ):
        self.info(
            f"EMBEDDING_GENERATED | "
            f"track={track_id}"
        )

    def new_face_registered(
        self,
        face_id
    ):
        self.info(
            f"NEW_FACE_REGISTERED | "
            f"face_id={face_id}"
        )

    def face_recognized(
        self,
        face_id,
        similarity
    ):
        self.info(
            f"FACE_RECOGNIZED | "
            f"face_id={face_id} | "
            f"similarity={similarity:.3f}"
        )

    def entry(
        self,
        face_id,
        track_id
    ):
        self.info(
            f"ENTRY | "
            f"face_id={face_id} | "
            f"track={track_id}"
        )

    def exit(
        self,
        face_id,
        track_id
    ):
        self.info(
            f"EXIT | "
            f"face_id={face_id} | "
            f"track={track_id}"
        )

    def track_exit(
        self,
        track_id
    ):
        self.info(
            f"TRACK_EXIT | "
            f"track={track_id}"
        )