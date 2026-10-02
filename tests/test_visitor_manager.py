import cv2
import numpy as np

from app.database import Database
from app.visitor_manager import VisitorManager


# ============================================================
# MOCK LOGGER
# ============================================================

class MockLogger:

    def face_recognized(self, *args, **kwargs):
        pass

    def new_face_registered(self, *args, **kwargs):
        pass

    def entry(self, *args, **kwargs):
        pass

    def exit(self, *args, **kwargs):
        pass

    def track_exit(self, *args, **kwargs):
        pass


# ============================================================
# MOCK RECOGNIZER
# ============================================================

class MockRecognizer:

    def __init__(self):
        self.embedding = np.ones(
            512,
            dtype=np.float32
        )

    def match_face_to_bbox(
        self,
        face,
        bbox
    ):
        return self.embedding


# ============================================================
# TEST HELPERS
# ============================================================

def create_manager(
    tmp_path,
    max_missing_frames=50
):

    db_path = (
        tmp_path /
        "test.db"
    )

    database = Database(
        str(db_path)
    )

    logger = MockLogger()

    recognizer = MockRecognizer()

    manager = VisitorManager(
        database=database,
        logger=logger,
        recognizer=recognizer,
        similarity_threshold=0.45,
        max_missing_frames=max_missing_frames,
        save_entry_images=False,
        save_exit_images=False
    )

    return (
        manager,
        database,
        logger
    )


def create_frame():

    return np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )


def create_track(
    track_id=1,
    bbox=None,
    confidence=0.95
):

    if bbox is None:

        bbox = [
            100,
            100,
            250,
            300
        ]

    return {
        "track_id": track_id,
        "bbox": bbox,
        "confidence": confidence
    }


def register_new_face(
    manager,
    frame,
    track,
    attempts=3
):

    face_id = None

    for _ in range(attempts):

        face_id = manager.process_track(
            frame,
            track,
            ["mock_face"]
        )

    return face_id


# ============================================================
# NEW FACE REGISTRATION
# ============================================================

def test_new_face_registration(
    tmp_path
):

    manager, database, logger = (
        create_manager(tmp_path)
    )

    frame = create_frame()

    track = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track
    )

    assert face_id == "FACE_0001"

    faces = database.get_all_faces()

    assert len(faces) == 1

    assert faces[0]["face_id"] == "FACE_0001"


# ============================================================
# ENTRY EVENT
# ============================================================

def test_entry_event_created(
    tmp_path
):

    manager, database, logger = (
        create_manager(tmp_path)
    )

    frame = create_frame()

    track = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track
    )

    assert face_id == "FACE_0001"

    events = database.get_events()

    assert len(events) == 1

    assert events[0]["event_type"] == "ENTRY"

    assert events[0]["face_id"] == "FACE_0001"

    assert events[0]["track_id"] == 1


# ============================================================
# DUPLICATE ENTRY PREVENTION
# ============================================================

def test_duplicate_entry_prevention(
    tmp_path
):

    manager, database, logger = (
        create_manager(tmp_path)
    )

    frame = create_frame()

    track = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track
    )

    assert face_id == "FACE_0001"

    # Same track again
    manager.process_track(
        frame,
        track,
        ["mock_face"]
    )

    events = database.get_events()

    entry_events = [
        event
        for event in events
        if event["event_type"] == "ENTRY"
    ]

    assert len(entry_events) == 1


# ============================================================
# UPDATE TRACK
# ============================================================

def test_update_track(
    tmp_path
):

    manager, database, logger = (
        create_manager(tmp_path)
    )

    frame = create_frame()

    track = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track
    )

    assert face_id == "FACE_0001"

    # Simulate missing frames
    manager.active_tracks[
        1
    ]["missing_frames"] = 2

    # Track appears again
    new_bbox = [
        120,
        120,
        270,
        320
    ]

    manager.update_track(
        1,
        frame,
        new_bbox
    )

    assert (
        manager.active_tracks[1]
        ["missing_frames"]
        == 0
    )

    assert (
        manager.active_tracks[1]
        ["face_id"]
        == "FACE_0001"
    )

    assert (
        manager.track_to_face[1]
        == "FACE_0001"
    )


# ============================================================
# EXIT EVENT
# ============================================================

def test_exit_event(
    tmp_path
):

    manager, database, logger = (
        create_manager(tmp_path)
    )

    frame = create_frame()

    track = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track
    )

    assert face_id == "FACE_0001"

    manager.close_track(1)

    events = database.get_events()

    assert len(events) == 2

    event_types = [
        event["event_type"]
        for event in events
    ]

    assert "ENTRY" in event_types
    assert "EXIT" in event_types


# ============================================================
# DUPLICATE EXIT PREVENTION
# ============================================================

def test_duplicate_exit_prevention(
    tmp_path
):

    manager, database, logger = (
        create_manager(tmp_path)
    )

    frame = create_frame()

    track = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track
    )

    assert face_id == "FACE_0001"

    manager.close_track(1)

    # Second close must do nothing
    manager.close_track(1)

    events = database.get_events()

    exit_events = [
        event
        for event in events
        if event["event_type"] == "EXIT"
    ]

    assert len(exit_events) == 1


# ============================================================
# UNIQUE VISITOR COUNT
# ============================================================

def test_unique_visitor_count(
    tmp_path
):

    manager, database, logger = (
        create_manager(tmp_path)
    )

    frame = create_frame()

    track1 = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track1
    )

    assert face_id == "FACE_0001"

    assert len(
        manager.unique_visitors
    ) == 1

    # Close first appearance
    manager.close_track(1)

    # Same person appears with a new tracking ID
    track2 = create_track(2)

    face_id_2 = manager.process_track(
        frame,
        track2,
        ["mock_face"]
    )

    assert face_id_2 == "FACE_0001"

    # Must still be one unique visitor
    assert len(
        manager.unique_visitors
    ) == 1


# ============================================================
# MISSING TRACK -> EXIT
# ============================================================

def test_missing_track_causes_exit(
    tmp_path
):

    manager, database, logger = (
        create_manager(
            tmp_path,
            max_missing_frames=2
        )
    )

    frame = create_frame()

    track = create_track(1)

    face_id = register_new_face(
        manager,
        frame,
        track
    )

    assert face_id == "FACE_0001"

    assert 1 in manager.active_tracks

    # First missing frame
    manager.update_missing_tracks(
        set()
    )

    assert 1 in manager.active_tracks

    # Second missing frame
    manager.update_missing_tracks(
        set()
    )

    # Track should now be closed
    assert 1 not in manager.active_tracks

    events = database.get_events()

    event_types = [
        event["event_type"]
        for event in events
    ]

    assert event_types.count(
        "ENTRY"
    ) == 1

    assert event_types.count(
        "EXIT"
    ) == 1