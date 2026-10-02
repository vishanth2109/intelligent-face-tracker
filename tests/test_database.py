import numpy as np

from app.database import Database


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def test_database_creates_tables(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    conn = database.connect()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
    """)

    tables = {
        row[0]
        for row in cursor.fetchall()
    }

    conn.close()

    assert "persons" in tables
    assert "events" in tables
    assert "tracks" in tables


# ============================================================
# FACE REGISTRATION
# ============================================================

def test_register_face(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id = database.register_face(
        embedding
    )

    assert face_id == "FACE_0001"


def test_multiple_face_registration(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding1 = np.random.rand(
        512
    ).astype(
        np.float32
    )

    embedding2 = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id1 = database.register_face(
        embedding1
    )

    face_id2 = database.register_face(
        embedding2
    )

    assert face_id1 == "FACE_0001"
    assert face_id2 == "FACE_0002"


# ============================================================
# EMBEDDING STORAGE
# ============================================================

def test_get_all_faces(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id = database.register_face(
        embedding
    )

    faces = database.get_all_faces()

    assert len(faces) == 1
    assert faces[0]["face_id"] == face_id

    stored_embedding = faces[0][
        "embedding"
    ]

    assert isinstance(
        stored_embedding,
        np.ndarray
    )

    assert stored_embedding.shape == (
        512,
    )


# ============================================================
# UPDATE LAST SEEN
# ============================================================

def test_update_last_seen(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id = database.register_face(
        embedding
    )

    database.update_last_seen(
        face_id
    )

    faces = database.get_all_faces()

    assert len(faces) == 1


# ============================================================
# EVENT INSERTION
# ============================================================

def test_add_event(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id = database.register_face(
        embedding
    )

    event_id = database.add_event(
        face_id=face_id,
        track_id=10,
        event_type="ENTRY",
        image_path="logs/entries/test.jpg",
        confidence=0.95
    )

    assert event_id == 1

    events = database.get_events()

    assert len(events) == 1
    assert events[0]["face_id"] == face_id
    assert events[0]["track_id"] == 10
    assert events[0]["event_type"] == "ENTRY"


# ============================================================
# TRACK CREATION
# ============================================================

def test_add_and_get_active_track(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id = database.register_face(
        embedding
    )

    track_id = database.add_track(
        face_id=face_id,
        track_id=100,
        start_time="2026-10-02T10:00:00",
        status="ACTIVE"
    )

    assert track_id == 1

    track = database.get_active_track(
        100
    )

    assert track is not None
    assert track["face_id"] == face_id
    assert track["track_id"] == 100
    assert track["status"] == "ACTIVE"


# ============================================================
# TRACK CLOSURE
# ============================================================

def test_close_track(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id = database.register_face(
        embedding
    )

    database.add_track(
        face_id=face_id,
        track_id=200,
        start_time="2026-10-02T10:00:00"
    )

    rows_updated = database.close_track(
        200,
        end_time="2026-10-02T10:05:00"
    )

    assert rows_updated == 1

    active_track = database.get_active_track(
        200
    )

    assert active_track is None


# ============================================================
# VISITOR HISTORY
# ============================================================

def test_visitor_history(tmp_path):

    db_path = tmp_path / "test_visitors.db"

    database = Database(
        str(db_path)
    )

    embedding = np.random.rand(
        512
    ).astype(
        np.float32
    )

    face_id = database.register_face(
        embedding
    )

    database.add_track(
        face_id=face_id,
        track_id=300,
        start_time="2026-10-02T10:00:00"
    )

    database.close_track(
        300,
        end_time="2026-10-02T10:10:00"
    )

    history = database.get_visitor_history(
        face_id
    )

    assert len(history) == 1
    assert history[0]["track_id"] == 300
    assert history[0]["status"] == "EXITED"