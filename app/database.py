import os
import sqlite3
import numpy as np
from datetime import datetime


class Database:

    def __init__(
        self,
        db_path="database/visitors.db"
    ):
        self.db_path = db_path

        # Make sure database directory exists
        db_directory = os.path.dirname(
            self.db_path
        )

        if db_directory:
            os.makedirs(
                db_directory,
                exist_ok=True
            )

        self.create_tables()

    # ============================================================
    # DATABASE CONNECTION
    # ============================================================

    def connect(self):
        return sqlite3.connect(
            self.db_path
        )

    # ============================================================
    # CREATE TABLES
    # ============================================================

    def create_tables(self):

        conn = self.connect()

        try:

            cursor = conn.cursor()

            # ----------------------------------------------------
            # Persistent visitor identities
            # ----------------------------------------------------

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT UNIQUE NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            # ----------------------------------------------------
            # Entry / Exit events
            # ----------------------------------------------------

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT NOT NULL,
                    track_id INTEGER,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_path TEXT,
                    confidence REAL
                )
            """)

            # ----------------------------------------------------
            # Tracking sessions
            # ----------------------------------------------------

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    face_id TEXT,
                    track_id INTEGER NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT
                )
            """)

            conn.commit()

        finally:

            conn.close()

    # ============================================================
    # GENERATE FACE ID
    # ============================================================

    def generate_face_id(self):

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM persons
            """)

            count = cursor.fetchone()[0]

        finally:

            conn.close()

        return f"FACE_{count + 1:04d}"

    # ============================================================
    # REGISTER NEW FACE
    # ============================================================

    def register_face(
        self,
        embedding
    ):

        now = datetime.now().isoformat()

        face_id = self.generate_face_id()

        # Convert embedding to float32
        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        embedding_blob = (
            embedding.tobytes()
        )

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO persons
                (
                    face_id,
                    first_seen,
                    last_seen,
                    embedding,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                face_id,
                now,
                now,
                embedding_blob,
                now
            ))

            conn.commit()

        finally:

            conn.close()

        return face_id

    # ============================================================
    # GET ALL REGISTERED FACES
    # ============================================================

    def get_all_faces(self):

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                SELECT face_id, embedding
                FROM persons
            """)

            rows = cursor.fetchall()

        finally:

            conn.close()

        faces = []

        for face_id, embedding_blob in rows:

            embedding = np.frombuffer(
                embedding_blob,
                dtype=np.float32
            ).copy()

            faces.append({
                "face_id": face_id,
                "embedding": embedding
            })

        return faces

    # ============================================================
    # UPDATE LAST SEEN
    # ============================================================

    def update_last_seen(
        self,
        face_id
    ):

        now = datetime.now().isoformat()

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                UPDATE persons
                SET last_seen = ?
                WHERE face_id = ?
            """, (
                now,
                face_id
            ))

            conn.commit()

        finally:

            conn.close()

    # ============================================================
    # ADD EVENT
    # ============================================================

    def add_event(
        self,
        face_id,
        track_id,
        event_type,
        image_path=None,
        confidence=None,
        timestamp=None
    ):
        """
        Add ENTRY or EXIT event.

        timestamp can optionally be supplied so that
        VisitorManager can use the exact same timestamp
        for the track session.
        """

        if timestamp is None:

            timestamp = datetime.now().isoformat()

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO events
                (
                    face_id,
                    track_id,
                    event_type,
                    timestamp,
                    image_path,
                    confidence
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                face_id,
                track_id,
                event_type,
                timestamp,
                image_path,
                confidence
            ))

            conn.commit()

            return cursor.lastrowid

        finally:

            conn.close()

    # ============================================================
    # ADD TRACK
    # ============================================================

    def add_track(
        self,
        face_id,
        track_id,
        start_time,
        status="ACTIVE"
    ):
        """
        Create a tracking session.

        start_time is required.
        """

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO tracks
                (
                    face_id,
                    track_id,
                    start_time,
                    status
                )
                VALUES (?, ?, ?, ?)
            """, (
                face_id,
                track_id,
                start_time,
                status
            ))

            conn.commit()

            return cursor.lastrowid

        finally:

            conn.close()

    # ============================================================
    # CLOSE TRACK
    # ============================================================

    def close_track(
        self,
        track_id,
        end_time=None
    ):
        """
        Close an active tracking session.
        """

        if end_time is None:

            end_time = datetime.now().isoformat()

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                UPDATE tracks
                SET
                    end_time = ?,
                    status = 'EXITED'
                WHERE track_id = ?
                  AND status = 'ACTIVE'
            """, (
                end_time,
                track_id
            ))

            conn.commit()

            return cursor.rowcount

        finally:

            conn.close()

    # ============================================================
    # GET ACTIVE TRACK
    # ============================================================

    def get_active_track(
        self,
        track_id
    ):

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    id,
                    face_id,
                    track_id,
                    start_time,
                    end_time,
                    status
                FROM tracks
                WHERE track_id = ?
                  AND status = 'ACTIVE'
                ORDER BY id DESC
                LIMIT 1
            """, (
                track_id,
            ))

            row = cursor.fetchone()

        finally:

            conn.close()

        if row is None:
            return None

        return {
            "id": row[0],
            "face_id": row[1],
            "track_id": row[2],
            "start_time": row[3],
            "end_time": row[4],
            "status": row[5]
        }

    # ============================================================
    # GET VISITOR HISTORY
    # ============================================================

    def get_visitor_history(
        self,
        face_id
    ):

        conn = self.connect()

        try:

            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    track_id,
                    start_time,
                    end_time,
                    status
                FROM tracks
                WHERE face_id = ?
                ORDER BY id DESC
            """, (
                face_id,
            ))

            rows = cursor.fetchall()

        finally:

            conn.close()

        history = []

        for row in rows:

            history.append({
                "track_id": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "status": row[3]
            })

        return history

    # ============================================================
    # GET EVENTS
    # ============================================================

    def get_events(
        self,
        face_id=None
    ):

        conn = self.connect()

        try:

            cursor = conn.cursor()

            if face_id is None:

                cursor.execute("""
                    SELECT
                        id,
                        face_id,
                        track_id,
                        event_type,
                        timestamp,
                        image_path,
                        confidence
                    FROM events
                    ORDER BY id DESC
                """)

            else:

                cursor.execute("""
                    SELECT
                        id,
                        face_id,
                        track_id,
                        event_type,
                        timestamp,
                        image_path,
                        confidence
                    FROM events
                    WHERE face_id = ?
                    ORDER BY id DESC
                """, (
                    face_id,
                ))

            rows = cursor.fetchall()

        finally:

            conn.close()

        events = []

        for row in rows:

            events.append({
                "id": row[0],
                "face_id": row[1],
                "track_id": row[2],
                "event_type": row[3],
                "timestamp": row[4],
                "image_path": row[5],
                "confidence": row[6]
            })

        return events
