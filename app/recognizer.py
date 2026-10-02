import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:

    def __init__(
        self,
        model_name="buffalo_l",
        similarity_threshold=0.45,
        det_size=(640, 640)
    ):
        print("Loading InsightFace model...")

        self.similarity_threshold = similarity_threshold

        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=det_size
        )

        print("InsightFace model loaded!")

    # =========================================================
    # DETECT FACES
    # =========================================================

    def get_faces(self, frame):
        """
        Detect faces in an image using InsightFace.

        Returns a list of InsightFace Face objects.
        """

        if frame is None:
            return []

        try:
            faces = self.app.get(frame)

            if faces is None:
                return []

            return list(faces)

        except Exception as error:
            print(
                f"Warning: InsightFace detection failed: {error}"
            )
            return []

    # =========================================================
    # NORMALIZE EMBEDDING
    # =========================================================

    def normalize_embedding(self, embedding):
        """
        Normalize a face embedding.
        """

        if embedding is None:
            return None

        try:
            embedding = np.asarray(
                embedding,
                dtype=np.float32
            )

            norm = np.linalg.norm(embedding)

            if norm < 1e-8:
                return None

            return embedding / norm

        except Exception:
            return None

    # =========================================================
    # BOUNDING BOX IOU
    # =========================================================

    def bbox_iou(self, box_a, box_b):
        """
        Calculate Intersection over Union between two boxes.

        box format:
        [x1, y1, x2, y2]
        """

        if box_a is None or box_b is None:
            return 0.0

        try:
            ax1, ay1, ax2, ay2 = map(
                float,
                box_a
            )

            bx1, by1, bx2, by2 = map(
                float,
                box_b
            )

        except Exception:
            return 0.0

        # Intersection
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)

        intersection = iw * ih

        # Areas
        area_a = max(0.0, ax2 - ax1) * max(
            0.0,
            ay2 - ay1
        )

        area_b = max(0.0, bx2 - bx1) * max(
            0.0,
            by2 - by1
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    # =========================================================
    # CENTER INSIDE
    # =========================================================

    def center_inside(self, face_bbox, track_bbox):
        """
        Check whether the center of the InsightFace bounding
        box lies inside the YOLO/ByteTrack bounding box.
        """

        if face_bbox is None or track_bbox is None:
            return False

        try:
            fx1, fy1, fx2, fy2 = map(
                float,
                face_bbox
            )

            tx1, ty1, tx2, ty2 = map(
                float,
                track_bbox
            )

        except Exception:
            return False

        center_x = (fx1 + fx2) / 2.0
        center_y = (fy1 + fy2) / 2.0

        return (
            tx1 <= center_x <= tx2
            and
            ty1 <= center_y <= ty2
        )

    # =========================================================
    # GET FACE BBOX
    # =========================================================

    def get_face_bbox(self, face):
        """
        Safely extract bbox from an InsightFace Face object.
        """

        if face is None:
            return None

        try:
            bbox = face.bbox

            if bbox is None:
                return None

            bbox = np.asarray(
                bbox,
                dtype=np.float32
            ).reshape(-1)

            if len(bbox) < 4:
                return None

            return [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3])
            ]

        except Exception:
            return None

    # =========================================================
    # MATCH INSIGHTFACE FACE TO TRACK BBOX
    # =========================================================

    def match_face_to_bbox(self, faces, track_bbox):
        """
        Find the InsightFace Face object corresponding to
        a YOLO/ByteTrack bounding box.

        Returns:
            InsightFace Face object

        or:
            None
        """

        if faces is None:
            return None

        if track_bbox is None:
            return None

        # -----------------------------------------------------
        # Make sure faces is actually a list of Face objects
        # -----------------------------------------------------

        try:
            faces = list(faces)
        except Exception:
            return None

        if len(faces) == 0:
            return None

        best_face = None
        best_score = 0.0

        for face in faces:

            # -------------------------------------------------
            # Important:
            # InsightFace returns Face objects.
            # Do not treat them as dictionaries/lists.
            # -------------------------------------------------

            face_bbox = self.get_face_bbox(face)

            if face_bbox is None:
                continue

            # IoU between InsightFace bbox and tracker bbox
            iou = self.bbox_iou(
                face_bbox,
                track_bbox
            )

            # -------------------------------------------------
            # Center-inside bonus
            # -------------------------------------------------

            score = iou

            if self.center_inside(
                face_bbox,
                track_bbox
            ):
                score += 0.25

            if score > best_score:

                best_score = score
                best_face = face

        # -----------------------------------------------------
        # Minimum matching score
        # -----------------------------------------------------

        if best_face is None:
            return None

        if best_score < 0.10:
            return None

        return best_face

    # =========================================================
    # GET EMBEDDING
    # =========================================================

    def get_embedding(self, face):
        """
        Extract and normalize embedding from an InsightFace
        Face object.
        """

        if face is None:
            return None

        try:
            embedding = face.embedding

        except Exception:
            return None

        return self.normalize_embedding(
            embedding
        )

    # =========================================================
    # FACE EMBEDDING FROM BBOX
    # =========================================================

    def match_embedding_to_bbox(
        self,
        faces,
        track_bbox
    ):
        """
        Convenience method.

        Finds the matching InsightFace face and returns its
        normalized embedding.
        """

        face = self.match_face_to_bbox(
            faces,
            track_bbox
        )

        if face is None:
            return None

        return self.get_embedding(face)





