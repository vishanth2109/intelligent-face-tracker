import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self):
        print("Loading InsightFace model...")

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("InsightFace model loaded!")

    def get_faces(self, frame):
        """
        Detect all faces in the FULL frame using InsightFace.

        Returns:
            List of InsightFace Face objects.
        """
        if frame is None:
            return []

        if len(frame.shape) != 3:
            return []

        faces = self.app.get(frame)

        return faces

    def normalize_embedding(self, embedding):
        """
        Normalize an embedding for cosine similarity.
        """
        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        return embedding / norm

    def bbox_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.

        Format:
        [x1, y1, x2, y2]
        """

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0, x2 - x1)
        intersection_height = max(0, y2 - y1)

        intersection = (
            intersection_width *
            intersection_height
        )

        area1 = max(0, box1[2] - box1[0]) * \
                max(0, box1[3] - box1[1])

        area2 = max(0, box2[2] - box2[0]) * \
                max(0, box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def center_inside(self, insight_box, yolo_box):
        """
        Check whether the center of an InsightFace
        bounding box is inside the YOLO bounding box.
        """

        cx = (insight_box[0] + insight_box[2]) / 2
        cy = (insight_box[1] + insight_box[3]) / 2

        return (
            yolo_box[0] <= cx <= yolo_box[2]
            and
            yolo_box[1] <= cy <= yolo_box[3]
        )

    def match_face_to_bbox(
        self,
        yolo_bbox,
        insight_faces
    ):
        """
        Match a YOLO face detection with the
        corresponding InsightFace detection.

        Returns:
            normalized embedding, or None
        """

        if not insight_faces:
            return None

        best_face = None
        best_score = 0.0

        for face in insight_faces:

            insight_bbox = [
                float(face.bbox[0]),
                float(face.bbox[1]),
                float(face.bbox[2]),
                float(face.bbox[3])
            ]

            iou = self.bbox_iou(
                yolo_bbox,
                insight_bbox
            )

            inside = self.center_inside(
                insight_bbox,
                yolo_bbox
            )

            # Center containment helps when
            # YOLO and InsightFace boxes have
            # different sizes.
            score = iou

            if inside:
                score += 0.25

            if score > best_score:
                best_score = score
                best_face = face

        if best_face is None:
            return None

        # Require reasonable spatial correspondence.
        if best_score < 0.10:
            return None

        # InsightFace already produces an ArcFace-style
        # embedding after its internal face alignment.
        if hasattr(best_face, "normed_embedding"):
            embedding = best_face.normed_embedding
        else:
            embedding = best_face.embedding

        return self.normalize_embedding(embedding)



