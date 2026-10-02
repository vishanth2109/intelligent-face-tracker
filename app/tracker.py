from ultralytics import YOLO


class FaceTracker:

    def __init__(
        self,
        model_path,
        confidence=0.5,
        image_size=1280,
        tracker="bytetrack.yaml",
        device="cpu"
    ):

        print("Loading YOLO + ByteTrack...")

        self.model = YOLO(model_path)

        self.confidence = confidence
        self.image_size = image_size
        self.tracker_config = tracker
        self.device = device

        print(f"YOLO + ByteTrack loaded!")
        print(f"Device: {self.device}")

    def track(self, frame):

        results = self.model.track(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            tracker=self.tracker_config,
            persist=True,
            device=self.device,
            verbose=False
        )

        tracks = []

        for result in results:

            if result.boxes is None:
                continue

            if result.boxes.id is None:
                continue

            boxes = result.boxes.xyxy.cpu().tolist()

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )

            for bbox, track_id, confidence in zip(
                boxes,
                track_ids,
                confidences
            ):

                x1, y1, x2, y2 = bbox

                tracks.append({
                    "track_id": int(track_id),

                    "bbox": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ],

                    "confidence": float(confidence)
                })

        return tracks
