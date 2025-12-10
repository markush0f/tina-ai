# services/object_detection_service.py
from typing import List

from PIL import Image

from app.core.logger import get_logger
from app.utils.yolo_loader import YoloModelRegistry

logger = get_logger(__name__)


class ObjectDetectionService:
    """
    Service responsible for running object detection using YOLO.
    """

    def __init__(self, confidence_threshold: float = 0.35) -> None:
        self.model, self.device = YoloModelRegistry.get()
        self.model_id = YoloModelRegistry.model_id()
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image: Image.Image) -> List[str]:
        """
        Run YOLO on the given image and return a list of unique object names.
        """
        logger.info("Running YOLO object detection")

        # YOLO accepts PIL images directly.
        results = self.model(
            image,
            verbose=False,
            device=self.device,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None or result.boxes.cls is None:
            return []

        class_ids = result.boxes.cls.tolist()
        confidences = result.boxes.conf.tolist()
        names = result.names

        # Keep the best confidence per class id.
        best_conf_by_class: dict[int, float] = {}
        for cls_id, conf in zip(class_ids, confidences):
            if conf < self.confidence_threshold:
                continue
            current_best = best_conf_by_class.get(int(cls_id))
            if current_best is None or conf > current_best:
                best_conf_by_class[int(cls_id)] = float(conf)

        # Map selected class ids to human-readable labels.
        detected_objects = [
            names[cls_id] for cls_id in best_conf_by_class.keys()
        ]

        # Sort for deterministic output.
        detected_objects.sort()

        logger.info("YOLO detected objects: %s", detected_objects)

        return detected_objects
