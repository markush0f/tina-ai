# utils/yolo_loader.py
from typing import Optional

import torch
from ultralytics.models import YOLO

from app.core.logger import get_logger

logger = get_logger(__name__)


class YoloModelRegistry:
    """
    Centralized and lazy-loaded registry for the YOLO object detection model.
    """

    # Load model
    _model: Optional[YOLO] = None

    # CUDA OR CPU
    _device: Optional[str] = None

    # UPLOAD MODEL OF HUGGINGFACE
    _model_id: str = "yolov8s.pt"

    @classmethod
    def get(cls) -> tuple[YOLO, str]:
        """
        Return the YOLO model and target device, loading them if needed.
        """
        if cls._model is not None and cls._device is not None:
            return cls._model, cls._device

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading YOLO model '%s' on device '%s'", cls._model_id, device)

        # Load yolo model
        model = YOLO(cls._model_id)
        model.to(device)

        cls._model = model
        cls._device = device

        logger.info("YOLO model loaded successfully")

        return model, device

    @classmethod
    def model_id(cls) -> str:
        """
        Expose the configured YOLO model identifier.
        """
        return cls._model_id
