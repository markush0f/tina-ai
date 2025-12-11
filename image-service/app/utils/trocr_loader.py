from typing import Optional
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.core.logger import get_logger

logger = get_logger(__name__)


class TrocrModelRegistry:
    """
    Lazy-loaded registry for the TrOCR model.
    Ensures processor & model are loaded a single time per process.
    """

    _processor: Optional[TrOCRProcessor] = None
    _model: Optional[VisionEncoderDecoderModel] = None
    _device: Optional[str] = None

    _model_id: str = "microsoft/trocr-base-printed"

    @classmethod
    def get(cls):
        # Already loaded
        if cls._processor and cls._model and cls._device:
            return cls._processor, cls._model, cls._device

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading TrOCR model '%s' on device '%s'", cls._model_id, device)

        processor = TrOCRProcessor.from_pretrained(
            cls._model_id,
            use_fast=True,
        )
        model = VisionEncoderDecoderModel.from_pretrained(cls._model_id).to(device)  # type: ignore
        model.eval()

        cls._processor = processor
        cls._model = model  # type: ignore
        cls._device = device

        logger.info("TrOCR model loaded successfully")

        return processor, model, device

    @classmethod
    def model_id(cls) -> str:
        return cls._model_id
