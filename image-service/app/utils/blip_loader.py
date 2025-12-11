from typing import Optional, Tuple
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from app.core.logger import get_logger

logger = get_logger(__name__)


class BlipModelRegistry:
    """
    Centralized and lazy-loaded registry for the BLIP captioning model.

    This loads the BLIP (NOT BLIP-2) processor and model only once per process
    to minimize memory usage and startup time.
    """

    _processor: Optional[BlipProcessor] = None
    _model: Optional[BlipForConditionalGeneration] = None
    _device: Optional[str] = None

    # Name of the BLIP model in HuggingFace
    _model_id: str = "Salesforce/blip-image-captioning-large"

    # More ligth:
    # _model_id: str = "Salesforce/blip-image-captioning-base"

    @classmethod
    def get(cls) -> Tuple[BlipProcessor, BlipForConditionalGeneration, str]:
        """
        Return the BLIP processor, model, and device.
        Load them into memory if not already loaded.
        """

        if cls._processor and cls._model and cls._device:
            return cls._processor, cls._model, cls._device

        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading BLIP model '%s' on device '%s'", cls._model_id, device)

        processor = BlipProcessor.from_pretrained(
            cls._model_id,
            use_fast=True,
        )

        model = BlipForConditionalGeneration.from_pretrained(
            cls._model_id,
            dtype=dtype,
        ).to(
            device  # type: ignore
        )

        model.eval()

        cls._processor = processor
        cls._model = model
        cls._device = device

        logger.info("BLIP model loaded successfully")

        return processor, model, device

    @classmethod
    def model_id(cls) -> str:
        """Expose the configured BLIP model identifier."""
        return cls._model_id
