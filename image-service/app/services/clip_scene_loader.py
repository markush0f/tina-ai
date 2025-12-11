from typing import Optional, Tuple
import torch
from transformers import CLIPProcessor, CLIPModel

from app.core.logger import get_logger

logger = get_logger(__name__)


class ClipSceneModelRegistry:
    """
    Lazy-loaded registry for CLIP used for scene classification.
    Loads the processor and model only once per process.
    """

    _processor: Optional[CLIPProcessor] = None
    _model: Optional[CLIPModel] = None
    _device: Optional[str] = None

    _model_id: str = "openai/clip-vit-base-patch32"

    @classmethod
    def get(cls) -> Tuple[CLIPProcessor, CLIPModel, str]:
        if cls._processor and cls._model and cls._device:
            return cls._processor, cls._model, cls._device

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading CLIP model '%s' on device '%s'", cls._model_id, device)

        processor = CLIPProcessor.from_pretrained(
            cls._model_id,
            use_fast=True,
        )
        model = CLIPModel.from_pretrained(cls._model_id).to(device)  # type: ignore

        model.eval()

        cls._processor = processor
        cls._model = model
        cls._device = device

        logger.info("CLIP model loaded successfully")

        return processor, model, device

    @classmethod
    def model_id(cls) -> str:
        return cls._model_id
