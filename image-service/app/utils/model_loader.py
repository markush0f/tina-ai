# utils/model_loader.py
from typing import Optional, Tuple

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from app.core.logger import get_logger

logger = get_logger(__name__)


class Blip2ModelRegistry:
    """
    Centralized and lazy-loaded registry for the BLIP-2 model and processor.

    The model is loaded only once per process to reduce startup cost and memory usage.
    Access is provided via the `get()` method.
    """

    # SAVE PREPROCESSOR of BLIP-2
    _processor: Optional[Blip2Processor] = None
    
    # SAVE MODEL OF BLIP-2
    _model: Optional[Blip2ForConditionalGeneration] = None

    # CUDA OR CPU
    _device: Optional[str] = None

    # NAME OF MODEL IN HUGGINFACAE
    _model_id: str = "Salesforce/blip2-opt-2.7b"

    @classmethod
    def get(cls) -> Tuple[Blip2Processor, Blip2ForConditionalGeneration, str]:
        """
        Return the BLIP-2 processor, model and device.

        If not already loaded, load them into memory.
        """
        
        if cls._processor is not None and cls._model is not None and  cls._device is not None:
            return cls._processor, cls._model, cls._device

        # GPU = float16
        # CPU = float32
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("Loading BLIP-2 model '%s' on device '%s'", cls._model_id, device)

        # Load preprocessor
        processor = Blip2Processor.from_pretrained(cls._model_id)
        # Load model
        model = Blip2ForConditionalGeneration.from_pretrained(
            cls._model_id,
            torch_dtype=torch_dtype,
        ).to(device) # type: ignore

        model.eval()

        cls._processor = processor
        cls._model = model
        cls._device = device

        logger.info("BLIP-2 model loaded successfully")

        return processor, model, device

    @classmethod
    def model_id(cls) -> str:
        """Expose the configured BLIP-2 model identifier."""
        return cls._model_id
