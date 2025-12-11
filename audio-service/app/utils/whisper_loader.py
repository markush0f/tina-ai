from typing import Optional
import torch
import whisper

from app.core.logger import get_logger

logger = get_logger(__name__)


class WhisperModelRegistry:
    """Centralized lazy-loaded registry for the Whisper audio transcription model."""

    _model: Optional[whisper.Whisper] = None
    _device: Optional[str] = None
    _model_id: str = "medium"  # Local model size (changeable)

    @classmethod
    def get(cls) -> tuple[whisper.Whisper, str]:
        # Return model if already loaded
        if cls._model is not None and cls._device is not None:
            return cls._model, cls._device

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading Whisper model '%s' on device '%s'", cls._model_id, device)

        # Load Whisper model locally
        model = whisper.load_model(cls._model_id).to(device)

        cls._model = model
        cls._device = device

        logger.info("Whisper model loaded successfully")
        return model, device
