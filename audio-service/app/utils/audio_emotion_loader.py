from typing import Optional
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from app.core.logger import get_logger

logger = get_logger(__name__)


class AudioEmotionModelRegistry:
    """Centralized lazy-loaded registry for the Speech Emotion Recognition model."""

    _model = None
    _extractor = None
    _device: Optional[str] = None
    _model_id: str = "superb/wav2vec2-base-superb-er"

    @classmethod
    def get(cls):
        # Return if already loaded
        if cls._model is not None and cls._extractor is not None:
            return cls._model, cls._extractor, cls._device

        # Select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading voice emotion model '%s' on '%s'", cls._model_id, device)

        # Load model + extractor
        extractor = AutoFeatureExtractor.from_pretrained(cls._model_id)
        model = AutoModelForAudioClassification.from_pretrained(cls._model_id).to(
            device
        )

        cls._model = model
        cls._extractor = extractor
        cls._device = device

        logger.info("Voice emotion model loaded successfully")
        return model, extractor, device
