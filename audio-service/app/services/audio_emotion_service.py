from typing import Dict, Any
import torch
import librosa
import numpy as np

from app.utils.audio_emotion_loader import AudioEmotionModelRegistry
from app.core.logger import get_logger

logger = get_logger(__name__)


EMOTION_LABELS = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
}


class AudioEmotionService:
    """Service that performs prosodic emotion detection on audio files."""

    def __init__(self):
        self.model, self.extractor, self.device = AudioEmotionModelRegistry.get()

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        # Load raw audio
        audio, sr = librosa.load(audio_path, sr=16000)

        # Prepare inputs for the model
        inputs = self.extractor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True
        ).to(self.device)

        # Run prediction
        with torch.no_grad():
            output = self.model(**inputs)
            scores = torch.softmax(output.logits, dim=-1)[0].cpu().numpy()

        raw_scores = {
            raw_label: float(score)
            for raw_label, score in zip(self.model.config.id2label.values(), scores)
        }

        # Normalize primary emotion to readable text
        raw_primary = max(raw_scores, key=raw_scores.get)  # type: ignore
        primary = EMOTION_LABELS.get(raw_primary, raw_primary)

        readable_scores = {EMOTION_LABELS.get(k, k): v for k, v in raw_scores.items()}

        # Build response
        return {
            "primary_emotion": primary,
            "scores": readable_scores,
            "metadata": {
                "device": self.device,
                "model": AudioEmotionModelRegistry._model_id,
            },
        }
