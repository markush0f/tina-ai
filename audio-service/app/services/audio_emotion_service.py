from typing import Dict, Any
import torch
import librosa
import numpy as np

from app.utils.audio_emotion_loader import AudioEmotionModelRegistry
from app.core.logger import get_logger

logger = get_logger(__name__)


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

        # Map classes to probabilities
        emotion_map = {
            label: float(score)
            for label, score in zip(self.model.config.id2label.values(), scores)
        }

        # Select primary emotion
        primary_emotion = max(emotion_map, key=emotion_map.get) # type: ignore

        # Build response
        return {
            "primary_emotion": primary_emotion,
            "scores": emotion_map,
            "metadata": {
                "model": AudioEmotionModelRegistry._model_id,
                "device": self.device,
            },
        }
