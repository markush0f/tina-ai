from typing import Dict, Any

from app.utils.whisper_loader import WhisperModelRegistry
from app.core.logger import get_logger

logger = get_logger(__name__)


class AudioAnalysisService:
    """Service that performs audio transcription using Whisper."""

    def __init__(self):
        self.model, self.device = WhisperModelRegistry.get()

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        # Transcribe audio
        logger.info("Running transcription on file: %s", audio_path)

        result = self.model.transcribe(audio_path)

        # Build structured response
        return {
            "transcription": result.get("text", "").strip(),  # type: ignore
            "language": result.get("language", None),
            "segments": [
                {
                    "start": seg["start"],  # type: ignore
                    "end": seg["end"],  # type: ignore
                    "text": seg["text"].strip(),  # type: ignore
                }
                for seg in result.get("segments", [])
            ],
            "metadata": {
                "duration": result.get("duration", None),
                "model": WhisperModelRegistry._model_id,
                "device": self.device,
            },
        }
