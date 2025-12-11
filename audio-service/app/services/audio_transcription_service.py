from typing import Dict, Any
from app.utils.whisper_loader import WhisperModelRegistry
from app.core.logger import get_logger

logger = get_logger(__name__)


class AudioTranscriptionService:
    """Service that performs pure Whisper transcription."""

    def __init__(self):
        self.model, self.device = WhisperModelRegistry.get()

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        logger.info("Running transcription on file: %s", audio_path)

        try:
            result = self.model.transcribe(audio_path)

            # Ensure safe fallback structure
            text = result.get("text", "") or ""
            segments = result.get("segments", []) or []

            return {
                "text": text,
                "language": result.get("language", None),
                "segments": [
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", "").strip(),
                    }
                    for seg in segments
                ],
                "metadata": {
                    "duration": result.get("duration", None),
                    "device": self.device,
                    "model": WhisperModelRegistry._model_id,
                },
            }

        except Exception as e:
            logger.error("Whisper transcription failed: %s", str(e))
            # Guaranteed fallback
            return {
                "text": "",
                "language": None,
                "segments": [],
                "metadata": {
                    "duration": None,
                    "device": self.device,
                    "model": WhisperModelRegistry._model_id,
                },
            }
