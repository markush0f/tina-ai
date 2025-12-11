from typing import Dict, Any
import os
import tempfile

from app.services.audio_transcription_service import AudioTranscriptionService
from app.services.audio_emotion_service import AudioEmotionService
from app.utils.audio_extractor import extract_audio_to_wav
from app.core.logger import get_logger

logger = get_logger(__name__)


class AudioAnalysisService:
    """High-level service that orchestrates audio transcription and emotion detection."""

    def __init__(self):
        self.transcriber = AudioTranscriptionService()
        self.emotion_service = AudioEmotionService()

    def analyze(self, input_path: str) -> Dict[str, Any]:
        wav_path = None

        try:
            # Audio extraction to normalized 16kHz WAV
            wav_path = extract_audio_to_wav(input_path)

            # Transcription
            transcription = self.transcriber.transcribe(wav_path)

            # Emotion analysis
            emotion = self.emotion_service.analyze(wav_path)

            # Build unified response
            return {
                "transcription": transcription.get("text", ""),
                "language": transcription.get("language", None),
                "segments": transcription.get("segments", []),
                "emotion": {
                    "primary": emotion["primary_emotion"],
                    "scores": emotion["scores"],
                },
                "metadata": {
                    "duration": transcription["metadata"]["duration"],
                    "whisper_model": transcription["metadata"]["model"],
                    "emotion_model": emotion["metadata"]["model"],
                    "device": {
                        "whisper": transcription["metadata"]["device"],
                        "emotion": emotion["metadata"]["device"],
                    },
                },
            }

        except Exception as e:
            # Error raised inside audio pipeline
            logger.error("Audio analysis failed: %s", str(e))
            raise e

        finally:
            # Cleanup for the temp WAV file
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
