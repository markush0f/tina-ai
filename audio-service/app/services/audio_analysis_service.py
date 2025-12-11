from typing import Dict, Any
import os
import tempfile

import librosa

from app.services.audio_emotion_timeline_service import AudioEmotionTimelineService
from app.services.audio_transcription_service import AudioTranscriptionService
from app.services.audio_emotion_service import AudioEmotionService
from app.utils.audio_extractor import extract_audio_to_wav
from app.core.logger import get_logger
from app.utils.vad import run_vad

logger = get_logger(__name__)


class AudioAnalysisService:
    """High-level service that orchestrates audio transcription and emotion detection."""

    def __init__(self):
        self.transcriber = AudioTranscriptionService()
        self.emotion_service = AudioEmotionService()
        self.timeline_service = AudioEmotionTimelineService()

    def analyze(self, input_path: str) -> Dict[str, Any]:
        wav_path = None

        try:
            # Audio extraction to normalized 16kHz WAV
            wav_path = extract_audio_to_wav(input_path)

            # Load audio for VAD
            audio, sr = librosa.load(wav_path, sr=16000)

            frames, speech_mask = run_vad(audio, sample_rate=int(sr))

            if not any(speech_mask):
                return {
                    "error": "No speech detected in the audio file.",
                    "has_voice": False,
                }

            # Transcription
            transcription = self.transcriber.transcribe(wav_path)

            # Emotion analysis
            emotion = self.emotion_service.analyze(wav_path)

            # Emotion timeline
            timeline = self.timeline_service.analyze_timeline(wav_path)

            # Build unified response
            return {
                "has_voice": True,
                "transcription": transcription["text"],
                "language": transcription["language"],
                "segments": transcription["segments"],
                "emotion": {
                    "primary": emotion["primary_emotion"],
                    "scores": emotion["scores"],
                },
                "emotion_timeline": timeline,
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
