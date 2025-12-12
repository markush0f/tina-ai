from typing import Dict, Any
import os

import librosa

from app.services.audio_emotion_timeline_service import AudioEmotionTimelineService
from app.services.audio_transcription_service import AudioTranscriptionService
from app.services.audio_emotion_service import AudioEmotionService
from app.utils.audio_extractor import extract_audio_to_wav
from app.core.logger import get_logger
from app.utils.vad import detect_speech, run_vad

logger = get_logger(__name__)


class AudioAnalysisService:
    """High-level service combining transcription, global emotion and segment-based emotion timeline."""

    def __init__(self):
        self.transcriber = AudioTranscriptionService()
        self.emotion_service = AudioEmotionService()
        self.timeline_service = AudioEmotionTimelineService()

    def analyze(self, input_path: str) -> Dict[str, Any]:
        wav_path = None

        try:
            wav_path = extract_audio_to_wav(input_path)

            audio, sr = librosa.load(wav_path, sr=16000)

            if not detect_speech(audio, sample_rate=int(sr)):
                return {
                    "has_voice": False,
                    "error": "No speech detected in the audio file.",
                }

            transcription = self.transcriber.transcribe(wav_path)

            emotion = self.emotion_service.analyze(wav_path)

            segments = transcription.get("segments", [])

            emotion_timeline = self.timeline_service.analyze_timeline(
                audio=audio,
                sample_rate=int(sr),
                whisper_segments=segments,
            )

            return {
                "has_voice": True,
                "transcription": transcription.get("text", ""),
                "language": transcription.get("language", None),
                "segments": segments,
                "emotion": {
                    "primary": emotion.get("primary_emotion"),
                    "scores": emotion.get("scores", {}),
                },
                "emotion_timeline": emotion_timeline,
                "metadata": {
                    "duration": transcription.get("metadata", {}).get("duration"),
                    "whisper_model": transcription.get("metadata", {}).get("model"),
                    "emotion_model": emotion.get("metadata", {}).get("model"),
                    "device": {
                        "whisper": transcription.get("metadata", {}).get("device"),
                        "emotion": emotion.get("metadata", {}).get("device"),
                    },
                },
            }

        except Exception as e:
            logger.error("Audio analysis failed: %s", str(e))
            raise e

        finally:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)