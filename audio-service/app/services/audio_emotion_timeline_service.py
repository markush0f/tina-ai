import os
import tempfile
from typing import List, Dict, Any
import librosa

from app.core.logger import get_logger
from app.services.audio_emotion_service import AudioEmotionService
from app.utils.vad import run_vad

logger = get_logger(__name__)
import soundfile as sf


class AudioEmotionTimelineService:
    """Builds an emotion timeline based on Whisper segments."""

    def __init__(self):
        self.emotion_service = AudioEmotionService()

    def analyze_timeline(
        self,
        audio,
        sample_rate: int,
        whisper_segments: List[Dict[str, Any]],
        min_segment_duration: float = 0.4,
    ) -> List[Dict[str, Any]]:
        timeline: List[Dict[str, Any]] = []

        for seg in whisper_segments:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            text = seg.get("text", "").strip()

            duration = end - start
            if duration <= 0 or duration < min_segment_duration:
                continue

            start_idx = int(start * sample_rate)
            end_idx = int(end * sample_rate)

            if start_idx >= len(audio):
                continue

            end_idx = min(end_idx, len(audio))
            segment_audio = audio[start_idx:end_idx]

            if len(segment_audio) == 0:
                continue

            tmp_path = None

            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_path = tmp.name
                tmp.close()

                sf.write(tmp_path, segment_audio, sample_rate)

                emotion = self.emotion_service.analyze(tmp_path)

                primary = emotion.get("primary_emotion")
                scores = emotion.get("scores", {})
                score = float(max(scores.values())) if scores else 0.0

                timeline.append(
                    {
                        "start": start,
                        "end": end,
                        "emotion": primary,
                        "score": score,
                        "text": text,
                    }
                )

            except Exception as e:
                logger.error(
                    "Failed to analyze emotion for segment %.2f-%.2f: %s",
                    start,
                    end,
                    str(e),
                )

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return timeline
