from typing import List, Dict, Any
import librosa

from app.services.audio_emotion_service import AudioEmotionService
from app.utils.vad import run_vad


class AudioEmotionTimelineService:
    """
    Splits audio into VAD segments and runs emotion classification on each.
    Returns emotional timeline with timestamps.
    """

    def __init__(self):
        self.emotion_service = AudioEmotionService()

    def analyze_timeline(self, wav_path: str) -> List[Dict[str, Any]]:
        audio, sr = librosa.load(wav_path, sr=16000)

        frames, speech_mask = run_vad(audio, sample_rate=16000)

        timeline = []

        for (start, end), is_speech in zip(frames, speech_mask):
            if not is_speech:
                continue

            # Run emotion for this segment only
            start_idx = int(start * sr)
            end_idx = int(end * sr)
            segment_audio = audio[start_idx:end_idx]

            segment_path = self._save_segment_temp(segment_audio, sr)

            emotion = self.emotion_service.analyze(segment_path)

            timeline.append({
                "start": start,
                "end": end,
                "emotion": emotion["primary_emotion"],
                "score": max(emotion["scores"].values())
            })

        return timeline

    def _save_segment_temp(self, audio, sr):
        import tempfile
        import soundfile as sf

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()

        sf.write(tmp_path, audio, sr)

        return tmp_path
