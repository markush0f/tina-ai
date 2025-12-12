import webrtcvad
import numpy as np


def run_vad(audio, sample_rate: int = 16000, frame_ms: int = 30, aggressiveness: int = 2):
    """Run VAD over audio and return frame time ranges plus a speech mask."""
    vad = webrtcvad.Vad(aggressiveness)

    frame_len = int(sample_rate * frame_ms / 1000)
    frames = []
    speech_mask = []

    for i in range(0, len(audio), frame_len):
        frame = audio[i:i + frame_len]

        if len(frame) < frame_len:
            break

        pcm = (frame * 32768).astype(np.int16).tobytes()
        is_speech = vad.is_speech(pcm, sample_rate)

        frames.append((i / sample_rate, (i + frame_len) / sample_rate))
        speech_mask.append(is_speech)

    return frames, speech_mask


def detect_speech(audio, sample_rate: int = 16000, frame_ms: int = 30, aggressiveness: int = 2) -> bool:
    """Return True if any speech is detected in the audio."""
    _, speech_mask = run_vad(audio, sample_rate=sample_rate, frame_ms=frame_ms, aggressiveness=aggressiveness)
    return any(speech_mask)
