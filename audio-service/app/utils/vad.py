import webrtcvad
import numpy as np
import librosa


def run_vad(audio, sample_rate=16000, frame_ms=30, aggressiveness=2):
    """
    Runs WebRTC VAD over an audio waveform.
    Returns a boolean mask indicating which frames contain speech.
    """
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
