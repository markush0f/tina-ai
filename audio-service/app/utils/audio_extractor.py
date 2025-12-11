import subprocess
import tempfile
from app.core.logger import get_logger

logger = get_logger(__name__)


def extract_audio_to_wav(input_path: str) -> str:
    # Create temporary WAV file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav_path = tmp.name
    tmp.close()

    # FFmpeg extraction
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        wav_path,
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    logger.info("Extracted WAV audio to: %s", wav_path)
    return wav_path
