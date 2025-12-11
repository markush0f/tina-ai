from fastapi import APIRouter, UploadFile, File
import tempfile
import os

from app.services.audio_analysis_service import AudioAnalysisService

router = APIRouter(prefix="/analyze/audio", tags=["audio"])

audio_service = AudioAnalysisService()


@router.post("/")
async def analyze_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Run analysis
    result = audio_service.analyze(tmp_path)

    # Remove temp file
    os.remove(tmp_path)

    return result
