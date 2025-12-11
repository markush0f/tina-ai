from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os

from app.services.audio_analysis_service import AudioAnalysisService

router = APIRouter(prefix="/analyze/audio", tags=["audio"])

analysis_service = AudioAnalysisService()


@router.post("/")
async def analyze_audio(file: UploadFile = File(...)):
    tmp_path = None

    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        return analysis_service.analyze(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Remove original uploaded file
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
