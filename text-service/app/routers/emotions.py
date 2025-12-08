"""Routes that expose the emotion analysis service."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict

from app.services.emotional_service import EmotionalAnalyzerService, get_emotion_service

router = APIRouter(prefix="/emotions", tags=["emotions"])


class EmotionRequest(BaseModel):
    text: str = Field(..., description="Input text to analyze.")


@router.post("/analyze")
def analyze_emotion(
    payload: EmotionRequest,
    service: EmotionalAnalyzerService = Depends(get_emotion_service),
) -> Any:
    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    return service.analyze(text)


@router.post("/analyze-comparison")
def analyze_emotion_comparison(
    payload: EmotionRequest,
    service: EmotionalAnalyzerService = Depends(get_emotion_service),
) -> Dict[str, Any]:
    text = payload.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    return service.analyze_with_sentence_comparison(text)
