from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.services.text_analysis_service import get_text_analysis_service, TextAnalysisService


router = APIRouter(prefix="/emotions", tags=["emotions-analysis"])


# ----------------------------------------------------------
# Request / Response models
# ----------------------------------------------------------
class TopicInput(BaseModel):
    topic: str = Field(..., description="General topic name.")
    keywords: List[str] = Field(default_factory=list, description="Keywords that define this topic.")


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., description="User text to analyze.")
    topics: List[TopicInput] = Field(
        default_factory=list,
        description="General topics retrieved from the database (Spring Boot)."
    )


# ----------------------------------------------------------
# Route
# ----------------------------------------------------------
@router.post("/analyze")
def analyze_text(
    payload: TextAnalysisRequest,
    service: TextAnalysisService = Depends(get_text_analysis_service),
) -> Dict[str, Any]:
    """
    Performs a full emotional + sentiment + topic analysis on the input text.
    """

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    general_topics = [topic.dict() for topic in payload.topics]

    result = service.analyze(text=text, general_topics=general_topics)

    return result
