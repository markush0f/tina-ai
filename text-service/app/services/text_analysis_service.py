from datetime import datetime
from typing import Dict, Any, List

from app.services.emotional_service import EmotionalAnalyzerService
from app.services.sentiment_service import SentimentService
from app.services.topic_service import TopicService


class TextAnalysisService:
    """
    Central orchestrator that combines:
    - Emotional analysis (emotion scores)
    - Sentiment reasoning (dominance, polarity, intensity, mixed emotions)
    - Topic classification (topics + subtopics)
    
    This service returns a complete structured package of the user's
    emotional and topical interpretation for a given text input.
    """

    def __init__(
        self,
        emotional_service: EmotionalAnalyzerService,
        sentiment_service: SentimentService,
        topic_service: TopicService,
    ):
        self.emotional = emotional_service
        self.sentiment = sentiment_service
        self.topics = topic_service

    # High-level analysis entrypoint
    def analyze(self, text: str, general_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Performs the full text analysis pipeline:
        1. Extract raw emotion scores from text
        2. Apply sentiment reasoning
        3. Extract subtopics
        4. Match subtopics with general topics
        5. Return unified analysis package
        """

        # 1. Raw emotional scores
        raw_emotions = self._format_emotions(
            self.emotional.analyze(text)
        )  # convert list -> dict

        # 2. Sentiment reasoning
        sentiment_result = self.sentiment.analyze(raw_emotions)

        # 3. Topic/subtopic extraction
        topic_result = self.topics.analyze_with_topics(text, general_topics)

        # 4. Build final response
        return {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "emotions": raw_emotions,
                "sentiment": sentiment_result,
                "topics": topic_result,
            },
        }

    #  convert emotional pipeline list to {label: score}
    def _format_emotions(self, emotion_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        EmotionalAnalyzerService returns a list of dicts:
        [
            {"label": "sadness", "score": 0.83},
            {"label": "joy", "score": 0.14},
            ...
        ]

        This converts it into a dictionary:
        {
            "sadness": 0.83,
            "joy": 0.14,
            ...
        }
        """
        return {e["label"]: float(e["score"]) for e in emotion_list}


# Dependency provider
def get_text_analysis_service():
    return TextAnalysisService(
        emotional_service=EmotionalAnalyzerService(),
        sentiment_service=SentimentService(),
        topic_service=TopicService(),
    )

