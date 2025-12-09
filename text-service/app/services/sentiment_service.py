from typing import Dict, List


class SentimentService:
    """
    Service that performs higher-level sentiment reasoning on top of
    raw emotion scores returned by the EmotionalAnalyzerService.

    This service does NOT use any ML models. All computations are
    deterministic and rule-based, which keeps the analysis fast,
    transparent, and easy to adjust.
    """

    def get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """
        Returns the emotion with the highest score.
        """
        return max(emotions.items(), key=lambda x: x[1])[0]

    def get_mixed_emotions(
        self, emotions: Dict[str, float], threshold: float = 0.15
    ) -> List[str]:
        """
        If multiple emotions have similiar scores (difference <= threshold),
        they are consireded "mixed emotions".
        """
        sorted_scores = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_score = sorted_scores[0]

        mixed = [top_emotion]

        for emotion, score in sorted_scores[1:]:
            if top_score - score <= threshold:
                mixed.append(emotion)

        # If only the dominant emotion qualifies, no mixed state exists.
        return mixed if len(mixed) > 1 else []
        # If only the dominant emotion qualifies, no mixed state exists.
        # ----------------------------------------------------------

    # Sentiment (positive / negative / neutral)
    def compute_sentiment(self, emotions: Dict[str, float]) -> Dict[str, float | str]:
        """
        Computes a general sentiment score using a rule-based formula.
        Positive emotions increase the score; negative emotions decrease it.
        """

        # Weighted positive signals
        positive = emotions.get("joy", 0) + emotions.get("surprise", 0) * 0.3

        # Weighted negative signals
        negative = (
            emotions.get("sadness", 0)
            + emotions.get("anger", 0)
            + emotions.get("fear", 0)
            + emotions.get("disgust", 0)
        )

        score = positive - negative

        # Determine sentiment category
        if score > 0.05:
            label = "positive"
        elif score < -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {"label": label, "score": score}

    # ---------------------------------------------------------
    # Emotional intensity
    def compute_intensity(self, emotions: Dict[str, float]) -> Dict[str, float | str]:
        """
        Measures how 'strong' the emotional signal is.
        Based on the highest emotional probability.
        """

        score = max(emotions.values())

        if score < 0.30:
            label = "low"
        elif score < 0.60:
            label = "medium"
        else:
            label = "high"

        return {"label": label, "score": score}

    # Final combined sentiment package
    def analyze(self, emotions: Dict[str, float]) -> Dict[str, any]:
        """
        Produces the full sentiment reasoning package:
        - raw emotions
        - dominant emotion
        - mixed emotions
        - sentiment polarity
        - emotional intensity
        """

        dominant = self.get_dominant_emotion(emotions)
        mixed = self.get_mixed_emotions(emotions)
        sentiment = self.compute_sentiment(emotions)
        intensity = self.compute_intensity(emotions)

        return {
            "emotions": emotions,
            "dominant_emotion": dominant,
            "mixed_emotions": mixed,
            "sentiment": sentiment,
            "intensity": intensity,
        }


def get_sentiment_service():
    return SentimentService()

# TEST
service = SentimentService()

sample_emotions = {
    "joy": 0.42,
    "sadness": 0.12,
    "anger": 0.05,
    "fear": 0.08,
    "disgust": 0.03,
    "surprise": 0.31,
}

result = service.analyze(sample_emotions)
print(result)
