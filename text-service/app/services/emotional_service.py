from typing import List, Dict, Any

from app.utils.model_loader import build_classification_pipeline


class EmotionalAnalyzerService:
    """Service wrapper that loads the local emotion classification pipeline."""

    def __init__(self):
        model_name = "emotion_model"
        self.pipeline = build_classification_pipeline(model_name)

    def analyze(self, text: str) -> List[Dict[str, Any]]:
        """Run the model over text and return emotions sorted by confidence."""
        # Hugging Face pipelines accept either str or list[str] and return batch-specific formats.
        output = self.pipeline(text)

        if isinstance(output, list):
            first = output[0]

            if isinstance(first, list):
                scores: List[Dict[str, Any]] = first  # Normal case
            elif isinstance(first, dict):
                scores = output  # Pipeline sometimes returns list[dict] even for batches
            else:
                raise ValueError("Unexpected pipeline output format.")
        else:
            raise ValueError("Pipeline output is not a list.")

        return sorted(
            scores,
            key=lambda item: float(item["score"]),
            reverse=True,
        )
