from typing import List
import torch
from PIL import Image

from app.core.logger import get_logger
from app.services.clip_scene_loader import ClipSceneModelRegistry
from app.common.SCENES import SCENES

logger = get_logger(__name__)


class SceneClassificationService:
    """
    High-level service that uses CLIP to classify the scene of an image.
    """

    CONFIDENCE_THRESHOLD: float = 0.10
    FALLBACK_SCENE: str = "unknown"

    def __init__(self) -> None:
        self.processor, self.model, self.device = ClipSceneModelRegistry.get()

    @torch.inference_mode()
    def classify(self, image: Image.Image) -> str:
        """
        Return the most likely scene label from the predefined vocabulary.
        If CLIP confidence is too low, return a fallback label instead.
        """

        # Create natural language prompts for CLIP text encoder
        text_inputs = [f"a photo of a {scene}" for scene in SCENES]

        # Tokenize image + text
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",  # type: ignore
            padding=True,  # type: ignore
        ).to(self.device)

        # Run CLIP forward pass
        outputs = self.model(**inputs)

        # Extract probabilities
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)

        # Select best scene
        best_idx = probs.argmax(dim=1).item()
        best_prob = probs[0, best_idx].item()
        best_scene = SCENES[best_idx]

        # ---- FALLBACK LOGIC ----
        if best_prob < self.CONFIDENCE_THRESHOLD:
            logger.info(
                f"Scene fallback triggered: CLIP confidence {best_prob:.3f} < threshold {self.CONFIDENCE_THRESHOLD}"
            )
            return self.FALLBACK_SCENE

        return best_scene
