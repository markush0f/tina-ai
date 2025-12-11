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



    def __init__(self) -> None:
        self.processor, self.model, self.device = ClipSceneModelRegistry.get()

    @torch.inference_mode()
    def classify(self, image: Image.Image) -> str:
        """
        Return the most likely scene label from the predefined vocabulary.
        """

        # Prepare CLIP text prompts
        text_inputs = [f"a photo of a {scene}" for scene in SCENES]

        # Tokenize text and image
        inputs = self.processor(
            text=text_inputs,
            images=image,
            return_tensors="pt",  # type: ignore
            padding=True,  # type: ignore
        ).to(self.device)

        outputs = self.model(**inputs)

        # Extract similarities between image and each scene text
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # Pick the scene with highest probability
        best_scene_idx = probs.argmax(dim=1).item()
        best_scene = SCENES[best_scene_idx]

        return best_scene
