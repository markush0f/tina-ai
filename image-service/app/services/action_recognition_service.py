from typing import List
import torch
from PIL import Image

from app.utils.clip_loader import ClipModelRegistry
from app.common.action_labels import ACTIONS
from app.core.logger import get_logger

logger = get_logger(__name__)


class ActionRecognitionService:
    """
    Zero-shot action recognition for static images using CLIP.
    """

    def __init__(self) -> None:
        self.processor, self.model, self.device = ClipModelRegistry.get()
        self.model_id = ClipModelRegistry.model_id()

    @torch.inference_mode()
    def recognize(self, image: Image.Image) -> List[str]:
        """
        Predict up to 3 likely actions in a static image.
        """

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build prompts
        prompts = [f"a person {a}" for a in ACTIONS]

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        outputs = self.model(**inputs)

        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)[0]

        # top 3 predictions
        topk = probs.topk(3)

        results = [ACTIONS[i] for i in topk.indices.tolist()]

        logger.info("Recognized actions: %s", results)

        return results
