from typing import List
from PIL import Image
import torch

from app.utils.trocr_loader import TrocrModelRegistry
from app.core.logger import get_logger

logger = get_logger(__name__)


class OcrService:
    """
    OCR service using Microsoft's TrOCR model.
    Produces accurate text even for stylized, digital, or large fonts.
    """

    def __init__(self) -> None:
        self.processor, self.model, self.device = TrocrModelRegistry.get()
        self.model_id = TrocrModelRegistry.model_id()

    @torch.inference_mode()
    def extract_text(self, image: Image.Image) -> List[str]:
        """
        Recognize text from the image using TrOCR.
        Returns a list with a single clean text string or empty if nothing found.
        """

        logger.info("Running TrOCR on image")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess for TrOCR
        inputs = self.processor(image, return_tensors="pt").to(self.device) # type: ignore

        # Generate sequence
        generated_ids = self.model.generate(**inputs) # type: ignore

        # Decode output
        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        logger.info("TrOCR extracted text: %s", text)

        if not text:
            return []
        return [text]
