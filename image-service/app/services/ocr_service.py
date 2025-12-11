from typing import List
from PIL import Image
import numpy as np

from core.logger import get_logger
from utils.ocr_loader import OcrModelRegistry

logger = get_logger(__name__)

# https://github.com/PaddlePaddle/PaddleOCR
class OcrService:
    """
    Service responsible for extracting text from an image using PaddleOCR.
    """

    def __init__(self) -> None:
        self.ocr = OcrModelRegistry.get()

    def extract_text(self, image: Image.Image) -> List[str]:
        """
        Run OCR on the given image and return a list of detected text strings.
        """
        logger.info("Running OCR")

        # PaddleOCR requires numpy array in BGR format
        img_np = np.array(image)[:, :, ::-1]

        # results = self.ocr.ocr(img_np, cls=True)
        results = self.ocr.predict(img_np)

        if not results:
            return []

        extracted = []

        for line in results:
            for box_info in line:
                text = box_info[1][0]  # recognized text
                extracted.append(text)

        # Remove duplicates and sort
        deduped = sorted(set(extracted))

        logger.info("OCR detected text: %s", deduped)

        return deduped
