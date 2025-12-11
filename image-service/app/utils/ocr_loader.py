from typing import Optional
from paddleocr import PaddleOCR

from core.logger import get_logger

logger = get_logger(__name__)


class OcrModelRegistry:
    """
    Centralized registry for the OCR model.
    Loads PaddleOCR only once.
    """

    # Model
    _ocr: Optional[PaddleOCR] = None

    @classmethod
    def get(cls) -> PaddleOCR:
        if cls._ocr is not None:
            return cls._ocr

        logger.info("Loading PaddleOCR model")

        cls._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en", 
            show_log=False,
        )

        logger.info("PaddleOCR loaded successfully")

        return cls._ocr
