from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import io

from PIL import Image
import torch

from app.core.logger import get_logger
from app.services.object_detection_service import ObjectDetectionService
from app.utils.blip_loader import Blip2ModelRegistry

logger = get_logger(__name__)


@dataclass
class ImageAnalysisResult:
    """
    Domain object representing the normalized output of an image analysis pipeline.
    """

    description: str
    objects: List[str]
    scene: Optional[str]
    actions: List[str]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert object into plain dictionary for JSON responses."""
        return asdict(self)


class ImageAnalysisService:
    """
    High-level service for image analysis using BLIP-2.

    Handles:
    - image decoding
    - caption generation
    - construction of the domain result
    """

    def __init__(self) -> None:
        self.processor, self.model, self.device = Blip2ModelRegistry.get()
        self.model_id = Blip2ModelRegistry.model_id()
        # load object detection service
        self.object_detection_service = ObjectDetectionService()

    def _load_image(self, image_bytes: bytes) -> Image.Image:
        """
        Convert raw bytes to a Pillow Image in RGB format.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            logger.exception("Failed to decode image bytes")
            raise ValueError("Invalid or corrupted image") from exc

        return image

        
    @torch.inference_mode()
    def _generate_caption(self, image: Image.Image) -> str:
        """
        Generate a natural caption for the given image using BLIP-2.
        """
        # https://huggingface.co/docs/transformers/model_doc/blip-2
        # get data converted in tensors of pytorch
        inputs = self.processor(images=image, return_tensors="pt")  # type: ignore

        # Move tensors to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Captioning
        output_ids = self.model.generate(**inputs, max_new_tokens=60)

        # IDS to text
        caption = self.processor.tokenizer.decode(  # type: ignore
            output_ids[0], skip_special_tokens=True
        ).strip()

        return caption

    def analyze(self, image_bytes: bytes) -> ImageAnalysisResult:
        """
        Full pipeline: decode image, caption it, and return structured result.
        """
        logger.info("Starting image analysis")

        image = self._load_image(image_bytes)
        description = self._generate_caption(image)
        
        # run object detection with YOLO.
        objects = self.object_detection_service.detect_objects(image)
        result = ImageAnalysisResult(
            description=description,
            objects=objects,
            scene=None,
            actions=[],
            meta={
                "model_id": self.model_id,
                "device": self.device,
            },
        )

        logger.info("Image analysis completed successfully")

        return result
