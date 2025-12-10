from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.core.logger import get_logger
from app.services.image_analysis_service import ImageAnalysisService

router = APIRouter(prefix="/v1/image", tags=["image"])
logger = get_logger(__name__)

# The service is instantiated once when the module loads.

image_service = ImageAnalysisService()

@router.get("/ping")
def ping() -> dict:
    """
    Basic health endpoint for this router.
    """
    return {"message": "image-service router is alive"}


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Receive an uploaded image file, extract the raw bytes, and delegate
    the analysis to the ImageAnalysisService.

    This endpoint performs only:
    - file validation
    - logging
    - delegation to the domain service
    - response formatting
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"): # type: ignore
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not recognized as an image."
        )

    # Read raw bytes
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    logger.info(
        "Received image for analysis: name=%s type=%s size=%d bytes",
        file.filename,
        file.content_type,
        len(image_bytes)
    )

    # Delegate to the domain service
    try:
        result = image_service.analyze(image_bytes)
    except ValueError as ve:
        # Known validation issue (e.g. corrupted image)
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        logger.exception("Unexpected error during image analysis")
        raise HTTPException(status_code=500, detail="Image analysis failed") from exc

    # Convert domain object to plain dictionary
    return JSONResponse(content=result.to_dict())
