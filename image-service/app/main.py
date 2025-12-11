from fastapi import FastAPI
from app.routers.image import router as image_router
import os

# DISABLE CHECK FOR MODEL SOURCE IN HUGGINGFACE
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# FastAPI application instance
app = FastAPI()

# Mounting the image router
# Added router registration for image endpoints
app.include_router(image_router)


@app.get("/ping")
def root_ping():
    # Added basic health endpoint for main application
    return {"message": "image-service is running"}
