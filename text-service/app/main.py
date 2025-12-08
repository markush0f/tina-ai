"""FastAPI entry point for the text emotion service."""

from fastapi import FastAPI

from app.core.logger import configure_logging
from app.routers.emotions import router as emotions_router

configure_logging()

app = FastAPI(title="Text Service", version="0.1.0")
app.include_router(emotions_router)


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Simple readiness endpoint."""
    return {"status": "ok"}
