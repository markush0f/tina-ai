from app.routers.audio import router as audio_router
from fastapi import FastAPI

app = FastAPI(title="Audio Analysis Service")


@app.get("/")
async def health():
    return {"message": "Audio Analysis Service is running."}


app.include_router(audio_router)
