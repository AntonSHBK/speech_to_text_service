from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan, title="Speech-to-Text API")

@app.get("/")
def health_check():
    """Простой эндпоинт для проверки работоспособности API."""
    return {"status": "API is running"}

app.include_router(router)