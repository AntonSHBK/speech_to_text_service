from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.routers import router
from app.settings import settings
from app.service.transcriber import transcriber_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    transcriber_service.init(
        model_name=settings.MODEL_NAME,
        device=settings.DEVICE,
        cache_dir=settings.CACHE_DIR,
        token=None,
        compute_type="default",
        cpu_threads=settings.MODEL_CPU_THREADS,
        num_workers=settings.MODEL_NUM_WORKERS,
    )

    yield

app = FastAPI(lifespan=lifespan, title="Speech-to-Text API")
app.include_router(router)

@app.get("/")
def health_check():
    """Простой эндпоинт для проверки работоспособности API."""
    return {"status": "API is running"}