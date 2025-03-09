from fastapi import FastAPI
from app.routers.routers import router
from app.config import CONFIG, init_dir

app = FastAPI(title="Speech-to-Text API")

init_dir()

app.include_router(router)

@app.get("/")
def health_check():
    """Простой эндпоинт для проверки работоспособности API."""
    return {"status": "API is running"}
