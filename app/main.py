from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from app.middleware.api_logging import ApiLoggingMiddleware
from app.routers import router
from app.utils.logging import get_logger


api_logger = get_logger("api.requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan, title="Speech-to-Text API")
app.add_middleware(ApiLoggingMiddleware)
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "-")
    api_logger.warning(
        "request validation failed | request_id=%s | method=%s | path=%s | errors=%s",
        request_id,
        request.method,
        request.url.path,
        exc.errors(),
    )
    return await request_validation_exception_handler(request, exc)


@app.get("/")
def health_check():
    """Простой эндпоинт для проверки работоспособности API."""
    return {"status": "API is running"}
