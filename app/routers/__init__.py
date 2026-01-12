from fastapi import APIRouter

from app.settings import settings

from app.routers import transcribe
from app.routers import ui

router = APIRouter()
router.include_router(transcribe.router)

if settings.USE_INTERFACE:
    router.include_router(ui.router)


