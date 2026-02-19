from app.celery_app import celery_app
from app.tasks import transcribe  # noqa: F401

__all__ = ("celery_app",)