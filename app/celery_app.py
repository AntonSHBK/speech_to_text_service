from celery import Celery

from app.settings import settings


celery_app = Celery(
    "speech_to_text_service",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    # Marks task as STARTED in backend before execution.
    task_track_started=True,
    # Use JSON for task payload serialization.
    task_serializer="json",
    # Use JSON for result serialization in backend.
    result_serializer="json",
    # Only accept JSON messages from broker.
    accept_content=["json"],
    # Acknowledge task only after execution to reduce task loss on worker crash.
    task_acks_late=True,
    # Re-queue task if worker process is lost while processing it.
    task_reject_on_worker_lost=True,
    # Ack failed/timeout tasks explicitly to keep broker state consistent.
    task_acks_on_failure_or_timeout=True,
    # Reserve one task per worker process to avoid long task starvation.
    worker_prefetch_multiplier=1,
    # Keep root handlers configured by app.utils.logging so Celery tracebacks go to worker files.
    worker_hijack_root_logger=False,
    # Keep task results in backend for 24 hours.
    result_expires=86400,
    # Time (seconds) before an unacked task is considered visible again in broker.
    broker_transport_options={"visibility_timeout": 3600},
)
celery_app.conf.beat_schedule = {
    "cleanup-old-files": {
        "task": "cleanup.old_files",
        "schedule": max(settings.CLEANUP_INTERVAL_MINUTES, 1) * 60,
    },
}
