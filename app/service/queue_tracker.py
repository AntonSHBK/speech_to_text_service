from __future__ import annotations

from redis import Redis

from app.settings import settings

QUEUE_KEY = "transcribe:queue:pending"

_redis_client: Redis | None = None


def _client() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    return _redis_client


def enqueue_task(task_id: str) -> int | None:
    try:
        client = _client()
        client.rpush(QUEUE_KEY, task_id)
        return get_queue_position(task_id)
    except Exception:
        return None


def mark_task_started(task_id: str | None) -> None:
    if not task_id:
        return
    try:
        _client().lrem(QUEUE_KEY, 0, task_id)
    except Exception:
        return


def get_queue_position(task_id: str) -> int | None:
    try:
        queue_items = _client().lrange(QUEUE_KEY, 0, -1)
        idx = queue_items.index(task_id)
        return idx + 1
    except ValueError:
        return None
    except Exception:
        return None
