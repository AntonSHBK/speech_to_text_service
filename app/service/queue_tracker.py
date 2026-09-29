from __future__ import annotations

from redis import Redis

from app.settings import settings

QUEUE_KEY = "transcribe:queue:pending"
CANCELLED_TASK_KEY_PREFIX = "transcribe:task:cancelled:"
CANCELLATION_TTL_SECONDS = 24 * 60 * 60

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


def remove_task(task_id: str | None) -> None:
    """Удаляет задачу из списка ожидающих задач интерфейса."""
    if not task_id:
        return
    try:
        _client().lrem(QUEUE_KEY, 0, task_id)
    except Exception:
        return


def request_task_cancellation(task_id: str) -> bool:
    """Сохраняет запрос на отмену, доступный worker после перезапуска."""
    try:
        _client().set(
            f"{CANCELLED_TASK_KEY_PREFIX}{task_id}",
            "1",
            ex=CANCELLATION_TTL_SECONDS,
        )
        return True
    except Exception:
        return False


def is_task_cancellation_requested(task_id: str | None) -> bool:
    """Проверяет, запросил ли API отмену задачи."""
    if not task_id:
        return False
    try:
        return bool(_client().exists(f"{CANCELLED_TASK_KEY_PREFIX}{task_id}"))
    except Exception:
        return False


def get_queue_position(task_id: str) -> int | None:
    try:
        queue_items = _client().lrange(QUEUE_KEY, 0, -1)
        idx = queue_items.index(task_id)
        return idx + 1
    except ValueError:
        return None
    except Exception:
        return None
