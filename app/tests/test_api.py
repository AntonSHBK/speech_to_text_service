import sys
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# чтобы тест видел app/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.main import app
from app.routers import transcribe as transcribe_router
from app.settings import settings
from app.service.transcriber import transcriber_service


TEST_AUDIO_FILE: Path = settings.AUDIO_DIR / "test_video_3.mp4"


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as client:
        yield client

@pytest.mark.order(1)
def test_service_ready(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}


class FakeAsyncResult:
    def __init__(self, state: str):
        self.state = state
        self.info = {}


def test_enqueue_transcription_tracks_task_before_publishing(client: TestClient, monkeypatch):
    task_id = "039cd2f0-852b-4ce4-8cb6-55140c245359"
    operations: list[tuple[str, str]] = []

    monkeypatch.setattr(transcribe_router, "uuid4", lambda: task_id)
    monkeypatch.setattr(
        transcribe_router,
        "enqueue_task",
        lambda tracked_task_id: operations.append(("enqueue", tracked_task_id)) or 1,
    )
    monkeypatch.setattr(
        transcribe_router.process_transcription,
        "apply_async",
        lambda *, kwargs, task_id: operations.append(("publish", task_id)),
    )

    response = transcribe_router._enqueue_transcription_task(source_url="https://example.com/a.mp3")

    assert operations == [("enqueue", task_id), ("publish", task_id)]
    assert response["task_id"] == task_id
    assert response["queue_position"] == 1


def test_enqueue_transcription_removes_tracker_entry_when_publishing_fails(
    client: TestClient,
    monkeypatch,
):
    task_id = "ca61ee4a-6f14-4c5f-8f2f-9156d99acfa"
    removed_task_ids: list[str] = []

    monkeypatch.setattr(transcribe_router, "uuid4", lambda: task_id)
    monkeypatch.setattr(transcribe_router, "enqueue_task", lambda _: 1)
    monkeypatch.setattr(transcribe_router, "remove_task", removed_task_ids.append)
    monkeypatch.setattr(
        transcribe_router.process_transcription,
        "apply_async",
        lambda **_: (_ for _ in ()).throw(RuntimeError("Celery is unavailable")),
    )

    with pytest.raises(RuntimeError, match="Celery is unavailable"):
        transcribe_router._enqueue_transcription_task(source_url="https://example.com/a.mp3")

    assert removed_task_ids == [task_id]


def test_cancel_queued_transcription_task(client: TestClient, monkeypatch):
    task_id = "fe5fc685-c720-49ea-9796-3379c2fd4e11"
    revoked_task_ids: list[str] = []
    removed_task_ids: list[str] = []
    stored_task_ids: list[str] = []

    monkeypatch.setattr(
        transcribe_router.celery_app,
        "AsyncResult",
        lambda requested_task_id: FakeAsyncResult("PENDING"),
    )
    monkeypatch.setattr(
        transcribe_router.celery_app.control,
        "revoke",
        revoked_task_ids.append,
    )
    monkeypatch.setattr(
        transcribe_router,
        "_store_cancelled_task_result",
        lambda stored_task_id: stored_task_ids.append(stored_task_id) or True,
    )
    monkeypatch.setattr(
        transcribe_router,
        "request_task_cancellation",
        lambda requested_task_id: requested_task_id == task_id,
    )
    monkeypatch.setattr(transcribe_router, "remove_task", removed_task_ids.append)

    response = client.delete(f"/transcribe/tasks/{task_id}")

    assert response.status_code == 202
    payload = response.json()
    assert payload["task_id"] == task_id
    assert payload["status"] == "cancellation_requested"
    assert payload["state_before_cancellation"] == "PENDING"
    assert payload["is_running"] is False
    assert revoked_task_ids == [task_id]
    assert removed_task_ids == [task_id]
    assert stored_task_ids == [task_id]


def test_cancel_completed_transcription_task_returns_conflict(client: TestClient, monkeypatch):
    task_id = "10a4d58b-dd87-466b-9af7-4d9d0561141e"

    monkeypatch.setattr(
        transcribe_router.celery_app,
        "AsyncResult",
        lambda requested_task_id: FakeAsyncResult("SUCCESS"),
    )

    response = client.delete(f"/transcribe/tasks/{task_id}")

    assert response.status_code == 409
    assert response.json() == {"detail": "Задача уже завершена со статусом SUCCESS."}


def test_cancel_running_transcription_task_persists_cancelled_status(
    client: TestClient,
    monkeypatch,
):
    task_id = "9f02ef67-7dc3-496a-8ddd-1ad0854c3700"
    stored_task_ids: list[str] = []
    revoked_task_ids: list[str] = []

    monkeypatch.setattr(
        transcribe_router.celery_app,
        "AsyncResult",
        lambda requested_task_id: FakeAsyncResult("PROGRESS"),
    )
    monkeypatch.setattr(transcribe_router, "request_task_cancellation", lambda _: True)
    monkeypatch.setattr(
        transcribe_router,
        "_store_cancelled_task_result",
        lambda stored_task_id: stored_task_ids.append(stored_task_id) or True,
    )
    monkeypatch.setattr(
        transcribe_router.celery_app.control,
        "revoke",
        revoked_task_ids.append,
    )
    monkeypatch.setattr(transcribe_router, "remove_task", lambda _: None)

    response = client.delete(f"/transcribe/tasks/{task_id}")

    assert response.status_code == 202
    assert response.json()["status"] == "cancellation_requested"
    assert response.json()["is_running"] is True
    assert stored_task_ids == [task_id]
    assert revoked_task_ids == [task_id]


def test_cancelled_transcription_task_status(client: TestClient, monkeypatch):
    task_id = "bf6a39b2-2026-4501-8976-44d655711ff7"
    monkeypatch.setattr(
        transcribe_router.celery_app,
        "AsyncResult",
        lambda requested_task_id: FakeAsyncResult("CANCELLED"),
    )

    response = client.get(f"/transcribe/tasks/{task_id}")

    assert response.status_code == 200
    assert response.json()["task_id"] == task_id
    assert response.json()["status"] == "cancelled"


def test_cancel_queued_task_returns_service_unavailable_when_status_is_not_saved(
    client: TestClient,
    monkeypatch,
):
    task_id = "039cd2f0-852b-4ce4-8cb6-55140c245359"
    monkeypatch.setattr(
        transcribe_router.celery_app,
        "AsyncResult",
        lambda requested_task_id: FakeAsyncResult("PENDING"),
    )
    monkeypatch.setattr(transcribe_router, "request_task_cancellation", lambda _: True)
    monkeypatch.setattr(transcribe_router, "_store_cancelled_task_result", lambda _: False)

    response = client.delete(f"/transcribe/tasks/{task_id}")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Не удалось сохранить статус отмененной задачи в Celery backend."
    }
