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


def test_cancel_queued_transcription_task(client: TestClient, monkeypatch):
    task_id = "fe5fc685-c720-49ea-9796-3379c2fd4e11"
    revoked_task_ids: list[str] = []
    removed_task_ids: list[str] = []

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
        "request_task_cancellation",
        lambda requested_task_id: requested_task_id == task_id,
    )
    monkeypatch.setattr(transcribe_router, "remove_task", removed_task_ids.append)

    response = client.delete(f"/transcribe/tasks/{task_id}")

    assert response.status_code == 202
    assert response.json() == {
        "task_id": task_id,
        "status": "cancellation_requested",
        "state_before_cancellation": "PENDING",
        "is_running": False,
        "detail": "Задача отменена до начала обработки.",
    }
    assert revoked_task_ids == [task_id]
    assert removed_task_ids == [task_id]


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
