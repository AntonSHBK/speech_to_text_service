import sys
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# чтобы тест видел app/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.main import app
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


@pytest.mark.order(2)
def test_transcription_flow(client: TestClient):
    assert TEST_AUDIO_FILE.exists(), f"Test file not found: {TEST_AUDIO_FILE}"

    with open(TEST_AUDIO_FILE, "rb") as f:
        response = client.post(
            "/transcribe/",
            files={"file": ("test_video_3.mp4", f, "video/mp4")},
            params={
                "language": "ru",
                "save_file": True,
                "save_result": True,
            },
        )

    assert response.status_code == 200, response.text

    data = response.json()

    # Проверка, что транскрипция выполнена
    assert "text" in data
    assert len(data["text"]) > 0

    # Проверка сегментов
    assert "segments" in data
    assert isinstance(data["segments"], list)
    assert len(data["segments"]) > 0

    # Проверка сохранения результата
    assert "result_file" in data
    result_path = Path(data["result_file"])
    assert result_path.exists(), f"Result file not found: {result_path}"

    # Проверка, что исходный файл сохранён
    saved_files = list(settings.AUDIO_DIR.glob("*test_video_3*"))
    assert len(saved_files) > 0, "Original audio file was not saved"
