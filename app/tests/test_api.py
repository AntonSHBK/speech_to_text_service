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

