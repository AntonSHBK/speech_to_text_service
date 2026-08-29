import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.models.transcriber import FastWhisperTranscriber
from app.settings import settings

TEST_AUDIO_FILE = settings.AUDIO_DIR / "test_video_3.mp4"
MODEL_NAME = "Systran/faster-whisper-tiny"


@pytest.mark.order(1)
def test_model_load():
    """Проверка, что модель корректно загружается"""

    transcriber = FastWhisperTranscriber(
        model_name=MODEL_NAME,
        cache_dir=settings.CACHE_DIR,
        device=settings.DEVICE,
    )

    assert transcriber.model is not None, "Модель не была загружена"
    print("Модель успешно загружена")

