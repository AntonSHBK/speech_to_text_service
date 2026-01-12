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


@pytest.mark.order(2)
def test_transcription():
    """Проверка транскрибации тестового аудио"""

    assert TEST_AUDIO_FILE.exists(), f"Тестовый файл не найден: {TEST_AUDIO_FILE}"

    transcriber = FastWhisperTranscriber(
        model_name=MODEL_NAME,
        cache_dir=settings.CACHE_DIR,
        device=settings.DEVICE,
    )

    result = transcriber.transcribe(
        audio_path=str(TEST_AUDIO_FILE),
        language="ru",
    )

    assert isinstance(result, dict), "Результат не является словарём"
    assert "segments" in result, "В результате отсутствует поле 'segments'"
    assert isinstance(result["segments"], list), "'segments' должен быть списком"
    assert len(result["segments"]) > 0, "Транскрипция вернула пустой результат"

    first = result["segments"][0]
    for key in ("start", "end", "text"):
        assert key in first, f"В сегменте отсутствует поле '{key}'"

    full_text = " ".join(seg["text"] for seg in result["segments"])
    print("\n===== TRANSCRIPTION RESULT =====")
    print(full_text)
    print("================================")
