import sys
import os
import pytest
import httpx
from fastapi.testclient import TestClient

# Добавляем путь к корневой директории проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.main import app

client = TestClient(app)

@pytest.fixture
def test_audio():
    """Фикстура для тестового аудиофайла."""
    return "./app/data/input/test.wav"


def test_transcribe_endpoint(test_audio):
    """Тестируем эндпоинт `/transcribe/` с реальным аудиофайлом."""
    with open(test_audio, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        response = client.post("/transcribe/", files=files)

    assert response.status_code == 200, "Ошибка: статус-код API не 200"
    data = response.json()

    assert "transcription" in data, "Ошибка: API не вернуло транскрипцию"
    assert isinstance(data["transcription"], str), "Ошибка: транскрипция должна быть строкой"
    assert len(data["transcription"]) > 0, "Ошибка: транскрипция пустая"


def test_transcribe_with_params(test_audio):
    """Тестируем передачу параметров `kwargs` в API."""
    with open(test_audio, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        response = client.post(
            "/transcribe/?temperature=0.5&max_new_tokens=50", files=files
        )

    assert response.status_code == 200, "Ошибка: API не вернуло 200"
    data = response.json()

    assert "transcription" in data, "Ошибка: API не вернуло транскрипцию"
    assert "parameters" in data, "Ошибка: API не вернуло параметры"
    assert data["parameters"]["temperature"] == 0.5, "Ошибка: параметр `temperature` не передался"
    assert data["parameters"]["max_new_tokens"] == 50, "Ошибка: параметр `max_new_tokens` не передался"

# def test_invalid_file():
#     """Тестируем API с неверным файлом (не аудио)."""
#     files = {"file": ("invalid.txt", b"not an audio file", "text/plain")}
    
#     response = client.post("/transcribe/", files=files)

#     # Проверяем, что API возвращает код ошибки 400
#     assert response.status_code == 400
#     assert response.json() == {"detail": "Формат файла app/data/input/invalid.txt не поддерживается!"}



