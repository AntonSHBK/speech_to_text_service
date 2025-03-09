import sys
import os
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.models.transcriber import WhisperTranscriber


@pytest.fixture(scope="module")
def transcriber():
    """Фикстура для загрузки модели перед тестами."""
    return WhisperTranscriber(model_name="openai/whisper-tiny", cache_dir="./app/data/cache_dir")


def test_model_load(transcriber: WhisperTranscriber):
    """Проверяем, что модель загружается без ошибок."""
    assert transcriber.model is not None, "Ошибка: модель не загружена"
    assert transcriber.processor is not None, "Ошибка: процессор не загружен"


def test_transcribe_audio(transcriber: WhisperTranscriber):
    """Проверяем, что транскрипция аудио работает."""
    audio_tensor = torch.randn(1, 16000)  # Генерируем случайный аудиосигнал
    transcript = transcriber.transcribe_audio(audio_tensor, language="ru", task="transcribe")

    assert isinstance(transcript, str), "Ошибка: транскрипция должна возвращать строку"
    assert len(transcript) > 0, "Ошибка: транскрипция пустая"


def test_transcribe_with_kwargs(transcriber: WhisperTranscriber):
    """Проверяем передачу `kwargs` в генерацию текста."""
    audio_tensor = torch.randn(1, 16000)

    transcript = transcriber.transcribe_audio(
        audio_tensor, 
        language="ru", 
        task="transcribe", 
        temperature=0.7, 
        max_new_tokens=50
    )

    assert isinstance(transcript, str), "Ошибка: транскрипция должна быть строкой"
    assert len(transcript) > 0, "Ошибка: транскрипция пустая"


# def test_transcribe_audio_file(transcriber: WhisperTranscriber):
#     """Проверяем транскрипцию аудиофайла."""
#     audio_path = "./app/data/input/test.wav"
#     transcript = transcriber.transcribe_audio_file(audio_path, language="ru", task="transcribe")

#     assert isinstance(transcript, str), "Ошибка: транскрипция должна быть строкой"
#     assert len(transcript) > 0, "Ошибка: транскрипция пустая"
