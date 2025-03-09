import asyncio

import torch

from app.models.transcriber import WhisperTranscriber


class ModelHandler:
    """
    Управляет загрузкой модели и выполнением транскрипции.
    Позволяет легко заменять или обновлять модели в будущем.
    """

    def __init__(self, model_name="openai/whisper-base", cache_dir=None, device=None, **kwargs):
        """
        Инициализация обработчика модели.

        Args:
            model_name (str): Название модели Hugging Face (по умолчанию "openai/whisper-base").
            cache_dir (str, optional): Директория для кеша модели.
            device (str, optional): Устройство для выполнения ("cpu" или "cuda").
            **kwargs: Дополнительные параметры для модели (например, `language`, `temperature`).
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_kwargs = kwargs
        self.lock = asyncio.Lock()

        self.transcriber = WhisperTranscriber(
            model_name=self.model_name,
            cache_dir=self.cache_dir,
            device=self.device,
            **self.model_kwargs
        )

    async def transcribe(self, audio_path: str, language: str = "ru", task: str = "transcribe", **kwargs) -> str:
        """
        Выполняет транскрипцию аудиофайла.

        Args:
            audio_path (str): Путь к аудиофайлу.
            language (str, optional): Язык транскрипции (по умолчанию "ru").
            task (str, optional): Тип задачи ("transcribe" — обычная транскрибация, "translate" — перевод в английский).
            **kwargs: Дополнительные параметры для `transcribe_audio_file()`.

        Returns:
            str: Транскрибированный текст.
        """
        async with self.lock:  # Гарантируем, что `generate()` вызывается только одним потоком
            return self.transcriber.transcribe_audio_file(audio_path, language=language, task=task, **kwargs)

    def batch_transcribe(self, audio_files: list, output_dir: str, language: str = "ru", task: str = "transcribe", **kwargs):
        """
        Выполняет транскрипцию нескольких аудиофайлов.

        Args:
            audio_files (list): Список путей к аудиофайлам.
            output_dir (str): Папка для сохранения результатов.
            language (str, optional): Язык транскрипции (по умолчанию "ru").
            task (str, optional): Тип задачи ("transcribe" или "translate").
            **kwargs: Дополнительные параметры для генерации (`temperature`, `max_new_tokens` и т. д.).

        Returns:
            dict: Словарь с транскрипциями {файл: текст}.
        """
        return self.transcriber.batch_transcribe(audio_files, output_dir, language=language, task=task, **kwargs)


model_handler = ModelHandler(
    model_name="openai/whisper-tiny",
    cache_dir="app/data/cache_dir"
)

# 🔥 **Пример использования**
if __name__ == "__main__":
    import asyncio

    model_handler = ModelHandler(
        model_name="openai/whisper-tiny",
        cache_dir="app/data/cache_dir",
        device="cpu",
    )

    async def main():
        audio_file = "./app/data/input/test.wav"
        transcript = await model_handler.transcribe(audio_file, language="ru", task="transcribe")
        print("Транскрипция:", transcript)

    asyncio.run(main())
