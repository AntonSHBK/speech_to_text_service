from app.models.transcriber import WhisperTranscriber

class ModelHandler:
    """
    Управляет загрузкой модели и выполнением транскрипции.
    Позволяет легко заменять или обновлять модели в будущем.
    """

    def __init__(self, model_name="openai/whisper-base", cache_dir=None):
        """
        Инициализация обработчика модели.

        Args:
            model_name (str): Название модели Hugging Face (по умолчанию "openai/whisper-base").
            cache_dir (str, optional): Директория для кеша модели.
        """
        self.model_name = model_name
        self.transcriber = self.load_model(cache_dir)

    def load_model(self, cache_dir):
        """
        Загружает модель для транскрипции.

        Args:
            cache_dir (str, optional): Путь к кешу модели.

        Returns:
            WhisperTranscriber: Экземпляр класса транскрипции.
        """
        return WhisperTranscriber(model_name=self.model_name, cache_dir=cache_dir)

    def transcribe(self, audio_path: str, **kwargs) -> str:
        """
        Выполняет транскрипцию аудиофайла.

        Args:
            audio_path (str): Путь к аудиофайлу.
            **kwargs: Дополнительные параметры для `transcribe_audio()`.

        Returns:
            str: Транскрибированный текст.
        """
        return self.transcriber.transcribe_audio(audio_path, **kwargs)

    def batch_transcribe(self, audio_files: list, output_dir: str):
        """
        Выполняет транскрипцию нескольких аудиофайлов.

        Args:
            audio_files (list): Список путей к аудиофайлам.
            output_dir (str): Папка для сохранения результатов.

        Returns:
            dict: Словарь с транскрипциями {файл: текст}.
        """
        return self.transcriber.batch_transcribe(audio_files, output_dir)


# 🔥 **Пример использования**
if __name__ == "__main__":
    model_handler = ModelHandler()
    audio_file = "app/data/input/sample.wav"

    transcript = model_handler.transcribe(audio_file)
    print("Транскрипция:", transcript)
