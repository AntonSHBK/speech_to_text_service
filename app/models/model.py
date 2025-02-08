from models.transcriber import WhisperTranscriber

class ModelHandler:
    """Класс для управления моделью транскрипции."""
    
    def __init__(self, model_name="openai/whisper-base"):
        """Инициализирует обработчик модели."""
        self.model_name = model_name
        self.transcriber = self.load_model()

    def load_model(self):
        """Загружает модель в зависимости от указанного имени."""
        if "whisper" in self.model_name:
            return WhisperTranscriber(model_name=self.model_name)
        else:
            raise ValueError(f"Неизвестная модель: {self.model_name}")

    def transcribe(self, audio_path: str) -> str:
        """Запускает процесс транскрипции для одного аудиофайла."""
        return self.transcriber.transcribe_audio(audio_path)

    def batch_transcribe(self, audio_files: list, output_dir: str):
        """Запускает процесс транскрипции для списка аудиофайлов."""
        return self.transcriber.batch_transcribe(audio_files, output_dir)


if __name__ == "__main__":
    model_handler = ModelHandler()
    audio_file = "./app/data/input/test.wav"
    transcript = model_handler.transcribe(audio_file)
    print("Результат:", transcript)
