import os
import json
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import torchaudio
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration


DATA_PATH = Path('app/data/')
DATA_PATH.mkdir(parents=True, exist_ok=True)

DATA_CACHE = Path('app/data/cache_dir/')
DATA_CACHE.mkdir(parents=True, exist_ok=True)

DATA_INPUT = Path('app/data/input/')
DATA_INPUT.mkdir(parents=True, exist_ok=True)

DATA_OUTPUT = Path('app/data/output/')
DATA_OUTPUT.mkdir(parents=True, exist_ok=True)


class BaseModel(ABC):
    """Базовый класс для моделей транскрипции."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    @abstractmethod
    def load_model(self):
        """Загрузить модель."""
        pass

    @abstractmethod
    def transcribe_audio(self, audio_path: str) -> str:
        """Транскрибировать аудиофайл."""
        pass


class WhisperTranscriber(BaseModel):
    """Класс транскрипции на основе openai/whisper-base."""

    def __init__(self, model_name="openai/whisper-base"):
        super().__init__(model_name)
        self.load_model()

    def load_model(self):
        """Загрузка модели и процессора."""
        print(f"Загружаем модель {self.model_name}...")
        self.processor = WhisperProcessor.from_pretrained(self.model_name, cache_dir='')
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name, cache_dir='')
        self.model.to(self.device)
        print("Модель загружена.")

    def preprocess_audio(self, audio_path: str):
        """Загрузка и обработка аудиофайла (конвертация в нужный формат)."""
        audio, rate = torchaudio.load(audio_path)
        if rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=rate, new_freq=16000)(audio)
        return audio

    def transcribe_audio(self, audio_path: str) -> str:
        """Обработать аудиофайл и вернуть транскрипцию."""
        audio = self.preprocess_audio(audio_path)
        inputs = self.processor(audio.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad():
            predicted_ids = self.model.generate(inputs)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def save_transcription(self, text: str, output_path: str):
        """Сохранить транскрипцию в файл JSON."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"transcription": text}, f, ensure_ascii=False, indent=4)
        print(f"Транскрипция сохранена: {output_path}")

    def batch_transcribe(self, audio_files: list, output_dir: str):
        """Обработать несколько аудиофайлов и сохранить результаты."""
        results = {}
        for audio_path in audio_files:
            try:
                text = self.transcribe_audio(audio_path)
                output_path = os.path.join(output_dir, Path(audio_path).stem + ".json")
                self.save_transcription(text, output_path)
                results[audio_path] = text
            except Exception as e:
                print(f"Ошибка при обработке {audio_path}: {e}")
        return results


if __name__ == "__main__":
    transcriber = WhisperTranscriber()
    audio_file = "app/data/input/sample.wav"
    transcript = transcriber.transcribe_audio(audio_file)
    print("Транскрипция:", transcript)
    transcriber.save_transcription(transcript, "../data/output/sample.json")
