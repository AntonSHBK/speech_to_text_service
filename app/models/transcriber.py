import os
import json
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod

import torch
import torchaudio
import torchaudio.transforms as transforms
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf


class BaseModel(ABC):
    """Базовый класс для моделей транскрипции."""
    
    def __init__(self, model_name: str, cache_dir: Path = ''):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.cache_dir = cache_dir

    @abstractmethod
    def load_model(self):
        """Загрузить модель."""
        pass

    @abstractmethod
    def transcribe_audio(self, audio_path: str) -> str:
        """Транскрибировать аудиофайл."""
        pass


class WhisperTranscriber(BaseModel):
    """Класс транскрипции на основе openai/whisper"""

    def __init__(self, model_name="openai/whisper-tiny", cache_dir: str = ''):
        super().__init__(model_name, cache_dir)
        self.load_model()

    def load_model(self):
        """Загрузка модели и процессора."""
        print(f"Загружаем модель {self.model_name}... cache_dir - {self.cache_dir}")
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.model.to(self.device)
        print("Модель загружена.")

    @staticmethod
    def load_audio_file(audio_path: str, target_sample_rate: int = 16000):
        """
        Загружает аудиофайл и конвертирует его в 16 kHz WAV при необходимости.

        Поддерживаемые форматы: WAV, MP3, FLAC, OGG, AAC, M4A, WMA, AIFF.

        Если файл не в формате WAV, он автоматически конвертируется в WAV с помощью FFmpeg.

        Args:
            audio_path (str): Путь к аудиофайлу.
            target_sample_rate (int): Целевая частота дискретизации (по умолчанию 16 kHz).

        Returns:
            torch.Tensor: Аудиоданные в формате (1, N).
            int: Частота дискретизации (обычно 16000).
        """
        audio_path = Path(audio_path)

        # Проверяем, является ли файл WAV
        if audio_path.suffix.lower() not in [".wav"]:
            print(f"⚠ Конвертируем {audio_path.suffix} в WAV...")
            temp_wav_path = audio_path.with_suffix(".wav")

            # Используем FFmpeg для конвертации
            subprocess.run([
                "ffmpeg", "-i", str(audio_path), "-ar", str(target_sample_rate), 
                "-ac", "1", "-y", str(temp_wav_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            audio_path = temp_wav_path

        # Загружаем аудиофайл через torchaudio
        audio, sample_rate = torchaudio.load(audio_path)

        # Если частота не совпадает с 16 kHz, ресемплируем
        if sample_rate != target_sample_rate:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            audio = resampler(audio)

        # Преобразуем стерео в моно (усредняем каналы)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio, target_sample_rate

    @staticmethod
    def load_audio_sf(audio_path):
        """Загружает аудиофайл через soundfile, если torchaudio не работает."""
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        return torch.tensor(audio), sample_rate

    def preprocess_audio(self, audio_path: str):
        """Загрузка и обработка аудиофайла (конвертация в WAV, ресемплирование)."""

        # Проверяем, существует ли файл
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Файл {audio_path} не найден!")

        # Загружаем аудио (через torchaudio или soundfile)
        try:
            audio, rate = torchaudio.load(audio_path)
        except RuntimeError:
            print(f"⚠ Ошибка загрузки {audio_path} через torchaudio, пробуем soundfile...")
            audio, rate = self.load_audio_sf(audio_path)

        # Если частота не 16 kHz, ресемплируем
        if rate != 16000:
            resampler = transforms.Resample(orig_freq=rate, new_freq=16000)
            audio = resampler(audio)

        # Приводим стерео в моно
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio.to(torch.float32)


    def transcribe_audio(
        self, 
        audio_path: str, 
        language: str = "ru", 
        task: str = "transcribe", 
        temperature: float = 0.1, 
        max_new_tokens: int = 100, 
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Обработать аудиофайл и вернуть транскрипцию.

        Args:
            audio_path (str): Путь к аудиофайлу.
            language (str): Язык транскрипции (по умолчанию "ru").
            task (str): Тип задачи ("transcribe" — транскрибация, "translate" — перевод в английский).
            temperature (float): Контроль случайности генерации (0.1 по умолчанию, 0 — строгое соответствие).
            max_new_tokens (int): Максимальное количество новых токенов в выходе.
            repetition_penalty (float): Штраф за повторяющиеся слова (чем больше, тем меньше повторов).

        Returns:
            str: Транскрибированный текст.
        """

        # Загружаем и предобрабатываем аудио
        audio = self.preprocess_audio(audio_path)
        inputs = self.processor(
            audio.squeeze(0), sampling_rate=16000, return_tensors="pt"
        )

        # Явно передаём attention_mask
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        # Принудительно указываем язык и задачу (транскрипция или перевод)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,  # Добавляем mask
                forced_decoder_ids=forced_decoder_ids,  # Принудительно задаём язык
                temperature=temperature,  # Контроль случайности
                # max_new_tokens=max_new_tokens,  # Ограничиваем длину
                # repetition_penalty=repetition_penalty,  # Наказание за повторы
                # suppress_tokens=[50257]  # Запрещаем вставку английских слов
            )

        # Декодируем текст
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Удаляем зацикленные повторения
        words = transcription.split()
        if len(set(words)) < len(words) * 0.3:  # Если 70% слов одинаковые
            transcription = "Ошибка транскрипции: слишком много повторов"

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
    transcriber = WhisperTranscriber(model_name="openai/whisper-tiny", cache_dir='./app/data/cache_dir')
    audio_file = "./app/data/input/test.wav"
    transcript = transcriber.transcribe_audio(audio_file)
    print("Транскрипция:", transcript)
    transcriber.save_transcription(transcript, "./app/data/output/sample.json")
