import os
import json
from pathlib import Path

from langcodes import Language
import torch
import torchaudio
import torchaudio.transforms as transforms
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import ffmpeg

from app.models.base import BaseModel


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".aiff"}


class WhisperTranscriber(BaseModel):
    """Класс транскрипции на основе openai/whisper"""

    def __init__(self, model_name="openai/whisper-tiny", cache_dir: str = '', device: str = None, **kwargs):
        """
        Инициализация WhisperTranscriber.

        Args:
            model_name (str): Название модели Hugging Face.
            cache_dir (str): Директория для кеша модели.
            device (str): Устройство (cpu/gpu).
            **kwargs: Дополнительные параметры для модели.
        """
        # Используем параметры из kwargs
        device = device or kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_name, cache_dir, device, **kwargs)
        
        # Загружаем модель
        self.load_model()

    def load_model(self):
        """Загрузка модели и процессора."""
        print(f"Загружаем модель {self.model_name}... cache_dir - {self.cache_dir}")
        
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir)
        
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, cache_dir=self.cache_dir, device_map="auto"
        ).to(self.device)
        
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

        # Проверяем, является ли файл WAV, если нет - конвертируем
        if audio_path.suffix.lower() not in [".wav"]:
            print(f"⚠ Конвертируем {audio_path.suffix} в WAV...")

            temp_wav_path = audio_path.with_suffix(".wav")

            try:
                (
                    ffmpeg
                    .input(str(audio_path))
                    .output(str(temp_wav_path), format="wav", ar=target_sample_rate, ac=1)
                    .run(overwrite_output=True, quiet=True)
                )
                audio_path = temp_wav_path  # Обновляем путь на новый WAV
            except Exception as e:
                raise RuntimeError(f"Ошибка при конвертации {audio_path}: {e}")

        # Загружаем аудиофайл через torchaudio
        try:
            audio, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки аудио: {e}")

        # Если частота не совпадает с 16 kHz, ресемплируем
        if sample_rate != target_sample_rate:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            audio = resampler(audio)

        # Преобразуем стерео в моно (усредняем каналы)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        return audio, sample_rate

    @staticmethod
    def load_audio_torchaudio(audio_path: str, **kwargs):
        """Загружает аудиофайл через torchaudio"""
        audio, rate = torchaudio.load(audio_path)
        return torch.tensor(audio), rate
    
    @staticmethod
    def load_audio_soundfile(audio_path: str, **kwargs):
        """Загружает аудиофайл через soundfile"""
        dtype = kwargs.get('load_dtype', 'float32')
        audio, rate = sf.read(audio_path, dtype=dtype)
        return torch.tensor(audio), rate

    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Загрузка и обработка аудиофайла (конвертация в WAV, ресемплирование)."""

        # Проверяем, существует ли файл
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Файл {audio_path} не найден!")
        
        # Проверка на формат
        if Path(audio_path).suffix.lower() not in ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(f"Формат файла {audio_path} не поддерживается!")

        # Загружаем аудио (через torchaudio или soundfile)
        try:
            audio, rate = self.load_audio_torchaudio(audio_path, **self.params_dict)
        except RuntimeError:
            print(f"⚠ Ошибка загрузки {audio_path} через torchaudio, пробуем soundfile...")
            audio, rate = self.load_audio_soundfile(audio_path, **self.params_dict)

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
        audio_tensor: torch.Tensor, 
        language: str = "ru", 
        task: str = "transcribe", 
        **kwargs
    ) -> str:
        """
        Выполняет транскрипцию аудиоданных в формате тензора с возможностью гибкой настройки параметров генерации.

        Args:
            audio_tensor (torch.Tensor): Аудиоданные в формате (1, N), предобработанные методом `preprocess_audio`.
            language (str, optional): Язык транскрипции. По умолчанию `"ru"`.
                - `"ru"` - русский
                - `"en"` - английский
            task (str, optional): Тип задачи. По умолчанию `"transcribe"`.
                - `"transcribe"` - стандартная транскрипция (оставляет исходный язык)
                - `"translate"` - перевод на английский
            **kwargs: Дополнительные параметры для метода `self.model.generate()`:
                - `temperature` (float, optional): Параметр, контролирующий случайность предсказания. По умолчанию `0.1`.
                    - `0.0` - детерминированный результат
                    - `>0.0` - больше случайности (например, `0.1`)
                - `max_new_tokens` (int, optional): Максимальное количество новых токенов в выходе. 
                    - Рекомендуется устанавливать в соответствии с ожидаемой длиной транскрипции.
                    - Например, для коротких аудиозаписей `100`, для длинных — `500` и более.
                - `repetition_penalty` (float, optional): Штраф за повторяющиеся слова.
                    - `1.0` - без штрафа.
                    - `> 1.0` - чем выше, тем меньше повторов (рекомендуется `1.1-1.5`).
                - `length_penalty` (float, optional): Регулировка длины выхода.
                    - `> 1.0` - способствует более длинным транскрипциям.
                    - `< 1.0` - сокращает текст.
                - `num_beams` (int, optional): Количество "лучей" в beam search (по умолчанию `1`).
                    - Большее значение может улучшить качество, но увеличивает вычислительную нагрузку.
                    - Оптимально `3-5` для улучшенного поиска.
                - `early_stopping` (bool, optional): Останавливает генерацию при `eos_token` (по умолчанию `True`).
                    - Полезно для предотвращения лишних продолжений в тексте.
                - `suppress_tokens` (list, optional): Запрещенные токены.
                    - Например, `[50257]` исключает английские слова.
                - `top_k` (int, optional): Выбирает следующее слово из `k` наиболее вероятных.
                    - Используется для управления разнообразием.
                    - Например, `top_k=50` ограничит выбор 50 самыми вероятными вариантами.
                - `top_p` (float, optional): Ограничивает выбор следующего слова на основе совокупной вероятности `p`.
                    - Альтернатива `top_k`, полезно для более естественной речи.
                    - Рекомендуется `top_p=0.95` для баланса между точностью и разнообразием.
                - `do_sample` (bool, optional): Включает случайное сэмплирование.
                    - Полезно, если требуется более творческое или вариативное предсказание.

        **Рекомендации по выбору параметров:**
            - **Детальная и точная транскрипция** → `num_beams=5, early_stopping=True, repetition_penalty=1.1`
            - **Максимальная скорость обработки** → `num_beams=1, temperature=0.0`
            - **Генерация длинных текстов** → `length_penalty=1.5, max_new_tokens=300`
            - **Борьба с повторами** → `repetition_penalty=1.2`
            - **Эксперимент с творческим режимом** → `do_sample=True, top_k=50, top_p=0.95, temperature=0.7`

        Returns:
            str: Транскрибированный текст.
        """

        # Преобразуем аудио в формат Whisper
        inputs = self.processor(audio_tensor.squeeze(0), sampling_rate=16000, return_tensors="pt")

        # Подготавливаем input для модели
        input_features = inputs.input_features.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None
        
        # Принудительно указываем язык и задачу
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)

        # Передаем параметры в `generate`
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids,
                **kwargs
            )

        # Проверяем, есть ли результат
        if predicted_ids is None or predicted_ids.shape[1] == 0:
            return "Ошибка: транскрипция не удалась"

        # Декодируем текст
        skip_special_tokens = kwargs.get('decode_skip_special_tokens', True)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=skip_special_tokens)[0]
        return transcription if transcription.strip() else "Ошибка: пустая транскрипция"

    def transcribe_audio_file(
            self, 
            audio_path: str, 
            language: str = "ru", 
            task: str = "transcribe", 
            **kwargs
        ) -> str:
        """
        Выполняет транскрипцию аудиофайла (загружает его, предобрабатывает и передает в `transcribe_audio`).

        Args:
            audio_path (str): Путь к аудиофайлу.
            language (str, optional): Язык транскрипции. По умолчанию `"ru"`.
            task (str, optional): Тип задачи (`"transcribe"` или `"translate"`).
            **kwargs: Дополнительные параметры для `self.model.generate()`.

        Returns:
            str: Транскрибированный текст.
        """
        audio_tensor = self.preprocess_audio(audio_path)
        transcription = self.transcribe_audio(audio_tensor, language, task, **kwargs)
        return transcription

    def save_transcription(self, text: str, output_path: str):
        """Сохранить транскрипцию в файл JSON."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"transcription": text}, f, ensure_ascii=False, indent=4)
        print(f"Транскрипция сохранена: {output_path}")

    def batch_transcribe(self, audio_files: list, output_dir: str, **kwargs):
        """Обработать несколько аудиофайлов и сохранить результаты."""
        results = {}
        for audio_path in audio_files:
            try:
                language = kwargs.get('language', 'ru')
                task = kwargs.get('task', 'transcribe')
                text = self.transcribe_audio_file(audio_path, language, task, **kwargs)
                output_path = os.path.join(output_dir, Path(audio_path).stem + ".json")
                self.save_transcription(text, output_path)
                results[audio_path] = text
            except Exception as e:
                print(f"Ошибка при обработке {audio_path}: {e}")
        return results


if __name__ == "__main__":  
    transcriber = WhisperTranscriber(model_name="openai/whisper-tiny", cache_dir='app/data/cache_dir')
    audio_file = "./app/data/input/test.wav"
    transcript = transcriber.transcribe_audio_file(audio_file)
    print("Транскрипция:", transcript)
    transcriber.save_transcription(transcript, "./app/data/output/sample.json")
