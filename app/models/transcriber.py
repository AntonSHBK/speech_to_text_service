import time
from pathlib import Path
from typing import Callable, List, Optional, Union, Tuple, Iterable

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from app.models.base import BaseModel


class FastWhisperTranscriber(BaseModel):
    def __init__(
        self,
        model_name: str,
        cache_dir: Path = "cache_dir",
        device: str = "cpu",
        token: str = None,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
    ):
        super().__init__(model_name, cache_dir, device)
        self.model = self.load_model(
            model_name=model_name,
            use_auth_token=token,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def load_model(
        self,
        model_name: str,
        use_auth_token: str = None,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
    ) -> WhisperModel:
        model = WhisperModel(
            model_size_or_path=model_name,
            device=self.device,
            download_root=str(self.cache_dir),
            use_auth_token=use_auth_token,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        self.logger.info("Модель загружена: %s", model_name)
        return model

    def process(self, audio_path: Union[str, Path]) -> Path:
        return Path(audio_path)

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = "ru",
        task: str = "transcribe",
        beam_size: int = 3,
        chunk_length: int = 10,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        multilingual: bool = False,
        on_progress: Callable[[float], None] | None = None,

        log_progress: bool = False,
        best_of: int = 5,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = (
            0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        ),
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = (-1,),
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, "VadOptions"]] = None,
        max_new_tokens: Optional[int] = None,
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        hotwords: Optional[str] = None,
        language_detection_threshold: Optional[float] = 0.5,
        language_detection_segments: int = 1,
    ) -> dict:
        """
        Выполняет транскрибацию аудиофайла с использованием модели Faster-Whisper.

        Параметры:
            audio_path: Путь к аудиофайлу.
            language: Код языка (например, "ru", "en"). Если не задан, определяется автоматически.
            task: Тип задачи ("transcribe" или "translate").
            beam_size: Размер beam search при декодировании.
            chunk_length: Длина сегментов аудио (в секундах).
            patience: Коэффициент «терпения» beam search.
            length_penalty: Штраф за длину (экспоненциальный).
            repetition_penalty: Штраф за повторение токенов (>1 уменьшает повторы).
            multilingual: Определять язык для каждого сегмента.
            on_progress: Callback для отслеживания прогресса (0–100).

            log_progress: Включить встроенный прогресс модели.
            best_of: Количество кандидатов при sampling.
            no_repeat_ngram_size: Запрет повторов n-грамм (0 — отключено).
            temperature: Температура (или список температур для fallback).
            compression_ratio_threshold: Порог сжатия gzip для детекции ошибок.
            log_prob_threshold: Порог средней лог-вероятности.
            no_speech_threshold: Порог вероятности отсутствия речи.

            condition_on_previous_text: Использовать предыдущий текст как prompt.
            prompt_reset_on_temperature: Сбрасывать prompt при высокой температуре.
            initial_prompt: Начальный prompt.
            prefix: Префикс для первого сегмента.

            suppress_blank: Подавлять пустые токены в начале.
            suppress_tokens: Список токенов для подавления (-1 — дефолтные).
            without_timestamps: Отключить таймкоды.
            max_initial_timestamp: Максимальный начальный таймкод.

            word_timestamps: Включить таймкоды слов.
            prepend_punctuations: Знаки препинания, присоединяемые к следующему слову.
            append_punctuations: Знаки препинания, присоединяемые к предыдущему слову.

            vad_filter: Включить VAD (Silero) для удаления тишины.
            vad_parameters: Параметры VAD.
            max_new_tokens: Максимум токенов на сегмент.

            clip_timestamps: Список интервалов для обработки (в секундах).
            hallucination_silence_threshold: Порог тишины для подавления «галлюцинаций».

            hotwords: Подсказки (ключевые слова) для модели.
            language_detection_threshold: Порог уверенности определения языка.
            language_detection_segments: Количество сегментов для определения языка.

        Возвращает:
            dict:
                {
                    "language": str,
                    "duration": float,
                    "text": str,
                    "segments": List[dict]
                }
        """

        audio_path = self.process(audio_path)

        self.logger.info(
            "Транскрибация начата | модель=%s | файл=%s | язык=%s | задача=%s",
            self.model_name,
            audio_path,
            language,
            task,
        )

        start_time = time.perf_counter()

        segments, info = self.model.transcribe(
            audio=str(audio_path),
            language=language,
            task=task,
            log_progress=log_progress,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=list(suppress_tokens) if suppress_tokens else None,
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
            language_detection_threshold=language_detection_threshold,
            language_detection_segments=language_detection_segments,
        )

        collected_segments: List[Segment] = []
        text_parts: List[str] = []
        last_logged = 0

        duration = info.duration or 0.0
        
        self.logger.info(
            "Длительность аудио: %.2fс.",
            duration,
        )

        for segment in segments:
            collected_segments.append(segment)
            text_parts.append(segment.text.strip())

            if duration > 0:
                progress = min(100.0, (segment.end / duration) * 100)
                if int(progress) >= last_logged + 2:
                    last_logged = int(progress)
                    self.logger.info(
                        "Прогресс | модель=%s | файл=%s | выполнено=%.1f%%",
                        self.model_name,
                        audio_path.name,
                        progress,
                    )
                    if on_progress:
                        on_progress(progress)

        processing_time = time.perf_counter() - start_time
        speed_ratio = duration / processing_time if processing_time > 0 else 0.0

        self.logger.info(
            "Транскрибация завершена | модель=%s | файл=%s | длительность=%.2fс | обработка=%.2fс | скорость=x%.2f",
            self.model_name,
            audio_path.name,
            duration,
            processing_time,
            speed_ratio,
        )

        full_text = " ".join(part.strip() for part in text_parts if part and part.strip())

        return {
            "language": info.language,
            "duration": duration,
            "text": full_text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in collected_segments
            ],
        }