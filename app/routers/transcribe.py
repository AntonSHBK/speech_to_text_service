from pathlib import Path
from typing import Optional
import mimetypes

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.celery_app import celery_app
from app.models.catalog import ModelDiarizationType, ModelTranscribeSize
from app.service.queue_tracker import enqueue_task, get_queue_position
from app.service.transcriber import transcriber_service
from app.settings import settings
from app.tasks.transcribe import process_transcription
from app.utils.export import ExportFormat

router = APIRouter(tags=["Транскрибация"])


DEFAULT_TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_SUPPRESS_TOKENS = [-1]


def _enqueue_transcription_task(
    audio_source: Path | None = None,
    source_filename: str | None = None,
    source_url: str | None = None,
    model: ModelTranscribeSize = "large",
    language: Optional[str] = None,
    task: str = "transcribe",
    log_progress: bool = False,
    beam_size: int = 3,
    best_of: int = 3,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    temperature: list[float] = DEFAULT_TEMPERATURE,
    compression_ratio_threshold: float | None = 2.4,
    log_prob_threshold: float | None = -1.0,
    no_speech_threshold: float | None = 0.6,
    condition_on_previous_text: bool = True,
    prompt_reset_on_temperature: float = 0.5,
    initial_prompt: str | None = None,
    prefix: str | None = None,
    suppress_blank: bool = True,
    suppress_tokens: list[int] = DEFAULT_SUPPRESS_TOKENS,
    without_timestamps: bool = False,
    max_initial_timestamp: float = 1.0,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    multilingual: bool = False,
    vad_filter: bool = False,
    vad_parameters: str | None = None,
    max_new_tokens: int | None = None,
    chunk_length: int | None = None,
    clip_timestamps: str = "0",
    hallucination_silence_threshold: float | None = None,
    hotwords: str | None = None,
    language_detection_threshold: float | None = 0.5,
    language_detection_segments: int = 1,
    diarization: bool = False,
    diarization_model: ModelDiarizationType = "pyannote_1",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    result_format: ExportFormat = "docx",
    export_timestamps: bool = False,
    save_source: bool = False,
    save_result: bool = True,
) -> dict:
    audio_path = str(audio_source) if audio_source else None
    queued_task = process_transcription.delay(
        audio_path=audio_path,
        source_filename=source_filename,
        source_url=source_url,
        model=model,
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
        suppress_tokens=suppress_tokens,
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
        diarization=diarization,
        diarization_model=diarization_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        result_format=result_format,
        export_timestamps=export_timestamps,
        save_source=save_source,
        save_result=save_result,
    )
    queue_position = enqueue_task(queued_task.id)

    return {
        "task_id": queued_task.id,
        "status": "queued",
        "queue_position": queue_position,
        "status_url": f"/transcribe/tasks/{queued_task.id}",
    }


@router.post("/transcribe/file/")
async def submit_transcription_file(
    file: UploadFile = File(...),
    model: ModelTranscribeSize = Query(
        "large",
        description="Размер модели Whisper: small, medium, large.",
    ),
    language: Optional[str] = Query(
        'ru',
        description="Код языка речи (например: ru, en). Если не задан, язык определяется автоматически.",
    ),
    task: str = Query(
        "transcribe",
        description="Задача: transcribe (транскрибация) или translate (перевод).",
    ),
    log_progress: bool = Query(
        False,
        description="Включить встроенный прогресс faster-whisper.",
    ),
    beam_size: int = Query(
        3,
        ge=1,
        le=20,
        description="Размер beam search.",
    ),
    best_of: int = Query(
        3,
        ge=1,
        le=20,
        description="Количество кандидатов при sampling (для ненулевой температуры).",
    ),
    patience: float = Query(
        1.0,
        ge=0.0,
        description="Коэффициент терпения для beam search.",
    ),
    length_penalty: float = Query(
        1.0,
        ge=0.0,
        description="Штраф за длину последовательности.",
    ),
    repetition_penalty: float = Query(
        1.1,
        ge=0.0,
        description="Штраф за повтор токенов (>1 усиливает штраф).",
    ),
    no_repeat_ngram_size: int = Query(
        0,
        ge=0,
        description="Запрет повторения n-грамм.",
    ),
    temperature: list[float] = Query(
        DEFAULT_TEMPERATURE,
        description="Температура декодирования.",
    ),
    compression_ratio_threshold: float | None = Query(
        2.4,
        description="Порог коэффициента сжатия gzip для детекции неудачного результата.",
    ),
    log_prob_threshold: float | None = Query(
        -1.0,
        description="Порог средней лог-вероятности токенов.",
    ),
    no_speech_threshold: float | None = Query(
        0.6,
        description="Порог вероятности отсутствия речи.",
    ),
    condition_on_previous_text: bool = Query(
        True,
        description="Использовать предыдущий текст как prompt для следующего окна.",
    ),
    prompt_reset_on_temperature: float = Query(
        0.5,
        description="Сбрасывать prompt при температуре выше этого порога.",
    ),
    initial_prompt: str | None = Query(
        None,
        description="Начальный prompt для первого окна.",
    ),
    prefix: str | None = Query(
        None,
        description="Текстовый префикс для первого окна.",
    ),
    suppress_blank: bool = Query(
        True,
        description="Подавлять пустые токены в начале генерации.",
    ),
    suppress_tokens: list[int] = Query(
        DEFAULT_SUPPRESS_TOKENS,
        description="Список ID токенов для подавления. -1 = стандартный список non-speech токенов.",
    ),
    without_timestamps: bool = Query(
        False,
        description="Генерировать только текст без таймкодов.",
    ),
    max_initial_timestamp: float = Query(
        1.0,
        ge=0.0,
        description="Максимальное значение начального таймкода.",
    ),
    word_timestamps: bool = Query(
        False,
        description="Извлекать таймкоды на уровне слов.",
    ),
    prepend_punctuations: str = Query(
        "\"'“¿([{-",
        description="Знаки пунктуации, присоединяемые к следующему слову при word_timestamps=True.",
    ),
    append_punctuations: str = Query(
        "\"'.。,，!！?？:：”)]}、",
        description="Знаки пунктуации, присоединяемые к предыдущему слову при word_timestamps=True.",
    ),
    multilingual: bool = Query(
        False,
        description="Определять язык для каждого сегмента.",
    ),
    vad_filter: bool = Query(
        False,
        description="Включить VAD (Silero) для удаления участков без речи.",
    ),
    vad_parameters: str | None = Query(
        None,
        description="Параметры VAD в JSON-строке (например: {\"min_silence_duration_ms\":500}).",
    ),
    max_new_tokens: int | None = Query(
        None,
        ge=1,
        description="Максимум новых токенов на сегмент.",
    ),
    chunk_length: int | None = Query(
        None,
        ge=1,
        description="Длина чанка аудио (сек), переопределяет настройку FeatureExtractor.",
    ),
    clip_timestamps: str = Query(
        "0",
        description="Список интервалов в секундах: start,end,start,end,...",
    ),
    hallucination_silence_threshold: float | None = Query(
        None,
        ge=0.0,
        description="Порог тишины для фильтрации возможных галлюцинаций при word_timestamps=True.",
    ),
    hotwords: str | None = Query(
        None,
        description="Ключевые слова/подсказки для модели.",
    ),
    language_detection_threshold: float | None = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Порог уверенности для определения языка.",
    ),
    language_detection_segments: int = Query(
        1,
        ge=1,
        description="Количество сегментов для определения языка.",
    ),
    diarization: bool = Query(
        False,
        description="Включить diarization (разметку спикеров).",
    ),
    diarization_model: ModelDiarizationType = Query(
        "pyannote_1",
        description="Модель diarization: pyannote_1 или pyannote_3_1.",
    ),
    num_speakers: int | None = Query(
        None,
        ge=1,
        description="Точное число спикеров (если известно заранее).",
    ),
    min_speakers: int | None = Query(
        None,
        ge=1,
        description="Минимальное число спикеров.",
    ),
    max_speakers: int | None = Query(
        None,
        ge=1,
        description="Максимальное число спикеров.",
    ),
    result_format: ExportFormat = Query(
        "docx",
        description="Формат экспортируемого файла результата.",
    ),
    export_timestamps: bool = Query(
        False,
        description="Добавлять таймкоды в экспортируемый файл результата.",
    ),
    save_source: bool = Query(
        False,
        description="Сохранить загруженный исходный файл.",
    ),
    save_result: bool = Query(
        True,
        description="Сохранить экспортированный файл результата.",
    ),
):
    filename = Path(file.filename or "uploaded_file")
    audio_source = await transcriber_service.prepare_audio_stream(
        file=file,
        filename=filename,
        save_source=save_source,
    )
    return _enqueue_transcription_task(
        audio_source=audio_source,
        source_filename=filename.name,
        model=model,
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
        suppress_tokens=suppress_tokens,
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
        diarization=diarization,
        diarization_model=diarization_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        result_format=result_format,
        export_timestamps=export_timestamps,
        save_source=save_source,
        save_result=save_result,
    )


@router.post("/transcribe/url/")
async def submit_transcription_url(
    source_url: str = Query(
        ...,
        description="Публичный URL медиа (YouTube, Rutube и т.д.)."
    ),
    model: ModelTranscribeSize = Query(
        "large",
        description="Размер модели Whisper: small, medium, large.",
    ),
    language: Optional[str] = Query(
        'ru',
        description="Код языка речи (например: ru, en). Если не задан, язык определяется автоматически.",
    ),
    task: str = Query(
        "transcribe",
        description="Задача: transcribe (транскрибация) или translate (перевод).",
    ),
    log_progress: bool = Query(
        False,
        description="Включить встроенный прогресс faster-whisper.",
    ),
    beam_size: int = Query(
        3,
        ge=1,
        le=20,
        description="Размер beam search.",
    ),
    best_of: int = Query(
        3,
        ge=1,
        le=20,
        description="Количество кандидатов при sampling (для ненулевой температуры).",
    ),
    patience: float = Query(
        1.0,
        ge=0.0,
        description="Коэффициент терпения для beam search.",
    ),
    length_penalty: float = Query(
        1.0,
        ge=0.0,
        description="Штраф за длину последовательности.",
    ),
    repetition_penalty: float = Query(
        1.1,
        ge=0.0,
        description="Штраф за повтор токенов (>1 усиливает штраф).",
    ),
    no_repeat_ngram_size: int = Query(
        0,
        ge=0,
        description="Запрет повторения n-грамм (0 = отключено).",
    ),
    temperature: list[float] = Query(
        DEFAULT_TEMPERATURE,
        description="Температура декодирования. Можно передавать несколько значений как fallback.",
    ),
    compression_ratio_threshold: float | None = Query(
        2.4,
        description="Порог коэффициента сжатия gzip для детекции неудачного результата.",
    ),
    log_prob_threshold: float | None = Query(
        -1.0,
        description="Порог средней лог-вероятности токенов.",
    ),
    no_speech_threshold: float | None = Query(
        0.6,
        description="Порог вероятности отсутствия речи.",
    ),
    condition_on_previous_text: bool = Query(
        True,
        description="Использовать предыдущий текст как prompt для следующего окна.",
    ),
    prompt_reset_on_temperature: float = Query(
        0.5,
        description="Сбрасывать prompt при температуре выше этого порога.",
    ),
    initial_prompt: str | None = Query(
        None,
        description="Начальный prompt для первого окна.",
    ),
    prefix: str | None = Query(
        None,
        description="Текстовый префикс для первого окна.",
    ),
    suppress_blank: bool = Query(
        True,
        description="Подавлять пустые токены в начале генерации.",
    ),
    suppress_tokens: list[int] = Query(
        DEFAULT_SUPPRESS_TOKENS,
        description="Список ID токенов для подавления. -1 = стандартный список non-speech токенов.",
    ),
    without_timestamps: bool = Query(
        False,
        description="Генерировать только текст без таймкодов.",
    ),
    max_initial_timestamp: float = Query(
        1.0,
        ge=0.0,
        description="Максимальное значение начального таймкода.",
    ),
    word_timestamps: bool = Query(
        False,
        description="Извлекать таймкоды на уровне слов.",
    ),
    prepend_punctuations: str = Query(
        "\"'“¿([{-",
        description="Знаки пунктуации, присоединяемые к следующему слову при word_timestamps=True.",
    ),
    append_punctuations: str = Query(
        "\"'.。,，!！?？:：”)]}、",
        description="Знаки пунктуации, присоединяемые к предыдущему слову при word_timestamps=True.",
    ),
    multilingual: bool = Query(
        False,
        description="Определять язык для каждого сегмента.",
    ),
    vad_filter: bool = Query(
        False,
        description="Включить VAD (Silero) для удаления участков без речи.",
    ),
    vad_parameters: str | None = Query(
        None,
        description="Параметры VAD в JSON-строке (например: {\"min_silence_duration_ms\":500}).",
    ),
    max_new_tokens: int | None = Query(
        None,
        ge=1,
        description="Максимум новых токенов на сегмент.",
    ),
    chunk_length: int | None = Query(
        None,
        ge=1,
        description="Длина чанка аудио (сек), переопределяет настройку FeatureExtractor.",
    ),
    clip_timestamps: str = Query(
        "0",
        description="Список интервалов в секундах: start,end,start,end,...",
    ),
    hallucination_silence_threshold: float | None = Query(
        None,
        ge=0.0,
        description="Порог тишины для фильтрации возможных галлюцинаций при word_timestamps=True.",
    ),
    hotwords: str | None = Query(
        None,
        description="Ключевые слова/подсказки для модели.",
    ),
    language_detection_threshold: float | None = Query(
        0.5,
        ge=0.0,
        le=1.0,
        description="Порог уверенности для определения языка.",
    ),
    language_detection_segments: int = Query(
        1,
        ge=1,
        description="Количество сегментов для определения языка.",
    ),
    diarization: bool = Query(
        False,
        description="Включить diarization (разметку спикеров).",
    ),
    diarization_model: ModelDiarizationType = Query(
        "pyannote_1",
        description="Модель diarization: pyannote_1 или pyannote_3_1.",
    ),
    num_speakers: int | None = Query(
        None,
        ge=1,
        description="Точное число спикеров (если известно заранее).",
    ),
    min_speakers: int | None = Query(
        None,
        ge=1,
        description="Минимальное число спикеров.",
    ),
    max_speakers: int | None = Query(
        None,
        ge=1,
        description="Максимальное число спикеров.",
    ),
    result_format: ExportFormat = Query(
        "docx",
        description="Формат экспортируемого файла результата.",
    ),
    export_timestamps: bool = Query(
        False,
        description="Добавлять таймкоды в экспортируемый файл результата.",
    ),
    save_source: bool = Query(
        False,
        description="Сохранить скачанный исходный файл.",
    ),
    save_result: bool = Query(
        True,
        description="Сохранить экспортированный файл результата.",
    ),
):
    return _enqueue_transcription_task(
        source_url=source_url,
        source_filename=None,
        model=model,
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
        suppress_tokens=suppress_tokens,
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
        diarization=diarization,
        diarization_model=diarization_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        result_format=result_format,
        export_timestamps=export_timestamps,
        save_source=save_source,
        save_result=save_result,
    )


@router.get("/transcribe/tasks/{task_id}")
def get_transcription_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    meta = result.info if isinstance(result.info, dict) else {}
    progress = meta.get("progress")
    progress_overall = meta.get("progress_overall", progress)
    progress_transcription = meta.get("progress_transcription")
    progress_diarization = meta.get("progress_diarization")
    media_duration_sec = meta.get("media_duration_sec")

    if result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "failed",
            "queue_position": None,
            "progress": progress_overall,
            "progress_overall": progress_overall,
            "progress_transcription": progress_transcription,
            "progress_diarization": progress_diarization,
            "media_duration_sec": media_duration_sec,
            "error": str(result.result),
        }

    if result.state == "SUCCESS":
        success_duration = None
        if isinstance(result.result, dict):
            raw_duration = result.result.get("media_duration_sec", result.result.get("duration"))
            if isinstance(raw_duration, (int, float)):
                success_duration = float(raw_duration)
        return {
            "task_id": task_id,
            "status": "done",
            "queue_position": None,
            "progress": 100.0,
            "progress_overall": 100.0,
            "progress_transcription": 100.0,
            "progress_diarization": (
                100.0 if isinstance(result.result, dict) and "diarization" in result.result else None
            ),
            "media_duration_sec": success_duration,
            "result": result.result,
        }

    status_map = {
        "PENDING": "queued",
        "PROGRESS": "processing",
        "STARTED": "processing",
        "RETRY": "retrying",
    }
    status = status_map.get(result.state, result.state.lower())
    queue_position = get_queue_position(task_id) if status == "queued" else None
    if status == "queued" and progress is None:
        progress = 0.0
    if status == "queued" and progress_overall is None:
        progress_overall = 0.0
    return {
        "task_id": task_id,
        "status": status,
        "queue_position": queue_position,
        "progress": progress_overall,
        "progress_overall": progress_overall,
        "progress_transcription": progress_transcription,
        "progress_diarization": progress_diarization,
        "media_duration_sec": media_duration_sec,
    }


@router.get("/transcribe/get_result/{filename:path}")
def download_transcription_file(filename: str):
    safe_name = Path(filename).name
    file_path = settings.TRANSCRIBE_RESULTS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(
        path=file_path,
        filename=safe_name,
        media_type=media_type or "application/octet-stream",
    )
