from pathlib import Path
import json

from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.models.catalog import ModelSize, resolve_model_name
from app.service.source_downloader import download_source
from app.service.queue_tracker import mark_task_started
from app.service.transcriber import transcriber_service
from app.settings import settings
from app.utils.export import ExportFormat
from app.utils.logging import get_logger

logger = get_logger("worker.init")


@worker_process_init.connect
def init_transcriber_worker(**kwargs):
    default_model_key: ModelSize = "medium"
    default_model_name = resolve_model_name(default_model_key)
    compute_type = settings.get_model_compute_type(default_model_key)
    logger.info("Начинается инициализация модели по умолчанию: %s", default_model_name)
    transcriber_service.init(
        model_name=default_model_name,
        device=settings.DEVICE,
        cache_dir=settings.CACHE_DIR,
        token=settings.HF_TOKEN,
        compute_type=compute_type,
        cpu_threads=settings.MODEL_CPU_THREADS,
        num_workers=settings.MODEL_NUM_WORKERS,
    )
    logger.info("Инициализация модели по умолчанию завершена: %s", default_model_name)
    logger.info("Другие модели будут загружаться по требованию задачи.")


@celery_app.task(name="transcribe.process")
def process_transcription(
    audio_path: str | None = None,
    source_filename: str | None = None,
    source_url: str | None = None,
    model: ModelSize = "medium",
    language: str | None = None,
    task: str = "transcribe",
    log_progress: bool = False,
    beam_size: int = 3,
    best_of: int = 3,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    temperature: list[float] | float = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: float | None = 2.4,
    log_prob_threshold: float | None = -1.0,
    no_speech_threshold: float | None = 0.6,
    condition_on_previous_text: bool = True,
    prompt_reset_on_temperature: float = 0.5,
    initial_prompt: str | None = None,
    prefix: str | None = None,
    suppress_blank: bool = True,
    suppress_tokens: list[int] | None = None,
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
    result_format: ExportFormat = "docx",
    save_source: bool = False,
    save_result: bool = True,
) -> dict:
    input_path = Path(audio_path) if audio_path else None
    result_file = None
    compute_type = settings.get_model_compute_type(model)
    resolved_model_name = resolve_model_name(model)
    mark_task_started(process_transcription.request.id)
    process_transcription.update_state(state="PROGRESS", meta={"progress": 0.0})

    parsed_vad_parameters = None
    if vad_parameters:
        try:
            parsed_vad_parameters = json.loads(vad_parameters)
        except json.JSONDecodeError as exc:
            raise ValueError("Некорректный JSON в vad_parameters") from exc

    try:
        if source_url:
            logger.info("Скачивание source_url начато: %s", source_url)
            input_path = download_source(source_url)
            logger.info("Скачивание source_url завершено: %s", input_path)

        if not input_path:
            raise ValueError("Не передан источник аудио: audio_path или source_url")

        source_filename = source_filename or input_path.name

        transcriber = transcriber_service.get_or_init(
            model_name=resolved_model_name,
            device=settings.DEVICE,
            cache_dir=settings.CACHE_DIR,
            token=settings.HF_TOKEN,
            compute_type=compute_type,
            cpu_threads=settings.MODEL_CPU_THREADS,
            num_workers=settings.MODEL_NUM_WORKERS,
        )

        result = transcriber.transcribe(
            input_path,
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
            vad_parameters=parsed_vad_parameters,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
            language_detection_threshold=language_detection_threshold,
            language_detection_segments=language_detection_segments,
            on_progress=lambda progress: process_transcription.update_state(
                state="PROGRESS",
                meta={"progress": round(progress, 1)},
            ),
        )

        if save_result:
            result_file = transcriber_service.export_result(
                result=result,
                source_filename=source_filename,
                format=result_format,
            )
            result["result_file"] = str(result_file)
            result["result_filename"] = result_file.name
        else:
            result["result_file"] = None
            result["result_filename"] = None

        return result
    finally:
        if input_path and not save_source and input_path.exists():
            input_path.unlink(missing_ok=True)
