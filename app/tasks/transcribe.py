from pathlib import Path

from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.models.catalog import MODEL_CATALOG, ModelSize, resolve_model_name
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
    logger.info("Начинается инициализация модели по умолчанию: %s", default_model_name)
    transcriber_service.init(
        model_name=default_model_name,
        device=settings.DEVICE,
        cache_dir=settings.CACHE_DIR,
        token=None,
        compute_type=settings.MODEL_COMPUTE_TYPE,
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
    language: str = "ru",
    task: str = "transcribe",
    beam_size: int = 1,
    chunk_length: int = 20,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    multilingual: bool = False,
    result_format: ExportFormat = "docx",
    save_source: bool = False,
    save_result: bool = True,
) -> dict:
    input_path = Path(audio_path) if audio_path else None
    result_file = None
    resolved_model_name = resolve_model_name(model)
    mark_task_started(process_transcription.request.id)
    process_transcription.update_state(state="PROGRESS", meta={"progress": 0.0})

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
            token=None,
            compute_type=settings.MODEL_COMPUTE_TYPE,
            cpu_threads=settings.MODEL_CPU_THREADS,
            num_workers=settings.MODEL_NUM_WORKERS,
        )

        result = transcriber.transcribe(
            input_path,
            language=language,
            task=task,
            beam_size=beam_size,
            chunk_length=chunk_length,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            multilingual=multilingual,
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
