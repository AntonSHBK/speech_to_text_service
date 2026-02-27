from pathlib import Path

from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.models.catalog import MODEL_CATALOG, ModelSize, resolve_model_name
from app.service.queue_tracker import mark_task_started
from app.service.transcriber import transcriber_service
from app.settings import settings
from app.utils.export import ExportFormat


@worker_process_init.connect
def init_transcriber_worker(**kwargs):
    for model_name in MODEL_CATALOG.values():
        transcriber_service.init(
            model_name=model_name,
            device=settings.DEVICE,
            cache_dir=settings.CACHE_DIR,
            token=None,
            compute_type="default",
            cpu_threads=settings.MODEL_CPU_THREADS,
            num_workers=settings.MODEL_NUM_WORKERS,
        )

@celery_app.task(name="transcribe.process")
def process_transcription(
    audio_path: str,
    source_filename: str,
    model: ModelSize = "small",
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
    input_path = Path(audio_path)
    result_file = None
    resolved_model_name = resolve_model_name(model)
    mark_task_started(process_transcription.request.id)
    process_transcription.update_state(state="PROGRESS", meta={"progress": 0.0})

    try:
        transcriber = transcriber_service.get_or_init(
            model_name=resolved_model_name,
            device=settings.DEVICE,
            cache_dir=settings.CACHE_DIR,
            token=None,
            compute_type="default",
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
        else:
            result["result_file"] = None

        return result
    finally:
        if not save_source and input_path.exists():
            input_path.unlink(missing_ok=True)
