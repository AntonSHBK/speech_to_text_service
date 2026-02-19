from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile

from app.celery_app import celery_app
from app.models.catalog import ModelSize
from app.service.queue_tracker import enqueue_task, get_queue_position
from app.service.transcriber import transcriber_service
from app.tasks.transcribe import process_transcription
from app.utils.export import ExportFormat

router = APIRouter(tags=["Transcription"])


@router.post("/transcribe/")
async def submit_transcription(
    file: UploadFile = File(...),
    model: ModelSize = Query("small", description="Whisper model size: small, medium, large."),
    language: str = Query("ru", description="Transcription language code, for example 'ru' or 'en'."),
    task: str = Query("transcribe", description="Task type: 'transcribe' or 'translate'."),
    beam_size: int = Query(1, ge=1, le=10, description="Beam search size."),
    chunk_length: int = Query(20, ge=5, le=60, description="Chunk length in seconds."),
    patience: float = Query(1.0, ge=0.0, description="Decoding patience."),
    length_penalty: float = Query(1.0, ge=0.0, description="Length penalty."),
    repetition_penalty: float = Query(1.0, ge=0.0, description="Repetition penalty."),
    multilingual: bool = Query(False, description="Enable multilingual decoding."),
    result_format: ExportFormat = Query("docx", description="Exported result file format."),
    save_file: bool = Query(False, description="Keep uploaded source file."),
    save_result: bool = Query(True, description="Keep exported result file."),
):
    raw_bytes = await file.read()
    filename = Path(file.filename or "uploaded_file")

    audio_source = transcriber_service.prepare_audio(
        raw_bytes=raw_bytes,
        filename=filename,
        save_file=True,
    )

    queued_task = process_transcription.delay(
        audio_path=str(audio_source),
        source_filename=filename.name,
        model=model,
        language=language,
        task=task,
        beam_size=beam_size,
        chunk_length=chunk_length,
        patience=patience,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        multilingual=multilingual,
        result_format=result_format,
        save_result=save_result,
        remove_source_after=not save_file,
    )
    queue_position = enqueue_task(queued_task.id)

    return {
        "task_id": queued_task.id,
        "status": "queued",
        "queue_position": queue_position,
        "status_url": f"/transcribe/tasks/{queued_task.id}",
    }


@router.get("/transcribe/tasks/{task_id}")
def get_transcription_status(task_id: str):
    result = celery_app.AsyncResult(task_id)
    meta = result.info if isinstance(result.info, dict) else {}
    progress = meta.get("progress")

    if result.state == "FAILURE":
        return {
            "task_id": task_id,
            "status": "failed",
            "queue_position": None,
            "progress": progress,
            "error": str(result.result),
        }

    if result.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "done",
            "queue_position": None,
            "progress": 100.0,
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
    return {
        "task_id": task_id,
        "status": status,
        "queue_position": queue_position,
        "progress": progress,
    }
