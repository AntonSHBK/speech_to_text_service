from pathlib import Path
from typing import Optional
import mimetypes

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.celery_app import celery_app
from app.models.catalog import ModelTranscribeSize
from app.service.queue_tracker import enqueue_task, get_queue_position
from app.service.transcriber import transcriber_service
from app.settings import settings
from app.tasks.transcribe import process_transcription
from app.utils.export import ExportFormat

router = APIRouter(tags=["Транскрибация"])


def _normalize_optional_language(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = language.strip().lower()
    if normalized in {"", "null", "none"}:
        return None
    return normalized


def _enqueue_transcription_task(
    audio_source: Path | None = None,
    source_filename: str | None = None,
    source_url: str | None = None,
    model: ModelTranscribeSize = "large",
    language: Optional[str] = None,
    diarization: bool = False,
    num_speakers: int | None = None,
    result_format: ExportFormat = "docx",
    export_timestamps: bool = False,
    save_source: bool = False,
    save_result: bool = True,
) -> dict:
    audio_path = str(audio_source) if audio_source else None
    language = _normalize_optional_language(language)
    queued_task = process_transcription.delay(
        audio_path=audio_path,
        source_filename=source_filename,
        source_url=source_url,
        model=model,
        language=language,
        diarization=diarization,
        num_speakers=num_speakers,
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
        None,
        description="Код языка речи (например: ru, en). Если не задан, язык определяется автоматически.",
    ),
    diarization: bool = Query(
        False,
        description="Включить diarization (разметку спикеров).",
    ),
    num_speakers: int | None = Query(
        None,
        ge=1,
        description="Exact number of speakers for Sherpa-ONNX, if known.",
    ),
    result_format: ExportFormat = Query(
        "docx",
        description=(
            "Формат экспортируемого файла результата: docx, txt, md, pdf, "
            "srt, vtt или ass."
        ),
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
        diarization=diarization,
        num_speakers=num_speakers,
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
        None,
        description="Код языка речи (например: ru, en). Если не задан, язык определяется автоматически.",
    ),
    diarization: bool = Query(
        False,
        description="Включить diarization (разметку спикеров).",
    ),
    num_speakers: int | None = Query(
        None,
        ge=1,
        description="Exact number of speakers for Sherpa-ONNX, if known.",
    ),
    result_format: ExportFormat = Query(
        "docx",
        description=(
            "Формат экспортируемого файла результата: docx, txt, md, pdf, "
            "srt, vtt или ass."
        ),
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
        diarization=diarization,
        num_speakers=num_speakers,
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
