from pathlib import Path
from urllib.parse import urlparse

from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from yt_dlp import YoutubeDL

from app.celery_app import celery_app
from app.models.catalog import ModelSize
from app.service.queue_tracker import enqueue_task, get_queue_position
from app.service.transcriber import transcriber_service
from app.settings import settings
from app.tasks.transcribe import process_transcription
from app.utils.export import ExportFormat

router = APIRouter(tags=["Транскрибация"])


def _enqueue_transcription_task(
    audio_source: Path,
    source_filename: str,
    model: ModelSize = Query("medium", description="Размер модели Whisper: small, medium, large."),
    language: str = Query("ru", description="Код языка транскрибации, например 'ru' или 'en'."),
    task: str = Query("transcribe", description="Тип задачи: 'transcribe' или 'translate'."),
    beam_size: int = Query(1, ge=1, le=10, description="Размер beam search."),
    chunk_length: int = Query(20, ge=5, le=60, description="Длина чанка в секундах."),
    patience: float = Query(1.0, ge=0.0, description="Параметр терпения декодирования."),
    length_penalty: float = Query(1.0, ge=0.0, description="Штраф за длину."),
    repetition_penalty: float = Query(1.0, ge=0.0, description="Штраф за повторы."),
    multilingual: bool = Query(False, description="Включить многоязычное декодирование."),
    result_format: ExportFormat = Query("docx", description="Формат экспортируемого файла результата."),
    save_source: bool = Query(False, description="Сохранить исходный файл."),
    save_result: bool = Query(True, description="Сохранить экспортированный файл результата."),
) -> dict:
    queued_task = process_transcription.delay(
        audio_path=str(audio_source),
        source_filename=source_filename,
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


def _download_source(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError("source_url должен использовать схему http или https")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "socket_timeout": settings.YTDLP_SOCKET_TIMEOUT_SEC,
        "outtmpl": str(settings.AUDIO_DIR / "%(title).120s_%(id)s.%(ext)s"),
    }
    with YoutubeDL(ydl_opts) as ydl:
        metadata = ydl.extract_info(url, download=False)
        if not metadata:
            raise RuntimeError("Не удалось получить метаданные источника")
        if "entries" in metadata and metadata["entries"]:
            metadata = metadata["entries"][0]

        duration = metadata.get("duration")
        if duration and duration > settings.YTDLP_MAX_DURATION_SEC:
            raise RuntimeError(
                f"Длительность источника превышает лимит: {int(duration)}с > {settings.YTDLP_MAX_DURATION_SEC}с"
            )

        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("Не удалось получить медиа по URL")

        if "entries" in info and info["entries"]:
            info = info["entries"][0]

        filepath = None
        if info.get("requested_downloads"):
            filepath = info["requested_downloads"][0].get("filepath")
        if not filepath:
            filepath = ydl.prepare_filename(info)

    path = Path(filepath)
    if not path.exists():
        raise RuntimeError("Скачанный файл не найден")
    return path


@router.post("/transcribe/file/")
async def submit_transcription_file(
    file: UploadFile = File(...),
    model: ModelSize = Query("medium", description="Размер модели Whisper: small, medium, large."),
    language: str = Query("ru", description="Код языка транскрибации, например 'ru' или 'en'."),
    task: str = Query("transcribe", description="Тип задачи: 'transcribe' или 'translate'."),
    beam_size: int = Query(3, ge=1, le=10, description="Размер beam search."),
    chunk_length: int = Query(20, ge=5, le=60, description="Длина чанка в секундах."),
    patience: float = Query(1.0, ge=0.0, description="Параметр терпения декодирования."),
    length_penalty: float = Query(1.0, ge=0.0, description="Штраф за длину."),
    repetition_penalty: float = Query(1.5, ge=0.0, description="Штраф за повторы."),
    multilingual: bool = Query(False, description="Включить многоязычное декодирование."),
    result_format: ExportFormat = Query("docx", description="Формат экспортируемого файла результата."),
    save_source: bool = Query(False, description="Сохранить загруженный исходный файл."),
    save_result: bool = Query(True, description="Сохранить экспортированный файл результата."),
):
    raw_bytes = await file.read()
    filename = Path(file.filename or "uploaded_file")
    audio_source = transcriber_service.prepare_audio(raw_bytes=raw_bytes, filename=filename, save_source=save_source)
    return _enqueue_transcription_task(
        audio_source=audio_source,
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
        save_source=save_source,
        save_result=save_result,
    )


@router.post("/transcribe/url/")
async def submit_transcription_url(
    source_url: str = Query(..., description="Публичный URL медиа (YouTube, Rutube и т.д.)."),
    model: ModelSize = Query("medium", description="Размер модели Whisper: small, medium, large."),
    language: str = Query("ru", description="Код языка транскрибации, например 'ru' или 'en'."),
    task: str = Query("transcribe", description="Тип задачи: 'transcribe' или 'translate'."),
    beam_size: int = Query(3, ge=1, le=10, description="Размер beam search."),
    chunk_length: int = Query(20, ge=5, le=60, description="Длина чанка в секундах."),
    patience: float = Query(1.0, ge=0.0, description="Параметр терпения декодирования."),
    length_penalty: float = Query(1.0, ge=0.0, description="Штраф за длину."),
    repetition_penalty: float = Query(1.5, ge=0.0, description="Штраф за повторы."),
    multilingual: bool = Query(False, description="Включить многоязычное декодирование."),
    result_format: ExportFormat = Query("docx", description="Формат экспортируемого файла результата."),
    save_source: bool = Query(False, description="Сохранить скачанный исходный файл."),
    save_result: bool = Query(True, description="Сохранить экспортированный файл результата."),
):
    try:
        audio_source = await run_in_threadpool(_download_source, source_url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось скачать source_url: {exc}") from exc

    return _enqueue_transcription_task(
        audio_source=audio_source,
        source_filename=audio_source.name,
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
        save_source=save_source,
        save_result=save_result,
    )


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


@router.get("/transcribe/files/{filename}")
def download_transcription_file(filename: str):
    safe_name = Path(filename).name
    file_path = settings.TRANSCRIBE_RESULTS_DIR / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(path=file_path, filename=safe_name)
