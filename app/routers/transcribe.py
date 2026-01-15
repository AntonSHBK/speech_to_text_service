import json
from pathlib import Path

from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText


from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from fastapi import APIRouter, UploadFile, File, Query, HTTPException

from app.settings import settings
from app.utils.export import ExportFormat
from app.service.transcriber import transcriber_service

router = APIRouter(tags=["Transcription"])

@router.post("/transcribe/")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query("ru", description="Язык транскрипции (например, 'ru', 'en')"),
    task: str = Query("transcribe", description="Тип задачи ('transcribe' или 'translate')"),
    beam_size: int = Query(1, ge=1, le=10, description="Размер beam search"),
    chunk_length: int = Query(20, ge=5, le=60, description="Длина чанка в секундах"),
    patience: float = Query(1.0, ge=0.0, description="Patience"),
    length_penalty: float = Query(1.0, ge=0.0, description="Length penalty"),
    repetition_penalty: float = Query(1.0, ge=0.0, description="Repetition penalty"),
    multilingual: bool = Query(False, description="Поддержка нескольких языков"),
    result_format: ExportFormat = Query("docx", description="Формат экспортированного файла"),
    save_file: bool = Query(False, description="Сохранять ли исходный файл"),
    save_result: bool = Query(False, description="Сохранять ли результат транскрипции"),
):
    if not transcriber_service.is_ready():
        raise HTTPException(503, "Transcription service not ready")

    raw_bytes = await file.read()
    filename = Path(file.filename)

    audio_source = await transcriber_service.prepare_audio(
        raw_bytes=raw_bytes,
        filename=filename,
        save_file=save_file
    )

    try:    
        result = await run_in_threadpool(
            transcriber_service.get().transcribe,
            audio_source,
            language=language,
            task=task,
            beam_size=beam_size,
            chunk_length=chunk_length,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            multilingual=multilingual
        )
        
        result_file = await transcriber_service.export_result(
            result=result, 
            source_filename=filename, 
            format=result_format
        )
        
        result["result_file"] = str(result_file)

        # json_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
        # file_bytes = result_file.read_bytes()
    
    finally:
        if not save_file and audio_source.exists():
            audio_source.unlink(missing_ok=True)

        # if not save_result and 'result_file' in locals():
        #     result_file.unlink(missing_ok=True)

    # msg = MIMEMultipart("mixed")

    # json_part = MIMEText(
    #     _text=json_bytes.decode("utf-8"),
    #     _subtype="json",
    #     _charset="utf-8"
    # )
    # msg.attach(json_part)

    # file_part = MIMEApplication(file_bytes)
    # file_part.add_header(
    #     "Content-Disposition",
    #     "attachment",
    #     filename=result_file.name
    # )
    # msg.attach(file_part)

    # return Response(
    #     content=msg.as_bytes(),
    #     media_type=msg.get("Content-Type")
    # )
    
    return result

