from fastapi import APIRouter, UploadFile, File
from app.models.model import ModelHandler

router = APIRouter()
model_handler = ModelHandler()

@router.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """Загружает аудиофайл, выполняет транскрипцию и возвращает текст."""
    audio_path = f"app/data/input/{file.filename}"

    with open(audio_path, "wb") as buffer:
        buffer.write(file.file.read())

    transcript = model_handler.transcribe(audio_path)
    
    return {"filename": file.filename, "transcription": transcript}

