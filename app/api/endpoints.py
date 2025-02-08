from fastapi import APIRouter, HTTPException
from app.translater.translate_service import TranslateService
from app.api.models import TranslationRequest, TranslationResponse, BatchTranslationResponse

router = APIRouter()

# Инициализация сервиса перевода
translate_service = TranslateService()

@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    """
    Переводит текст на несколько языков.
    """
    try:
        # Перевод текста на несколько языков
        translated_texts = translate_service.translate_batch(request.text, request.source_lang, request.target_langs)
        return {"translations": translated_texts}  # Возвращаем переведенные тексты
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_translate", response_model=BatchTranslationResponse)
async def batch_translate(request: TranslationRequest):
    """
    Переводит несколько текстов на несколько языков.
    """
    try:
        # Переводим несколько текстов
        translated_batch = translate_service.translate_batch(request.text, request.source_lang, request.target_langs)
        return {"translations": translated_batch}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
