from pydantic import BaseModel
from typing import List, Dict

class TranslationRequest(BaseModel):
    text: List[str]  # Список текстов для перевода
    source_lang: str  # Исходный язык
    target_langs: List[str]  # Список целевых языков

class TranslationResponse(BaseModel):
    translations: List[Dict[str, List[str]]]  # Список переведенных текстов с целевыми языками

class BatchTranslationResponse(BaseModel):
    translations: List[List[Dict[str, List[str]]]]  # Список переведенных текстов для каждого входного текста
