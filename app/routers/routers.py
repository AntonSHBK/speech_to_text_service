from fastapi import APIRouter, UploadFile, File, Query, Depends
from typing import Optional, List
from app.models.model_handler import model_handler

router = APIRouter()

@router.post("/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query("ru", description="Язык транскрипции (например, 'ru', 'en')"),
    task: str = Query("transcribe", description="Тип задачи ('transcribe' или 'translate')"),
    temperature: float = Query(None, description="Контроль случайности (чем выше, тем больше вариативность)"),
    max_new_tokens: Optional[int] = Query(None, description="Максимальное количество новых токенов"),
    repetition_penalty: Optional[float] = Query(None, description="Штраф за повторения (рекомендуется 1.1-1.5)"),
    length_penalty: Optional[float] = Query(None, description="Регулировка длины выхода"),
    num_beams: Optional[int] = Query(None, description="Количество 'лучей' в beam search (оптимально 3-5)"),
    early_stopping: Optional[bool] = Query(None, description="Останавливать генерацию при eos_token"),
    suppress_tokens: Optional[List[int]] = Query(None, description="Запрещенные токены (список ID)"),
    top_k: Optional[int] = Query(None, description="Выбор следующего слова из k наиболее вероятных"),
    top_p: Optional[float] = Query(None, description="Ограничение вероятности выбора следующего слова"),
    do_sample: Optional[bool] = Query(None, description="Использовать случайное сэмплирование"),
    model_handler=Depends(lambda: model_handler)  # Используем глобальную модель
):
    """
    Загружает аудиофайл, выполняет транскрипцию и возвращает текст.
    Поддерживает гибкую настройку параметров модели.

    **Допустимые параметры:**
    - **`temperature`** (float): Чем выше, тем более случайные результаты (`0.0` - строгое соответствие, `>0.0` - больше случайности).
    - **`max_new_tokens`** (int): Ограничивает количество новых токенов в выходе (рекомендуется `100-500`).
    - **`repetition_penalty`** (float): Чем выше, тем меньше повторов (оптимально `1.1-1.5`).
    - **`length_penalty`** (float): `>1.0` увеличивает длину транскрипции, `<1.0` - сокращает.
    - **`num_beams`** (int): Использует beam search для поиска наилучшего варианта (`3-5` дают лучший результат).
    - **`early_stopping`** (bool): Останавливает генерацию при достижении конца предложения.
    - **`suppress_tokens`** (list): Запрещенные токены (например, `[50257]` исключает английские слова).
    - **`top_k`** (int): Выбирает следующее слово из `k` наиболее вероятных.
    - **`top_p`** (float): Ограничивает выбор слов на основе вероятности (`top_p=0.95` - хорошая настройка).
    - **`do_sample`** (bool): Включает случайное сэмплирование (больше разнообразия в тексте).

    **Пример запроса:**
    ```
    curl -X 'POST' \
      'http://127.0.0.1:8000/transcribe/?temperature=0.7&max_new_tokens=200' \
      -F 'file=@sample.wav'
    ```
    """
    audio_path = f"app/data/input/{file.filename}"

    # Сохраняем файл
    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())

    # Собираем параметры в kwargs
    kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "num_beams": num_beams,
        "early_stopping": early_stopping,
        "suppress_tokens": suppress_tokens,
        "top_k": top_k,
        "top_p": top_p,
        "do_sample": do_sample,
    }

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Выполняем транскрипцию
    transcript = await model_handler.transcribe(audio_path, language=language, task=task, **kwargs)

    return {
        "filename": file.filename,
        "transcription": transcript,
        "parameters": kwargs
    }
