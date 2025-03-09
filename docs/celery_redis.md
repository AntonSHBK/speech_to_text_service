### **📌 Настройка `Celery + Redis` для фоновой обработки транскрипции**
Сейчас FastAPI **обрабатывает аудио синхронно**, что может **замедлять API**.  
Мы **перенесем обработку в фон** с помощью **Celery + Redis**. 🚀  

---

## **🔹 Что мы сделаем?**
✅ **API принимает аудиофайл и отправляет задачу Celery**.  
✅ **Celery в фоне выполняет транскрипцию** (без блокировки API).  
✅ **Когда транскрипция готова, результат сохраняется в БД / файле / возвращается через API**.  
✅ **Можно параллельно запускать много задач без перегрузки сервера**.

---

## **📌 1. Установим зависимости**
```bash
pip install celery redis fastapi uvicorn
```
Если **Redis еще не установлен**, можно запустить его через Docker:
```bash
docker run -d --name redis -p 6379:6379 redis
```
---

## **📌 2. Настроим Celery в `celery_worker.py`**
Создадим **новый файл** `app/celery_worker.py`, который будет **запускать фоновые задачи**.

📌 **Добавляем Celery и подключаем Redis**
```python
from celery import Celery
from app.models.model_handler import ModelHandler
from app.config import CONFIG

# Настраиваем Celery с Redis
celery_app = Celery(
    "transcription_tasks",
    broker="redis://localhost:6379/0",  # Redis для очереди задач
    backend="redis://localhost:6379/0"  # Redis для хранения результатов
)

# Загружаем модель один раз (для всех задач)
model_handler = ModelHandler(
    model_name=CONFIG["model_name"],
    cache_dir=CONFIG["cache_dir"],
    device=CONFIG["device"],
    language=CONFIG["default_language"],
    temperature=CONFIG["temperature"]
)

@celery_app.task
def transcribe_task(audio_path: str, language: str = "ru", task: str = "transcribe"):
    """
    Фоновая задача для транскрипции аудиофайла.
    """
    transcription = model_handler.transcribe(audio_path, language=language, task=task)
    
    # Можно сохранить результат в базу данных или Redis
    return {"filename": audio_path, "transcription": transcription}
```

✅ **Почему это важно?**
- Celery **запускает транскрипцию в фоне**, освобождая API.  
- **Модель загружается один раз**, и все задачи используют ее.  
- **Redis хранит результаты**, их можно получать через API.

---

## **📌 3. Обновляем `routers.py`**
Теперь API **отправляет запрос в Celery и сразу возвращает task_id**, а результат можно запрашивать позже.

📌 **Обновляем эндпоинты в `routers.py`**
```python
from fastapi import APIRouter, UploadFile, File
from app.celery_worker import transcribe_task

router = APIRouter()

@router.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), language: str = "ru", task: str = "transcribe"):
    """
    Загружает аудиофайл, отправляет в очередь Celery и возвращает task_id.
    """
    audio_path = f"app/data/input/{file.filename}"

    # Сохраняем файл
    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())

    # Отправляем задачу в Celery
    task = transcribe_task.delay(audio_path, language, task)

    return {"task_id": task.id, "message": "Transcription is in progress"}

@router.get("/transcribe/{task_id}")
async def get_transcription_result(task_id: str):
    """
    Получает результат транскрипции по task_id.
    """
    result = transcribe_task.AsyncResult(task_id)
    
    if result.ready():
        return result.result  # Возвращаем результат
    return {"task_id": task_id, "status": result.status}
```

✅ **Что улучшилось?**
- Запрос **сразу получает `task_id`**, а транскрипция выполняется в фоне.  
- **Можно запрашивать статус задачи** (`PENDING`, `STARTED`, `SUCCESS`).  
- **Нет блокировки API** → сервер принимает новые запросы, пока Celery работает.  

---

## **📌 4. Запускаем сервер и Celery**
Теперь **запускаем все процессы**:

1️⃣ **Запускаем FastAPI сервер**
```bash
uvicorn app.main:app --reload
```

2️⃣ **Запускаем Celery воркер**
```bash
celery -A app.celery_worker.celery_app worker --loglevel=info
```

---

## **📌 5. Тестируем API**
### **1. Отправляем аудиофайл**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -F 'file=@sample.wav'
```
**Ответ:**
```json
{"task_id": "e82b0d92-a7bc-4c80-888f-24d7b08b824d", "message": "Transcription is in progress"}
```

### **2. Запрашиваем статус задачи**
```bash
curl -X 'GET' 'http://127.0.0.1:8000/transcribe/e82b0d92-a7bc-4c80-888f-24d7b08b824d'
```
**Ответ, если обработка еще идет:**
```json
{"task_id": "e82b0d92-a7bc-4c80-888f-24d7b08b824d", "status": "PENDING"}
```
**Ответ, если транскрипция завершена:**
```json
{
    "filename": "app/data/input/sample.wav",
    "transcription": "Привет, это тестовый аудиофайл."
}
```

---

## **📌 Итог**
✅ **API больше не блокируется при транскрипции**.  
✅ **Можно обрабатывать много файлов одновременно**.  
✅ **Celery + Redis делают систему масштабируемой**.  
✅ **Легко добавить новые фоновые задачи (например, перевод)**.  

🚀 **Готово! Теперь API поддерживает высокую нагрузку!**  
❓ **Дальше тестируем или улучшаем API?** 😊