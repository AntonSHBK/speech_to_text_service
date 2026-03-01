### **📌 `RUN.md` – Руководство по запуску Speech-to-Text API**

Этот файл содержит инструкции по **развёртыванию и запуску** сервиса **Speech-to-Text API** с использованием **FastAPI** и **Whisper (OpenAI)**.

---

## **🔹 1. Запуск без Docker (локально)**

### **1️⃣ Установка зависимостей**
> **Требования:**  
> - **Python 3.9+**  
> - **FFmpeg** (необходим для обработки аудио)  

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/your-repo/speech-to-text.git
   cd speech-to-text
   ```

2. **Создайте виртуальное окружение (рекомендуется)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate     # Windows
   ```

3. **Установите зависимости**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### **2️⃣ Запуск сервера**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Теперь API будет доступно по адресу:  
📜 **http://127.0.0.1:8000/docs** (Swagger UI)

---

## **🔹 2. Запуск через Docker**
> **Требования:**  
> - **Docker 20+**  
> - **Docker Compose**

1. **Соберите Docker-образ**:
   ```bash
   docker-compose build
   ```

2. **Запустите контейнер в фоновом режиме**:
   ```bash
   docker-compose up -d
   ```

3. **Проверка работы**:
   ```bash
   docker ps  # Проверяем, запущен ли контейнер
   docker logs -f speech-to-text-service  # Просмотр логов
   ```

API доступно по **http://localhost:8000/docs** 🚀

---

## **🔹 3. Использование API**
### **1️⃣ Транскрипция аудиофайла**
📌 **POST `/transcribe/`** – Отправка аудиофайла:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -F 'file=@audio.mp3'
```
🔹 **Ответ:**
```json
{
  "filename": "audio.mp3",
  "transcription": "Пример текста транскрипции"
}
```

---

### **2️⃣ Транскрипция с переводом в английский**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -F 'file=@audio.mp3' \
  -F 'task=translate'
```
🔹 **Ответ:**
```json
{
  "filename": "audio.mp3",
  "transcription": "This is an example transcription in English"
}
```

---

## **🔹 4. Остановка сервиса**
### **Остановка локального запуска**
```bash
CTRL + C  # Прервать процесс uvicorn
```

### **Остановка Docker-контейнера**
```bash
docker-compose down
```

---

## **🔹 5. Решение проблем**
### **1️⃣ Ошибка: `ffmpeg not found`**
> FFmpeg не установлен. Установите вручную:
```bash
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS
choco install ffmpeg     # Windows (через Chocolatey)
```

### **2️⃣ Ошибка: `torch.cuda.is_available() == False`**
> PyTorch не видит GPU. Проверьте установку CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Если вывод `False`, установите PyTorch с поддержкой GPU:  
👉 [Руководство по установке](https://pytorch.org/get-started/locally/)


```bash
docker compose -f docker/docker-compose.yml build
```