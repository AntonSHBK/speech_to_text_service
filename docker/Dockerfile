# Используем официальный образ Debian
FROM debian:bullseye-slim

# Обновляем систему и устанавливаем зависимости для Python и необходимых библиотек
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    curl \
    ffmpeg \
    && apt-get clean

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . /app

# Устанавливаем зависимости Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Используем переменные окружения из .env
ENV PYTHONUNBUFFERED=1

# Открываем порт для API
EXPOSE 8000

# Запускаем FastAPI через Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
