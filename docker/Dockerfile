# Используем официальный образ Debian
FROM debian:bullseye-slim

# Обновляем систему и устанавливаем зависимости для Python и других необходимых библиотек
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    curl \
    && apt-get clean

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем все файлы в контейнер
COPY . /app

# Устанавливаем зависимости из requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Запускаем приложение с помощью Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
