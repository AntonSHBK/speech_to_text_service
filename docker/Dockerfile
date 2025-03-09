FROM debian:bullseye-slim

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-pip \
    python3-dev \
    curl \
    ffmpeg \
    && apt-get clean

WORKDIR /app

COPY . /app

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]
