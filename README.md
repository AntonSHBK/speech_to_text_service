# Speech-to-Text API (Whisper)

**Speech-to-Text API** is a production-ready microservice for automatic speech recognition built with **FastAPI** and **OpenAI Whisper** (via **faster-whisper**).
The service provides high-quality transcription for audio and video files with support for **multilingual speech recognition**, optimized for CPU inference.

## Features

* Audio transcription (WAV, MP3, FLAC, AAC, OGG, M4A, etc.)
* Audio extraction from video files (MP4, MKV, AVI)
* Multilingual transcription (99+ languages supported by Whisper)
* Optional translation to English
* High-performance inference using **faster-whisper**
* Clean service architecture: models / service / routers separation
* Asynchronous FastAPI interface
* Persistent storage for source files and transcription results
* DOCX export with formatted transcription output
* Docker-ready deployment

## Supported Languages

Whisper supports more than **99 languages**, including but not limited to:

| Language | Code | Language   | Code |
| -------- | ---- | ---------- | ---- |
| English  | en   | Spanish    | es   |
| Russian  | ru   | Portuguese | pt   |
| Chinese  | zh   | Arabic     | ar   |
| French   | fr   | Japanese   | ja   |
| German   | de   | Hindi      | hi   |

The full language list is available in the official Whisper repository.

## Architecture Overview

```
app/
 ├─ models/        # Model definitions (no runtime objects)
 ├─ service/       # Runtime services and business logic
 ├─ routers/       # HTTP API endpoints
 ├─ settings.py    # Global configuration (Pydantic Settings)
 ├─ main.py        # Application lifecycle & startup
```

* **models**: Pure class definitions for transcription engines
* **service**: Holds live model instances and processing pipeline
* **routers**: API layer (FastAPI endpoints)
* **main**: Controls application startup and lifecycle via lifespan

## Installation & Setup

### Run Locally

### Requirements

**Python 3.10+, FFmpeg**

FFmpeg is required for audio processing.

You can download FFmpeg from the official website: 
[https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

#### Windows

For Windows users, download the **static build** of FFmpeg (ZIP archive), extract it, and add the path to the `bin` directory (where `ffmpeg.exe` is located) to your system **PATH** environment variable.

After that, make sure FFmpeg is accessible from the command line:

```bash
ffmpeg -version
```

If the command works, FFmpeg is correctly installed.

### Next

```bash
git clone https://github.com/AntonSHBK/speech_to_text_service
cd speech_to_text_service

pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API documentation will be available at:

```
http://127.0.0.1:8000/docs
```

### Docker Deployment

```bash
docker-compose build
docker-compose up -d
```

The API will be available at:

```
http://localhost:8000/docs
```

### Optional Web Interface (Local UI)

The service also includes a lightweight built-in web interface for local usage and testing.
It allows uploading audio files, adjusting transcription parameters and saving the resulting text without using the API client.

To enable the interface, set the following environment variable:

```env
USE_INTERFACE=true
```

Then start the server as usual.

The web interface will be available at:

```
http://127.0.0.1:8000/ui
```

You can use this page as a simple local application for interactive transcription.

## API Usage

### Health Check

```http
GET /
```

Response:

```json
{ "status": "API is running" }
```

### Transcription Endpoint

```http
POST /transcribe/
```

#### Parameters

| Name               | Type   | Description                 |
| ------------------ | ------ | --------------------------- |
| file               | file   | Audio or video file         |
| language           | string | Language code (default: ru) |
| task               | string | transcribe or translate     |
| beam_size          | int    | Beam search size            |
| chunk_length       | int    | Chunk length in seconds     |
| patience           | float  | Decoding patience           |
| length_penalty     | float  | Length penalty              |
| repetition_penalty | float  | Repetition penalty          |
| save_file          | bool   | Save uploaded audio         |
| save_result        | bool   | Save DOCX transcription     |

#### Example

```bash
curl -X POST http://127.0.0.1:8000/transcribe/ \
  -F "file=@audio.mp3" \
  -F "language=ru" \
  -F "result_format=docx" \
  -F "save_file=true" \
  -F "save_result=true"
```

#### Example Response

```json
{
  "language": "ru",
  "duration": 42.8,
  "text": "Full transcription text",
  "segments": [
    { "start": 0.0, "end": 4.2, "text": "Segment text" }
  ],
  "result_file": "data/transcriptions/audio_20240110_123045.docx"
}
```

## Support & Contributions
If you have ideas for improvement or found a bug, create an **Issue** or **Pull Request** in the [project repository](https://github.com/AntonSHBK/speech_to_text_service).

**Author:** [Anton Pisarenko](https://github.com/AntonSHBK)  
**License:** Apache License 2.0  