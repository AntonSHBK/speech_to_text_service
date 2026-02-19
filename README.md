# Speech-to-Text API (Whisper)

**Speech-to-Text API** is a production-ready microservice for automatic speech recognition built with **FastAPI** and **OpenAI Whisper** (via **faster-whisper**).
The service provides high-quality transcription for audio and video files with support for **multilingual speech recognition**, optimized for CPU inference.

## What Changed

- Transcription is now asynchronous (queue-based).
- `POST /transcribe/` creates a task and returns `task_id`.
- Result is fetched via `GET /transcribe/tasks/{task_id}`.
- Queue status includes:
  - `status` (`queued`, `processing`, `retrying`, `done`, `failed`)
  - `queue_position` (best-effort)
  - `progress` (0..100 during processing)
- Added Flower monitoring UI with basic auth.

## Architecture

- `speech-to-text-service` (FastAPI): accepts files and enqueues tasks
- `worker` (Celery): processes transcription tasks
- `redis`: broker/result backend + queue position tracking
- `flower`: Celery monitoring UI

## Supported Models

`small`, `medium`, `large`

Model is selected per request using `model` query parameter.

## Export Formats

`docx`, `txt`, `md`, `pdf`

## Quick Start (Docker)

From project root:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Available locally:

- API docs: `http://127.0.0.1:8000/docs`
- Flower: `http://127.0.0.1:5555`

## Flower Auth

Credentials are set in `docker/.env`:

```env
FLOWER_BASIC_AUTH=admin:admin
```

Format: `username:password`

Change this value before production use.

## Main Endpoints

### 1) Health check

```http
GET /
```

Response:

```json
{ "status": "API is running" }
```

### 2) Submit transcription task

```http
POST /transcribe/
```

#### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `file` | file | Audio/video file |
| `model` | string | `small` / `medium` / `large` |
| `language` | string | Language code (`ru`, `en`, `auto`, ...) |
| `task` | string | `transcribe` or `translate` |
| `beam_size` | int | 1..10 |
| `chunk_length` | int | 5..60 seconds |
| `patience` | float | Decoder patience |
| `length_penalty` | float | Length penalty |
| `repetition_penalty` | float | Repetition penalty |
| `multilingual` | bool | Enable multilingual mode |
| `result_format` | string | `docx` / `txt` / `md` / `pdf` |
| `save_file` | bool | Keep uploaded source file |
| `save_result` | bool | Keep result file |

Example:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/?model=medium&language=ru&result_format=pdf" \
  -F "file=@audio.mp3"
```

Response:

```json
{
  "task_id": "7c1e8f4d-5c43-4fc8-bd7d-8ce8b2c4f53e",
  "status": "queued",
  "queue_position": 2,
  "status_url": "/transcribe/tasks/7c1e8f4d-5c43-4fc8-bd7d-8ce8b2c4f53e"
}
```

### 3) Get task status/result

```http
GET /transcribe/tasks/{task_id}
```

Possible responses:

Queued:

```json
{
  "task_id": "7c1e8f4d-5c43-4fc8-bd7d-8ce8b2c4f53e",
  "status": "queued",
  "queue_position": 2,
  "progress": 0.0
}
```

Processing:

```json
{
  "task_id": "7c1e8f4d-5c43-4fc8-bd7d-8ce8b2c4f53e",
  "status": "processing",
  "queue_position": null,
  "progress": 47.3
}
```

Done:

```json
{
  "task_id": "7c1e8f4d-5c43-4fc8-bd7d-8ce8b2c4f53e",
  "status": "done",
  "queue_position": null,
  "progress": 100.0,
  "result": {
    "language": "ru",
    "duration": 42.8,
    "text": "Full transcription text",
    "segments": [
      { "start": 0.0, "end": 4.2, "text": "Segment text" }
    ],
    "result_file": "/app/data/transcriptions/audio_19_02_2026_103000.pdf"
  }
}
```

Failed:

```json
{
  "task_id": "7c1e8f4d-5c43-4fc8-bd7d-8ce8b2c4f53e",
  "status": "failed",
  "queue_position": null,
  "progress": 12.5,
  "error": "error message"
}
```

## Queue Position Note

`queue_position` is best-effort and may be approximate, especially with multiple workers.

## Local UI

Enable UI with:

```env
USE_INTERFACE=true
```

Then open:

`http://127.0.0.1:8000/ui`

## Utility Command

Clean transcription result directory:

```bash
python -m app.scripts.clean_transcriptions
```

## License

Apache License 2.0
