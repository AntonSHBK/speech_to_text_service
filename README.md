# Speech-to-Text API (Whisper)

**Speech-to-Text API** is a production-ready microservice for automatic speech recognition built with **FastAPI** and **OpenAI Whisper** (via **faster-whisper**).
The service provides high-quality transcription for audio and video files with support for **multilingual speech recognition**, optimized for CPU inference.

## What Changed

- Transcription is now asynchronous (queue-based).
- `POST /transcribe/file/` creates a task from uploaded file and returns `task_id`.
- `POST /transcribe/url/` creates a task from source URL (YouTube/Rutube/etc.) and returns `task_id`.
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

## Runtime Settings (.env)

Key inference settings:

```env
DEVICE=cpu
MODEL_CPU_THREADS=8
MODEL_NUM_WORKERS=1

# Global fallback compute type
MODEL_COMPUTE_TYPE=default

# Per-model compute type overrides
MODEL_COMPUTE_TYPE_SMALL=default
MODEL_COMPUTE_TYPE_MEDIUM=default
MODEL_COMPUTE_TYPE_LARGE=int8

HF_TOKEN=your_hf_token
```

Per-model compute type is resolved at model load time:
- `small` -> `MODEL_COMPUTE_TYPE_SMALL` (fallback: `MODEL_COMPUTE_TYPE`)
- `medium` -> `MODEL_COMPUTE_TYPE_MEDIUM` (fallback: `MODEL_COMPUTE_TYPE`)
- `large` -> `MODEL_COMPUTE_TYPE_LARGE` (fallback: `MODEL_COMPUTE_TYPE`)

## Export Formats

`docx`, `txt`, `md`, `pdf`

## Quick Start (Docker)

The project now has two Docker Compose configurations:

- CPU: `docker/docker-compose.yml`
- CUDA (GPU): `docker/docker-compose.cuda.yml`

From project root, run one of the following:

CPU:

```bash
docker compose -f docker/docker-compose.yml up --build
```

CUDA:

```bash
docker compose -f docker/docker-compose.cuda.yml up --build
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

### 2) Submit transcription task from file

```http
POST /transcribe/file/
```

#### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `file` | file | Audio/video file |
| `model` | string | `small` / `medium` / `large` |
| `language` | string | Language code (`ru`, `en`, ...). If omitted, auto-detection is used |
| `task` | string | `transcribe` or `translate` |
| `beam_size` | int | Beam search size (default `5`) |
| `best_of` | int | Number of candidates for sampling mode (default `5`) |
| `chunk_length` | int | Chunk length in seconds (optional) |
| `patience` | float | Decoder patience |
| `length_penalty` | float | Length penalty |
| `repetition_penalty` | float | Repetition penalty |
| `multilingual` | bool | Enable multilingual mode |
| `result_format` | string | `docx` / `txt` / `md` / `pdf` |
| `save_source` | bool | Keep uploaded source file |
| `save_result` | bool | Keep result file |

Advanced faster-whisper parameters are also supported directly in API query:

`log_progress`, `no_repeat_ngram_size`, `temperature`, `compression_ratio_threshold`,
`log_prob_threshold`, `no_speech_threshold`, `condition_on_previous_text`,
`prompt_reset_on_temperature`, `initial_prompt`, `prefix`, `suppress_blank`,
`suppress_tokens`, `without_timestamps`, `max_initial_timestamp`, `word_timestamps`,
`prepend_punctuations`, `append_punctuations`, `vad_filter`, `vad_parameters` (JSON string),
`max_new_tokens`, `clip_timestamps`, `hallucination_silence_threshold`, `hotwords`,
`language_detection_threshold`, `language_detection_segments`.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/file/?model=medium&language=ru&result_format=pdf&beam_size=5&word_timestamps=true" \
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

### 3) Submit transcription task from URL

```http
POST /transcribe/url/
```

#### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `source_url` | string | Public media URL |
| `model` | string | `small` / `medium` / `large` |
| `language` | string | Language code (`ru`, `en`, ...). If omitted, auto-detection is used |
| `task` | string | `transcribe` or `translate` |
| `beam_size` | int | Beam search size (default `5`) |
| `best_of` | int | Number of candidates for sampling mode (default `5`) |
| `chunk_length` | int | Chunk length in seconds (optional) |
| `patience` | float | Decoder patience |
| `length_penalty` | float | Length penalty |
| `repetition_penalty` | float | Repetition penalty |
| `multilingual` | bool | Enable multilingual mode |
| `result_format` | string | `docx` / `txt` / `md` / `pdf` |
| `save_source` | bool | Keep downloaded source file |
| `save_result` | bool | Keep result file |

The same advanced faster-whisper parameters as in `/transcribe/file/` are supported here too.
For `vad_parameters` pass JSON as string in query.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/transcribe/url/?source_url=https://www.youtube.com/watch?v=VIDEO_ID&model=small&result_format=pdf&beam_size=5"
```

Response is the same queue object as for `/transcribe/file/`.
`source_url` download now happens in worker during task processing (not in API request lifecycle).

### 4) Get task status/result

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
    "result_file": "/app/data/transcriptions/audio_19_02_2026_103000.pdf",
    "result_filename": "audio_19_02_2026_103000.pdf"
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

### 5) Download exported file

```http
GET /transcribe/files/{filename}
```

Example:

```bash
curl -OJ "http://127.0.0.1:8000/transcribe/files/audio_19_02_2026_103000.pdf"
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
