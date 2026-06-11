import json
import subprocess
import uuid
from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.settings import settings


def _validate_downloaded_media(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("Скачанный медиафайл отсутствует или пуст")

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        error = (completed.stderr or "").strip()
        raise RuntimeError(f"ffprobe не смог прочитать скачанный файл: {error}")

    try:
        metadata = json.loads(completed.stdout or "{}")
        duration = float(metadata.get("format", {}).get("duration") or 0.0)
        streams = metadata.get("streams") or []
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("ffprobe вернул некорректные метаданные") from exc

    if not streams:
        raise RuntimeError("В скачанном файле отсутствует аудиопоток")
    if duration <= 0:
        raise RuntimeError("Скачанный аудиопоток имеет нулевую длительность")


def _remove_download_attempt_files(download_id: str) -> None:
    for path in settings.AUDIO_DIR.glob(f"*_{download_id}.*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def _resolve_downloaded_path(ydl: YoutubeDL, info: dict) -> Path:
    if "entries" in info and info["entries"]:
        info = info["entries"][0]

    filepath = None
    if info.get("requested_downloads"):
        filepath = info["requested_downloads"][0].get("filepath")
    if not filepath:
        filepath = ydl.prepare_filename(info)
    return Path(filepath)


def download_source(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError("source_url должен использовать схему http или https")

    common_opts = {
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "socket_timeout": settings.YTDLP_SOCKET_TIMEOUT_SEC,
        "continuedl": False,
        "overwrites": True,
        "retries": 3,
        "fragment_retries": 3,
    }

    with YoutubeDL(common_opts) as ydl:
        metadata = ydl.extract_info(url, download=False)
        if not metadata:
            raise RuntimeError("Не удалось получить метаданные источника")
        if "entries" in metadata and metadata["entries"]:
            metadata = metadata["entries"][0]

        duration = metadata.get("duration")
        if duration and duration > settings.YTDLP_MAX_DURATION_SEC:
            raise RuntimeError(
                f"Длительность источника превышает лимит: {int(duration)}с > {settings.YTDLP_MAX_DURATION_SEC}с"
            )

    attempts = (
        {
            "name": "default",
            "format": "bestaudio[ext=m4a]/bestaudio/best",
        },
        {
            "name": "web_safari_hls",
            "format": "bestaudio[protocol^=m3u8]/best[protocol^=m3u8]",
            "extractor_args": {
                "youtube": {
                    "player_client": ["web_safari"],
                }
            },
        },
        {
            "name": "web_embedded",
            "format": "bestaudio/best",
            "extractor_args": {
                "youtube": {
                    "player_client": ["web_embedded"],
                }
            },
        },
    )
    errors: list[str] = []

    for attempt in attempts:
        download_id = uuid.uuid4().hex
        ydl_opts = {
            **common_opts,
            **attempt,
            "outtmpl": str(
                settings.AUDIO_DIR
                / f"%(title).100s_%(id)s_{download_id}.%(ext)s"
            ),
        }
        attempt_name = str(ydl_opts.pop("name"))

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    raise RuntimeError("yt-dlp не вернул данные скачивания")
                path = _resolve_downloaded_path(ydl, info)

            _validate_downloaded_media(path)
            return path
        except (DownloadError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            errors.append(f"{attempt_name}: {exc}")
            _remove_download_attempt_files(download_id)

    raise RuntimeError(
        "Не удалось скачать корректный аудиопоток. "
        + " | ".join(errors)
    )
