from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL

from app.settings import settings


def download_source(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError("source_url должен использовать схему http или https")

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "socket_timeout": settings.YTDLP_SOCKET_TIMEOUT_SEC,
        "outtmpl": str(settings.AUDIO_DIR / "%(title).120s_%(id)s.%(ext)s"),
    }
    with YoutubeDL(ydl_opts) as ydl:
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

        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("Не удалось получить медиа по URL")

        if "entries" in info and info["entries"]:
            info = info["entries"][0]

        filepath = None
        if info.get("requested_downloads"):
            filepath = info["requested_downloads"][0].get("filepath")
        if not filepath:
            filepath = ydl.prepare_filename(info)

    path = Path(filepath)
    if not path.exists():
        raise RuntimeError("Скачанный файл не найден")
    return path
