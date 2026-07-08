from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.celery_app import celery_app
from app.settings import settings
from app.utils.logging import get_logger


logger = get_logger("worker.cleanup")


def _iter_cleanup_files(directories: list[Path]):
    for directory in directories:
        if not directory.exists():
            logger.info("Директория очистки не найдена | директория=%s", directory)
            continue

        for path in directory.rglob("*"):
            if path.is_file():
                yield path


@celery_app.task(name="cleanup.old_files")
def cleanup_old_files() -> dict[str, int | bool]:
    """Удаляет старые исходные файлы и результаты обработки."""
    if not settings.AUTO_CLEANUP_ENABLED:
        logger.info("Автоочистка отключена настройкой AUTO_CLEANUP_ENABLED")
        return {"enabled": False, "deleted": 0, "failed": 0}

    ttl = timedelta(hours=settings.CLEANUP_FILE_TTL_HOURS)
    cutoff = datetime.now(timezone.utc) - ttl
    directories = [
        Path(settings.AUDIO_DIR),
        Path(settings.TRANSCRIBE_RESULTS_DIR),
    ]

    deleted = 0
    failed = 0

    logger.info(
        "Автоочистка файлов начата | ttl_hours=%s | cutoff=%s | directories=%s",
        settings.CLEANUP_FILE_TTL_HOURS,
        cutoff.isoformat(),
        [str(directory) for directory in directories],
    )

    for path in _iter_cleanup_files(directories):
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if modified_at >= cutoff:
                continue

            path.unlink()
            deleted += 1
            logger.info(
                "Удалён старый файл | файл=%s | modified_at=%s",
                path,
                modified_at.isoformat(),
            )
        except OSError:
            failed += 1
            logger.exception("Не удалось удалить старый файл | файл=%s", path)

    logger.info(
        "Автоочистка файлов завершена | удалено=%s | ошибок=%s",
        deleted,
        failed,
    )
    return {"enabled": True, "deleted": deleted, "failed": failed}
