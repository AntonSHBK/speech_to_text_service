import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(processName)s[%(process)d] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


WORKER_LOG_PREFIXES = (
    "worker.",
    "model.",
    "diarization.",
)


def _formatter() -> logging.Formatter:
    return logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)


def _ensure_log_path(target_path: Path) -> None:
    if not target_path.is_dir():
        return

    try:
        target_path.rmdir()
    except OSError as exc:
        raise RuntimeError(
            f"Путь лога должен быть файлом, но является директорией: {target_path}"
        ) from exc


def _has_file_handler(logger: logging.Logger, target_path: Path) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            try:
                if Path(handler.baseFilename).resolve() == target_path.resolve():
                    return True
            except Exception:
                continue
    return False


def _add_file_handler(
    logger: logging.Logger,
    target_path: Path,
    level: str | int,
) -> None:
    _ensure_log_path(target_path)
    if _has_file_handler(logger, target_path):
        return

    file_handler = RotatingFileHandler(
        target_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_formatter())
    logger.addHandler(file_handler)


def _safe_log_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _is_worker_logger(name: str) -> bool:
    return name.startswith(WORKER_LOG_PREFIXES)


def _worker_log_stem() -> str:
    worker_name = os.getenv("WORKER_NAME", "worker").strip() or "worker"
    return _safe_log_name(worker_name)


def _resolve_log_file_by_name(name: str) -> str:
    """Возвращает имя основного файла лога для логгера."""
    if name.startswith("api"):
        return "api.log"
    if _is_worker_logger(name):
        return f"{_worker_log_stem()}.log"

    return f"{_safe_log_name(name)}.log"


def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO"):
    """Настраивает базовое логирование в stdout."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = log_level.upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_formatter())
        root_logger.addHandler(console_handler)


def get_logger(
    name: str,
    log_dir: Path = Path("logs"),
    log_file: str | None = None,
    log_level: str = "INFO",
):
    """Создаёт или возвращает именованный логгер."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    resolved_log_file = log_file or _resolve_log_file_by_name(name)
    log_dir.mkdir(parents=True, exist_ok=True)

    _add_file_handler(
        logger=logger,
        target_path=(log_dir / resolved_log_file).resolve(),
        level=log_level.upper(),
    )

    if name.startswith("api"):
        _add_file_handler(
            logger=logger,
            target_path=(log_dir / "api.warning.log").resolve(),
            level=logging.WARNING,
        )

    if _is_worker_logger(name):
        _add_file_handler(
            logger=logger,
            target_path=(log_dir / f"{_worker_log_stem()}.warning.log").resolve(),
            level=logging.WARNING,
        )

    return logger
