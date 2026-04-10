import sys
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler


def _create_logger(name: str, log_dir: Path, log_file: str, log_level: str, max_bytes: int, backup_count: int):
    """Вспомогательная функция для создания именованных логгеров."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    target_path = (log_dir / log_file).resolve()
    has_same_file_handler = False
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            try:
                if Path(handler.baseFilename).resolve() == target_path:
                    has_same_file_handler = True
                    break
            except Exception:
                continue

    if not has_same_file_handler:
        file_handler = RotatingFileHandler(
            target_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _resolve_log_file_by_name(name: str) -> str:
    """Возвращает имя файла лога для логгера."""
    if name.startswith("api"):
        return "api.log"
    if name.startswith("model.") or name.startswith("diarization.") or name == "worker.models":
        return "model.log"
    if name.startswith("worker."):
        return "worker.log"

    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return f"{safe_name}.log"


def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO"):
    """Настройка базового логирования и стандартных логгеров."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = log_level.upper()

    # Общий форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        # Консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # Общий файл
        file_handler = RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, log_dir: Path = Path("logs"), log_file: str | None = None, log_level: str = "INFO"):
    """
    Создаёт или возвращает именованный логгер.
    Если указан log_file, логи будут писаться в отдельный файл.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    resolved_log_file = log_file or _resolve_log_file_by_name(name)
    log_dir.mkdir(parents=True, exist_ok=True)
    _create_logger(
        name=name,
        log_dir=log_dir,
        log_file=resolved_log_file,
        log_level=log_level.upper(),
        max_bytes=5 * 1024 * 1024,
        backup_count=3,
    )

    return logger
