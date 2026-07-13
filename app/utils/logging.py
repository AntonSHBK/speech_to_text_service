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
            f"Log path must be a file, but it is a directory: {target_path}"
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
    """Return the main log file name for a logger."""
    if name.startswith("api"):
        return "api.log"
    if _is_worker_logger(name):
        return f"{_worker_log_stem()}.log"

    return f"{_safe_log_name(name)}.log"


def setup_logging(log_dir: Path = Path("logs"), log_level: str = "INFO"):
    """Configure base stdout logging and process log files."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = log_level.upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_formatter())
        root_logger.addHandler(console_handler)

    worker_name = os.getenv("WORKER_NAME", "").strip()
    if worker_name:
        process_log_stem = _safe_log_name(worker_name)
        _add_file_handler(
            logger=root_logger,
            target_path=(log_dir / f"{process_log_stem}.log").resolve(),
            level=log_level,
        )
        _add_file_handler(
            logger=root_logger,
            target_path=(log_dir / f"{process_log_stem}.warning.log").resolve(),
            level=logging.WARNING,
        )


def get_logger(
    name: str,
    log_dir: Path = Path("logs"),
    log_file: str | None = None,
    log_level: str = "INFO",
):
    """Create or return a named logger."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())
    logger.propagate = True

    log_dir.mkdir(parents=True, exist_ok=True)
    worker_name = os.getenv("WORKER_NAME", "").strip()
    is_worker_logger = _is_worker_logger(name)

    # Worker root logger owns worker-N.log and worker-N.warning.log.
    # Named worker/model/diarization loggers propagate to root to avoid duplicate lines.
    should_add_named_file = not is_worker_logger or not worker_name or log_file is not None

    # Do not create fallback worker.log when API imports task modules outside a worker process.
    if is_worker_logger and not worker_name and log_file is None:
        should_add_named_file = False

    if should_add_named_file:
        resolved_log_file = log_file or _resolve_log_file_by_name(name)
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

    return logger
