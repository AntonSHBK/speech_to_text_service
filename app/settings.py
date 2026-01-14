import os
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field

from app.utils.logging import setup_logging

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Глобальные настройки приложения."""
    
    DEVICE: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME: str = Field(default="Systran/faster-whisper-small")
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    AUDIO_DIR: Path = BASE_DIR / "data" / "audio"
    TRANSCRIBE_RESULTS_DIR: Path = BASE_DIR / "data" / "transcriptions"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache_dir"
    LOG_DIR: Path = BASE_DIR / "logs"

    LOG_LEVEL: str = "INFO"
    
    USE_INTERFACE: bool = False
    MODEL_CPU_THREADS: int = 8
    MODEL_NUM_WORKERS: int = 16

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


    @field_validator(
        "CACHE_DIR",
        "LOG_DIR",
        "DATA_DIR",
        "AUDIO_DIR",
        "TRANSCRIBE_RESULTS_DIR",
        mode="before"
    )
    @classmethod
    def create_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

settings = Settings()

setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)