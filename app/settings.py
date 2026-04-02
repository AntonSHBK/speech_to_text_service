import os
from pathlib import Path
from typing import Literal

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field

from app.utils.logging import setup_logging
from app.models.catalog import ModelSize

BASE_DIR = Path(__file__).resolve().parent.parent
ModelComputeType = Literal[
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "int16",
    "float16",
    "bfloat16",
    "float32",
]


class Settings(BaseSettings):
    """Глобальные настройки приложения."""
    
    DEVICE: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    AUDIO_DIR: Path = BASE_DIR / "data" / "audio"
    TRANSCRIBE_RESULTS_DIR: Path = BASE_DIR / "data" / "transcriptions"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache_dir"
    LOG_DIR: Path = BASE_DIR / "logs"

    LOG_LEVEL: str = "INFO"
    
    USE_INTERFACE: bool = False
    
    MODEL_CPU_THREADS: int = 2
    MODEL_NUM_WORKERS: int = 4
    MODEL_COMPUTE_TYPE: ModelComputeType = "default"
    MODEL_COMPUTE_TYPE_SMALL: ModelComputeType = "default"
    MODEL_COMPUTE_TYPE_MEDIUM: ModelComputeType = "default"
    MODEL_COMPUTE_TYPE_LARGE: ModelComputeType = "default"

    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    YTDLP_SOCKET_TIMEOUT_SEC: int = 180
    YTDLP_MAX_DURATION_SEC: int = 10800
    HF_TOKEN: str | None = None

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

    def get_model_compute_type(self, model: ModelSize) -> ModelComputeType:
        per_model: dict[ModelSize, ModelComputeType | None] = {
            "small": self.MODEL_COMPUTE_TYPE_SMALL,
            "medium": self.MODEL_COMPUTE_TYPE_MEDIUM,
            "large": self.MODEL_COMPUTE_TYPE_LARGE,
        }
        return per_model.get(model) or self.MODEL_COMPUTE_TYPE

settings = Settings()

setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)
