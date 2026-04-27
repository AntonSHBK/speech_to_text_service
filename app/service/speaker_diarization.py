import gc
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from app.models.catalog import ModelDiarizationType, resolve_speaker_diarization_model_name
from app.models.speaker_diarization import SpeakerDiarizationModel
from app.settings import settings
from app.utils.logging import get_logger


class SpeakerDiarizationService:
    def __init__(self):
        self.diarizer: Optional[SpeakerDiarizationModel] = None
        self.current_model_name: str | None = None
        self.logger = get_logger("worker.diarization")

    def init(
        self,
        model: ModelDiarizationType = "pyannote_1",
        device: str | None = None,
        token: str | None = None,
    ) -> SpeakerDiarizationModel:
        model_name = resolve_speaker_diarization_model_name(model)

        if self.diarizer and self.current_model_name == model_name:
            self.logger.info("Модель diarization уже активна: %s", model_name)
            return self.diarizer

        if self.diarizer is not None:
            self.logger.info(
                "Переключение модели diarization: %s -> %s. Выгружаем предыдущую модель.",
                self.current_model_name,
                model_name,
            )
            self.diarizer = None
            gc.collect()

            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                except Exception:
                    pass

        diarizer = SpeakerDiarizationModel(
            model_name=model_name,
            device=device or settings.DEVICE,
            token=token or settings.HF_TOKEN,
        )
        self.diarizer = diarizer
        self.current_model_name = model_name
        self.logger.info("Модель diarization активирована: %s", model_name)
        return diarizer

    def get_or_init(
        self,
        model: ModelDiarizationType = "pyannote_1",
        device: str | None = None,
        token: str | None = None,
    ) -> SpeakerDiarizationModel:
        model_name = resolve_speaker_diarization_model_name(model)
        if self.diarizer and self.current_model_name == model_name:
            return self.diarizer
        return self.init(model=model, device=device, token=token)

    def release(self) -> None:
        if self.diarizer is None:
            return

        model_name = self.current_model_name
        self.diarizer = None
        self.current_model_name = None
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass

        self.logger.info("Модель diarization выгружена из памяти: %s", model_name)

    def diarize(
        self,
        audio_path: str | Path,
        model: ModelDiarizationType = "pyannote_1",
        hook: Any = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        diarizer = self.get_or_init(model=model)
        return diarizer.diarize(
            audio_path=audio_path,
            hook=hook,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            on_progress=on_progress,
        )


diary_service = SpeakerDiarizationService()
