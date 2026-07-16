import gc
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from app.models.catalog import ModelDiarizationType, resolve_speaker_diarization_model_name
from app.models.pyannote_speaker_diarization import PyannoteSpeakerDiarizationModel
from app.models.sherpa_speaker_diarization import (
    SherpaOnnxProvider,
    SherpaOnnxSpeakerDiarizationModel,
)
from app.settings import settings
from app.utils.logging import get_logger


class PyannoteSpeakerDiarizationService:
    def __init__(self):
        self.diarizer: Optional[PyannoteSpeakerDiarizationModel] = None
        self.current_model_name: str | None = None
        self.logger = get_logger("worker.diarization.pyannote")

    def init(
        self,
        model: ModelDiarizationType = "pyannote_1",
        device: str | None = None,
        token: str | None = None,
    ) -> PyannoteSpeakerDiarizationModel:
        model_name = resolve_speaker_diarization_model_name(model)

        if self.diarizer and self.current_model_name == model_name:
            self.logger.info("Модель Pyannote diarization уже активна: %s", model_name)
            return self.diarizer

        if self.diarizer is not None:
            self.logger.info(
                "Переключение модели Pyannote diarization: %s -> %s. Выгружаем предыдущую модель.",
                self.current_model_name,
                model_name,
            )
            self.release()

        diarizer = PyannoteSpeakerDiarizationModel(
            model_name=model_name,
            device=device or settings.DEVICE,
            token=token or settings.HF_TOKEN,
        )
        self.diarizer = diarizer
        self.current_model_name = model_name
        self.logger.info("Модель Pyannote diarization активирована: %s", model_name)
        return diarizer

    def get_or_init(
        self,
        model: ModelDiarizationType = "pyannote_1",
        device: str | None = None,
        token: str | None = None,
    ) -> PyannoteSpeakerDiarizationModel:
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

        self.logger.info("Модель Pyannote diarization выгружена из памяти: %s", model_name)

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


class SherpaSpeakerDiarizationService:
    def __init__(self):
        self.diarizer: Optional[SherpaOnnxSpeakerDiarizationModel] = None
        self.current_config: tuple[Any, ...] | None = None
        self.logger = get_logger("worker.diarization.sherpa")

    def init(
        self,
        provider: SherpaOnnxProvider | None = None,
        token: str | None = None,
        cluster_threshold: float = 1.0,
        num_threads: int = 1,
        min_duration_on: float = 0.4,
        min_duration_off: float = 0.4,
        merge_gap: float = 0.2,
        min_segment_duration: float = 0.3,
        local_files_only: bool | None = None,
    ) -> SherpaOnnxSpeakerDiarizationModel:
        effective_provider = "cpu"
        effective_num_threads = num_threads
        effective_local_files_only = (
            settings.DIARIZATION_LOCAL_FILES_ONLY
            if local_files_only is None
            else local_files_only
        )
        config = (
            effective_provider,
            token or settings.HF_TOKEN,
            cluster_threshold,
            effective_num_threads,
            min_duration_on,
            min_duration_off,
            merge_gap,
            min_segment_duration,
            effective_local_files_only,
        )

        if self.diarizer and self.current_config == config:
            self.logger.info("Модель Sherpa-ONNX diarization уже активна")
            return self.diarizer

        if self.diarizer is not None:
            self.logger.info("Переинициализация модели Sherpa-ONNX diarization. Выгружаем предыдущую модель.")
            self.release()

        diarizer = SherpaOnnxSpeakerDiarizationModel(
            cache_dir=settings.CACHE_DIR,
            token=token or settings.HF_TOKEN,
            cluster_threshold=cluster_threshold,
            provider=effective_provider,
            num_threads=effective_num_threads,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            merge_gap=merge_gap,
            min_segment_duration=min_segment_duration,
            local_files_only=effective_local_files_only,
        )
        self.diarizer = diarizer
        self.current_config = config
        self.logger.info(
            "Модель Sherpa-ONNX diarization активирована | provider=%s | num_threads=%s | cluster_threshold=%s",
            effective_provider,
            effective_num_threads,
            cluster_threshold,
        )
        return diarizer

    def get_or_init(
        self,
        provider: SherpaOnnxProvider | None = None,
        token: str | None = None,
        cluster_threshold: float = 1.0,
        num_threads: int = 1,
        min_duration_on: float = 0.4,
        min_duration_off: float = 0.4,
        merge_gap: float = 0.2,
        min_segment_duration: float = 0.3,
        local_files_only: bool | None = None,
    ) -> SherpaOnnxSpeakerDiarizationModel:
        effective_provider = "cpu"
        effective_num_threads = num_threads
        effective_local_files_only = (
            settings.DIARIZATION_LOCAL_FILES_ONLY
            if local_files_only is None
            else local_files_only
        )
        config = (
            effective_provider,
            token or settings.HF_TOKEN,
            cluster_threshold,
            effective_num_threads,
            min_duration_on,
            min_duration_off,
            merge_gap,
            min_segment_duration,
            effective_local_files_only,
        )
        if self.diarizer and self.current_config == config:
            return self.diarizer
        return self.init(
            provider=provider,
            token=token,
            cluster_threshold=cluster_threshold,
            num_threads=num_threads,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            merge_gap=merge_gap,
            min_segment_duration=min_segment_duration,
            local_files_only=effective_local_files_only,
        )

    def release(self) -> None:
        if self.diarizer is None:
            return

        self.diarizer = None
        self.current_config = None
        gc.collect()
        self.logger.info("Модель Sherpa-ONNX diarization выгружена из памяти")

    def diarize(
        self,
        audio_path: str | Path,
        num_speakers: int | None = None,
        on_progress: Callable[[float], None] | None = None,
        provider: SherpaOnnxProvider | None = None,
        cluster_threshold: float = 1.0,
        num_threads: int = 1,
        min_duration_on: float = 0.4,
        min_duration_off: float = 0.4,
        merge_gap: float = 0.2,
        min_segment_duration: float = 0.3,
    ) -> dict:
        diarizer = self.get_or_init(
            provider=provider,
            cluster_threshold=cluster_threshold,
            num_threads=num_threads,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
            merge_gap=merge_gap,
            min_segment_duration=min_segment_duration,
        )
        return diarizer.diarize(
            audio_path=audio_path,
            num_speakers=num_speakers,
            on_progress=on_progress,
        )


# pyannote_diary_service = PyannoteSpeakerDiarizationService()
# diary_service = pyannote_diary_service

sherpa_diary_service = SherpaSpeakerDiarizationService()
diary_service = sherpa_diary_service
