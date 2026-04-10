import gc
from pathlib import Path
from typing import Any, Callable, Optional

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
            self.logger.info("Diarization model already active: %s", model_name)
            return self.diarizer

        if self.diarizer is not None:
            self.logger.info(
                "Switching diarization model: %s -> %s. Releasing previous model.",
                self.current_model_name,
                model_name,
            )
            self.diarizer = None
            gc.collect()

        diarizer = SpeakerDiarizationModel(
            model_name=model_name,
            device=device or settings.DEVICE,
            token=token or settings.HF_TOKEN,
        )
        self.diarizer = diarizer
        self.current_model_name = model_name
        self.logger.info("Diarization model activated: %s", model_name)
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
