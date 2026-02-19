import time
from pathlib import Path
from typing import Callable, List, Optional, Union

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from app.models.base import BaseModel


class FastWhisperTranscriber(BaseModel):
    def __init__(
        self,
        model_name: str,
        cache_dir: Path = "cache_dir",
        device: str = "cpu",
        token: str = None,
        compute_type: str = "default",
        cpu_threads: int = 4,
        num_workers: int = 8,
    ):
        super().__init__(model_name, cache_dir, device)
        self.model = self.load_model(
            model_name=model_name,
            use_auth_token=token,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    def load_model(
        self,
        model_name: str,
        use_auth_token: str = None,
        compute_type: str = "default",
        cpu_threads: int = 4,
        num_workers: int = 8,
    ) -> WhisperModel:
        model = WhisperModel(
            model_size_or_path=model_name,
            device=self.device,
            download_root=str(self.cache_dir),
            use_auth_token=use_auth_token,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )
        self.logger.info("Model loaded: %s", model_name)
        return model

    def process(self, audio_path: Union[str, Path]) -> Path:
        return Path(audio_path)

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = "ru",
        task: str = "transcribe",
        beam_size: int = 3,
        chunk_length: int = 10,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        multilingual: bool = False,
        on_progress: Callable[[float], None] | None = None,
        **kwargs,
    ) -> dict:
        audio_path = self.process(audio_path)
        self.logger.info(
            "Transcription started | model=%s | file=%s | language=%s | task=%s",
            self.model_name,
            audio_path,
            language,
            task,
        )

        start_time = time.perf_counter()
        segments, info = self.model.transcribe(
            audio=str(audio_path),
            language=language,
            task=task,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            beam_size=beam_size,
            chunk_length=chunk_length,
            multilingual=multilingual,
            **kwargs,
        )

        collected_segments: List[Segment] = []
        text_parts: List[str] = []
        last_logged = 0

        duration = info.duration or 0.0
        for segment in segments:
            collected_segments.append(segment)
            text_parts.append(segment.text.strip())

            if duration > 0:
                progress = min(100.0, (segment.end / duration) * 100)
                if int(progress) >= last_logged + 2:
                    last_logged = int(progress)
                    self.logger.info(
                        "Progress | model=%s | file=%s | value=%.1f%%",
                        self.model_name,
                        audio_path,
                        progress,
                    )
                    if on_progress:
                        on_progress(progress)

        processing_time = time.perf_counter() - start_time
        speed_ratio = duration / processing_time if processing_time > 0 else 0.0

        self.logger.info(
            "Transcription finished | model=%s | file=%s | duration=%.2fs | processing=%.2fs | speed=x%.2f",
            self.model_name,
            audio_path,
            duration,
            processing_time,
            speed_ratio,
        )

        full_text = " ".join(part.strip() for part in text_parts if part and part.strip())
        return {
            "language": info.language,
            "duration": duration,
            "text": full_text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in collected_segments
            ],
        }
