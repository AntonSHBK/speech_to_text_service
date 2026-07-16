import time
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Callable

import soundfile as sf
import torch
from pyannote.audio import Pipeline

from app.utils.logging import get_logger


class _PyannoteProgressBridgeHook:
    """Передаёт прогресс pyannote hook в числовой callback [0..100]."""

    def __init__(self, on_progress: Callable[[float], None] | None = None):
        self.on_progress = on_progress

    def __call__(self, *args, **kwargs):
        if not self.on_progress:
            return

        completed = kwargs.get("completed")
        total = kwargs.get("total")

        if completed is None or total is None:
            numeric_args = [a for a in args if isinstance(a, (int, float))]
            if len(numeric_args) >= 2:
                completed, total = numeric_args[-2], numeric_args[-1]

        if not isinstance(completed, (int, float)) or not isinstance(total, (int, float)):
            return
        if total <= 0:
            return

        progress = max(0.0, min(100.0, float(completed) / float(total) * 100.0))
        self.on_progress(progress)


class _PyannoteCompositeHook:
    """Вызывает несколько hook с одинаковыми данными прогресса pyannote."""

    def __init__(self, hooks: list[Any]):
        self.hooks = [h for h in hooks if h is not None]

    def __call__(self, *args, **kwargs):
        for hook in self.hooks:
            hook(*args, **kwargs)


class PyannoteSpeakerDiarizationModel:
    """Обёртка над pyannote для определения спикеров."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        token: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        safe_model_name = model_name.replace("/", ".")
        self.logger = get_logger(f"diarization.{safe_model_name}")
        self.pipeline = self.load_pipeline(model_name=model_name, token=token)

    def load_pipeline(self, model_name: str, token: str | None = None) -> Pipeline:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
        if self.device == "cuda":
            pipeline.to(torch.device("cuda"))
        self.logger.info("Пайплайн diarization загружен: %s", model_name)
        return pipeline

    def _prepare_audio_for_diarization(
        self,
        source: Path,
    ) -> tuple[dict[str, Any], Path]:
        """Преобразует медиа в mono 16 kHz PCM и загружает без TorchCodec."""
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
            dir=source.parent,
        ) as tmp:
            prepared_path = Path(tmp.name)

        try:
            base_command = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
            ]
            output_options = [
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(prepared_path),
            ]
            attempts = (
                ("строгий", []),
                (
                    "восстановление",
                    [
                        "-fflags",
                        "+discardcorrupt",
                        "-err_detect",
                        "ignore_err",
                    ],
                ),
            )
            errors: list[str] = []
            audio = None
            sample_rate = 0

            for attempt_name, input_options in attempts:
                prepared_path.unlink(missing_ok=True)
                completed = subprocess.run(
                    [
                        *base_command,
                        *input_options,
                        "-i",
                        str(source),
                        *output_options,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if completed.returncode != 0:
                    error = (completed.stderr or "").strip()
                    errors.append(
                        f"{attempt_name}: ffmpeg code={completed.returncode}: {error}"
                    )
                    continue

                try:
                    audio, sample_rate = sf.read(
                        prepared_path,
                        dtype="float32",
                        always_2d=True,
                    )
                except Exception as exc:
                    errors.append(f"{attempt_name}: WAV не читается: {exc}")
                    continue

                if audio.size == 0 or audio.shape[0] == 0:
                    errors.append(f"{attempt_name}: ffmpeg создал пустой аудиофайл")
                    audio = None
                    continue

                break

            if audio is None:
                raise RuntimeError("; ".join(errors))

            waveform = torch.from_numpy(audio.T.copy())
            self.logger.info(
                "Аудио подготовлено для diarization | исходник=%s | "
                "подготовленный=%s | sample_rate=%s | channels=%s",
                source,
                prepared_path,
                sample_rate,
                waveform.shape[0],
            )

            return {
                "waveform": waveform,
                "sample_rate": int(sample_rate),
            }, prepared_path
        except Exception as exc:
            prepared_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Не удалось подготовить аудио для diarization: {source}"
            ) from exc

    def _apply_test_pipeline_parameters(self) -> None:
        self.pipeline.instantiate( 
            {
                "segmentation": {
                    # Минимальная длительность паузы между речевыми сегментами
                    "min_duration_off": 0.0,
                },
                "clustering": {
                    # Порог кластеризации
                    "threshold": 0.70,

                    # Параметры алгоритма VBx
                    # "Fa": 0.07,
                    # "Fb": 0.80,
                },
            }
        )

    def diarize(
        self,
        audio_path: str | Path,
        hook: Any = None,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        source = Path(audio_path)
        if not source.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {source}")

        self.logger.info(
            "Diarization запущен | модель=%s | файл=%s | num_speakers=%s | min_speakers=%s | max_speakers=%s",
            self.model_name,
            source,
            num_speakers,
            min_speakers,
            max_speakers,
        )

        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        bridge_hook = _PyannoteProgressBridgeHook(on_progress=on_progress) if on_progress else None
        effective_hook = _PyannoteCompositeHook([hook, bridge_hook]) if (hook or bridge_hook) else None
        if effective_hook is not None:
            kwargs["hook"] = effective_hook

        started_at = time.perf_counter()
        self._apply_test_pipeline_parameters()
        prepared_audio, prepared_path = self._prepare_audio_for_diarization(source)

        try:
            output = self.pipeline(prepared_audio, **kwargs)
        finally:
            prepared_path.unlink(missing_ok=True)

        segments: list[dict[str, Any]] = []
        speakers: set[str] = set()
        max_end = 0.0

        # --- основной diarization ---
        diarization = getattr(output, "speaker_diarization", None)

        if diarization is not None:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)

                max_end = max(max_end, end)
                speakers.add(str(speaker))

                segments.append(
                    {
                        "speaker": str(speaker),
                        "start": start,
                        "end": end,
                        "duration": end - start,
                    }
                )

        # --- эксклюзивный diarization ---
        exclusive_segments: list[dict[str, Any]] = []
        exclusive = getattr(output, "exclusive_speaker_diarization", None)

        if exclusive is not None:
            for turn, _, speaker in exclusive.itertracks(yield_label=True):
                start = float(turn.start)
                end = float(turn.end)

                exclusive_segments.append(
                    {
                        "speaker": str(speaker),
                        "start": start,
                        "end": end,
                        "duration": end - start,
                    }
                )

        processing_time = time.perf_counter() - started_at
        self.logger.info(
            "Diarization завершён | модель=%s | файл=%s | спикеров=%s | сегментов=%s | длительность=%.2fs | обработка=%.2fs",
            self.model_name,
            source,
            len(speakers),
            len(segments),
            max_end,
            processing_time,
        )

        return {
            "duration": max_end,
            "num_speakers": len(speakers),
            "speakers": sorted(speakers),
            "segments": segments,
            "exclusive_segments": exclusive_segments,
            "processing_time": processing_time,
        }
