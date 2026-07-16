from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
import sherpa_onnx

from app.utils.logging import get_logger


DEFAULT_SEGMENTATION_REPO_ID = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
# DEFAULT_SEGMENTATION_REPO_ID = "csukuangfj/sherpa-onnx-reverb-diarization-v1"
DEFAULT_SEGMENTATION_FILENAME = "model.onnx"

DEFAULT_EMBEDDING_REPO_ID = "csukuangfj/speaker-embedding-models"
DEFAULT_EMBEDDING_FILENAME = "nemo_en_titanet_small.onnx"
# DEFAULT_EMBEDDING_FILENAME = "nemo_en_titanet_large.onnx"
# DEFAULT_EMBEDDING_FILENAME = "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"


SherpaOnnxProvider = Literal["cpu", "cuda", "coreml"]


@dataclass(slots=True)
class SherpaDiarizationSegment:
    start: float
    end: float
    speaker_id: int
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


class SherpaOnnxSpeakerDiarizationModel:
    """Офлайн-модель определения спикеров на базе sherpa-onnx.

    Модель загружает два ONNX-файла с Hugging Face в cache_dir:
    - сегментация спикеров: sherpa-onnx-pyannote-segmentation-3-0/model.onnx;
    - эмбеддинги спикеров: nemo_en_titanet_small.onnx.

    Возвращаемый словарь совместим по структуре с PyannoteSpeakerDiarizationModel:
    duration, num_speakers, speakers, segments, exclusive_segments, processing_time.

    Параметры, влияющие на качество определения спикеров:

    cache_dir:
        Каталог, куда кэшируются файлы моделей Hugging Face.

    token:
        Токен Hugging Face. Для публичных моделей обычно не обязателен, но полезен
        для увеличения лимитов запросов.

    segmentation_repo_id / segmentation_filename:
        Репозиторий Hugging Face и имя файла ONNX-модели сегментации спикеров.
        Обычно лучше оставлять значения по умолчанию.

    embedding_repo_id / embedding_filename:
        Репозиторий Hugging Face и имя файла ONNX-модели эмбеддингов спикеров.
        Обычно лучше оставлять значения по умолчанию.

    cluster_threshold:
        Порог кластеризации, который используется, когда num_speakers не задан.
        Это главный параметр, если модель находит слишком много или слишком мало спикеров.
        Меньшие значения обычно создают больше спикеров. Большие значения обычно
        сильнее объединяют похожие голоса.
        Если один реальный спикер разбивается на двух, увеличивайте значение, например:
        0.5 -> 0.6 -> 0.7.
        Если разные реальные спикеры сливаются в одного, уменьшайте значение, например:
        0.5 -> 0.4 -> 0.3.
        Значение по умолчанию: 1.0.

    provider:
        Провайдер ONNX Runtime, который использует sherpa-onnx.
        Поддерживаемые значения в этой обертке: "cpu", "cuda", "coreml".
        Используйте "cpu" для предсказуемой обработки на процессоре.
        Используйте "cuda" только если установленная сборка sherpa-onnx/ONNX Runtime
        поддерживает CUDA.
        Значение по умолчанию: "cpu".

    num_threads:
        Количество CPU-потоков для моделей сегментации и эмбеддингов.
        В основном влияет на provider="cpu". При provider="cuda" может влиять
        на вспомогательную CPU-часть пайплайна.
        Для CPU обычно стоит начать с 2-4 потоков. Слишком большое значение может
        замедлить весь сервис из-за конкуренции потоков.
        Значение по умолчанию: 1.

    min_duration_on:
        Минимальная длительность речевого сегмента в секундах.
        Увеличивайте значение, чтобы удалить очень короткие фрагменты/шум.
        Если diarization создает много мелких сегментов, попробуйте 0.4-0.8.
        Если короткие фразы пропадают, уменьшайте значение.
        Значение по умолчанию: 0.4.

    min_duration_off:
        Минимальная длительность паузы/разрыва между речевыми участками в секундах.
        Увеличивайте значение, чтобы агрессивнее объединять близкие речевые участки.
        Уменьшайте значение, если отдельные фразы склеиваются слишком сильно.
        Значение по умолчанию: 1.0.

    merge_gap:
        Постобработка: максимальный разрыв в секундах между соседними сегментами
        одного и того же спикера, при котором они объединяются.
        Если один и тот же спикер разбивается на много соседних сегментов,
        увеличивайте значение, например: 0.2 -> 0.5 -> 1.0.
        Значение по умолчанию: 0.2.

    min_segment_duration:
        Постобработка: удаляет финальные сегменты короче указанного значения.
        Увеличивайте значение, чтобы убрать короткие ложные переключения спикеров.
        Будьте осторожны: слишком большое значение может удалить реальные короткие ответы.
        Значение по умолчанию: 0.3.

    num_speakers в diarize():
        Если количество спикеров известно, передавайте его явно.
        Для аудио с одним спикером используйте diarize(..., num_speakers=1).
        Это самый надежный способ убрать ложных дополнительных спикеров.
        Если num_speakers=None, кластеризация использует cluster_threshold.

    """

    def __init__(
        self,
        cache_dir: str | Path = "cache_dir",
        token: str | None = None,
        segmentation_repo_id: str = DEFAULT_SEGMENTATION_REPO_ID,
        segmentation_filename: str = DEFAULT_SEGMENTATION_FILENAME,
        embedding_repo_id: str = DEFAULT_EMBEDDING_REPO_ID,
        embedding_filename: str = DEFAULT_EMBEDDING_FILENAME,
        cluster_threshold: float = 1.0,
        provider: SherpaOnnxProvider = "cpu",
        num_threads: int = 1,
        min_duration_on: float = 0.4,
        min_duration_off: float = 0.4,
        merge_gap: float = 0.2,
        min_segment_duration: float = 0.3,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = "sherpa-onnx-speaker-diarization"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.segmentation_repo_id = segmentation_repo_id
        self.segmentation_filename = segmentation_filename
        self.embedding_repo_id = embedding_repo_id
        self.embedding_filename = embedding_filename
        self.cluster_threshold = cluster_threshold
        self.provider = provider
        self.num_threads = num_threads
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.merge_gap = merge_gap
        self.min_segment_duration = min_segment_duration
        self.local_files_only = local_files_only
        self.logger = get_logger("diarization.sherpa_onnx")

        self._validate_static_parameters()
        self.segmentation_model = self._download_model_file(
            repo_id=self.segmentation_repo_id,
            filename=self.segmentation_filename,
        )
        self.embedding_model = self._download_model_file(
            repo_id=self.embedding_repo_id,
            filename=self.embedding_filename,
        )
        self._diarizer_by_num_speakers: dict[int, sherpa_onnx.OfflineSpeakerDiarization] = {}

        self.logger.info(
            "Модели Sherpa-ONNX diarization готовы | segmentation=%s | embedding=%s",
            self.segmentation_model,
            self.embedding_model,
        )

    def _validate_static_parameters(self) -> None:
        if not 0.0 < self.cluster_threshold <= 1.0:
            raise ValueError("cluster_threshold должен быть в диапазоне (0, 1]")
        if self.provider not in {"cpu", "cuda", "coreml"}:
            raise ValueError("provider должен быть одним из значений: cpu, cuda, coreml")
        if self.num_threads < 1:
            raise ValueError("num_threads должен быть >= 1")
        if self.min_duration_on < 0:
            raise ValueError("min_duration_on не может быть отрицательным")
        if self.min_duration_off < 0:
            raise ValueError("min_duration_off не может быть отрицательным")
        if self.merge_gap < 0:
            raise ValueError("merge_gap не может быть отрицательным")
        if self.min_segment_duration < 0:
            raise ValueError("min_segment_duration не может быть отрицательным")

    def _download_model_file(self, repo_id: str, filename: str) -> Path:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=self.token,
            cache_dir=str(self.cache_dir),
            local_files_only=self.local_files_only,
        )
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Загруженный файл модели не найден: {model_path}")
        return model_path

    def _create_diarizer(self, num_speakers: int | None = None) -> sherpa_onnx.OfflineSpeakerDiarization:
        num_clusters = self._normalize_num_speakers(num_speakers)
        cached = self._diarizer_by_num_speakers.get(num_clusters)
        if cached is not None:
            return cached

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(self.segmentation_model),
                ),
                num_threads=self.num_threads,
                provider=self.provider,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(self.embedding_model),
                num_threads=self.num_threads,
                provider=self.provider,
            ),
            clustering=sherpa_onnx.FastClusteringConfig(
                num_clusters=num_clusters,
                threshold=self.cluster_threshold,
            ),
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

        if not config.validate():
            raise RuntimeError(
                "Некорректная конфигурация sherpa-onnx diarization. Проверьте пути к моделям и их совместимость."
            )

        diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
        self._diarizer_by_num_speakers[num_clusters] = diarizer
        self.logger.info(
            "Sherpa-ONNX diarizer инициализирован | num_clusters=%s | sample_rate=%s | provider=%s | num_threads=%s",
            num_clusters,
            diarizer.sample_rate,
            self.provider,
            self.num_threads,
        )
        return diarizer

    @staticmethod
    def _normalize_num_speakers(num_speakers: int | None) -> int:
        if num_speakers is None:
            return -1
        if num_speakers <= 0:
            raise ValueError("num_speakers должен быть положительным числом или None")
        return int(num_speakers)

    def _prepare_audio_for_diarization(
        self,
        source: Path,
        sample_rate: int,
    ) -> tuple[np.ndarray, Path]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=source.parent) as tmp:
            prepared_path = Path(tmp.name)

        try:
            command = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-fflags",
                "+discardcorrupt",
                "-err_detect",
                "ignore_err",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_f32le",
                str(prepared_path),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                error = (completed.stderr or "").strip()
                raise RuntimeError(f"ffmpeg завершился с ошибкой code={completed.returncode}: {error}")

            audio, actual_sample_rate = sf.read(
                prepared_path,
                dtype="float32",
                always_2d=False,
            )
            if actual_sample_rate != sample_rate:
                raise RuntimeError(
                    f"Неожиданная частота дискретизации: ожидалось={sample_rate}, получено={actual_sample_rate}"
                )

            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1, dtype=np.float32)
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            if audio.size == 0:
                raise RuntimeError("Подготовленное аудио пустое")

            return audio, prepared_path
        except Exception:
            prepared_path.unlink(missing_ok=True)
            raise

    def diarize(
        self,
        audio_path: str | Path,
        num_speakers: int | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        source = Path(audio_path)
        if not source.exists():
            raise FileNotFoundError(f"Аудиофайл не найден: {source}")

        started_at = time.perf_counter()
        diarizer = self._create_diarizer(num_speakers=num_speakers)
        audio, prepared_path = self._prepare_audio_for_diarization(
            source=source,
            sample_rate=int(diarizer.sample_rate),
        )

        self.logger.info(
            "Sherpa-ONNX diarization запущен | файл=%s | num_speakers=%s | sample_rate=%s | provider=%s",
            source,
            num_speakers,
            diarizer.sample_rate,
            self.provider,
        )

        def callback(processed_chunks: int, total_chunks: int) -> int:
            if on_progress and total_chunks > 0:
                progress = max(0.0, min(100.0, processed_chunks / total_chunks * 100.0))
                on_progress(progress)
            return 0

        try:
            raw_result = diarizer.process(audio, callback=callback).sort_by_start_time()
        finally:
            prepared_path.unlink(missing_ok=True)

        segments = self._normalize_segments(raw_result)
        # segments = self._merge_adjacent_segments(segments)
        # segments = self._remove_short_segments(segments)
        segments = self._renumber_speakers(segments)

        speakers = sorted({segment.speaker for segment in segments})
        duration = max((segment.end for segment in segments), default=0.0)
        processing_time = time.perf_counter() - started_at

        self.logger.info(
            "Sherpa-ONNX diarization завершен | файл=%s | спикеров=%s | сегментов=%s | длительность=%.2fs | обработка=%.2fs",
            source,
            len(speakers),
            len(segments),
            duration,
            processing_time,
        )

        return {
            "duration": duration,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "segments": [segment.to_dict() for segment in segments],
            "exclusive_segments": [],
            "processing_time": processing_time,
        }

    @staticmethod
    def _normalize_segments(raw_result: Any) -> list[SherpaDiarizationSegment]:
        segments: list[SherpaDiarizationSegment] = []
        for item in raw_result:
            start = float(item.start)
            end = float(item.end)
            if end <= start:
                continue

            speaker_id = int(item.speaker)
            segments.append(
                SherpaDiarizationSegment(
                    start=start,
                    end=end,
                    speaker_id=speaker_id,
                    speaker=f"SPEAKER_{speaker_id:02d}",
                )
            )
        return segments

    def _remove_short_segments(
        self,
        segments: Sequence[SherpaDiarizationSegment],
    ) -> list[SherpaDiarizationSegment]:
        return [
            segment
            for segment in segments
            if segment.duration >= self.min_segment_duration
        ]

    @staticmethod
    def _renumber_speakers(
        segments: Sequence[SherpaDiarizationSegment],
    ) -> list[SherpaDiarizationSegment]:
        speaker_id_map: dict[int, int] = {}
        renumbered: list[SherpaDiarizationSegment] = []

        for segment in sorted(segments, key=lambda item: (item.start, item.end)):
            if segment.speaker_id not in speaker_id_map:
                speaker_id_map[segment.speaker_id] = len(speaker_id_map) + 1

            new_speaker_id = speaker_id_map[segment.speaker_id]
            renumbered.append(
                SherpaDiarizationSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker_id=new_speaker_id,
                    speaker=f"SPEAKER_{new_speaker_id:02d}",
                )
            )

        return renumbered

    def _merge_adjacent_segments(
        self,
        segments: Sequence[SherpaDiarizationSegment],
    ) -> list[SherpaDiarizationSegment]:
        if not segments:
            return []

        sorted_segments = sorted(segments, key=lambda item: (item.start, item.end))
        merged: list[SherpaDiarizationSegment] = [
            SherpaDiarizationSegment(
                start=sorted_segments[0].start,
                end=sorted_segments[0].end,
                speaker_id=sorted_segments[0].speaker_id,
                speaker=sorted_segments[0].speaker,
            )
        ]

        for current in sorted_segments[1:]:
            previous = merged[-1]
            gap = current.start - previous.end
            if current.speaker_id == previous.speaker_id and gap <= self.merge_gap:
                previous.end = max(previous.end, current.end)
            else:
                merged.append(
                    SherpaDiarizationSegment(
                        start=current.start,
                        end=current.end,
                        speaker_id=current.speaker_id,
                        speaker=current.speaker,
                    )
                )

        return merged
