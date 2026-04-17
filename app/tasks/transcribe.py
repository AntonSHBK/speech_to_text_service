from pathlib import Path
import json
import torchaudio
import subprocess

from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.models.catalog import ModelDiarizationType, ModelTranscribeSize, resolve_model_name
from app.service.source_downloader import download_source
from app.service.queue_tracker import mark_task_started
from app.service.transcriber import transcriber_service
from app.settings import settings
from app.utils.export import ExportFormat
from app.utils.logging import get_logger

logger = get_logger("worker.init")

TRANSCRIPTION_WEIGHT = 0.50
DIARIZATION_WEIGHT = 0.50


def _probe_media_duration_seconds(audio_path: Path) -> float | None:
    def _probe_with_ffprobe() -> float | None:
        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            raw = (completed.stdout or "").strip()
            if not raw:
                return None
            duration = float(raw)
            return duration if duration > 0 else None
        except Exception:
            return None

    try:
        if hasattr(torchaudio, "info"):
            info = torchaudio.info(str(audio_path))
            sample_rate = int(getattr(info, "sample_rate", 0) or 0)
            num_frames = int(getattr(info, "num_frames", 0) or 0)
        else:
            waveform, sample_rate = torchaudio.load(str(audio_path))
            sample_rate = int(sample_rate or 0)
            num_frames = int(waveform.shape[-1]) if waveform is not None else 0

        if sample_rate <= 0 or num_frames <= 0:
            raise RuntimeError("torchaudio вернул пустые метаданные")

        duration = float(num_frames) / float(sample_rate)
        if duration <= 0:
            raise RuntimeError("torchaudio вернул неположительную длительность")
        return duration
    except Exception as exc:
        fallback_duration = _probe_with_ffprobe()
        if fallback_duration is not None:
            logger.info(
                "Длительность определена через ffprobe (fallback): %.3f c | файл=%s",
                fallback_duration,
                audio_path,
            )
            return fallback_duration

        logger.warning(
            "Не удалось определить длительность медиа через torchaudio/ffprobe: %s | файл=%s",
            exc,
            audio_path,
        )
        return None


def _select_model_by_duration(
    model: ModelTranscribeSize,
    audio_path: Path,
    duration: float | None = None,
) -> ModelTranscribeSize:
    if duration is None:
        duration = _probe_media_duration_seconds(audio_path)
    if duration is None:
        logger.info(
            "Автовыбор модели пропущен (длительность не определена), используем модель из запроса: %s",
            model,
        )
        return model

    # Временная логика:
    # < 1 минуты -> small
    # 1 минута .. 1 час -> medium
    # > 1 часа -> small
    if duration < 60:
        selected_model: ModelTranscribeSize = "small"
    elif duration <= 3600:
        selected_model = "medium"
    else:
        selected_model = "small"

    logger.info(
        "Автовыбор модели по длительности | файл=%s | длительность=%.2fs | модель_запроса=%s | модель_выбрана=%s",
        audio_path.name,
        duration,
        model,
        selected_model,
    )
    return selected_model


@worker_process_init.connect
def init_transcriber_worker(**kwargs):
    default_model_key: ModelTranscribeSize = "medium"
    default_model_name = resolve_model_name(default_model_key)
    compute_type = settings.get_model_compute_type(default_model_key)
    logger.info("Начинается инициализация модели по умолчанию: %s", default_model_name)
    transcriber_service.init(
        model_name=default_model_name,
        device=settings.DEVICE,
        cache_dir=settings.CACHE_DIR,
        token=settings.HF_TOKEN,
        compute_type=compute_type,
        cpu_threads=settings.MODEL_CPU_THREADS,
        num_workers=settings.MODEL_NUM_WORKERS,
    )
    logger.info("Инициализация модели по умолчанию завершена: %s", default_model_name)
    logger.info("Другие модели будут загружаться по требованию задачи.")


@celery_app.task(name="transcribe.process")
def process_transcription(
    audio_path: str | None = None,
    source_filename: str | None = None,
    source_url: str | None = None,
    model: ModelTranscribeSize = "medium",
    language: str | None = None,
    task: str = "transcribe",
    log_progress: bool = False,
    beam_size: int = 3,
    best_of: int = 3,
    patience: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    temperature: list[float] | float = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: float | None = 2.4,
    log_prob_threshold: float | None = -1.0,
    no_speech_threshold: float | None = 0.6,
    condition_on_previous_text: bool = True,
    prompt_reset_on_temperature: float = 0.5,
    initial_prompt: str | None = None,
    prefix: str | None = None,
    suppress_blank: bool = True,
    suppress_tokens: list[int] | None = None,
    without_timestamps: bool = False,
    max_initial_timestamp: float = 1.0,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    multilingual: bool = False,
    vad_filter: bool = False,
    vad_parameters: str | None = None,
    max_new_tokens: int | None = None,
    chunk_length: int | None = None,
    clip_timestamps: str = "0",
    hallucination_silence_threshold: float | None = None,
    hotwords: str | None = None,
    language_detection_threshold: float | None = 0.5,
    language_detection_segments: int = 1,
    diarization: bool = False,
    diarization_model: ModelDiarizationType = "pyannote_1",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    result_format: ExportFormat = "docx",
    export_timestamps: bool = False,
    save_source: bool = False,
    save_result: bool = True,
) -> dict:
    input_path = Path(audio_path) if audio_path else None
    result_file = None
    media_duration_sec: float | None = None
    selected_model: ModelTranscribeSize = model
    mark_task_started(process_transcription.request.id)
    stage_progress: dict[str, float | None] = {
        "transcription": 0.0,
        "diarization": 0.0 if diarization else None,
    }

    def _emit_progress() -> None:
        transcription_progress = float(stage_progress.get("transcription") or 0.0)
        diarization_progress_raw = stage_progress.get("diarization")
        diarization_progress = float(diarization_progress_raw or 0.0)

        if diarization:
            overall = (
                transcription_progress * TRANSCRIPTION_WEIGHT
                + diarization_progress * DIARIZATION_WEIGHT
            )
        else:
            overall = transcription_progress

        process_transcription.update_state(
            state="PROGRESS",
            meta={
                "progress": round(overall, 1),
                "progress_overall": round(overall, 1),
                "progress_transcription": round(transcription_progress, 1),
                "progress_diarization": (
                    round(diarization_progress, 1)
                    if diarization
                    else None
                ),
                "media_duration_sec": media_duration_sec,
            },
        )

    _emit_progress()

    parsed_vad_parameters = None
    if vad_parameters:
        try:
            parsed_vad_parameters = json.loads(vad_parameters)
        except json.JSONDecodeError as exc:
            raise ValueError("Некорректный JSON в vad_parameters") from exc

    try:
        if source_url:
            logger.info("Скачивание source_url начато: %s", source_url)
            input_path = download_source(source_url)
            logger.info("Скачивание source_url завершено: %s", input_path)

        if not input_path:
            raise ValueError("Не передан источник аудио: audio_path или source_url")

        source_filename = source_filename or input_path.name

        media_duration_sec = _probe_media_duration_seconds(input_path)
        _emit_progress()
        selected_model = _select_model_by_duration(
            model=model,
            audio_path=input_path,
            duration=media_duration_sec,
        )
        
        compute_type = settings.get_model_compute_type(selected_model)
        resolved_model_name = resolve_model_name(selected_model)

        transcriber = transcriber_service.get_or_init(
            model_name=resolved_model_name,
            device=settings.DEVICE,
            cache_dir=settings.CACHE_DIR,
            token=settings.HF_TOKEN,
            compute_type=compute_type,
            cpu_threads=settings.MODEL_CPU_THREADS,
            num_workers=settings.MODEL_NUM_WORKERS,
        )

        def _transcription_progress(progress: float):
            stage_progress["transcription"] = max(0.0, min(100.0, float(progress)))
            _emit_progress()

        result = transcriber.transcribe(
            input_path,
            language=language,
            task=task,
            log_progress=log_progress,
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=suppress_tokens,
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            vad_filter=vad_filter,
            vad_parameters=parsed_vad_parameters,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
            language_detection_threshold=language_detection_threshold,
            language_detection_segments=language_detection_segments,
            on_progress=_transcription_progress,
        )
        result_duration = result.get("duration")
        if isinstance(result_duration, (int, float)):
            media_duration_sec = float(result_duration)
            result["media_duration_sec"] = media_duration_sec
        stage_progress["transcription"] = 100.0
        _emit_progress()

        if diarization:
            from app.service.speaker_diarization import diary_service
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            def _diarization_progress(progress: float):
                stage_progress["diarization"] = max(0.0, min(100.0, float(progress)))
                _emit_progress()

            with ProgressHook() as active_hook:
                diarization_result = diary_service.diarize(
                    audio_path=input_path,
                    model=diarization_model,
                    hook=active_hook,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    on_progress=_diarization_progress,
                )

            result["diarization"] = diarization_result
            stage_progress["diarization"] = 100.0
            _emit_progress()

        if save_result:
            result_file = transcriber_service.export_result(
                result=result,
                source_filename=source_filename,
                format=result_format,
                export_timestamps=export_timestamps,
            )
            result["result_file"] = str(result_file)
            result["result_filename"] = result_file.name
            result["download_url"] = f"/transcribe/get_result/{result_file.name}"
        else:
            result["result_file"] = None
            result["result_filename"] = None
            result["download_url"] = None

        return result
    finally:
        if input_path and not save_source and input_path.exists():
            input_path.unlink(missing_ok=True)
