from pathlib import Path
import torchaudio
import subprocess

from celery.signals import worker_process_init

from app.celery_app import celery_app
from app.models.catalog import ModelTranscribeSize, resolve_model_name
from app.service.source_downloader import download_source
from app.service.queue_tracker import mark_task_started
from app.service.transcriber import transcriber_service
from app.settings import settings
from app.utils.export import ExportFormat
from app.utils.exporters.common import build_paragraph_blocks
from app.utils.logging import get_logger

logger = get_logger("worker.init")

TRANSCRIPTION_WEIGHT = 0.50
DIARIZATION_WEIGHT = 0.50
GPU_RETRY_COUNTDOWN_SEC = 60
GPU_MAX_RETRIES = 3

# Параметры faster-whisper, общие для всех задач транскрибации.
TRANSCRIPTION_TASK = "transcribe"
TRANSCRIPTION_LOG_PROGRESS = False
TRANSCRIPTION_BEAM_SIZE = 3
TRANSCRIPTION_BEST_OF = 3
TRANSCRIPTION_PATIENCE = 1.0
TRANSCRIPTION_LENGTH_PENALTY = 1.0
TRANSCRIPTION_REPETITION_PENALTY = 1.0
TRANSCRIPTION_NO_REPEAT_NGRAM_SIZE = 0
TRANSCRIPTION_TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TRANSCRIPTION_COMPRESSION_RATIO_THRESHOLD = 2.4
TRANSCRIPTION_LOG_PROB_THRESHOLD = -1.0
TRANSCRIPTION_NO_SPEECH_THRESHOLD = 0.6
TRANSCRIPTION_CONDITION_ON_PREVIOUS_TEXT = False
TRANSCRIPTION_PROMPT_RESET_ON_TEMPERATURE = 0.5
TRANSCRIPTION_INITIAL_PROMPT = (
    "Расставляй знаки препинания: точки, запятые, "
    "вопросительные и восклицательные знаки. "
    "Разделяй текст на законченные предложения."
)
TRANSCRIPTION_PREFIX = None
TRANSCRIPTION_SUPPRESS_BLANK = True
TRANSCRIPTION_SUPPRESS_TOKENS = [-1]
TRANSCRIPTION_WITHOUT_TIMESTAMPS = False
TRANSCRIPTION_MAX_INITIAL_TIMESTAMP = 1.0
TRANSCRIPTION_WORD_TIMESTAMPS = False
TRANSCRIPTION_PREPEND_PUNCTUATIONS = "\"'“¿([{-"
TRANSCRIPTION_APPEND_PUNCTUATIONS = "\"'.。,，!！?？:：”)]}、"
TRANSCRIPTION_MULTILINGUAL = False
TRANSCRIPTION_VAD_FILTER = True
TRANSCRIPTION_VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
}
TRANSCRIPTION_MAX_NEW_TOKENS = None
TRANSCRIPTION_CHUNK_LENGTH = None
TRANSCRIPTION_CLIP_TIMESTAMPS = "0"
TRANSCRIPTION_HALLUCINATION_SILENCE_THRESHOLD = None
TRANSCRIPTION_HOTWORDS = None
TRANSCRIPTION_LANGUAGE_DETECTION_THRESHOLD = 0.5
TRANSCRIPTION_LANGUAGE_DETECTION_SEGMENTS = 1

# Параметры Sherpa-ONNX diarization, общие для всех задач.
DIARIZATION_NUM_THREADS = 2
DIARIZATION_CLUSTER_THRESHOLD = 1.0
DIARIZATION_MIN_DURATION_ON = 0.4
DIARIZATION_MIN_DURATION_OFF = 0.4
DIARIZATION_MERGE_GAP = 0.2
DIARIZATION_MIN_SEGMENT_DURATION = 0.3


def _is_retryable_gpu_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_markers = (
        "cuda",
        "cublas",
        "cudnn",
        "out of memory",
        "invalid device ordinal",
    )
    return any(marker in message for marker in retryable_markers)


def _release_models_after_failure() -> None:
    try:
        transcriber_service.release()
    except Exception as release_exc:
        logger.warning("Не удалось выгрузить модель transcriber после ошибки: %s", release_exc)

    try:
        from app.service.speaker_diarization import diary_service
        diary_service.release()
    except Exception as release_exc:
        logger.warning("Не удалось выгрузить модель diarization после ошибки: %s", release_exc)


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
        selected_model: ModelTranscribeSize = "medium"
    elif duration <= 3600:
        selected_model = "large"
    else:
        selected_model = "medium"

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
    if settings.RELEASE_MODELS_ON_IDLE:
        logger.info("Предзагрузка моделей отключена. Модели будут загружаться по требованию задачи.")
        return

    default_model_key: ModelTranscribeSize = "large"
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
        num_workers=settings.MODEL_NUM_CPU_WORKERS,
        local_files_only=settings.MODEL_LOCAL_FILES_ONLY,
    )
    logger.info("Инициализация модели по умолчанию завершена: %s", default_model_name)
    logger.info("Модели будут сохраняться в памяти между задачами.")


@celery_app.task(bind=True, name="transcribe.process")
def process_transcription(
    self,
    audio_path: str | None = None,
    source_filename: str | None = None,
    source_url: str | None = None,
    model: ModelTranscribeSize = "large",
    language: str | None = None,
    diarization: bool = False,
    num_speakers: int | None = None,
    result_format: ExportFormat = "docx",
    export_timestamps: bool = False,
    save_source: bool = False,
    save_result: bool = True,
) -> dict:

    input_path = Path(audio_path) if audio_path else None
    result_file = None
    media_duration_sec: float | None = None
    selected_model: ModelTranscribeSize = model
    mark_task_started(self.request.id)
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

        self.update_state(
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

    retry_scheduled = False

    try:
        if source_url:
            logger.info("Скачивание source_url начато: %s", source_url)
            input_path = download_source(source_url)
            logger.info("Скачивание source_url завершено: %s", input_path)

        if not input_path:
            raise ValueError("Не передан источник аудио: audio_path или source_url")

        source_filename = source_filename or input_path.name

        # media_duration_sec = _probe_media_duration_seconds(input_path)
        # _emit_progress()
        
        # selected_model = _select_model_by_duration(
        #     model=model,
        #     audio_path=input_path,
        #     duration=media_duration_sec,
        # )
        
        selected_model = model
        
        compute_type = settings.get_model_compute_type(selected_model)
        resolved_model_name = resolve_model_name(selected_model)

        transcriber = transcriber_service.get_or_init(
            model_name=resolved_model_name,
            device=settings.DEVICE,
            cache_dir=settings.CACHE_DIR,
            token=settings.HF_TOKEN,
            compute_type=compute_type,
            cpu_threads=settings.MODEL_CPU_THREADS,
            num_workers=settings.MODEL_NUM_CPU_WORKERS,
            local_files_only=settings.MODEL_LOCAL_FILES_ONLY,
        )

        def _transcription_progress(progress: float):
            stage_progress["transcription"] = max(0.0, min(100.0, float(progress)))
            _emit_progress()

        result = transcriber.transcribe(
            input_path,
            language=language,
            task=TRANSCRIPTION_TASK,
            log_progress=TRANSCRIPTION_LOG_PROGRESS,
            beam_size=TRANSCRIPTION_BEAM_SIZE,
            best_of=TRANSCRIPTION_BEST_OF,
            patience=TRANSCRIPTION_PATIENCE,
            length_penalty=TRANSCRIPTION_LENGTH_PENALTY,
            repetition_penalty=TRANSCRIPTION_REPETITION_PENALTY,
            no_repeat_ngram_size=TRANSCRIPTION_NO_REPEAT_NGRAM_SIZE,
            temperature=TRANSCRIPTION_TEMPERATURE,
            compression_ratio_threshold=TRANSCRIPTION_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=TRANSCRIPTION_LOG_PROB_THRESHOLD,
            no_speech_threshold=TRANSCRIPTION_NO_SPEECH_THRESHOLD,
            condition_on_previous_text=TRANSCRIPTION_CONDITION_ON_PREVIOUS_TEXT,
            prompt_reset_on_temperature=TRANSCRIPTION_PROMPT_RESET_ON_TEMPERATURE,
            initial_prompt=TRANSCRIPTION_INITIAL_PROMPT,
            prefix=TRANSCRIPTION_PREFIX,
            suppress_blank=TRANSCRIPTION_SUPPRESS_BLANK,
            suppress_tokens=TRANSCRIPTION_SUPPRESS_TOKENS,
            without_timestamps=TRANSCRIPTION_WITHOUT_TIMESTAMPS,
            max_initial_timestamp=TRANSCRIPTION_MAX_INITIAL_TIMESTAMP,
            word_timestamps=TRANSCRIPTION_WORD_TIMESTAMPS,
            prepend_punctuations=TRANSCRIPTION_PREPEND_PUNCTUATIONS,
            append_punctuations=TRANSCRIPTION_APPEND_PUNCTUATIONS,
            multilingual=TRANSCRIPTION_MULTILINGUAL,
            vad_filter=TRANSCRIPTION_VAD_FILTER,
            vad_parameters=TRANSCRIPTION_VAD_PARAMETERS,
            max_new_tokens=TRANSCRIPTION_MAX_NEW_TOKENS,
            chunk_length=TRANSCRIPTION_CHUNK_LENGTH,
            clip_timestamps=TRANSCRIPTION_CLIP_TIMESTAMPS,
            hallucination_silence_threshold=TRANSCRIPTION_HALLUCINATION_SILENCE_THRESHOLD,
            hotwords=TRANSCRIPTION_HOTWORDS,
            language_detection_threshold=TRANSCRIPTION_LANGUAGE_DETECTION_THRESHOLD,
            language_detection_segments=TRANSCRIPTION_LANGUAGE_DETECTION_SEGMENTS,
            on_progress=_transcription_progress,
        )
        result_duration = result.get("duration")
        if isinstance(result_duration, (int, float)):
            media_duration_sec = float(result_duration)
            result["media_duration_sec"] = media_duration_sec
        stage_progress["transcription"] = 100.0
        _emit_progress()

        if diarization:
            transcriber_service.release()

            from app.service.speaker_diarization import (
                PyannoteSpeakerDiarizationService,
                SherpaSpeakerDiarizationService,
                diary_service,
            )

            def _diarization_progress(progress: float):
                stage_progress["diarization"] = max(0.0, min(100.0, float(progress)))
                _emit_progress()

            try:
                if isinstance(diary_service, PyannoteSpeakerDiarizationService):
                    from pyannote.audio.pipelines.utils.hook import ProgressHook

                    with ProgressHook() as active_hook:
                        diarization_result = diary_service.diarize(
                            audio_path=input_path,
                            hook=active_hook,
                            num_speakers=num_speakers,
                            on_progress=_diarization_progress,
                        )
                elif isinstance(diary_service, SherpaSpeakerDiarizationService):
                    diarization_result = diary_service.diarize(
                        audio_path=input_path,
                        num_speakers=num_speakers,
                        on_progress=_diarization_progress,
                        provider=settings.DEVICE,
                        cluster_threshold=DIARIZATION_CLUSTER_THRESHOLD,
                        num_threads=DIARIZATION_NUM_THREADS,
                        min_duration_on=DIARIZATION_MIN_DURATION_ON,
                        min_duration_off=DIARIZATION_MIN_DURATION_OFF,
                        merge_gap=DIARIZATION_MERGE_GAP,
                        min_segment_duration=DIARIZATION_MIN_SEGMENT_DURATION,
                    )
                else:
                    raise TypeError(
                        f"Unknown diarization service: {type(diary_service).__name__}"
                    )

                result["diarization"] = diarization_result
                result["diarization_error"] = None
            except Exception as exc:
                logger.exception(
                    "Определение спикеров завершилось с ошибкой | файл=%s",
                    input_path,
                )
                result["diarization"] = None
                result["diarization_error"] = str(exc)
            finally:
                stage_progress["diarization"] = 100.0
                _emit_progress()

        paragraph_blocks = build_paragraph_blocks(result)
        result["text"] = "\n\n".join(
            " ".join(
                str(part.get("text", "")).strip()
                for part in block.get("parts", [])
                if part.get("text")
            )
            for block in paragraph_blocks
            if block.get("parts")
        )

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
    except Exception as exc:
        if _is_retryable_gpu_error(exc):
            _release_models_after_failure()

            if self.request.retries < GPU_MAX_RETRIES:
                retry_scheduled = True
                logger.exception(
                    "GPU/CUDA ошибка при транскрибации, задача будет возвращена в очередь | task_id=%s | retry=%s/%s | countdown=%ss",
                    self.request.id,
                    self.request.retries + 1,
                    GPU_MAX_RETRIES,
                    GPU_RETRY_COUNTDOWN_SEC,
                )
                raise self.retry(
                    exc=exc,
                    countdown=GPU_RETRY_COUNTDOWN_SEC,
                    max_retries=GPU_MAX_RETRIES,
                ) from exc

            logger.exception(
                "GPU/CUDA ошибка при транскрибации, лимит повторов исчерпан | task_id=%s | retries=%s/%s",
                self.request.id,
                self.request.retries,
                GPU_MAX_RETRIES,
            )
        raise
    finally:
        if settings.RELEASE_MODELS_ON_IDLE:
            try:
                transcriber_service.release()
            except Exception as exc:
                logger.warning("Не удалось выгрузить модель transcriber: %s", exc)

            try:
                from app.service.speaker_diarization import diary_service
                diary_service.release()
            except Exception as exc:
                logger.warning("Не удалось выгрузить модель diarization: %s", exc)

        if input_path and not retry_scheduled and not save_source and input_path.exists():
            input_path.unlink(missing_ok=True)
