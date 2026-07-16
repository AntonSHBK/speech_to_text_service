import gc
import os
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

pytest.importorskip("soundfile")
pytest.importorskip("pyannote.audio")

from app.models.pyannote_speaker_diarization import PyannoteSpeakerDiarizationModel
from app.settings import settings

TEST_AUDIO_FILE = settings.AUDIO_DIR / "test_video_3.mp4"
MODEL_NAME = "pyannote/speaker-diarization-community-1"


class FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class FakeDiarization:
    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, yield_label: bool = False):
        assert yield_label is True
        for start, end, speaker in self.tracks:
            yield FakeTurn(start, end), None, speaker


class FakePipelineOutput:
    def __init__(self):
        self.speaker_diarization = FakeDiarization(
            [
                (0.0, 1.5, "SPEAKER_00"),
                (1.5, 3.0, "SPEAKER_01"),
                (3.0, 4.25, "SPEAKER_00"),
            ]
        )
        self.exclusive_speaker_diarization = FakeDiarization(
            [
                (0.0, 1.4, "SPEAKER_00"),
                (1.6, 3.0, "SPEAKER_01"),
            ]
        )


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        hook = kwargs.get("hook")
        if hook is not None:
            hook(completed=1, total=4)
            hook(completed=4, total=4)
        return FakePipelineOutput()


@pytest.fixture
def fake_pipeline(monkeypatch):
    pipeline = FakePipeline()

    def fake_from_pretrained(model_name: str, token: str | None = None):
        assert model_name == MODEL_NAME
        assert token == settings.HF_TOKEN
        return pipeline

    monkeypatch.setattr(
        "app.models.pyannote_speaker_diarization.Pipeline.from_pretrained",
        fake_from_pretrained,
    )
    return pipeline


# @pytest.mark.order(1)
# def test_model_load(fake_pipeline):
#     """Check that the diarization pipeline is loaded correctly."""

#     diarizer = PyannoteSpeakerDiarizationModel(
#         model_name=MODEL_NAME,
#         device="cpu",
#         token=settings.HF_TOKEN,
#     )

#     assert diarizer.pipeline is fake_pipeline, "Diarization pipeline was not loaded"
#     assert fake_pipeline.device is None
#     print("Diarization pipeline loaded successfully")


# @pytest.mark.order(2)
# def test_diarization(monkeypatch, fake_pipeline):
#     """Check speaker diarization result structure on a test audio file."""

#     assert TEST_AUDIO_FILE.exists(), f"Test audio file not found: {TEST_AUDIO_FILE}"

#     prepared_path = settings.AUDIO_DIR / "test_prepared_diarization.wav"
#     prepared_path.write_bytes(b"fake wav")

#     prepared_audio = {
#         "waveform": torch.zeros((1, 16000), dtype=torch.float32),
#         "sample_rate": 16000,
#     }

#     def fake_prepare(self, source: Path):
#         assert source == TEST_AUDIO_FILE
#         return prepared_audio, prepared_path

#     monkeypatch.setattr(
#         PyannoteSpeakerDiarizationModel,
#         "_prepare_audio_for_diarization",
#         fake_prepare,
#     )

#     progress_values = []

#     diarizer = PyannoteSpeakerDiarizationModel(
#         model_name=MODEL_NAME,
#         device="cpu",
#         token=settings.HF_TOKEN,
#     )

#     result = diarizer.diarize(
#         audio_path=TEST_AUDIO_FILE,
#         num_speakers=2,
#         on_progress=progress_values.append,
#     )

#     assert fake_pipeline.calls, "Pipeline was not called"
#     called_audio, called_kwargs = fake_pipeline.calls[0]
#     assert called_audio is prepared_audio
#     assert called_kwargs["num_speakers"] == 2
#     assert "hook" in called_kwargs

#     assert progress_values == [25.0, 100.0]
#     assert isinstance(result, dict), "Result is not a dictionary"
#     assert result["duration"] == 4.25
#     assert result["num_speakers"] == 2
#     assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
#     assert isinstance(result["segments"], list)
#     assert len(result["segments"]) == 3
#     assert isinstance(result["exclusive_segments"], list)
#     assert len(result["exclusive_segments"]) == 2

#     first = result["segments"][0]
#     for key in ("speaker", "start", "end", "duration"):
#         assert key in first, f"Segment does not contain field '{key}'"

#     assert not prepared_path.exists(), "Temporary prepared audio was not deleted"

#     print("\n===== DIARIZATION RESULT =====")
#     print(result)
#     print("==============================")



# @pytest.mark.order(3)
# def test_diarization_peak_gpu_memory_real():
#     """Measure real peak GPU memory for pyannote diarization.

#     This is an integration/performance test: it loads the real pyannote pipeline,
#     runs ffmpeg audio preparation, and requires CUDA + HF access.
#     Run explicitly with RUN_REAL_DIARIZATION_GPU_TEST=1.
#     """

#     if os.getenv("RUN_REAL_DIARIZATION_GPU_TEST") != "1":
#         pytest.skip("Set RUN_REAL_DIARIZATION_GPU_TEST=1 to run real GPU memory test")
#     if not torch.cuda.is_available():
#         pytest.skip("CUDA is not available")

#     assert TEST_AUDIO_FILE.exists(), f"Test audio file not found: {TEST_AUDIO_FILE}"

#     torch.cuda.empty_cache()
#     torch.cuda.reset_peak_memory_stats()
#     before_allocated = torch.cuda.memory_allocated()
#     before_reserved = torch.cuda.memory_reserved()

#     diarizer = None
#     try:
#         diarizer = PyannoteSpeakerDiarizationModel(
#             model_name=MODEL_NAME,
#             device="cuda",
#             token=settings.HF_TOKEN,
#         )

#         result = diarizer.diarize(
#             audio_path=TEST_AUDIO_FILE,
#             num_speakers=2,
#         )

#         torch.cuda.synchronize()
#         peak_allocated = torch.cuda.max_memory_allocated()
#         after_allocated = torch.cuda.memory_allocated()
#         after_reserved = torch.cuda.memory_reserved()

#         peak_used_mb = (peak_allocated - before_allocated) / 1024 / 1024
#         peak_total_mb = peak_allocated / 1024 / 1024

#         assert isinstance(result, dict)
#         assert "segments" in result
#         assert peak_allocated >= before_allocated

#         print("\n===== REAL DIARIZATION GPU MEMORY =====")
#         print(f"before_allocated_mb={before_allocated / 1024 / 1024:.2f}")
#         print(f"before_reserved_mb={before_reserved / 1024 / 1024:.2f}")
#         print(f"peak_allocated_mb={peak_total_mb:.2f}")
#         print(f"peak_used_by_test_mb={peak_used_mb:.2f}")
#         print(f"after_allocated_mb={after_allocated / 1024 / 1024:.2f}")
#         print(f"after_reserved_mb={after_reserved / 1024 / 1024:.2f}")
#         print("=======================================")
#     finally:
#         del diarizer
#         gc.collect()
#         torch.cuda.empty_cache()
#         if hasattr(torch.cuda, "ipc_collect"):
#             torch.cuda.ipc_collect()




# @pytest.mark.order(4)
# def test_diarization_peak_cpu_memory_real():
#     """Measure real process RAM usage for pyannote diarization on CPU.

#     This is an integration/performance test: it loads the real pyannote pipeline,
#     runs ffmpeg audio preparation, and can be slow.
#     Run explicitly with RUN_REAL_DIARIZATION_CPU_TEST=1.
#     """

#     if os.getenv("RUN_REAL_DIARIZATION_CPU_TEST") != "1":
#         pytest.skip("Set RUN_REAL_DIARIZATION_CPU_TEST=1 to run real CPU memory test")

#     psutil = pytest.importorskip("psutil")

#     assert TEST_AUDIO_FILE.exists(), f"Test audio file not found: {TEST_AUDIO_FILE}"

#     process = psutil.Process(os.getpid())
#     gc.collect()

#     before_rss = process.memory_info().rss
#     peak_rss = before_rss
#     started_at = time.perf_counter()

#     diarizer = None
#     try:
#         diarizer = PyannoteSpeakerDiarizationModel(
#             model_name=MODEL_NAME,
#             device="cpu",
#             token=settings.HF_TOKEN,
#         )
#         after_load_rss = process.memory_info().rss
#         peak_rss = max(peak_rss, after_load_rss)

#         progress_values = []

#         def on_progress(progress: float):
#             nonlocal peak_rss
#             peak_rss = max(peak_rss, process.memory_info().rss)
#             progress_values.append(progress)

#         result = diarizer.diarize(
#             audio_path=TEST_AUDIO_FILE,
#             num_speakers=2,
#             on_progress=on_progress,
#         )

#         after_run_rss = process.memory_info().rss
#         peak_rss = max(peak_rss, after_run_rss)
#         elapsed_sec = time.perf_counter() - started_at

#         assert isinstance(result, dict)
#         assert "segments" in result

#         print("\n===== REAL DIARIZATION CPU MEMORY =====")
#         print(f"before_rss_mb={before_rss / 1024 / 1024:.2f}")
#         print(f"after_load_rss_mb={after_load_rss / 1024 / 1024:.2f}")
#         print(f"after_run_rss_mb={after_run_rss / 1024 / 1024:.2f}")
#         print(f"peak_observed_rss_mb={peak_rss / 1024 / 1024:.2f}")
#         print(f"peak_used_by_test_mb={(peak_rss - before_rss) / 1024 / 1024:.2f}")
#         print(f"elapsed_sec={elapsed_sec:.2f}")
#         print(f"progress_events={len(progress_values)}")
#         print("=======================================")
#     finally:
#         del diarizer
#         gc.collect()
