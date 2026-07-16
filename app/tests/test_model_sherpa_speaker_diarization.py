import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

pytest.importorskip("sherpa_onnx")
pytest.importorskip("soundfile")

from app.models.sherpa_speaker_diarization import SherpaOnnxSpeakerDiarizationModel
from app.settings import settings

TEST_AUDIO_FILE = settings.AUDIO_DIR / "test_video_3.mp4"
SEGMENTATION_REPO_ID = "test/segmentation"
SEGMENTATION_FILENAME = "model.onnx"
EMBEDDING_REPO_ID = "test/embedding"
EMBEDDING_FILENAME = "nemo_en_titanet_small.onnx"


class FakeSherpaResult(list):
    def sort_by_start_time(self):
        return FakeSherpaResult(sorted(self, key=lambda item: item.start))


class FakeSherpaDiarizer:
    def __init__(self, config):
        self.config = config
        self.sample_rate = 16000
        self.calls = []

    def process(self, audio, callback=None):
        self.calls.append((audio, callback))
        if callback is not None:
            callback(1, 4)
            callback(4, 4)
        return FakeSherpaResult(
            [
                SimpleNamespace(start=1.5, end=3.0, speaker=1),
                SimpleNamespace(start=0.0, end=1.0, speaker=0),
                SimpleNamespace(start=1.05, end=1.4, speaker=0),
            ]
        )


@pytest.fixture
def fake_model_files(tmp_path, monkeypatch):
    segmentation_model = tmp_path / "segmentation" / SEGMENTATION_FILENAME
    embedding_model = tmp_path / "embedding" / EMBEDDING_FILENAME
    segmentation_model.parent.mkdir(parents=True)
    embedding_model.parent.mkdir(parents=True)
    segmentation_model.write_bytes(b"fake segmentation onnx")
    embedding_model.write_bytes(b"fake embedding onnx")

    def fake_hf_hub_download(
        repo_id,
        filename,
        token=None,
        cache_dir=None,
        local_files_only=False,
    ):
        assert token == "test-token"
        assert cache_dir == str(tmp_path)
        if repo_id == SEGMENTATION_REPO_ID and filename == SEGMENTATION_FILENAME:
            return str(segmentation_model)
        if repo_id == EMBEDDING_REPO_ID and filename == EMBEDDING_FILENAME:
            return str(embedding_model)
        raise AssertionError(f"Unexpected model request: {repo_id}/{filename}")

    monkeypatch.setattr(
        "app.models.sherpa_speaker_diarization.hf_hub_download",
        fake_hf_hub_download,
    )
    return tmp_path, segmentation_model, embedding_model


@pytest.fixture
def fake_sherpa(monkeypatch):
    created = []

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def validate(self):
            return True

    def fake_diarization_factory(config):
        diarizer = FakeSherpaDiarizer(config)
        created.append(diarizer)
        return diarizer

    monkeypatch.setattr(
        "app.models.sherpa_speaker_diarization.sherpa_onnx.OfflineSpeakerDiarizationConfig",
        FakeConfig,
    )
    monkeypatch.setattr(
        "app.models.sherpa_speaker_diarization.sherpa_onnx.OfflineSpeakerSegmentationModelConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "app.models.sherpa_speaker_diarization.sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "app.models.sherpa_speaker_diarization.sherpa_onnx.SpeakerEmbeddingExtractorConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "app.models.sherpa_speaker_diarization.sherpa_onnx.FastClusteringConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "app.models.sherpa_speaker_diarization.sherpa_onnx.OfflineSpeakerDiarization",
        fake_diarization_factory,
    )

    return created


@pytest.mark.order(1)
def test_model_downloads_files_and_initializes(fake_model_files, fake_sherpa):
    cache_dir, segmentation_model, embedding_model = fake_model_files

    diarizer = SherpaOnnxSpeakerDiarizationModel(
        cache_dir=cache_dir,
        token="test-token",
        segmentation_repo_id=SEGMENTATION_REPO_ID,
        segmentation_filename=SEGMENTATION_FILENAME,
        embedding_repo_id=EMBEDDING_REPO_ID,
        embedding_filename=EMBEDDING_FILENAME,
        provider="cpu",
        num_threads=2,
    )

    assert diarizer.segmentation_model == segmentation_model
    assert diarizer.embedding_model == embedding_model
    assert diarizer.cache_dir == cache_dir
    assert diarizer.provider == "cpu"
    assert diarizer.num_threads == 2


@pytest.mark.order(2)
def test_diarization_returns_expected_result(tmp_path, monkeypatch, fake_model_files, fake_sherpa):
    cache_dir, _, _ = fake_model_files
    source_path = tmp_path / "audio.mp3"
    source_path.write_bytes(b"fake audio")
    prepared_path = tmp_path / "prepared.wav"
    prepared_path.write_bytes(b"fake prepared audio")

    prepared_audio = np.zeros(16000, dtype=np.float32)

    def fake_prepare(self, source: Path, sample_rate: int):
        assert source == source_path
        assert sample_rate == 16000
        return prepared_audio, prepared_path

    monkeypatch.setattr(
        SherpaOnnxSpeakerDiarizationModel,
        "_prepare_audio_for_diarization",
        fake_prepare,
    )

    progress_values = []
    diarizer = SherpaOnnxSpeakerDiarizationModel(
        cache_dir=cache_dir,
        token="test-token",
        segmentation_repo_id=SEGMENTATION_REPO_ID,
        segmentation_filename=SEGMENTATION_FILENAME,
        embedding_repo_id=EMBEDDING_REPO_ID,
        embedding_filename=EMBEDDING_FILENAME,
        merge_gap=0.2,
        provider="cpu",
        num_threads=3,
    )

    result = diarizer.diarize(
        audio_path=source_path,
        num_speakers=2,
        on_progress=progress_values.append,
    )

    assert fake_sherpa, "Sherpa diarizer was not created"
    config_kwargs = fake_sherpa[0].config.kwargs
    assert config_kwargs["segmentation"].provider == "cpu"
    assert config_kwargs["segmentation"].num_threads == 3
    assert config_kwargs["embedding"].provider == "cpu"
    assert config_kwargs["embedding"].num_threads == 3
    assert fake_sherpa[0].calls, "Sherpa diarizer process was not called"
    called_audio, called_callback = fake_sherpa[0].calls[0]
    assert called_audio is prepared_audio
    assert called_callback is not None
    assert progress_values == [25.0, 100.0]

    assert result["backend"] == "sherpa-onnx"
    assert result["duration"] == 3.0
    assert result["num_speakers"] == 2
    assert result["speakers"] == ["SPEAKER_01", "SPEAKER_02"]
    assert result["exclusive_segments"] == []
    assert len(result["segments"]) == 2
    assert result["segments"][0] == {
        "start": 0.0,
        "end": 1.4,
        "speaker_id": 1,
        "speaker": "SPEAKER_01",
        "duration": 1.4,
    }
    assert isinstance(result["processing_time"], float)
    assert not prepared_path.exists(), "Temporary prepared audio was not deleted"


@pytest.mark.order(3)
def test_diarization_raises_for_missing_audio(fake_model_files, fake_sherpa):
    cache_dir, _, _ = fake_model_files
    diarizer = SherpaOnnxSpeakerDiarizationModel(
        cache_dir=cache_dir,
        token="test-token",
        segmentation_repo_id=SEGMENTATION_REPO_ID,
        segmentation_filename=SEGMENTATION_FILENAME,
        embedding_repo_id=EMBEDDING_REPO_ID,
        embedding_filename=EMBEDDING_FILENAME,
    )

    with pytest.raises(FileNotFoundError):
        diarizer.diarize(settings.AUDIO_DIR / "missing.wav")


@pytest.mark.order(4)
def test_real_sherpa_diarization():
    """Run real sherpa-onnx diarization. Enable with RUN_REAL_SHERPA_DIARIZATION_TEST=1."""

    if os.getenv("RUN_REAL_SHERPA_DIARIZATION_TEST") != "1":
        pytest.skip("Set RUN_REAL_SHERPA_DIARIZATION_TEST=1 to run real sherpa-onnx test")

    assert TEST_AUDIO_FILE.exists(), f"Test audio file not found: {TEST_AUDIO_FILE}"

    diarizer = SherpaOnnxSpeakerDiarizationModel(
        cache_dir=settings.CACHE_DIR,
        token=settings.HF_TOKEN,
        provider="cpu",
    )
    progress_values = []
    result = diarizer.diarize(
        audio_path=TEST_AUDIO_FILE,
        num_speakers=2,
        on_progress=progress_values.append,
    )

    assert isinstance(result, dict)
    assert result["backend"] == "sherpa-onnx"
    assert "segments" in result
    assert isinstance(result["segments"], list)

    print("\n===== SHERPA-ONNX DIARIZATION RESULT =====")
    print(result)
    print(f"progress_events={len(progress_values)}")
    print("==========================================")
