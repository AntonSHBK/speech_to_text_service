from typing import Literal

ModelTranscribeSize = Literal["small", "medium", "large"]

MODEL_TRANSCRIBER_CATALOG: dict[ModelTranscribeSize, str] = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
}

ModelDiarizationType = Literal["pyannote_1", "pyannote_3_1"]

MODEL_SPEAKER_DIARIZATION_CATALOG: dict[ModelDiarizationType, str] = {
    "pyannote_1": "pyannote/speaker-diarization-community-1",
    "pyannote_3_1": "pyannote/speaker-diarization-3.1",
}   

def resolve_model_name(model: ModelTranscribeSize) -> str:
    return MODEL_TRANSCRIBER_CATALOG[model]

def resolve_speaker_diarization_model_name(model: ModelDiarizationType) -> str:
    return MODEL_SPEAKER_DIARIZATION_CATALOG[model]