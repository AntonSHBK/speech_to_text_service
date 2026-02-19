from typing import Literal

ModelSize = Literal["small", "medium", "large"]

MODEL_CATALOG: dict[ModelSize, str] = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
}


def resolve_model_name(model: ModelSize) -> str:
    return MODEL_CATALOG[model]
