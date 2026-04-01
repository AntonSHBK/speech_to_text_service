import os
import getpass
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download

MODEL_CATALOG: Dict[str, str] = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
}

CACHE_DIR = Path("/app/data/cache_dir")


def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if token:
        return token

    token = getpass.getpass("Введите Hugging Face token: ").strip()
    if not token:
        raise ValueError("Токен не может быть пустым")
    return token


def download_models():
    token = get_hf_token()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for size, repo_id in MODEL_CATALOG.items():
        print(f"Загрузка модели {size}: {repo_id}")
        print(f"cache_dir: {CACHE_DIR.resolve()}")

        try:
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(CACHE_DIR),
                token=token,
            )

            print(f"Скачано в: {path}\n")

        except Exception as e:
            print(f"Ошибка при загрузке {size}: {e}\n")

    print("Все модели обработаны")


if __name__ == "__main__":
    download_models()