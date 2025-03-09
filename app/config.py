import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "model_name": "openai/whisper-tiny",
    "cache_dir": os.path.join(BASE_DIR, "app/data/cache"),
    "input_dir": os.path.join(BASE_DIR, "app/data/input"),
    "output_dir": os.path.join(BASE_DIR, "app/data/output"),
    "device": "cuda",
    "default_language": "ru",
    "default_task": "transcribe",
}

def init_dir():

    DATA_PATH = Path('data/')
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    DATA_CACHE = Path('data/cache_dir/')
    DATA_CACHE.mkdir(parents=True, exist_ok=True)

    DATA_PATH_SAVE_MODELS = Path('data/models/')
    DATA_PATH_SAVE_MODELS.mkdir(parents=True, exist_ok=True)

    Path('/data/cache_dir').mkdir(parents=True, exist_ok=True)
    Path('/data/cache_dir').mkdir(parents=True, exist_ok=True)
    Path('/data/cache_dir').mkdir(parents=True, exist_ok=True)
