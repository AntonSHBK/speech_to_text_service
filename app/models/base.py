
from pathlib import Path
from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):
    """Базовый класс для моделей транскрипции."""
    
    def __init__(self, model_name: str, cache_dir: Path = ''):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.cache_dir = cache_dir

    @abstractmethod
    def load_model(self):
        """Загрузить модель."""
        pass

    @abstractmethod
    def transcribe_audio(self, audio_path: str) -> str:
        """Транскрибировать аудиофайл."""
        pass