
from pathlib import Path
from abc import ABC, abstractmethod

import torch


class BaseModel(ABC):
    """Базовый класс для моделей транскрипции."""
    
    def __init__(
        self, model_name: str, 
        cache_dir: Path = '', 
        device: str = 'cpu',
        **kwargs
    ):
        self.params_dict = kwargs
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Загрузить модель."""
        pass

    @abstractmethod
    def transcribe_audio(self, audio_path: str) -> str:
        """Транскрибировать аудиофайл."""
        pass
    
    @abstractmethod
    def preprocess_audio(self, audio_path: str):
        """Предобработка аудиофайла"""
        pass