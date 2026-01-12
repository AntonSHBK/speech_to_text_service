from pathlib import Path
from abc import ABC, abstractmethod

from app.utils.logging import get_logger


class BaseModel(ABC):
    """Базовый класс для моделей транскрипции."""
    
    def __init__(
        self, 
        model_name: str, 
        cache_dir: str = 'cache_dir', 
        device: str = 'cpu',
    ):
        self.model_name = model_name
        self.device = device
        
        self.cache_dir = Path(cache_dir)        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger("model")
        
    @abstractmethod
    def load_model(self, model_name: str):
        """Загрузить модель."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        """Транскрибировать аудиофайл."""
        pass
    
    @abstractmethod
    def process(self, audio_path: str):
        """Обработка аудиофайла"""
        pass