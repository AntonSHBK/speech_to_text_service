from re import A
import uuid
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.settings import settings
from app.utils.export import export_result, ExportFormat
from app.models.transcriber import FastWhisperTranscriber


class TranscriberService:
    def __init__(self):
        self.transcriber: Optional[FastWhisperTranscriber] = None

    def init(self, **kwargs):
        self.transcriber = FastWhisperTranscriber(**kwargs)

    def is_ready(self) -> bool:
        return self.transcriber is not None

    def get(self) -> FastWhisperTranscriber:
        if not self.transcriber:
            raise RuntimeError("Transcriber is not initialized")
        return self.transcriber
    
    async def prepare_audio(
        self, 
        raw_bytes: bytes, 
        filename: str, 
        save_file: bool
    ) -> Path:
        original = Path(filename)

        stem = original.stem[:20]
        suffix = original.suffix.lower()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        uid = uuid.uuid4().hex

        safe_name = f"{stem}_{timestamp}_{uid}{suffix}"

        if save_file:
            path = settings.AUDIO_DIR / safe_name
            path.write_bytes(raw_bytes)
            return path

        tmp = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix,
            dir=settings.AUDIO_DIR
        )
        tmp.write(raw_bytes)
        tmp.flush()
        return Path(tmp.name)
    
    async def export_result(
        self,
        result: dict,
        source_filename: str,
        format: ExportFormat = "docx"
    ) -> Path:
        stem = Path(source_filename).stem[:20]
        timestamp = datetime.now().strftime("%d_%m_%Y_%H%M%S")

        filename = f"{stem}_{timestamp}.{format}"
        path = settings.TRANSCRIBE_RESULTS_DIR / filename

        return export_result(result, path, format)


transcriber_service = TranscriberService()
