from pathlib import Path
from typing import Literal

from app.utils.exporters.docx_exporter import export_docx
from app.utils.exporters.markdown_exporter import export_markdown
from app.utils.exporters.pdf_exporter import export_pdf
from app.utils.exporters.ass_exporter import export_ass
from app.utils.exporters.srt_exporter import export_srt
from app.utils.exporters.txt_exporter import export_txt
from app.utils.exporters.vtt_exporter import export_vtt

ExportFormat = Literal["docx", "txt", "md", "pdf", "srt", "vtt", "ass"]


def export_result(
    result: dict,
    path: Path,
    format: ExportFormat,
    export_timestamps: bool = False,
    paragraph_pause_sec: float = 1.5,
    paragraph_max_chars: int = 350,
) -> Path:
    match format:
        case "docx":
            return export_docx(
                result,
                path,
                export_timestamps=export_timestamps,
                paragraph_pause_sec=paragraph_pause_sec,
                paragraph_max_chars=paragraph_max_chars,
            )
        case "txt":
            return export_txt(
                result,
                path,
                export_timestamps=export_timestamps,
                paragraph_pause_sec=paragraph_pause_sec,
                paragraph_max_chars=paragraph_max_chars,
            )
        case "md":
            return export_markdown(
                result,
                path,
                export_timestamps=export_timestamps,
                paragraph_pause_sec=paragraph_pause_sec,
                paragraph_max_chars=paragraph_max_chars,
            )
        case "pdf":
            return export_pdf(
                result,
                path,
                export_timestamps=export_timestamps,
                paragraph_pause_sec=paragraph_pause_sec,
                paragraph_max_chars=paragraph_max_chars,
            )
        case "srt":
            return export_srt(result, path)
        case "vtt":
            return export_vtt(result, path)
        case "ass":
            return export_ass(result, path)
        case _:
            raise ValueError(f"Unsupported export format: {format}")
