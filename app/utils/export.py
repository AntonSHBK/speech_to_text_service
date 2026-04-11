from pathlib import Path
from typing import Literal

from app.utils.exporters.docx_exporter import export_docx
from app.utils.exporters.markdown_exporter import export_markdown
from app.utils.exporters.pdf_exporter import export_pdf
from app.utils.exporters.txt_exporter import export_txt

ExportFormat = Literal["docx", "txt", "md", "pdf"]


def export_result(
    result: dict,
    path: Path,
    format: ExportFormat,
    export_timestamps: bool = False,
    paragraph_pause_sec: float = 1.0,
    paragraph_max_chars: int = 300,
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
        case _:
            raise ValueError(f"Unsupported export format: {format}")
