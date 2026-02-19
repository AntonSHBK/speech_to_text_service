from pathlib import Path
from textwrap import wrap
from typing import Literal

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ExportFormat = Literal["docx", "txt", "md", "pdf"]


def export_docx(result: dict, path: Path) -> Path:
    document = Document()

    heading = document.add_heading("Transcription Result", level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    paragraph = document.add_paragraph(result.get("text", ""))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    paragraph_format = paragraph.paragraph_format
    paragraph_format.line_spacing = 1.25
    paragraph_format.space_after = Pt(6)
    paragraph_format.space_before = Pt(6)

    document.save(path)
    return path


def export_txt(result: dict, path: Path) -> Path:
    path.write_text(result.get("text", ""), encoding="utf-8")
    return path


def export_markdown(result: dict, path: Path) -> Path:
    content = f"# Transcription Result\n\n{result.get('text', '')}"
    path.write_text(content, encoding="utf-8")
    return path


def export_pdf(result: dict, path: Path) -> Path:
    text = result.get("text", "") or ""

    pdf = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4

    x = 40
    y = height - 50
    line_height = 14
    max_chars = 100

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(x, y, "Transcription Result")
    y -= 28

    pdf.setFont("Helvetica", 11)
    for paragraph in text.splitlines() or [""]:
        lines = wrap(paragraph, width=max_chars) or [""]
        for line in lines:
            if y <= 40:
                pdf.showPage()
                pdf.setFont("Helvetica", 11)
                y = height - 50
            pdf.drawString(x, y, line)
            y -= line_height
        y -= 4

    pdf.save()
    return path


def export_result(result: dict, path: Path, format: ExportFormat) -> Path:
    match format:
        case "docx":
            return export_docx(result, path)
        case "txt":
            return export_txt(result, path)
        case "md":
            return export_markdown(result, path)
        case "pdf":
            return export_pdf(result, path)
        case _:
            raise ValueError(f"Unsupported export format: {format}")
