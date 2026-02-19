from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def export_docx(result: dict, path: Path) -> Path:
    document = Document()

    heading = document.add_heading("Результат транскрибации", level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    paragraph = document.add_paragraph(result.get("text", ""))
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    paragraph_format = paragraph.paragraph_format
    paragraph_format.line_spacing = 1.25
    paragraph_format.space_after = Pt(6)
    paragraph_format.space_before = Pt(6)

    document.save(path)
    return path
