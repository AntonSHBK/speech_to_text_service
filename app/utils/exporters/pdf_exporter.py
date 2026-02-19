from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PDF_FONT = "DejaVuSans"
PDF_FONT_BOLD = "DejaVuSans-Bold"


def _register_pdf_fonts() -> tuple[str, str]:
    regular_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    bold_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]

    regular_path = next((p for p in regular_candidates if p.exists()), None)
    bold_path = next((p for p in bold_candidates if p.exists()), None)

    if regular_path and bold_path:
        if PDF_FONT not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(PDF_FONT, str(regular_path)))
        if PDF_FONT_BOLD not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(PDF_FONT_BOLD, str(bold_path)))
        return PDF_FONT, PDF_FONT_BOLD

    return "Helvetica", "Helvetica-Bold"


def export_pdf(result: dict, path: Path) -> Path:
    text = result.get("text", "") or ""
    regular_font, bold_font = _register_pdf_fonts()

    pdf = canvas.Canvas(str(path), pagesize=A4)
    page_width, page_height = A4

    left_margin = 40
    right_margin = 40
    top_margin = 50
    bottom_margin = 40
    content_width = page_width - left_margin - right_margin
    y = page_height - top_margin
    line_height = 14
    font_size = 11

    pdf.setFont(bold_font, 14)
    pdf.drawString(left_margin, y, "Результат транскрибации")
    y -= 28

    pdf.setFont(regular_font, font_size)

    def split_long_token(token: str) -> list[str]:
        parts: list[str] = []
        current = ""
        for ch in token:
            candidate = current + ch
            if pdfmetrics.stringWidth(candidate, regular_font, font_size) <= content_width:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = ch
        if current:
            parts.append(current)
        return parts or [""]

    def wrap_paragraph(paragraph: str) -> list[str]:
        if not paragraph:
            return [""]

        wrapped: list[str] = []
        current = ""

        for token in paragraph.split(" "):
            candidate = token if not current else f"{current} {token}"
            if pdfmetrics.stringWidth(candidate, regular_font, font_size) <= content_width:
                current = candidate
                continue

            if current:
                wrapped.append(current)
                current = ""

            if pdfmetrics.stringWidth(token, regular_font, font_size) <= content_width:
                current = token
            else:
                token_parts = split_long_token(token)
                wrapped.extend(token_parts[:-1])
                current = token_parts[-1]

        if current:
            wrapped.append(current)
        return wrapped or [""]

    for paragraph in text.splitlines() or [""]:
        lines = wrap_paragraph(paragraph)
        for line in lines:
            if y <= bottom_margin:
                pdf.showPage()
                pdf.setFont(regular_font, font_size)
                y = page_height - top_margin
            pdf.drawString(left_margin, y, line)
            y -= line_height
        y -= 4

    pdf.save()
    return path
