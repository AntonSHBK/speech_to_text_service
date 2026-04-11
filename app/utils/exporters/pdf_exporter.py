from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.utils.exporters.common import build_paragraph_blocks, format_timestamp

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


def _part_to_chunk(part: dict, export_timestamps: bool) -> str:
    text = str(part.get("text", "")).strip()
    if not text:
        return ""
    if not export_timestamps:
        return text
    start = format_timestamp(part.get("start"))
    end = format_timestamp(part.get("end"))
    return f"[{start} - {end}] {text}"


def export_pdf(
    result: dict,
    path: Path,
    export_timestamps: bool = False,
    paragraph_pause_sec: float = 2.0,
    paragraph_max_chars: int = 350,
) -> Path:
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
    heading_size = 14

    def ensure_space(lines_count: int = 1) -> None:
        nonlocal y
        needed = line_height * lines_count
        if y - needed <= bottom_margin:
            pdf.showPage()
            y = page_height - top_margin

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

    def wrap_text(text: str) -> list[str]:
        if not text:
            return [""]

        wrapped: list[str] = []
        current = ""
        for token in text.split(" "):
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

    def draw_paragraph(text: str) -> None:
        nonlocal y
        lines = wrap_text(text)
        for line in lines:
            ensure_space(1)
            pdf.setFont(regular_font, font_size)
            pdf.drawString(left_margin, y, line)
            y -= line_height
        y -= 4

    pdf.setFont(bold_font, heading_size)
    pdf.drawString(left_margin, y, "Результат транскрибации")
    y -= 28

    blocks = build_paragraph_blocks(
        result,
        pause_sec=paragraph_pause_sec,
        max_chars=paragraph_max_chars,
    )
    if blocks:
        prev_speaker: str | None = None
        for block in blocks:
            speaker = block.get("speaker")
            parts = block.get("parts") or []
            chunks = [_part_to_chunk(part, export_timestamps) for part in parts]
            chunks = [chunk for chunk in chunks if chunk]
            if not chunks:
                continue

            if speaker and speaker != prev_speaker:
                ensure_space(1)
                pdf.setFont(bold_font, font_size)
                pdf.drawString(left_margin, y, f"{speaker}:")
                y -= line_height

            draw_paragraph(" ".join(chunks))
            prev_speaker = speaker if isinstance(speaker, str) else None
    else:
        draw_paragraph(result.get("text", "") or "")

    pdf.save()
    return path
