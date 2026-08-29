from pathlib import Path

from app.utils.exporters.common import build_speaker_blocks


def _format_ass_timestamp(seconds: float | int | None) -> str:
    """Format seconds using the ASS timestamp format."""
    total_centiseconds = max(0, round(float(seconds or 0.0) * 100))
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _clean_text(text: str) -> str:
    return " ".join(text.split()).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def export_ass(result: dict, path: Path) -> Path:
    """Export transcription segments to the Advanced SubStation Alpha format."""
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,60,60,45,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dialogue_lines: list[str] = []
    for block in build_speaker_blocks(result):
        speaker = str(block.get("speaker") or "")
        for part in block.get("parts") or []:
            text = _clean_text(str(part.get("text", "")))
            if not text:
                continue
            if speaker:
                text = f"{speaker}: {text}"

            dialogue_lines.append(
                "Dialogue: 0,"
                f"{_format_ass_timestamp(part.get('start'))},"
                f"{_format_ass_timestamp(part.get('end'))},"
                f"Default,{speaker},0,0,0,,{text}"
            )

    if not dialogue_lines:
        text = _clean_text(str(result.get("text", "")))
        if text:
            dialogue_lines.append(
                "Dialogue: 0,"
                f"{_format_ass_timestamp(0)},"
                f"{_format_ass_timestamp(result.get('duration', 0.0))},"
                f"Default,,0,0,0,,{text}"
            )

    path.write_text(header + "\n".join(dialogue_lines) + "\n", encoding="utf-8")
    return path
