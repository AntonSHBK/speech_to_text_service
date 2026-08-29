from pathlib import Path

from app.utils.exporters.common import build_speaker_blocks


def _format_srt_timestamp(seconds: float | int | None) -> str:
    """Format seconds using the SRT timestamp format."""
    total_milliseconds = max(0, round(float(seconds or 0.0) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def export_srt(result: dict, path: Path) -> Path:
    """Export transcription segments to the SubRip subtitle format."""
    blocks = build_speaker_blocks(result)
    entries: list[str] = []
    number = 1

    for block in blocks:
        speaker = block.get("speaker")
        for part in block.get("parts") or []:
            text = _clean_text(str(part.get("text", "")))
            if not text:
                continue

            if speaker:
                text = f"{speaker}: {text}"

            start = _format_srt_timestamp(part.get("start"))
            end = _format_srt_timestamp(part.get("end"))
            entries.append(f"{number}\n{start} --> {end}\n{text}")
            number += 1

    if not entries:
        text = _clean_text(str(result.get("text", "")))
        if text:
            duration = result.get("duration", 0.0)
            entries.append(
                "1\n"
                f"{_format_srt_timestamp(0)} --> "
                f"{_format_srt_timestamp(duration)}\n"
                f"{text}"
            )

    path.write_text("\n\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
    return path
