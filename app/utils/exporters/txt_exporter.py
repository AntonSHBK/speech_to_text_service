from pathlib import Path

from app.utils.exporters.common import build_paragraph_blocks, format_timestamp


def _build_merged_speaker_line(
    speaker: str | None,
    parts: list[dict],
    export_timestamps: bool,
    show_speaker: bool = True,
) -> str:
    chunks: list[str] = []
    for part in parts:
        text = str(part.get("text", "")).strip()
        if not text:
            continue
        if export_timestamps:
            start = format_timestamp(part.get("start"))
            end = format_timestamp(part.get("end"))
            chunks.append(f"[{start} - {end}] {text}")
        else:
            chunks.append(text)

    if not chunks:
        return ""

    if speaker and show_speaker:
        return f"[{speaker}] " + " ".join(chunks)
    return " ".join(chunks)


def export_txt(
    result: dict,
    path: Path,
    export_timestamps: bool = False,
    paragraph_pause_sec: float = 2.0,
    paragraph_max_chars: int = 350,
) -> Path:
    blocks = build_paragraph_blocks(
        result,
        pause_sec=paragraph_pause_sec,
        max_chars=paragraph_max_chars,
    )
    if blocks:
        lines: list[str] = []
        prev_speaker: str | None = None
        for block in blocks:
            speaker = block.get("speaker")
            line = _build_merged_speaker_line(
                speaker=speaker,
                parts=block.get("parts") or [],
                export_timestamps=export_timestamps,
                show_speaker=(speaker != prev_speaker),
            )
            if line:
                lines.append(line)
            prev_speaker = speaker if isinstance(speaker, str) else None

        if lines:
            path.write_text("\n\n".join(lines), encoding="utf-8")
            return path

    path.write_text(result.get("text", "") or "", encoding="utf-8")
    return path
