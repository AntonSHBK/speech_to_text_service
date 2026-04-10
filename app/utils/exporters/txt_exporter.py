from pathlib import Path

from app.utils.exporters.common import build_speaker_blocks, format_timestamp


def _build_merged_speaker_line(
    speaker: str | None,
    parts: list[dict],
    export_timestamps: bool,
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

    if speaker:
        return f"[{speaker}] " + " ".join(chunks)
    return " ".join(chunks)


def export_txt(result: dict, path: Path, export_timestamps: bool = False) -> Path:
    blocks = build_speaker_blocks(result)
    if blocks:
        lines: list[str] = []
        has_speakers = any(block.get("speaker") for block in blocks)
        for block in blocks:
            line = _build_merged_speaker_line(
                speaker=block.get("speaker"),
                parts=block.get("parts") or [],
                export_timestamps=export_timestamps,
            )
            if line:
                lines.append(line)

        if lines:
            separator = "\n" if has_speakers else " "
            path.write_text(separator.join(lines), encoding="utf-8")
            return path

    path.write_text(result.get("text", "") or "", encoding="utf-8")
    return path
