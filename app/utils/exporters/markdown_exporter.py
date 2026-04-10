from pathlib import Path

from app.utils.exporters.common import build_speaker_blocks, format_timestamp


def _render_block(block: dict, export_timestamps: bool) -> str:
    speaker = block.get("speaker")
    parts = block.get("parts") or []

    chunks: list[str] = []
    for part in parts:
        text = str(part.get("text", "")).strip()
        if not text:
            continue
        if export_timestamps:
            start = format_timestamp(part.get("start"))
            end = format_timestamp(part.get("end"))
            chunks.append(f"`[{start} - {end}]` {text}")
        else:
            chunks.append(text)

    if not chunks:
        return ""

    content = " ".join(chunks)
    if speaker:
        return f"### {speaker}\n{content}"
    return content


def export_markdown(result: dict, path: Path, export_timestamps: bool = False) -> Path:
    blocks = build_speaker_blocks(result)
    if blocks:
        has_speakers = any(block.get("speaker") for block in blocks)
        rendered_blocks = [_render_block(block, export_timestamps) for block in blocks]
        rendered_blocks = [block for block in rendered_blocks if block]
        if rendered_blocks:
            separator = "\n\n" if has_speakers else " "
            content = "# Результат транскрибации\n\n" + separator.join(rendered_blocks)
            path.write_text(content, encoding="utf-8")
            return path

    content = f"# Результат транскрибации\n\n{result.get('text', '') or ''}"
    path.write_text(content, encoding="utf-8")
    return path
