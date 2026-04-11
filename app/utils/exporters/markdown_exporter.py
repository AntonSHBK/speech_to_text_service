from pathlib import Path

from app.utils.exporters.common import build_paragraph_blocks, format_timestamp


def _render_block(block: dict, export_timestamps: bool, show_speaker: bool = True) -> str:
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
    if speaker and show_speaker:
        return f"### {speaker}\n{content}"
    return content


def export_markdown(result: dict, path: Path, export_timestamps: bool = False) -> Path:
    blocks = build_paragraph_blocks(result, pause_sec=2.0)
    if blocks:
        rendered_blocks: list[str] = []
        prev_speaker: str | None = None
        for block in blocks:
            speaker = block.get("speaker")
            rendered = _render_block(
                block,
                export_timestamps,
                show_speaker=(speaker != prev_speaker),
            )
            if rendered:
                rendered_blocks.append(rendered)
            prev_speaker = speaker if isinstance(speaker, str) else None
        rendered_blocks = [block for block in rendered_blocks if block]
        if rendered_blocks:
            separator = "\n\n"
            content = "# Результат транскрибации\n\n" + separator.join(rendered_blocks)
            path.write_text(content, encoding="utf-8")
            return path

    content = f"# Результат транскрибации\n\n{result.get('text', '') or ''}"
    path.write_text(content, encoding="utf-8")
    return path
