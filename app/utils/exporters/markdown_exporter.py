from pathlib import Path


def export_markdown(result: dict, path: Path) -> Path:
    content = f"# Результат транскрибации\n\n{result.get('text', '')}"
    path.write_text(content, encoding="utf-8")
    return path
