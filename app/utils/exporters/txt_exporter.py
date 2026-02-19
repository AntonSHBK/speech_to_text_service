from pathlib import Path


def export_txt(result: dict, path: Path) -> Path:
    path.write_text(result.get("text", ""), encoding="utf-8")
    return path
