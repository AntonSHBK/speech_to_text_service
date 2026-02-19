from pathlib import Path

from app.settings import settings


def clean_transcriptions() -> tuple[int, int]:
    files_removed = 0
    dirs_skipped = 0

    target_dir = settings.TRANSCRIBE_RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    for entry in target_dir.iterdir():
        if entry.is_file():
            entry.unlink(missing_ok=True)
            files_removed += 1
        else:
            dirs_skipped += 1

    return files_removed, dirs_skipped


if __name__ == "__main__":
    removed, skipped = clean_transcriptions()
    print(f"Removed files: {removed}")
    print(f"Skipped directories: {skipped}")
