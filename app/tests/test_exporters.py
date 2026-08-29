from pathlib import Path

import pytest

from app.utils.export import export_result


@pytest.fixture
def result_with_speakers() -> dict:
    return {
        "language": "ru",
        "duration": 8.0,
        "text": "Добрый день. Начинаем тестирование экспорта.",
        "segments": [
            {"start": 0.0, "end": 3.2, "text": "Добрый день."},
            {"start": 3.2, "end": 8.0, "text": "Начинаем тестирование экспорта."},
        ],
        "diarization": {
            "segments": [
                {
                    "speaker": "Докладчик 1",
                    "start": 0.0,
                    "end": 3.2,
                    "duration": 3.2,
                },
                {
                    "speaker": "Докладчик 2",
                    "start": 3.2,
                    "end": 8.0,
                    "duration": 4.8,
                },
            ]
        },
    }


@pytest.fixture
def result_without_speakers() -> dict:
    return {
        "duration": 5.0,
        "text": "Текст без определения спикеров.",
        "segments": [
            {"start": 0.0, "end": 5.0, "text": "Текст без определения спикеров."},
        ],
    }


@pytest.mark.parametrize(
    ("export_format", "extension"),
    [
        ("txt", ".txt"),
        ("md", ".md"),
        ("docx", ".docx"),
        ("pdf", ".pdf"),
        ("srt", ".srt"),
        ("vtt", ".vtt"),
        ("ass", ".ass"),
    ],
)
def test_export_result_supports_all_formats(
    tmp_path: Path,
    result_with_speakers: dict,
    export_format: str,
    extension: str,
) -> None:
    output_path = tmp_path / f"result{extension}"

    exported_path = export_result(
        result=result_with_speakers,
        path=output_path,
        format=export_format,
        export_timestamps=True,
    )

    assert exported_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_srt_export_contains_srt_structure(
    tmp_path: Path,
    result_with_speakers: dict,
) -> None:
    output_path = tmp_path / "result.srt"

    export_result(result_with_speakers, output_path, format="srt")

    content = output_path.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:03,200" in content
    assert "Докладчик 1: Добрый день." in content
    assert "Докладчик 2: Начинаем тестирование экспорта." in content


def test_vtt_export_contains_webvtt_header(
    tmp_path: Path,
    result_with_speakers: dict,
) -> None:
    output_path = tmp_path / "result.vtt"

    export_result(result_with_speakers, output_path, format="vtt")

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("WEBVTT\n\n")
    assert "00:00:00.000 --> 00:00:03.200" in content
    assert "Докладчик 1: Добрый день." in content


def test_ass_export_contains_events(
    tmp_path: Path,
    result_with_speakers: dict,
) -> None:
    output_path = tmp_path / "result.ass"

    export_result(result_with_speakers, output_path, format="ass")

    content = output_path.read_text(encoding="utf-8")
    assert "[Events]" in content
    assert "Dialogue: 0,0:00:00.00,0:00:03.20" in content
    assert "Докладчик 1: Добрый день." in content


@pytest.mark.parametrize("export_format", ["srt", "vtt", "ass"])
def test_subtitle_export_works_without_speakers(
    tmp_path: Path,
    result_without_speakers: dict,
    export_format: str,
) -> None:
    output_path = tmp_path / f"result.{export_format}"

    export_result(result_without_speakers, output_path, format=export_format)

    content = output_path.read_text(encoding="utf-8")
    assert "Текст без определения спикеров." in content
    assert "Докладчик" not in content
