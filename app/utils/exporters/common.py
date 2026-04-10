from typing import Any


def format_timestamp(seconds: float | int | None) -> str:
    value = float(seconds or 0.0)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = value % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _interval_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _resolve_diarization_segments(result: dict) -> list[dict[str, Any]]:
    diarization = result.get("diarization")
    if not isinstance(diarization, dict):
        return []

    segments = diarization.get("exclusive_segments") or diarization.get("segments") or []
    if not isinstance(segments, list):
        return []
    return [s for s in segments if isinstance(s, dict)]


def _pick_speaker(
    segment: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
) -> str | None:
    start = float(segment.get("start", 0.0))
    end = float(segment.get("end", start))

    best_speaker: str | None = None
    best_overlap = 0.0

    for ds in diarization_segments:
        ds_start = float(ds.get("start", 0.0))
        ds_end = float(ds.get("end", ds_start))
        overlap = _interval_overlap(start, end, ds_start, ds_end)
        if overlap > best_overlap:
            best_overlap = overlap
            label = ds.get("speaker")
            best_speaker = str(label) if label is not None else None

    return best_speaker


def build_speaker_blocks(result: dict) -> list[dict[str, Any]]:
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return []

    diarization_segments = _resolve_diarization_segments(result)
    blocks: list[dict[str, Any]] = []

    if diarization_segments:
        current_speaker: str | None = None
        current_parts: list[dict[str, Any]] = []

        for segment in segments:
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue

            speaker = _pick_speaker(segment, diarization_segments)
            part = {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text,
            }

            if current_parts and speaker != current_speaker:
                blocks.append({"speaker": current_speaker, "parts": current_parts})
                current_parts = []

            current_speaker = speaker
            current_parts.append(part)

        if current_parts:
            blocks.append({"speaker": current_speaker, "parts": current_parts})
        return blocks

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        blocks.append(
            {
                "speaker": None,
                "parts": [
                    {
                        "start": segment.get("start"),
                        "end": segment.get("end"),
                        "text": text,
                    }
                ],
            }
        )
    return blocks
