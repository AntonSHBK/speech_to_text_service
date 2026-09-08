import random
import re
from typing import Any


_EPS = 1e-6


def format_timestamp(seconds: float | int | None) -> str:
    value = float(seconds or 0.0)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = value % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def format_speaker_label(label: Any) -> str | None:
    if label is None:
        return None

    value = str(label).strip()
    match = re.fullmatch(r"speaker(?:[\s_-]*(.*))?", value, flags=re.IGNORECASE)
    if not match:
        return value

    suffix = (match.group(1) or "").strip()
    return f"Спикер {suffix}".rstrip()


def _speaker_score_for_interval(
    interval_start: float,
    interval_end: float,
    diarization_segment: dict[str, Any],
) -> tuple[float, float]:
    ds_start = _safe_float(diarization_segment.get("start"))
    ds_end = _safe_float(diarization_segment.get("end"), ds_start)
    ds_duration = max(_EPS, ds_end - ds_start)
    overlap = _interval_overlap(interval_start, interval_end, ds_start, ds_end)
    normalized_score = overlap / ds_duration
    return normalized_score, overlap


def _pick_speaker_for_interval(
    interval_start: float,
    interval_end: float,
    diarization_segments: list[dict[str, Any]],
) -> str | None:
    best_speaker: str | None = None
    best_score = 0.0
    best_overlap = 0.0

    for ds in diarization_segments:
        score, overlap = _speaker_score_for_interval(interval_start, interval_end, ds)
        if overlap <= _EPS:
            continue
        if score > best_score or (score == best_score and overlap > best_overlap):
            best_score = score
            best_overlap = overlap
            best_speaker = format_speaker_label(ds.get("speaker"))

    return best_speaker


def _pick_nearest_speaker_for_interval(
    interval_start: float,
    interval_end: float,
    diarization_segments: list[dict[str, Any]],
) -> str | None:
    center = (interval_start + interval_end) / 2
    best_speaker: str | None = None
    best_distance: float | None = None

    for ds in diarization_segments:
        ds_start = _safe_float(ds.get("start"))
        ds_end = _safe_float(ds.get("end"), ds_start)
        if ds_end < ds_start:
            continue

        if ds_start <= center <= ds_end:
            distance = 0.0
        else:
            distance = min(abs(center - ds_start), abs(center - ds_end))

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_speaker = format_speaker_label(ds.get("speaker"))

    return best_speaker


def _resolve_speaker_for_interval(
    interval_start: float,
    interval_end: float,
    diarization_segments: list[dict[str, Any]],
) -> str | None:
    return _pick_speaker_for_interval(
        interval_start,
        interval_end,
        diarization_segments,
    ) or _pick_nearest_speaker_for_interval(
        interval_start,
        interval_end,
        diarization_segments,
    )


def _pick_speaker(
    segment: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
) -> str | None:
    start = _safe_float(segment.get("start"))
    end = _safe_float(segment.get("end"), start)
    return _resolve_speaker_for_interval(start, end, diarization_segments)


def _word_boundary_index_by_time(
    words: list[str],
    segment_start: float,
    segment_end: float,
    boundary_time: float,
) -> int:
    if not words:
        return 0

    duration = segment_end - segment_start
    if duration <= _EPS:
        return len(words)

    ratio = (boundary_time - segment_start) / duration
    ratio = max(0.0, min(1.0, ratio))

    return max(0, min(len(words), round(ratio * len(words))))


def _ends_with_sentence_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    tail = stripped.rstrip("\"'??)]}")
    return bool(tail) and tail[-1] in ".!??"


def _starts_with_upper(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped:
        return False
    for ch in stripped:
        if ch.isalpha():
            return ch.isupper()
    return False


def _is_natural_text_boundary(left_text: str, right_text: str) -> bool:
    return _ends_with_sentence_punctuation(left_text) or _starts_with_upper(right_text)


def _single_speaker_part(
    segment: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    text: str,
) -> list[dict[str, Any]]:
    start = _safe_float(segment.get("start"))
    end = _safe_float(segment.get("end"), start)
    return [
        {
            "speaker": _resolve_speaker_for_interval(start, end, diarization_segments),
            "start": segment.get("start"),
            "end": segment.get("end"),
            "text": text,
        }
    ]


def _split_segment_by_speakers(
    segment: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    text = str(segment.get("text", "")).strip()
    if not text:
        return []

    words = re.findall(r"\S+", text)
    if not words:
        return []

    start = _safe_float(segment.get("start"))
    end = _safe_float(segment.get("end"), start)
    if end <= start:
        return _single_speaker_part(segment, diarization_segments, text)

    boundaries = {start, end}
    for ds in diarization_segments:
        ds_start = _safe_float(ds.get("start"))
        ds_end = _safe_float(ds.get("end"), ds_start)
        if _interval_overlap(start, end, ds_start, ds_end) <= _EPS:
            continue
        boundaries.add(max(start, ds_start))
        boundaries.add(min(end, ds_end))

    sorted_boundaries = sorted(boundaries)
    slices: list[dict[str, Any]] = []
    for left, right in zip(sorted_boundaries, sorted_boundaries[1:]):
        if right - left <= _EPS:
            continue
        speaker = _resolve_speaker_for_interval(left, right, diarization_segments)
        if slices and slices[-1]["speaker"] == speaker:
            slices[-1]["end"] = right
        else:
            slices.append({"speaker": speaker, "start": left, "end": right})

    if not slices:
        return _single_speaker_part(segment, diarization_segments, text)

    parts: list[dict[str, Any]] = []
    previous_word_idx = 0
    for idx, item in enumerate(slices):
        if idx == len(slices) - 1:
            next_word_idx = len(words)
        else:
            next_word_idx = _word_boundary_index_by_time(
                words=words,
                segment_start=start,
                segment_end=end,
                boundary_time=item["end"],
            )
            next_word_idx = max(previous_word_idx, next_word_idx)

        part_words = words[previous_word_idx:next_word_idx]
        previous_word_idx = next_word_idx
        if not part_words:
            continue

        part = {
            "speaker": item["speaker"],
            "start": item["start"],
            "end": item["end"],
            "text": " ".join(part_words),
        }

        if parts and parts[-1]["speaker"] == part["speaker"]:
            parts[-1]["end"] = part["end"]
            parts[-1]["text"] = f'{parts[-1]["text"]} {part["text"]}'.strip()
        else:
            parts.append(part)

    if len(parts) > 1:
        for left, right in zip(parts, parts[1:]):
            if not _is_natural_text_boundary(left["text"], right["text"]):
                return _single_speaker_part(segment, diarization_segments, text)

    return parts


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

            for speaker_part in _split_segment_by_speakers(segment, diarization_segments):
                text = str(speaker_part.get("text", "")).strip()
                if not text:
                    continue

                speaker = speaker_part.get("speaker")
                part = {
                    "start": speaker_part.get("start"),
                    "end": speaker_part.get("end"),
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


def _split_parts_by_pause(
    parts: list[dict[str, Any]],
    pause_sec: float,
    min_chars: int,
    max_chars: int,
) -> list[list[dict[str, Any]]]:
    def _parts_len(items: list[dict[str, Any]]) -> int:
        length = 0
        for item in items:
            text_item = str(item.get("text", "")).strip()
            if not text_item:
                continue
            if length == 0:
                length = len(text_item)
            else:
                length += 1 + len(text_item)
        return length

    paragraphs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_len = 0
    prev_end: float | None = None
    current_max_chars = random.randint(min_chars, max_chars)

    for part in parts:
        if not isinstance(part, dict):
            continue

        text = str(part.get("text", "")).strip()
        if not text:
            continue

        start_raw = part.get("start")
        end_raw = part.get("end")
        start = float(start_raw) if isinstance(start_raw, (int, float)) else None
        end = float(end_raw) if isinstance(end_raw, (int, float)) else None

        should_split = (
            bool(current)
            and start is not None
            and prev_end is not None
            and (start - prev_end) > pause_sec
        )
        text_len = len(text)
        should_split_by_len = bool(current) and (
            current_len + 1 + text_len > current_max_chars
        )
        if should_split:
            paragraphs.append(current)
            current = []
            current_len = 0
            current_max_chars = random.randint(min_chars, max_chars)
        elif should_split_by_len:
            prev_text = str(current[-1].get("text", "")).strip() if current else ""

            if current and _is_natural_text_boundary(prev_text, text):
                paragraphs.append(current)
                current = []
                current_len = 0
                current_max_chars = random.randint(min_chars, max_chars)
            else:
                split_idx = None
                for idx in range(len(current) - 2, -1, -1):
                    left_text = str(current[idx].get("text", "")).strip()
                    right_text = str(current[idx + 1].get("text", "")).strip()
                    if _is_natural_text_boundary(left_text, right_text):
                        split_idx = idx + 1
                        break

                if split_idx is not None:
                    paragraphs.append(current[:split_idx])
                    current = current[split_idx:]
                    current_len = _parts_len(current)
                    current_max_chars = random.randint(min_chars, max_chars)

        normalized_part: dict[str, Any] = {
            "start": start_raw,
            "end": end_raw,
            "text": text,
        }
        current.append(normalized_part)
        if current_len == 0:
            current_len = text_len
        else:
            current_len += 1 + text_len
        if end is not None:
            prev_end = end

    if current:
        paragraphs.append(current)
    return paragraphs


def build_paragraph_blocks(
    result: dict,
    pause_sec: float = 2.0,
    min_chars: int = 200,
    max_chars: int = 350,
) -> list[dict[str, Any]]:
    """
    Build paragraph blocks by pauses and speaker changes.
    - With speakers: paragraph split is performed inside each speaker block.
    - Without speakers: paragraph split is performed on the full text sequence.
    """
    speaker_blocks = build_speaker_blocks(result)
    if not speaker_blocks:
        return []

    has_speakers = any(block.get("speaker") for block in speaker_blocks)
    paragraph_blocks: list[dict[str, Any]] = []

    if has_speakers:
        for block in speaker_blocks:
            speaker = block.get("speaker")
            parts = block.get("parts") or []
            for paragraph_parts in _split_parts_by_pause(
                parts,
                pause_sec=pause_sec,
                min_chars=min_chars,
                max_chars=max_chars,
            ):
                paragraph_blocks.append({"speaker": speaker, "parts": paragraph_parts})
        return paragraph_blocks

    all_parts: list[dict[str, Any]] = []
    for block in speaker_blocks:
        all_parts.extend(block.get("parts") or [])

    for paragraph_parts in _split_parts_by_pause(
        all_parts,
        pause_sec=pause_sec,
        min_chars=min_chars,
        max_chars=max_chars,
    ):
        paragraph_blocks.append({"speaker": None, "parts": paragraph_parts})

    return paragraph_blocks
