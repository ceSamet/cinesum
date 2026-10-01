import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.audio.whisper_transcriber import transcribe_audio_with_whisper
from src.summary.speech_boundaries import is_linguistically_complete_text


TARGETED_ASR_MAX_SEGMENTS = 8
TARGETED_ASR_EVENT_THRESHOLD = 0.90
TARGETED_ASR_LOOKBEHIND_SEC = 15.0
TARGETED_ASR_LOOKAHEAD_SEC = 8.0
TARGETED_ASR_MAX_NO_SPEECH_PROB = 0.65
TARGETED_ASR_SPEECH_PADDING_SEC = 0.45

# The full-video Whisper pass can mistime a word by more than the normal 0.45s
# acoustic pad near scene cuts, music beds or dense dialogue (confirmed on real
# footage: a word's real audio started/ended up to ~0.5s away from where the
# full-pass transcript placed it). These constants bound a short, focused
# re-check of each segment's own edges against the real audio.
BOUNDARY_VERIFY_LOOKAROUND_SEC = 1.5
BOUNDARY_VERIFY_OVERLAP_SEC = 0.5
BOUNDARY_VERIFY_PAD_SEC = 0.30
BOUNDARY_VERIFY_MAX_EXTENSION_SEC = 1.5
BOUNDARY_VERIFY_EDGE_SLOP_SEC = 0.05
BOUNDARY_VERIFY_MAX_NO_SPEECH_PROB = 0.65


def _event_score(segment: Dict[str, Any], shots_by_id: Dict[int, Dict[str, Any]]) -> float:
    return max(
        (
            float(shots_by_id.get(int(scene_id), {}).get("narrative_event_score", 0.0))
            for scene_id in segment.get("core_shot_ids", [])
        ),
        default=0.0,
    )


def _useful_asr_segments(
    transcript: Dict[str, Any],
    *,
    window_start: float,
    original_start: float,
    original_end: float,
) -> List[Dict[str, Any]]:
    useful = []
    for row in transcript.get("segments", []):
        text = " ".join(str(row.get("text", "")).split()).strip()
        if not text:
            continue
        no_speech_probability = float(row.get("no_speech_prob", 0.0) or 0.0)
        if no_speech_probability > TARGETED_ASR_MAX_NO_SPEECH_PROB:
            continue
        absolute_start = window_start + float(row.get("start", 0.0))
        absolute_end = window_start + float(row.get("end", 0.0))
        if absolute_end < window_start or absolute_start > original_end + TARGETED_ASR_LOOKAHEAD_SEC:
            continue
        useful.append({
            "start": absolute_start,
            "end": absolute_end,
            "text": text,
            "no_speech_prob": no_speech_probability,
        })
    return useful


def refine_critical_silent_segments(
    segments: Iterable[Dict[str, Any]],
    scored_scenes: Iterable[Dict[str, Any]],
    *,
    audio_wav_path: Path,
    language: Optional[str] = None,
    max_segments: int = TARGETED_ASR_MAX_SEGMENTS,
    transcribe_fn: Callable[..., Dict[str, Any]] = transcribe_audio_with_whisper,
    audio_runner: Callable[..., Any] = subprocess.run,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Recheck visually critical, apparently silent summary windows with VAD disabled.

    This is deliberately a post-selection pass: it does not retranscribe the full
    film and therefore keeps the normal cached summary path fast.
    """
    rows = [dict(segment) for segment in segments]
    wav_path = Path(audio_wav_path)
    if not wav_path.exists():
        return rows, {"status": "skipped", "reason": "audio_wav_missing"}

    shots_by_id = {int(shot["scene_id"]): shot for shot in scored_scenes}
    candidates = []
    for index, segment in enumerate(rows):
        event_score = _event_score(segment, shots_by_id)
        unanswered_question = "trimmed_unanswered_question_reaction" in str(
            segment.get("boundary_reason", "")
        )
        critical_silence = (
            not bool(segment.get("has_speech"))
            and event_score >= TARGETED_ASR_EVENT_THRESHOLD
        )
        suspicious_speech = (
            bool(segment.get("has_speech"))
            and not is_linguistically_complete_text(
                str(segment.get("transcript_text", ""))
            )
        )
        if unanswered_question or critical_silence or suspicious_speech:
            candidates.append((index, max(
                event_score,
                1.0 if unanswered_question else 0.95 if suspicious_speech else 0.0,
            )))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    candidates = candidates[:max(0, int(max_segments))]
    if not candidates:
        return rows, {"status": "skipped", "reason": "no_critical_silent_segments"}

    applied = []
    errors = []
    with tempfile.TemporaryDirectory(prefix="cinesum_targeted_asr_") as temp_dir:
        temp_root = Path(temp_dir)
        for index, event_score in candidates:
            segment = rows[index]
            original_start = float(segment["start"])
            original_end = float(segment["end"])
            window_start = max(0.0, original_start - TARGETED_ASR_LOOKBEHIND_SEC)
            window_end = original_end + TARGETED_ASR_LOOKAHEAD_SEC
            clip_path = temp_root / f"segment_{index:03d}.wav"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{window_start:.3f}",
                "-t", f"{window_end - window_start:.3f}",
                "-i", str(wav_path),
                "-ac", "1", "-ar", "16000", str(clip_path),
            ]
            try:
                audio_runner(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                transcript = transcribe_fn(
                    str(clip_path),
                    model_size="small",
                    device="auto",
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=False,
                    language=language,
                )
                useful = _useful_asr_segments(
                    transcript,
                    window_start=window_start,
                    original_start=original_start,
                    original_end=original_end,
                )
                if not useful:
                    continue

                recovered_text = " ".join(row["text"] for row in useful)
                if not is_linguistically_complete_text(recovered_text):
                    continue

                detected_start = min(row["start"] for row in useful)
                detected_end = max(row["end"] for row in useful)
                refined_start = min(
                    original_start,
                    max(window_start, detected_start - TARGETED_ASR_SPEECH_PADDING_SEC),
                )
                refined_end = max(
                    original_end,
                    min(window_end, detected_end + TARGETED_ASR_SPEECH_PADDING_SEC),
                )
                segment["start"] = round(refined_start, 3)
                segment["end"] = round(refined_end, 3)
                segment["duration"] = round(refined_end - refined_start, 3)
                segment["has_speech"] = True
                segment["complete_utterance"] = True
                segment["boundary_mode"] = "targeted_asr"
                segment["boundary_reason"] = "targeted_asr_vad_disabled"
                segment["transcript_text"] = recovered_text
                segment["targeted_asr_applied"] = True
                segment["targeted_asr_event_score"] = round(event_score, 4)
                segment["targeted_asr_original_start"] = round(original_start, 3)
                segment["targeted_asr_original_end"] = round(original_end, 3)
                applied.append(index)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
                errors.append({"segment_index": index, "error": str(exc)})

    status = "applied" if applied else "no_speech_detected"
    return rows, {
        "status": status,
        "candidate_count": len(candidates),
        "applied_count": len(applied),
        "applied_segment_indexes": applied,
        "errors": errors,
    }


def _words_from_transcript(
    transcript: Dict[str, Any],
    window_start: float,
    *,
    max_no_speech_prob: float = BOUNDARY_VERIFY_MAX_NO_SPEECH_PROB,
) -> List[Dict[str, Any]]:
    words = []
    for row in transcript.get("segments", []):
        no_speech_probability = float(row.get("no_speech_prob", 0.0) or 0.0)
        if no_speech_probability > max_no_speech_prob:
            continue
        for word in row.get("words") or []:
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            words.append({
                "start": window_start + float(word.get("start", 0.0)),
                "end": window_start + float(word.get("end", 0.0)),
                "word": text,
            })
    return words


def verify_and_extend_speech_boundaries(
    segments: Iterable[Dict[str, Any]],
    *,
    audio_wav_path: Path,
    video_duration: Optional[float] = None,
    language: Optional[str] = None,
    transcribe_fn: Callable[..., Dict[str, Any]] = transcribe_audio_with_whisper,
    audio_runner: Callable[..., Any] = subprocess.run,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Re-check every speech segment's own edges against the real audio.

    The full-video Whisper pass can mistime a word near a scene cut, music bed,
    or dense dialogue by more than the normal acoustic safety pad, so a segment
    already built from "safe" word/sentence boundaries can still silently clip
    the start or end of a sentence in the exported clip (confirmed on real
    footage). This runs a short, VAD-disabled re-transcription of the audio
    immediately around each segment's own edges and extends the boundary if
    real speech is found straddling it. Scope is bounded to the final,
    already-selected segment list -- a handful of short windows, not the whole
    film -- so cached summary generation stays fast. It never shrinks a
    segment, and it clamps against the neighboring segment so segments cannot
    overlap.
    """
    rows = [dict(segment) for segment in segments]
    wav_path = Path(audio_wav_path)
    if not wav_path.exists():
        return rows, {"status": "skipped", "reason": "audio_wav_missing"}

    order = sorted(range(len(rows)), key=lambda i: float(rows[i]["start"]))
    speech_positions = [position for position, index in enumerate(order) if rows[index].get("has_speech")]
    if not speech_positions:
        return rows, {"status": "skipped", "reason": "no_speech_segments"}

    extended_start_indexes: List[int] = []
    extended_end_indexes: List[int] = []
    errors: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="cinesum_boundary_verify_") as temp_dir:
        temp_root = Path(temp_dir)

        def _transcribe_window(window_start: float, window_end: float, tag: str) -> List[Dict[str, Any]]:
            window_start = max(0.0, window_start)
            if window_end - window_start < 0.4:
                return []
            clip_path = temp_root / f"{tag}.wav"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{window_start:.3f}",
                "-t", f"{window_end - window_start:.3f}",
                "-i", str(wav_path),
                "-ac", "1", "-ar", "16000", str(clip_path),
            ]
            audio_runner(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            transcript = transcribe_fn(
                str(clip_path),
                model_size="small",
                device="auto",
                beam_size=5,
                word_timestamps=True,
                vad_filter=False,
                language=language,
            )
            return _words_from_transcript(transcript, window_start)

        for position in speech_positions:
            index = order[position]
            segment = rows[index]
            start = float(segment["start"])
            end = float(segment["end"])
            prev_end = float(rows[order[position - 1]]["end"]) if position > 0 else 0.0
            next_start = (
                float(rows[order[position + 1]]["start"])
                if position + 1 < len(order)
                else (video_duration if video_duration is not None else end + BOUNDARY_VERIFY_LOOKAROUND_SEC)
            )

            try:
                # End edge: does real speech extend past what we exported?
                end_window_limit = min(
                    end + BOUNDARY_VERIFY_LOOKAROUND_SEC,
                    next_start,
                    video_duration if video_duration is not None else float("inf"),
                )
                end_words = _transcribe_window(
                    end - BOUNDARY_VERIFY_OVERLAP_SEC, end_window_limit, f"end_{index}",
                )
                straddling_end = [
                    w for w in end_words
                    if w["start"] < end + BOUNDARY_VERIFY_EDGE_SLOP_SEC
                    and w["end"] > end + BOUNDARY_VERIFY_EDGE_SLOP_SEC
                ]
                if straddling_end:
                    detected_end = max(w["end"] for w in straddling_end)
                    new_end = min(
                        detected_end + BOUNDARY_VERIFY_PAD_SEC,
                        end_window_limit,
                        end + BOUNDARY_VERIFY_MAX_EXTENSION_SEC,
                    )
                    if new_end > end + BOUNDARY_VERIFY_EDGE_SLOP_SEC:
                        segment["end"] = round(new_end, 3)
                        segment["duration"] = round(new_end - start, 3)
                        segment["boundary_verified_end"] = True
                        extended_end_indexes.append(index)
                        end = new_end

                # Start edge: does real speech begin before what we exported?
                start_window_limit = max(start - BOUNDARY_VERIFY_LOOKAROUND_SEC, prev_end, 0.0)
                start_words = _transcribe_window(
                    start_window_limit, start + BOUNDARY_VERIFY_OVERLAP_SEC, f"start_{index}",
                )
                straddling_start = [
                    w for w in start_words
                    if w["start"] < start - BOUNDARY_VERIFY_EDGE_SLOP_SEC
                    and w["end"] > start - BOUNDARY_VERIFY_EDGE_SLOP_SEC
                ]
                if straddling_start:
                    detected_start = min(w["start"] for w in straddling_start)
                    new_start = max(
                        detected_start - BOUNDARY_VERIFY_PAD_SEC,
                        start_window_limit,
                        start - BOUNDARY_VERIFY_MAX_EXTENSION_SEC,
                    )
                    if new_start < start - BOUNDARY_VERIFY_EDGE_SLOP_SEC:
                        segment["start"] = round(new_start, 3)
                        segment["duration"] = round(float(segment["end"]) - new_start, 3)
                        segment["boundary_verified_start"] = True
                        extended_start_indexes.append(index)
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
                errors.append({"segment_index": index, "error": str(exc)})

    status = "applied" if (extended_start_indexes or extended_end_indexes) else "no_extension_needed"
    return rows, {
        "status": status,
        "checked_segment_count": len(speech_positions),
        "extended_start_count": len(extended_start_indexes),
        "extended_end_count": len(extended_end_indexes),
        "errors": errors,
    }
