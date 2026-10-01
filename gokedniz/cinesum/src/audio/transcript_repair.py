import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.audio.whisper_transcriber import transcribe_audio_with_whisper


TRANSCRIPT_REPAIR_VERSION = "1.0"
MAX_NORMAL_SEGMENT_SEC = 45.0
FORCE_REPAIR_SEGMENT_SEC = 90.0
MIN_WORDS_PER_SECOND = 0.20
REPAIR_CHUNK_SEC = 30.0
REPAIR_CONTEXT_SEC = 1.5
MAX_REPAIR_ANOMALIES = 8


def detect_transcript_anomalies(
    transcript: Dict[str, Any],
    *,
    max_segment_sec: float = MAX_NORMAL_SEGMENT_SEC,
) -> List[Dict[str, Any]]:
    """Find timestamp spans that cannot plausibly represent one ASR utterance."""
    anomalies = []
    for index, row in enumerate(transcript.get("segments", [])):
        start = float(row.get("start", 0.0))
        end = float(row.get("end", start))
        duration = max(0.0, end - start)
        words = row.get("words") or []
        word_rate = len(words) / duration if duration > 0 else 0.0
        reasons = []
        if duration >= FORCE_REPAIR_SEGMENT_SEC:
            reasons.append("extreme_duration")
        if duration > max_segment_sec and word_rate < MIN_WORDS_PER_SECOND:
            reasons.append("sparse_word_timestamps")
        if reasons:
            anomalies.append({
                "index": index,
                "start": start,
                "end": end,
                "duration": duration,
                "word_count": len(words),
                "word_rate": round(word_rate, 4),
                "reasons": reasons,
            })
    return anomalies


def _absolute_segment(row: Dict[str, Any], offset: float) -> Dict[str, Any]:
    result = dict(row)
    result["start"] = round(offset + float(row.get("start", 0.0)), 3)
    result["end"] = round(offset + float(row.get("end", 0.0)), 3)
    result["words"] = [
        {
            **word,
            "start": round(offset + float(word.get("start", 0.0)), 3),
            "end": round(offset + float(word.get("end", 0.0)), 3),
        }
        for word in (row.get("words") or [])
    ]
    return result


def _repair_one_span(
    audio_wav_path: Path,
    anomaly: Dict[str, Any],
    *,
    temp_root: Path,
    language: Optional[str],
    model_size: str,
    device: str,
    compute_type: Optional[str],
    transcribe_fn: Callable[..., Dict[str, Any]],
    audio_runner: Callable[..., Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    repaired = []
    errors = []
    span_start = float(anomaly["start"])
    span_end = float(anomaly["end"])
    core_start = span_start
    chunk_index = 0
    while core_start < span_end - 1e-6:
        core_end = min(span_end, core_start + REPAIR_CHUNK_SEC)
        extract_start = max(0.0, core_start - REPAIR_CONTEXT_SEC)
        extract_end = span_end if core_end >= span_end else core_end + REPAIR_CONTEXT_SEC
        clip_path = temp_root / f"repair_{int(anomaly['index']):03d}_{chunk_index:03d}.wav"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{extract_start:.3f}",
            "-t", f"{extract_end - extract_start:.3f}",
            "-i", str(audio_wav_path),
            "-ac", "1", "-ar", "16000", str(clip_path),
        ]
        try:
            audio_runner(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = transcribe_fn(
                str(clip_path),
                model_size=model_size,
                backend="faster-whisper",
                device=device,
                compute_type=compute_type,
                beam_size=3,
                word_timestamps=True,
                vad_filter=False,
                language=language,
            )
            for row in result.get("segments", []):
                absolute = _absolute_segment(row, extract_start)
                midpoint = (float(absolute["start"]) + float(absolute["end"])) / 2.0
                if core_start - 1e-6 <= midpoint < core_end + 1e-6:
                    repaired.append(absolute)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            errors.append(f"chunk_{chunk_index}:{type(exc).__name__}")
        core_start = core_end
        chunk_index += 1
    return repaired, errors


def repair_transcript_anomalies(
    transcript: Dict[str, Any],
    audio_wav_path: Path,
    *,
    language: Optional[str] = None,
    model_size: str = "small",
    device: str = "auto",
    compute_type: Optional[str] = None,
    max_anomalies: int = MAX_REPAIR_ANOMALIES,
    transcribe_fn: Callable[..., Dict[str, Any]] = transcribe_audio_with_whisper,
    audio_runner: Callable[..., Any] = subprocess.run,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Locally retranscribe only implausibly long/sparse Whisper spans."""
    source = dict(transcript)
    source_segments = [dict(row) for row in transcript.get("segments", [])]
    anomalies = detect_transcript_anomalies(transcript)[:max(0, int(max_anomalies))]
    if not anomalies:
        return source, {
            "status": "clean",
            "version": TRANSCRIPT_REPAIR_VERSION,
            "anomaly_count": 0,
            "repaired_count": 0,
        }
    audio_path = Path(audio_wav_path)
    if not audio_path.exists():
        return source, {
            "status": "skipped",
            "reason": "audio_wav_missing",
            "version": TRANSCRIPT_REPAIR_VERSION,
            "anomaly_count": len(anomalies),
            "repaired_count": 0,
        }

    replacements: Dict[int, List[Dict[str, Any]]] = {}
    errors = []
    with tempfile.TemporaryDirectory(prefix="cinesum_transcript_repair_") as temp_dir:
        temp_root = Path(temp_dir)
        for anomaly in anomalies:
            rows, span_errors = _repair_one_span(
                audio_path,
                anomaly,
                temp_root=temp_root,
                language=language or transcript.get("language"),
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                transcribe_fn=transcribe_fn,
                audio_runner=audio_runner,
            )
            if rows and not span_errors:
                replacements[int(anomaly["index"])] = rows
            else:
                errors.extend(
                    {"segment_index": anomaly["index"], "error": value}
                    for value in (span_errors or ["no_speech_detected"])
                )

    merged = []
    for index, row in enumerate(source_segments):
        merged.extend(replacements.get(index, [row]))
    merged.sort(key=lambda row: (float(row.get("start", 0.0)), float(row.get("end", 0.0))))
    for index, row in enumerate(merged):
        row["id"] = index
    source["segments"] = merged
    source["text"] = " ".join(
        str(row.get("text", "")).strip() for row in merged if str(row.get("text", "")).strip()
    ).strip()
    metadata = {
        "status": "applied" if replacements else "unchanged",
        "version": TRANSCRIPT_REPAIR_VERSION,
        "anomaly_count": len(anomalies),
        "repaired_count": len(replacements),
        "replaced_segment_indexes": sorted(replacements),
        "generated_segment_count": sum(len(rows) for rows in replacements.values()),
        "errors": errors[:8],
        "device": device,
        "compute_type": compute_type,
    }
    source["repair"] = metadata
    return source, metadata
