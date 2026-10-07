import json
import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable


_whisper_models: Dict[Tuple[str, str, str, str], Any] = {}


def _cuda_runtime_ready() -> bool:
    """Torch seeing a GPU does not mean CTranslate2 can load CUDA libraries."""
    if sys.platform == "win32":
        try:
            import ctranslate2

            return ctranslate2.get_cuda_device_count() > 0
        except (ImportError, RuntimeError, OSError):
            return False
    try:
        ctypes.CDLL("libcublas.so.12")
        ctypes.CDLL("libcudnn.so.9")
        return True
    except OSError:
        return False


def resolve_whisper_device(device: str = "auto") -> str:
    """Resolve the requested Whisper device against the installed Torch runtime."""
    if device != "auto":
        return device
    configured = os.getenv("THANOS_WHISPER_DEVICE", "auto").strip().lower()
    if configured == "cpu":
        return "cpu"
    if configured not in {"auto", "cuda"}:
        raise ValueError("THANOS_WHISPER_DEVICE auto, cpu veya cuda olmalıdır")
    try:
        import torch

        return "cuda" if torch.cuda.is_available() and _cuda_runtime_ready() else "cpu"
    except ImportError:
        return "cpu"


def resolve_whisper_compute_type(
    device: str,
    compute_type: Optional[str] = None,
) -> str:
    """Use fast GPU FP16 while keeping the memory-efficient CPU INT8 fallback."""
    return compute_type or ("float16" if device == "cuda" else "int8")


def get_whisper_model(
    model_size: str = "small",
    backend: str = "faster-whisper",
    device: str = "auto",
    compute_type: Optional[str] = None,
):
    """Load and cache Whisper models by backend, size, device and compute type."""
    resolved_device = resolve_whisper_device(device)
    resolved_compute = resolve_whisper_compute_type(resolved_device, compute_type)
    cache_key = (backend, model_size, resolved_device, resolved_compute)

    if cache_key in _whisper_models:
        return _whisper_models[cache_key]

    if backend == "faster-whisper":
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper kurulu değil. 'pip install faster-whisper' ile kurun "
                "veya backend='openai-whisper' kullanın."
            ) from exc

        print(
            f"[FASTER-WHISPER] Model yükleniyor: '{model_size}' "
            f"({resolved_device}/{resolved_compute})..."
        )
        shared_model = Path(__file__).resolve().parents[3] / "samet" / "models" / "whisper"
        model_source = str(shared_model) if model_size == "small" and (shared_model / "model.bin").exists() else model_size
        model = WhisperModel(
            model_source,
            device=resolved_device,
            compute_type=resolved_compute,
        )
    elif backend == "openai-whisper":
        import whisper

        print(f"[WHISPER] Model yükleniyor: '{model_size}' ({resolved_device})...")
        model = whisper.load_model(model_size, device=resolved_device)
    else:
        raise ValueError(f"Desteklenmeyen Whisper backend'i: {backend}")

    _whisper_models[cache_key] = model
    return model


def _serialize_faster_segment(segment: Any) -> Dict[str, Any]:
    words = []
    for word in segment.words or []:
        words.append({
            "start": round(float(word.start), 3),
            "end": round(float(word.end), 3),
            "word": str(word.word),
            "probability": round(float(word.probability), 4),
        })
    return {
        "id": int(segment.id),
        "start": round(float(segment.start), 3),
        "end": round(float(segment.end), 3),
        "text": str(segment.text).strip(),
        "avg_logprob": float(segment.avg_logprob),
        "no_speech_prob": float(segment.no_speech_prob),
        "words": words,
    }


def transcribe_audio_with_whisper(
    audio_wav_path: str,
    model_size: str = "small",
    *,
    backend: str = "faster-whisper",
    device: str = "auto",
    compute_type: Optional[str] = None,
    beam_size: int = 1,
    word_timestamps: bool = True,
    vad_filter: bool = True,
    language: Optional[str] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Transcribe audio into a backend-independent segment/word schema."""
    wav_path = Path(audio_wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_wav_path}")

    if backend == "faster-whisper":
        resolved_device = resolve_whisper_device(device)
        devices = [resolved_device] + (["cpu"] if resolved_device == "cuda" else [])
        for selected_device in devices:
            try:
                model = get_whisper_model(
                    model_size=model_size,
                    backend=backend,
                    device=selected_device,
                    compute_type="int8" if selected_device == "cpu" else compute_type,
                )
                segment_iterator, info = model.transcribe(
                    str(wav_path),
                    language=language,
                    beam_size=beam_size,
                    best_of=max(1, beam_size),
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                    condition_on_previous_text=False,
                )
                # CTranslate2 may only load libcublas while iterating.
                segments = []
                for segment in segment_iterator:
                    if cancel_check:
                        cancel_check()
                    segments.append(_serialize_faster_segment(segment))
                break
            except Exception as exc:
                error_text = str(exc).lower()
                gpu_failure = any(marker in error_text for marker in (
                    "libcublas", "libcudnn", "cuda", "cublas", "cudnn", "out of memory",
                ))
                if selected_device != "cuda" or not gpu_failure:
                    raise
                print(f"[FASTER-WHISPER] CUDA kullanılamıyor ({type(exc).__name__}); CPU/int8'e geçiliyor.")
        detected_language = info.language or language or "unknown"
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    else:
        model = get_whisper_model(
            model_size=model_size,
            backend=backend,
            device=device,
            compute_type=compute_type,
        )
        result = model.transcribe(
            str(wav_path),
            verbose=False,
            language=language,
            word_timestamps=word_timestamps,
            beam_size=beam_size,
        )
        segments = result.get("segments", [])
        detected_language = result.get("language", language or "unknown")
        language_probability = 0.0

    return {
        "backend": backend,
        "model": model_size,
        "text": " ".join(segment.get("text", "").strip() for segment in segments).strip(),
        "language": detected_language,
        "language_probability": round(language_probability, 4),
        "segments": segments,
    }


def save_transcript(result: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


def map_transcript_to_scenes(
    whisper_result: Dict[str, Any],
    scenes_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map timestamped Whisper segments to shots and compute speech features."""
    segments = whisper_result.get("segments", [])
    scene_transcripts = []

    for scene in scenes_data:
        start_sec = scene["start_seconds"]
        end_sec = scene["end_seconds"]
        duration = scene["duration_seconds"]
        scene_text_parts = []
        scene_words = []
        speech_intervals = []

        for segment in segments:
            segment_start = float(segment.get("start", 0.0))
            segment_end = float(segment.get("end", 0.0))
            overlap_start = max(start_sec, segment_start)
            overlap_end = min(end_sec, segment_end)

            if overlap_end <= overlap_start:
                continue

            words = segment.get("words") or []
            overlapping_words = []
            for word in words:
                word_start = float(word.get("start", segment_start))
                word_end = float(word.get("end", segment_end))
                if word_end > start_sec and word_start < end_sec:
                    overlapping_words.append(word)
                    speech_intervals.append((
                        max(start_sec, word_start),
                        min(end_sec, word_end),
                    ))

            if words:
                # Faster-Whisper can occasionally emit a very long segment whose
                # text contains only a handful of timestamped words. Treating the
                # whole segment as speech incorrectly marks every intervening shot
                # as dialogue. Word timestamps are the authoritative boundaries.
                if overlapping_words:
                    scene_words.extend(overlapping_words)
                    word_text = " ".join(
                        str(word.get("word", "")).strip()
                        for word in overlapping_words
                        if str(word.get("word", "")).strip()
                    ).strip()
                    if word_text:
                        scene_text_parts.append(word_text)
            else:
                # Models/configurations without word timestamps retain the safe
                # segment-level fallback.
                speech_intervals.append((overlap_start, overlap_end))
                text = segment.get("text", "").strip()
                if text:
                    scene_text_parts.append(text)

        speech_intervals.sort()
        merged_intervals = []
        for interval_start, interval_end in speech_intervals:
            if merged_intervals and interval_start <= merged_intervals[-1][1]:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval_end)
            else:
                merged_intervals.append([interval_start, interval_end])

        speech_time = sum(end - start for start, end in merged_intervals)
        speech_ratio = round(min(speech_time / duration, 1.0), 4) if duration > 0 else 0.0
        combined_text = " ".join(scene_text_parts).strip()
        word_count = len(scene_words) if scene_words else len(combined_text.split())
        density = round(word_count / duration, 4) if duration > 0 else 0.0

        scene_transcripts.append({
            "scene_id": scene["scene_id"],
            "start_seconds": start_sec,
            "end_seconds": end_sec,
            "duration_seconds": duration,
            "speech_time_seconds": round(speech_time, 3),
            "speech_ratio": speech_ratio,
            "word_count": word_count,
            "transcript_density": density,
            "transcript_text": combined_text,
            "words": scene_words,
        })

    return scene_transcripts
