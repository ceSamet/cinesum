from __future__ import annotations

import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass
class SpeechTurn:
    start_sec: float
    end_sec: float
    text: str
    speaker: str = "Konuşmacı 1"
    actor: str | None = None


def extract_audio(video_path: Path, wav_path: Path) -> bool:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    return process.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44


def transcribe_audio(
    wav_path: Path,
    model_path: str,
    device: str = "auto",
    language: str | None = None,
) -> tuple[list[SpeechTurn], str | None]:
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    def run(selected_device: str) -> tuple[list[SpeechTurn], str | None]:
        compute_type = "float16" if selected_device == "cuda" else "int8"
        model = WhisperModel(model_path, device=selected_device, compute_type=compute_type)
        segments, info = model.transcribe(
            str(wav_path),
            language=language,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
        )
        turns = [
            SpeechTurn(round(float(item.start), 3), round(float(item.end), 3), item.text.strip())
            for item in segments
            if item.text.strip()
        ]
        return turns, getattr(info, "language", None)

    try:
        return run(device)
    except Exception:
        if device != "cuda":
            raise
        # CTranslate2 allocates outside PyTorch, so torch's free-memory reading
        # can occasionally be optimistic. CPU int8 is still comfortably fast.
        return run("cpu")


def _voice_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    if len(samples) < sample_rate // 2:
        samples = np.pad(samples, (0, sample_rate // 2 - len(samples)))
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    vector = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1), delta.mean(axis=1)])
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _face_agreement(labels: np.ndarray, hints: list[str | None]) -> float | None:
    """Measure how well voice clusters preserve conservative face hints."""
    pairs = [(int(label), hint) for label, hint in zip(labels, hints) if hint]
    if len(pairs) < 3 or len({hint for _, hint in pairs}) < 1:
        return None
    clusters: dict[int, dict[str, int]] = {}
    actors: dict[str, dict[int, int]] = {}
    for label, actor in pairs:
        clusters.setdefault(label, {})[actor] = clusters.setdefault(label, {}).get(actor, 0) + 1
        actors.setdefault(actor, {})[label] = actors.setdefault(actor, {}).get(label, 0) + 1
    purity = sum(max(votes.values()) for votes in clusters.values()) / len(pairs)
    cohesion = sum(max(votes.values()) / sum(votes.values()) for votes in actors.values()) / len(actors)
    return (purity + cohesion) / 2.0


def _attach_actor_labels(
    turns: list[SpeechTurn],
    usable: list[int],
    labels: np.ndarray,
    actor_hints: list[str | None],
) -> None:
    """Attach an actor only when one voice cluster has a decisive face vote."""
    votes: dict[int, dict[str, int]] = {}
    for source_index, label in zip(usable, labels):
        actor = actor_hints[source_index]
        if actor:
            bucket = votes.setdefault(int(label), {})
            bucket[actor] = bucket.get(actor, 0) + 1

    candidates: list[tuple[float, int, int, str]] = []
    for label, bucket in votes.items():
        actor, count = max(bucket.items(), key=lambda item: item[1])
        total = sum(bucket.values())
        confidence = count / total
        if count >= 2 and confidence >= 0.72:
            candidates.append((confidence, count, label, actor))

    # One face identity may name only one voice cluster; weaker duplicate
    # mappings remain anonymous instead of inventing two voices for one actor.
    actor_to_label: dict[str, int] = {}
    label_to_actor: dict[int, str] = {}
    for _, _, label, actor in sorted(candidates, reverse=True):
        if actor not in actor_to_label:
            actor_to_label[actor] = label
            label_to_actor[label] = actor
    for source_index, label in zip(usable, labels):
        turns[source_index].actor = label_to_actor.get(int(label))


def assign_speakers(
    wav_path: Path,
    turns: list[SpeechTurn],
    max_speakers: int = 8,
    actor_hints: list[str | None] | None = None,
    timings: dict[str, float] | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> int:
    """Cluster voice prints, using reliable on-screen faces as a weak prior."""
    if not turns:
        return 0
    if actor_hints is None or len(actor_hints) != len(turns):
        actor_hints = [None] * len(turns)
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    load_started = time.perf_counter()
    audio, sample_rate = librosa.load(str(wav_path), sr=16000, mono=True)
    if timings is not None:
        timings["voice_audio_load_sec"] = round(time.perf_counter() - load_started, 3)
    usable: list[int] = []
    features: list[np.ndarray] = []
    feature_started = time.perf_counter()
    for index, turn in enumerate(turns):
        if cancel_check and index % 16 == 0:
            cancel_check()
        start = max(0, int(turn.start_sec * sample_rate))
        end = min(len(audio), int(turn.end_sec * sample_rate))
        if end - start < sample_rate * 0.45:
            continue
        usable.append(index)
        features.append(_voice_features(audio[start:end], sample_rate))
    if timings is not None:
        timings["voice_mfcc_sec"] = round(time.perf_counter() - feature_started, 3)
    if len(features) < 3:
        return 1

    matrix = StandardScaler().fit_transform(np.asarray(features))
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    best_labels = np.zeros(len(matrix), dtype=int)
    best_score = -1.0
    upper = min(max_speakers, len(matrix) - 1, max(2, round(len(matrix) ** 0.5)))
    cluster_started = time.perf_counter()
    for count in range(2, upper + 1):
        if cancel_check:
            cancel_check()
        try:
            labels = AgglomerativeClustering(n_clusters=count, metric="cosine", linkage="average").fit_predict(matrix)
            if len(set(labels)) < 2:
                continue
            counts = np.bincount(labels)
            # A tiny outlier cluster is not a real second speaker. Without this
            # guard, noisy turns can produce distributions such as 721 vs 2.
            minimum_cluster = max(3, math.ceil(len(matrix) * 0.015))
            if counts.min() < minimum_cluster:
                continue
            score = silhouette_score(matrix, labels, metric="cosine")
        except ValueError:
            continue
        # A modest complexity penalty avoids inventing a speaker for every turn.
        balance = float(counts.min() / counts.max())
        adjusted = float(score) + min(0.08, balance * 0.12) - count * 0.025
        agreement = _face_agreement(
            labels, [actor_hints[source_index] for source_index in usable]
        )
        if agreement is not None:
            adjusted += (agreement - 0.5) * 0.12
        if adjusted > best_score:
            best_score, best_labels = adjusted, labels
    if timings is not None:
        timings["voice_clustering_sec"] = round(time.perf_counter() - cluster_started, 3)
    if best_score < 0.035:
        best_labels[:] = 0

    for source_index, label in zip(usable, best_labels):
        turns[source_index].speaker = f"Konuşmacı {int(label) + 1}"
    _attach_actor_labels(turns, usable, best_labels, actor_hints)
    # Very short turns inherit the nearest temporal speaker.
    for index, turn in enumerate(turns):
        if index in usable:
            continue
        nearest = min(usable, key=lambda other: abs(turn.start_sec - turns[other].start_sec))
        turn.speaker = turns[nearest].speaker
        turn.actor = turns[nearest].actor
    return len({turn.speaker for turn in turns})


EVENT_TRANSLATIONS = {
    "Explosion": "patlama", "Boom": "patlama", "Gunshot, gunfire": "silah sesi",
    "Machine gun": "makineli tüfek", "Screaming": "çığlık", "Shout": "bağırma",
    "Yell": "bağırma", "Crying, sobbing": "ağlama", "Laughter": "kahkaha",
    "Music": "müzik", "Silence": "sessizlik", "Vehicle": "araç sesi",
    "Thunder": "gök gürültüsü", "Glass": "cam sesi", "Breaking": "kırılma",
}


def analyze_audio_events(
    wav_path: Path,
    scenes,
    model_path: str,
    max_windows: int = 96,
    cancel_check=None,
) -> dict[int, list[dict]]:
    """Run AudioSet classification once per scene, capped for predictable runtime."""
    import librosa
    from transformers import pipeline

    audio, sample_rate = librosa.load(str(wav_path), sr=16000, mono=True)
    if not len(audio):
        return {}
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            pipeline_device = 0 if free_bytes >= 1.2 * 1024**3 else -1
        else:
            pipeline_device = -1
    except Exception:
        pipeline_device = -1
    classifier = pipeline("audio-classification", model=model_path, device=pipeline_device)
    chosen = scenes if len(scenes) <= max_windows else scenes[:: max(1, len(scenes) // max_windows)][:max_windows]
    output: dict[int, list[dict]] = {}
    for scene in chosen:
        if cancel_check and cancel_check():
            break
        center = (scene.start_sec + scene.end_sec) / 2
        start_sec = max(scene.start_sec, center - 3.5)
        end_sec = min(scene.end_sec, center + 3.5)
        clip = audio[int(start_sec * sample_rate): int(end_sec * sample_rate)]
        if len(clip) < sample_rate // 2:
            continue
        predictions = classifier({"array": clip, "sampling_rate": sample_rate}, top_k=8)
        events = []
        for prediction in predictions:
            label = prediction["label"]
            translated = next((tr for en, tr in EVENT_TRANSLATIONS.items() if en.lower() in label.lower()), None)
            if translated and float(prediction["score"]) >= 0.04:
                events.append({"label": translated, "score": round(float(prediction["score"]), 3)})
        output[scene.index] = events[:3]
    return output


def turns_for_scene(turns: list[SpeechTurn], start: float, end: float) -> list[SpeechTurn]:
    return [turn for turn in turns if min(end, turn.end_sec) - max(start, turn.start_sec) > 0.05]
