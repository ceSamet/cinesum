from __future__ import annotations

import gc
import copy
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .audio_analysis import (
    SpeechTurn,
    analyze_audio_events,
    assign_speakers,
    extract_audio,
    transcribe_audio,
    turns_for_scene,
)
from .captioners import create_captioner
from .face_analysis import analyze_faces
from .scene_detector import DetectedScene, extract_scene_keyframes


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


class AnalysisCancelled(RuntimeError):
    pass


def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check and cancel_check():
        raise AnalysisCancelled("Analiz kullanıcı tarafından durduruldu.")


@dataclass
class SceneResult:
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    keyframe_sec: float
    frame_path: str
    description: str
    semantic_summary: str = ""
    summary_start_sec: float | None = None
    summary_end_sec: float | None = None
    visual_change_score: float = 0.0
    transcript: list[dict] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    actors: list[str] = field(default_factory=list)
    faces: list[dict] = field(default_factory=list)
    audio_events: list[dict] = field(default_factory=list)
    importance_score: float = 0.0
    selected: bool = False
    selection_rank: int | None = None
    selection_reasons: list[str] = field(default_factory=list)
    scoring_breakdown: dict = field(default_factory=dict)
    credit_probability: float = 0.0


def _model_path(name: str, fallback: str) -> str:
    local = MODEL_DIR / name
    if local.exists() and any(local.iterdir()):
        return str(local)
    raise FileNotFoundError(
        f"{name} yerel modeli bulunamadı; otomatik model indirme devre dışı."
    )


def _release_models() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _preferred_device(min_free_gb: float) -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free_bytes, _ = torch.cuda.mem_get_info()
        return "cuda" if free_bytes >= min_free_gb * 1024**3 else "cpu"
    except Exception:
        return "cpu"


def _caption_scenes(
    scenes: list[DetectedScene],
    backend: str,
    on_caption: Callable[[int, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[str]:
    if backend == "mock":
        return ["Görsel açıklama modeli devre dışı; sahne ana karesi hazırlandı." for _ in scenes]
    if backend == "llava":
        model = _model_path("llava", "llava-hf/llava-interleave-qwen-0.5b-hf")
    elif backend == "blip":
        model = _model_path("blip", "Salesforce/blip-image-captioning-base")
    else:
        model = None
    minimum_free = 2.7 if backend == "llava" else 1.2
    captioner = create_captioner(
        backend, model_name=model, device=_preferred_device(minimum_free)
    )
    prompt = (
        "What is happening in this movie image? Describe only visible people, "
        "place, and action in one concrete factual sentence."
    )
    retry_prompt = "What is visibly happening in this image? Answer with one factual sentence."

    def clean_description(text: str) -> str:
        text = " ".join(text.strip().split())
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return sentences[0].strip() if sentences else text

    def invalid_description(text: str) -> bool:
        lowered = text.casefold()
        blocked = (
            "i'm sorry", "sorry,", "cannot help", "can't help", "cannot assist",
            "üzgünüm", "yardımcı olamam", "bilgi vermek için buradayım",
            "this film scene is being used",
        )
        return len(text.split()) < 4 or any(phrase in lowered for phrase in blocked)

    descriptions: list[str] = []
    for index, scene in enumerate(scenes):
        _check_cancelled(cancel_check)
        try:
            description = clean_description(captioner.caption(scene.frame_path, prompt=prompt))
            if invalid_description(description):
                description = clean_description(
                    captioner.caption(scene.frame_path, prompt=retry_prompt)
                )
            if invalid_description(description):
                description = "The scene contains people in a movie setting."
        except Exception:
            description = "The movie scene is visible, but its details could not be described."
        descriptions.append(description)
        if on_caption:
            on_caption(index, description)
    del captioner
    _release_models()

    return descriptions


SCORE_WEIGHTS = {
    "dialogue": 24,
    "audio": 16,
    "character": 16,
    "visual_change": 12,
    "semantic_uniqueness": 21,
    "event_structure": 11,
}

SCORE_LABELS = {
    "dialogue": "Konuşma yoğunluğu",
    "audio": "Önemli ses olayı",
    "character": "Karakter görünürlüğü",
    "visual_change": "Görsel değişim",
    "semantic_uniqueness": "Semantik özgünlük",
    "event_structure": "Olay yapısı ve sınırı",
}

SELECTION_SCENE_TYPES = {"action", "dialogue", "silent", "music"}
SELECTION_CHARACTER_ROLES = {"main", "supporting"}


def _normalize_selection_request(request: dict | None) -> dict:
    request = request or {}
    prompt = " ".join(str(request.get("prompt") or "").split())[:1000]
    lowered = prompt.casefold()
    scene_types = {
        str(value).strip().casefold()
        for value in request.get("scene_types", [])
        if str(value).strip().casefold() in SELECTION_SCENE_TYPES
    }
    inferred = {
        "action": ("aksiyon", "action", "dövüş", "kovalamaca", "çatışma"),
        "dialogue": ("konuşma", "konuşmalı", "diyalog", "dialogue"),
        "silent": ("konuşmasız", "sessiz", "silent", "no dialogue"),
        "music": ("müzik", "müzikli", "music"),
    }
    for key, words in inferred.items():
        if any(word in lowered for word in words):
            scene_types.add(key)
    character_roles = {
        str(value).strip().casefold()
        for value in request.get("character_roles", [])
        if str(value).strip().casefold() in SELECTION_CHARACTER_ROLES
    }
    people = [
        value.strip()[:80]
        for value in re.split(r"[,;\n]+", str(request.get("people") or ""))
        if value.strip()
    ][:12]
    strict = bool(request.get("strict")) or any(
        marker in lowered for marker in ("sadece", "yalnız", "yalnızca", "only ")
    )
    return {
        "prompt": prompt,
        "scene_types": sorted(scene_types),
        "character_roles": sorted(character_roles),
        "people": people,
        "strict": strict,
    }


def _scene_type_flags(scene: SceneResult) -> set[str]:
    text = " ".join([
        scene.description or "",
        *(str(item.get("label") or "") for item in scene.audio_events),
    ]).casefold()
    action_words = (
        "fight", "run", "chase", "gun", "shoot", "explosion", "attack", "punch",
        "kick", "crash", "dövş", "koş", "kovala", "silah", "ateş", "patlama",
        "saldır", "yumruk", "tekme", "çarpış",
    )
    music_words = ("music", "müzik", "sing", "song", "şarkı")
    flags: set[str] = set()
    if scene.transcript:
        flags.add("dialogue")
    else:
        flags.add("silent")
    if any(word in text for word in action_words) or scene.visual_change_score >= 0.72:
        flags.add("action")
    if any(word in text for word in music_words):
        flags.add("music")
    return flags


def _selection_profile(results: list[SceneResult]) -> dict:
    """Infer an observable editing profile without guessing a screenplay template."""
    count = max(1, len(results))
    flags = [_scene_type_flags(scene) for scene in results]
    dialogue_ratio = sum("dialogue" in item for item in flags) / count
    action_ratio = sum("action" in item for item in flags) / count
    music_ratio = sum("music" in item for item in flags) / count
    silent_ratio = sum("silent" in item for item in flags) / count
    visual_activity = float(np.mean([scene.visual_change_score for scene in results] or [0.0]))
    actors = {actor for scene in results for actor in scene.actors}
    actor_presence = {
        actor: sum(actor in scene.actors for scene in results) / count for actor in actors
    }
    actor_dominance = max(actor_presence.values(), default=0.0)

    if action_ratio >= 0.34:
        key = "action_led"
        label = "Hareket ve eylem ağırlıklı"
        weights = {
            "dialogue": 12, "audio": 23, "character": 12,
            "visual_change": 23, "semantic_uniqueness": 18, "event_structure": 12,
        }
    elif silent_ratio >= 0.58 and (visual_activity >= 0.35 or music_ratio >= 0.25):
        key = "atmospheric"
        label = "Atmosfer ve görsel motif ağırlıklı"
        weights = {
            "dialogue": 8, "audio": 16, "character": 10,
            "visual_change": 24, "semantic_uniqueness": 30, "event_structure": 12,
        }
    elif len(actors) >= 4 and actor_dominance < 0.58:
        key = "ensemble"
        label = "Çok karakterli / paralel hatlı"
        weights = {
            "dialogue": 21, "audio": 12, "character": 25,
            "visual_change": 10, "semantic_uniqueness": 20, "event_structure": 12,
        }
    elif dialogue_ratio >= 0.62:
        key = "dialogue_led"
        label = "Diyalog ve ilişki ağırlıklı"
        weights = {
            "dialogue": 31, "audio": 11, "character": 16,
            "visual_change": 8, "semantic_uniqueness": 22, "event_structure": 12,
        }
    else:
        key = "mixed"
        label = "Karma / şablonsuz"
        weights = dict(SCORE_WEIGHTS)
    return {
        "key": key,
        "label": label,
        "weights": weights,
        "signals": {
            "dialogue_ratio": round(dialogue_ratio, 3),
            "action_ratio": round(action_ratio, 3),
            "silent_ratio": round(silent_ratio, 3),
            "music_ratio": round(music_ratio, 3),
            "actor_dominance": round(actor_dominance, 3),
        },
    }


def _knapsack_select(values: list[float], costs: list[int], budget: int) -> list[int]:
    """Return the maximum-value subset under an integer duration budget."""
    if not values or budget <= 0:
        return []
    dp = np.zeros(budget + 1, dtype=np.float32)
    take = np.zeros((len(values), budget + 1), dtype=np.bool_)
    for index, (value, cost) in enumerate(zip(values, costs)):
        if cost > budget:
            continue
        previous = dp.copy()
        candidates = previous[: budget + 1 - cost] + value
        better = candidates > dp[cost:] + 1e-6
        dp[cost:][better] = candidates[better]
        take[index, cost:][better] = True
    cursor = int(np.argmax(dp))
    selected: list[int] = []
    for index in range(len(values) - 1, -1, -1):
        cost = costs[index]
        if cursor >= cost and take[index, cursor]:
            selected.append(index)
            cursor -= cost
    return list(reversed(selected))


def _speech_safe_bounds(
    start: float,
    end: float,
    turns: list[SpeechTurn],
    duration: float,
    padding: float = 0.22,
) -> tuple[float, float]:
    """Expand a shot until neither boundary falls inside a detected utterance."""
    start = max(0.0, float(start))
    end = min(float(duration), float(end))
    for _ in range(len(turns) + 1):
        previous = (start, end)
        for turn in turns:
            if turn.start_sec < start < turn.end_sec:
                start = max(0.0, turn.start_sec - padding)
            if turn.start_sec < end < turn.end_sec:
                end = min(float(duration), turn.end_sec + padding)
        if abs(start - previous[0]) < 1e-6 and abs(end - previous[1]) < 1e-6:
            break
    return round(start, 3), round(end, 3)


def _speech_safe_intervals(
    selected,
    turns: list[SpeechTurn],
    duration: float,
) -> list[tuple[float, float]]:
    """Build chronological, non-overlapping highlight intervals with whole speech turns."""
    intervals = []
    for scene in selected:
        start = scene.summary_start_sec if scene.summary_start_sec is not None else scene.start_sec
        end = scene.summary_end_sec if scene.summary_end_sec is not None else scene.end_sec
        intervals.append(_speech_safe_bounds(start, end, turns, duration))
    intervals.sort()
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.08:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [
        _speech_safe_bounds(start, end, turns, duration)
        for start, end in merged
    ]


def _adaptive_diversity_select(
    values: list[float],
    costs: list[int],
    positions: list[float],
    budget: int,
    excluded: set[int] | None = None,
    event_groups: list[int] | None = None,
) -> list[int]:
    """Select globally, then softly reduce event/time concentration.

    Time windows are an anti-collapse signal only. They neither represent acts
    nor force weak material from an otherwise empty part of the video.
    """
    excluded = excluded or set()
    candidates = [index for index in range(len(values)) if index not in excluded]
    if not candidates:
        return []
    groups = event_groups or list(range(len(values)))
    adjusted = [max(0.0, float(value)) for value in values]
    selected: list[int] = []

    # Iteratively penalize only concentrations that actually appear in the
    # selected set. Empty timeline windows never receive a compulsory quota.
    for _ in range(5):
        local = _knapsack_select(
            [adjusted[index] for index in candidates],
            [costs[index] for index in candidates],
            budget,
        )
        selected = [candidates[index] for index in local]
        if not selected:
            break
        changed = False
        time_members: dict[int, list[int]] = {}
        event_members: dict[int, list[int]] = {}
        for index in selected:
            time_bin = min(9, int(max(0.0, min(0.999999, positions[index])) * 10))
            time_members.setdefault(time_bin, []).append(index)
            event_members.setdefault(groups[index], []).append(index)

        for members, share_limit, decay in (
            (time_members, 0.34, 0.86),
            (event_members, 0.28, 0.78),
        ):
            for indices in members.values():
                duration = sum(costs[index] for index in indices)
                if duration <= max(1, budget * share_limit):
                    continue
                ranked = sorted(
                    indices,
                    key=lambda index: values[index] / max(1, costs[index]),
                    reverse=True,
                )
                running = 0
                for rank, index in enumerate(ranked):
                    running += costs[index]
                    if rank == 0 or running <= budget * share_limit:
                        continue
                    new_value = max(values[index] * 0.30, adjusted[index] * decay)
                    if new_value < adjusted[index] - 1e-6:
                        adjusted[index] = new_value
                        changed = True
        if not changed:
            break
    return sorted(selected)


def _temporally_balanced_select(
    values: list[float],
    costs: list[int],
    positions: list[float],
    budget: int,
    excluded: set[int] | None = None,
) -> list[int]:
    """Compatibility alias for the former API; no three-act quota is imposed."""
    return _adaptive_diversity_select(values, costs, positions, budget, excluded)


def _visual_embeddings(results: list[SceneResult]) -> np.ndarray:
    """Cheap pixel-level appearance signal using existing OpenCV only."""
    import cv2

    vectors = []
    for scene in results:
        image = cv2.imread(scene.frame_path)
        if image is None:
            vectors.append(np.zeros(256, dtype=np.float32))
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
        norm = float(np.linalg.norm(histogram))
        vectors.append(histogram / max(norm, 1e-8))
    return np.asarray(vectors, dtype=np.float32)


def _project_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Project high-dimensional scene vectors to stable, normalized PCA coordinates."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if not len(matrix):
        return np.empty((0, 2), dtype=np.float32), [0.0, 0.0]
    if len(matrix) == 1:
        return np.asarray([[0.5, 0.5]], dtype=np.float32), [1.0, 0.0]
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        left, singular, _ = np.linalg.svd(centered, full_matrices=False)
        raw = left[:, :2] * singular[:2]
        if raw.shape[1] == 1:
            raw = np.column_stack([raw[:, 0], np.zeros(len(raw), dtype=np.float32)])
        variance = singular ** 2
        total_variance = max(float(variance.sum()), 1e-8)
        explained = [
            round(float(variance[index] / total_variance), 4) if index < len(variance) else 0.0
            for index in range(2)
        ]
    except np.linalg.LinAlgError:
        raw = np.column_stack([np.arange(len(matrix), dtype=np.float32), np.zeros(len(matrix))])
        explained = [0.0, 0.0]

    normalized = np.empty_like(raw, dtype=np.float32)
    for axis in range(2):
        values = raw[:, axis]
        if len(values) >= 10:
            low, high = np.percentile(values, [2, 98])
        else:
            low, high = float(values.min()), float(values.max())
        if high - low < 1e-8:
            normalized[:, axis] = 0.5
        else:
            normalized[:, axis] = 0.06 + 0.88 * np.clip((values - low) / (high - low), 0.0, 1.0)
    return normalized, explained


def _build_embedding_map(
    results: list[SceneResult],
    embeddings: np.ndarray,
    story_scenes: list[dict],
) -> dict:
    coordinates, explained = _project_embeddings(embeddings)
    story_by_cut = {
        cut_index: story["index"]
        for story in story_scenes
        for cut_index in story.get("cut_indices", [])
    }
    points = [
        {
            "cut_index": scene.index,
            "story_index": story_by_cut.get(scene.index),
            "x": round(float(coordinates[index, 0]), 5),
            "y": round(float(coordinates[index, 1]), 5),
            "start_sec": scene.start_sec,
            "end_sec": scene.end_sec,
            "importance": scene.importance_score,
            "selected": scene.selected,
            "description": scene.description,
            "actors": scene.actors,
        }
        for index, scene in enumerate(results)
        if index < len(coordinates)
    ]
    point_by_cut = {point["cut_index"]: point for point in points}
    story_points = []
    for story in story_scenes:
        members = [
            point_by_cut[index]
            for index in story.get("cut_indices", [])
            if index in point_by_cut
        ]
        if not members:
            continue
        story_points.append({
            "story_index": story["index"],
            "x": round(sum(point["x"] for point in members) / len(members), 5),
            "y": round(sum(point["y"] for point in members) / len(members), 5),
            "label": story.get("label"),
            "description": story.get("description"),
            "cut_count": len(members),
            "start_sec": story.get("start_sec"),
            "end_sec": story.get("end_sec"),
            "selected": story.get("selected", False),
        })
    return {
        "method": "PCA",
        "explained_variance": explained,
        "points": points,
        "stories": story_points,
    }


def _credit_probability(scene: SceneResult, total_duration: float) -> float:
    """Estimate title/credit likelihood from position, text, dialogue and frame style."""
    import cv2

    midpoint = (scene.start_sec + scene.end_sec) / 2.0
    position = midpoint / max(total_duration, 1e-6)
    text = scene.description.casefold()
    keywords = (
        "credit", "credits", "jenerik", "cast and crew", "directed by",
        "produced by", "executive producer", "yönetmen", "yapımcı",
        "oyuncular", "başroller", "title card", "film title",
    )
    keyword_hit = any(keyword in text for keyword in keywords)
    no_dialogue = not scene.transcript
    score = 0.0
    if keyword_hit:
        score += 0.50
    if position >= 0.90:
        score += 0.22
    elif position <= 0.055:
        score += 0.10
    if no_dialogue:
        score += 0.13
    if not scene.actors:
        score += 0.05
    if any(item.get("label") == "müzik" for item in scene.audio_events):
        score += 0.05

    image = cv2.imread(scene.frame_path)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.mean(gray < 55))
        edge_ratio = float(np.mean(cv2.Canny(gray, 80, 180) > 0))
        if dark_ratio >= 0.62:
            score += 0.08
        if 0.015 <= edge_ratio <= 0.16:
            score += 0.04
    if scene.transcript:
        score -= 0.22
    if scene.actors:
        score -= 0.08
    # Position alone must never exclude a narrative opening or ending.
    if not keyword_hit and position < 0.90:
        score = min(score, 0.58)
    return round(max(0.0, min(1.0, score)), 3)


def _compact_model_copy(text: str, minimum: int, maximum: int) -> str:
    text = re.sub(
        r"(?i)^(başlık|title|açıklama|description)\s*[:\-]\s*",
        "",
        " ".join(text.replace('"', "").strip().split()),
    )
    words = text.split()
    if len(words) < minimum:
        return ""
    return " ".join(words[:maximum]).strip(" .,:;|-")


def _fallback_story_copy(group: list[SceneResult], index: int) -> tuple[str, str]:
    text = next(
        (scene.description for scene in group if scene.description and "hazırlanıyor" not in scene.description),
        "",
    )
    text = re.sub(
        r"(?i)^(the image shows|the scene shows|in the image,?|the image of|this image shows)\s+",
        "",
        text,
    )
    words = text.split()
    title = _compact_model_copy(" ".join(words[:3]), 1, 3) or f"Sahne {index}"
    description = _compact_model_copy(" ".join(words[:5]), 3, 5) or "Model açıklaması üretilemedi"
    return title, description


def _story_context(story: dict, group: list[SceneResult]) -> dict:
    """Build compact multimodal evidence for a text-only story summarizer."""
    observations = []
    dialogue = []
    seen_turns: set[tuple] = set()
    actors = []
    for scene in group:
        description = " ".join((scene.description or "").split())
        if description and "hazırlanıyor" not in description and description not in observations:
            observations.append(description[:360])
        for actor in scene.actors:
            if actor not in actors:
                actors.append(actor)
        for turn in scene.transcript:
            key = (turn.get("start_sec"), turn.get("end_sec"), turn.get("text"))
            if key in seen_turns:
                continue
            seen_turns.add(key)
            speaker = turn.get("actor") or turn.get("speaker") or "Konuşmacı"
            text = " ".join(str(turn.get("text") or "").split())
            if text:
                dialogue.append(f"{speaker}: {text[:300]}")
    return {
        "index": story["index"],
        "duration_sec": round(float(story.get("duration_sec") or 0), 1),
        "actors": actors[:8],
        "visual_observations": observations[:10],
        "dialogue": dialogue[:18],
    }


def _group_story_scenes(
    results: list[SceneResult],
    text_embeddings: np.ndarray | None = None,
    visual_embeddings: np.ndarray | None = None,
) -> list[dict]:
    """Group adjacent camera cuts into higher-level narrative scenes."""
    if not results:
        return []
    if visual_embeddings is None:
        visual_embeddings = _visual_embeddings(results)
    groups: list[list[int]] = []
    current = [0]

    for index in range(1, len(results)):
        previous = results[index - 1]
        scene = results[index]
        current_duration = results[current[-1]].end_sec - results[current[0]].start_sec
        actors_left, actors_right = set(previous.actors), set(scene.actors)
        actor_overlap = (
            len(actors_left & actors_right) / max(1, len(actors_left | actors_right))
            if actors_left or actors_right else 0.35
        )
        visual_similarity = float(visual_embeddings[index - 1] @ visual_embeddings[index])
        text_similarity = 0.5
        if text_embeddings is not None and len(text_embeddings) == len(results):
            text_similarity = float(text_embeddings[index - 1] @ text_embeddings[index])
        continuity = text_similarity * 0.52 + visual_similarity * 0.25 + actor_overlap * 0.23
        previous_credit = previous.credit_probability >= 0.65
        current_credit = scene.credit_probability >= 0.65
        credit_change = previous_credit != current_credit
        boundary = (
            credit_change
            or (current_duration >= 8.0 and continuity < 0.40)
            or current_duration >= 72.0
        )
        if boundary:
            groups.append(current)
            current = [index]
        else:
            current.append(index)
    groups.append(current)

    story_scenes = []
    for story_index, indices in enumerate(groups, 1):
        shots = [results[index] for index in indices]
        label, description = _fallback_story_copy(shots, story_index)
        story_scenes.append({
            "index": story_index,
            "label": label,
            "description": description,
            "label_source": "cut_caption",
            "start_sec": round(shots[0].start_sec, 3),
            "end_sec": round(shots[-1].end_sec, 3),
            "duration_sec": round(shots[-1].end_sec - shots[0].start_sec, 3),
            "cut_indices": [shot.index for shot in shots],
            "cut_count": len(shots),
            "selected": any(shot.selected for shot in shots),
        })
    return story_scenes


def _embed_and_select(
    results: list[SceneResult],
    budget_ratio: float,
    story_scenes_override: list[dict] | None = None,
    rag_database_path: Path | None = None,
    use_llm_selection: bool = False,
    cancel_check: Callable[[], bool] | None = None,
    embeddings_override: np.ndarray | None = None,
    selection_request: dict | None = None,
) -> tuple[np.ndarray, dict, list[dict]]:
    selection_request = _normalize_selection_request(selection_request)
    user_prompt = selection_request["prompt"]
    texts = []
    dialogue_values = []
    audio_values = []
    character_values = []
    dialogues = [" ".join(item["text"] for item in scene.transcript) for scene in results]
    actor_seconds: dict[str, float] = {}
    for scene in results:
        for actor in set(scene.actors):
            actor_seconds[actor] = actor_seconds.get(actor, 0.0) + scene.duration_sec
    maximum_actor_time = max(actor_seconds.values(), default=1.0)

    for index, scene in enumerate(results):
        dialogue = " ".join(item["text"] for item in scene.transcript)
        events = " ".join(item["label"] for item in scene.audio_events)
        context = " ".join(
            dialogues[other]
            for other in range(max(0, index - 1), min(len(results), index + 2))
            if other != index and dialogues[other]
        )
        # Repeating local dialogue keeps neighbouring context useful without
        # letting an adjacent scene dominate this scene's identity.
        texts.append(
            f"{scene.semantic_summary} {scene.description} {dialogue} {dialogue} "
            f"{events} Bağlam: {context}".strip()
            or "sessiz sahne"
        )
        dialogue_values.append(
            min(1.0, len(dialogue.split()) / max(6.0, scene.duration_sec * 2.3))
        )
        audio_values.append(
            min(1.0, 2.0 * max([item["score"] for item in scene.audio_events] or [0.0]))
        )
        character_values.append(max(
            [math.sqrt(actor_seconds.get(actor, 0.0) / maximum_actor_time) for actor in scene.actors]
            or [0.0]
        ))

    query_vector: np.ndarray | None = None
    if embeddings_override is not None and len(embeddings_override) == len(results):
        embeddings = np.asarray(embeddings_override, dtype=np.float32)
        if user_prompt:
            try:
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(_model_path(
                    "embeddings", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ))
                query_vector = np.asarray(
                    encoder.encode([user_prompt], normalize_embeddings=True, show_progress_bar=False)[0],
                    dtype=np.float32,
                )
                del encoder
                _release_models()
            except Exception:
                query_vector = None
    else:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(_model_path(
                "embeddings", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ))
            encoded = np.asarray(encoder.encode(
                texts + ([user_prompt] if user_prompt else []),
                normalize_embeddings=True,
                show_progress_bar=False,
            ))
            embeddings = encoded[:len(texts)]
            if user_prompt:
                query_vector = np.asarray(encoded[-1], dtype=np.float32)
            del encoder
            _release_models()
        except Exception:
            # Keeps the core pipeline usable in light installations.
            from sklearn.feature_extraction.text import TfidfVectorizer
            embeddings = TfidfVectorizer(max_features=512).fit_transform(texts).toarray()
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-8)

    query_relevance = np.zeros(len(results), dtype=np.float32)
    if user_prompt and query_vector is not None and len(query_vector) == embeddings.shape[1]:
        similarities = np.asarray(embeddings @ query_vector, dtype=np.float32)
        if similarities.max() > similarities.min():
            query_relevance = (
                (similarities - similarities.min())
                / (similarities.max() - similarities.min())
            )
        else:
            query_relevance.fill(0.5)
    elif user_prompt and results:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            matrix = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=2048
            ).fit_transform([*texts, user_prompt])
            query_relevance = np.asarray(
                (matrix[:-1] @ matrix[-1].T).toarray()
            ).reshape(-1).astype(np.float32)
            if query_relevance.max() > query_relevance.min():
                query_relevance = (
                    (query_relevance - query_relevance.min())
                    / (query_relevance.max() - query_relevance.min())
                )
        except Exception:
            pass

    if not results:
        return embeddings, {
            "algorithm": "0/1 Knapsack", "duration_budget_sec": 0,
            "selected_duration_sec": 0, "weights": SCORE_WEIGHTS,
        }, []

    visual_embeddings = _visual_embeddings(results)
    if len(results) == 1:
        uniqueness = np.ones(1, dtype=np.float32)
    else:
        text_similarities = np.asarray(embeddings @ embeddings.T, dtype=np.float32)
        visual_similarities = np.asarray(visual_embeddings @ visual_embeddings.T, dtype=np.float32)
        similarities = text_similarities * 0.78 + visual_similarities * 0.22
        np.fill_diagonal(similarities, -1.0)
        uniqueness = 1.0 - similarities.max(axis=1)
        if uniqueness.max() > uniqueness.min():
            uniqueness = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min())
        else:
            uniqueness = np.full(len(results), 0.5, dtype=np.float32)
    visual = np.asarray([scene.visual_change_score for scene in results], dtype=np.float32)
    if visual.max() > visual.min():
        visual = (visual - visual.min()) / (visual.max() - visual.min())
    elif len(visual):
        visual.fill(0.5)

    story_scenes = (
        copy.deepcopy(story_scenes_override)
        if story_scenes_override is not None
        else _group_story_scenes(results, embeddings, visual_embeddings)
    )
    source_by_cut = {scene.index: index for index, scene in enumerate(results)}
    story_by_source: dict[int, int] = {}
    story_members: dict[int, list[int]] = {}
    for fallback_group, story in enumerate(story_scenes):
        group_id = int(story.get("index", fallback_group + 1))
        members = [
            source_by_cut[cut] for cut in story.get("cut_indices", [])
            if cut in source_by_cut
        ]
        story_members[group_id] = members
        for source_index in members:
            story_by_source[source_index] = group_id
    event_groups = [story_by_source.get(index, -(index + 1)) for index in range(len(results))]

    # Structural salience comes from observed event boundaries and local change,
    # not from assumed screenplay beats at fixed percentages.
    event_structure = np.full(len(results), 0.35, dtype=np.float32)
    for index in range(len(results)):
        changes = []
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(results):
                semantic_change = 1.0 - float(embeddings[index] @ embeddings[neighbour])
                visual_change = 1.0 - float(visual_embeddings[index] @ visual_embeddings[neighbour])
                changes.append(max(0.0, min(1.0, semantic_change * 0.72 + visual_change * 0.28)))
        local_change = max(changes or [0.35])
        members = story_members.get(event_groups[index], [index])
        is_boundary = len(members) == 1 or index in (members[0], members[-1])
        event_structure[index] = min(1.0, 0.20 + 0.58 * local_change + (0.22 if is_boundary else 0.0))

    profile = _selection_profile(results)
    active_weights = profile["weights"]

    total_scores: list[float] = []
    costs: list[int] = []
    positions: list[float] = []
    total_duration = sum(scene.duration_sec for scene in results)
    selected_types = set(selection_request["scene_types"])
    selected_roles = set(selection_request["character_roles"])
    requested_people = [value.casefold() for value in selection_request["people"]]
    actor_ranking = sorted(actor_seconds, key=actor_seconds.get, reverse=True)
    main_actor_count = max(1, min(3, math.ceil(len(actor_ranking) * 0.25)))
    main_actors = set(actor_ranking[:main_actor_count])
    intent_active = bool(user_prompt or selected_types or selected_roles or requested_people)
    excluded: set[int] = set()
    for index, scene in enumerate(results):
        position = ((scene.start_sec + scene.end_sec) / 2.0) / max(total_duration, 1e-6)
        positions.append(position)
        scene.credit_probability = _credit_probability(scene, total_duration)
        raw = {
            "dialogue": float(dialogue_values[index]),
            "audio": float(audio_values[index]),
            "character": float(character_values[index]),
            "visual_change": float(visual[index]),
            "semantic_uniqueness": float(uniqueness[index]),
            "event_structure": float(event_structure[index]),
        }
        breakdown = {
            key: {
                "label": SCORE_LABELS[key],
                "raw": round(value, 3),
                "weight": weight,
                "points": round(value * weight, 1),
            }
            for key, (value, weight) in (
                (key, (raw[key], active_weights[key])) for key in active_weights
            )
        }
        type_flags = _scene_type_flags(scene)
        type_match = not selected_types or bool(type_flags & selected_types)
        role_match = (
            not selected_roles
            or ("main" in selected_roles and bool(set(scene.actors) & main_actors))
            or ("supporting" in selected_roles and bool(set(scene.actors) - main_actors))
        )
        searchable = " ".join([
            scene.description or "",
            *scene.actors,
            *(str(item.get("text") or "") for item in scene.transcript),
        ]).casefold()
        people_match = not requested_people or any(value in searchable for value in requested_people)
        explicit_scores = [
            value for enabled, value in (
                (bool(selected_types), 1.0 if type_match else 0.0),
                (bool(selected_roles), 1.0 if role_match else 0.0),
                (bool(requested_people), 1.0 if people_match else 0.0),
            ) if enabled
        ]
        intent_score = float(query_relevance[index]) if user_prompt else 0.0
        if explicit_scores:
            explicit_score = sum(explicit_scores) / len(explicit_scores)
            intent_score = max(intent_score, explicit_score) if user_prompt else explicit_score
        if intent_active:
            for item in breakdown.values():
                item["weight"] = round(item["weight"] * 0.65, 1)
                item["points"] = round(item["points"] * 0.65, 1)
            breakdown["user_intent"] = {
                "label": "Kullanıcı isteğine uygunluk",
                "raw": round(intent_score, 3),
                "weight": 35,
                "points": round(intent_score * 35, 1),
            }
        subtotal = sum(item["points"] for item in breakdown.values())
        total = round(subtotal * (1.0 - scene.credit_probability), 1)
        if scene.credit_probability:
            breakdown["credits"] = {
                "label": "Jenerik cezası",
                "raw": scene.credit_probability,
                "weight": 0,
                "points": round(total - subtotal, 1),
            }
        if scene.credit_probability >= 0.72:
            excluded.add(index)
        if selection_request["strict"]:
            explicit_mismatch = not (type_match and role_match and people_match)
            prompt_mismatch = bool(user_prompt and not explicit_scores and query_relevance[index] < 0.55)
            if explicit_mismatch or prompt_mismatch:
                excluded.add(index)
        scene.scoring_breakdown = breakdown
        scene.importance_score = round(total / 100.0, 3)
        total_scores.append(total)
        safe_start = scene.summary_start_sec if scene.summary_start_sec is not None else scene.start_sec
        safe_end = scene.summary_end_sec if scene.summary_end_sec is not None else scene.end_sec
        costs.append(max(1, math.ceil(safe_end - safe_start)))

    penalized_cut_count = 0
    for story in story_scenes:
        members = [source_by_cut[index] for index in story["cut_indices"] if index in source_by_cut]
        ranked = sorted(members, key=lambda index: total_scores[index], reverse=True)
        for repetition_rank, index in enumerate(ranked):
            flags = _scene_type_flags(results[index])
            free_count = 3 if "action" in flags else 2
            earlier = ranked[:repetition_rank]
            semantic_similarity = max(
                (
                    float(embeddings[index] @ embeddings[other]) * 0.78
                    + float(visual_embeddings[index] @ visual_embeddings[other]) * 0.22
                    for other in earlier
                ),
                default=0.0,
            )
            redundancy = max(0.0, min(1.0, (semantic_similarity - 0.52) / 0.48))
            penalty_ratio = 0.0 if repetition_rank < free_count else min(
                0.55, (repetition_rank - free_count + 1) * 0.14 * redundancy
            )
            if not penalty_ratio:
                continue
            original = total_scores[index]
            penalty = round(original * penalty_ratio, 1)
            total_scores[index] = round(original - penalty, 1)
            results[index].importance_score = round(total_scores[index] / 100.0, 3)
            results[index].scoring_breakdown["story_repetition"] = {
                "label": "Benzer olay tekrarı",
                "raw": round(redundancy, 3),
                "weight": 0,
                "points": -penalty,
            }
            penalized_cut_count += 1

    duration_budget = max(1, round(total_duration * budget_ratio))
    selected = _adaptive_diversity_select(
        total_scores, costs, positions, duration_budget, excluded, event_groups
    )
    llm_selection = {
        "status": "disabled",
        "provider": "ParalonCloud",
        "model": os.getenv("PARALON_MODEL", "qwen3.8-27b"),
    }
    if use_llm_selection and rag_database_path is not None:
        from .scene_rag import select_with_paralon_rag

        selected, llm_selection = select_with_paralon_rag(
            results=results,
            story_scenes=story_scenes,
            embeddings=embeddings,
            base_selected=selected,
            scores=total_scores,
            costs=costs,
            positions=positions,
            budget=duration_budget,
            excluded=excluded,
            database_path=rag_database_path,
            knapsack_select=_knapsack_select,
            cancel_check=cancel_check,
            selection_request=selection_request,
            selection_profile=profile,
            diversity_select=_adaptive_diversity_select,
        )
    selected_set = set(selected)
    for story in story_scenes:
        story["selected"] = any(
            source_by_cut.get(cut_index) in selected_set
            for cut_index in story["cut_indices"]
        )

    for rank, index in enumerate(selected, 1):
        scene = results[index]
        scene.selected = True
        scene.selection_rank = rank
        strongest = sorted(
            scene.scoring_breakdown.values(), key=lambda item: item["points"], reverse=True
        )[:2]
        llm_reasons = list(scene.selection_reasons)
        scene.selection_reasons = [
            *llm_reasons,
            *(f"{item['label']}: +{item['points']:.1f}" for item in strongest),
            (
                "Paralon önerisi süre bütçesine knapsack ile yerleştirildi"
                if llm_selection.get("status") == "applied"
                else "olay çeşitliliği korunarak süre bütçesine yerleşti"
            ),
        ]
    for index, scene in enumerate(results):
        if index not in selected_set:
            strongest = max(
                scene.scoring_breakdown.values(), key=lambda item: item["points"]
            )
            scene.selection_reasons = [
                f"en güçlü katkı {strongest['label']}: +{strongest['points']:.1f}",
                "olay çeşitliliği ve süre bütçesinde başka sahneler tercih edildi",
            ]
    selected_duration = sum(costs[index] for index in selected)
    temporal_distribution = [
        sum(costs[index] for index in selected if min(9, int(positions[index] * 10)) == bin_index)
        for bin_index in range(10)
    ]
    meta = {
        "algorithm": "İçerik uyarlamalı çeşitli Knapsack",
        "duration_budget_sec": duration_budget,
        "selected_duration_sec": round(selected_duration, 1),
        "total_video_duration_sec": round(total_duration, 1),
        "budget_ratio": round(budget_ratio, 3),
        "temporal_distribution_sec": temporal_distribution,
        "excluded_credit_scenes": sum(scene.credit_probability >= 0.72 for scene in results),
        "excluded_by_request": sum(
            index in excluded and scene.credit_probability < 0.72
            for index, scene in enumerate(results)
        ),
        "story_repetition_penalty": "Yalnız semantik olarak benzer olay tekrarlarına azalan getiri; aksiyonda üç, diğer yapılarda iki temsilciye kadar serbest",
        "story_penalized_cut_count": penalized_cut_count,
        "narrative_distribution": "Sabit perde veya dönüm noktası yok; 10 zaman penceresi yalnız aşırı yığılmayı yumuşak biçimde sınırlar",
        "selection_profile": profile,
        "selection_request": selection_request,
        "weights": [
            {"key": key, "label": SCORE_LABELS[key], "max_points": value}
            for key, value in active_weights.items()
        ],
        "objective": "Kullanıcı isteğine uyan, yeni bilgi taşıyan ve birbirini gereksiz tekrarlamayan olayları seç",
        "llm_selection": llm_selection,
    }
    return embeddings, meta, story_scenes


def _make_highlight(
    video_path: Path,
    results: list[SceneResult],
    output_path: Path,
    turns: list[SpeechTurn],
    duration: float,
    has_audio: bool = True,
) -> tuple[bool, list[tuple[float, float]]]:
    selected = [scene for scene in results if scene.selected]
    if not selected:
        return False, []
    intervals = _speech_safe_intervals(selected, turns, duration)
    if not intervals:
        return False, []
    filter_parts = []
    concat_inputs = []
    for index, (start, end) in enumerate(intervals):
        video_filter = f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{index}]"
        if has_audio:
            filter_parts.append(video_filter + ";" + f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{index}]")
            concat_inputs.append(f"[v{index}][a{index}]")
        else:
            filter_parts.append(video_filter)
            concat_inputs.append(f"[v{index}]")
    graph = ";".join(filter_parts) + ";" + "".join(concat_inputs) + f"concat=n={len(intervals)}:v=1:a={1 if has_audio else 0}[v]" + ("[a]" if has_audio else "")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
        "-filter_complex", graph, "-map", "[v]",
    ]
    if has_audio:
        command.extend(["-map", "[a]"])
    command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"])
    if has_audio:
        command.extend(["-c:a", "aac"])
    command.extend(["-movflags", "+faststart", str(output_path)])
    process = subprocess.run(command, capture_output=True, text=True)
    return process.returncode == 0 and output_path.exists(), intervals


def _video_has_audio_stream(video_path: Path) -> bool:
    """Inspect the source container instead of relying on cached WAV artifacts."""
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return process.returncode == 0 and bool(process.stdout.strip())


def _analysis_snapshot(
    results: list[SceneResult],
    duration: float,
    started: float,
    speaker_count: int = 0,
    language: str | None = None,
    cast: list[dict] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "duration_sec": round(duration, 2),
        "processing_sec": round(time.perf_counter() - started, 2),
        "scene_count": len(results),
        "selected_count": sum(scene.selected for scene in results),
        "speaker_count": speaker_count,
        "language": language,
        "cast": cast or [],
        "warnings": warnings or [],
        "highlight_path": None,
        "scenes": [asdict(scene) for scene in results],
    }


def _emit_progress(
    callback: Callable | None,
    message: str,
    analysis: dict | None = None,
    percent: int | None = None,
) -> None:
    if callback is None:
        return
    try:
        callback(message, analysis, percent)
    except TypeError:
        callback(message)


def _actor_hints_for_turns(
    turns: list[SpeechTurn],
    scenes: list[DetectedScene],
    actor_map: dict[int, list[str]],
) -> list[str | None]:
    """Return a face hint only when exactly one recurring actor is on screen."""
    hints: list[str | None] = []
    for turn in turns:
        midpoint = (turn.start_sec + turn.end_sec) / 2.0
        scene = next(
            (item for item in scenes if item.start_sec <= midpoint < item.end_sec),
            None,
        )
        actors = actor_map.get(scene.index, []) if scene else []
        hints.append(actors[0] if len(actors) == 1 else None)
    return hints


def analyze_video(
    video_path: Path,
    job_dir: Path,
    backend: str = "llava",
    budget_ratio: float = 0.22,
    language: str | None = None,
    audio_events: bool = True,
    selection_request: dict | None = None,
    progress: Callable | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    started = time.perf_counter()
    warnings: list[str] = []
    _check_cancelled(cancel_check)
    _emit_progress(progress, "Sahne geçişleri bulunuyor", percent=5)
    frames_dir = job_dir / "frames"
    scenes, duration = extract_scene_keyframes(video_path, frames_dir)
    results = [
        SceneResult(
            index=scene.index,
            start_sec=scene.start_sec,
            end_sec=scene.end_sec,
            duration_sec=round(scene.duration_sec, 3),
            keyframe_sec=scene.keyframe_sec,
            frame_path=str(scene.frame_path),
            description="Görsel açıklama hazırlanıyor…",
            visual_change_score=scene.cut_score,
        )
        for scene in scenes
    ]
    _emit_progress(
        progress,
        f"{len(scenes)} sahne bulundu",
        _analysis_snapshot(results, duration, started),
        15,
    )
    _check_cancelled(cancel_check)

    actor_map: dict[int, list[str]] = {}
    cast: list[dict] = []
    face_map: dict[int, list[dict]] = {}
    _emit_progress(progress, "Karakterler erkenden taranıyor", percent=17)
    try:
        actor_map, cast, face_map = analyze_faces(
            scenes,
            MODEL_DIR / "opencv",
            video_path=video_path,
            portraits_dir=job_dir / "portraits",
        )
        for result in results:
            result.actors = actor_map.get(result.index, [])
            result.faces = face_map.get(result.index, [])
        _emit_progress(
            progress,
            f"{len(cast)} tekrarlayan karakter bulundu",
            _analysis_snapshot(results, duration, started, cast=cast, warnings=warnings),
            28,
        )
    except Exception:
        warnings.append("Karakter görünürlüğü tamamlanamadı.")
    _check_cancelled(cancel_check)

    wav_path = job_dir / "audio.wav"
    turns: list[SpeechTurn] = []
    detected_language = None
    speaker_count = 0
    event_map: dict[int, list[dict]] = {}
    _emit_progress(progress, "Konuşmalar karakter ipuçlarıyla çözümleniyor", percent=30)
    has_audio = extract_audio(video_path, wav_path)
    if has_audio:
        try:
            turns, detected_language = transcribe_audio(
                wav_path,
                _model_path("whisper", "Systran/faster-whisper-small"),
                device="cpu",
                language=language,
            )
        except Exception:
            warnings.append("Konuşmalar bu videoda çözümlenemedi.")
        if turns:
            try:
                speaker_count = assign_speakers(
                    wav_path,
                    turns,
                    actor_hints=_actor_hints_for_turns(turns, scenes, actor_map),
                )
            except Exception:
                speaker_count = 1
                warnings.append("Konuşmacılar ayrı ayrı gruplanamadı.")
            for scene, result in zip(scenes, results):
                scene_turns = turns_for_scene(turns, scene.start_sec, scene.end_sec)
                result.transcript = [asdict(turn) for turn in scene_turns]
                result.speakers = sorted({turn.speaker for turn in scene_turns})
            _emit_progress(
                progress,
                f"{len(turns)} replik sahnelere yerleştirildi",
                _analysis_snapshot(
                    results, duration, started, speaker_count, detected_language,
                    cast=cast, warnings=warnings
                ),
                40,
            )
        _check_cancelled(cancel_check)
        if audio_events:
            _emit_progress(progress, "Ses olayları taranıyor", percent=43)
            try:
                event_map = analyze_audio_events(
                    wav_path,
                    scenes,
                    _model_path("audio-events", "MIT/ast-finetuned-audioset-10-10-0.4593"),
                    cancel_check=cancel_check,
                )
                _check_cancelled(cancel_check)
                for result in results:
                    result.audio_events = event_map.get(result.index, [])
            except AnalysisCancelled:
                raise
            except Exception:
                warnings.append("Bazı ses olayları tanımlanamadı.")
        _release_models()
    else:
        warnings.append("Bu videoda ses kanalı bulunamadı.")
    for result in results:
        result.summary_start_sec, result.summary_end_sec = _speech_safe_bounds(
            result.start_sec, result.end_sec, turns, duration
        )
    _check_cancelled(cancel_check)

    last_published = -1
    def publish_caption(index: int, description: str) -> None:
        nonlocal last_published
        results[index].description = description
        # Avoid rewriting a large job state for every single shot.
        if index == len(results) - 1 or index - last_published >= 4:
            last_published = index
            percent = 46 + round(27 * (index + 1) / max(1, len(results)))
            _emit_progress(
                progress,
                f"Sahne {index + 1}/{len(results)} görsel olarak açıklanıyor",
                _analysis_snapshot(
                    results, duration, started, speaker_count, detected_language,
                    cast=cast, warnings=warnings
                ),
                percent,
            )

    _emit_progress(progress, "Görsel açıklamalar hazırlanıyor", percent=46)
    try:
        _caption_scenes(
            scenes,
            backend,
            on_caption=publish_caption,
            cancel_check=cancel_check,
        )
    except AnalysisCancelled:
        raise
    except Exception:
        warnings.append("Görsel açıklamalar tamamlanamadı.")

    _check_cancelled(cancel_check)
    _emit_progress(progress, "Anlatısal sahneler gruplanıyor", percent=82)
    embeddings, _, story_scenes = _embed_and_select(
        results, min(0.5, max(0.05, budget_ratio))
    )
    _emit_progress(progress, "Sahne belgeleri RAG için hazırlanıyor", percent=86)
    by_cut = {scene.index: scene for scene in results}
    for story in story_scenes:
        semantic_summary = " ".join(
            part for part in (story.get("label"), story.get("description")) if part
        )
        for cut_index in story.get("cut_indices", []):
            if cut_index in by_cut:
                by_cut[cut_index].semantic_summary = semantic_summary
    # Re-embed deterministic story summaries, then let Paralon review the
    # duration-safe knapsack proposal with retrieved narrative context.
    _emit_progress(progress, "Paralon RAG bağlamıyla sahneleri seçiyor", percent=89)
    for result in results:
        result.selected = False
        result.selection_rank = None
        result.selection_reasons = []
        result.scoring_breakdown = {}
    for story in story_scenes:
        story["selected"] = False
    embeddings, selection_meta, story_scenes = _embed_and_select(
        results,
        min(0.5, max(0.05, budget_ratio)),
        story_scenes_override=story_scenes,
        rag_database_path=job_dir / "scene_rag.sqlite3",
        use_llm_selection=True,
        cancel_check=cancel_check,
        selection_request=selection_request,
    )
    llm_status = selection_meta.get("llm_selection", {}).get("status")
    if llm_status == "fallback_knapsack":
        reason = selection_meta["llm_selection"].get("reason", "bilinmeyen hata")
        warnings.append(f"Paralon seçimi kullanılamadı; knapsack seçimi korundu: {reason}")
    embedding_map = _build_embedding_map(results, embeddings, story_scenes)
    np.save(job_dir / "scene_embeddings.npy", embeddings)
    _check_cancelled(cancel_check)
    highlight_path = job_dir / "highlight.mp4"
    _emit_progress(
        progress,
        "Özet video hazırlanıyor",
        _analysis_snapshot(
            results, duration, started, speaker_count, detected_language, cast, warnings
        ),
        92,
    )
    highlight_created, highlight_intervals = _make_highlight(
        video_path,
        results,
        highlight_path,
        turns,
        duration,
        has_audio=has_audio,
    )
    selection_meta["speech_safe_clip_count"] = len(highlight_intervals)
    selection_meta["speech_safe_duration_sec"] = round(
        sum(end - start for start, end in highlight_intervals), 1
    )
    selection_meta["speech_safe_intervals"] = [
        {"start_sec": start, "end_sec": end}
        for start, end in highlight_intervals
    ]
    if not highlight_created:
        warnings.append("Özet video oluşturulamadı; sahne sonuçları hazır.")

    payload = {
        "duration_sec": round(duration, 2),
        "processing_sec": round(time.perf_counter() - started, 2),
        "scene_count": len(results),
        "story_scene_count": len(story_scenes),
        "selected_count": sum(scene.selected for scene in results),
        "speaker_count": speaker_count,
        "language": detected_language,
        "cast": cast,
        "selection": selection_meta,
        "story_scenes": story_scenes,
        "embedding_map": embedding_map,
        "warnings": warnings,
        "highlight_path": str(highlight_path) if highlight_created else None,
        "scenes": [asdict(scene) for scene in results],
    }
    (job_dir / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def reanalyze_from_cache(
    video_path: Path,
    job_dir: Path,
    cached_analysis: dict,
    cached_job_dir: Path,
    budget_ratio: float = 0.22,
    selection_request: dict | None = None,
    progress: Callable | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Reuse deterministic/expensive analysis and rerun selection providers."""
    started = time.perf_counter()
    _check_cancelled(cancel_check)
    scene_fields = set(SceneResult.__dataclass_fields__)
    results: list[SceneResult] = []
    for raw_scene in cached_analysis.get("scenes", []):
        values = {key: copy.deepcopy(value) for key, value in raw_scene.items() if key in scene_fields}
        values["frame_path"] = str(job_dir / "frames" / Path(values["frame_path"]).name)
        results.append(SceneResult(**values))
    if not results:
        raise ValueError("Önceki analizde yeniden kullanılabilir sahne bulunamadı")

    cast = copy.deepcopy(cached_analysis.get("cast") or [])
    for person in cast:
        if person.get("portrait_path"):
            person["portrait_path"] = str(
                job_dir / "portraits" / Path(person["portrait_path"]).name
            )
    warnings = [
        warning for warning in cached_analysis.get("warnings", [])
        if not str(warning).startswith("Paralon seçimi kullanılamadı")
    ]
    duration = float(cached_analysis.get("duration_sec") or sum(x.duration_sec for x in results))
    speaker_count = int(cached_analysis.get("speaker_count") or 0)
    language = cached_analysis.get("language")
    story_scenes = copy.deepcopy(cached_analysis.get("story_scenes") or [])
    _emit_progress(
        progress,
        "Aynı videonun konuşma ve görsel analizleri yeniden kullanılıyor",
        _analysis_snapshot(results, duration, started, speaker_count, language, cast, warnings),
        84,
    )

    embeddings_path = cached_job_dir / "scene_embeddings.npy"
    cached_embeddings = (
        np.load(embeddings_path, allow_pickle=False) if embeddings_path.exists() else None
    )
    for result in results:
        result.selected = False
        result.selection_rank = None
        result.selection_reasons = []
        result.scoring_breakdown = {}
    for story in story_scenes:
        story["selected"] = False
    _emit_progress(progress, "Paralon RAG bağlamıyla sahneleri yeniden seçiyor", percent=89)
    embeddings, selection_meta, story_scenes = _embed_and_select(
        results,
        min(0.5, max(0.05, budget_ratio)),
        story_scenes_override=story_scenes or None,
        rag_database_path=job_dir / "scene_rag.sqlite3",
        use_llm_selection=True,
        cancel_check=cancel_check,
        embeddings_override=cached_embeddings,
        selection_request=selection_request,
    )
    llm_status = selection_meta.get("llm_selection", {}).get("status")
    if llm_status == "fallback_knapsack":
        reason = selection_meta["llm_selection"].get("reason", "bilinmeyen hata")
        warnings.append(f"Paralon seçimi kullanılamadı; knapsack seçimi korundu: {reason}")

    turns_by_key: dict[tuple, SpeechTurn] = {}
    for scene in results:
        for item in scene.transcript:
            key = (item.get("start_sec"), item.get("end_sec"), item.get("text"))
            if None in key or key in turns_by_key:
                continue
            turns_by_key[key] = SpeechTurn(
                float(key[0]), float(key[1]), str(key[2]),
                str(item.get("speaker") or "Konuşmacı 1"), item.get("actor"),
            )
    turns = sorted(turns_by_key.values(), key=lambda item: item.start_sec)
    embedding_map = _build_embedding_map(results, embeddings, story_scenes)
    np.save(job_dir / "scene_embeddings.npy", embeddings)
    _check_cancelled(cancel_check)
    _emit_progress(progress, "Özet video yeniden hazırlanıyor", percent=92)
    highlight_path = job_dir / "highlight.mp4"
    has_audio = _video_has_audio_stream(video_path)
    highlight_created, highlight_intervals = _make_highlight(
        video_path, results, highlight_path, turns, duration, has_audio=has_audio
    )
    selection_meta["speech_safe_clip_count"] = len(highlight_intervals)
    selection_meta["speech_safe_duration_sec"] = round(
        sum(end - start for start, end in highlight_intervals), 1
    )
    selection_meta["speech_safe_intervals"] = [
        {"start_sec": start, "end_sec": end} for start, end in highlight_intervals
    ]
    if not highlight_created:
        warnings.append("Özet video oluşturulamadı; sahne sonuçları hazır.")
    payload = {
        "duration_sec": round(duration, 2),
        "processing_sec": round(time.perf_counter() - started, 2),
        "scene_count": len(results),
        "story_scene_count": len(story_scenes),
        "selected_count": sum(scene.selected for scene in results),
        "speaker_count": speaker_count,
        "language": language,
        "cast": cast,
        "selection": selection_meta,
        "story_scenes": story_scenes,
        "embedding_map": embedding_map,
        "warnings": warnings,
        "highlight_path": str(highlight_path) if highlight_created else None,
        "scenes": [asdict(scene) for scene in results],
        "cache_reused": True,
    }
    (job_dir / "analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
