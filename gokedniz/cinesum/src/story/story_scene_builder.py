import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


STORY_SCHEMA_VERSION = "1.0"
MAX_STORY_DURATION_SEC = 45.0
MAX_SHOTS_PER_STORY = 12
MAX_CONTINUITY_GAP_SEC = 2.0


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _aggregate_shot_embeddings(
    scenes: List[Dict[str, Any]],
    clip_vectors: np.ndarray,
    clip_metadata: List[Dict[str, Any]],
) -> np.ndarray:
    indexes_by_scene: Dict[int, List[int]] = {}
    for fallback_index, metadata in enumerate(clip_metadata):
        scene_id = int(metadata["scene_id"])
        feature_index = int(metadata.get("feature_vector_index", fallback_index))
        if 0 <= feature_index < len(clip_vectors):
            indexes_by_scene.setdefault(scene_id, []).append(feature_index)

    feature_size = int(clip_vectors.shape[1]) if clip_vectors.ndim == 2 and len(clip_vectors) else 512
    aggregated = []
    for scene in scenes:
        indexes = indexes_by_scene.get(int(scene["scene_id"]), [])
        if indexes:
            aggregated.append(np.mean(clip_vectors[indexes], axis=0))
        else:
            aggregated.append(np.zeros(feature_size, dtype=np.float32))
    return _normalize_rows(np.asarray(aggregated, dtype=np.float32))


def _build_text_embeddings(texts: List[str]) -> np.ndarray:
    if not any(text.strip() for text in texts):
        return np.zeros((len(texts), 1), dtype=np.float32)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts).astype(np.float32)
        return matrix.toarray()
    except (ImportError, ValueError):
        # Deterministic lightweight fallback for minimal installations.
        token_sets = [set(text.lower().split()) for text in texts]
        vocabulary = sorted(set().union(*token_sets))
        if not vocabulary:
            return np.zeros((len(texts), 1), dtype=np.float32)
        positions = {token: index for index, token in enumerate(vocabulary)}
        matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
        for row_index, tokens in enumerate(token_sets):
            for token in tokens:
                matrix[row_index, positions[token]] = 1.0
        return _normalize_rows(matrix)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return min(max(float(np.dot(left, right) / denominator), 0.0), 1.0)


def _deduplicate_transcripts(texts: List[str]) -> str:
    unique = []
    normalized_seen = set()
    for text in texts:
        stripped = " ".join((text or "").split()).strip()
        normalized = stripped.casefold()
        if stripped and normalized not in normalized_seen:
            normalized_seen.add(normalized)
            unique.append(stripped)
    return " ".join(unique)


def _adaptive_threshold(scores: List[float]) -> float:
    if not scores:
        return 0.52
    median = float(np.median(scores))
    standard_deviation = float(np.std(scores))
    return round(min(max(median + 0.15 * standard_deviation, 0.46), 0.68), 4)


def _dominant_mode(shots: List[Dict[str, Any]]) -> str:
    total_duration = sum(float(shot.get("duration_seconds", 0.0)) for shot in shots) or 1.0
    speech = sum(
        float(shot.get("speech_ratio", 0.0)) * float(shot.get("duration_seconds", 0.0))
        for shot in shots
    ) / total_duration
    energy = sum(
        float(shot.get("normalized_audio_energy", 0.0)) * float(shot.get("duration_seconds", 0.0))
        for shot in shots
    ) / total_duration
    if speech >= 0.55:
        return "dialogue"
    if energy >= 0.60:
        return "action_audio"
    return "general"


def build_story_scenes(
    scenes: List[Dict[str, Any]],
    clip_vectors: np.ndarray,
    clip_metadata: List[Dict[str, Any]],
    audio_features: List[Dict[str, Any]],
    *,
    max_story_duration_sec: float = MAX_STORY_DURATION_SEC,
    max_shots_per_story: int = MAX_SHOTS_PER_STORY,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Group adjacent shots into explainable StoryScene units."""
    if not scenes:
        return {
            "schema_version": STORY_SCHEMA_VERSION,
            "story_scene_count": 0,
            "continuity_threshold": 0.52,
            "shot_to_story_scene": {},
            "story_scenes": [],
        }, np.empty((0, 0), dtype=np.float32)

    ordered_scenes = sorted(scenes, key=lambda shot: float(shot["start_seconds"]))
    audio_by_scene = {int(row["scene_id"]): row for row in audio_features}
    enriched_shots = []
    for scene in ordered_scenes:
        audio = audio_by_scene.get(int(scene["scene_id"]), {})
        enriched_shots.append({
            **scene,
            "transcript_text": audio.get("transcript_text", ""),
            "speech_ratio": float(audio.get("speech_ratio", 0.0)),
            "normalized_audio_energy": float(audio.get("normalized_audio_energy", 0.0)),
        })

    visual_embeddings = _aggregate_shot_embeddings(ordered_scenes, clip_vectors, clip_metadata)
    text_embeddings = _build_text_embeddings(
        [shot.get("transcript_text", "") for shot in enriched_shots]
    )
    text_embeddings = _normalize_rows(np.asarray(text_embeddings, dtype=np.float32))

    boundaries = []
    for index in range(len(enriched_shots) - 1):
        left = enriched_shots[index]
        right = enriched_shots[index + 1]
        gap = max(0.0, float(right["start_seconds"]) - float(left["end_seconds"]))
        visual_similarity = _cosine(visual_embeddings[index], visual_embeddings[index + 1])
        text_similarity = _cosine(text_embeddings[index], text_embeddings[index + 1])
        speech_continuity = 1.0 - min(
            abs(float(left["speech_ratio"]) - float(right["speech_ratio"])),
            1.0,
        )
        temporal_continuity = max(0.0, 1.0 - gap / MAX_CONTINUITY_GAP_SEC)
        score = (
            0.52 * visual_similarity
            + 0.28 * text_similarity
            + 0.12 * speech_continuity
            + 0.08 * temporal_continuity
        )
        boundaries.append({
            "left_scene_id": int(left["scene_id"]),
            "right_scene_id": int(right["scene_id"]),
            "gap_seconds": round(gap, 4),
            "visual_similarity": round(visual_similarity, 4),
            "text_similarity": round(text_similarity, 4),
            "speech_continuity": round(speech_continuity, 4),
            "temporal_continuity": round(temporal_continuity, 4),
            "continuity_score": round(score, 4),
        })

    threshold = _adaptive_threshold([boundary["continuity_score"] for boundary in boundaries])
    groups: List[List[int]] = [[0]]
    for boundary_index, boundary in enumerate(boundaries):
        current_group = groups[-1]
        next_index = boundary_index + 1
        proposed_start = float(enriched_shots[current_group[0]]["start_seconds"])
        proposed_end = float(enriched_shots[next_index]["end_seconds"])
        proposed_duration = proposed_end - proposed_start
        transcript_bridge = boundary["text_similarity"] >= 0.80
        should_merge = (
            boundary["gap_seconds"] <= MAX_CONTINUITY_GAP_SEC
            and (
                boundary["continuity_score"] >= threshold
                or (transcript_bridge and boundary["visual_similarity"] >= 0.30)
            )
            and proposed_duration <= max_story_duration_sec
            and len(current_group) < max_shots_per_story
        )
        if should_merge:
            current_group.append(next_index)
        else:
            groups.append([next_index])

    story_scenes = []
    story_embeddings = []
    shot_to_story_scene = {}
    for story_index, indexes in enumerate(groups, start=1):
        shots = [enriched_shots[index] for index in indexes]
        shot_ids = [int(shot["scene_id"]) for shot in shots]
        story_vector = np.mean(visual_embeddings[indexes], axis=0)
        story_vector = _normalize_rows(story_vector.reshape(1, -1))[0]
        story_embeddings.append(story_vector)
        for position, shot_id in enumerate(shot_ids):
            shot_to_story_scene[str(shot_id)] = {
                "story_scene_id": story_index,
                "position": position,
                "shot_count": len(shot_ids),
            }

        start = float(shots[0]["start_seconds"])
        end = float(shots[-1]["end_seconds"])
        duration = end - start
        weighted_speech = sum(
            float(shot["speech_ratio"]) * float(shot["duration_seconds"])
            for shot in shots
        ) / max(sum(float(shot["duration_seconds"]) for shot in shots), 0.001)
        internal_boundaries = boundaries[indexes[0]:indexes[-1]] if len(indexes) > 1 else []

        story_scenes.append({
            "story_scene_id": story_index,
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "duration_seconds": round(duration, 3),
            "shot_ids": shot_ids,
            "shot_count": len(shot_ids),
            "transcript_text": _deduplicate_transcripts(
                [shot.get("transcript_text", "") for shot in shots]
            ),
            "speech_ratio": round(weighted_speech, 4),
            "dominant_mode": _dominant_mode(shots),
            "visual_centroid_index": story_index - 1,
            "mean_internal_continuity": round(
                sum(boundary["continuity_score"] for boundary in internal_boundaries)
                / max(len(internal_boundaries), 1),
                4,
            ),
        })

    story_matrix = np.asarray(story_embeddings, dtype=np.float32)
    result = {
        "schema_version": STORY_SCHEMA_VERSION,
        "story_scene_count": len(story_scenes),
        "continuity_threshold": threshold,
        "weights": {
            "visual": 0.52,
            "text": 0.28,
            "speech": 0.12,
            "temporal": 0.08,
        },
        "shot_to_story_scene": shot_to_story_scene,
        "boundaries": boundaries,
        "story_scenes": story_scenes,
    }
    return result, story_matrix


def build_story_scenes_from_files(
    scenes_json_path: Path,
    clip_npy_path: Path,
    clip_metadata_path: Path,
    audio_features_path: Path,
    output_json_path: Path,
    output_npy_path: Path,
) -> Dict[str, Any]:
    with Path(scenes_json_path).open("r", encoding="utf-8") as handle:
        scenes = json.load(handle)
    clip_vectors = np.load(str(clip_npy_path))
    with Path(clip_metadata_path).open("r", encoding="utf-8") as handle:
        clip_metadata = json.load(handle)
    with Path(audio_features_path).open("r", encoding="utf-8") as handle:
        audio_features = json.load(handle)

    result, story_embeddings = build_story_scenes(
        scenes,
        clip_vectors,
        clip_metadata,
        audio_features,
    )
    output_json_path = Path(output_json_path)
    output_npy_path = Path(output_npy_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    temporary_json.replace(output_json_path)
    np.save(str(output_npy_path), story_embeddings)
    return result
