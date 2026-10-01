from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalize(left), normalize(right)))


def extract_segment_embeddings(
    audio: np.ndarray,
    turns: list[dict],
    embedding_model,
    sample_rate: int = 16_000,
    min_duration: float = 1.0,
    max_duration: float = 8.0,
) -> dict[int, np.ndarray]:
    """Extract identity embeddings only from sufficiently long turns."""
    embeddings: dict[int, np.ndarray] = {}
    for index, turn in enumerate(turns):
        duration = turn["end"] - turn["start"]
        if duration < min_duration:
            continue
        start = max(0, round(turn["start"] * sample_rate))
        end = min(len(audio), start + round(min(duration, max_duration) * sample_rate))
        waveform = torch.from_numpy(audio[start:end]).float().reshape(1, 1, -1)
        if waveform.shape[-1] < embedding_model.min_num_samples:
            continue
        embedding = embedding_model(waveform)[0]
        if np.all(np.isfinite(embedding)):
            embeddings[index] = normalize(embedding.astype(np.float32))
    return embeddings


def build_prototypes(
    turns: list[dict], embeddings: dict[int, np.ndarray]
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for index, embedding in embeddings.items():
        turn = turns[index]
        grouped[turn["speaker"]].append((turn["end"] - turn["start"], embedding))

    prototypes = {}
    for speaker, samples in grouped.items():
        # Cap duration weights so that one long segment cannot dominate a speaker.
        weights = np.asarray([min(duration, 5.0) for duration, _ in samples])
        matrix = np.vstack([embedding for _, embedding in samples])
        prototypes[speaker] = normalize(np.average(matrix, axis=0, weights=weights))
    return prototypes


def validate_assignments(
    turns: list[dict],
    embeddings: dict[int, np.ndarray],
    prototypes: dict[str, np.ndarray],
    min_similarity: float,
    margin: float,
) -> int:
    corrected = 0
    for index, embedding in embeddings.items():
        turn = turns[index]
        assigned = turn["speaker"]
        similarities = {
            speaker: cosine(embedding, prototype)
            for speaker, prototype in prototypes.items()
        }
        if assigned not in similarities:
            continue
        best = max(similarities, key=similarities.get)
        best_score = similarities[best]
        assigned_score = similarities[assigned]
        if (
            best != assigned
            and best_score >= min_similarity
            and best_score - assigned_score >= margin
        ):
            turn.update(
                {
                    "original_speaker": turn.get("original_speaker", assigned),
                    "speaker": best,
                    "corrected": True,
                    "assigned_similarity": round(assigned_score, 4),
                    "best_similarity": round(best_score, 4),
                    "correction_reason": "speaker_embedding",
                }
            )
            corrected += 1
    return corrected


def _overlap_blocks_merge(left: str, right: str, overlaps: list[dict]) -> bool:
    pair = {left, right}
    return any(pair.issubset(set(region["speakers"])) for region in overlaps)


def merge_duplicate_clusters(
    turns: list[dict],
    prototypes: dict[str, np.ndarray],
    overlaps: list[dict],
    threshold: float,
) -> tuple[dict[str, str], dict[str, float]]:
    speakers = sorted(prototypes)
    similarities = {}
    candidates = []
    for pos, left in enumerate(speakers):
        for right in speakers[pos + 1 :]:
            score = cosine(prototypes[left], prototypes[right])
            similarities[f"{left}<->{right}"] = round(score, 4)
            if score >= threshold and not _overlap_blocks_merge(left, right, overlaps):
                candidates.append((score, left, right))

    remap: dict[str, str] = {}
    for _, left, right in sorted(candidates, reverse=True):
        target = remap.get(left, left)
        source = remap.get(right, right)
        if target != source:
            remap[source] = target

    def resolve(speaker: str) -> str:
        while speaker in remap:
            speaker = remap[speaker]
        return speaker

    final_remap = {speaker: resolve(speaker) for speaker in speakers if resolve(speaker) != speaker}
    for turn in turns:
        original = turn["speaker"]
        final = resolve(original)
        if final != original:
            turn.update(
                {
                    "original_speaker": turn.get("original_speaker", original),
                    "speaker": final,
                    "corrected": True,
                    "correction_reason": "duplicate_cluster_merge",
                }
            )
    return final_remap, similarities


def fix_short_interruptions(
    turns: list[dict], embeddings: dict[int, np.ndarray], max_duration: float = 0.25
) -> int:
    corrected = 0
    for index in range(1, len(turns) - 1):
        previous, current, following = turns[index - 1 : index + 2]
        if (
            previous["speaker"] == following["speaker"]
            and current["speaker"] != previous["speaker"]
            and current["end"] - current["start"] <= max_duration
            and current["start"] - previous["end"] <= 0.08
            and following["start"] - current["end"] <= 0.08
        ):
            left_embedding = embeddings.get(index - 1)
            right_embedding = embeddings.get(index + 1)
            # Duration alone is not evidence. Require both surrounding turns to
            # have reliable, mutually consistent speaker embeddings.
            if (
                left_embedding is None
                or right_embedding is None
                or cosine(left_embedding, right_embedding) < 0.75
            ):
                continue
            original = current["speaker"]
            current.update(
                {
                    "original_speaker": current.get("original_speaker", original),
                    "speaker": previous["speaker"],
                    "corrected": True,
                    "correction_reason": "short_aba_glitch",
                }
            )
            corrected += 1
    return corrected


def merge_adjacent(turns: list[dict], max_gap: float = 0.08) -> list[dict]:
    merged: list[dict] = []
    for turn in sorted(turns, key=lambda item: item["start"]):
        if (
            merged
            and merged[-1]["speaker"] == turn["speaker"]
            and 0 <= turn["start"] - merged[-1]["end"] <= max_gap
        ):
            merged[-1]["end"] = turn["end"]
            if turn.get("corrected"):
                merged[-1]["corrected"] = True
        else:
            merged.append(turn.copy())
    return merged


def detect_underclustered(
    turns: list[dict], embeddings: dict[int, np.ndarray], prototypes: dict[str, np.ndarray]
) -> list[dict]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for index, embedding in embeddings.items():
        grouped[turns[index]["speaker"]].append(embedding)
    warnings = []
    for speaker, samples in grouped.items():
        if len(samples) < 4 or speaker not in prototypes:
            continue
        similarities = [cosine(sample, prototypes[speaker]) for sample in samples]
        low_ratio = sum(score < 0.55 for score in similarities) / len(similarities)
        if low_ratio >= 0.30:
            warnings.append(
                {
                    "speaker": speaker,
                    "segments": len(samples),
                    "mean_similarity": round(float(np.mean(similarities)), 4),
                    "low_similarity_ratio": round(low_ratio, 4),
                }
            )
    return warnings


def _two_means(matrix: np.ndarray, iterations: int = 20) -> np.ndarray:
    # Deterministic farthest-pair initialization.
    similarities = matrix @ matrix.T
    left, right = np.unravel_index(np.argmin(similarities), similarities.shape)
    centroids = np.vstack((matrix[left], matrix[right]))
    labels = np.zeros(len(matrix), dtype=np.int64)
    for _ in range(iterations):
        new_labels = np.argmax(matrix @ centroids.T, axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        if len(set(labels.tolist())) < 2:
            break
        centroids = np.vstack(
            [normalize(matrix[labels == label].mean(axis=0)) for label in (0, 1)]
        )
    return labels


def split_underclustered_speakers(
    turns: list[dict],
    embeddings: dict[int, np.ndarray],
    max_centroid_similarity: float = 0.55,
    min_improvement: float = 0.10,
    min_segments_per_cluster: int = 1,
) -> list[dict]:
    """Conservatively split a pyannote label whose embeddings are bimodal."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in embeddings:
        grouped[turns[index]["speaker"]].append(index)

    splits = []
    next_number = 1 + max(
        (int(turn["speaker"].rsplit("_", 1)[-1]) for turn in turns if turn["speaker"].rsplit("_", 1)[-1].isdigit()),
        default=-1,
    )
    for speaker, indices in grouped.items():
        if len(indices) < 2 * min_segments_per_cluster:
            continue
        matrix = np.vstack([embeddings[index] for index in indices])
        labels = _two_means(matrix)
        sizes = [int(np.sum(labels == label)) for label in (0, 1)]
        if min(sizes) < min_segments_per_cluster:
            continue
        centroids = [normalize(matrix[labels == label].mean(axis=0)) for label in (0, 1)]
        centroid_similarity = cosine(centroids[0], centroids[1])
        single_centroid = normalize(matrix.mean(axis=0))
        before = float(np.mean(matrix @ single_centroid))
        after = float(
            np.mean([cosine(matrix[pos], centroids[int(label)]) for pos, label in enumerate(labels)])
        )
        improvement = after - before
        if centroid_similarity >= max_centroid_similarity or improvement < min_improvement:
            continue

        new_speaker = f"SPEAKER_{next_number:02d}"
        next_number += 1
        # Keep the larger/earlier cluster on the original ID.
        split_label = 1 if sizes[1] < sizes[0] else 0 if sizes[0] < sizes[1] else int(labels[-1])
        split_indices = [indices[pos] for pos, label in enumerate(labels) if label == split_label]
        split_centroid = centroids[split_label]
        original_centroid = centroids[1 - split_label]
        for index in split_indices:
            turn = turns[index]
            turn.update(
                {
                    "original_speaker": turn.get("original_speaker", speaker),
                    "speaker": new_speaker,
                    "corrected": True,
                    "correction_reason": "undercluster_split",
                }
            )
        # Assign short, embedding-less turns only with strong local continuity.
        for index, turn in enumerate(turns):
            if turn["speaker"] != speaker or index in embeddings:
                continue
            previous = next((turns[pos] for pos in range(index - 1, -1, -1) if turns[pos]["speaker"] in {speaker, new_speaker}), None)
            following = next((turns[pos] for pos in range(index + 1, len(turns)) if turns[pos]["speaker"] in {speaker, new_speaker}), None)
            if previous and following and previous["speaker"] == following["speaker"] == new_speaker:
                turn["speaker"] = new_speaker
        splits.append(
            {
                "original_speaker": speaker,
                "new_speaker": new_speaker,
                "moved_segments": len(split_indices),
                "cluster_sizes": sizes,
                "centroid_similarity": round(centroid_similarity, 4),
                "cohesion_improvement": round(improvement, 4),
            }
        )
    return splits


def correct_diarization(
    audio: np.ndarray,
    raw_turns: list[dict],
    overlaps: list[dict],
    embedding_model,
    min_embedding_duration: float = 0.5,
    reassign_similarity: float = 0.72,
    reassign_margin: float = 0.15,
    merge_threshold: float = 0.88,
    enable_intra_speaker_split: bool = True,
) -> tuple[list[dict], dict]:
    turns = [turn.copy() for turn in raw_turns]
    embeddings = extract_segment_embeddings(
        audio, turns, embedding_model, min_duration=min_embedding_duration
    )
    total_reassigned = 0
    for _ in range(2):
        prototypes = build_prototypes(turns, embeddings)
        changed = validate_assignments(
            turns, embeddings, prototypes, reassign_similarity, reassign_margin
        )
        total_reassigned += changed
        if not changed:
            break

    intra_speaker_splits = (
        split_underclustered_speakers(turns, embeddings)
        if enable_intra_speaker_split
        else []
    )
    prototypes = build_prototypes(turns, embeddings)
    remap, prototype_similarities = merge_duplicate_clusters(
        turns, prototypes, overlaps, merge_threshold
    )
    short_corrections = fix_short_interruptions(turns, embeddings)
    prototypes = build_prototypes(turns, embeddings)
    underclustered = detect_underclustered(turns, embeddings, prototypes)
    turns = merge_adjacent(turns)
    report = {
        "raw_speaker_count": len({turn["speaker"] for turn in raw_turns}),
        "final_speaker_count": len({turn["speaker"] for turn in turns}),
        "embedded_segments": len(embeddings),
        "reassigned_segments": total_reassigned,
        "short_temporal_corrections": short_corrections,
        "merged_speaker_clusters": remap,
        "intra_speaker_splits": intra_speaker_splits,
        "potential_underclustered_speakers": underclustered,
        "prototype_similarities": prototype_similarities,
        "thresholds": {
            "min_embedding_duration": min_embedding_duration,
            "reassign_similarity": reassign_similarity,
            "reassign_margin": reassign_margin,
            "speaker_merge": merge_threshold,
        },
    }
    return turns, report
