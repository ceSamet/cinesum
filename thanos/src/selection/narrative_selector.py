import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from src.selection.three_act import ACT_QUOTAS, act_for
from src.selection.adaptive_diversity import select_adaptive


NARRATIVE_QUOTAS = {
    "intro": 0.15,
    "early_middle": 0.18,
    "middle": 0.22,
    "ending_early": 0.05,
    "ending": 0.05,
    "turning_climax_early": 0.05,
    "turning_climax": 0.05,
    "late_climax": 0.10,
    "resolution_early": 0.05,
    "resolution": 0.10,
}

DIVERSITY_WEIGHTS = {
    "action": 0.20,
    "dialogue": 0.12,
    "importance": 0.25,
    "custom": 0.18,
}

KNAPSACK_RESOLUTION_SEC = 0.5
MIN_SHOTS_FOR_NARRATIVE_SELECTION = 6
MAX_SELECTED_SHOTS_PER_STORY = 2
MIN_MMR_POOL_SIZE = 80
MAX_MMR_POOL_SIZE = 400
MMR_POOL_OVERSAMPLE_FACTOR = 4
MMR_RANK_OVERSAMPLE_FACTOR = 2


def narrative_region(shot: Dict[str, Any], video_duration: float) -> str:
    midpoint = (float(shot["start_seconds"]) + float(shot["end_seconds"])) / 2.0
    # Scene-order position is more robust than runtime for films with long end
    # credits. Scored shots provide it; synthetic/legacy inputs fall back to time.
    ratio = float(shot.get(
        "story_position_ratio",
        midpoint / max(video_duration, 0.001),
    ))
    if ratio <= 0.15:
        return "intro"
    if ratio <= 0.38:
        return "early_middle"
    if ratio <= 0.65:
        return "middle"
    if ratio <= 0.75:
        return "ending_early"
    if ratio <= 0.80:
        return "ending"
    if ratio <= 0.81:
        return "turning_climax_early"
    if ratio <= 0.82:
        return "turning_climax"
    if ratio <= 0.90:
        return "late_climax"
    if ratio <= 0.925:
        return "resolution_early"
    return "resolution"


def _compact_multimodal_vector(
    shot: Dict[str, Any],
    video_duration: float,
) -> Tuple[float, ...]:
    midpoint = (float(shot["start_seconds"]) + float(shot["end_seconds"])) / 2.0
    duration = float(shot.get("duration_seconds", 0.0))
    return (
        float(shot.get("clip_action_similarity", shot.get("action_score", 0.0))),
        float(shot.get("clip_dialogue_similarity", shot.get("dialogue_score", 0.0))),
        float(shot.get("importance_score", 0.0)),
        float(shot.get("normalized_audio_energy", 0.0)),
        float(shot.get("speech_ratio", 0.0)),
        min(duration / 20.0, 1.0),
        min(max(midpoint / max(video_duration, 0.001), 0.0), 1.0),
    )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return min(max(dot / (left_norm * right_norm), 0.0), 1.0)


def _estimate_segment_duration(
    shot: Dict[str, Any],
    min_duration: float,
    max_duration: float,
    pre_context: float,
    post_context: float,
) -> float:
    if "_selection_duration" in shot:
        return min(
            max(float(shot["_selection_duration"]), min_duration),
            max_duration,
        )
    raw = float(shot.get("duration_seconds", 0.0)) + pre_context + post_context
    return min(max(raw, min_duration), max_duration)


def mmr_rank(
    candidates: Iterable[Dict[str, Any]],
    score_key: str,
    diversity_weight: float,
    video_duration: float,
    seed_shots: Optional[Iterable[Dict[str, Any]]] = None,
    max_results: Optional[int] = None,
) -> List[Dict[str, Any]]:
    remaining = list(candidates)
    selected = list(seed_shots or [])
    ranked = []
    vectors = {
        int(shot["scene_id"]): _compact_multimodal_vector(shot, video_duration)
        for shot in remaining + selected
    }

    # Keep each candidate's maximum similarity to the selected set.  The old
    # implementation recomputed every candidate against every previously selected
    # shot on every iteration, which becomes cubic for feature-length movies.
    max_similarities: Dict[int, float] = {}
    for shot in remaining:
        scene_id = int(shot["scene_id"])
        vector = vectors[scene_id]
        best_similarity = 0.0
        for chosen in selected:
            similarity = cosine_similarity(vector, vectors[int(chosen["scene_id"])])
            shot_story = shot.get("story_scene_id")
            chosen_story = chosen.get("story_scene_id")
            if shot_story is not None and shot_story == chosen_story:
                similarity = 1.0
            best_similarity = max(best_similarity, similarity)
        max_similarities[scene_id] = best_similarity

    result_limit = len(remaining) if max_results is None else max(0, min(max_results, len(remaining)))
    while remaining and len(ranked) < result_limit:
        best_shot = None
        best_mmr = float("-inf")
        best_similarity = 0.0

        for shot in remaining:
            relevance = float(shot.get(score_key, 0.0))
            max_similarity = max_similarities[int(shot["scene_id"])]
            mmr_score = relevance - diversity_weight * max_similarity
            if mmr_score > best_mmr:
                best_shot = shot
                best_mmr = mmr_score
                best_similarity = max_similarity

        ranked_row = dict(best_shot)
        ranked_row["_mmr_score"] = round(best_mmr, 6)
        ranked_row["_max_selected_similarity"] = round(best_similarity, 6)
        ranked.append(ranked_row)
        selected.append(best_shot)
        remaining.remove(best_shot)

        chosen_id = int(best_shot["scene_id"])
        chosen_vector = vectors[chosen_id]
        chosen_story = best_shot.get("story_scene_id")
        for shot in remaining:
            scene_id = int(shot["scene_id"])
            similarity = cosine_similarity(vectors[scene_id], chosen_vector)
            shot_story = shot.get("story_scene_id")
            if shot_story is not None and shot_story == chosen_story:
                similarity = 1.0
            if similarity > max_similarities[scene_id]:
                max_similarities[scene_id] = similarity

    return ranked


def _bounded_mmr_pool(
    shots: Iterable[Dict[str, Any]],
    score_key: str,
    budget_sec: float,
    min_duration_sec: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """Pre-filter large pools and return a budget-aware MMR ranking limit."""
    estimated_slots = max(1, int(math.ceil(budget_sec / max(min_duration_sec, 0.5))))
    rank_limit = max(16, estimated_slots * MMR_RANK_OVERSAMPLE_FACTOR)
    pool_limit = min(
        MAX_MMR_POOL_SIZE,
        max(MIN_MMR_POOL_SIZE, estimated_slots * MMR_POOL_OVERSAMPLE_FACTOR),
    )
    ordered = sorted(shots, key=lambda shot: float(shot.get(score_key, 0.0)), reverse=True)
    bounded = ordered[:pool_limit]
    return bounded, min(rank_limit, len(bounded))


def knapsack_select(
    candidates: Iterable[Dict[str, Any]],
    budget_sec: float,
    *,
    resolution_sec: float = KNAPSACK_RESOLUTION_SEC,
) -> List[Dict[str, Any]]:
    items = list(candidates)
    capacity = max(0, int(round(budget_sec / resolution_sec)))
    if not items or capacity <= 0:
        return []

    # dp[capacity] = (value, tuple of selected item indexes)
    dp: List[Tuple[float, Tuple[int, ...]]] = [(0.0, tuple()) for _ in range(capacity + 1)]
    for index, shot in enumerate(items):
        weight = max(1, int(round(float(shot["_estimated_duration"]) / resolution_sec)))
        value = max(float(shot.get("_mmr_score", 0.0)), 0.000001)
        if weight > capacity:
            continue
        for current_capacity in range(capacity, weight - 1, -1):
            previous_value, previous_indexes = dp[current_capacity - weight]
            candidate_value = previous_value + value
            if candidate_value > dp[current_capacity][0]:
                dp[current_capacity] = (candidate_value, previous_indexes + (index,))

    _, selected_indexes = max(
        dp,
        key=lambda entry: (entry[0], len(entry[1])),
    )
    return [items[index] for index in selected_indexes]


def select_narrative_shots(
    shots: List[Dict[str, Any]],
    *,
    score_key: str,
    category: str,
    target_duration_sec: float,
    video_duration: float,
    segment_config: Dict[str, float],
    required_shot_ids: Optional[Iterable[int]] = None,
    distribution_mode: str = "adaptive",
) -> Dict[str, Any]:
    """Select shots with narrative quotas, MMR diversity and duration knapsack."""
    if category.lower() == "importance" and distribution_mode == "adaptive" and 0 < target_duration_sec < 9999:
        prepared = []
        for shot in shots:
            row = dict(shot)
            row["_estimated_duration"] = _estimate_segment_duration(
                row,
                segment_config["min_duration"],
                segment_config["max_duration"],
                segment_config["pre_context"],
                segment_config["post_context"],
            )
            prepared.append(row)
        required_ids = {int(value) for value in (required_shot_ids or ())}
        # Keep the candidate pool bounded on feature-length videos and avoid
        # feeding many near-identical shots from one StoryScene to knapsack.
        by_story = {}
        for row in sorted(prepared, key=lambda item: float(item.get(score_key, 0.0)), reverse=True):
            key = row.get("story_scene_id", f"shot:{row['scene_id']}")
            members = by_story.setdefault(key, [])
            if len(members) < MAX_SELECTED_SHOTS_PER_STORY or int(row["scene_id"]) in required_ids:
                members.append(row)
        pool = sorted(
            (row for members in by_story.values() for row in members),
            key=lambda item: float(item.get(score_key, 0.0)), reverse=True,
        )[:MAX_MMR_POOL_SIZE]
        selected, report = select_adaptive(
            pool,
            budget_sec=target_duration_sec,
            score=lambda row: float(row.get(score_key, 0.0)),
            duration=lambda row: float(row["_estimated_duration"]),
            position=lambda row: float(row.get(
                "story_position_ratio",
                ((float(row["start_seconds"]) + float(row["end_seconds"])) / 2)
                / max(video_duration, 0.001),
            )),
            event_group=lambda row: row.get("story_scene_id") or f"shot:{row['scene_id']}",
            required=lambda row: int(row["scene_id"]) in required_ids,
        )
        if not selected and pool:
            selected = [pool[0]]
            report["fallback"] = "best_single_shot_exceeds_estimated_budget"
        return {
            **report,
            "selected_shots": selected,
            "selected_shot_ids": [int(row["scene_id"]) for row in selected],
            "required_shot_ids": sorted(required_ids),
            "decisions": [],
            "region_budgets": {},
            "region_selected_counts": {},
            "candidate_pool_counts": {"adaptive": len(pool)},
            "ranked_candidate_counts": {"adaptive": len(pool)},
        }
    if (
        target_duration_sec <= 0
        or target_duration_sec >= 9999
        or len(shots) < MIN_SHOTS_FOR_NARRATIVE_SELECTION
    ):
        ordered = sorted(shots, key=lambda shot: shot.get(score_key, 0.0), reverse=True)
        return {
            "strategy": "legacy_small_pool" if len(shots) < MIN_SHOTS_FOR_NARRATIVE_SELECTION else "select_all",
            "selected_shots": ordered,
            "selected_shot_ids": [int(shot["scene_id"]) for shot in ordered],
            "diversity_weight": 0.0,
            "region_budgets": {},
            "region_selected_counts": {},
            "decisions": [],
        }

    diversity_weight = DIVERSITY_WEIGHTS.get(category.lower(), DIVERSITY_WEIGHTS["importance"])
    min_duration = float(segment_config["min_duration"])
    use_three_acts = category.lower() == "importance"
    use_compact_regions = not use_three_acts and target_duration_sec < min_duration * len(NARRATIVE_QUOTAS)
    active_quotas = (
        ACT_QUOTAS if use_three_acts else (
            {"intro": 0.20, "middle": 0.55, "ending": 0.25}
            if use_compact_regions else NARRATIVE_QUOTAS
        )
    )
    prepared = []
    for shot in shots:
        row = dict(shot)
        if use_three_acts:
            row["_narrative_region"] = act_for(row, video_duration)
        elif use_compact_regions:
            midpoint = (float(row["start_seconds"]) + float(row["end_seconds"])) / 2.0
            ratio = midpoint / max(video_duration, 0.001)
            row["_narrative_region"] = (
                "intro" if ratio <= 0.20 else "ending" if ratio >= 0.75 else "middle"
            )
        else:
            row["_narrative_region"] = narrative_region(row, video_duration)
        row["_estimated_duration"] = _estimate_segment_duration(
            row,
            segment_config["min_duration"],
            segment_config["max_duration"],
            segment_config["pre_context"],
            segment_config["post_context"],
        )
        prepared.append(row)

    region_budgets = {
        region: round(target_duration_sec * quota, 3)
        for region, quota in active_quotas.items()
    }
    required_ids = {int(value) for value in (required_shot_ids or ())}
    selected: List[Dict[str, Any]] = [row for row in prepared if int(row["scene_id"]) in required_ids]
    candidate_pool_counts: Dict[str, int] = {}
    ranked_candidate_counts: Dict[str, int] = {}

    def diversify_story_candidates(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if category.lower() != "importance":
            return list(rows)
        diversified = []
        seen_stories = set()
        for row in rows:
            story_id = row.get("story_scene_id", f"shot:{row['scene_id']}")
            if story_id in seen_stories:
                continue
            diversified.append(row)
            seen_stories.add(story_id)
        return diversified

    def append_with_story_cap(rows: Iterable[Dict[str, Any]]) -> None:
        existing_ids = {int(row["scene_id"]) for row in selected}
        story_counts: Dict[Any, int] = {}
        for row in selected:
            story_id = row.get("story_scene_id", f"shot:{row['scene_id']}")
            story_counts[story_id] = story_counts.get(story_id, 0) + 1

        for row in sorted(rows, key=lambda item: item.get("_mmr_score", 0.0), reverse=True):
            scene_id = int(row["scene_id"])
            story_id = row.get("story_scene_id", f"shot:{scene_id}")
            if scene_id in existing_ids:
                continue
            if story_counts.get(story_id, 0) >= MAX_SELECTED_SHOTS_PER_STORY:
                continue
            selected.append(row)
            existing_ids.add(scene_id)
            story_counts[story_id] = story_counts.get(story_id, 0) + 1

    for region in active_quotas:
        reserved = sum(float(shot["_estimated_duration"]) for shot in selected if shot["_narrative_region"] == region)
        available = max(0.0, region_budgets[region] - reserved)
        full_pool = [shot for shot in prepared if shot["_narrative_region"] == region and int(shot["scene_id"]) not in required_ids]
        pool, rank_limit = _bounded_mmr_pool(
            full_pool,
            score_key,
            available,
            segment_config["min_duration"],
        )
        candidate_pool_counts[region] = len(pool)
        ranked = mmr_rank(
            pool,
            score_key,
            diversity_weight,
            video_duration,
            seed_shots=selected,
            max_results=rank_limit,
        )
        ranked = diversify_story_candidates(ranked)
        ranked_candidate_counts[region] = len(ranked)
        chosen = knapsack_select(ranked, available)
        append_with_story_cap(chosen)

    selected_ids = {int(shot["scene_id"]) for shot in selected}
    used_budget = sum(float(shot["_estimated_duration"]) for shot in selected)
    remaining_budget = max(0.0, target_duration_sec - used_budget)
    remaining_pool = [shot for shot in prepared if int(shot["scene_id"]) not in selected_ids]

    if remaining_budget >= KNAPSACK_RESOLUTION_SEC and remaining_pool:
        remaining_pool, rank_limit = _bounded_mmr_pool(
            remaining_pool,
            score_key,
            remaining_budget,
            segment_config["min_duration"],
        )
        candidate_pool_counts["remainder"] = len(remaining_pool)
        ranked_remaining = mmr_rank(
            remaining_pool,
            score_key,
            diversity_weight,
            video_duration,
            seed_shots=selected,
            max_results=rank_limit,
        )
        ranked_remaining = diversify_story_candidates(ranked_remaining)
        ranked_candidate_counts["remainder"] = len(ranked_remaining)
        extra = knapsack_select(ranked_remaining, remaining_budget)
        append_with_story_cap(extra)
        selected_ids.update(int(shot["scene_id"]) for shot in extra)

    if not selected and prepared:
        fallback = max(prepared, key=lambda shot: shot.get(score_key, 0.0))
        fallback["_mmr_score"] = float(fallback.get(score_key, 0.0))
        fallback["_max_selected_similarity"] = 0.0
        selected = [fallback]
        selected_ids = {int(fallback["scene_id"])}

    selected_ids = {int(shot["scene_id"]) for shot in selected}
    selected.sort(key=lambda shot: shot.get(score_key, 0.0), reverse=True)
    region_selected_counts = {
        region: sum(1 for shot in selected if shot["_narrative_region"] == region)
        for region in active_quotas
    }
    decisions = [{
        "scene_id": int(shot["scene_id"]),
        "region": shot["_narrative_region"],
        "relevance_score": round(float(shot.get(score_key, 0.0)), 6),
        "mmr_score": round(float(shot.get("_mmr_score", shot.get(score_key, 0.0))), 6),
        "max_selected_similarity": round(float(shot.get("_max_selected_similarity", 0.0)), 6),
        "estimated_duration": round(float(shot["_estimated_duration"]), 3),
        "reasons": [
            f"narrative_region:{shot['_narrative_region']}",
            "duration_budget_fit",
            "mmr_diversity_adjusted",
        ],
    } for shot in selected]

    return {
        "strategy": "narrative_mmr_knapsack",
        "selected_shots": selected,
        "selected_shot_ids": sorted(selected_ids),
        "required_shot_ids": sorted(required_ids & selected_ids),
        "diversity_weight": diversity_weight,
        "region_budgets": region_budgets,
        "region_selected_counts": region_selected_counts,
        "candidate_pool_counts": candidate_pool_counts,
        "ranked_candidate_counts": ranked_candidate_counts,
        "estimated_selected_duration": round(
            sum(float(shot["_estimated_duration"]) for shot in selected), 3
        ),
        "decisions": decisions,
    }
