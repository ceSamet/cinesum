import os
import json
import logging
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.llm.narrative_reranker import rerank_narrative_shots
from src.selection.narrative_selector import select_narrative_shots
from src.selection.adaptive_diversity import select_adaptive
from src.selection.three_act import ACT_QUOTAS, act_for, find_turning_points
from src.summary.speech_boundaries import (
    align_segment_boundaries,
    build_speech_context,
    describe_segment_window,
    find_safe_budget_end,
)

logger = logging.getLogger("CineSum.TemporalSegmentBuilder")
logger.setLevel(logging.INFO)

# Category-aware Segment Configuration Constraints
SEGMENT_CONFIG = {
    "action": {
        "min_duration": 4.0,
        "target_duration": 7.0,
        "max_duration": 12.0,
        "pre_context": 1.5,
        "post_context": 2.0,
        "merge_gap": 2.5,
    },
    "dialogue": {
        "min_duration": 6.0,
        "target_duration": 10.0,
        "max_duration": 18.0,
        "pre_context": 2.0,
        "post_context": 2.5,
        "merge_gap": 3.0,
    },
    "importance": {
        "min_duration": 7.0,
        "target_duration": 11.0,
        "max_duration": 30.0,
        "pre_context": 5.0,
        "post_context": 3.0,
        "merge_gap": 3.0,
    },
    "custom": {
        "min_duration": 4.0,
        "target_duration": 7.0,
        "max_duration": 15.0,
        "pre_context": 1.5,
        "post_context": 2.0,
        "merge_gap": 2.5,
    }
}

# General Constants
TEMPORAL_SMOOTHING_ENABLED = True
TEMPORAL_SMOOTHING_WEIGHTS = [0.20, 0.60, 0.20]
MIN_SECONDS_PER_SUMMARY_SEGMENT = 3.0
SUMMARY_DURATION_TOLERANCE = 0.10  # Internal candidate-sizing search width only.
# The exported summary's final duration is capped exactly at target_duration_sec by
# enforce_duration_ceiling() in speech_boundaries.py, called after all later stages
# (targeted ASR, adjacent re-merge) so nothing downstream can push the real output
# past the user's requested duration.
CREDIT_EXCLUSION_THRESHOLD = 0.65  # scored_scene["credit_probability"] >= this -> excluded from every category
SUMMARY_ALGORITHM_VERSION = "3.9.0"
SPEECH_PADDING_SEC = 0.45
CONFLICT_BRIDGE_MIN_GAP_SEC = 90.0
BALANCED_REGION_QUOTAS = {
    "setup": 0.15,
    "development": 0.23,
    "turning_points": 0.22,
    "final_setup": 0.10,
    "climax_payoff": 0.25,
    "resolution": 0.05,
}
BALANCED_REGION_HARD_MULTIPLIERS = {
    "setup": 1.25,
    "development": 1.20,
    "turning_points": 1.20,
    "final_setup": 1.00,
    "climax_payoff": 1.00,
    "resolution": 1.20,
}
BALANCED_RESERVES_PER_REGION = 8
EVENT_CHAIN_MAX_DURATION_SEC = 30.0
EVENT_CHAIN_FORWARD_SEC = 26.0
EVENT_CHAIN_MARKERS = (
    "sacrifice", "death", "dies", "killed", "victory", "wins", "reveal",
    "hammer", "snaps", "final blow", "fedak", "öl", "oldur", "öldür",
    "kazan", "zafer", "çekic", "cekic", "şaşırt", "sasirt", "ortaya çık",
)

@dataclass
class SummarySegment:
    start: float
    end: float
    duration: float
    category: str
    core_shot_ids: List[int] = field(default_factory=list)
    context_shot_ids: List[int] = field(default_factory=list)
    peak_score: float = 0.0
    segment_score: float = 0.0
    has_speech: bool = False
    transcript_text: str = ""
    transition_in: str = "auto"
    transition_out: str = "auto"
    reason: str = "temporal_coherence_segment"
    boundary_mode: str = "shot"
    original_start: float = 0.0
    original_end: float = 0.0
    aligned_start: float = 0.0
    aligned_end: float = 0.0
    start_delta: float = 0.0
    end_delta: float = 0.0
    boundary_reason: str = "shot_boundary"
    complete_utterance: bool = True
    budget_trimmed: bool = False
    word_count: int = 0
    narrative_chain_role: str = ""
    protected_narrative_chain: bool = False
    narrative_region: str = ""
    balanced_reserve: bool = False
    hard_anchor_roles: List[str] = field(default_factory=list)
    late_resolution_chain: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["start"] = round(d["start"], 3)
        d["end"] = round(d["end"], 3)
        d["duration"] = round(d["duration"], 3)
        d["peak_score"] = round(d["peak_score"], 4)
        d["segment_score"] = round(d["segment_score"], 4)
        for key in (
            "original_start", "original_end", "aligned_start", "aligned_end",
            "start_delta", "end_delta",
        ):
            d[key] = round(d[key], 3)
        return d

def smooth_shot_scores(
    shots: List[Dict[str, Any]],
    score_key: str,
    weights: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Applies temporal score smoothing across neighboring shots:
    smoothed_score[i] = 0.20 * score[i-1] + 0.60 * score[i] + 0.20 * score[i+1]
    Preserves both raw_score and smoothed_score on each shot dictionary.
    """
    if not shots:
        return []

    if weights is None:
        weights = TEMPORAL_SMOOTHING_WEIGHTS

    w_prev, w_curr, w_next = weights[0], weights[1], weights[2]
    n = len(shots)

    for i in range(n):
        raw_val = float(shots[i].get(score_key, 0.0))
        shots[i]["raw_score"] = raw_val

        prev_val = float(shots[i - 1].get(score_key, raw_val)) if i > 0 else raw_val
        next_val = float(shots[i + 1].get(score_key, raw_val)) if i < n - 1 else raw_val

        smoothed_val = w_prev * prev_val + w_curr * raw_val + w_next * next_val
        shots[i]["smoothed_score"] = round(smoothed_val, 4)

    return shots

def snap_to_shot_boundaries(
    start_sec: float,
    end_sec: float,
    shots: List[Dict[str, Any]],
    video_duration: float = None
) -> tuple[float, float, List[int], List[int]]:
    """
    Snaps expanded segment boundaries to the nearest valid PySceneDetect shot boundaries
    so cuts do not occur at arbitrary mid-shot frames.
    Identifies core_shot_ids and context_shot_ids.
    """
    if not shots:
        s = max(0.0, start_sec)
        e = min(video_duration if video_duration else end_sec, end_sec)
        return s, e, [], []

    first_start = shots[0]["start_seconds"]
    last_end = shots[-1]["end_seconds"]
    max_dur = video_duration if video_duration else last_end

    # Clamp bounds
    target_start = max(0.0, start_sec)
    target_end = min(max_dur, end_sec)

    # Find nearest shot start boundary for start
    snapped_start = target_start
    min_start_diff = float("inf")
    for sc in shots:
        sc_s = sc["start_seconds"]
        diff = abs(sc_s - target_start)
        if diff < min_start_diff:
            min_start_diff = diff
            snapped_start = sc_s

    # Find nearest shot end boundary for end
    snapped_end = target_end
    min_end_diff = float("inf")
    for sc in shots:
        sc_e = sc["end_seconds"]
        diff = abs(sc_e - target_end)
        if diff < min_end_diff:
            min_end_diff = diff
            snapped_end = sc_e

    if snapped_end <= snapped_start:
        snapped_end = snapped_start + 1.0

    core_ids = []
    context_ids = []

    for sc in shots:
        sc_id = sc["scene_id"]
        sc_s = sc["start_seconds"]
        sc_e = sc["end_seconds"]

        # Overlaps with snapped range
        if sc_e > snapped_start and sc_s < snapped_end:
            # Check if it was in original target core window
            if sc_e > start_sec and sc_s < end_sec:
                core_ids.append(sc_id)
            else:
                context_ids.append(sc_id)

    return snapped_start, snapped_end, core_ids, context_ids

def align_segment_to_speech_boundaries(
    segment_start: float,
    segment_end: float,
    shots: List[Dict[str, Any]],
    max_segment_dur: float = 18.0,
    category: str = "importance",
) -> tuple[float, float, bool, str]:
    """
    Aligns segment boundaries with Whisper speech/transcript boundaries
    so sentences and words are not truncated mid-speech.
    """
    alignment = align_segment_boundaries(
        segment_start,
        segment_end,
        shots,
        category=category,
        max_segment_duration=max_segment_dur,
    )
    return (
        alignment["start"],
        alignment["end"],
        alignment["has_speech"],
        alignment["transcript_text"],
    )


def add_conflict_onset_bridges(
    selected_shots: List[Dict[str, Any]],
    all_shots: List[Dict[str, Any]],
    *,
    video_duration: float,
) -> tuple[List[Dict[str, Any]], List[int]]:
    """Protect the beginning of a sustained final conflict before later highlights.

    A score-only summary can jump from pre-conflict dialogue to a late battle payoff.
    For large gaps in the final act, find the first sustained cluster of visual action
    rather than simply choosing the highest-scoring shot in the middle of the fight.
    """
    if len(selected_shots) < 2:
        return list(selected_shots), []
    ordered_selected = sorted(
        selected_shots,
        key=lambda row: (float(row["start_seconds"]), int(row["scene_id"])),
    )
    ordered_all = sorted(
        all_shots,
        key=lambda row: (float(row["start_seconds"]), int(row["scene_id"])),
    )
    existing_ids = {int(row["scene_id"]) for row in ordered_selected}
    bridges: List[Dict[str, Any]] = []

    def position(row: Dict[str, Any]) -> float:
        return float(row.get(
            "story_position_ratio",
            float(row.get("start_seconds", 0.0)) / max(video_duration, 0.001),
        ))

    for left, right in zip(ordered_selected, ordered_selected[1:]):
        gap_start = float(left["end_seconds"])
        gap_end = float(right["start_seconds"])
        gap = gap_end - gap_start
        if (
            gap < CONFLICT_BRIDGE_MIN_GAP_SEC
            or position(left) < 0.72
            or not 0.80 <= position(right) <= 0.94
        ):
            continue
        search_end = min(gap_end - 5.0, gap_start + min(gap * 0.45, 180.0))
        eligible = [
            row for row in ordered_all
            if gap_start + 5.0 <= float(row["start_seconds"]) <= search_end
            and int(row["scene_id"]) not in existing_ids
            and float(row.get("action_score", 0.0)) >= 0.50
            and float(row.get("narrative_event_score", 0.0)) >= 0.80
            and float(row.get("speech_ratio", 0.0)) <= 0.35
        ]
        onset = None
        for candidate in eligible:
            window_end = float(candidate["start_seconds"]) + 30.0
            sustained = [
                row for row in eligible
                if float(candidate["start_seconds"]) <= float(row["start_seconds"]) <= window_end
            ]
            if len(sustained) >= 3:
                onset = candidate
                break
        if onset is None:
            continue
        bridge = dict(onset)
        bridge["narrative_chain_role"] = "conflict_setup"
        bridge["protected_narrative_chain"] = True
        bridges.append(bridge)
        existing_ids.add(int(bridge["scene_id"]))
        if len(bridges) >= 2:
            break

    combined = ordered_selected + bridges
    combined.sort(key=lambda row: (float(row["start_seconds"]), int(row["scene_id"])))
    return combined, [int(row["scene_id"]) for row in bridges]


def add_protected_event_chains(
    selected_shots: List[Dict[str, Any]],
    all_shots: List[Dict[str, Any]],
    *,
    video_duration: float,
    target_duration_sec: float,
) -> tuple[List[Dict[str, Any]], List[int]]:
    """Protect a bounded setup/action/payoff window around high-value events."""
    if target_duration_sec < 90.0 or not selected_shots:
        return list(selected_shots), []
    ordered_all = sorted(
        all_shots,
        key=lambda row: (float(row["start_seconds"]), int(row["scene_id"])),
    )
    positions = {int(row["scene_id"]): index for index, row in enumerate(ordered_all)}

    def semantic_bonus(row: Dict[str, Any]) -> float:
        evidence = " ".join((
            str(row.get("transcript_text", "")),
            str(row.get("visual_caption", "")),
            str(row.get("llm_reason", "")),
        )).casefold()
        return 0.20 if any(marker in evidence for marker in EVENT_CHAIN_MARKERS) else 0.0

    def anchor_value(row: Dict[str, Any]) -> float:
        return (
            0.40 * float(row.get("narrative_event_score", 0.0))
            + 0.30 * float(row.get("llm_narrative_score", 0.0))
            + 0.20 * float(row.get("hybrid_narrative_score", row.get("smoothed_score", 0.0)))
            + 0.10 * float(row.get("importance_score", 0.0))
            + semantic_bonus(row)
        )

    qualified = []
    for row in selected_shots:
        if row.get("balanced_reserve"):
            continue
        if (
            float(row.get("narrative_event_score", 0.0)) >= 0.94
            or float(row.get("llm_narrative_score", 0.0)) >= 0.75
            or float(row.get("hybrid_narrative_score", 0.0)) >= 0.94
            or semantic_bonus(row) > 0.0
        ):
            qualified.append(row)

    per_region: Dict[str, List[Dict[str, Any]]] = {}
    for row in qualified:
        per_region.setdefault(act_for(row, video_duration), []).append(row)
    anchors = []
    for region, rows in per_region.items():
        rows.sort(key=lambda row: (-anchor_value(row), int(row["scene_id"])))
        region_limit = 2 if region in {"act_ii", "act_iii"} else 1
        anchors.extend(rows[:region_limit])
    max_chains = max(2, min(8, round(target_duration_sec / 45.0)))
    anchors.sort(key=lambda row: (-anchor_value(row), int(row["scene_id"])))
    anchor_ids = {int(row["scene_id"]) for row in anchors[:max_chains]}

    result = []
    protected_ids = []
    for original in selected_shots:
        scene_id = int(original["scene_id"])
        if scene_id not in anchor_ids or scene_id not in positions:
            result.append(original)
            continue
        row = dict(original)
        index = positions[scene_id]
        anchor_start = float(row["start_seconds"])
        anchor_end = float(row["end_seconds"])
        previous = ordered_all[index - 1] if index > 0 else row
        setup_start = (
            float(previous["start_seconds"])
            if anchor_start - float(previous["end_seconds"]) <= 4.0
            else anchor_start
        )
        forward = [
            candidate for candidate in ordered_all[index + 1:]
            if float(candidate["start_seconds"]) <= anchor_end + EVENT_CHAIN_FORWARD_SEC
        ]
        payoff = max(
            forward,
            key=lambda candidate: (
                0.50 * float(candidate.get("narrative_event_score", 0.0))
                + 0.25 * float(candidate.get("action_score", 0.0))
                + 0.25 * float(candidate.get("importance_score", 0.0)),
                float(candidate["end_seconds"]),
            ),
            default=row,
        )
        chain_end = max(anchor_end, float(payoff["end_seconds"]))
        late_resolution = (
            float(row.get("narrative_event_score", 0.0)) >= 0.90
            and 0.80 <= (anchor_start / max(video_duration, 0.001)) <= 0.94
        )
        if late_resolution:
            # A climax is a recap only when the visible consequence follows it.
            # Keep the latest high-event shot that fully fits a bounded 50s horizon.
            late_payoffs = [
                candidate for candidate in ordered_all[index + 1:]
                if float(candidate["end_seconds"]) <= anchor_end + 50.0
                and float(candidate.get("narrative_event_score", 0.0)) >= 0.90
            ]
            if late_payoffs:
                chain_end = max(chain_end, float(late_payoffs[-1]["end_seconds"]))
        chain_start = max(setup_start, chain_end - (50.0 if late_resolution else EVENT_CHAIN_MAX_DURATION_SEC))
        row["event_chain_start"] = round(chain_start, 3)
        row["event_chain_end"] = round(chain_end, 3)
        row["late_resolution_chain"] = late_resolution
        row["narrative_chain_role"] = "event_anchor"
        row["protected_narrative_chain"] = True
        row["event_anchor_score"] = round(anchor_value(row), 4)
        result.append(row)
        protected_ids.append(scene_id)
    return result, protected_ids


def balanced_narrative_region(row: Dict[str, Any], video_duration: float) -> str:
    midpoint = (
        float(row.get("start_seconds", row.get("start", 0.0)))
        + float(row.get("end_seconds", row.get("end", 0.0)))
    ) / 2.0
    ratio = float(row.get(
        "story_position_ratio",
        midpoint / max(video_duration, 0.001),
    ))
    if ratio <= 0.18:
        return "setup"
    if ratio <= 0.45:
        return "development"
    if ratio <= 0.72:
        return "turning_points"
    if ratio <= 0.82:
        return "final_setup"
    if ratio <= 0.94:
        return "climax_payoff"
    return "resolution"


def add_balanced_region_reserves(
    selected_shots: List[Dict[str, Any]],
    all_shots: List[Dict[str, Any]],
    *,
    video_duration: float,
    per_region: int = BALANCED_RESERVES_PER_REGION,
) -> tuple[List[Dict[str, Any]], List[int]]:
    """Supply post-alignment alternatives when selected dialogue is rejected."""
    result = list(selected_shots)
    existing_ids = {int(row["scene_id"]) for row in result}
    existing_stories = {
        row.get("story_scene_id", f"shot:{row['scene_id']}") for row in result
    }
    reserve_ids = []
    for region in ACT_QUOTAS:
        pool = [
            row for row in all_shots
            if int(row["scene_id"]) not in existing_ids
            and act_for(row, video_duration) == region
        ]
        pool.sort(key=lambda row: (
            -float(row.get("smoothed_score", row.get("importance_score", 0.0))),
            -float(row.get("narrative_event_score", 0.0)),
            int(row["scene_id"]),
        ))
        added = 0
        for row in pool:
            story_id = row.get("story_scene_id", f"shot:{row['scene_id']}")
            if story_id in existing_stories:
                continue
            reserve = dict(row)
            reserve["balanced_reserve"] = True
            result.append(reserve)
            existing_ids.add(int(reserve["scene_id"]))
            existing_stories.add(story_id)
            reserve_ids.append(int(reserve["scene_id"]))
            added += 1
            if added >= per_region:
                break
    return result, reserve_ids


def _segment_knapsack(
    segments: List[SummarySegment], budget_sec: float
) -> List[SummarySegment]:
    resolution = 0.5
    capacity = max(0, int(round(budget_sec / resolution)))
    if not segments or capacity <= 0:
        return []
    protected_indexes = []
    protected_weight = 0
    for index, segment in sorted(
        enumerate(segments),
        key=lambda item: (
            item[1].narrative_chain_role == "event_anchor",
            item[1].segment_score,
        ),
        reverse=True,
    ):
        if not segment.protected_narrative_chain:
            continue
        weight = max(1, int(math.ceil(segment.duration / resolution)))
        if protected_weight + weight <= capacity:
            protected_indexes.append(index)
            protected_weight += weight

    remaining_capacity = capacity - protected_weight
    dp = [(0.0, tuple()) for _ in range(remaining_capacity + 1)]
    for index, segment in enumerate(segments):
        if index in protected_indexes:
            continue
        weight = max(1, int(math.ceil(segment.duration / resolution)))
        if weight > remaining_capacity:
            continue
        value = segment.segment_score * math.sqrt(max(segment.duration, 1.0))
        if segment.protected_narrative_chain:
            value += 0.75
        if not segment.balanced_reserve:
            value += 0.20
        for current in range(remaining_capacity, weight - 1, -1):
            previous_value, previous_indexes = dp[current - weight]
            candidate_value = previous_value + value
            if candidate_value > dp[current][0]:
                dp[current] = (candidate_value, previous_indexes + (index,))
    _, indexes = max(dp, key=lambda item: (item[0], len(item[1])))
    selected_indexes = tuple(protected_indexes) + indexes
    return [segments[index] for index in selected_indexes]


def select_balanced_narrative_segments(
    segments: List[SummarySegment], target_duration_sec: float
) -> tuple[List[SummarySegment], Dict[str, Any]]:
    """Allocate actual aligned durations, capping final battle dominance."""
    mandatory = [segment for segment in segments if segment.hard_anchor_roles]
    mandatory_duration = sum(segment.duration for segment in mandatory)
    if mandatory_duration > target_duration_sec + 1e-6:
        raise ValueError(
            f"Zorunlu dönüm noktaları hedef süreye sığmıyor: en az {mandatory_duration:.1f} sn gerekiyor"
        )
    selected: List[SummarySegment] = list(mandatory)
    region_budgets = {
        region: target_duration_sec * quota
        for region, quota in ACT_QUOTAS.items()
    }
    for region, budget in region_budgets.items():
        reserved = sum(segment.duration for segment in mandatory if segment.narrative_region == region)
        options = [segment for segment in segments if segment.narrative_region == region and not segment.hard_anchor_roles]
        selected.extend(_segment_knapsack(options, max(0.0, budget - reserved)))

    selected_ids = {id(segment) for segment in selected}
    region_durations = {
        region: sum(segment.duration for segment in selected if segment.narrative_region == region)
        for region in ACT_QUOTAS
    }
    total = sum(segment.duration for segment in selected)
    max_total = target_duration_sec
    remaining = sorted(
        (segment for segment in segments if id(segment) not in selected_ids),
        key=lambda segment: (
            segment.segment_score / max(segment.duration, 1.0),
            segment.segment_score,
        ),
        reverse=True,
    )
    for segment in remaining:
        region = segment.narrative_region
        hard_cap = region_budgets[region] * 1.20
        if region_durations[region] + segment.duration > hard_cap + 1e-6:
            continue
        if total + segment.duration > max_total + 1e-6:
            continue
        selected.append(segment)
        selected_ids.add(id(segment))
        region_durations[region] += segment.duration
        total += segment.duration

    selected.sort(key=lambda segment: segment.start)
    return selected, {
        "strategy": "post_alignment_three_acts",
        "region_quotas": ACT_QUOTAS,
        "region_budgets": {key: round(value, 3) for key, value in region_budgets.items()},
        "region_hard_caps": {
            key: round(value * 1.20, 3)
            for key, value in region_budgets.items()
        },
        "region_actual_durations": {
            key: round(value, 3) for key, value in region_durations.items()
        },
        "mandatory_duration": round(mandatory_duration, 3),
    }

def build_temporal_segments(
    shots: List[Dict[str, Any]],
    category: str = "importance",
    target_duration_sec: float = 30.0,
    score_key: Optional[str] = None,
    video_duration: Optional[float] = None,
    narrative_mode: str = "local",
    base_dir: Optional[Path] = None,
    video_alias: Optional[str] = None,
    custom_prompt: str = "",
    narrative_progress_callback: Optional[Any] = None,
    distribution_mode: str = "adaptive",
) -> Dict[str, Any]:
    """
    Master Temporal Segment Builder:
    1. Temporal score smoothing
    2. Peak shot selection
    3. Boundary-aware context expansion & shot snapping
    4. Adjacent/overlapping segment merging (MERGE_GAP = 2.5s)
    5. Speech boundary alignment via Whisper
    6. Segment score aggregation
    7. Budget optimization & cut density constraint
    8. Chronological ordering
    """
    if not shots:
        return {
            "target_duration": target_duration_sec,
            "actual_duration": 0.0,
            "category": category,
            "candidate_shots": 0,
            "selected_core_shots": 0,
            "final_segments": 0,
            "cut_density": 0.0,
            "segments": [],
        }

    # Hard-exclude opening/closing credits from EVERY category before any scoring,
    # selection or LLM reranking sees them. credit_probability is content-based
    # (CLIP credits/title-card similarity + real video-time edge position), so this
    # applies uniformly regardless of category or video length. If every shot were
    # (implausibly) flagged, fall back to the unfiltered list rather than crash.
    excluded_credit_shots = 0
    non_credit_shots = [
        shot for shot in shots
        if float(shot.get("credit_probability", 0.0) or 0.0) < CREDIT_EXCLUSION_THRESHOLD
    ]
    if non_credit_shots:
        excluded_credit_shots = len(shots) - len(non_credit_shots)
        shots = non_credit_shots

    cat_cfg = SEGMENT_CONFIG.get(category.lower(), SEGMENT_CONFIG["importance"])
    min_seg_dur = cat_cfg["min_duration"]
    pref_seg_dur = cat_cfg["target_duration"]
    max_seg_dur = cat_cfg["max_duration"]
    pre_ctx = cat_cfg["pre_context"]
    post_ctx = cat_cfg["post_context"]
    merge_gap = cat_cfg["merge_gap"]

    if score_key is None:
        if category.lower() == "custom":
            score_key = "query_max_similarity" if "query_max_similarity" in shots[0] else "importance_score"
        else:
            score_key = f"{category.lower()}_score"
            if score_key not in shots[0]:
                score_key = "importance_score"

    # Step 1: Temporal Score Smoothing
    smooth_shots = smooth_shot_scores(shots, score_key)
    speech_context = build_speech_context(smooth_shots)
    resolved_video_duration = video_duration or max(
        (float(shot.get("end_seconds", 0.0)) for shot in smooth_shots), default=0.0,
    )
    use_fixed_acts = category.lower() == "importance" and distribution_mode == "three_act"
    initial_turning_points = (
        find_turning_points(smooth_shots, resolved_video_duration)
        if use_fixed_acts else {}
    )
    required_shot_ids = [
        item["scene_id"] for item in initial_turning_points.values()
        if item["status"] == "evidence_found"
    ]

    # Step 2: Build peak candidate pool, then apply narrative quotas + MMR + Knapsack.
    scores = [s.get("smoothed_score", 0.0) for s in smooth_shots]
    max_score = max(scores) if scores else 0.5
    mean_score = sum(scores) / len(scores) if scores else 0.2

    eligible_shots = [
        shot for shot in smooth_shots
        if int(shot["scene_id"]) in required_shot_ids or not (
            shot.get("smoothed_score", 0.0) < (max_score * 0.40)
            and shot.get("smoothed_score", 0.0) < mean_score
        )
    ]
    selection_result = select_narrative_shots(
        eligible_shots,
        score_key="smoothed_score",
        category=category,
        target_duration_sec=target_duration_sec,
        video_duration=resolved_video_duration,
        segment_config=cat_cfg,
        required_shot_ids=required_shot_ids,
        distribution_mode=distribution_mode,
    )
    if narrative_mode == "rag_llm":
        if base_dir is None or not video_alias:
            selection_result["llm_selection"] = {
                "status": "fallback_local",
                "reason": "narrative_context_missing",
            }
        else:
            selection_result = rerank_narrative_shots(
                smooth_shots,
                selection_result,
                base_dir=base_dir,
                video_alias=video_alias,
                category=category,
                target_duration_sec=target_duration_sec,
                score_key=score_key,
                segment_config=cat_cfg,
                video_duration=resolved_video_duration,
                custom_prompt=custom_prompt,
                progress_callback=narrative_progress_callback,
                required_shot_ids=required_shot_ids,
            )
    else:
        selection_result["llm_selection"] = {
            "status": "disabled",
            "reason": "local_mode",
        }
    ranked_shots = selection_result["selected_shots"]
    anchor_roles_by_id = {}
    if use_fixed_acts:
        # Reranking enriches selected rows with captions and reasons. Preserve
        # that evidence when evaluating the complete shot pool.
        enriched_by_id = {int(row["scene_id"]): row for row in ranked_shots}
        evidence_pool = [
            {**row, **{key: enriched_by_id[int(row["scene_id"])][key]
                       for key in ("visual_caption", "llava_caption", "llm_reason")
                       if int(row["scene_id"]) in enriched_by_id
                       and key in enriched_by_id[int(row["scene_id"])]}}
            for row in smooth_shots
        ]
        turning_points = find_turning_points(evidence_pool, resolved_video_duration)
        anchor_roles_by_id = {
            item["scene_id"]: role for role, item in turning_points.items()
            if item["status"] == "evidence_found"
        }
        ranked_by_id = {int(row["scene_id"]): row for row in ranked_shots}
        for row in smooth_shots:
            scene_id = int(row["scene_id"])
            if scene_id in anchor_roles_by_id:
                chosen = dict(ranked_by_id.get(scene_id, row))
                chosen["hard_anchor_role"] = anchor_roles_by_id[scene_id]
                chosen["protected_narrative_chain"] = True
                ranked_by_id[scene_id] = chosen
        ranked_shots = list(ranked_by_id.values())
        selection_result["turning_points"] = turning_points
        ranked_shots, narrative_bridge_ids = add_conflict_onset_bridges(
            ranked_shots,
            smooth_shots,
            video_duration=resolved_video_duration,
        )
        selection_result["narrative_bridge_ids"] = narrative_bridge_ids
        ranked_shots, protected_event_ids = add_protected_event_chains(
            ranked_shots,
            smooth_shots,
            video_duration=resolved_video_duration,
            target_duration_sec=target_duration_sec,
        )
        selection_result["protected_event_ids"] = protected_event_ids
        ranked_shots, balanced_reserve_ids = add_balanced_region_reserves(
            ranked_shots,
            smooth_shots,
            video_duration=resolved_video_duration,
        )
        selection_result["balanced_reserve_ids"] = balanced_reserve_ids

    # Step 3: Build initial candidate expanded segments from peaks
    raw_segments: List[Dict[str, Any]] = []
    ordered_shots = sorted(
        smooth_shots,
        key=lambda shot: (float(shot["start_seconds"]), int(shot["scene_id"])),
    )
    shot_position = {
        int(shot["scene_id"]): index for index, shot in enumerate(ordered_shots)
    }

    for sc in ranked_shots:
        if use_fixed_acts and int(sc["scene_id"]) in anchor_roles_by_id:
            sc = dict(sc)
            sc["hard_anchor_role"] = anchor_roles_by_id[int(sc["scene_id"])]
            sc["protected_narrative_chain"] = True
        sc_score = sc.get("smoothed_score", 0.0)
        raw_s = sc.get("event_chain_start", sc["start_seconds"])
        raw_e = sc.get("event_chain_end", sc["end_seconds"])

        # Expand context
        is_conflict_setup = sc.get("narrative_chain_role") == "conflict_setup"
        is_event_anchor = sc.get("narrative_chain_role") == "event_anchor"
        if is_event_anchor:
            expanded_s = max(0.0, raw_s)
            expanded_e = raw_e
        else:
            expanded_s = max(0.0, raw_s - (8.0 if is_conflict_setup else pre_ctx))
            expanded_e = raw_e + (8.0 if is_conflict_setup else post_ctx)

        # A high-salience visual event often begins in a short setup shot and pays
        # off in the immediately following shot. Keep that continuation together
        # (still bounded later by max segment duration).
        if (
            category.lower() == "importance"
            and not is_event_anchor
            and float(sc.get("narrative_event_score", 0.0)) >= 0.90
        ):
            index = shot_position.get(int(sc["scene_id"]))
            if index is not None:
                for neighbor in ordered_shots[index + 1:index + 3]:
                    if float(neighbor.get("narrative_event_score", 0.0)) < 0.90:
                        break
                    expanded_e = max(expanded_e, float(neighbor["end_seconds"]))

        # Ensure segment satisfies minimum segment duration
        if (expanded_e - expanded_s) < min_seg_dur:
            needed = min_seg_dur - (expanded_e - expanded_s)
            expanded_s = max(0.0, expanded_s - (needed / 2.0))
            expanded_e = expanded_e + (needed / 2.0)

        # Snap to PySceneDetect shot boundaries
        snap_s, snap_e, core_ids, ctx_ids = snap_to_shot_boundaries(expanded_s, expanded_e, smooth_shots, video_duration)

        raw_segments.append({
            "start": snap_s,
            "end": snap_e,
            "duration": snap_e - snap_s,
            "core_shot_ids": core_ids if core_ids else [sc["scene_id"]],
            "context_shot_ids": ctx_ids,
            "peak_score": sc.get("raw_score", sc_score),
            "smoothed_peak_score": sc_score,
            "narrative_chain_role": sc.get("narrative_chain_role", ""),
            "protected_narrative_chain": bool(sc.get("protected_narrative_chain", False)),
            "balanced_reserve": bool(sc.get("balanced_reserve", False)),
            "hard_anchor_roles": [sc["hard_anchor_role"]] if sc.get("hard_anchor_role") else [],
            "late_resolution_chain": bool(sc.get("late_resolution_chain", False)),
        })

    # Step 4: Sort candidate segments chronologically and MERGE adjacent/overlapping segments
    raw_segments.sort(key=lambda seg: seg["start"])

    merged_segments: List[Dict[str, Any]] = []
    for seg in raw_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            prev = merged_segments[-1]
            # Merge if gap between segments <= MERGE_GAP_SECONDS
            if (seg["start"] - prev["end"]) <= merge_gap:
                prev["end"] = max(prev["end"], seg["end"])
                prev["duration"] = prev["end"] - prev["start"]
                prev["core_shot_ids"] = sorted(list(set(prev["core_shot_ids"] + seg["core_shot_ids"])))
                prev["context_shot_ids"] = sorted(list(set(prev["context_shot_ids"] + seg["context_shot_ids"])))
                prev["peak_score"] = max(prev["peak_score"], seg["peak_score"])
                prev["smoothed_peak_score"] = max(prev["smoothed_peak_score"], seg["smoothed_peak_score"])
                if seg.get("protected_narrative_chain"):
                    prev["protected_narrative_chain"] = True
                    prev["narrative_chain_role"] = seg.get("narrative_chain_role", "")
                prev["balanced_reserve"] = (
                    bool(prev.get("balanced_reserve", False))
                    and bool(seg.get("balanced_reserve", False))
                )
                prev["hard_anchor_roles"] = sorted(set(prev.get("hard_anchor_roles", []) + seg.get("hard_anchor_roles", [])))
                prev["late_resolution_chain"] = bool(prev.get("late_resolution_chain") or seg.get("late_resolution_chain"))
            else:
                merged_segments.append(seg)

    # Step 5: Speech Boundary Alignment & Segment Score Aggregation
    shot_by_id = {s["scene_id"]: s for s in smooth_shots}
    candidate_summary_segments: List[SummarySegment] = []
    rejected_incomplete_segments = 0

    for seg in merged_segments:
        s_start = seg["start"]
        s_end = seg["end"]

        # Align with Whisper speech boundaries
        alignment = align_segment_boundaries(
            s_start,
            s_end,
            smooth_shots,
            category=category,
            max_segment_duration=max(max_seg_dur, min(50.0, s_end - s_start)) if seg.get("late_resolution_chain") else max_seg_dur,
            context=speech_context,
        )
        aligned_s = alignment["start"]
        aligned_e = alignment["end"]

        final_dur = aligned_e - aligned_s
        if final_dur < min_seg_dur:
            desired_end = min(
                resolved_video_duration,
                max(s_end, aligned_s + min_seg_dur),
            )
            alignment = align_segment_boundaries(
                aligned_s,
                desired_end,
                smooth_shots,
                category=category,
                max_segment_duration=max_seg_dur,
                context=speech_context,
            )
            aligned_s = alignment["start"]
            aligned_e = alignment["end"]

        if alignment["has_speech"]:
            aligned_s = max(0.0, aligned_s - SPEECH_PADDING_SEC)
            aligned_e = min(resolved_video_duration, aligned_e + SPEECH_PADDING_SEC)
            padded_description = describe_segment_window(
                aligned_s, aligned_e, smooth_shots, speech_context
            )
            alignment.update(padded_description)
            alignment["start"] = aligned_s
            alignment["end"] = aligned_e
            alignment["start_delta"] = aligned_s - float(alignment["original_start"])
            alignment["end_delta"] = aligned_e - float(alignment["original_end"])
            alignment["boundary_reason"] = (
                f"{alignment['boundary_reason']},speech_padding_{SPEECH_PADDING_SEC:.2f}s"
            )

        # Calculate robust segment score (core shots weighted 60% max + 40% mean)
        c_scores = [shot_by_id[cid].get("smoothed_score", 0.0) for cid in seg["core_shot_ids"] if cid in shot_by_id]
        if not c_scores:
            c_scores = [seg["smoothed_peak_score"]]

        max_c = max(c_scores)
        mean_c = sum(c_scores) / len(c_scores)
        agg_segment_score = 0.60 * max_c + 0.40 * mean_c

        summary_seg = SummarySegment(
            start=round(aligned_s, 3),
            end=round(aligned_e, 3),
            duration=round(aligned_e - aligned_s, 3),
            category=category,
            core_shot_ids=seg["core_shot_ids"],
            context_shot_ids=seg["context_shot_ids"],
            peak_score=round(seg["peak_score"], 4),
            segment_score=round(agg_segment_score, 4),
            has_speech=alignment["has_speech"],
            transcript_text=alignment["transcript_text"],
            transition_in="auto",
            transition_out="auto",
            reason=f"temporal_{category}_segment",
            boundary_mode=alignment["boundary_mode"],
            original_start=alignment["original_start"],
            original_end=alignment["original_end"],
            aligned_start=aligned_s,
            aligned_end=aligned_e,
            start_delta=alignment["start_delta"],
            end_delta=alignment["end_delta"],
            boundary_reason=alignment["boundary_reason"],
            complete_utterance=alignment["complete_utterance"],
            word_count=alignment["word_count"],
            narrative_chain_role=seg.get("narrative_chain_role", ""),
            protected_narrative_chain=bool(seg.get("protected_narrative_chain", False)),
            narrative_region=act_for({"start": aligned_s, "end": aligned_e}, resolved_video_duration) if use_fixed_acts else balanced_narrative_region(
                {
                    "start_seconds": aligned_s,
                    "end_seconds": aligned_e,
                    "story_position_ratio": sum(
                        float(shot_by_id[scene_id].get(
                            "story_position_ratio",
                            ((float(shot_by_id[scene_id]["start_seconds"])
                              + float(shot_by_id[scene_id]["end_seconds"])) / 2.0)
                            / max(resolved_video_duration, 0.001),
                        ))
                        for scene_id in seg["core_shot_ids"] if scene_id in shot_by_id
                    ) / max(
                        sum(1 for scene_id in seg["core_shot_ids"] if scene_id in shot_by_id),
                        1,
                    ),
                },
                resolved_video_duration,
            ),
            balanced_reserve=bool(seg.get("balanced_reserve", False)),
            hard_anchor_roles=seg.get("hard_anchor_roles", []),
            late_resolution_chain=bool(seg.get("late_resolution_chain", False)),
        )
        if summary_seg.has_speech and not summary_seg.complete_utterance:
            rejected_incomplete_segments += 1
            if summary_seg.hard_anchor_roles:
                selection_result.setdefault("anchor_violations", []).append({"roles": summary_seg.hard_anchor_roles, "reason": "incomplete_utterance"})
            continue
        candidate_summary_segments.append(summary_seg)

    if selection_result.get("anchor_violations"):
        raise ValueError(
            "Zorunlu dönüm noktası konuşma bütünlüğü korunarak hizalanamadı: "
            + ", ".join(sorted({
                role for violation in selection_result["anchor_violations"]
                for role in violation["roles"]
            }))
        )

    # Rank candidate segments by segment_score for budget optimization
    candidate_summary_segments.sort(
        key=lambda seg: (bool(seg.protected_narrative_chain), seg.segment_score),
        reverse=True,
    )

    # Step 6: Budget Optimizer & Cut Density Constraint
    # Target duration optimization: select best coherent segments up to target_duration_sec
    is_select_all = (target_duration_sec <= 0 or target_duration_sec >= 9999)
    selected_segments: List[SummarySegment] = []
    accumulated_dur = 0.0
    post_alignment_balance: Dict[str, Any] = {
        "strategy": "disabled",
        "reason": "not_importance_or_select_all",
    }

    if is_select_all:
        selected_segments = list(candidate_summary_segments)
        accumulated_dur = sum(s.duration for s in selected_segments)
    elif category.lower() == "importance" and distribution_mode == "adaptive":
        selected_segments, post_alignment_balance = select_adaptive(
            candidate_summary_segments,
            budget_sec=target_duration_sec,
            score=lambda seg: seg.segment_score,
            duration=lambda seg: seg.duration,
            position=lambda seg: ((seg.start + seg.end) / 2) / max(resolved_video_duration, 0.001),
            event_group=lambda seg: (
                next((shot_by_id[cid].get("story_scene_id") for cid in seg.core_shot_ids
                      if cid in shot_by_id and shot_by_id[cid].get("story_scene_id") is not None),
                     f"segment:{seg.start}")
            ),
        )
        if not selected_segments and candidate_summary_segments:
            selected_segments = [max(candidate_summary_segments, key=lambda seg: seg.segment_score)]
            post_alignment_balance["fallback"] = "best_single_segment"
        accumulated_dur = sum(seg.duration for seg in selected_segments)
    elif category.lower() == "importance" and target_duration_sec >= 90.0:
        for seg in candidate_summary_segments:
            if seg.hard_anchor_roles:
                seg.protected_narrative_chain = True
                seg.narrative_region = act_for({"start": seg.start, "end": seg.end}, resolved_video_duration)
        selected_segments, post_alignment_balance = select_balanced_narrative_segments(
            candidate_summary_segments,
            target_duration_sec,
        )
        accumulated_dur = sum(s.duration for s in selected_segments)
    else:
        if category.lower() == "importance":
            selected_segments = [seg for seg in candidate_summary_segments if seg.hard_anchor_roles]
            accumulated_dur = sum(seg.duration for seg in selected_segments)
            if accumulated_dur > target_duration_sec + 1e-6:
                raise ValueError(
                    f"Zorunlu dönüm noktaları hedef süreye sığmıyor: en az {accumulated_dur:.1f} sn gerekiyor"
                )
        max_budget = target_duration_sec * (1.0 + SUMMARY_DURATION_TOLERANCE)
        max_allowed_segments = max(1, round(target_duration_sec / MIN_SECONDS_PER_SUMMARY_SEGMENT))

        for seg in candidate_summary_segments:
            if seg in selected_segments:
                continue
            if len(selected_segments) >= max_allowed_segments and accumulated_dur >= (target_duration_sec * 0.85):
                break

            # Avoid adding tiny 1-second fragments just to hit exact budget
            if (accumulated_dur + seg.duration) > max_budget:
                if accumulated_dur >= (target_duration_sec * 0.80):
                    break
                else:
                    # Fit at a safe speech/shot boundary; never cut at an arbitrary timestamp.
                    remaining = target_duration_sec - accumulated_dur
                    if remaining >= min_seg_dur:
                        safe_end = find_safe_budget_end(
                            seg.start,
                            seg.start + remaining,
                            smooth_shots,
                            category=category,
                            min_duration=min_seg_dur,
                            context=speech_context,
                        )
                        if safe_end is not None:
                            trimmed_end = float(safe_end["end"])
                            # A budget-trimmed tail must not lose the same acoustic
                            # safety pad that untrimmed segments get (step 5 above),
                            # or Whisper's slightly-imprecise word timestamps can
                            # shave off the trailing sound of the last word. The
                            # pad is only clamped by the video length here; the
                            # true hard duration cap is enforced later, once, by
                            # enforce_duration_ceiling() over the final segment list.
                            trial_description = describe_segment_window(
                                seg.start, trimmed_end, smooth_shots, speech_context
                            )
                            final_end = trimmed_end
                            if trial_description["has_speech"]:
                                final_end = min(
                                    resolved_video_duration, trimmed_end + SPEECH_PADDING_SEC
                                )
                            seg.end = final_end
                            seg.aligned_end = seg.end
                            seg.duration = seg.end - seg.start
                            seg.end_delta = seg.end - seg.original_end
                            seg.boundary_mode = str(safe_end["mode"])
                            reason = str(safe_end["reason"])
                            if trial_description["has_speech"]:
                                reason = f"{reason},speech_padding_{SPEECH_PADDING_SEC:.2f}s"
                            seg.boundary_reason = reason
                            seg.complete_utterance = bool(safe_end["complete_utterance"])
                            seg.budget_trimmed = True
                            description = describe_segment_window(
                                seg.start, seg.end, smooth_shots, speech_context
                            )
                            seg.has_speech = description["has_speech"]
                            seg.transcript_text = description["transcript_text"]
                            seg.word_count = description["word_count"]
                            seg.complete_utterance = description["complete_utterance"]
                            selected_segments.append(seg)
                            accumulated_dur += seg.duration
                    break

            selected_segments.append(seg)
            accumulated_dur += seg.duration

            if accumulated_dur >= target_duration_sec:
                break

    # Step 7: Chronological Ordering for Final Source Storyline Export
    selected_segments.sort(key=lambda seg: seg.start)

    # Re-merge any newly overlapping segments after budget trimming
    final_segments: List[SummarySegment] = []
    for seg in selected_segments:
        if not final_segments:
            final_segments.append(seg)
        else:
            prev = final_segments[-1]
            if (seg.start - prev.end) <= merge_gap:
                prev.end = max(prev.end, seg.end)
                prev.duration = round(prev.end - prev.start, 3)
                prev.core_shot_ids = sorted(list(set(prev.core_shot_ids + seg.core_shot_ids)))
                prev.context_shot_ids = sorted(list(set(prev.context_shot_ids + seg.context_shot_ids)))
                prev.peak_score = max(prev.peak_score, seg.peak_score)
                prev.segment_score = max(prev.segment_score, seg.segment_score)
                prev.budget_trimmed = prev.budget_trimmed or seg.budget_trimmed
                prev.protected_narrative_chain = (
                    prev.protected_narrative_chain or seg.protected_narrative_chain
                )
                if seg.narrative_chain_role:
                    prev.narrative_chain_role = seg.narrative_chain_role
                prev.balanced_reserve = prev.balanced_reserve and seg.balanced_reserve
                prev.hard_anchor_roles = sorted(set(prev.hard_anchor_roles + seg.hard_anchor_roles))
                prev.late_resolution_chain = prev.late_resolution_chain or seg.late_resolution_chain
                prev.boundary_reason = f"{prev.boundary_reason},merged_adjacent"
                if seg.transcript_text and seg.transcript_text not in prev.transcript_text:
                    prev.transcript_text = (prev.transcript_text + " " + seg.transcript_text).strip()
            else:
                final_segments.append(seg)

    for seg in final_segments:
        description = describe_segment_window(
            seg.start, seg.end, smooth_shots, speech_context
        )
        seg.aligned_start = seg.start
        seg.aligned_end = seg.end
        seg.duration = round(seg.end - seg.start, 3)
        seg.has_speech = description["has_speech"]
        seg.transcript_text = description["transcript_text"]
        seg.word_count = description["word_count"]
        seg.complete_utterance = description["complete_utterance"]

    final_dur = sum(s.duration for s in final_segments)
    cut_density = (len(final_segments) / final_dur) if final_dur > 0 else 0.0
    all_core_shots = set()
    for s in final_segments:
        all_core_shots.update(s.core_shot_ids)

    debug_data = {
        "summary_algorithm_version": SUMMARY_ALGORITHM_VERSION,
        "rejected_incomplete_segments": rejected_incomplete_segments,
        "excluded_credit_shots": excluded_credit_shots,
        "target_duration": target_duration_sec,
        "actual_duration": round(final_dur, 2),
        "category": category,
        "narrative_mode": narrative_mode,
        "distribution_mode": distribution_mode,
        "candidate_shots": len(shots),
        "selected_core_shots": len(all_core_shots),
        "final_segments": len(final_segments),
        "cut_density": round(cut_density, 4),
        "selection": {
            key: value
            for key, value in selection_result.items()
            if key != "selected_shots"
        },
        "post_alignment_balance": post_alignment_balance,
        "segments": [s.to_dict() for s in final_segments]
    }

    logger.info(
        f"[TemporalBuilder] {len(shots)} candidate shots -> {len(final_segments)} coherent segments "
        f"({final_dur:.1f}s / target {target_duration_sec}s, cut_density={cut_density:.3f} seg/s)"
    )

    return debug_data
