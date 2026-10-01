from typing import Any, Dict, Iterable, List


DEFAULT_NARRATIVE_CANDIDATES = 32
MAX_NARRATIVE_CANDIDATES = 72
MAX_TRANSCRIPT_CHARS = 900
MAX_LLM_TRANSCRIPT_CHARS = 300
MAX_LLM_CONTEXT_CHARS = 180
MAX_LLM_VISUAL_CAPTION_CHARS = 420


def narrative_candidate_limit(video_duration_seconds: float) -> int:
    """Scale the LLM pool conservatively with video duration.

    The tiers deliberately grow slower than the video itself: local analysis still
    covers every shot, while the remote LLM receives only a representative shortlist.
    """
    duration_minutes = max(0.0, float(video_duration_seconds)) / 60.0
    if duration_minutes <= 10:
        return 16
    if duration_minutes <= 30:
        return 24
    if duration_minutes <= 60:
        return 32
    if duration_minutes <= 90:
        return 40
    if duration_minutes <= 150:
        return 48
    if duration_minutes <= 165:
        return 56
    if duration_minutes <= 210:
        return 72
    return MAX_NARRATIVE_CANDIDATES


def _score(shot: Dict[str, Any], score_key: str) -> float:
    return float(shot.get(score_key, shot.get("importance_score", 0.0)))


def build_narrative_candidate_pool(
    shots: Iterable[Dict[str, Any]],
    local_selected_ids: Iterable[int],
    *,
    score_key: str = "importance_score",
    max_candidates: int = DEFAULT_NARRATIVE_CANDIDATES,
    required_shot_ids: Iterable[int] = (),
) -> List[Dict[str, Any]]:
    """Build a deterministic bounded pool without sending the full video to an LLM."""
    limit = max(1, min(int(max_candidates), MAX_NARRATIVE_CANDIDATES))
    ordered = sorted(shots, key=lambda row: (float(row["start_seconds"]), int(row["scene_id"])))
    by_id = {int(row["scene_id"]): row for row in ordered}
    position = {int(row["scene_id"]): index for index, row in enumerate(ordered)}
    result: List[Dict[str, Any]] = []
    seen = set()

    def add(rows: Iterable[Dict[str, Any]], source: str) -> None:
        for shot in rows:
            scene_id = int(shot["scene_id"])
            if scene_id in seen or len(result) >= limit:
                continue
            row = dict(shot)
            row["candidate_source"] = source
            transcript = " ".join(str(row.get("transcript_text", "")).split())
            row["transcript_text"] = transcript[:MAX_TRANSCRIPT_CHARS]
            result.append(row)
            seen.add(scene_id)

    local_ids = [int(value) for value in local_selected_ids if int(value) in by_id]
    add((by_id[value] for value in required_shot_ids if int(value) in by_id), "required_anchor")
    # Keep the local result as the backbone, but reserve room for coverage signals.
    # Previously, local shots + neighbours filled the entire pool before temporal
    # bins and distinct story scenes were ever considered on feature-length films.
    local_quota = max(2, int(round(limit * 0.35)))
    add((by_id[value] for value in local_ids[:local_quota]), "local_selected")

    if ordered:
        video_start = float(ordered[0]["start_seconds"])
        video_end = max(float(row["end_seconds"]) for row in ordered)
        span = max(video_end - video_start, 0.001)
        bin_count = 8 if limit <= 32 else 12 if limit <= 48 else 16
        temporal_representatives = []
        for bin_index in range(bin_count):
            in_bin = [
                row for row in ordered
                if min(
                    int(
                        bin_count * (
                            ((float(row["start_seconds"]) + float(row["end_seconds"])) / 2.0)
                            - video_start
                        ) / span
                    ),
                    bin_count - 1,
                ) == bin_index
            ]
            if in_bin:
                temporal_representatives.append(max(
                    in_bin,
                    key=lambda row: (_score(row, score_key), -int(row["scene_id"])),
                ))
        add(temporal_representatives, "temporal_bin")

        # Importance-only representatives tend to over-select sustained action
        # and dialogue. Reserve a separate, temporally diverse lane for sharp
        # narrative events such as sacrifices, reveals and decisive actions.
        event_representatives = []
        for bin_index in range(bin_count):
            in_bin = [
                row for row in ordered
                if min(
                    int(
                        bin_count * (
                            ((float(row["start_seconds"]) + float(row["end_seconds"])) / 2.0)
                            - video_start
                        ) / span
                    ),
                    bin_count - 1,
                ) == bin_index
            ]
            if in_bin:
                event_representatives.append(max(
                    in_bin,
                    key=lambda row: (
                        float(row.get("narrative_event_score", 0.0)),
                        _score(row, score_key),
                        -int(row["scene_id"]),
                    ),
                ))
        event_quota_end = min(limit, len(result) + max(2, int(round(limit * 0.15))))
        add(
            event_representatives[:max(0, event_quota_end - len(result))],
            "event_peak",
        )

    neighbors = []
    for scene_id in local_ids:
        index = position[scene_id]
        if index > 0:
            neighbors.append(ordered[index - 1])
        if index + 1 < len(ordered):
            neighbors.append(ordered[index + 1])
    neighbor_quota_end = min(limit, len(result) + max(2, int(round(limit * 0.10))))
    add(neighbors[:max(0, neighbor_quota_end - len(result))], "local_neighbor")

    ranked = sorted(ordered, key=lambda row: (-_score(row, score_key), int(row["scene_id"])))
    best_by_story: Dict[Any, Dict[str, Any]] = {}
    for shot in ranked:
        story_id = shot.get("story_scene_id", f"shot:{shot['scene_id']}")
        best_by_story.setdefault(story_id, shot)
    story_quota_end = min(limit, len(result) + max(2, int(round(limit * 0.20))))
    add(
        list(best_by_story.values())[:max(0, story_quota_end - len(result))],
        "strong_story_scene",
    )

    add(ranked, "top_local_score")
    return result


def serialize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Keep prompt payload compact and intentionally omit embeddings/word arrays."""
    related_context = []
    for context in candidate.get("related_context", [])[:2]:
        related_context.append({
            "document_id": context.get("document_id"),
            "score": round(float(context.get("score", 0.0)), 6),
            "content": str(context.get("content", ""))[:MAX_LLM_CONTEXT_CHARS],
        })
    return {
        "scene_id": int(candidate["scene_id"]),
        "story_scene_id": candidate.get("story_scene_id"),
        "start_seconds": round(float(candidate["start_seconds"]), 3),
        "end_seconds": round(float(candidate["end_seconds"]), 3),
        "local_score": round(
            float(candidate.get("local_score", candidate.get("importance_score", 0.0))), 6
        ),
        "mode": candidate.get("story_scene_mode", "general"),
        "transcript": str(candidate.get("transcript_text", ""))[:MAX_LLM_TRANSCRIPT_CHARS],
        "visual_caption": str(candidate.get("visual_caption", ""))[:MAX_LLM_VISUAL_CAPTION_CHARS],
        "visual_caption_source": candidate.get("visual_caption_source"),
        "candidate_source": candidate.get("candidate_source", "unknown"),
        "related_context": related_context,
    }
