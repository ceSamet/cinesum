"""Shared time-based act policy and conservative turning-point evidence."""

import re
from typing import Any, Dict, Iterable, List


ACT_QUOTAS = {"act_i": 0.20, "act_ii": 0.50, "act_iii": 0.30}
TURNING_WINDOWS = {
    "inciting_incident": (0.15, 0.30),
    "midpoint": (0.45, 0.55),
    "climax": (0.80, 0.95),
    "resolution": (0.85, 0.99),
}
EVENT_TERMS = {
    "inciting_incident": ("must", "need to", "have to", "attack", "threat", "mission", "disappear", "tehdit", "görev", "saldır", "zorunday"),
    "midpoint": ("decide", "truth", "reveal", "betray", "sacrifice", "choose", "karar", "gerçek", "ihanet", "fedak"),
    "climax": ("defeat", "victory", "win", "save", "die", "sacrifice", "final", "end", "kazan", "zafer", "kurtar", "öld", "fedak"),
    "resolution": ("saved", "survived", "peace", "won", "returned", "reunited", "safe", "kurtuldu", "kazandı", "barış", "geri dönd", "birleş"),
}


def time_ratio(row: Dict[str, Any], video_duration: float) -> float:
    middle = (float(row.get("start_seconds", row.get("start", 0.0))) + float(row.get("end_seconds", row.get("end", 0.0)))) / 2
    return max(0.0, min(1.0, middle / max(video_duration, 0.001)))


def act_for(row: Dict[str, Any], video_duration: float) -> str:
    ratio = time_ratio(row, video_duration)
    return "act_i" if ratio < 0.25 else "act_ii" if ratio < 0.75 else "act_iii"


def act_overlap_durations(segments: Iterable[Dict[str, Any]], video_duration: float) -> Dict[str, float]:
    """Measure source seconds per act, including segments that cross act boundaries."""
    totals = {name: 0.0 for name in ACT_QUOTAS}
    limits = (("act_i", 0.0, 0.25), ("act_ii", 0.25, 0.75), ("act_iii", 0.75, 1.0))
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        for name, low, high in limits:
            totals[name] += max(0.0, min(end, high * video_duration) - max(start, low * video_duration))
    return {name: round(value, 3) for name, value in totals.items()}


def find_turning_points(shots: Iterable[Dict[str, Any]], video_duration: float) -> Dict[str, Any]:
    """Only certify a turn with text or visual event evidence; time alone is insufficient."""
    rows = list(shots)
    result: Dict[str, Any] = {}
    for role, (lower, upper) in TURNING_WINDOWS.items():
        candidates: List[Dict[str, Any]] = []
        for row in rows:
            ratio = time_ratio(row, video_duration)
            if not lower <= ratio <= upper:
                continue
            text = " ".join(str(row.get(key, "")) for key in ("transcript_text", "visual_caption", "llava_caption", "llm_reason")).casefold()
            matches = [term for term in EVENT_TERMS[role] if re.search(r"(?<!\w)" + re.escape(term), text)]
            event = float(row.get("narrative_event_score", 0.0))
            # A high CLIP percentile alone only says the image looks dramatic.
            # It cannot certify which story event occurred.
            if not matches:
                continue
            confidence = min(1.0, 0.45 + 0.35 * event + 0.20 * min(len(matches), 2) / 2)
            if confidence < 0.55:
                continue
            visual_text = " ".join(str(row.get(key, "")) for key in ("visual_caption", "llava_caption")).casefold()
            independent_visual = any(re.search(r"(?<!\w)" + re.escape(term), visual_text) for term in matches)
            candidates.append({"scene_id": int(row["scene_id"]), "confidence": round(confidence, 3), "evidence_terms": matches[:3], "event_score": round(event, 3), "position_ratio": round(ratio, 4), "evidence": text[:280], "independent_visual": independent_visual})
        if role == "resolution" and result.get("climax", {}).get("status") == "evidence_found":
            climax_id = result["climax"]["scene_id"]
            climax_row = next((row for row in rows if int(row["scene_id"]) == climax_id), None)
            if climax_row is not None:
                candidates = [item for item in candidates if item["position_ratio"] > time_ratio(climax_row, video_duration) and item["scene_id"] != climax_id]
        if candidates:
            best = max(candidates, key=lambda item: (item["confidence"], item["event_score"]))
            corroborated = len(best["evidence_terms"]) >= 2 or best["independent_visual"]
            result[role] = {"status": "evidence_found" if corroborated else "candidate_unverified", **best}
        else:
            result[role] = {"status": "unverified", "reason": "no_sufficient_content_evidence"}
    return result
