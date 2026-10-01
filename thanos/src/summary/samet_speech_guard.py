"""Final speech-safe export guard based on Samet's whole-turn cut policy.

Whisper segment boundaries, not individual words, are atomic here. This guard
runs *after* Gökdeniz's alignment, targeted ASR and budget decisions.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def load_turns(transcript_path: Path | None) -> list[tuple[float, float]]:
    if not transcript_path or not Path(transcript_path).exists():
        return []
    data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    raw_turns = []
    for row in data.get("segments", []):
        if not str(row.get("text", "")).strip():
            continue
        try:
            start, end = float(row["start"]), float(row["end"])
        except (KeyError, TypeError, ValueError):
            continue
        # Transcript repair can leave a hallucinated, nearly film-length turn.
        if 0 <= start < end and end - start <= 90:
            raw_turns.append((start, end, str(row.get("text", "")).strip()))
    merged: list[list[Any]] = []
    for start, end, text in sorted(set(raw_turns)):
        if merged:
            previous = merged[-1]
            complete = bool(re.search(r"[.!?…][\"'”’)]*$", str(previous[2])))
            if not complete and 0 <= start - float(previous[1]) <= 0.8 and end - float(previous[0]) <= 45:
                previous[1] = max(float(previous[1]), end)
                previous[2] = (str(previous[2]) + " " + text).strip()
                continue
        merged.append([start, end, text])
    return [(float(start), float(end)) for start, end, _ in merged]


def speech_safe_bounds(
    start: float,
    end: float,
    turns: list[tuple[float, float]],
    duration: float,
    padding: float = 0.22,
) -> tuple[float, float]:
    """Samet's iterative whole-turn expansion, with its 0.22 s acoustic pad."""
    start = max(0.0, float(start))
    end = min(float(duration), float(end))
    for _ in range(len(turns) + 1):
        previous = start, end
        for turn_start, turn_end in turns:
            if turn_start < start < turn_end:
                start = max(0.0, turn_start - padding)
            if turn_start < end < turn_end:
                end = min(float(duration), turn_end + padding)
        if abs(start - previous[0]) < 1e-6 and abs(end - previous[1]) < 1e-6:
            break
    return round(start, 3), round(end, 3)


def guard_and_fit_segments(
    segments: list[dict[str, Any]],
    *,
    turns: list[tuple[float, float]],
    video_duration: float,
    target_duration_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand cuts to whole speech turns, merge overlaps, drop weak clips if over budget.

    Crucially, duration is charged *after* expansion. Never shorten a spoken
    turn just to hit an exact time target.
    """
    ordered = []
    for raw in sorted(segments, key=lambda row: float(row["start"])):
        row = dict(raw)
        start, end = speech_safe_bounds(row["start"], row["end"], turns, video_duration)
        if end <= start:
            continue
        row.update(start=start, end=end, duration=round(end - start, 3))
        row["boundary_reason"] = f"{row.get('boundary_reason', '')},samet_whole_turn_guard".strip(",")
        row["complete_utterance"] = True
        if ordered and start <= ordered[-1]["end"] + 0.08:
            prior = ordered[-1]
            prior["end"] = max(prior["end"], end)
            prior["duration"] = round(prior["end"] - prior["start"], 3)
            prior["segment_score"] = max(float(prior.get("segment_score", 0)), float(row.get("segment_score", 0)))
            prior["has_speech"] = bool(prior.get("has_speech") or row.get("has_speech"))
            for field in ("core_shot_ids", "context_shot_ids", "hard_anchor_roles"):
                prior[field] = sorted(set(prior.get(field, [])) | set(row.get(field, [])))
            if row.get("transcript_text") and row["transcript_text"] not in prior.get("transcript_text", ""):
                prior["transcript_text"] = (prior.get("transcript_text", "") + " " + row["transcript_text"]).strip()
        else:
            ordered.append(row)

    for row in ordered:
        row["start"], row["end"] = speech_safe_bounds(row["start"], row["end"], turns, video_duration)
        row["duration"] = round(row["end"] - row["start"], 3)
    before = round(sum(row["duration"] for row in ordered), 3)
    fit_strategy = "within_budget"
    if 0 < target_duration_sec < 9999:
        if len(ordered) == 1 and ordered[0]["duration"] > target_duration_sec and not ordered[0].get("has_speech"):
            # A silent isolated peak may safely use the remaining time budget.
            row = ordered[0]
            row["end"] = round(min(row["end"], row["start"] + target_duration_sec), 3)
            row["duration"] = round(row["end"] - row["start"], 3)
        if sum(row["duration"] for row in ordered) > target_duration_sec + 0.001:
            mandatory = [row for row in ordered if row.get("hard_anchor_roles")]
            optional = [row for row in ordered if not row.get("hard_anchor_roles")]
            mandatory_duration = sum(row["duration"] for row in mandatory)
            if mandatory_duration > target_duration_sec + 0.001:
                raise ValueError("Konuşma bütünlüğü ve zorunlu sahneler hedef süreye birlikte sığmıyor")

            # A greedy weakest-clip deletion can turn a nearly full 30-second
            # summary into 11 seconds. Select whole intervals by actual,
            # speech-expanded duration; among equally full subsets, keep the
            # higher-scoring material. Rounding costs upward keeps the hard cap.
            tick = 0.1
            budget = max(0, int((target_duration_sec - mandatory_duration + 0.000001) / tick))
            states: list[tuple[float, int] | None] = [None] * (budget + 1)
            states[0] = (0.0, 0)
            for index, row in enumerate(optional):
                cost = max(1, math.ceil((float(row["duration"]) - 0.000001) / tick))
                if cost > budget:
                    continue
                value = max(0.0, float(row.get("segment_score", 0.0)))
                for capacity in range(budget, cost - 1, -1):
                    previous = states[capacity - cost]
                    if previous is None:
                        continue
                    score = previous[0] + value
                    current = states[capacity]
                    if current is None or score > current[0] + 0.000001:
                        states[capacity] = (score, previous[1] | (1 << index))
            best = next((state for state in reversed(states) if state is not None), None)
            chosen_mask = best[1] if best is not None else 0
            ordered = sorted(
                mandatory + [row for index, row in enumerate(optional) if chosen_mask & (1 << index)],
                key=lambda row: float(row["start"]),
            )
            fit_strategy = "whole_interval_duration_knapsack"
    if not ordered:
        raise ValueError("Hedef süreye cümleyi bölmeden sığan sahne yok; süreyi artırın")
    return ordered, {
        "strategy": "samet_whole_turn_final_guard",
        "transcript_turns": len(turns),
        "duration_before_fit": before,
        "duration_after_fit": round(sum(row["duration"] for row in ordered), 3),
        "dropped_intervals": len(segments) - len(ordered),
        "fit_strategy": fit_strategy,
    }
