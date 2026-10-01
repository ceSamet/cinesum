"""Low-cost multilingual dialogue cues for local narrative scoring.

These are lexical cues, not a claim that a sentence's emotion is understood.
They remain useful when the optional remote model is unavailable.
"""

import re
from typing import Any, Dict, Iterable, List


POSITIVE = {"love", "hope", "happy", "saved", "safe", "won", "victory", "trust", "sev", "umut", "mutlu", "kurtul", "kazan", "zafer", "güven"}
NEGATIVE = {"hate", "fear", "afraid", "die", "dead", "lost", "betray", "hurt", "kill", "terrible", "nefret", "kork", "öl", "kaybet", "ihanet", "acı", "öldür"}
DECISION = {"decide", "choose", "must", "promise", "confess", "truth", "sacrifice", "betray", "karar", "seç", "zorunda", "itiraf", "gerçek", "fedak", "ihanet"}
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _has_prefix(tokens: List[str], cue: str) -> bool:
    return any(token.startswith(cue) for token in tokens)


def dialogue_cues(text: str) -> Dict[str, float]:
    tokens = TOKEN_RE.findall((text or "").casefold())
    if not tokens:
        return {"polarity": 0.0, "intensity": 0.0, "decision": 0.0, "confidence": 0.0}
    positive = sum(_has_prefix(tokens, cue) for cue in POSITIVE)
    negative = sum(_has_prefix(tokens, cue) for cue in NEGATIVE)
    decision = sum(_has_prefix(tokens, cue) for cue in DECISION)
    signal = positive + negative
    return {
        "polarity": round((positive - negative) / max(signal, 1), 4),
        "intensity": round(min(1.0, signal / 2.0), 4),
        "decision": round(min(1.0, decision / 2.0), 4),
        "confidence": round(min(0.75, 0.25 * signal + 0.15 * decision), 4),
    }


def emotion_peaks(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    ordered = sorted(rows, key=lambda row: float(row["start_seconds"]))
    cues = [dialogue_cues(str(row.get("transcript_text", ""))) for row in ordered]
    result = {}
    for index, row in enumerate(ordered):
        current = cues[index]
        neighbors = cues[max(0, index - 2):index] + cues[index + 1:index + 3]
        baseline = sum(item["polarity"] for item in neighbors) / max(len(neighbors), 1)
        change = abs(current["polarity"] - baseline) / 2.0
        peak = min(1.0, 0.45 * current["intensity"] + 0.35 * change + 0.20 * current["decision"])
        result[int(row["scene_id"])] = {
            **current,
            "emotion_change": round(change, 4),
            "emotion_peak": round(peak * current["confidence"], 4),
        }
    return result
