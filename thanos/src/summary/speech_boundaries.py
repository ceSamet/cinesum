import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


SILENCE_BOUNDARY_SEC = 2.75
BOUNDARY_EPSILON_SEC = 0.02
QUESTION_RESPONSE_GAP_SEC = 5.0
CONTINUATION_GAP_SEC = 12.0
LEADING_CONTEXT_GAP_SEC = 6.0
DURATION_CEILING_MIN_KEEP_SEC = 1.5
DURATION_CEILING_UNBOUNDED_SEC = 9999.0

SPEECH_BOUNDARY_POLICIES = {
    "action": {"max_expansion": 0.6, "prefer_sentence": False},
    "dialogue": {"max_expansion": 3.0, "prefer_sentence": True},
    "importance": {"max_expansion": 8.0, "prefer_sentence": True},
    "custom": {"max_expansion": 1.5, "prefer_sentence": True},
}

_SENTENCE_END_RE = re.compile(r"[.!?…]+(?:[\"'”’\)\]]+)?$")
_QUESTION_END_RE = re.compile(r"\?+(?:[\"'”’\)\]]+)?$")
_PUNCTUATION_SPACING_RE = re.compile(r"\s+([,.;:!?…])")
_LEXICAL_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_DANGLING_ENDINGS = {
    "ama", "ancak", "fakat", "çünkü", "ve", "veya", "yahut", "ki",
    "yani", "oysa", "halbuki", "eğer", "şayet", "iken", "ile",
}
_CONTINUATION_STARTERS = _DANGLING_ENDINGS | {
    "de", "da", "ise", "sonra", "ardından", "böylece", "bu yüzden",
}


@dataclass(frozen=True)
class SpeechContext:
    words: List[Dict[str, Any]]
    utterances: List[Dict[str, Any]]


def _normalized_word(raw_word: Any) -> str:
    return str(raw_word or "").strip()


def _word_key(word: Dict[str, Any]) -> tuple[float, float, str]:
    return (
        round(float(word["start"]), 3),
        round(float(word["end"]), 3),
        _normalized_word(word.get("word")).casefold(),
    )


def _join_word_text(words: Iterable[Dict[str, Any]]) -> str:
    text = " ".join(
        token
        for token in (_normalized_word(word.get("word")) for word in words)
        if token
    )
    return _PUNCTUATION_SPACING_RE.sub(r"\1", text).strip()


def _last_lexical_token(text: str) -> str:
    tokens = _LEXICAL_TOKEN_RE.findall(str(text or "").casefold())
    return tokens[-1] if tokens else ""


def is_linguistically_complete_text(text: str) -> bool:
    """Reject explicit Turkish continuation markers masquerading as sentence ends."""
    value = str(text or "").strip()
    if not value:
        return True
    if value.endswith((",", ";", ":")):
        return False
    return _last_lexical_token(value) not in _DANGLING_ENDINGS


def _starts_as_continuation(text: str) -> bool:
    value = str(text or "").lstrip("\"'“”‘’([{ ")
    if not value:
        return False
    tokens = _LEXICAL_TOKEN_RE.findall(value.casefold())
    first_token = tokens[0] if tokens else ""
    # Whisper casing is not reliable enough to use a lowercase first letter as
    # proof that two utterances belong to the same sentence.  Only explicit
    # Turkish continuation markers are safe at this stage; semantic
    # incompleteness of the previous utterance is checked by the caller.
    return first_token in _CONTINUATION_STARTERS


def build_speech_context(
    shots: Iterable[Dict[str, Any]],
    *,
    silence_boundary_sec: float = SILENCE_BOUNDARY_SEC,
) -> SpeechContext:
    """Collect, validate and deduplicate word timestamps, then group utterances."""
    unique_words: Dict[tuple[float, float, str], Dict[str, Any]] = {}
    for shot in shots:
        for raw_word in shot.get("words") or []:
            try:
                start = float(raw_word.get("start", 0.0))
                end = float(raw_word.get("end", start))
            except (TypeError, ValueError):
                continue
            token = _normalized_word(raw_word.get("word"))
            if not token or end <= start or start < 0:
                continue
            try:
                probability = float(raw_word.get("probability", 0.0) or 0.0)
            except (TypeError, ValueError):
                probability = 0.0
            word = {
                "start": round(start, 3),
                "end": round(end, 3),
                "word": token,
                "probability": round(probability, 4),
            }
            key = _word_key(word)
            previous = unique_words.get(key)
            if previous is None or word["probability"] > previous["probability"]:
                unique_words[key] = word

    words = sorted(
        unique_words.values(),
        key=lambda word: (word["start"], word["end"], word["word"]),
    )
    utterances: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    for index, word in enumerate(words):
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        silence_gap = (
            max(0.0, float(next_word["start"]) - float(word["end"]))
            if next_word is not None
            else 0.0
        )
        punctuation_boundary = bool(_SENTENCE_END_RE.search(word["word"]))
        silence_boundary = next_word is not None and silence_gap >= silence_boundary_sec
        stream_end = next_word is None
        if punctuation_boundary or silence_boundary or stream_end:
            utterances.append({
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "text": _join_word_text(current),
                "words": list(current),
                "semantic_complete": is_linguistically_complete_text(
                    _join_word_text(current)
                ),
                "closed_by": (
                    "punctuation"
                    if punctuation_boundary
                    else "silence"
                    if silence_boundary
                    else "stream_end"
                ),
            })
            current = []

    return SpeechContext(words=words, utterances=utterances)


def _overlaps(start: float, end: float, item: Dict[str, Any]) -> bool:
    return float(item["end"]) > start and float(item["start"]) < end


def _word_containing(
    words: Iterable[Dict[str, Any]],
    timestamp: float,
) -> Optional[Dict[str, Any]]:
    return next(
        (
            word
            for word in words
            if float(word["start"]) + BOUNDARY_EPSILON_SEC
            < timestamp
            < float(word["end"]) - BOUNDARY_EPSILON_SEC
        ),
        None,
    )


def _fallback_transcript(
    start: float,
    end: float,
    shots: Iterable[Dict[str, Any]],
) -> tuple[bool, str]:
    texts = []
    has_speech = False
    for shot in shots:
        if float(shot.get("end_seconds", 0.0)) <= start:
            continue
        if float(shot.get("start_seconds", 0.0)) >= end:
            continue
        text = str(shot.get("transcript_text", "") or "").strip()
        if float(shot.get("speech_ratio", 0.0) or 0.0) > 0.1 or text:
            has_speech = True
        if text and text not in texts:
            texts.append(text)
    return has_speech, " ".join(texts).strip()


def describe_segment_window(
    start: float,
    end: float,
    shots: Iterable[Dict[str, Any]],
    context: Optional[SpeechContext] = None,
) -> Dict[str, Any]:
    context = context or build_speech_context(shots)
    selected_words = [word for word in context.words if _overlaps(start, end, word)]
    if not selected_words:
        has_speech, transcript = _fallback_transcript(start, end, shots)
        return {
            "has_speech": has_speech,
            "transcript_text": transcript,
            "word_count": 0,
            "complete_utterance": True,
        }

    overlapping_utterances = [
        utterance for utterance in context.utterances if _overlaps(start, end, utterance)
    ]
    complete_utterance = True
    for utterance in overlapping_utterances:
        utterance_index = context.utterances.index(utterance)
        temporal_complete = (
            start <= float(utterance["start"]) + BOUNDARY_EPSILON_SEC
            and end >= float(utterance["end"]) - BOUNDARY_EPSILON_SEC
        )
        semantic_complete = bool(utterance.get("semantic_complete", True))
        if not semantic_complete and utterance_index + 1 < len(context.utterances):
            continuation = context.utterances[utterance_index + 1]
            gap = float(continuation["start"]) - float(utterance["end"])
            semantic_complete = (
                0.0 <= gap <= CONTINUATION_GAP_SEC
                and end >= float(continuation["end"]) - BOUNDARY_EPSILON_SEC
            )
        if not temporal_complete or not semantic_complete:
            complete_utterance = False
            break
    return {
        "has_speech": True,
        "transcript_text": _join_word_text(selected_words),
        "word_count": len(selected_words),
        "complete_utterance": complete_utterance,
    }


def expand_to_complete_utterances(
    start: float,
    end: float,
    context: SpeechContext,
    video_duration: float,
    padding: float = 0.22,
) -> tuple[float, float]:
    """Keep a selected visual interval, extending its edges across whole speech turns.

    Unlike the category-specific alignment policy, this is a final safety repair:
    an action clip must not be discarded merely because a sentence crosses its
    shot boundary. The expansion is charged to the actual duration budget later.
    """
    start = max(0.0, float(start))
    end = min(float(video_duration), float(end))
    utterances = context.utterances
    for _ in range(len(utterances) + 1):
        previous = start, end
        for index, utterance in enumerate(utterances):
            if not _overlaps(start, end, utterance):
                continue
            start = min(start, float(utterance["start"]))
            end = max(end, float(utterance["end"]))
            if not utterance.get("semantic_complete", True) and index + 1 < len(utterances):
                continuation = utterances[index + 1]
                gap = float(continuation["start"]) - float(utterance["end"])
                if 0.0 <= gap <= CONTINUATION_GAP_SEC:
                    end = max(end, float(continuation["end"]))
        if (start, end) == previous:
            break
    # Add acoustic room only where doing so does not create a new, partial turn.
    padded_start = max(0.0, start - padding)
    padded_end = min(float(video_duration), end + padding)
    if all(not _overlaps(padded_start, start, row) for row in utterances):
        start = padded_start
    if all(not _overlaps(end, padded_end, row) for row in utterances):
        end = padded_end
    return round(start, 3), round(end, 3)


def _safe_end_at_or_before(
    context: SpeechContext,
    start: float,
    target_end: float,
    category: str,
    min_end: float,
) -> Optional[Dict[str, Any]]:
    prefer_sentence = SPEECH_BOUNDARY_POLICIES.get(
        category.lower(),
        SPEECH_BOUNDARY_POLICIES["importance"],
    )["prefer_sentence"]
    if prefer_sentence:
        utterance_ends = [
            float(utterance["end"])
            for utterance in context.utterances
            if min_end <= float(utterance["end"]) <= target_end + BOUNDARY_EPSILON_SEC
            and float(utterance["end"]) > start
            and bool(utterance.get("semantic_complete", True))
        ]
        if utterance_ends:
            return {"end": max(utterance_ends), "mode": "sentence", "complete": True}

    word_ends = [
        float(word["end"])
        for word in context.words
        if min_end <= float(word["end"]) <= target_end + BOUNDARY_EPSILON_SEC
        and float(word["end"]) > start
    ]
    if word_ends:
        return {"end": max(word_ends), "mode": "word", "complete": False}
    return None


def align_segment_boundaries(
    segment_start: float,
    segment_end: float,
    shots: List[Dict[str, Any]],
    *,
    category: str,
    max_segment_duration: float,
    context: Optional[SpeechContext] = None,
) -> Dict[str, Any]:
    """Align a segment without cutting words and, when allowed, whole utterances."""
    context = context or build_speech_context(shots)
    original_start = max(0.0, float(segment_start))
    original_end = max(original_start, float(segment_end))
    aligned_start = original_start
    aligned_end = original_end
    policy = SPEECH_BOUNDARY_POLICIES.get(
        category.lower(),
        SPEECH_BOUNDARY_POLICIES["importance"],
    )
    reasons: List[str] = []
    boundary_mode = "shot"

    if not context.words:
        overlapping_shots = [
            shot
            for shot in shots
            if float(shot.get("end_seconds", 0.0)) > original_start
            and float(shot.get("start_seconds", 0.0)) < original_end
        ]
        speech_shots = [
            shot
            for shot in overlapping_shots
            if float(shot.get("speech_ratio", 0.0) or 0.0) > 0.1
            or str(shot.get("transcript_text", "") or "").strip()
        ]
        if speech_shots and policy["prefer_sentence"]:
            candidate_start = float(speech_shots[0]["start_seconds"])
            candidate_end = float(speech_shots[-1]["end_seconds"])
            if (
                original_end - original_start > max_segment_duration
                and candidate_end - candidate_start <= max_segment_duration
            ):
                aligned_start = candidate_start
                aligned_end = candidate_end
                reasons.append("focused_fallback_speech_window")
            else:
                if 0.0 < original_start - candidate_start <= policy["max_expansion"]:
                    aligned_start = candidate_start
                if 0.0 < candidate_end - original_end <= policy["max_expansion"]:
                    aligned_end = candidate_end
            boundary_mode = "fallback"
            reasons.append("word_timestamps_missing")
        if aligned_end - aligned_start > max_segment_duration:
            aligned_end = aligned_start + max_segment_duration
            reasons.append("fallback_max_duration_clamp")
        description = describe_segment_window(aligned_start, aligned_end, shots, context)
        return {
            "start": aligned_start,
            "end": aligned_end,
            "original_start": original_start,
            "original_end": original_end,
            "start_delta": aligned_start - original_start,
            "end_delta": aligned_end - original_end,
            "boundary_mode": boundary_mode,
            "boundary_reason": ",".join(reasons) or "shot_boundary",
            **description,
        }

    overlapping_utterances = [
        utterance
        for utterance in context.utterances
        if _overlaps(original_start, original_end, utterance)
    ]

    if policy["prefer_sentence"] and overlapping_utterances:
        first_utterance = overlapping_utterances[0]
        last_utterance = overlapping_utterances[-1]

        first_index = context.utterances.index(first_utterance)
        previous = context.utterances[first_index - 1] if first_index > 0 else None
        needs_leading_context = bool(
            previous is not None
            and (
                not bool(previous.get("semantic_complete", True))
                or _starts_as_continuation(str(first_utterance.get("text", "")))
            )
        )
        if needs_leading_context:
            leading_gap = float(first_utterance["start"]) - float(previous["end"])
            if (
                0.0 <= leading_gap <= LEADING_CONTEXT_GAP_SEC
                and float(last_utterance["end"]) - float(previous["start"])
                <= max_segment_duration
            ):
                first_utterance = previous
                overlapping_utterances.insert(0, previous)
                reasons.append("expanded_leading_continuation_context")

        last_index = context.utterances.index(last_utterance)
        if (
            not bool(last_utterance.get("semantic_complete", True))
            and last_index + 1 < len(context.utterances)
        ):
            continuation = context.utterances[last_index + 1]
            continuation_gap = float(continuation["start"]) - float(last_utterance["end"])
            if (
                0.0 <= continuation_gap <= CONTINUATION_GAP_SEC
                and float(continuation["end"]) - float(first_utterance["start"])
                <= max_segment_duration
            ):
                last_utterance = continuation
                overlapping_utterances.append(continuation)
                reasons.append("expanded_dangling_continuation")
        start_expansion = original_start - float(first_utterance["start"])
        end_expansion = float(last_utterance["end"]) - original_end
        if BOUNDARY_EPSILON_SEC < start_expansion <= policy["max_expansion"]:
            aligned_start = float(first_utterance["start"])
            boundary_mode = "sentence"
            reasons.append("expanded_to_utterance_start")
        if BOUNDARY_EPSILON_SEC < end_expansion <= policy["max_expansion"]:
            aligned_end = float(last_utterance["end"])
            boundary_mode = "sentence"
            reasons.append("expanded_to_utterance_end")

        # A question without its response is not a coherent dialogue unit. Include
        # the next utterance when it starts shortly afterwards, even if the camera
        # has already cut to the listener/reply shot.
        if _QUESTION_END_RE.search(str(last_utterance.get("text", "")).strip()):
            try:
                last_index = context.utterances.index(last_utterance)
            except ValueError:
                last_index = -1
            next_utterance = (
                context.utterances[last_index + 1]
                if 0 <= last_index < len(context.utterances) - 1
                else None
            )
            if next_utterance is not None:
                response_gap = (
                    float(next_utterance["start"]) - float(last_utterance["end"])
                )
                response_end = float(next_utterance["end"])
                if (
                    0.0 <= response_gap <= QUESTION_RESPONSE_GAP_SEC
                    and response_end - aligned_start <= max_segment_duration
                ):
                    aligned_end = max(aligned_end, response_end)
                    boundary_mode = "exchange"
                    reasons.append("expanded_question_to_response")
                elif original_end > float(last_utterance["end"]) + BOUNDARY_EPSILON_SEC:
                    aligned_end = float(last_utterance["end"])
                    boundary_mode = "sentence"
                    reasons.append("trimmed_unanswered_question_reaction")
            elif original_end > float(last_utterance["end"]) + BOUNDARY_EPSILON_SEC:
                aligned_end = float(last_utterance["end"])
                boundary_mode = "sentence"
                reasons.append("trimmed_unanswered_question_reaction")

        # Metadata must never knowingly call a partial spoken thought complete.
        # If the complete utterance fits the category limit, prefer semantic
        # completeness over the normal expansion-distance policy.
        full_start = float(first_utterance["start"])
        full_end = float(last_utterance["end"])
        if full_end - full_start <= max_segment_duration:
            if aligned_start > full_start + BOUNDARY_EPSILON_SEC:
                aligned_start = full_start
                boundary_mode = "sentence"
                reasons.append("forced_complete_utterance_start")
            if aligned_end < full_end - BOUNDARY_EPSILON_SEC:
                aligned_end = full_end
                boundary_mode = "sentence"
                reasons.append("forced_complete_utterance_end")

    start_word = _word_containing(context.words, aligned_start)
    if start_word is not None:
        expansion = aligned_start - float(start_word["start"])
        if expansion <= policy["max_expansion"]:
            aligned_start = float(start_word["start"])
            if boundary_mode == "shot":
                boundary_mode = "word"
            reasons.append("avoided_start_mid_word")

    end_word = _word_containing(context.words, aligned_end)
    if end_word is not None:
        expansion = float(end_word["end"]) - aligned_end
        if expansion <= policy["max_expansion"]:
            aligned_end = float(end_word["end"])
            if boundary_mode == "shot":
                boundary_mode = "word"
            reasons.append("avoided_end_mid_word")

    if aligned_end - aligned_start > max_segment_duration:
        hard_end = aligned_start + max_segment_duration
        safe_end = _safe_end_at_or_before(
            context,
            aligned_start,
            hard_end,
            category,
            aligned_start + 0.25,
        )
        if safe_end is not None:
            aligned_end = float(safe_end["end"])
            boundary_mode = str(safe_end["mode"])
            reasons.append("max_duration_safe_boundary")
        else:
            aligned_end = hard_end
            boundary_mode = "fallback"
            reasons.append("max_duration_no_safe_boundary")

    if aligned_end <= aligned_start:
        aligned_start = original_start
        aligned_end = original_end
        boundary_mode = "fallback"
        reasons.append("invalid_alignment_reverted")

    description = describe_segment_window(aligned_start, aligned_end, shots, context)
    return {
        "start": aligned_start,
        "end": aligned_end,
        "original_start": original_start,
        "original_end": original_end,
        "start_delta": aligned_start - original_start,
        "end_delta": aligned_end - original_end,
        "boundary_mode": boundary_mode,
        "boundary_reason": ",".join(reasons) or "shot_boundary",
        **description,
    }


def find_safe_budget_end(
    segment_start: float,
    target_end: float,
    shots: List[Dict[str, Any]],
    *,
    category: str,
    min_duration: float,
    context: Optional[SpeechContext] = None,
) -> Optional[Dict[str, Any]]:
    """Find a budget-fitting end without cutting a sentence in speech-aware modes."""
    context = context or build_speech_context(shots)
    start = float(segment_start)
    target = float(target_end)
    min_end = start + float(min_duration)
    if target < min_end:
        return None

    speech_words = [word for word in context.words if _overlaps(start, target, word)]

    # Budget trimming is stricter than ordinary action alignment: a clipped
    # sentence is not an acceptable way to fill the last few seconds.
    if speech_words:
        utterance_ends = [
            float(utterance["end"])
            for utterance in context.utterances
            if min_end <= float(utterance["end"]) <= target + BOUNDARY_EPSILON_SEC
            and _overlaps(start, target, utterance)
            and bool(utterance.get("semantic_complete", True))
        ]
        if not utterance_ends:
            return None
        return {
            "end": max(utterance_ends),
            "mode": "sentence",
            "reason": "budget_trimmed_to_utterance_end",
            "complete_utterance": True,
        }

    shot_ends = sorted({
        float(shot.get("end_seconds", 0.0))
        for shot in shots
        if min_end <= float(shot.get("end_seconds", 0.0)) <= target + BOUNDARY_EPSILON_SEC
    })
    for shot_end in reversed(shot_ends):
        if _word_containing(context.words, shot_end) is None:
            return {
                "end": shot_end,
                "mode": "shot",
                "reason": "budget_trimmed_to_shot_end",
                "complete_utterance": not speech_words,
            }

    safe_word = _safe_end_at_or_before(context, start, target, category, min_end)
    if safe_word is not None:
        return {
            "end": float(safe_word["end"]),
            "mode": str(safe_word["mode"]),
            "reason": "budget_trimmed_to_word_end",
            "complete_utterance": bool(safe_word["complete"]),
        }
    return None


def enforce_duration_ceiling(
    segments: List[Dict[str, Any]],
    target_duration_sec: float,
    shots: List[Dict[str, Any]],
    *,
    category: str = "importance",
    context: Optional[SpeechContext] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Guarantee the exported summary never exceeds target_duration_sec.

    This is the single, final authority on total output duration. It runs after
    every other stage (budget optimization, adjacent re-merge, targeted ASR
    re-expansion) that could have pushed the running total past the requested
    length. Segments are walked in chronological order; once the running total
    would exceed the target, the current segment is trimmed to the nearest safe
    sentence/word/shot boundary that still fits (never mid-word), and every later
    segment is dropped outright. If no safe boundary fits within what remains of
    the budget, the segment itself is dropped instead of being cut unsafely.
    """
    is_unbounded = (
        target_duration_sec is None
        or target_duration_sec <= 0
        or target_duration_sec >= DURATION_CEILING_UNBOUNDED_SEC
    )
    if is_unbounded:
        return list(segments), {"applied": False, "reason": "unbounded_target"}

    context = context or build_speech_context(shots)
    ordered = sorted(segments, key=lambda seg: float(seg["start"]))
    pre_total = round(sum(float(seg["duration"]) for seg in ordered), 3)

    mandatory = [seg for seg in ordered if seg.get("hard_anchor_roles")]
    protected = [seg for seg in ordered if seg.get("protected_narrative_chain")]
    if mandatory or (category.lower() == "importance" and protected):
        mandatory_total = sum(float(seg["end"]) - float(seg["start"]) for seg in mandatory)
        if mandatory_total > float(target_duration_sec) + BOUNDARY_EPSILON_SEC:
            raise ValueError(
                f"Zorunlu dönüm noktaları hedef süreye sığmıyor: en az {mandatory_total:.1f} sn gerekiyor"
            )
        kept = list(ordered)
        dropped_protected = []
        def current_total():
            return sum(float(seg["end"]) - float(seg["start"]) for seg in kept)
        while current_total() > float(target_duration_sec) + BOUNDARY_EPSILON_SEC:
            removable = [seg for seg in kept if not seg.get("hard_anchor_roles") and not seg.get("protected_narrative_chain")]
            if not removable:
                removable = [seg for seg in kept if not seg.get("hard_anchor_roles")]
            if not removable:
                raise ValueError("Zorunlu dönüm noktaları hedef süreye sığmıyor")
            excess = current_total() - float(target_duration_sec)
            if category.lower() == "importance":
                from src.selection.three_act import ACT_QUOTAS
                region_totals = {
                    region: sum(float(seg["duration"]) for seg in kept if seg.get("narrative_region") == region)
                    for region in ACT_QUOTAS
                }
                most_over = max(ACT_QUOTAS, key=lambda region: region_totals[region] - float(target_duration_sec) * ACT_QUOTAS[region])
                regional = [seg for seg in removable if seg.get("narrative_region") == most_over]
                if regional:
                    removable = regional
            victim = min(removable, key=lambda seg: (
                float(seg["duration"]) < excess,
                abs(float(seg["duration"]) - excess),
                float(seg.get("segment_score", 0.0)),
            ))
            if victim.get("protected_narrative_chain"):
                dropped_protected.append(victim.get("narrative_chain_role", "event"))
            kept.remove(victim)
        return kept, {
            "applied": len(kept) != len(ordered),
            "target_duration": round(float(target_duration_sec), 3),
            "pre_ceiling_duration": pre_total,
            "post_ceiling_duration": round(sum(float(seg["end"]) - float(seg["start"]) for seg in kept), 3),
            "dropped_segment_count": len(ordered) - len(kept),
            "trimmed_last_segment": False,
            "protected_anchor_roles": sorted({role for seg in mandatory for role in seg["hard_anchor_roles"]}),
            "dropped_soft_protected_roles": dropped_protected,
        }

    kept: List[Dict[str, Any]] = []
    accumulated = 0.0
    trimmed_segment = False

    for seg in ordered:
        start = float(seg["start"])
        end = float(seg["end"])
        duration = end - start

        if accumulated + duration <= float(target_duration_sec) + BOUNDARY_EPSILON_SEC:
            kept.append(seg)
            accumulated += duration
            continue

        remaining = float(target_duration_sec) - accumulated
        if remaining >= DURATION_CEILING_MIN_KEEP_SEC:
            safe_end = find_safe_budget_end(
                start,
                start + remaining,
                shots,
                category=category,
                min_duration=DURATION_CEILING_MIN_KEEP_SEC,
                context=context,
            )
            if safe_end is not None:
                new_end = float(safe_end["end"])
                description = describe_segment_window(start, new_end, shots, context)
                new_seg = dict(seg)
                previous_reason = str(seg.get("boundary_reason", "") or "")
                new_seg.update({
                    "end": round(new_end, 3),
                    "duration": round(new_end - start, 3),
                    "aligned_end": round(new_end, 3),
                    "end_delta": round(new_end - float(seg.get("original_end", end)), 3),
                    "boundary_mode": str(safe_end["mode"]),
                    "boundary_reason": (
                        f"{previous_reason},duration_ceiling_trim"
                        if previous_reason else "duration_ceiling_trim"
                    ),
                    "complete_utterance": bool(safe_end["complete_utterance"]),
                    "has_speech": description["has_speech"],
                    "transcript_text": description["transcript_text"],
                    "word_count": description["word_count"],
                    "duration_ceiling_trimmed": True,
                })
                kept.append(new_seg)
                accumulated += new_seg["duration"]
                trimmed_segment = True
        break

    dropped_count = len(ordered) - len(kept)
    post_total = round(sum(float(seg["duration"]) for seg in kept), 3)
    return kept, {
        "applied": dropped_count > 0 or trimmed_segment,
        "target_duration": round(float(target_duration_sec), 3),
        "pre_ceiling_duration": pre_total,
        "post_ceiling_duration": post_total,
        "dropped_segment_count": dropped_count,
        "trimmed_last_segment": trimmed_segment,
    }
