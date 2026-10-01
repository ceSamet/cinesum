import os
from typing import List, Dict, Any, Optional

AUDIO_FADE_IN = 0.15
AUDIO_FADE_OUT = 0.25
AUDIO_CROSSFADE = 0.25
VIDEO_CROSSFADE_DURATION = 0.20

def get_audio_fade_filter(
    segment_duration: float,
    fade_in: float = AUDIO_FADE_IN,
    fade_out: float = AUDIO_FADE_OUT,
    *,
    preserve_speech: bool = False,
) -> str:
    """
    Generates FFmpeg audio fade filter string for segment boundaries:
    afade=t=in:ss=0:d=in_dur,afade=t=out:st=out_start:d=out_dur
    Clamps fade duration to avoid audio clipping or sync drift on short segments.
    """
    if preserve_speech:
        return ""

    actual_fade_in = min(fade_in, segment_duration * 0.10)
    actual_fade_out = min(fade_out, segment_duration * 0.10)

    if actual_fade_in <= 0.01 and actual_fade_out <= 0.01:
        return ""

    filters = []
    if actual_fade_in > 0.01:
        filters.append(f"afade=t=in:ss=0:d={actual_fade_in:.3f}")
    if actual_fade_out > 0.01:
        out_start = max(0.0, segment_duration - actual_fade_out)
        filters.append(f"afade=t=out:st={out_start:.3f}:d={actual_fade_out:.3f}")

    return ",".join(filters)

def get_transition_type(temporal_gap: float, merge_gap: float = 2.5, hard_cut_gap: float = 30.0) -> str:
    """
    Determines transition type based on temporal gap between source segments:
    - temporal_gap <= merge_gap: merge (continuous source interval)
    - temporal_gap < hard_cut_gap: hard_cut with audio fade
    - temporal_gap >= hard_cut_gap: crossfade / dissolve
    """
    if temporal_gap <= merge_gap:
        return "merge"
    elif temporal_gap < hard_cut_gap:
        return "hard_cut"
    else:
        return "crossfade"
