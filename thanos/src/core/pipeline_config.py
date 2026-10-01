from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


PIPELINE_VERSION = "3.1.0"
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AnalysisProfile:
    name: str
    whisper_model: str
    whisper_beam_size: int
    whisper_compute_type: Optional[str]
    clip_batch_size: Optional[int]
    word_timestamps: bool = True
    vad_filter: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


ANALYSIS_PROFILES = {
    "fast": AnalysisProfile(
        name="fast",
        whisper_model="small",
        whisper_beam_size=1,
        whisper_compute_type=None,
        clip_batch_size=None,
    ),
    "balanced": AnalysisProfile(
        name="balanced",
        whisper_model="small",
        whisper_beam_size=3,
        whisper_compute_type=None,
        clip_batch_size=None,
    ),
    "quality": AnalysisProfile(
        name="quality",
        whisper_model="medium",
        whisper_beam_size=5,
        whisper_compute_type=None,
        clip_batch_size=None,
    ),
}


def get_analysis_profile(name: str = "balanced") -> AnalysisProfile:
    normalized = (name or "balanced").strip().lower()
    if normalized not in ANALYSIS_PROFILES:
        choices = ", ".join(sorted(ANALYSIS_PROFILES))
        raise ValueError(f"Bilinmeyen analiz profili: {name}. Geçerli profiller: {choices}")
    return ANALYSIS_PROFILES[normalized]
