import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_LLAVA_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


@dataclass(frozen=True)
class VLMConfig:
    enabled: bool = True
    blip_model: str = DEFAULT_BLIP_MODEL
    llava_model: str = DEFAULT_LLAVA_MODEL
    max_frames: int = 40
    llava_max_frames: int = 12
    time_budget_sec: float = 60.0

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "blip_model": self.blip_model,
            "llava_model": self.llava_model,
            "max_frames": self.max_frames,
            "llava_max_frames": self.llava_max_frames,
            "time_budget_sec": self.time_budget_sec,
        }


def _load_dotenv_file(base_dir: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=base_dir / ".env", override=False)


def _boolean_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} true/false olmalıdır")


def _integer_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} tam sayı olmalıdır") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} {minimum} ile {maximum} arasında olmalıdır")
    return value


def load_vlm_config(
    base_dir: Optional[Path] = None,
    *,
    load_env_file: bool = True,
) -> VLMConfig:
    if load_env_file:
        _load_dotenv_file(base_dir or Path(__file__).resolve().parents[2])
    try:
        budget = float(os.getenv("VLM_TIME_BUDGET_SEC", "60"))
    except (TypeError, ValueError) as exc:
        raise ValueError("VLM_TIME_BUDGET_SEC sayısal olmalıdır") from exc
    if not 10.0 <= budget <= 300.0:
        raise ValueError("VLM_TIME_BUDGET_SEC 10 ile 300 saniye arasında olmalıdır")
    max_frames = _integer_env("VLM_MAX_FRAMES", 40, 1, 64)
    llava_max_frames = _integer_env("VLM_LLAVA_MAX_FRAMES", 12, 0, 32)
    return VLMConfig(
        enabled=_boolean_env("VLM_ENABLED", True),
        blip_model=(os.getenv("VLM_BLIP_MODEL") or DEFAULT_BLIP_MODEL).strip(),
        llava_model=(os.getenv("VLM_LLAVA_MODEL") or DEFAULT_LLAVA_MODEL).strip(),
        max_frames=max_frames,
        llava_max_frames=min(llava_max_frames, max_frames),
        time_budget_sec=budget,
    )
