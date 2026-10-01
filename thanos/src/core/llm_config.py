import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse


DEFAULT_PARALON_BASE_URL = "https://paraloncloud.com/v1"
DEFAULT_PARALON_MODEL = "gemma3-4b-mac"
DEFAULT_LLM_TIMEOUT_SEC = 60.0


@dataclass(frozen=True)
class LLMConfig:
    api_key: Optional[str] = field(default=None, repr=False)
    base_url: str = DEFAULT_PARALON_BASE_URL
    model: str = DEFAULT_PARALON_MODEL
    timeout_sec: float = DEFAULT_LLM_TIMEOUT_SEC

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def to_public_dict(self) -> Dict[str, Any]:
        """Return diagnostics that are safe for API responses, logs and caches."""
        return {
            "enabled": self.enabled,
            "provider": "paralon",
            "base_url": self.base_url,
            "model": self.model,
            "timeout_sec": self.timeout_sec,
        }


def _load_dotenv_file(base_dir: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=base_dir / ".env", override=False)
    load_dotenv(dotenv_path=base_dir.parent / ".env", override=False)


def _validated_base_url(value: str) -> str:
    base_url = (value or DEFAULT_PARALON_BASE_URL).strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("PARALON_BASE_URL geçerli bir HTTP(S) adresi olmalıdır")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise ValueError("PARALON_BASE_URL uzak bağlantılarda HTTPS kullanmalıdır")
    return base_url


def load_llm_config(
    base_dir: Optional[Path] = None,
    *,
    load_env_file: bool = True,
) -> LLMConfig:
    if load_env_file:
        _load_dotenv_file(base_dir or Path(__file__).resolve().parents[2])

    raw_timeout = os.getenv("PARALON_TIMEOUT_SEC", str(DEFAULT_LLM_TIMEOUT_SEC))
    try:
        timeout_sec = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("PARALON_TIMEOUT_SEC sayısal olmalıdır") from exc
    if not 1.0 <= timeout_sec <= 120.0:
        raise ValueError("PARALON_TIMEOUT_SEC 1 ile 120 saniye arasında olmalıdır")

    api_key = (os.getenv("PARALON_API_KEY") or "").strip() or None
    model = (os.getenv("PARALON_MODEL") or DEFAULT_PARALON_MODEL).strip()
    if not model:
        raise ValueError("PARALON_MODEL boş olamaz")

    return LLMConfig(
        api_key=api_key,
        base_url=_validated_base_url(os.getenv("PARALON_BASE_URL", "")),
        model=model,
        timeout_sec=timeout_sec,
    )
