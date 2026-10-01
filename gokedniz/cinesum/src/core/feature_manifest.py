import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.core.pipeline_config import PIPELINE_VERSION, SCHEMA_VERSION


def stable_config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


class FeatureManifest:
    """Versioned, stage-aware cache manifest for one source video."""

    def __init__(self, manifest_path: Path, video_path: Path):
        self.path = Path(manifest_path)
        self.video_path = Path(video_path)
        self.source = fingerprint_file(self.video_path)
        self.data = self._load()

        if self.data.get("source", {}).get("sha256") != self.source["sha256"]:
            self.data = self._new_data()
            self.save()

    def _new_data(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "source": self.source,
            "stages": {},
            "created_at_unix": time.time(),
            "updated_at_unix": time.time(),
        }

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._new_data()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return self._new_data()

        if data.get("schema_version") != SCHEMA_VERSION:
            return self._new_data()
        return data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["pipeline_version"] = PIPELINE_VERSION
        self.data["updated_at_unix"] = time.time()
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.path)

    def stage_is_valid(
        self,
        name: str,
        config: Dict[str, Any],
        artifacts: Iterable[Path],
    ) -> bool:
        stage = self.data.get("stages", {}).get(name, {})
        if stage.get("status") != "complete":
            return False
        if stage.get("config_hash") != stable_config_hash(config):
            return False
        return all(Path(artifact).exists() for artifact in artifacts)

    def mark_running(self, name: str, config: Dict[str, Any]) -> None:
        self.data.setdefault("stages", {})[name] = {
            "status": "running",
            "config": config,
            "config_hash": stable_config_hash(config),
            "started_at_unix": time.time(),
        }
        self.save()

    def mark_complete(
        self,
        name: str,
        config: Dict[str, Any],
        artifacts: Iterable[Path],
        processing_sec: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.data.setdefault("stages", {})[name] = {
            "status": "complete",
            "config": config,
            "config_hash": stable_config_hash(config),
            "artifacts": [str(Path(artifact).resolve()) for artifact in artifacts],
            "processing_sec": round(processing_sec, 4),
            "metadata": metadata or {},
            "completed_at_unix": time.time(),
        }
        self.save()

    def mark_failed(self, name: str, config: Dict[str, Any], error: Exception) -> None:
        self.data.setdefault("stages", {})[name] = {
            "status": "failed",
            "config": config,
            "config_hash": stable_config_hash(config),
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at_unix": time.time(),
        }
        self.save()

