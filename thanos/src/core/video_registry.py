"""Content-addressed upload registry: identical files reuse the same analysis."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import BinaryIO


_registry_lock = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _store_video_unlocked(file: BinaryIO, video_dir: Path) -> tuple[str, Path, bool]:
    video_dir.mkdir(parents=True, exist_ok=True)
    registry_path = video_dir / "content_index.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        registry = {}
    temp_path = video_dir / f".incoming_{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    try:
        with temp_path.open("wb") as target:
            for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
                target.write(block)
                digest.update(block)
        content_hash = digest.hexdigest()
        # An old video may predate the index; hash it once and keep the result.
        for existing in video_dir.glob("*.mp4"):
            if existing.stem not in registry:
                registry[existing.stem] = _sha256(existing)
        for alias, previous_hash in registry.items():
            existing = video_dir / f"{alias}.mp4"
            if previous_hash == content_hash and existing.exists():
                temp_path.unlink()
                return alias, existing, True
        numbers = [
            int(match.group(1))
            for path in video_dir.glob("video*.mp4")
            if (match := re.fullmatch(r"video(\d+)", path.stem))
        ]
        alias = f"video{max(numbers, default=0) + 1}"
        destination = video_dir / f"{alias}.mp4"
        os.replace(temp_path, destination)
        registry[alias] = content_hash
        registry_temp = registry_path.with_suffix(".json.tmp")
        registry_temp.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        os.replace(registry_temp, registry_path)
        return alias, destination, False
    finally:
        temp_path.unlink(missing_ok=True)


def store_video(file: BinaryIO, video_dir: Path) -> tuple[str, Path, bool]:
    with _registry_lock:
        return _store_video_unlocked(file, video_dir)
