import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.core.vlm_config import VLMConfig, load_vlm_config


CAPTION_SCHEMA_VERSION = "1.0"
MAX_CAPTION_CHARS = 500
_GENERIC_CAPTION_PHRASES = (
    "movie trailer",
    "coming to theaters",
    "coming to cinemas",
    "the avengers movie",
    "transformers transformers",
)


def _compact(text: str, limit: int = MAX_CAPTION_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit]


def _caption_is_informative(text: str) -> bool:
    normalized = _compact(text).casefold()
    if not normalized:
        return False
    tokens = re.findall(r"[\w']+", normalized, flags=re.UNICODE)
    if len(tokens) >= 6 and len(set(tokens)) / len(tokens) < 0.35:
        return False
    if any(phrase in normalized for phrase in _GENERIC_CAPTION_PHRASES):
        return False
    return len(tokens) >= 3


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_keyframes(base_dir: Path, video_alias: str) -> Dict[int, Path]:
    directory = base_dir / "outputs" / "pyscenedetect" / "keyframes" / video_alias
    metadata_path = directory / "keyframes_metadata.json"
    if not metadata_path.exists():
        return {}
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["scene_id"]), []).append(row)
    result = {}
    for scene_id, frames in grouped.items():
        frames.sort(key=lambda row: int(row.get("kf_index", 1)))
        chosen = frames[len(frames) // 2]
        path = Path(str(chosen.get("file_path", "")))
        if not path.exists():
            path = directory / str(chosen.get("file_name", ""))
        if path.exists():
            result[scene_id] = path
    return result


def _context_frame_paths(keyframes: Dict[int, Path], scene_id: int) -> List[Path]:
    """Return previous/current/next visual evidence for a short movie event."""
    ordered_ids = sorted(keyframes)
    try:
        index = ordered_ids.index(int(scene_id))
    except ValueError:
        return []
    selected_ids = ordered_ids[max(0, index - 1):min(len(ordered_ids), index + 2)]
    return [keyframes[value] for value in selected_ids]


def _context_hash(paths: List[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(_image_hash(path).encode("ascii"))
    return digest.hexdigest()


def _select_caption_candidates(
    candidates: List[Dict[str, Any]], max_frames: int
) -> List[Dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            0 if row.get("candidate_source") == "local_selected" else 1,
            -float(row.get("narrative_event_score", 0.0)),
            -float(row.get("importance_score", 0.0)),
            int(row["scene_id"]),
        ),
    )
    result = []
    story_counts: Dict[Any, int] = {}
    for row in ranked:
        story_id = row.get("story_scene_id", f"shot:{row['scene_id']}")
        if story_counts.get(story_id, 0) >= 2:
            continue
        result.append(row)
        story_counts[story_id] = story_counts.get(story_id, 0) + 1
        if len(result) >= max_frames:
            break
    return result


class TransformersVisualCaptioner:
    """Lazy BLIP + lightweight LLaVA backend; heavy imports happen only when used."""

    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self._blip = None
        self._blip_processor = None
        self._llava = None
        self._llava_processor = None
        self._device = None

    def _torch(self):
        import torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch

    @property
    def device(self) -> str:
        self._torch()
        return str(self._device)

    def _load_blip(self) -> None:
        if self._blip is not None:
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor
        torch = self._torch()
        self._blip_processor = BlipProcessor.from_pretrained(self.config.blip_model)
        self._blip = BlipForConditionalGeneration.from_pretrained(
            self.config.blip_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def caption_blip(self, image_path: Path) -> str:
        self._load_blip()
        torch = self._torch()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = self._blip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self._blip.generate(**inputs, max_new_tokens=45)
        return _compact(self._blip_processor.decode(output[0], skip_special_tokens=True))

    def release_blip(self) -> None:
        if self._blip is None:
            return
        import gc
        torch = self._torch()
        self._blip = None
        self._blip_processor = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_llava(self) -> None:
        if self._llava is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor
        torch = self._torch()
        self._llava_processor = AutoProcessor.from_pretrained(self.config.llava_model)
        self._llava = AutoModelForImageTextToText.from_pretrained(
            self.config.llava_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

    def caption_llava(self, image_paths: List[Path], blip_caption: str) -> str:
        self._load_llava()
        torch = self._torch()
        instruction = (
            "The images are consecutive movie frames in chronological order. "
            "Describe the visible change from before, through the decisive action, to its result. "
            "Focus on objects changing hands, falls, deaths, victories, power reveals, "
            "gestures and emotional reactions. Do not invent identities, dialogue or events. "
            f"A basic caption says: {blip_caption or '[unreliable or unavailable]'}."
        )
        conversation = [{
            "role": "user",
            "content": (
                [{"type": "image"} for _ in image_paths]
                + [{"type": "text", "text": instruction}]
            ),
        }]
        prompt = self._llava_processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as source:
                frame = source.convert("RGB")
                frame.thumbnail((336, 336), Image.Resampling.LANCZOS)
                images.append(frame.copy())
        inputs = self._llava_processor(images=images, text=prompt, return_tensors="pt").to(
            self.device
        )
        input_length = inputs["input_ids"].shape[-1]
        try:
            with torch.inference_mode():
                output = self._llava.generate(**inputs, max_new_tokens=48, do_sample=False)
        except Exception:
            del inputs
            del images
            if self.device == "cuda":
                torch.cuda.empty_cache()
            raise
        generated = output[0][input_length:]
        return _compact(self._llava_processor.decode(generated, skip_special_tokens=True))

    def release_all(self) -> None:
        import gc
        torch = self._torch()
        self._blip = None
        self._blip_processor = None
        self._llava = None
        self._llava_processor = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _read_cache(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") == CAPTION_SCHEMA_VERSION:
            return value
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    return {"schema_version": CAPTION_SCHEMA_VERSION, "records": {}}


def _write_cache(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def enrich_candidates_with_visual_captions(
    candidates: List[Dict[str, Any]],
    *,
    base_dir: Path,
    video_alias: str,
    config: Optional[VLMConfig] = None,
    captioner: Optional[Any] = None,
) -> Dict[str, Any]:
    """Add cached visual semantics to a bounded shortlist without risking selection."""
    config = config or load_vlm_config(Path(base_dir))
    if not config.enabled:
        return {"status": "disabled", "reason": "vlm_disabled"}
    keyframes = _load_keyframes(Path(base_dir), video_alias)
    shortlisted = [
        row for row in _select_caption_candidates(candidates, config.max_frames)
        if int(row["scene_id"]) in keyframes
    ]
    if not shortlisted:
        return {"status": "fallback_no_captions", "reason": "keyframes_missing"}

    # Scoring has already consumed the cached CLIP embeddings. Keeping the CLIP
    # model resident while BLIP/LLaVA load can exhaust a 6 GB laptop GPU.
    try:
        from src.features.clip_extractor import release_clip_model
        release_clip_model()
    except Exception:
        pass

    cache_path = Path(base_dir) / "outputs" / "cache" / "vlm" / video_alias / "captions.json"
    cache = _read_cache(cache_path)
    records = cache.setdefault("records", {})
    backend = captioner or TransformersVisualCaptioner(config)
    started = time.perf_counter()
    blip_deadline = started + config.time_budget_sec * 0.55
    errors = []
    cache_hits = 0
    blip_count = 0
    llava_count = 0
    low_quality_blip_count = 0

    prepared = []
    for row in shortlisted:
        scene_id = int(row["scene_id"])
        image_path = keyframes[scene_id]
        image_digest = _image_hash(image_path)
        key = str(scene_id)
        record = records.get(key, {})
        valid = (
            record.get("image_hash") == image_digest
            and record.get("blip_model") == config.blip_model
        )
        if valid and record.get("blip_caption"):
            cache_hits += 1
        elif time.perf_counter() <= blip_deadline:
            try:
                record = {
                    "scene_id": scene_id,
                    "image_hash": image_digest,
                    "image_file": image_path.name,
                    "blip_model": config.blip_model,
                    "llava_model": config.llava_model,
                    "blip_caption": _compact(backend.caption_blip(image_path)),
                }
                records[key] = record
                blip_count += 1
            except Exception as exc:
                errors.append(f"blip:{type(exc).__name__}")
                break
        context_paths = _context_frame_paths(keyframes, scene_id) or [image_path]
        prepared.append((row, image_path, context_paths, record))

    llava_ranked = sorted(
        prepared,
        key=lambda item: (
            float(item[0].get("speech_ratio", 0.0)),
            -float(item[0].get("narrative_event_score", 0.0)),
            -float(item[0].get("importance_score", 0.0)),
        ),
    )[:config.llava_max_frames]
    if hasattr(backend, "release_blip"):
        backend.release_blip()
    llava_scene_ids = {int(item[0]["scene_id"]) for item in llava_ranked}
    for row, image_path, context_paths, record in llava_ranked:
        if time.perf_counter() - started > config.time_budget_sec:
            break
        context_digest = _context_hash(context_paths)
        valid = (
            record.get("llava_model") == config.llava_model
            and bool(record.get("llava_caption"))
            and (
                record.get("llava_context_hash") == context_digest
                or (
                    record.get("llava_context_mode") == "single_frame_after_oom"
                    and record.get("llava_context_hash") == _context_hash([context_paths[len(context_paths) // 2]])
                )
            )
        )
        if valid:
            continue
        try:
            record["llava_model"] = config.llava_model
            record["llava_context_hash"] = context_digest
            record["llava_context_frames"] = [path.name for path in context_paths]
            blip_for_prompt = record.get("blip_caption", "")
            if not _caption_is_informative(blip_for_prompt):
                blip_for_prompt = ""
            record["llava_caption"] = _compact(
                backend.caption_llava(context_paths, blip_for_prompt)
            )
            records[str(int(row["scene_id"]))] = record
            llava_count += 1
        except Exception as exc:
            if (type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).casefold()) and len(context_paths) > 1:
                # Three frames can exceed 6 GB VRAM even after BLIP is released.
                # Retry only the center frame and record the reduced context.
                try:
                    center_path = context_paths[len(context_paths) // 2]
                    record["llava_caption"] = _compact(
                        backend.caption_llava([center_path], blip_for_prompt)
                    )
                    record["llava_context_hash"] = _context_hash([center_path])
                    record["llava_context_frames"] = [center_path.name]
                    record["llava_context_mode"] = "single_frame_after_oom"
                    records[str(int(row["scene_id"]))] = record
                    llava_count += 1
                    continue
                except Exception as retry_exc:
                    errors.append(f"llava_single_frame:{type(retry_exc).__name__}")
            else:
                errors.append(f"llava:{type(exc).__name__}")
            break

    captioned = 0
    for row, _, _, record in prepared:
        parts = []
        if _caption_is_informative(record.get("blip_caption", "")):
            parts.append(f"BLIP: {record['blip_caption']}")
        elif record.get("blip_caption"):
            low_quality_blip_count += 1
        if int(row["scene_id"]) in llava_scene_ids and record.get("llava_caption"):
            parts.append(f"LLaVA: {record['llava_caption']}")
        if parts:
            row["visual_caption"] = _compact(" | ".join(parts))
            row["visual_caption_source"] = "blip_llava" if len(parts) == 2 else "blip"
            captioned += 1

    cache["updated_at_unix"] = round(time.time(), 3)
    cache["blip_model"] = config.blip_model
    cache["llava_model"] = config.llava_model
    if records:
        _write_cache(cache_path, cache)
    if hasattr(backend, "release_all"):
        backend.release_all()
    elapsed = round(time.perf_counter() - started, 3)
    if captioned == 0:
        status = "fallback_no_captions"
    elif errors or captioned < len(shortlisted):
        status = "partial"
    elif blip_count == 0 and llava_count == 0:
        status = "cached"
    else:
        status = "applied"
    return {
        "status": status,
        "captioned_scene_count": captioned,
        "shortlist_count": len(shortlisted),
        "cache_hit_count": cache_hits,
        "blip_generated_count": blip_count,
        "llava_generated_count": llava_count,
        "low_quality_blip_count": low_quality_blip_count,
        "elapsed_sec": elapsed,
        "device": getattr(backend, "device", "test"),
        "errors": errors[:2],
        "time_budget_sec": config.time_budget_sec,
    }
