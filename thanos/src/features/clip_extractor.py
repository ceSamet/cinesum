import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

# Fix Windows Symlink Permission Issue for HuggingFace Hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

_clip_model = None
_clip_processor = None


def release_clip_model() -> None:
    """Release the cached CLIP model before another GPU-heavy inference stage."""
    global _clip_model, _clip_processor
    if _clip_model is None and _clip_processor is None:
        return
    import gc
    try:
        if _clip_model is not None:
            _clip_model = _clip_model.to("cpu")
    except Exception:
        pass
    _clip_model = None
    _clip_processor = None
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_clip_device() -> str:
    """Return the best Torch device available for CLIP inference."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Lazy loader for HuggingFace CLIP model and processor.
    """
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        local_model = Path(__file__).resolve().parents[2] / "models" / "clip"
        source = str(local_model) if (local_model / "config.json").exists() else model_name
        print(f"[CLIP] Model yükleniyor: {source}...")
        try:
            _clip_processor = CLIPProcessor.from_pretrained(source, local_files_only=True)
            _clip_model = CLIPModel.from_pretrained(source, local_files_only=True)
        except (OSError, ValueError):
            print("[CLIP] Yerel model eksik; Hugging Face kaynağı deneniyor...")
            _clip_processor = CLIPProcessor.from_pretrained(model_name, local_files_only=False)
            _clip_model = CLIPModel.from_pretrained(model_name, local_files_only=False)

        _clip_model.eval()
    return _clip_model, _clip_processor

def load_or_reconstruct_keyframes_metadata(alias: str, keyframes_dir: Path) -> List[Dict[str, Any]]:
    """
    Load keyframes_metadata.json if present, or reconstruct metadata list directly from .jpg files in the folder.
    """
    v_dir = keyframes_dir / alias
    # Check if v_dir or nested v_dir/v_dir exists
    if not v_dir.exists():
        return []

    json_path = v_dir / "keyframes_metadata.json"
    nested_json = v_dir / alias / "keyframes_metadata.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if nested_json.exists():
        with open(nested_json, "r", encoding="utf-8") as f:
            return json.load(f)

    # Reconstruct from .jpg files directly in v_dir or v_dir/alias
    jpg_files = list(v_dir.glob("*.jpg"))
    if not jpg_files and (v_dir / alias).exists():
        jpg_files = list((v_dir / alias).glob("*.jpg"))

    if not jpg_files:
        return []

    # Sort files naturally by scene and kf index
    jpg_files.sort(key=lambda p: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', p.name)])

    reconstructed_meta = []
    for idx, img_path in enumerate(jpg_files):
        filename = img_path.name
        # Parse scene_id and kf_index from filename like video4_scene_003_kf2.jpg
        match = re.search(r'scene_(\d+)_kf(\d+)', filename)
        scene_id = int(match.group(1)) if match else (idx + 1)
        kf_index = int(match.group(2)) if match else 1

        reconstructed_meta.append({
            "video_stem": alias,
            "scene_id": scene_id,
            "kf_index": kf_index,
            "file_name": filename,
            "file_path": str(img_path.resolve()),
        })

    # Save reconstructed metadata for future use
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(reconstructed_meta, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    return reconstructed_meta

def extract_clip_features_for_video(
    alias: str,
    keyframes_base_dir: str,
    output_npy_path: str,
    output_json_path: str,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: Optional[int] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extract L2-normalized 512-dim CLIP visual embeddings for keyframes of a video.
    """
    kf_base_path = Path(keyframes_base_dir)
    keyframes_meta = load_or_reconstruct_keyframes_metadata(alias, kf_base_path)

    if not keyframes_meta:
        return np.array([]), []

    model, processor = get_clip_model(model_name)
    device = resolve_clip_device()
    model = model.to(device)
    print(f"[CLIP] Cihaz: {device} | batch_size: {batch_size or (32 if device == 'cuda' else 8)}")

    valid_meta = []

    for kf in keyframes_meta:
        img_path = kf["file_path"]
        if Path(img_path).exists():
            valid_meta.append(kf)

    if not valid_meta:
        return np.array([]), []

    resolved_batch_size = batch_size or (32 if device == "cuda" else 8)
    feature_batches = []
    processed_meta = []

    for batch_start in range(0, len(valid_meta), resolved_batch_size):
        if cancel_check:
            cancel_check()
        batch_meta = valid_meta[batch_start:batch_start + resolved_batch_size]
        images = []
        loaded_meta = []
        for meta in batch_meta:
            try:
                with Image.open(meta["file_path"]) as source:
                    images.append(source.convert("RGB"))
                loaded_meta.append(meta)
            except Exception as exc:
                print(f"[UYARI] Görsel okunamadı: {meta['file_path']} ({exc})")

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.get_image_features(**inputs)
            if hasattr(outputs, "image_embeds"):
                image_features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                image_features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                image_features = outputs
            else:
                image_features = outputs[0]

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            feature_batches.append(image_features.cpu().numpy())
            processed_meta.extend(loaded_meta)

        images.clear()

    if not feature_batches:
        return np.array([]), []

    features_np = np.concatenate(feature_batches, axis=0)
    valid_meta = processed_meta

    # Add feature index to metadata
    for i, meta in enumerate(valid_meta):
        meta["feature_vector_index"] = i

    # Save outputs
    npy_path = Path(output_npy_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_path), features_np)

    json_out_path = Path(output_json_path)
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(valid_meta, f, indent=2, ensure_ascii=False)

    return features_np, valid_meta
