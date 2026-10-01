import os
import json
import re
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import numpy as np

from src.features.clip_extractor import get_clip_model
from src.summary.temporal_segment_builder import build_temporal_segments
from src.summary.summary_exporter import export_summary_segments
from src.summary.samet_speech_guard import guard_and_fit_segments, load_turns

def get_clip_model_and_processor():
    return get_clip_model("openai/clip-vit-base-patch32")

def generate_custom_query_summary(
    video_alias: str,
    text_prompt: str,
    base_dir: Path,
    target_duration_sec: float = 30.0,
    output_mp4_path: str = None,
    progress_callback: Optional[Callable[[int, str, str, str], None]] = None,
    narrative_mode: str = "local",
    base_scored_scenes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Encode natural language text query into 512-dim CLIP vector.
    Applies Contrastive Neutral Normalization, Z-Score Peak Filtering, and
    Temporal Segment Building to create coherent custom query video summaries.
    """
    video_path = base_dir / "dataset" / "video" / f"{video_alias}.mp4"
    visual_dir = base_dir / "outputs" / "features" / "visual"
    scene_lists_dir = base_dir / "outputs" / "pyscenedetect" / "scene_lists"
    summaries_dir = base_dir / "outputs" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    npy_file = visual_dir / f"{video_alias}_clip_features.npy"
    v_meta_file = visual_dir / f"{video_alias}_clip_metadata.json"
    scenes_json_file = scene_lists_dir / f"{video_alias}_scenes.json"

    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    if not npy_file.exists() or not v_meta_file.exists():
        raise FileNotFoundError(f"CLIP özellikleri bulunamadı: {video_alias}")
    if not scenes_json_file.exists():
        raise FileNotFoundError(f"Sahne listesi bulunamadı: {video_alias}")

    if progress_callback:
        progress_callback(30, "CLIP vektör verileri yükleniyor...", f"Sorgu: '{text_prompt}'", "stepClip")

    # Load data
    image_vectors = np.load(str(npy_file))
    with open(v_meta_file, "r", encoding="utf-8") as f:
        clip_meta = json.load(f)
    with open(scenes_json_file, "r", encoding="utf-8") as f:
        scenes_data = json.load(f)

    if progress_callback:
        progress_callback(45, "CLIP Text Encoder (Kontrastlı Filtreleme) çalıştırılıyor...", f"Prompt: '{text_prompt}'", "stepClip")

    # 1. Encode Target Text Prompt & Neutral Baseline Prompt
    neutral_prompt = "a video scene"
    model, processor = get_clip_model_and_processor()

    device = next(model.parameters()).device
    inputs = processor(text=[text_prompt, neutral_prompt], return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        text_out = model.get_text_features(**inputs)
        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            text_vecs = text_out.pooler_output
        elif hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
            text_vecs = text_out.text_embeds
        elif isinstance(text_out, torch.Tensor):
            text_vecs = text_out
        else:
            text_vecs = text_out[0]

        if text_vecs.ndim == 3:
            text_vecs = text_vecs[:, 0, :]

        # L2 Normalize
        text_vecs = text_vecs / text_vecs.norm(p=2, dim=-1, keepdim=True)
        target_vec = text_vecs[0:1].cpu().numpy().T  # Shape: (512, 1)
        neutral_vec = text_vecs[1:2].cpu().numpy().T  # Shape: (512, 1)

    # 2. Compute Target & Neutral Cosine Similarities
    target_sims = np.dot(image_vectors, target_vec).flatten()
    neutral_sims = np.dot(image_vectors, neutral_vec).flatten()

    # Contrastive score = Target Similarity - 0.35 * Neutral Similarity
    contrastive_sims = target_sims - 0.35 * neutral_sims

    # Map similarities to scene_ids
    scene_sims = {}
    raw_target_sims = {}

    for idx, kf in enumerate(clip_meta):
        sc_id = kf["scene_id"]
        if sc_id not in scene_sims:
            scene_sims[sc_id] = []
            raw_target_sims[sc_id] = []
        scene_sims[sc_id].append(float(contrastive_sims[idx]))
        raw_target_sims[sc_id].append(float(target_sims[idx]))

    # Rank scenes by peak contrastive similarity
    ranked_scenes = []
    scored_by_id = {
        int(row["scene_id"]): row for row in (base_scored_scenes or [])
    }
    for sc in scenes_data:
        sc_id = sc["scene_id"]
        sim_list = scene_sims.get(sc_id, [0.0])
        raw_list = raw_target_sims.get(sc_id, [0.0])

        max_sim = float(np.max(raw_list))
        contrastive_max = float(np.max(sim_list))

        sc_copy = {**scored_by_id.get(int(sc_id), {}), **sc}
        sc_copy["query_max_similarity"] = round(max_sim, 4)
        sc_copy["contrastive_score"] = round(contrastive_max, 4)
        sc_copy["importance_score"] = round(max_sim, 4)
        ranked_scenes.append(sc_copy)

    # 3. Build Coherent Temporal Segments using Temporal Segment Builder
    if progress_callback:
        progress_callback(60, "Coherent Temporal Segmentler oluşturuluyor...", f"Kategori: CUSTOM ('{text_prompt}')", "stepClip")

    seg_builder_res = build_temporal_segments(
        shots=ranked_scenes,
        category="custom",
        target_duration_sec=target_duration_sec,
        score_key="query_max_similarity",
        video_duration=ranked_scenes[-1]["end_seconds"] if ranked_scenes else None,
        narrative_mode=narrative_mode,
        base_dir=base_dir,
        video_alias=video_alias,
        custom_prompt=text_prompt,
        narrative_progress_callback=(
            (lambda pct, desc: progress_callback(pct, desc, "Opsiyonel anlatı iyileştirme", "stepClip"))
            if progress_callback else None
        ),
    )

    segments = seg_builder_res["segments"]
    segments, guard_report = guard_and_fit_segments(
        segments,
        turns=load_turns(base_dir / "outputs" / "features" / "audio" / f"{video_alias}_transcript.json"),
        video_duration=max((float(row["end_seconds"]) for row in ranked_scenes), default=0.0),
        target_duration_sec=target_duration_sec,
    )
    seg_builder_res["segments"] = segments
    seg_builder_res["actual_duration"] = round(sum(row["duration"] for row in segments), 3)
    seg_builder_res["speech_guard"] = guard_report

    if output_mp4_path is None:
        slug = re.sub(r'[^a-zA-Z0-9]+', '_', text_prompt).strip('_').lower()[:20]
        mode_suffix = "all" if (target_duration_sec <= 0 or target_duration_sec >= 9999) else f"{int(target_duration_sec)}s"
        output_mp4_path = str(summaries_dir / f"{video_alias}_custom_{slug}_{mode_suffix}_{narrative_mode}.mp4")

    output_p = Path(output_mp4_path)

    # Save Debug Metadata JSON
    debug_dir = output_p.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"{video_alias}_custom_{int(target_duration_sec)}s_segments.json"

    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(seg_builder_res, f, indent=2, ensure_ascii=False)

    # 4. Export smooth summary MP4
    out_path = export_summary_segments(
        video_path=str(video_path),
        segments=segments,
        output_mp4_path=str(output_p),
        category="custom",
        progress_callback=progress_callback
    )

    # Construct selected_scenes list matching segment timestamps for frontend playback
    selected_scenes = []
    for idx, seg in enumerate(segments, 1):
        selected_scenes.append({
            "scene_id": idx,
            "start_timecode": f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}",
            "end_timecode": f"{int(seg['end']//60):02d}:{int(seg['end']%60):02d}",
            "start_seconds": seg["start"],
            "end_seconds": seg["end"],
            "duration_seconds": seg["duration"],
            "query_max_similarity": seg["peak_score"],
            "action_score": seg["peak_score"],
            "importance_score": seg["segment_score"],
            "transcript_text": seg.get("transcript_text", ""),
        })

    return {
        "video_alias": video_alias,
        "text_prompt": text_prompt,
        "selected_scenes": selected_scenes,
        "total_duration_seconds": seg_builder_res["actual_duration"],
        "output_mp4_path": out_path,
        "debug_metadata": seg_builder_res,
    }
