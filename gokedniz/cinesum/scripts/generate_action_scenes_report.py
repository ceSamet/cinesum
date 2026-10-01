import os
import json
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

def generate_action_report_for_all_videos():
    base_dir = Path(__file__).resolve().parent.parent
    features_dir = base_dir / "outputs" / "features" / "visual"
    
    # Get all extracted CLIP feature files
    npy_files = sorted(
        list(features_dir.glob("*_clip_features.npy")),
        key=lambda x: int(x.stem.replace("_clip_features", "").replace("video", "")) if x.stem.replace("_clip_features", "").replace("video", "").isdigit() else 999
    )

    if not npy_files:
        print("[HATA] Çıkarılmış CLIP özellik matrisi bulunamadı!")
        return

    # Load CLIP model & encode action prompt
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    action_prompt = ["a fight, fast movement, explosion, or action scene"]
    inputs = processor(text=action_prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_out = model.get_text_features(**inputs)
        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            text_vec = text_out.pooler_output
        elif hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
            text_vec = text_out.text_embeds
        elif isinstance(text_out, torch.Tensor):
            text_vec = text_out
        else:
            text_vec = text_out[1]

        if text_vec.ndim == 3:
            text_vec = text_vec[:, 0, :]

        text_vec = text_vec / text_vec.norm(p=2, dim=-1, keepdim=True)
        action_text_np = text_vec.cpu().numpy().T  # Shape: (512, 1)

    print("==========================================================================")
    print("VIDEOLARDAKI EN BELIRGIN AKSIYON SAHNELERI RAPORU (CLIP ANLAMSAL ANALIZ)")
    print("==========================================================================")

    for npy_file in npy_files:
        alias = npy_file.stem.replace("_clip_features", "")
        meta_file = features_dir / f"{alias}_clip_metadata.json"
        
        if not meta_file.exists():
            continue

        image_vectors = np.load(str(npy_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if len(image_vectors) == 0 or len(metadata) == 0:
            continue

        # Cosine similarity scores: [Num_Keyframes, 512] x [512, 1] -> [Num_Keyframes, 1]
        scores = np.dot(image_vectors, action_text_np).flatten()

        # Find top 2 action scenes for this video
        top_indices = np.argsort(scores)[::-1][:2]

        print(f"\n---> {alias.upper()} (Toplam {len(metadata)} Keyframe):")
        for rank, idx in enumerate(top_indices, 1):
            meta = metadata[idx]
            sc = scores[idx]
            sec = meta.get("timestamp_seconds", (meta.get("frame_number", 0) / 25.0))
            timecode = f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}.{int((sec%1)*1000):03d}"
            
            print(
                f"   Top #{rank} Aksiyon Sahnesi -> Sahne {meta.get('scene_id', idx+1):2d} | "
                f"Zaman: {timecode} ({sec:6.2f}. saniye) | "
                f"CLIP Skor: {sc:.4f} | Resim: {meta.get('file_name', 'N/A')}"
            )

    print("\n==========================================================================")
    print("[BASARILI] Aksiyon sahneleri raporlama tamamlandi!")

if __name__ == "__main__":
    generate_action_report_for_all_videos()
