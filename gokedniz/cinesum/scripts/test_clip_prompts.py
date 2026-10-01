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

def test_clip_zero_shot_classification(video_alias: str = "video1"):
    base_dir = Path(__file__).resolve().parent.parent
    features_dir = base_dir / "outputs" / "features" / "visual"
    
    npy_file = features_dir / f"{video_alias}_clip_features.npy"
    meta_file = features_dir / f"{video_alias}_clip_metadata.json"

    if not npy_file.exists() or not meta_file.exists():
        print(f"[HATA] {video_alias} için CLIP matrisi veya metadata bulunamadı!")
        return

    # 1. Load pre-extracted keyframe image vectors (Shape: [Num_Keyframes, 512])
    image_vectors = np.load(str(npy_file))
    with open(meta_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. Define natural language text queries (Prompts)
    prompts = [
        "a fight or action scene with fast movement",
        "two people having a conversation or dialogue",
        "a calm indoor or outdoor scene"
    ]

    print("==========================================================================")
    print(f"CLIP CANLI TESTI: {video_alias.upper()} SAHNELERI ANLAMSAL OLARAK SINIFLANDIRILIYOR")
    print("==========================================================================")
    print(f"Toplam Analiz Edilen Keyframe Sayisi: {len(image_vectors)}")
    print(f"Test Edilen Dogal Dil Sorgulari (Prompts):")
    for p in prompts:
        print(f"  - '{p}'")
    print("--------------------------------------------------------------------------")

    # 3. Load CLIP Text Encoder
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # 4. Convert text prompts to 512-dim vectors
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_out = model.get_text_features(**inputs)

        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            text_vectors = text_out.pooler_output
        elif hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
            text_vectors = text_out.text_embeds
        elif isinstance(text_out, torch.Tensor):
            text_vectors = text_out
        else:
            text_vectors = text_out[1]

        # Handle 3D tensor if last_hidden_state returned
        if text_vectors.ndim == 3:
            text_vectors = text_vectors[:, 0, :]

        # Normalize text vectors (L2 Norm)
        text_vectors = text_vectors / text_vectors.norm(p=2, dim=-1, keepdim=True)
        text_vectors_np = text_vectors.cpu().numpy()  # Shape: (3, 512)

    # 5. Compute Cosine Similarity between Image Vectors & Text Vectors
    # Matrix Multiplication: [Num_Keyframes, 512] x [512, 3] -> [Num_Keyframes, 3]
    similarity_matrix = np.dot(image_vectors, text_vectors_np.T)

    # 6. Display Top Action Scenes and Top Dialogue Scenes
    action_scores = similarity_matrix[:, 0]
    dialogue_scores = similarity_matrix[:, 1]

    # Top 3 Action Keyframes
    top_action_indices = np.argsort(action_scores)[::-1][:3]
    print("\n--- TOP 3 AKSIYON SAHNESI (Action Scene Matches) ---")
    for rank, idx in enumerate(top_action_indices, 1):
        meta = metadata[idx]
        score = action_scores[idx]
        print(
            f"   #{rank} -> Sahne: {meta['scene_id']:2d} | Zaman: {meta['timestamp_seconds']:6.2f}s | "
            f"Dosya: {meta['file_name']} | CLIP Benzerlik Skoru: {score:.4f}"
        )

    # Top 3 Dialogue Keyframes
    top_dialogue_indices = np.argsort(dialogue_scores)[::-1][:3]
    print("\n--- TOP 3 DIYALOG SAHNESI (Dialogue Scene Matches) ---")
    for rank, idx in enumerate(top_dialogue_indices, 1):
        meta = metadata[idx]
        score = dialogue_scores[idx]
        print(
            f"   #{rank} -> Sahne: {meta['scene_id']:2d} | Zaman: {meta['timestamp_seconds']:6.2f}s | "
            f"Dosya: {meta['file_name']} | CLIP Benzerlik Skoru: {score:.4f}"
        )

    print("\n==========================================================================")
    print("[BASARILI] CLIP anlamsal siniflandirma testi tamamlandi!")

if __name__ == "__main__":
    test_clip_zero_shot_classification("video1")
