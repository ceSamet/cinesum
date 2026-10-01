import os
import sys
import json
import time
from pathlib import Path

# Disable HuggingFace Windows Symlink Requirement
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.clip_extractor import extract_clip_features_for_video

def run_clip_extraction_all_50_videos():
    base_dir = Path(__file__).resolve().parent.parent
    keyframes_dir = base_dir / "outputs" / "pyscenedetect" / "keyframes"
    visual_features_dir = base_dir / "outputs" / "features" / "visual"
    visual_features_dir.mkdir(parents=True, exist_ok=True)

    # Always generate video1 through video50 alias list dynamically
    all_video_aliases = [f"video{i}" for i in range(1, 51)]

    print("==================================================")
    print(f"CLIP Görsel Özellik Çıkarma ({len(all_video_aliases)} Video / ViT-B/32)")
    print("==================================================")

    start_total_time = time.time()
    success_count = 0
    skipped_count = 0
    total_vectors_extracted = 0

    for idx, alias in enumerate(all_video_aliases, 1):
        v_dir = keyframes_dir / alias
        if not v_dir.exists():
            print(f"[{idx:02d}/50] [HENÜZ YOK] Keyframe klasörü bulunamadı: {alias}")
            continue

        npy_out = visual_features_dir / f"{alias}_clip_features.npy"
        json_out = visual_features_dir / f"{alias}_clip_metadata.json"

        # Check if CLIP features already extracted
        if npy_out.exists() and json_out.exists():
            skipped_count += 1
            print(f"[{idx:02d}/50] [ATLANIYOR] CLIP zaten çıkarılmış: {alias}")
            continue

        t0 = time.time()
        features, valid_meta = extract_clip_features_for_video(
            alias=alias,
            keyframes_base_dir=str(keyframes_dir),
            output_npy_path=str(npy_out),
            output_json_path=str(json_out),
        )

        if len(valid_meta) == 0:
            print(f"[{idx:02d}/50] [UYARI] {alias} klasöründe keyframe resmi bulunamadı.")
            continue

        elapsed = round(time.time() - t0, 2)
        vec_count = len(valid_meta)
        total_vectors_extracted += vec_count
        success_count += 1

        print(
            f"[{idx:02d}/50] [BAŞARILI] {alias:7s} -> {vec_count:2d} keyframe için {features.shape} CLIP vektörü üretildi | "
            f"Süre: {elapsed:.2f}s"
        )

    total_elapsed = round(time.time() - start_total_time, 2)
    print("\n==================================================")
    print(f"[BAŞARILI] İşlem Tamamlandı! ({success_count} yeni CLIP vektörleşti, {skipped_count} önceden hazır)")
    print(f"Toplam Üretilen Vektör Sayısı: {total_vectors_extracted}")
    print(f"Toplam Geçen Süre: {total_elapsed:.2f} saniye (~{total_elapsed/60:.2f} dakika)")
    print("==================================================")

if __name__ == "__main__":
    run_clip_extraction_all_50_videos()
