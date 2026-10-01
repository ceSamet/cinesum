import sys
import json
import time
from pathlib import Path
import cv2

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scene_detection.pyscenedetect_runner import (
    detect_scenes_pyscenedetect,
    save_scenes_to_json,
    save_scenes_to_csv,
)
from src.scene_detection.keyframe_extractor import extract_keyframes_for_scenes

def process_all_50_videos():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"
    outputs_dir = base_dir / "outputs" / "pyscenedetect"
    scene_lists_dir = outputs_dir / "scene_lists"
    keyframes_dir = outputs_dir / "keyframes"

    scene_lists_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    # Get all video1.mp4 to video50.mp4 sorted
    video_files = sorted(
        list(video_dir.glob("video*.mp4")),
        key=lambda x: int(x.stem.replace("video", "")) if x.stem.replace("video", "").isdigit() else 999
    )

    print("==================================================")
    print(f"Tüm Dataset Videoları İşleniyor ({len(video_files)} Video)")
    print("Yöntem: PySceneDetect (adaptive_t3.0) + Netlik Tabanlı Keyframe")
    print("==================================================")

    start_total_time = time.time()
    processed_count = 0
    skipped_count = 0
    total_scenes_found = 0
    total_keyframes_extracted = 0

    for idx, v_path in enumerate(video_files, 1):
        alias = v_path.stem
        v_kf_meta_file = keyframes_dir / alias / "keyframes_metadata.json"

        # Check if already processed
        if v_kf_meta_file.exists():
            skipped_count += 1
            print(f"[{idx:02d}/{len(video_files):02d}] [ATLANIYOR] Zaten işlenmiş: {alias}")
            continue

        t0 = time.time()
        print(f"\n[{idx:02d}/{len(video_files):02d}] İşleniyor: {alias} ({v_path.name})")

        # 1. Detect scenes using PySceneDetect (adaptive detector t3.0)
        scenes = detect_scenes_pyscenedetect(
            video_path=str(v_path),
            detector_type="adaptive",
            threshold=3.0,
        )

        # Save scene list JSON and CSV
        json_out = scene_lists_dir / f"{alias}_adaptive_t3.0.json"
        csv_out = scene_lists_dir / f"{alias}_adaptive_t3.0.csv"
        save_scenes_to_json(scenes, str(json_out))
        save_scenes_to_csv(scenes, str(csv_out))

        # Also save a canonical default scene list JSON: {alias}_scenes.json
        canonical_json = scene_lists_dir / f"{alias}_scenes.json"
        save_scenes_to_json(scenes, str(canonical_json))

        # 2. Extract keyframes with Laplacian sharpness scoring
        v_kf_dir = keyframes_dir / alias
        kf_meta = extract_keyframes_for_scenes(
            video_path=str(v_path),
            scenes_data=scenes,
            output_dir=str(keyframes_dir),
        )

        elapsed = round(time.time() - t0, 2)
        scene_count = len(scenes)
        kf_count = len(kf_meta)

        total_scenes_found += scene_count
        total_keyframes_extracted += kf_count
        processed_count += 1

        print(
            f"   -> {scene_count:2d} sahne tespit edildi | "
            f"{kf_count:2d} keyframe çıkarıldı | "
            f"Süre: {elapsed:.2f}s"
        )

    total_elapsed = round(time.time() - start_total_time, 2)
    print("\n==================================================")
    print(f"[BAŞARILI] İşlem Tamamlandı! ({processed_count} yeni işlendi, {skipped_count} önceden hazır)")
    print(f"Toplam Geçen Süre: {total_elapsed:.2f} saniye (~{total_elapsed/60:.2f} dakika)")
    print("==================================================")

if __name__ == "__main__":
    process_all_50_videos()
