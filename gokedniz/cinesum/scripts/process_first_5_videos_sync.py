import sys
import json
import csv
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scene_detection.pyscenedetect_runner import (
    detect_scenes_pyscenedetect,
    save_scenes_to_json,
    save_scenes_to_csv,
)
from src.scene_detection.keyframe_extractor import (
    extract_keyframes_for_scenes,
    save_keyframe_metadata,
)
from scripts.calculate_evaluation_metrics import calculate_shot_boundary_metrics

def main():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"
    outputs_dir = base_dir / "outputs" / "pyscenedetect"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    first_5_videos = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4", "video5.mp4"]
    configs = [
        {"detector": "content", "threshold": 27.0, "label": "content_t27"},
        {"detector": "content", "threshold": 15.0, "label": "content_t15"},
        {"detector": "content", "threshold": 35.0, "label": "content_t35"},
        {"detector": "adaptive", "adaptive_threshold": 3.0, "label": "adaptive_t3.0"},
    ]

    print("==================================================")
    print("İLK 5 VİDEO İÇİN SAHNE LİSTELERİ VE KEYFRAME'LER ÇIKARILIYOR")
    print("==================================================")

    for v_filename in first_5_videos:
        v_path = video_dir / v_filename
        v_stem = v_path.stem
        if not v_path.exists():
            print(f"[UYARI] Video bulunamadı: {v_path}")
            continue

        print(f"\n---> Video İşleniyor: {v_stem} ({v_filename})")

        # 1. Generate scene lists for all 4 configs
        default_scenes = []
        for cfg in configs:
            label = cfg["label"]
            det_type = cfg["detector"]
            thresh = cfg.get("threshold", 27.0)
            adap_thresh = cfg.get("adaptive_threshold", 3.0)

            scenes = detect_scenes_pyscenedetect(
                video_path=str(v_path),
                detector_type=det_type,
                threshold=thresh,
                adaptive_threshold=adap_thresh,
            )

            json_out = outputs_dir / "scene_lists" / f"{v_stem}_{label}.json"
            csv_out = outputs_dir / "scene_lists" / f"{v_stem}_{label}.csv"
            save_scenes_to_json(scenes, str(json_out))
            save_scenes_to_csv(scenes, str(csv_out))

            if label == "content_t27":
                default_scenes = scenes

            print(f"    [{label:15s}] -> Bulunan Sahne Sayısı: {len(scenes)}")

        # 2. Extract keyframes for the default (content_t27) scenes
        kf_dir = outputs_dir / "keyframes"
        keyframes_meta = extract_keyframes_for_scenes(
            video_path=str(v_path),
            scenes_data=default_scenes,
            output_dir=str(kf_dir),
            long_scene_threshold=10.0,
            sample_step=2,
        )
        kf_meta_json = kf_dir / v_stem / "keyframes_metadata.json"
        save_keyframe_metadata(keyframes_meta, str(kf_meta_json))
        print(f"    [Keyframe Extraction] -> {len(keyframes_meta)} adet en net keyframe çıkarıldı.")

    # 3. Calculate evaluation metrics on these 5 videos
    print("\n==================================================")
    print("İLK 5 VİDEO İÇİN AKADEMİK SKORLAR HESAPLANIYOR")
    print("==================================================")
    calculate_shot_boundary_metrics()

if __name__ == "__main__":
    main()
