import os
import sys
import csv
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scene_detection.pyscenedetect_runner import (
    detect_scenes_pyscenedetect,
    save_scenes_to_json,
    save_scenes_to_csv,
)

def run_experiment():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"
    outputs_dir = base_dir / "outputs" / "pyscenedetect"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Pick representative sample videos (video51 was -esJrBWj2d8, video14 was 0tmA_C6XwfM, video20 was 37rzWOQsNIw)
    sample_videos = [
        "video51.mp4",  # Sample 1 (Will A Cat Eat Dog Food?)
        "video14.mp4",  # Sample 2 (Nail clipper Gloria Pets)
        "video20.mp4",  # Sample 3 (Saigon Sandwich)
    ]

    test_configs = [
        {"detector": "content", "threshold": 15.0, "adaptive_threshold": None, "label": "content_t15"},
        {"detector": "content", "threshold": 27.0, "adaptive_threshold": None, "label": "content_t27"},
        {"detector": "content", "threshold": 35.0, "adaptive_threshold": None, "label": "content_t35"},
        {"detector": "adaptive", "threshold": None, "adaptive_threshold": 3.0, "label": "adaptive_t3.0"},
    ]

    summary_records = []

    for video_name in sample_videos:
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"[UYARI] Video bulunamadı: {video_path}")
            continue

        print(f"\n==========================================")
        print(f"Video Analiz Ediliyor: {video_name}")
        print(f"==========================================")

        for cfg in test_configs:
            label = cfg["label"]
            detector_type = cfg["detector"]
            threshold = cfg["threshold"] or 27.0
            adaptive_threshold = cfg["adaptive_threshold"] or 3.0

            scenes = detect_scenes_pyscenedetect(
                video_path=str(video_path),
                detector_type=detector_type,
                threshold=threshold,
                adaptive_threshold=adaptive_threshold,
            )

            video_stem = video_path.stem
            json_out = outputs_dir / "scene_lists" / f"{video_stem}_{label}.json"
            csv_out = outputs_dir / "scene_lists" / f"{video_stem}_{label}.csv"

            save_scenes_to_json(scenes, str(json_out))
            save_scenes_to_csv(scenes, str(csv_out))

            scene_count = len(scenes)
            avg_duration = (
                sum(s["duration_seconds"] for s in scenes) / scene_count
                if scene_count > 0
                else 0
            )

            print(
                f"[{label}] -> Bulunan Sahne Sayısı: {scene_count:2d} | "
                f"Ort. Sahne Süresi: {avg_duration:.2f}s"
            )

            summary_records.append({
                "video_name": video_name,
                "video_stem": video_stem,
                "config_label": label,
                "detector_type": detector_type,
                "threshold": threshold if detector_type == "content" else adaptive_threshold,
                "detected_scene_count": scene_count,
                "avg_scene_duration_sec": round(avg_duration, 2),
                "json_output": str(json_out.relative_to(base_dir)),
                "csv_output": str(csv_out.relative_to(base_dir)),
            })

    # Save summary report using standard csv module
    summary_csv_path = reports_dir / "pyscenedetect_threshold_comparison.csv"
    if summary_records:
        fieldnames = list(summary_records[0].keys())
        with open(summary_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_records)

    print(f"\n[BAŞARILI] Test tamamlandı! Özet rapor kaydedildi: {summary_csv_path}")

if __name__ == "__main__":
    run_experiment()
