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

def process_all_videos():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"
    outputs_dir = base_dir / "outputs" / "pyscenedetect"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mapping_file = base_dir / "dataset" / "video_mapping.json"
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        mapping = [{"alias": f"video{i}", "new_filename": f"video{i}.mp4"} for i in range(1, 51)]

    mapping = mapping[:5]

    print(f"==================================================")
    print(f"TVMum50 İlk 5 Video İşleniyor (video1 - video5)")
    print(f"==================================================")

    summary_rows = []
    start_total_time = time.time()

    for idx, item in enumerate(mapping, start=1):
        alias = item["alias"]
        v_filename = item["new_filename"]
        v_path = video_dir / v_filename

        if not v_path.exists():
            print(f"[{idx}/{len(mapping)}] [UYARI] Video bulunamadı: {v_path}")
            continue

        print(f"[{idx}/{len(mapping)}] İşleniyor: {alias} ({v_filename})...")
        t0 = time.time()

        # 1. Detect Scenes (PySceneDetect ContentDetector threshold=27.0)
        scenes = detect_scenes_pyscenedetect(
            video_path=str(v_path),
            detector_type="content",
            threshold=27.0,
        )

        # Save scene lists
        json_out = outputs_dir / "scene_lists" / f"{alias}_content_t27.json"
        csv_out = outputs_dir / "scene_lists" / f"{alias}_content_t27.csv"
        save_scenes_to_json(scenes, str(json_out))
        save_scenes_to_csv(scenes, str(csv_out))

        scene_count = len(scenes)

        # 2. Extract Keyframes
        kf_dir = outputs_dir / "keyframes"
        keyframes_meta = extract_keyframes_for_scenes(
            video_path=str(v_path),
            scenes_data=scenes,
            output_dir=str(kf_dir),
            long_scene_threshold=10.0,
            sample_step=2,
        )

        kf_meta_json = kf_dir / alias / "keyframes_metadata.json"
        save_keyframe_metadata(keyframes_meta, str(kf_meta_json))

        kf_count = len(keyframes_meta)
        elapsed = round(time.time() - t0, 2)

        print(
            f"   -> {scene_count:2d} sahne bulundu | {kf_count:2d} keyframe çıkarıldı | "
            f"Süre: {elapsed}s"
        )

        summary_rows.append({
            "video_alias": alias,
            "original_video_id": item.get("video_id", ""),
            "category": item.get("category", ""),
            "title": item.get("title", ""),
            "detected_scenes": scene_count,
            "extracted_keyframes": kf_count,
            "processing_time_sec": elapsed,
            "scene_json": str(json_out.relative_to(base_dir)),
            "keyframe_folder": str((kf_dir / alias).relative_to(base_dir)),
        })

    total_elapsed = round(time.time() - start_total_time, 2)
    print(f"\n==================================================")
    print(f"[BAŞARILI] Tüm videolar tamamlandı! Toplam Süre: {total_elapsed}s")
    print(f"==================================================")

    # Save master summary CSV
    master_csv = reports_dir / "all_videos_scene_and_keyframe_summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(master_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Master özet raporu kaydedildi: {master_csv}")

if __name__ == "__main__":
    process_all_videos()
