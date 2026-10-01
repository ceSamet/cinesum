import sys
import json
import csv
import time
from pathlib import Path
import cv2
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scene_detection.pyscenedetect_runner import save_scenes_to_json, save_scenes_to_csv

def run_transnetv2_experiment():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"
    outputs_dir = base_dir / "outputs" / "transnetv2"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    try:
        from transnetv2_pytorch import TransNetV2
    except ImportError:
        print("[HATA] transnetv2-pytorch bulunamadı!")
        return

    first_5_videos = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4", "video5.mp4"]
    thresholds = [0.4, 0.5, 0.6]

    print("==================================================")
    print("TransNetV2 Hassas FPS Hesabıyla Test Ediliyor (İlk 5 Video)")
    print("==================================================")

    model = TransNetV2()
    transnet_results = []

    for v_filename in first_5_videos:
        v_path = video_dir / v_filename
        v_stem = v_path.stem
        if not v_path.exists():
            print(f"[UYARI] Video bulunamadı: {v_path}")
            continue

        # Read exact FPS using OpenCV
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        print(f"\n---> TransNetV2 Analizi: {v_stem} (FPS: {fps:.2f})")
        t0 = time.time()

        video_frames, single_frame_predictions, _ = model.predict_video(str(v_path))
        total_frames = len(single_frame_predictions)

        for thresh in thresholds:
            raw_cuts = np.where(single_frame_predictions > thresh)[0]

            merged_cuts = []
            for c in raw_cuts:
                if not merged_cuts or (c - merged_cuts[-1]) >= 15:
                    merged_cuts.append(int(c))

            scene_boundaries = [0] + merged_cuts
            if not scene_boundaries or scene_boundaries[-1] < total_frames:
                scene_boundaries.append(total_frames)

            scenes = []
            for i in range(len(scene_boundaries) - 1):
                s_frame = scene_boundaries[i]
                e_frame = scene_boundaries[i+1]
                s_sec = round(s_frame / fps, 3)
                e_sec = round(e_frame / fps, 3)
                dur_sec = round(e_sec - s_sec, 3)

                scenes.append({
                    "scene_id": i + 1,
                    "start_frame": s_frame,
                    "end_frame": e_frame,
                    "start_timecode": f"{int(s_sec//3600):02d}:{int((s_sec%3600)//60):02d}:{int(s_sec%60):02d}.{int((s_sec%1)*1000):03d}",
                    "end_timecode": f"{int(e_sec//3600):02d}:{int((e_sec%3600)//60):02d}:{int(e_sec%60):02d}.{int((e_sec%1)*1000):03d}",
                    "start_seconds": s_sec,
                    "end_seconds": e_sec,
                    "duration_seconds": dur_sec,
                })

            label = f"transnetv2_t{int(thresh*100)}"
            json_out = outputs_dir / "scene_lists" / f"{v_stem}_{label}.json"
            csv_out = outputs_dir / "scene_lists" / f"{v_stem}_{label}.csv"

            save_scenes_to_json(scenes, str(json_out))
            save_scenes_to_csv(scenes, str(csv_out))

            scene_count = len(scenes)
            avg_dur = (
                sum(s["duration_seconds"] for s in scenes) / scene_count
                if scene_count > 0
                else 0
            )

            print(
                f"    [{label:15s}] -> Sahne Sayısı: {scene_count:2d} | "
                f"Ort. Sahne Süresi: {avg_dur:.2f}s"
            )

            transnet_results.append({
                "video_alias": v_stem,
                "config": label,
                "threshold": thresh,
                "scene_count": scene_count,
                "avg_scene_duration_sec": round(avg_dur, 2),
                "processing_time_sec": round(time.time() - t0, 2),
            })

    out_csv = reports_dir / "transnetv2_vs_pyscenedetect.csv"
    if transnet_results:
        fieldnames = list(transnet_results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(transnet_results)

    print(f"\n[BAŞARILI] TransNetV2 testi tamamlandı!")

if __name__ == "__main__":
    run_transnetv2_experiment()
