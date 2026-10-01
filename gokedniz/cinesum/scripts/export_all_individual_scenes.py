import sys
import json
import time
import subprocess
from pathlib import Path

def export_all_scenes_as_individual_mp4s(video_alias: str = "video54"):
    base_dir = Path(__file__).resolve().parent.parent
    video_path = base_dir / "dataset" / "video" / f"{video_alias}.mp4"
    scenes_json = base_dir / "outputs" / "pyscenedetect" / "scene_lists" / f"{video_alias}_scenes.json"
    output_dir = base_dir / "outputs" / "scene_clips" / video_alias

    if not video_path.exists():
        print(f"[HATA] Video bulunamadı: {video_path}")
        return
    if not scenes_json.exists():
        print(f"[HATA] Sahne listesi bulunamadı: {scenes_json}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(scenes_json, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    print("==================================================")
    print(f"[{video_alias.upper()}] TÜM SAHNELER TEK TEK MP4 OLARAK ÇIKARILIYOR")
    print(f"Toplam Sahne Sayısı: {len(scenes)}")
    print(f"Çıktı Klasörü: {output_dir}")
    print("==================================================")

    t0 = time.time()
    success_count = 0

    for sc in scenes:
        sc_id = sc["scene_id"]
        s_sec = sc["start_seconds"]
        dur = sc["duration_seconds"]
        out_mp4 = output_dir / f"{video_alias}_scene_{sc_id:03d}.mp4"

        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{s_sec:.3f}",
            "-i", str(video_path),
            "-t", f"{dur:.3f}",
            "-c:v", "libx264",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            str(out_mp4)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            success_count += 1
            print(
                f"  -> [{sc_id:03d}/{len(scenes)}] {out_mp4.name} | "
                f"Zaman: {sc['start_timecode']} -> {sc['end_timecode']} ({dur:.2f}s)"
            )
        except subprocess.CalledProcessError as e:
            print(f"  [HATA] Sahne {sc_id} kesilemedi: {e}")

    total_elapsed = round(time.time() - t0, 2)
    print("\n==================================================")
    print(f"[BAŞARILI] {success_count} adet sahne MP4 dosyası kaydedildi!")
    print(f"Klasör Adresi: {output_dir}")
    print(f"Toplam Süre: {total_elapsed} saniye (~{total_elapsed/60:.2f} dakika)")
    print("==================================================")

if __name__ == "__main__":
    export_all_scenes_as_individual_mp4s("video54")
