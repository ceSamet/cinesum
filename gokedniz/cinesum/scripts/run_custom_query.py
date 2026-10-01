import sys
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.summary.custom_query_summary import generate_custom_query_summary

def run_test_custom_queries():
    base_dir = Path(__file__).resolve().parent.parent

    # Test queries on video1 and video2
    test_cases = [
        {"video": "video1", "prompt": "a car or vehicle drive on the road", "target_dur": 30.0},
        {"video": "video2", "prompt": "two people speaking in a room", "target_dur": 35.0},
    ]

    print("==================================================")
    print("CineSum AI — Özel Prompt/Cümle Tabanlı Video Özetleyici")
    print("==================================================")

    for case in test_cases:
        v_alias = case["video"]
        prompt = case["prompt"]
        dur = case["target_dur"]

        print(f"\n---> [{v_alias.upper()}] Özel Sorgu İle Özetleniyor: '{prompt}'")
        t0 = time.time()

        try:
            res = generate_custom_query_summary(
                video_alias=v_alias,
                text_prompt=prompt,
                base_dir=base_dir,
                target_duration_sec=dur,
            )
            elapsed = round(time.time() - t0, 2)

            print(f"    [BAŞARILI] {len(res['selected_scenes'])} sahne seçildi! Toplam Süre: {res['total_duration_seconds']}s")
            print(f"    [ÇIKTI VIDEO] {res['output_mp4_path']} (Süre: {elapsed}s)")

            print("    Seçilen Sahnelerin Benzerlik Skorları:")
            for sc in res["selected_scenes"]:
                print(
                    f"      - Sahne {sc['scene_id']:2d} ({sc['start_timecode']} -> {sc['end_timecode']}) | "
                    f"CLIP Skor: {sc['query_max_similarity']:.4f}"
                )

        except Exception as e:
            print(f"    [HATA] Sorgu özeti oluşturulamadı: {e}")

    print("\n==================================================")
    print("[TEBRİKLER] Özel Prompt Tabanlı Özetleme Testi Tamamlandı!")
    print("==================================================")

if __name__ == "__main__":
    run_test_custom_queries()
