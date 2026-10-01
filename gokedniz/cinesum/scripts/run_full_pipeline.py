import sys
import json
import time
from pathlib import Path

# Add src directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.core.analysis_pipeline import analyze_video_features
from src.scoring.scoring_engine import compute_scene_scores_for_video
from src.summary.summary_exporter import export_category_summary

def ensure_features_exist_for_video(video_alias: str, profile: str = "balanced"):
    """
    Check if CLIP visual vectors and scene features exist for video_alias.
    If missing, run PySceneDetect, Keyframes, CLIP, and Whisper automatically.
    """
    v_mp4 = BASE_DIR / "dataset" / "video" / f"{video_alias}.mp4"
    print(f"\n[ANALİZ] {video_alias} için sürümlü feature pipeline çalıştırılıyor ({profile})...")

    def report_progress(percent, description, detail, _step):
        print(f"  [{percent:3d}%] {description} — {detail}")

    result = analyze_video_features(
        video_alias=video_alias,
        video_path=v_mp4,
        base_dir=BASE_DIR,
        profile_name=profile,
        progress_callback=report_progress,
    )
    hits = sum(result["cache_hits"].values())
    total = len(result["cache_hits"])
    print(
        f"  [TAMAMLANDI] {result['scenes_found']} shot / "
        f"{result['story_scenes_found']} StoryScene hazır; "
        f"{hits}/{total} aşama cache'ten kullanıldı.\n"
    )
    return result

def main():
    video_dir = BASE_DIR / "dataset" / "video"
    scores_out_dir = BASE_DIR / "outputs" / "scores"
    summaries_out_dir = BASE_DIR / "outputs" / "summaries"

    scores_out_dir.mkdir(parents=True, exist_ok=True)
    summaries_out_dir.mkdir(parents=True, exist_ok=True)

    print("==========================================================================")
    print("🎬 CineSum AI — İnteraktif Tüm Özet Türlerini Üretme Scripti")
    print("==========================================================================")

    if not video_dir.exists():
        print(f"[HATA] Video dizini bulunamadı: {video_dir}")
        return

    available_videos = sorted(
        [f.stem for f in video_dir.glob("*.mp4")],
        key=lambda x: int(x.replace("video", "")) if x.replace("video", "").isdigit() else 999
    )

    if not available_videos:
        print("[HATA] dataset/video klasöründe hiçbir MP4 video bulunamadı!")
        return

    print("\nMevcut Videolar:")
    for idx, v in enumerate(available_videos[:15], 1):
        print(f"  [{idx:2d}] {v}")
    if len(available_videos) > 15:
        print(f"  ... ve {len(available_videos) - 15} video daha ({available_videos[0]} .. {available_videos[-1]})")

    print("--------------------------------------------------------------------------")

    try:
        user_input = input("\n👉 Tüm özetlerini (Aksiyon, Diyalog, Önem) çıkarmak istediğin video numarası/adı (Örn: 1, 55, video55 veya HEPSİ için 'all') [Varsayılan: 1]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[ÇIKIŞ] İşlem iptal edildi.")
        return

    # Clean input if user types video55.mp4
    user_input = user_input.replace(".mp4", "").replace(".MOV", "").replace(".avi", "").strip()

    if not user_input or user_input == "1":
        selected_videos = ["video1"]
    elif user_input.lower() == "all":
        selected_videos = available_videos
    elif user_input.isdigit():
        v_idx = int(user_input) - 1
        if 0 <= v_idx < len(available_videos):
            selected_videos = [available_videos[v_idx]]
        else:
            selected_videos = [f"video{user_input}"]
    else:
        selected_videos = [user_input.lower().strip()]

    categories = ["action", "dialogue", "importance"]
    start_total = time.time()

    print(f"\nİşlenecek Video(lar): {', '.join(selected_videos)}")

    for alias in selected_videos:
        v_mp4 = video_dir / f"{alias}.mp4"
        if not v_mp4.exists():
            print(f"\n[UYARI] Video dosyası bulunamadı: {v_mp4}")
            continue

        # Automatic Feature Extraction Check
        try:
            ensure_features_exist_for_video(alias)
        except Exception as fe_err:
            print(f"[HATA] {alias} için analiz matrisleri çıkarılamadı: {fe_err}")
            continue

        print(f"\n==================================================")
        print(f"---> [{alias.upper()}] Skorlama ve Özet Üretimi Başladı...")
        print(f"==================================================")
        t0 = time.time()

        try:
            scored_scenes = compute_scene_scores_for_video(alias, BASE_DIR)
        except Exception as e:
            print(f"[HATA] {alias} skorlamasında hata: {e}")
            continue

        # Save scores JSON
        score_json = scores_out_dir / f"{alias}_scene_scores.json"
        with open(score_json, "w", encoding="utf-8") as f:
            json.dump(scored_scenes, f, indent=2, ensure_ascii=False)

        print(f"  [BAŞARILI] {len(scored_scenes)} sahne puanlandı! Dosya: {score_json.name} ({round(time.time()-t0, 2)}s)")

        # Export Category Summaries (Action, Dialogue, Importance MP4s)
        for cat in categories:
            print(f"  [EXPORT] ⚡ {cat.upper()} Özeti MP4 Kesiliyor...")
            out_mp4 = summaries_out_dir / f"{alias}_{cat}_summary.mp4"

            try:
                export_category_summary(
                    video_path=str(v_mp4),
                    scored_scenes=scored_scenes,
                    category=cat,
                    target_duration_sec=35.0,
                    output_mp4_path=str(out_mp4),
                )
                print(f"    └─ Tamamlandı: {out_mp4.name}")
            except Exception as e:
                print(f"    └─ [HATA] {cat} özeti oluşturulamadı: {e}")

    total_elapsed = round(time.time() - start_total, 2)
    print("\n==================================================")
    print(f"🎉 [TEBRİKLER] Seçilen Tüm Özetler Başarıyla Üretildi!")
    print(f"Toplam Süre: {total_elapsed:.2f} saniye (~{total_elapsed/60:.2f} dakika)")
    print(f"Özet Videolar Klasörü: {summaries_out_dir}")
    print("==================================================")

if __name__ == "__main__":
    main()
