import os
import sys
import time
from pathlib import Path

# Disable HuggingFace symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.summary.custom_query_summary import generate_custom_query_summary

def main():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"

    print("\n==========================================================================")
    print("🎬 CineSum AI — İnteraktif Özel Prompt Tabanlı Video Özetleyici")
    print("==========================================================================")

    # 1. List available videos
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

    # 2. Get Video Choice from User
    try:
        user_v_input = input("\n👉 Özetlemek istediğin videonun numarasını veya adını gir (Örn: 1 veya video1) [Varsayılan: 1]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[ÇIKIŞ] İşlem iptal edildi.")
        return

    if not user_v_input:
        selected_alias = "video1"
    elif user_v_input.isdigit():
        v_idx = int(user_v_input) - 1
        if 0 <= v_idx < len(available_videos):
            selected_alias = available_videos[v_idx]
        else:
            selected_alias = f"video{user_v_input}"
    else:
        selected_alias = user_v_input.lower().strip()

    if selected_alias not in available_videos:
        print(f"[UYARI] '{selected_alias}' listede bulunamadı, 'video1' seçiliyor.")
        selected_alias = "video1"

    # 3. Get Custom Prompt from User
    try:
        user_prompt = input("👉 Aratmak istediğin konuyu / cümleyi gir (Örn: 'a car chase', 'a fight', 'people conversation') [Varsayılan: 'a car chase']: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n[ÇIKIŞ] İşlem iptal edildi.")
        return

    if not user_prompt:
        user_prompt = "a car chase"

    # 4. Get Target Duration from User (supports 0 or 'hepsi' or 'all' for ALL matching scenes)
    try:
        print("\nSüre Seçenekleri:")
        print("  - Sabit Süre: 30, 45, 60 (saniye)")
        print("  - TÜM EŞLEŞEN SAHNELER: 0 veya 'hepsi' veya 'all'")
        user_dur_input = input("👉 Hedef özet süresi [Varsayılan: 30]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n[ÇIKIŞ] İşlem iptal edildi.")
        return

    if user_dur_input in ["0", "hepsi", "all", "tum", "tumu"]:
        target_dur = 0.0
        mode_label = "TÜM EŞLEŞEN SAHNELER (Süre Sınırı Yok)"
    elif not user_dur_input:
        target_dur = 30.0
        mode_label = "30.0 saniye"
    else:
        try:
            target_dur = float(user_dur_input)
            mode_label = f"{target_dur} saniye"
        except ValueError:
            target_dur = 30.0
            mode_label = "30.0 saniye"

    print("\n--------------------------------------------------------------------------")
    print(f"⚡ İŞLEM BAŞLATILIYOR:")
    print(f"   Video: {selected_alias.upper()}")
    print(f"   Sorgu: '{user_prompt}'")
    print(f"   Mod/Süre: {mode_label}")
    print("--------------------------------------------------------------------------\n")

    t0 = time.time()
    try:
        result = generate_custom_query_summary(
            video_alias=selected_alias,
            text_prompt=user_prompt,
            base_dir=base_dir,
            target_duration_sec=target_dur,
        )
        elapsed = round(time.time() - t0, 2)

        print("\n==========================================================================")
        print(f"🎉 ÖZET VİDEO BAŞARIYLA ÜRETİLDİ! (İşlem Süresi: {elapsed} saniye)")
        print("==========================================================================")
        print(f"📁 Çıktı Dosyası: {result['output_mp4_path']}")
        print(f"⏱️ Toplam Özet Süresi: {result['total_duration_seconds']} saniye ({len(result['selected_scenes'])} sahne seçildi)")
        print("\nSeçilen Sahnelerin Detayları:")
        for rank, sc in enumerate(result['selected_scenes'], 1):
            print(
                f"   #{rank} -> Sahne {sc['scene_id']:2d} | "
                f"Zaman: {sc['start_timecode']} -> {sc['end_timecode']} ({sc['duration_seconds']:.2f}s) | "
                f"CLIP Skor: {sc['query_max_similarity']:.4f}"
            )
        print("==========================================================================\n")

    except Exception as e:
        print(f"\n[HATA] Özet oluşturulurken bir sorun oluştu: {e}")

if __name__ == "__main__":
    main()
