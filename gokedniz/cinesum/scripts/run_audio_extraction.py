import sys
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.audio.audio_extractor import extract_audio_from_video, compute_scene_audio_energy
from src.audio.whisper_transcriber import transcribe_audio_with_whisper, map_transcript_to_scenes

def run_audio_extraction_all_videos():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"
    scene_lists_dir = base_dir / "outputs" / "pyscenedetect" / "scene_lists"
    audio_outputs_dir = base_dir / "outputs" / "features" / "audio"
    raw_audio_dir = base_dir / "outputs" / "audio"

    audio_outputs_dir.mkdir(parents=True, exist_ok=True)
    raw_audio_dir.mkdir(parents=True, exist_ok=True)

    videos = [f"video{i}" for i in range(1, 51)]

    print("==================================================")
    print("Audio Branch İşleniyor (FFmpeg + Librosa RMS + Whisper)")
    print("==================================================")

    start_total_time = time.time()
    success_count = 0
    skipped_count = 0

    for idx, alias in enumerate(videos, 1):
        v_path = video_dir / f"{alias}.mp4"
        scene_json = scene_lists_dir / f"{alias}_scenes.json"

        if not v_path.exists() or not scene_json.exists():
            print(f"[{idx:02d}/50] [HENÜZ YOK] Video veya sahne listesi bulunamadı: {alias}")
            continue

        out_json = audio_outputs_dir / f"{alias}_audio_features.json"
        if out_json.exists():
            skipped_count += 1
            print(f"[{idx:02d}/50] [ATLANIYOR] Ses özellikleri zaten çıkarılmış: {alias}")
            continue

        print(f"\n[{idx:02d}/50] Ses Analizi İşliyor: {alias}")
        t0 = time.time()

        with open(scene_json, "r", encoding="utf-8") as f:
            scenes = json.load(f)

        # 1. Extract WAV audio via FFmpeg
        wav_path = raw_audio_dir / f"{alias}.wav"
        extract_audio_from_video(str(v_path), str(wav_path))

        # 2. Compute Librosa RMS Audio Energy per scene
        audio_energy_list = compute_scene_audio_energy(str(wav_path), scenes)

        # 3. Transcribe audio with Whisper (tiny model for high speed)
        whisper_res = transcribe_audio_with_whisper(str(wav_path), model_size="tiny")
        scene_transcripts = map_transcript_to_scenes(whisper_res, scenes)

        # 4. Merge audio energy & transcript features per scene
        merged_audio_features = []
        for ae, st in zip(audio_energy_list, scene_transcripts):
            merged_audio_features.append({
                "scene_id": ae["scene_id"],
                "start_seconds": ae["start_seconds"],
                "end_seconds": ae["end_seconds"],
                "duration_seconds": ae["duration_seconds"],
                "mean_rms_energy": ae["mean_rms_energy"],
                "max_rms_energy": ae["max_rms_energy"],
                "normalized_audio_energy": ae["normalized_audio_energy"],
                "speech_ratio": st["speech_ratio"],
                "word_count": st["word_count"],
                "transcript_density": st["transcript_density"],
                "transcript_text": st["transcript_text"],
            })

        # Save audio features JSON
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(merged_audio_features, f, indent=2, ensure_ascii=False)

        elapsed = round(time.time() - t0, 2)
        success_count += 1
        print(
            f"   -> {len(merged_audio_features)} sahne için ses ve transkript özellikleri çıkarıldı | "
            f"Süre: {elapsed:.2f}s"
        )

    total_elapsed = round(time.time() - start_total_time, 2)
    print("\n==================================================")
    print(f"[BAŞARILI] İşlem Tamamlandı! ({success_count} yeni işlendi, {skipped_count} önceden hazır)")
    print(f"Toplam Geçen Süre: {total_elapsed:.2f} saniye (~{total_elapsed/60:.2f} dakika)")
    print("==================================================")

if __name__ == "__main__":
    run_audio_extraction_all_videos()
