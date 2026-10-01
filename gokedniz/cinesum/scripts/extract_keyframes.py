import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scene_detection.keyframe_extractor import (
    extract_keyframes_for_scenes,
    save_keyframe_metadata,
)

def run_keyframe_extraction():
    base_dir = Path(__file__).resolve().parent.parent
    video_dir = base_dir / "dataset" / "video"
    outputs_dir = base_dir / "outputs" / "pyscenedetect"
    
    sample_video_name = "video51.mp4"
    video_path = video_dir / sample_video_name
    video_stem = video_path.stem

    # Read default content_t27 scene list JSON
    scene_json_path = outputs_dir / "scene_lists" / f"{video_stem}_content_t27.json"
    if not scene_json_path.exists():
        print(f"[HATA] Sahne listesi JSON bulunamadı: {scene_json_path}")
        return

    with open(scene_json_path, "r", encoding="utf-8") as f:
        scenes_data = json.load(f)

    print(f"==========================================")
    print(f"Keyframe Çıkarma Başlıyor: {sample_video_name}")
    print(f"Toplam Sahne Sayısı: {len(scenes_data)}")
    print(f"==========================================")

    keyframes_output_dir = outputs_dir / "keyframes"
    
    metadata = extract_keyframes_for_scenes(
        video_path=str(video_path),
        scenes_data=scenes_data,
        output_dir=str(keyframes_output_dir),
        long_scene_threshold=10.0,
        sample_step=2,
    )

    metadata_json_path = keyframes_output_dir / video_stem / "keyframes_metadata.json"
    save_keyframe_metadata(metadata, str(metadata_json_path))

    print(f"\n[BAŞARILI] Toplam {len(metadata)} adet keyframe çıkarıldı!")
    print(f"Görsellerin kaydedildiği dizin: {keyframes_output_dir / video_stem}")
    print(f"Metadata JSON dosyası: {metadata_json_path}")

if __name__ == "__main__":
    run_keyframe_extraction()
