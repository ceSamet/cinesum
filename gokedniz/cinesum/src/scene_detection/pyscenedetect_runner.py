import json
import csv
from pathlib import Path
from typing import List, Dict, Any

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector

def detect_scenes_pyscenedetect(
    video_path: str,
    detector_type: str = "content",
    threshold: float = 27.0,
    adaptive_threshold: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Detect scenes in a video file using PySceneDetect.
    
    Args:
        video_path: Path to the video file.
        detector_type: 'content' or 'adaptive'.
        threshold: Threshold for ContentDetector.
        adaptive_threshold: Threshold for AdaptiveDetector.
        
    Returns:
        List of dicts representing scene segments with start/end frames & timestamps.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    
    if detector_type.lower() == "adaptive":
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=adaptive_threshold))
    else:
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    
    scenes_data = []
    for i, scene in enumerate(scene_list, start=1):
        start, end = scene
        scenes_data.append({
            "scene_id": i,
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "start_timecode": start.get_timecode(),
            "end_timecode": end.get_timecode(),
            "start_seconds": round(start.get_seconds(), 3),
            "end_seconds": round(end.get_seconds(), 3),
            "duration_seconds": round(end.get_seconds() - start.get_seconds(), 3)
        })
        
    return scenes_data

def save_scenes_to_json(scenes_data: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenes_data, f, indent=2, ensure_ascii=False)

def save_scenes_to_csv(scenes_data: List[Dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not scenes_data:
        return
    fieldnames = list(scenes_data[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenes_data)
