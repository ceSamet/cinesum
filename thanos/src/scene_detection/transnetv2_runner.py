import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

def detect_scenes_transnetv2(
    video_path: str,
    threshold: float = 0.5,
    min_scene_len_frames: int = 15,
) -> List[Dict[str, Any]]:
    """
    Detect scenes in a video using TransNetV2 PyTorch model with exact FPS calculation.
    """
    try:
        from transnetv2_pytorch import TransNetV2
    except ImportError:
        raise ImportError("transnetv2-pytorch paketi bulunamadı.")

    # Get exact video FPS using OpenCV
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    model = TransNetV2()
    video_frames, single_frame_predictions, _ = model.predict_video(video_path)
    
    total_frames = len(single_frame_predictions)
    if total_frames == 0:
        return []

    raw_cuts = np.where(single_frame_predictions > threshold)[0]

    merged_cuts = []
    for c in raw_cuts:
        if not merged_cuts or (c - merged_cuts[-1]) >= min_scene_len_frames:
            merged_cuts.append(int(c))

    scene_boundaries = [0] + merged_cuts
    if not scene_boundaries or scene_boundaries[-1] < total_frames:
        scene_boundaries.append(total_frames)

    scenes_data = []
    for i in range(len(scene_boundaries) - 1):
        s_frame = scene_boundaries[i]
        e_frame = scene_boundaries[i+1]
        
        s_sec = round(s_frame / fps, 3)
        e_sec = round(e_frame / fps, 3)
        dur_sec = round(e_sec - s_sec, 3)

        s_tc = time.strftime('%H:%M:%S', time.gmtime(s_sec)) + f".{int((s_sec % 1)*1000):03d}"
        e_tc = time.strftime('%H:%M:%S', time.gmtime(e_sec)) + f".{int((e_sec % 1)*1000):03d}"

        scenes_data.append({
            "scene_id": i + 1,
            "start_frame": s_frame,
            "end_frame": e_frame,
            "start_timecode": s_tc,
            "end_timecode": e_tc,
            "start_seconds": s_sec,
            "end_seconds": e_sec,
            "duration_seconds": dur_sec,
        })

    return scenes_data
