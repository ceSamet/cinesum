import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

def calculate_blur_score(frame: np.ndarray) -> float:
    """
    Calculate image sharpness/blur score using the variance of Laplacian.
    Higher values indicate a sharper, clearer image.
    """
    if frame is None or frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def extract_keyframes_for_scenes(
    video_path: str,
    scenes_data: List[Dict[str, Any]],
    output_dir: str,
    long_scene_threshold: float = 10.0,
    sample_step: int = 2,
) -> List[Dict[str, Any]]:
    """
    Extract sharpest keyframe(s) for each scene segment.
    - Scenes < long_scene_threshold: 1 keyframe (sharpest frame in middle window).
    - Scenes >= long_scene_threshold: 3 keyframes (sharpest frame in start, mid, end windows).
    """
    video_path_obj = Path(video_path)
    video_stem = video_path_obj.stem
    output_path = Path(output_dir) / video_stem
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video okunamadı: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    extracted_keyframes = []

    for scene in scenes_data:
        scene_id = scene["scene_id"]
        start_frame = scene["start_frame"]
        end_frame = scene["end_frame"]
        duration = scene["duration_seconds"]
        total_frames = end_frame - start_frame

        if total_frames <= 0:
            continue

        # Determine windows to search for sharpest frames
        if duration < long_scene_threshold:
            # 1 window in middle (30% to 70%)
            windows = [
                (int(start_frame + total_frames * 0.30), int(start_frame + total_frames * 0.70))
            ]
        else:
            # 3 windows: [10%-35%], [35%-65%], [65%-90%]
            windows = [
                (int(start_frame + total_frames * 0.10), int(start_frame + total_frames * 0.35)),
                (int(start_frame + total_frames * 0.35), int(start_frame + total_frames * 0.65)),
                (int(start_frame + total_frames * 0.65), int(start_frame + total_frames * 0.90)),
            ]

        for kf_idx, (w_start, w_end) in enumerate(windows, start=1):
            if w_end <= w_start:
                w_start = start_frame
                w_end = end_frame

            best_frame_num = w_start
            best_score = -1.0
            best_frame = None

            # Seek once per window. Repeated random seeks for every sample can
            # decode the same GOP thousands of times on a long movie.
            cap.set(cv2.CAP_PROP_POS_FRAMES, w_start)
            for fn in range(w_start, w_end + 1):
                if not cap.grab():
                    break
                if (fn - w_start) % sample_step:
                    continue
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    continue

                score = calculate_blur_score(frame)
                if score > best_score:
                    best_score = score
                    best_frame_num = fn
                    best_frame = frame

            # If no frame matched, fallback to window start
            if best_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, w_start)
                ret, best_frame = cap.read()
                best_frame_num = w_start
                best_score = calculate_blur_score(best_frame) if ret else 0.0

            if best_frame is not None:
                kf_filename = f"{video_stem}_scene_{scene_id:03d}_kf{kf_idx}.jpg"
                kf_file_path = output_path / kf_filename
                is_success, im_buf_arr = cv2.imencode(".jpg", best_frame)
                if is_success:
                    im_buf_arr.tofile(str(kf_file_path))

                timestamp_sec = round(best_frame_num / fps, 3)

                extracted_keyframes.append({
                    "video_stem": video_stem,
                    "scene_id": scene_id,
                    "kf_index": kf_idx,
                    "frame_number": best_frame_num,
                    "timestamp_seconds": timestamp_sec,
                    "blur_score": round(best_score, 2),
                    "file_name": kf_filename,
                    "file_path": str(kf_file_path),
                })

    cap.release()
    return extracted_keyframes

def save_keyframe_metadata(metadata: List[Dict[str, Any]], output_json_path: str) -> None:
    path = Path(output_json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
