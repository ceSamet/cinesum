from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedScene:
    index: int
    start_sec: float
    end_sec: float
    keyframe_sec: float
    cut_score: float
    frame_path: Path

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


def _visual_signature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def detect_scene_boundaries(
    video_path: Path,
    min_scene_seconds: float = 1.25,
    max_scene_seconds: float = 30.0,
    analysis_fps: float = 4.0,
) -> tuple[list[tuple[float, float, float]], float]:
    """Detect adaptive shot boundaries and cap unusually long static shots.

    The threshold is computed from the video's own median/MAD change profile,
    so dark dramas and fast action footage do not share a hard coded interval.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0.0
    step = max(1, round(fps / analysis_fps))
    samples: list[tuple[float, float]] = []
    previous = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                signature = _visual_signature(frame)
                if previous is not None:
                    distance = float(cv2.compareHist(previous, signature, cv2.HISTCMP_BHATTACHARYYA))
                    samples.append((index / fps, distance))
                previous = signature
            index += 1
    finally:
        cap.release()

    if duration <= 0:
        duration = samples[-1][0] if samples else 0.0
    if not samples or duration <= 0:
        return [(0.0, duration, 0.0)], duration

    scores = np.asarray([score for _, score in samples], dtype=np.float32)
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    threshold = min(0.72, max(0.20, median + max(0.08, 5.5 * mad)))

    cuts = [0.0]
    last_cut = 0.0
    for timestamp, score in samples:
        if score >= threshold and timestamp - last_cut >= min_scene_seconds:
            cuts.append(timestamp)
            last_cut = timestamp

    cuts.append(duration)
    normalized = [cuts[0]]
    for end in cuts[1:]:
        start = normalized[-1]
        while end - start > max_scene_seconds:
            start += max_scene_seconds
            normalized.append(start)
        if end - normalized[-1] >= min_scene_seconds or end == duration:
            normalized.append(end)
    if len(normalized) > 2 and normalized[-1] - normalized[-2] < min_scene_seconds:
        normalized.pop(-2)

    scenes: list[tuple[float, float, float]] = []
    score_by_time = dict(samples)
    for start, end in zip(normalized, normalized[1:]):
        scenes.append((round(start, 3), round(end, 3), round(score_by_time.get(start, 0.0), 4)))
    return scenes, duration


def extract_scene_keyframes(
    video_path: Path,
    output_dir: Path,
    min_scene_seconds: float = 1.25,
    max_scene_seconds: float = 30.0,
) -> tuple[list[DetectedScene], float]:
    boundaries, duration = detect_scene_boundaries(
        video_path, min_scene_seconds=min_scene_seconds, max_scene_seconds=max_scene_seconds
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    results: list[DetectedScene] = []
    try:
        for index, (start, end, score) in enumerate(boundaries, 1):
            # Slightly after the midpoint avoids fades at either boundary.
            keyframe = min(end - 0.05, start + max(0.05, (end - start) * 0.55))
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, keyframe) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            path = output_dir / f"scene_{index:04d}_{start:08.2f}-{end:08.2f}.jpg"
            cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            results.append(DetectedScene(index, start, end, round(keyframe, 3), score, path))
    finally:
        cap.release()
    return results, duration
