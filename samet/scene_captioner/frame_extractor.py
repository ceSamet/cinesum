from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedFrame:
    index: int
    timestamp_sec: float
    path: Path


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    interval_seconds: float = 2.0,
    max_frames: int | None = None,
    image_quality: int = 92,
) -> list[ExtractedFrame]:
    """Extract JPEG frames from a video at a fixed time interval."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    video_path = Path(video_path)
    frames_dir = Path(frames_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for frame extraction. Install with: "
            "pip install -r requirements-light.txt"
        ) from exc

    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 25.0

    extracted: list[ExtractedFrame] = []
    frame_index = 0
    next_timestamp = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = frame_index / fps
            if timestamp + 1e-6 >= next_timestamp:
                out_path = frames_dir / f"frame_{len(extracted) + 1:04d}_{timestamp:08.2f}s.jpg"
                written = cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(image_quality)],
                )
                if not written:
                    raise RuntimeError(f"Could not write frame: {out_path}")

                extracted.append(
                    ExtractedFrame(
                        index=len(extracted) + 1,
                        timestamp_sec=round(timestamp, 3),
                        path=out_path,
                    )
                )

                if max_frames is not None and len(extracted) >= max_frames:
                    break

                next_timestamp += interval_seconds

            frame_index += 1
    finally:
        cap.release()

    return extracted
