from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .captioners import Captioner
from .frame_extractor import extract_frames


@dataclass(frozen=True)
class SceneCaption:
    frame_index: int
    timestamp_sec: float
    frame_path: str
    caption: str


def run_scene_captioning(
    video_path: Path,
    frames_dir: Path,
    captioner: Captioner,
    interval_seconds: float = 2.0,
    max_frames: int | None = None,
    prompt: str | None = None,
) -> list[SceneCaption]:
    frames = extract_frames(
        video_path=video_path,
        frames_dir=frames_dir,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )

    results: list[SceneCaption] = []
    for frame in frames:
        caption = captioner.caption(frame.path, prompt=prompt)
        results.append(
            SceneCaption(
                frame_index=frame.index,
                timestamp_sec=frame.timestamp_sec,
                frame_path=str(frame.path),
                caption=caption,
            )
        )

    return results


def write_outputs(results: list[SceneCaption], json_path: Path, text_path: Path | None = None) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if text_path is None:
        return

    lines: list[str] = []
    for result in results:
        lines.append(f"Frame {result.frame_index} ({result.timestamp_sec:.2f}s)")
        lines.append(f'"{result.caption}"')
        lines.append("")

    text_path = Path(text_path)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
