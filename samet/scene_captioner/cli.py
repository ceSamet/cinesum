from __future__ import annotations

import argparse
import os
from pathlib import Path

from .captioners import create_captioner
from .pipeline import run_scene_captioning, write_outputs


def load_env_file() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    env_path = next(
        (candidate for candidate in (project_dir / ".env", project_dir.parent / ".env") if candidate.is_file()),
        None,
    )
    if env_path is None:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key.replace("_", "").isalnum():
                os.environ.setdefault(key, value.strip().strip("'\""))
        return

    load_dotenv(dotenv_path=env_path, override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video and generate scene captions."
    )
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between extracted frames. Default: 2.0",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "blip", "blip2", "llava", "video-llava", "gemini"],
        default="mock",
        help="Captioning backend. Default: mock",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model name for gemini, blip, blip2, llava, or video-llava backends",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force model device. Defaults to cuda when available.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional prompt passed to the captioning model.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap for fast smoke tests.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("outputs/frames"),
        help="Directory where extracted frames are written.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/captions.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=Path("outputs/captions.txt"),
        help="Text output path.",
    )
    return parser


def main() -> None:
    load_env_file()
    args = build_parser().parse_args()

    captioner = create_captioner(
        backend=args.backend,
        model_name=args.model,
        device=args.device,
    )
    results = run_scene_captioning(
        video_path=args.video,
        frames_dir=args.frames_dir,
        captioner=captioner,
        interval_seconds=args.interval,
        max_frames=args.max_frames,
        prompt=args.prompt,
    )
    write_outputs(results, json_path=args.output_json, text_path=args.output_text)

    print(f"Extracted and captioned {len(results)} frame(s).")
    print(f"JSON: {args.output_json}")
    print(f"Text: {args.output_text}")


if __name__ == "__main__":
    main()
