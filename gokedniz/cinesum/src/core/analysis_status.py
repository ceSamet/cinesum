from pathlib import Path
from typing import Any, Dict


def get_analysis_artifact_status(
    base_dir: Path,
    video_alias: str,
) -> Dict[str, Any]:
    """Report whether the minimum artifacts required for scoring are present."""
    base_dir = Path(base_dir)
    required = {
        "scene_list": (
            base_dir
            / "outputs"
            / "pyscenedetect"
            / "scene_lists"
            / f"{video_alias}_scenes.json"
        ),
        "clip_features": (
            base_dir
            / "outputs"
            / "features"
            / "visual"
            / f"{video_alias}_clip_features.npy"
        ),
        "clip_metadata": (
            base_dir
            / "outputs"
            / "features"
            / "visual"
            / f"{video_alias}_clip_metadata.json"
        ),
        "audio_features": (
            base_dir
            / "outputs"
            / "features"
            / "audio"
            / f"{video_alias}_audio_features.json"
        ),
    }
    optional = {
        "story_scenes": (
            base_dir
            / "outputs"
            / "features"
            / "story"
            / f"{video_alias}_story_scenes.json"
        ),
        "manifest": base_dir / "outputs" / "manifests" / f"{video_alias}.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return {
        "ready": not missing,
        "missing": missing,
        "required": {name: str(path) for name, path in required.items()},
        "optional": {name: path.exists() for name, path in optional.items()},
    }
