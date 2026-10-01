"""Optional Samet face and voice identity enrichment for Gökdeniz shot data."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.audio.samet_audio_analysis import SpeechTurn, assign_speakers
from src.features.samet_face_analysis import analyze_faces


def analyze_people(
    *,
    scenes: list[dict[str, Any]],
    keyframes: list[dict[str, Any]],
    transcript: dict[str, Any],
    video_path: Path,
    wav_path: Path,
    models_dir: Path,
    portraits_dir: Path,
) -> dict[str, Any]:
    frame_lookup = {}
    for row in keyframes:
        scene_id = int(row["scene_id"])
        frame = Path(row["file_path"])
        if frame.exists() and scene_id not in frame_lookup:
            frame_lookup[scene_id] = frame
    adapted = [
        SimpleNamespace(
            index=int(scene["scene_id"]),
            start_sec=float(scene["start_seconds"]),
            end_sec=float(scene["end_seconds"]),
            duration_sec=float(scene["duration_seconds"]),
            frame_path=frame_lookup[int(scene["scene_id"])],
        )
        for scene in scenes if int(scene["scene_id"]) in frame_lookup
    ]
    presence, cast, face_boxes = analyze_faces(
        adapted, models_dir, video_path=video_path, portraits_dir=portraits_dir,
    )
    turns = [
        SpeechTurn(float(row["start"]), float(row["end"]), str(row.get("text", "")))
        for row in transcript.get("segments", [])
        if row.get("text") and 0 <= float(row.get("start", -1)) < float(row.get("end", 0))
    ]
    hints = []
    for turn in turns:
        midpoint = (turn.start_sec + turn.end_sec) / 2
        scene = next((row for row in scenes if row["start_seconds"] <= midpoint < row["end_seconds"]), None)
        actors = presence.get(int(scene["scene_id"]), []) if scene else []
        hints.append(actors[0] if len(actors) == 1 else None)
    speaker_count = assign_speakers(wav_path, turns, actor_hints=hints) if wav_path.exists() else 0
    return {
        "cast": cast,
        "actors_by_scene": {str(key): value for key, value in presence.items()},
        "faces_by_scene": {str(key): value for key, value in face_boxes.items()},
        "speaker_count": speaker_count,
        "turns": [
            {"start": turn.start_sec, "end": turn.end_sec, "text": turn.text,
             "speaker": turn.speaker, "actor": turn.actor}
            for turn in turns
        ],
    }
