"""Optional Samet face and voice identity enrichment for Gökdeniz shot data."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.audio.samet_audio_analysis import SpeechTurn, assign_speakers
from src.audio.nezihat_diarization import diarize_transcript, pyannote_available
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
    try:
        presence, cast, face_boxes = analyze_faces(
            adapted, models_dir, video_path=video_path, portraits_dir=portraits_dir,
        )
        face_warning = None
    except Exception as exc:
        presence, cast, face_boxes = {}, [], {}
        face_warning = type(exc).__name__
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
    preferred = os.getenv("THANOS_DIARIZATION_BACKEND", "auto").strip().lower()
    if preferred not in {"auto", "pyannote", "samet"}:
        raise ValueError("THANOS_DIARIZATION_BACKEND auto, pyannote veya samet olmalıdır")
    diarization = None
    fallback_reason = None
    if preferred != "samet" and wav_path.exists() and pyannote_available():
        try:
            diarization = diarize_transcript(wav_path, transcript)
            if not diarization["turns"] and turns:
                diarization = None
                fallback_reason = "Pyannote konuşmacı turu bulamadı"
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"[:240]
    elif preferred == "pyannote":
        fallback_reason = "pyannote.audio veya Hugging Face model erişimi hazır değil"

    if diarization is not None:
        # Use the word-level transcript for scene speakers; keep raw pyannote
        # turns separately so simultaneous speech and correction remain inspectable.
        speech_rows = diarization["speaker_transcript"] or diarization["turns"]
        votes: dict[str, Counter] = defaultdict(Counter)
        for row in diarization["turns"]:
            midpoint = (row["start"] + row["end"]) / 2
            scene = next((item for item in scenes if item["start_seconds"] <= midpoint < item["end_seconds"]), None)
            actors = presence.get(int(scene["scene_id"]), []) if scene else []
            if len(actors) == 1:
                votes[row["speaker"]][actors[0]] += 1
        candidates = sorted((
            (count / sum(counts.values()), count, speaker, actor)
            for speaker, counts in votes.items()
            for actor, count in [counts.most_common(1)[0]]
            if count >= 2 and count / sum(counts.values()) >= 0.72
        ), reverse=True)
        actor_map = {}
        used_actors = set()
        for _, _, speaker, actor in candidates:
            if actor not in used_actors:
                actor_map[speaker] = actor
                used_actors.add(actor)
        final_turns = [{**row, "actor": actor_map.get(row["speaker"])} for row in speech_rows]
        speaker_count = len({row["speaker"] for row in diarization["turns"]})
        backend = "nezihat_pyannote"
    else:
        speaker_count = assign_speakers(wav_path, turns, actor_hints=hints) if wav_path.exists() else 0
        final_turns = [
            {"start": turn.start_sec, "end": turn.end_sec, "text": turn.text,
             "speaker": turn.speaker, "actor": turn.actor}
            for turn in turns
        ]
        backend = "samet_mfcc"
    return {
        "backend": backend,
        "fallback_reason": fallback_reason,
        "face_warning": face_warning,
        "cast": cast,
        "actors_by_scene": {str(key): value for key, value in presence.items()},
        "faces_by_scene": {str(key): value for key, value in face_boxes.items()},
        "speaker_count": speaker_count,
        "turns": final_turns,
        "diarization_turns": diarization["turns"] if diarization else [],
        "overlaps": diarization["overlaps"] if diarization else [],
        "correction": diarization["correction"] if diarization else {},
    }
