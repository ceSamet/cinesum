"""Nezihat's pyannote diarization and word-level speaker alignment for Thanos."""

from __future__ import annotations

import os
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ID = "pyannote/speaker-diarization-community-1"
INTEGRATION_VERSION = "nezihat-pyannote-1"


def _token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    try:
        from dotenv import load_dotenv

        base_dir = Path(__file__).resolve().parents[2]
        load_dotenv(base_dir / ".env", override=False)
        load_dotenv(base_dir.parent / ".env", override=False)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            return token.strip()
    except ImportError:
        pass
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        return None


def pyannote_available() -> bool:
    if not _token():
        return False
    try:
        import pyannote.audio  # noqa: F401
    except Exception:
        return False
    return True


def load_pcm_wav(path: Path) -> np.ndarray:
    """The analysis pipeline writes 16 kHz, mono, signed 16-bit PCM."""
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Diarization için 16 kHz mono PCM WAV gerekli")
        samples = source.readframes(source.getnframes())
    return np.frombuffer(samples, dtype="<i2").astype(np.float32) / 32768.0


def _annotation_turns(annotation: Any) -> list[dict]:
    return sorted((
        {"start": round(float(region.start), 3),
         "end": round(float(region.end), 3), "speaker": str(speaker)}
        for region, _, speaker in annotation.itertracks(yield_label=True)
        if region.end > region.start
    ), key=lambda row: (row["start"], row["end"]))


def overlap_regions(turns: list[dict], minimum: float = 0.30) -> list[dict]:
    """Keep simultaneous speech separate from exclusive word assignments."""
    boundaries = sorted({value for row in turns for value in (row["start"], row["end"])})
    regions: list[dict] = []
    for start, end in zip(boundaries, boundaries[1:]):
        speakers = sorted({row["speaker"] for row in turns if row["start"] < end and row["end"] > start})
        if len(speakers) < 2:
            continue
        if regions and regions[-1]["speakers"] == speakers and start <= regions[-1]["end"] + 0.02:
            regions[-1]["end"] = end
        else:
            regions.append({"start": start, "end": end, "speakers": speakers})
    return [row for row in regions if row["end"] - row["start"] >= minimum]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def _embeddings(audio: np.ndarray, turns: list[dict], model: Any) -> dict[int, np.ndarray]:
    import torch

    result = {}
    for index, turn in enumerate(turns):
        duration = turn["end"] - turn["start"]
        if duration < 0.5:
            continue
        start = max(0, round(turn["start"] * 16000))
        end = min(len(audio), start + round(min(duration, 8.0) * 16000))
        waveform = torch.from_numpy(audio[start:end]).float().reshape(1, 1, -1)
        if waveform.shape[-1] < model.min_num_samples:
            continue
        vector = np.asarray(model(waveform)[0], dtype=np.float32)
        if np.all(np.isfinite(vector)):
            result[index] = _normalize(vector)
    return result


def _prototypes(turns: list[dict], embeddings: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for index, vector in embeddings.items():
        turn = turns[index]
        grouped[turn["speaker"]].append((min(5.0, turn["end"] - turn["start"]), vector))
    return {
        speaker: _normalize(np.average(
            np.vstack([vector for _, vector in rows]), axis=0,
            weights=[duration for duration, _ in rows],
        ))
        for speaker, rows in grouped.items()
    }


def correct_speakers(
    turns: list[dict], embeddings: dict[int, np.ndarray], overlaps: list[dict],
) -> tuple[list[dict], dict]:
    """Conservative Nezihat-style prototype repair, without erasing overlaps."""
    corrected = [row.copy() for row in turns]
    reassigned = 0
    for _ in range(2):
        prototypes = _prototypes(corrected, embeddings)
        changes = 0
        for index, vector in embeddings.items():
            assigned = corrected[index]["speaker"]
            if assigned not in prototypes:
                continue
            scores = {speaker: float(vector @ prototype) for speaker, prototype in prototypes.items()}
            best = max(scores, key=scores.get)
            simultaneous = any(
                assigned in region["speakers"] and best in region["speakers"]
                and _overlap(corrected[index], region) > 0
                for region in overlaps
            )
            if best != assigned and not simultaneous and scores[best] >= 0.72 and scores[best] - scores[assigned] >= 0.15:
                corrected[index]["speaker"] = best
                changes += 1
        reassigned += changes
        if not changes:
            break

    prototypes = _prototypes(corrected, embeddings)
    parent = {speaker: speaker for speaker in prototypes}

    def root(speaker: str) -> str:
        while parent[speaker] != speaker:
            speaker = parent[speaker]
        return speaker

    # Nezihat preserves speakers who truly overlap, even when embeddings look similar.
    forbidden = [{*region["speakers"]} for region in overlaps]
    members = {speaker: {speaker} for speaker in prototypes}
    pairs = sorted((
        (float(left_vector @ prototypes[right]), left, right)
        for left, left_vector in prototypes.items()
        for right in prototypes if left < right
    ), reverse=True)
    for similarity, left, right in pairs:
        if similarity < 0.88:
            break
        a, b = root(left), root(right)
        if a == b or any(group & members[a] and group & members[b] for group in forbidden):
            continue
        parent[b] = a
        members[a].update(members.pop(b))
    for turn in corrected:
        if turn["speaker"] in parent:
            turn["speaker"] = root(turn["speaker"])
    return corrected, {
        "raw_speaker_count": len({row["speaker"] for row in turns}),
        "final_speaker_count": len({row["speaker"] for row in corrected}),
        "embedded_segments": len(embeddings),
        "reassigned_segments": reassigned,
        "merged_speaker_clusters": {speaker: root(speaker) for speaker in parent if root(speaker) != speaker},
    }


def _overlap(left: dict, right: dict) -> float:
    return max(0.0, min(left["end"], right["end"]) - max(left["start"], right["start"]))


def _nearest_speaker(item: dict, turns: list[dict]) -> str:
    if not turns:
        return "UNKNOWN"
    midpoint = (item["start"] + item["end"]) / 2
    return min(turns, key=lambda turn: min(
        abs(midpoint - turn["start"]), abs(midpoint - turn["end"]),
    ))["speaker"]


def assign_word_speakers(words: list[dict], turns: list[dict], switch_penalty: float = 0.35) -> list[str]:
    """Nezihat's duration overlap + switch penalty prevents one-word label flips."""
    if not words:
        return []
    speakers = sorted({turn["speaker"] for turn in turns})
    if not speakers:
        return ["UNKNOWN"] * len(words)
    emissions = []
    for word in words:
        duration = max(0.01, word["end"] - word["start"])
        scores = {speaker: sum(
            _overlap(word, turn) for turn in turns if turn["speaker"] == speaker
        ) / duration for speaker in speakers}
        if max(scores.values()) == 0:
            scores[_nearest_speaker(word, turns)] = 0.1
        emissions.append(scores)
    paths = {speaker: (emissions[0][speaker], [speaker]) for speaker in speakers}
    for scores in emissions[1:]:
        paths = {speaker: max((
            (value + scores[speaker] - (switch_penalty if previous != speaker else 0),
             path + [speaker])
            for previous, (value, path) in paths.items()
        ), key=lambda candidate: candidate[0]) for speaker in speakers}
    return max(paths.values(), key=lambda item: item[0])[1]


def align_transcript(transcript: dict, turns: list[dict]) -> list[dict]:
    """Split a Whisper segment only when the word-level speaker changes."""
    aligned = []
    for segment in transcript.get("segments", []):
        words = [
            {"start": float(word["start"]), "end": float(word["end"]),
             "text": str(word.get("word", word.get("text", ""))).strip()}
            for word in segment.get("words") or []
            if word.get("start") is not None and word.get("end") is not None
            and float(word["end"]) > float(word["start"])
        ]
        if words:
            for word, speaker in zip(words, assign_word_speakers(words, turns)):
                if aligned and aligned[-1]["speaker"] == speaker and word["start"] - aligned[-1]["end"] <= 0.08:
                    aligned[-1]["end"] = word["end"]
                    aligned[-1]["text"] += " " + word["text"]
                else:
                    aligned.append({**word, "speaker": speaker})
        elif str(segment.get("text", "")).strip():
            item = {"start": float(segment["start"]), "end": float(segment["end"]),
                    "text": str(segment["text"]).strip()}
            scores = {speaker: sum(_overlap(item, turn) for turn in turns if turn["speaker"] == speaker)
                      for speaker in {turn["speaker"] for turn in turns}}
            item["speaker"] = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else _nearest_speaker(item, turns)
            aligned.append(item)
    return aligned


def diarize_transcript(wav_path: Path, transcript: dict) -> dict:
    """Run Nezihat's gated pyannote model on Thanos' already extracted WAV."""
    import torch
    from pyannote.audio import Pipeline

    token = _token()
    if not token:
        raise RuntimeError("HF_TOKEN veya Hugging Face oturumu gerekli")
    audio = load_pcm_wav(wav_path)
    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    device = "cuda" if torch.cuda.is_available() and os.getenv("THANOS_DIARIZATION_DEVICE", "cpu") == "cuda" else "cpu"
    pipeline.to(torch.device(device))
    pipeline.instantiate({"clustering": {"threshold": 0.40}})
    result = pipeline({
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": 16000,
        "uri": str(wav_path),
    })
    regular = getattr(result, "speaker_diarization", result)
    exclusive = getattr(result, "exclusive_speaker_diarization", regular)
    raw_turns = _annotation_turns(exclusive)
    overlaps = overlap_regions(_annotation_turns(regular))
    try:
        embeddings = _embeddings(audio, raw_turns, pipeline._embedding)
        turns, report = correct_speakers(raw_turns, embeddings, overlaps)
    except Exception as exc:
        # The diarization itself is still useful if its optional correction fails.
        turns = raw_turns
        report = {"raw_speaker_count": len({row["speaker"] for row in raw_turns}),
                  "final_speaker_count": len({row["speaker"] for row in raw_turns}),
                  "status": "correction_unavailable", "reason": type(exc).__name__}
    return {
        "backend": "nezihat_pyannote",
        "model": MODEL_ID,
        "turns": turns,
        "overlaps": overlaps,
        "speaker_transcript": align_transcript(transcript, turns),
        "correction": report,
    }
