import os
import json
from pathlib import Path
from typing import List, Dict, Any

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import numpy as np
from src.features.clip_extractor import get_clip_model
from src.features.emotion_features import emotion_peaks

_action_text_vec = None
_dialogue_text_vec = None
_narrative_text_vec = None
_credit_text_vec = None

# Number of genre-neutral narrative-event prompts (kept separate from the
# credits/title-card prompt bank below so each group can be scored on its own
# within-video percentile scale).
_NARRATIVE_PROMPT_COUNT = 4
_CREDIT_PROMPT_COUNT = 3

def get_clip_text_vectors():
    """
    Lazy loader for CLIP model and normalized text prompt vectors
    (Action, Dialogue, Narrative-event, Credits/title-card).
    """
    global _action_text_vec, _dialogue_text_vec, _narrative_text_vec, _credit_text_vec
    if (
        _action_text_vec is None
        or _dialogue_text_vec is None
        or _narrative_text_vec is None
        or _credit_text_vec is None
    ):
        model_name = "openai/clip-vit-base-patch32"
        clip_model, clip_processor = get_clip_model(model_name)
        device = next(clip_model.parameters()).device

        prompts = [
            "a fight, fast movement, explosion, or action scene",
            "two people having a conversation, dialogue, or interview",
            "a decisive heroic turning point, major revelation, sacrifice, victory, or story climax",
            "a character gaining a new power or wielding a legendary weapon",
            "a character making a decisive gesture with glowing magical energy",
            "an emotional heroic sacrifice that saves everyone",
            "movie end credits, cast and crew names scrolling on a plain background",
            "an opening title card with a movie logo and studio name on a black screen",
            "a blank or static screen containing only text, with no characters or action",
        ]

        inputs = clip_processor(text=prompts, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            text_out = clip_model.get_text_features(**inputs)
            if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
                text_vecs = text_out.pooler_output
            elif hasattr(text_out, "text_embeds") and text_out.text_embeds is not None:
                text_vecs = text_out.text_embeds
            elif isinstance(text_out, torch.Tensor):
                text_vecs = text_out
            else:
                text_vecs = text_out[0]

            if text_vecs.ndim == 3:
                text_vecs = text_vecs[:, 0, :]

            # L2 Normalize
            text_vecs = text_vecs / text_vecs.norm(p=2, dim=-1, keepdim=True)
            text_np = text_vecs.cpu().numpy()

            _action_text_vec = text_np[0]     # Shape: (512,)
            _dialogue_text_vec = text_np[1]   # Shape: (512,)
            # Multiple genre-neutral event descriptions improve recall without
            # hard-coding a particular film, character, weapon, or ending.
            narrative_end = 2 + _NARRATIVE_PROMPT_COUNT
            _narrative_text_vec = text_np[2:narrative_end]              # Shape: (N, 512)
            _credit_text_vec = text_np[narrative_end:narrative_end + _CREDIT_PROMPT_COUNT]  # Shape: (M, 512)

    return _action_text_vec, _dialogue_text_vec, _narrative_text_vec, _credit_text_vec

def compute_scene_scores_for_video(
    video_alias: str,
    base_dir: Path
) -> List[Dict[str, Any]]:
    """
    Compute action_score, dialogue_score, and importance_score per scene by fusing
    Visual CLIP features, Audio Energy, and Speech Transcripts.
    Importance is content-led; chronology is handled by the soft adaptive selector.
    """
    visual_dir = base_dir / "outputs" / "features" / "visual"
    audio_dir = base_dir / "outputs" / "features" / "audio"
    scene_lists_dir = base_dir / "outputs" / "pyscenedetect" / "scene_lists"
    story_dir = base_dir / "outputs" / "features" / "story"

    npy_file = visual_dir / f"{video_alias}_clip_features.npy"
    v_meta_file = visual_dir / f"{video_alias}_clip_metadata.json"
    audio_json_file = audio_dir / f"{video_alias}_audio_features.json"
    scenes_json_file = scene_lists_dir / f"{video_alias}_scenes.json"
    story_json_file = story_dir / f"{video_alias}_story_scenes.json"
    people_json_file = audio_dir / f"{video_alias}_people.json"

    if not npy_file.exists() or not v_meta_file.exists():
        raise FileNotFoundError(f"Visual CLIP özellikleri bulunamadı: {video_alias}")
    if not audio_json_file.exists():
        raise FileNotFoundError(f"Audio ses özellikleri bulunamadı: {video_alias}")
    if not scenes_json_file.exists():
        raise FileNotFoundError(f"Sahne listesi bulunamadı: {video_alias}")

    # Load data
    image_vectors = np.load(str(npy_file))
    with open(v_meta_file, "r", encoding="utf-8") as f:
        clip_meta = json.load(f)
    with open(audio_json_file, "r", encoding="utf-8") as f:
        audio_feats = json.load(f)
    with open(scenes_json_file, "r", encoding="utf-8") as f:
        scenes_data = json.load(f)
    try:
        people_data = json.loads(people_json_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        people_data = {}
    actors_by_scene = people_data.get("actors_by_scene", {})
    speaker_turns = people_data.get("turns", [])

    story_lookup = {}
    story_modes = {}
    if story_json_file.exists():
        try:
            with open(story_json_file, "r", encoding="utf-8") as f:
                story_data = json.load(f)
            story_lookup = story_data.get("shot_to_story_scene", {})
            story_modes = {
                int(story["story_scene_id"]): story.get("dominant_mode", "general")
                for story in story_data.get("story_scenes", [])
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            story_lookup = {}
            story_modes = {}

    # Get prompt text vectors
    action_text_v, dialogue_text_v, narrative_text_v, credit_text_v = get_clip_text_vectors()

    # Compute CLIP similarities per keyframe
    clip_action_sims = np.dot(image_vectors, action_text_v)       # [Num_Keyframes]
    clip_dialogue_sims = np.dot(image_vectors, dialogue_text_v)   # [Num_Keyframes]
    clip_narrative_sims = np.dot(image_vectors, narrative_text_v.T) # [Num_Keyframes, N]
    clip_credit_sims = np.dot(image_vectors, credit_text_v.T)     # [Num_Keyframes, M]

    # Map keyframe CLIP similarities to scene_id
    scene_clip_action = {}
    scene_clip_dialogue = {}
    scene_clip_narrative = {}
    scene_clip_credit = {}

    for idx, kf in enumerate(clip_meta):
        sc_id = kf["scene_id"]
        if sc_id not in scene_clip_action:
            scene_clip_action[sc_id] = []
            scene_clip_dialogue[sc_id] = []
            scene_clip_narrative[sc_id] = []
            scene_clip_credit[sc_id] = []
        scene_clip_action[sc_id].append(float(clip_action_sims[idx]))
        scene_clip_dialogue[sc_id].append(float(clip_dialogue_sims[idx]))
        scene_clip_narrative[sc_id].append(clip_narrative_sims[idx])
        scene_clip_credit[sc_id].append(clip_credit_sims[idx])

    # CLIP's absolute similarity varies substantially by prompt. Convert each
    # narrative prompt to a within-video percentile, then keep the strongest event
    # signal. This lets rare visual turning points compete with sustained loud action
    # without assuming a particular title or character.
    narrative_scene_ids = sorted(scene_clip_narrative)
    narrative_means = np.asarray([
        np.mean(scene_clip_narrative[scene_id], axis=0)
        for scene_id in narrative_scene_ids
    ]) if narrative_scene_ids else np.empty((0, len(narrative_text_v)))
    narrative_percentiles = np.zeros_like(narrative_means)
    if len(narrative_scene_ids) > 1:
        for prompt_index in range(narrative_means.shape[1]):
            order = np.argsort(narrative_means[:, prompt_index], kind="stable")
            narrative_percentiles[order, prompt_index] = (
                np.arange(len(order), dtype=np.float32) / (len(order) - 1)
            )
    scene_narrative_salience = {
        scene_id: float(np.max(narrative_percentiles[index]))
        for index, scene_id in enumerate(narrative_scene_ids)
    }
    scene_narrative_raw = {
        scene_id: float(np.max(narrative_means[index]))
        for index, scene_id in enumerate(narrative_scene_ids)
    }

    # Same within-video percentile trick for the credits/title-card prompt bank:
    # absolute CLIP similarity to "text on a screen" prompts drifts by video, so we
    # rank shots against the rest of the same film instead of a fixed threshold.
    credit_scene_ids = narrative_scene_ids
    credit_means = np.asarray([
        np.mean(scene_clip_credit[scene_id], axis=0)
        for scene_id in credit_scene_ids
    ]) if credit_scene_ids else np.empty((0, len(credit_text_v)))
    credit_percentiles = np.zeros_like(credit_means)
    if len(credit_scene_ids) > 1:
        for prompt_index in range(credit_means.shape[1]):
            order = np.argsort(credit_means[:, prompt_index], kind="stable")
            credit_percentiles[order, prompt_index] = (
                np.arange(len(order), dtype=np.float32) / (len(order) - 1)
            )
    scene_credit_visual_salience = {
        scene_id: float(np.max(credit_percentiles[index]))
        for index, scene_id in enumerate(credit_scene_ids)
    }

    # Map audio features by scene_id
    audio_by_scene = {af["scene_id"]: af for af in audio_feats}
    emotion_by_scene = emotion_peaks(audio_feats)

    scored_scenes = []
    total_scenes = len(scenes_data)
    video_duration_seconds = max(
        (float(scene.get("end_seconds", 0.0)) for scene in scenes_data),
        default=0.0,
    )

    for i, sc in enumerate(scenes_data):
        sc_id = sc["scene_id"]
        dur = sc["duration_seconds"]

        # 1. Visual CLIP scores
        c_act_list = scene_clip_action.get(sc_id, [0.20])
        c_dia_list = scene_clip_dialogue.get(sc_id, [0.20])
        clip_act = float(np.mean(c_act_list))
        clip_dia = float(np.mean(c_dia_list))
        clip_narrative = scene_narrative_raw.get(sc_id, 0.20)

        # Normalize CLIP similarities to [0.0, 1.0] range
        clip_act_norm = min(max((clip_act - 0.15) / 0.20, 0.0), 1.0)
        clip_dia_norm = min(max((clip_dia - 0.15) / 0.20, 0.0), 1.0)
        clip_narrative_norm = scene_narrative_salience.get(sc_id, 0.0)

        # 2. Audio features
        af = audio_by_scene.get(sc_id, {})
        audio_energy = af.get("normalized_audio_energy", 0.0)
        audio_onset_contrast = float(af.get("audio_onset_contrast", 0.0))
        speech_ratio = af.get("speech_ratio", 0.0)
        transcript_density = min(af.get("transcript_density", 0.0) / 3.0, 1.0)
        transcript_text = af.get("transcript_text", "")
        words = af.get("words", [])
        emotion = emotion_by_scene.get(sc_id, {})
        story_meta = story_lookup.get(str(sc_id), {})
        story_scene_id = int(story_meta.get("story_scene_id", sc_id))

        # 3. Cut frequency / duration score
        cut_freq_score = min(max((15.0 - dur) / 15.0, 0.0), 1.0)

        # 4. Narrative position is metadata, not a universal three-act proxy.
        position_ratio = (i + 1) / total_scenes

        # 5. Content-based credits/title-card detection (opening AND closing,
        # any video length). Uses real video-time position, not shot index, so
        # uneven shot density near the edges doesn't skew the estimate. A shot
        # only scores high if it is BOTH near an edge AND visually resembles a
        # text/credits screen, so a legitimate mid-movie black/text shot is not
        # mistaken for the credits.
        time_ratio = float(sc["start_seconds"]) / max(video_duration_seconds, 0.001)
        edge_strength = max(
            max(0.0, (0.03 - time_ratio) / 0.03),
            max(0.0, (time_ratio - 0.96) / 0.04),
        )
        visual_credit_signal = scene_credit_visual_salience.get(sc_id, 0.0)
        low_dialogue_signal = 1.0 - min(speech_ratio * 2.0, 1.0)
        credit_probability = (
            (0.65 * visual_credit_signal)
            + (0.25 * edge_strength)
            + (0.10 * low_dialogue_signal)
        )
        if edge_strength <= 0.0:
            # Not near either edge of the video: heavily discount, this cannot be
            # the opening/closing credit roll no matter how "text-like" it looks.
            credit_probability *= 0.35
        credit_probability = min(max(credit_probability, 0.0), 1.0)

        # --- MASTER PLAN SCORING FORMULAS ---
        # Action Score
        raw_action_score = (0.40 * clip_act_norm) + (0.35 * audio_energy) + (0.25 * cut_freq_score)

        # Dialogue Score
        raw_dialogue_score = (0.45 * speech_ratio) + (0.30 * clip_dia_norm) + (0.25 * transcript_density)

        # Importance: content signals; no fixed beginning/end boost.
        visual_importance = max(clip_act_norm, clip_dia_norm)
        raw_importance_score = (
            (0.16 * visual_importance) +
            (0.39 * clip_narrative_norm) +
            (0.17 * speech_ratio) +
            (0.10 * audio_energy) +
            (0.10 * audio_onset_contrast) +
            (0.08 * float(emotion.get("emotion_peak", 0.0)))
        )

        # Determine Category Labels
        labels = []
        if raw_action_score >= 0.45:
            labels.append("action")
        if raw_dialogue_score >= 0.40:
            labels.append("dialogue")
        if raw_importance_score >= 0.40:
            labels.append("important")
        if not labels:
            labels.append("general")

        scored_scenes.append({
            "scene_id": sc_id,
            "actors": actors_by_scene.get(str(sc_id), []),
            "speakers": sorted({
                str(turn.get("speaker")) for turn in speaker_turns
                if float(turn.get("end", 0)) > float(sc["start_seconds"])
                and float(turn.get("start", 0)) < float(sc["end_seconds"])
                and turn.get("speaker")
            }),
            "speaker_transcript": [
                {"start": turn.get("start"), "end": turn.get("end"),
                 "speaker": turn.get("speaker"), "text": turn.get("text")}
                for turn in speaker_turns
                if turn.get("text")
                and float(turn.get("end", 0)) > float(sc["start_seconds"])
                and float(turn.get("start", 0)) < float(sc["end_seconds"])
            ],
            "start_timecode": sc["start_timecode"],
            "end_timecode": sc["end_timecode"],
            "start_seconds": sc["start_seconds"],
            "end_seconds": sc["end_seconds"],
            "duration_seconds": dur,
            "clip_action_similarity": round(clip_act, 4),
            "clip_dialogue_similarity": round(clip_dia, 4),
            "clip_narrative_similarity": round(clip_narrative, 4),
            "narrative_event_score": round(clip_narrative_norm, 4),
            "normalized_audio_energy": round(audio_energy, 4),
            "audio_onset_contrast": round(audio_onset_contrast, 4),
            "emotion_peak": float(emotion.get("emotion_peak", 0.0)),
            "emotion_change": float(emotion.get("emotion_change", 0.0)),
            "emotion_cue_confidence": float(emotion.get("confidence", 0.0)),
            "decision_cue": float(emotion.get("decision", 0.0)),
            "speech_ratio": round(speech_ratio, 4),
            "transcript_density": round(transcript_density, 4),
            "transcript_text": transcript_text,
            "words": words,
            "story_scene_id": story_scene_id,
            "story_scene_position": int(story_meta.get("position", 0)),
            "story_scene_shot_count": int(story_meta.get("shot_count", 1)),
            "story_scene_mode": story_modes.get(story_scene_id, "general"),
            "story_position_ratio": round(position_ratio, 6),
            "action_score": round(float(raw_action_score), 4),
            "dialogue_score": round(float(raw_dialogue_score), 4),
            "importance_score": round(float(raw_importance_score), 4),
            "category_labels": labels,
            "credit_probability": round(float(credit_probability), 4),
        })

    return scored_scenes
