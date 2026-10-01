import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from src.audio.audio_extractor import compute_scene_audio_energy, extract_audio_from_video
from src.audio.transcript_repair import (
    TRANSCRIPT_REPAIR_VERSION,
    repair_transcript_anomalies,
)
from src.audio.whisper_transcriber import (
    map_transcript_to_scenes,
    resolve_whisper_compute_type,
    resolve_whisper_device,
    save_transcript,
    transcribe_audio_with_whisper,
)
from src.core.feature_manifest import FeatureManifest, stable_config_hash
from src.core.pipeline_config import PIPELINE_VERSION, AnalysisProfile, get_analysis_profile
from src.features.clip_extractor import (
    extract_clip_features_for_video,
    load_or_reconstruct_keyframes_metadata,
    resolve_clip_device,
)
from src.scene_detection.keyframe_extractor import (
    extract_keyframes_for_scenes,
    save_keyframe_metadata,
)
from src.scene_detection.pyscenedetect_runner import (
    detect_scenes_pyscenedetect,
    save_scenes_to_csv,
    save_scenes_to_json,
)
from src.story.story_scene_builder import build_story_scenes_from_files


ProgressCallback = Callable[[int, str, str, str], None]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def _notify(
    callback: Optional[ProgressCallback],
    progress: int,
    description: str,
    detail: str,
    step: str,
) -> None:
    if callback:
        callback(progress, description, detail, step)


def find_reusable_keyframes(
    video_alias: str,
    keyframes_dir: Path,
    scenes: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Reuse legacy keyframes only when their scene coverage matches exactly."""
    existing_metadata = load_or_reconstruct_keyframes_metadata(
        video_alias,
        Path(keyframes_dir),
    )
    expected_scene_ids = {int(scene["scene_id"]) for scene in scenes}
    existing_scene_ids = {int(metadata["scene_id"]) for metadata in existing_metadata}
    if existing_metadata and existing_scene_ids == expected_scene_ids:
        return existing_metadata
    return []


def _run_stage(
    manifest: FeatureManifest,
    name: str,
    config: Dict[str, Any],
    artifacts: Iterable[Path],
    operation: Callable[[], Any],
    *,
    force: bool = False,
) -> tuple[Any, bool, float]:
    artifact_list = list(artifacts)
    if not force and manifest.stage_is_valid(name, config, artifact_list):
        return None, True, 0.0

    manifest.mark_running(name, config)
    started = time.perf_counter()
    try:
        result = operation()
        elapsed = time.perf_counter() - started
        manifest.mark_complete(name, config, artifact_list, elapsed)
        return result, False, elapsed
    except Exception as exc:
        manifest.mark_failed(name, config, exc)
        raise


def analyze_video_features(
    video_alias: str,
    video_path: Path,
    base_dir: Path,
    *,
    profile_name: str = "balanced",
    progress_callback: Optional[ProgressCallback] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Build reusable video features with stage-level cache validation."""
    video_path = Path(video_path)
    base_dir = Path(base_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    profile: AnalysisProfile = get_analysis_profile(profile_name)
    scene_lists_dir = base_dir / "outputs" / "pyscenedetect" / "scene_lists"
    keyframes_dir = base_dir / "outputs" / "pyscenedetect" / "keyframes"
    visual_dir = base_dir / "outputs" / "features" / "visual"
    raw_audio_dir = base_dir / "outputs" / "audio"
    audio_features_dir = base_dir / "outputs" / "features" / "audio"
    story_features_dir = base_dir / "outputs" / "features" / "story"
    manifest_path = base_dir / "outputs" / "manifests" / f"{video_alias}.json"

    for directory in (
        scene_lists_dir,
        keyframes_dir,
        visual_dir,
        raw_audio_dir,
        audio_features_dir,
        story_features_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    scenes_json = scene_lists_dir / f"{video_alias}_scenes.json"
    scenes_csv = scene_lists_dir / f"{video_alias}_adaptive_t3.0.csv"
    keyframes_metadata = keyframes_dir / video_alias / "keyframes_metadata.json"
    clip_npy = visual_dir / f"{video_alias}_clip_features.npy"
    clip_metadata = visual_dir / f"{video_alias}_clip_metadata.json"
    wav_path = raw_audio_dir / f"{video_alias}.wav"
    raw_transcript_path = audio_features_dir / f"{video_alias}_transcript_raw.json"
    transcript_path = audio_features_dir / f"{video_alias}_transcript.json"
    audio_features_path = audio_features_dir / f"{video_alias}_audio_features.json"
    story_scenes_path = story_features_dir / f"{video_alias}_story_scenes.json"
    story_embeddings_path = story_features_dir / f"{video_alias}_story_embeddings.npy"

    _notify(progress_callback, 5, "Video kimliği doğrulanıyor...", video_alias, "stepScene")
    manifest = FeatureManifest(manifest_path, video_path)

    scene_config = {
        "pipeline_version": PIPELINE_VERSION,
        "detector": "pyscenedetect-adaptive",
        "adaptive_threshold": 3.0,
    }

    def build_scenes():
        detected = detect_scenes_pyscenedetect(
            str(video_path),
            detector_type="adaptive",
            adaptive_threshold=3.0,
        )
        save_scenes_to_json(detected, str(scenes_json))
        save_scenes_to_csv(detected, str(scenes_csv))
        return detected

    _notify(progress_callback, 15, "Sahneler bölütleniyor...", "PySceneDetect AdaptiveDetector", "stepScene")
    scenes, scene_cache_hit, _ = _run_stage(
        manifest,
        "scene_detection",
        scene_config,
        [scenes_json, scenes_csv],
        build_scenes,
        force=force,
    )
    if scene_cache_hit:
        scenes = _read_json(scenes_json)
    if not scenes:
        raise RuntimeError("Video için hiçbir sahne tespit edilemedi.")

    keyframe_config = {
        "pipeline_version": PIPELINE_VERSION,
        "scene_config_hash": stable_config_hash(scene_config),
        "long_scene_threshold": 10.0,
        "sample_step": 2,
    }

    def build_keyframes():
        if not force:
            existing_metadata = find_reusable_keyframes(
                video_alias,
                keyframes_dir,
                scenes,
            )
            if existing_metadata:
                save_keyframe_metadata(existing_metadata, str(keyframes_metadata))
                return existing_metadata

        metadata = extract_keyframes_for_scenes(
            str(video_path),
            scenes,
            str(keyframes_dir),
            long_scene_threshold=10.0,
            sample_step=2,
        )
        if not metadata:
            raise RuntimeError("Hiçbir keyframe üretilemedi.")
        save_keyframe_metadata(metadata, str(keyframes_metadata))
        return metadata

    _notify(progress_callback, 30, "En net keyframe'ler çıkarılıyor...", f"{len(scenes)} sahne", "stepScene")
    _, keyframe_cache_hit, _ = _run_stage(
        manifest,
        "keyframes",
        keyframe_config,
        [keyframes_metadata],
        build_keyframes,
        force=force or not scene_cache_hit,
    )

    clip_device = resolve_clip_device()
    clip_batch_size = profile.clip_batch_size or (32 if clip_device == "cuda" else 8)
    clip_config = {
        "pipeline_version": PIPELINE_VERSION,
        "keyframe_config_hash": stable_config_hash(keyframe_config),
        "model": "openai/clip-vit-base-patch32",
        "device": clip_device,
        "batch_size": clip_batch_size,
    }

    def build_clip():
        features, metadata = extract_clip_features_for_video(
            video_alias,
            str(keyframes_dir),
            str(clip_npy),
            str(clip_metadata),
            model_name=clip_config["model"],
            batch_size=clip_batch_size,
        )
        if len(features) == 0 or not metadata:
            raise RuntimeError("CLIP görsel özellikleri üretilemedi.")
        return features

    _notify(progress_callback, 48, "CLIP görsel embedding'leri üretiliyor...", profile.name, "stepClip")
    _, clip_cache_hit, _ = _run_stage(
        manifest,
        "clip",
        clip_config,
        [clip_npy, clip_metadata],
        build_clip,
        force=force or not keyframe_cache_hit,
    )

    audio_config = {
        "pipeline_version": PIPELINE_VERSION,
        "sample_rate": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
    }
    _notify(progress_callback, 62, "Ses kanalı hazırlanıyor...", "16 kHz mono PCM", "stepAudio")
    _, audio_cache_hit, _ = _run_stage(
        manifest,
        "audio_extraction",
        audio_config,
        [wav_path],
        lambda: extract_audio_from_video(str(video_path), str(wav_path)),
        force=force,
    )

    whisper_device = resolve_whisper_device()
    whisper_compute_type = resolve_whisper_compute_type(
        whisper_device,
        profile.whisper_compute_type,
    )
    transcript_config = {
        "pipeline_version": PIPELINE_VERSION,
        "audio_config_hash": stable_config_hash(audio_config),
        "backend": "faster-whisper",
        "model": profile.whisper_model,
        "beam_size": profile.whisper_beam_size,
        "device": whisper_device,
        "compute_type": whisper_compute_type,
        "word_timestamps": profile.word_timestamps,
        "vad_filter": profile.vad_filter,
        "language": "auto",
    }

    def build_transcript():
        # One-time migration: preserve an existing cached transcript as the raw
        # source so adding the repair stage does not retranscribe a whole movie.
        if transcript_path.exists() and not raw_transcript_path.exists() and not force:
            transcript = _read_json(transcript_path)
            _write_json(raw_transcript_path, transcript)
            return transcript
        transcript = transcribe_audio_with_whisper(
            str(wav_path),
            model_size=profile.whisper_model,
            backend="faster-whisper",
            device=whisper_device,
            compute_type=whisper_compute_type,
            beam_size=profile.whisper_beam_size,
            word_timestamps=profile.word_timestamps,
            vad_filter=profile.vad_filter,
        )
        save_transcript(transcript, str(raw_transcript_path))
        return transcript

    _notify(
        progress_callback,
        74,
        "Faster-Whisper transkripsiyonu çalışıyor...",
        f"{profile.whisper_model}, beam={profile.whisper_beam_size}",
        "stepAudio",
    )
    transcript, transcript_cache_hit, _ = _run_stage(
        manifest,
        "transcript",
        transcript_config,
        [raw_transcript_path],
        build_transcript,
        force=force or not audio_cache_hit,
    )
    if transcript_cache_hit:
        transcript = _read_json(raw_transcript_path)

    transcript_repair_config = {
        "pipeline_version": PIPELINE_VERSION,
        "repair_version": TRANSCRIPT_REPAIR_VERSION,
        "transcript_config_hash": stable_config_hash(transcript_config),
        "max_segment_sec": 45.0,
        "force_repair_segment_sec": 90.0,
        "chunk_sec": 30.0,
        "max_anomalies": 8,
        "device": whisper_device,
        "compute_type": whisper_compute_type,
    }

    def build_repaired_transcript():
        repaired, _ = repair_transcript_anomalies(
            transcript,
            wav_path,
            language=transcript.get("language"),
            model_size=profile.whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
        )
        save_transcript(repaired, str(transcript_path))
        return repaired

    _notify(
        progress_callback,
        82,
        "ASR zaman anomalileri denetleniyor...",
        "Yalnız uzun ve seyrek konuşma blokları yeniden işlenir",
        "stepAudio",
    )
    transcript, transcript_repair_cache_hit, _ = _run_stage(
        manifest,
        "transcript_repair",
        transcript_repair_config,
        [transcript_path],
        build_repaired_transcript,
        force=force or not transcript_cache_hit,
    )
    if transcript_repair_cache_hit:
        transcript = _read_json(transcript_path)

    audio_features_config = {
        "pipeline_version": PIPELINE_VERSION,
        "audio_onset_version": "1.0",
        "transcript_scene_mapping_version": "2.0-word-boundaries",
        "scene_config_hash": stable_config_hash(scene_config),
        "audio_config_hash": stable_config_hash(audio_config),
        "transcript_config_hash": stable_config_hash(transcript_repair_config),
    }

    def build_audio_features():
        energy = compute_scene_audio_energy(str(wav_path), scenes)
        mapped_transcript = map_transcript_to_scenes(transcript, scenes)
        merged = []
        for audio_row, transcript_row in zip(energy, mapped_transcript):
            merged.append({
                **audio_row,
                "speech_ratio": transcript_row["speech_ratio"],
                "speech_time_seconds": transcript_row["speech_time_seconds"],
                "word_count": transcript_row["word_count"],
                "transcript_density": transcript_row["transcript_density"],
                "transcript_text": transcript_row["transcript_text"],
                "words": transcript_row["words"],
            })
        _write_json(audio_features_path, merged)
        return merged

    _notify(progress_callback, 90, "Ses ve konuşma özellikleri birleştiriliyor...", "Feature matrix", "stepAudio")
    _, audio_features_cache_hit, _ = _run_stage(
        manifest,
        "audio_features",
        audio_features_config,
        [audio_features_path],
        build_audio_features,
        force=(
            force
            or not scene_cache_hit
            or not audio_cache_hit
            or not transcript_repair_cache_hit
        ),
    )

    story_config = {
        "pipeline_version": PIPELINE_VERSION,
        "scene_config_hash": stable_config_hash(scene_config),
        "clip_config_hash": stable_config_hash(clip_config),
        "audio_features_config_hash": stable_config_hash(audio_features_config),
        "schema_version": "1.0",
        "weights": {
            "visual": 0.52,
            "text": 0.28,
            "speech": 0.12,
            "temporal": 0.08,
        },
        "max_story_duration_sec": 45.0,
        "max_shots_per_story": 12,
    }

    _notify(
        progress_callback,
        96,
        "Hikâye sahneleri gruplanıyor...",
        "CLIP + transcript + temporal continuity",
        "stepClip",
    )
    story_result, story_cache_hit, _ = _run_stage(
        manifest,
        "story_scenes",
        story_config,
        [story_scenes_path, story_embeddings_path],
        lambda: build_story_scenes_from_files(
            scenes_json,
            clip_npy,
            clip_metadata,
            audio_features_path,
            story_scenes_path,
            story_embeddings_path,
        ),
        force=(
            force
            or not scene_cache_hit
            or not clip_cache_hit
            or not audio_features_cache_hit
        ),
    )
    if story_cache_hit:
        story_result = _read_json(story_scenes_path)

    cache_hits = {
        "scene_detection": scene_cache_hit,
        "keyframes": keyframe_cache_hit,
        "clip": clip_cache_hit,
        "audio_extraction": audio_cache_hit,
        "transcript": transcript_cache_hit,
        "transcript_repair": transcript_repair_cache_hit,
        "audio_features": audio_features_cache_hit,
        "story_scenes": story_cache_hit,
    }
    _notify(
        progress_callback,
        100,
        "Analiz tamamlandı!",
        f"{len(scenes)} sahne, {sum(cache_hits.values())}/{len(cache_hits)} cache hit",
        "stepExport",
    )
    return {
        "video_alias": video_alias,
        "profile": profile.name,
        "scenes_found": len(scenes),
        "story_scenes_found": int(story_result.get("story_scene_count", 0)),
        "manifest_path": str(manifest_path),
        "cache_hits": cache_hits,
    }
