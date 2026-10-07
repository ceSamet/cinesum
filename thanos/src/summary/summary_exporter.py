import os
import json
import subprocess
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from src.summary.temporal_segment_builder import build_temporal_segments, SummarySegment
from src.summary.transition_manager import get_audio_fade_filter
from src.summary.targeted_asr import refine_critical_silent_segments, verify_and_extend_speech_boundaries
from src.summary.speech_boundaries import enforce_duration_ceiling
from src.summary.samet_speech_guard import guard_and_fit_segments, load_turns
from src.selection.three_act import ACT_QUOTAS, act_overlap_durations

logger = logging.getLogger("CineSum.SummaryExporter")
logger.setLevel(logging.INFO)

def export_summary_segments(
    video_path: str,
    segments: List[Dict[str, Any]],
    output_mp4_path: str = "summary.mp4",
    category: str = "importance",
    progress_callback: Optional[Callable[[int, str, str, str], None]] = None,
) -> str:
    """
    Exports coherent SummarySegment objects using continuous source interval cutting
    and audio fade smoothing at segment boundaries.
    """
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")

    output_p = Path(output_mp4_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if not segments:
        raise ValueError("Export için hiçbir temporal segment bulunamadı.")

    # Samet's single-pass trim/atrim + concat keeps original audio and exact
    # timestamps. No per-clip mux/copy pass can silently drop the audio track.
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_p)],
        capture_output=True, text=True, check=True,
    )
    has_audio = bool(probe.stdout.strip())
    graph_parts = []
    inputs = []
    for index, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        graph_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        inputs.append(f"[v{index}]")
        if has_audio:
            graph_parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )
            inputs.append(f"[a{index}]")
    graph_parts.append(
        "".join(inputs) + f"concat=n={len(segments)}:v=1:a={int(has_audio)}[v]"
        + ("[a]" if has_audio else "")
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_p),
        "-filter_complex", ";".join(graph_parts), "-map", "[v]",
    ]
    if has_audio:
        command.extend(["-map", "[a]"])
    def _get_fast_encoder_args() -> list[str]:
        try:
            res = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=2)
            enc = res.stdout.lower()
            if "h264_nvenc" in enc:
                return ["-c:v", "h264_nvenc", "-preset", "p2", "-cq", "26"]
            elif "h264_qsv" in enc:
                return ["-c:v", "h264_qsv", "-preset", "veryfast"]
            elif "h264_videotoolbox" in enc:
                return ["-c:v", "h264_videotoolbox", "-q:v", "50"]
        except Exception as e:
            logger.warning(f"FFmpeg encoder check failed: {e}")
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "26"]

    def _run_export_with_encoder(enc_args: list[str]):
        run_cmd = list(command) + enc_args
        if has_audio:
            run_cmd.extend(["-c:a", "aac"])
        run_cmd.extend(["-movflags", "+faststart", str(output_p)])
        return subprocess.run(run_cmd, capture_output=True, text=True)

    if progress_callback:
        progress_callback(85, "Güvenli konuşma kesimleri kodlanıyor...", f"{len(segments)} aralık", "stepExport")

    primary_enc_args = _get_fast_encoder_args()
    process = _run_export_with_encoder(primary_enc_args)

    # If hardware encoder (like h264_nvenc with outdated driver) fails, seamlessly fallback to CPU libx264
    if process.returncode != 0 and "libx264" not in primary_enc_args:
        logger.warning("Hardware video encoder failed; falling back to libx264 ultrafast.")
        process = _run_export_with_encoder(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "26"])

    if process.returncode != 0 or not output_p.exists():
        raise RuntimeError(f"FFmpeg özet üretimi başarısız: {process.stderr[-1200:]}")

    if progress_callback:
        progress_callback(100, "Özet Video Hazır!", f"{output_p.name}", "stepExport")

    return str(output_p)

def export_category_summary(
    video_path: str,
    scored_scenes: List[Dict[str, Any]],
    category: str = "importance",
    target_duration_sec: float = 45.0,
    output_mp4_path: str = "summary.mp4",
    progress_callback: Optional[Callable[[int, str, str, str], None]] = None,
    narrative_mode: str = "local",
    base_dir: Optional[Path] = None,
    video_alias: Optional[str] = None,
) -> str:
    """
    Main Entrypoint for category video summary export:
    Calls build_temporal_segments to convert fine-grained shot scores into coherent temporal viewing segments,
    saves debug metadata JSON, and exports smooth MP4.
    """
    video_p = Path(video_path)
    output_p = Path(output_mp4_path)

    # 1. Build Coherent Temporal Segments
    if progress_callback:
        progress_callback(72, "Özet adayları seçiliyor...", f"{category} sahneleri değerlendiriliyor", "stepClip")
    seg_builder_res = build_temporal_segments(
        shots=scored_scenes,
        category=category,
        target_duration_sec=target_duration_sec,
        video_duration=scored_scenes[-1]["end_seconds"] if scored_scenes else None,
        narrative_mode=narrative_mode,
        base_dir=base_dir,
        video_alias=video_alias or video_p.stem,
        narrative_progress_callback=(
            (lambda pct, desc: progress_callback(pct, desc, "Opsiyonel anlatı iyileştirme", "stepClip"))
            if progress_callback else None
        ),
    )

    segments = seg_builder_res["segments"]

    detected_language = None
    audio_wav_path = None
    if base_dir is not None and video_alias:
        transcript_path = Path(base_dir) / "outputs" / "features" / "audio" / f"{video_alias}_transcript.json"
        if transcript_path.exists():
            try:
                with transcript_path.open("r", encoding="utf-8") as handle:
                    detected_language = json.load(handle).get("language")
            except (OSError, json.JSONDecodeError):
                detected_language = None
        audio_wav_path = Path(base_dir) / "outputs" / "audio" / f"{video_alias}.wav"

    # Whisper VAD can miss quiet but narratively decisive dialogue. Recheck only
    # visually critical selected windows so cached summary generation stays bounded.
    if category.lower() == "importance" and audio_wav_path is not None:
        if progress_callback:
            progress_callback(79, "Kritik sessiz anlarda konuşma kontrolü...", f"{len(segments)} aday aralık", "stepAudio")
        segments, targeted_asr = refine_critical_silent_segments(
            segments,
            scored_scenes,
            audio_wav_path=audio_wav_path,
            language=detected_language,
        )
        seg_builder_res["segments"] = segments
        seg_builder_res["actual_duration"] = round(
            sum(float(segment["duration"]) for segment in segments), 3
        )
        seg_builder_res["targeted_asr"] = targeted_asr

    # 1a. The full-video Whisper pass can mistime a word near a scene cut or
    # dense dialogue by more than the normal acoustic pad, silently clipping the
    # start/end of a sentence even in an otherwise "safe" segment. Re-verify
    # every selected segment's own edges against the real audio and extend them
    # if needed, for every category (not just importance).
    if audio_wav_path is not None:
        if progress_callback:
            progress_callback(81, "Konuşma sınırları doğrulanıyor...", f"{len(segments)} aralık", "stepAudio")
        video_duration_hint = scored_scenes[-1]["end_seconds"] if scored_scenes else None
        segments, boundary_verification = verify_and_extend_speech_boundaries(
            segments,
            audio_wav_path=audio_wav_path,
            video_duration=video_duration_hint,
            language=detected_language,
        )
        seg_builder_res["segments"] = segments
        seg_builder_res["actual_duration"] = round(
            sum(float(segment["duration"]) for segment in segments), 3
        )
        seg_builder_res["boundary_verification"] = boundary_verification

    # 1b. Final, authoritative duration cap. Budget optimization allows a small
    # internal search tolerance and later stages (adjacent re-merge, targeted ASR
    # re-expansion above) can each add a little more length; this is the single
    # place that guarantees the exported video never exceeds what the user asked
    # for, without ever cutting a segment mid-word or mid-sentence.
    transcript_path = (
        Path(base_dir) / "outputs" / "features" / "audio" / f"{video_alias or video_p.stem}_transcript.json"
        if base_dir is not None else None
    )
    if progress_callback:
        progress_callback(83, "Tam replikler hedef süreye sığdırılıyor...", f"{len(segments)} aralık", "stepAudio")
    segments, ceiling_info = guard_and_fit_segments(
        segments,
        turns=load_turns(transcript_path),
        video_duration=max((float(row["end_seconds"]) for row in scored_scenes), default=0.0),
        target_duration_sec=target_duration_sec,
    )
    seg_builder_res["segments"] = segments
    seg_builder_res["actual_duration"] = round(
        sum(float(segment["duration"]) for segment in segments), 3
    )
    seg_builder_res["duration_ceiling"] = ceiling_info

    if category.lower() == "importance" and seg_builder_res.get("distribution_mode") == "three_act":
        video_duration_hint = max((float(row["end_seconds"]) for row in scored_scenes), default=0.0)
        act_durations = act_overlap_durations(segments, video_duration_hint)
        present_roles = set()
        for segment in segments:
            present_roles.update(segment.get("hard_anchor_roles", []))
        turning = seg_builder_res.get("selection", {}).get("turning_points", {})
        expected_roles = {role for role, data in turning.items() if data.get("status") == "evidence_found"}
        unverified_roles = sorted(role for role, data in turning.items() if data.get("status") != "evidence_found")
        missing_roles = sorted(expected_roles - present_roles)
        seg_builder_res["three_act_report"] = {
            "target_quotas": ACT_QUOTAS,
            "source_time_boundaries": {"act_i_end": round(0.25 * video_duration_hint, 3), "act_ii_end": round(0.75 * video_duration_hint, 3)},
            "excluded_credit_shots": seg_builder_res.get("excluded_credit_shots", 0),
            "actual_durations": {act: round(duration, 3) for act, duration in act_durations.items()},
            "actual_quotas": {act: round(duration / max(sum(act_durations.values()), 0.001), 4) for act, duration in act_durations.items()},
            "verified_anchor_roles": sorted(expected_roles),
            "present_anchor_roles": sorted(present_roles),
            "missing_anchor_roles": missing_roles,
            "unverified_anchor_roles": unverified_roles,
            "status": "constraint_unmet" if missing_roles else "narrative_constraints_unverified" if unverified_roles else "verified_anchors_preserved",
        }
        if missing_roles:
            raise ValueError(f"Zorunlu dönüm noktaları çıktıdan düştü: {', '.join(missing_roles)}")

    if not segments:
        raise ValueError(
            "Hedef süre çok kısa: konuşma/kelime bütünlüğünü bozmadan sığdırılabilecek "
            "hiçbir segment kalmadı."
        )

    # 2. Save Debug Metadata JSON
    debug_dir = output_p.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"{video_p.stem}_{category}_{int(target_duration_sec)}s_segments.json"

    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(seg_builder_res, f, indent=2, ensure_ascii=False)

    # 3. Export Summary MP4 using Temporal Segment Exporter
    return export_summary_segments(
        video_path=video_path,
        segments=segments,
        output_mp4_path=output_mp4_path,
        category=category,
        progress_callback=progress_callback,
    )
