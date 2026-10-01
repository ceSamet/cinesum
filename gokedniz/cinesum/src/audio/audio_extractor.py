import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import librosa
import numpy as np

def extract_audio_from_video(video_path: str, output_wav_path: str) -> str:
    """
    Extract 16kHz mono WAV audio from video using FFmpeg.
    """
    video_p = Path(video_path)
    out_p = Path(output_wav_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_p),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_p)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        raise RuntimeError(f"FFmpeg ses çıkarma işlemi başarısız oldu: {stderr[-1200:]}") from exc

    if not out_p.exists() or out_p.stat().st_size == 0:
        raise RuntimeError("FFmpeg ses çıktısı oluşturamadı veya çıktı boş.")
        
    return str(out_p)

def compute_scene_audio_energy(
    audio_wav_path: str,
    scenes_data: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compute normalized RMS Audio Energy for each scene segment in a video.
    High RMS energy indicates loudness, explosions, shouting, or intense action music.
    """
    wav_p = Path(audio_wav_path)
    if not wav_p.exists():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_wav_path}")

    # Load audio signal (sr=16000)
    y, sr = librosa.load(str(wav_p), sr=16000)
    total_duration = len(y) / sr

    # Compute frame-wise Root Mean Square (RMS) energy
    rms = librosa.feature.rms(y=y)[0]
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)

    audio_features = []

    for scene in scenes_data:
        s_sec = scene["start_seconds"]
        e_sec = scene["end_seconds"]

        # Filter audio frames falling within this scene window
        mask = (times >= s_sec) & (times <= e_sec)
        scene_rms = rms[mask]
        previous_rms = rms[(times >= max(0.0, s_sec - 3.0)) & (times < s_sec)]

        if len(scene_rms) > 0:
            mean_rms = float(np.mean(scene_rms))
            max_rms = float(np.max(scene_rms))
        else:
            mean_rms = 0.0
            max_rms = 0.0

        # Relative onset after quiet audio; loudness alone is not an onset.
        quiet_reference = float(np.mean(previous_rms)) if len(previous_rms) else mean_rms
        early_peak = float(np.max(scene_rms[:max(1, len(scene_rms) // 3)])) if len(scene_rms) else 0.0
        onset_contrast = max(0.0, (early_peak - quiet_reference) / max(early_peak, 1e-6))

        audio_features.append({
            "scene_id": scene["scene_id"],
            "start_seconds": s_sec,
            "end_seconds": e_sec,
            "duration_seconds": scene["duration_seconds"],
            "mean_rms_energy": round(mean_rms, 6),
            "max_rms_energy": round(max_rms, 6),
            "audio_onset_contrast": round(onset_contrast, 4),
        })

    # Min-Max Normalize mean_rms_energy across all scenes [0.0, 1.0]
    all_means = [af["mean_rms_energy"] for af in audio_features]
    min_e = min(all_means) if all_means else 0.0
    max_e = max(all_means) if all_means else 1.0
    denom = (max_e - min_e) if (max_e - min_e) > 0 else 1.0

    for af in audio_features:
        af["normalized_audio_energy"] = round((af["mean_rms_energy"] - min_e) / denom, 4)

    return audio_features
