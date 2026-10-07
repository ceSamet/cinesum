import json
import os
from pathlib import Path

from pyannote.audio import Pipeline


INPUT_FILE = Path("input/test.mp4")
OUTPUT_FILE = Path("output/diarization.json")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {INPUT_FILE}")

    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN bulunamadı. PowerShell'de şu komutu çalıştır:\n"
            '$env:HF_TOKEN="hf_..."'
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Pyannote modeli yükleniyor...")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )
    pipeline.instantiate({"clustering": {"threshold": 0.50}})

    print("Videonun sesi belleğe alınıyor...")

    # torchcodec sorunlarını aşmak ve diske dosya yazmamak için
    # FFmpeg ile MP4'ün sesini doğrudan belleğe (RAM) 16kHz mono olarak alıyoruz.
    import subprocess
    import numpy as np
    import torch
    
    ffmpeg_exe = r".venv\Scripts\ffmpeg.exe"
    raw_audio = subprocess.run([
        ffmpeg_exe, "-i", str(INPUT_FILE),
        "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"
    ], capture_output=True, check=True).stdout
    
    samples = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)
    
    audio_in_memory = {"waveform": waveform, "sample_rate": 16000, "uri": str(INPUT_FILE)}

    print("Konuşmacı ayrımı başlatılıyor...")

    result = pipeline(audio_in_memory)

    segments = []

    # Eşzamanlı konuşmaları kaybetmemek için exclusive olmayan çıktıyı kullan.
    annotation = result.speaker_diarization

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        item = {
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": speaker,
        }

        segments.append(item)

        print(
            f"[{item['start']:7.2f} - {item['end']:7.2f}] "
            f"{item['speaker']}"
        )

    speaker_count = len(
        {segment["speaker"] for segment in segments}
    )

    output = {
        "input_file": str(INPUT_FILE),
        "speaker_count": speaker_count,
        "segments": segments,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nBulunan konuşmacı sayısı: {speaker_count}")
    print(f"Sonuç kaydedildi: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
