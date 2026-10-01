import json
from pathlib import Path

from faster_whisper import WhisperModel


INPUT_FILE = Path("input/test.mp4")
OUTPUT_FILE = Path("output/transcript.json")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        str(INPUT_FILE),
        language="en",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )

    results = []

    for segment in segments:
        item = {
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "words": [
                {
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "text": word.word.strip(),
                }
                for word in (segment.words or [])
            ],
        }

        results.append(item)

        print(
            f"[{item['start']:7.2f} - {item['end']:7.2f}] "
            f"{item['text']}"
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nTranskript kaydedildi: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
