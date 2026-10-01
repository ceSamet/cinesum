import json
from pathlib import Path


TRANSCRIPT_FILE = Path("output/transcript.json")
DIARIZATION_FILE = Path("output/diarization.json")
OUTPUT_JSON = Path("output/speaker_transcript.json")
OUTPUT_TEXT = Path("output/speaker_transcript.txt")


def overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Return the duration shared by two time intervals."""
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def find_speaker(transcript_segment: dict, diarization_segments: list[dict]) -> str:
    """Pick the speaker whose turns overlap the transcript segment the most."""
    scores: dict[str, float] = {}

    for turn in diarization_segments:
        duration = overlap(
            transcript_segment["start"],
            transcript_segment["end"],
            turn["start"],
            turn["end"],
        )
        if duration > 0:
            speaker = turn["speaker"]
            scores[speaker] = scores.get(speaker, 0.0) + duration

    if scores:
        return max(scores, key=scores.get)

    # Çok kısa boşluklarda en yakın konuşma sırasını kullan.
    midpoint = (transcript_segment["start"] + transcript_segment["end"]) / 2
    if diarization_segments:
        nearest = min(
            diarization_segments,
            key=lambda turn: min(
                abs(midpoint - turn["start"]),
                abs(midpoint - turn["end"]),
            ),
        )
        return nearest["speaker"]

    return "UNKNOWN"


def merge_word_assignments(words: list[dict], diarization_segments: list[dict]) -> list[dict]:
    """Assign each word separately, then join consecutive words by speaker."""
    merged: list[dict] = []

    for word in words:
        speaker = find_speaker(word, diarization_segments)

        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["end"] = word["end"]
            merged[-1]["text"] += " " + word["text"]
        else:
            merged.append(
                {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": speaker,
                    "text": word["text"],
                }
            )

    return merged


def main() -> None:
    if not TRANSCRIPT_FILE.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {TRANSCRIPT_FILE}")
    if not DIARIZATION_FILE.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {DIARIZATION_FILE}")

    with TRANSCRIPT_FILE.open(encoding="utf-8") as file:
        transcript = json.load(file)
    with DIARIZATION_FILE.open(encoding="utf-8") as file:
        diarization = json.load(file)

    diarization_segments = diarization["segments"]
    combined_segments = []

    all_words = [
        word
        for segment in transcript["segments"]
        for word in segment.get("words", [])
    ]

    if all_words:
        combined_segments = merge_word_assignments(all_words, diarization_segments)
    else:
        print(
            "Uyarı: transcript.json kelime zamanları içermiyor. "
            "Önce transcribe.py dosyasını yeniden çalıştırın."
        )
        for segment in transcript["segments"]:
            combined_segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": find_speaker(segment, diarization_segments),
                    "text": segment["text"],
                }
            )

    output = {
        "input_file": diarization.get("input_file"),
        "language": transcript.get("language"),
        "speaker_count": diarization.get("speaker_count"),
        "segments": combined_segments,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    with OUTPUT_TEXT.open("w", encoding="utf-8") as file:
        for segment in combined_segments:
            line = (
                f"[{segment['start']:7.2f} - {segment['end']:7.2f}] "
                f"{segment['speaker']}: {segment['text']}"
            )
            print(line)
            file.write(line + "\n")

    print(f"\nBirleşik JSON kaydedildi: {OUTPUT_JSON}")
    print(f"Okunabilir metin kaydedildi: {OUTPUT_TEXT}")


if __name__ == "__main__":
    main()
