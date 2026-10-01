from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

# Windows'ta symlink uyarısı vermemesi için:
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Proje sesi FFmpeg ile kendisi yüklediği için pyannote'ın opsiyonel TorchCodec
# çözücüsüne ihtiyaç duymaz. Uzun ve yanıltıcı başlangıç uyarılarını gizle.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io")
warnings.filterwarnings(
    "ignore",
    message=r"std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
    module=r"pyannote\.audio\.models\.blocks\.pooling",
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)

from huggingface_hub import get_token


SAMPLE_RATE = 16_000
PROJECT_ROOT = Path(__file__).resolve().parent
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
EVENT_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

# AudioSet modelinin İngilizce etiketlerini kullanıcı dostu İngilizce adlara çevirir.
EVENT_LABELS = {
    "chink, clink": "Glass clink",
    "glass": "Glass sound",
    "cutlery": "Cutlery",
    "silverware": "Cutlery",
    "dishes": "Dishes",
    "pots, and pans": "Dishes",
    "laughter": "Laughter",
    "giggle": "Giggle",
    "chuckle": "Chuckle",
    "crying": "Crying",
    "sobbing": "Sobbing",
    "cough": "Cough",
    "sneeze": "Sneeze",
    "applause": "Applause",
    "clapping": "Clapping",
    "music": "Music",
    "siren": "Siren",
    "alarm": "Alarm",
    "vehicle": "Vehicle",
    "car": "Car",
    "motorcycle": "Motorcycle",
    "footsteps": "Footsteps",
    "typing": "Typing",
    "door": "Door",
    "knock": "Knock",
    "explosion": "Explosion",
    "gunshot": "Gunshot",
    "wind": "Wind",
    "rain": "Rain",
    "thunder": "Thunder",
    "noise": "Noise",
}

VOCAL_SOUND_EVENTS = {"Laughter", "Giggle", "Chuckle", "Crying", "Sobbing", "Cough", "Sneeze"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Videoyu tek seferde yazıya çevirir, konuşmacıları ve dış sesleri ayırır."
    )
    parser.add_argument("video", type=Path, help="İşlenecek video dosyasının yolu")
    parser.add_argument("--language", default="en", help="Konuşma dili (varsayılan: en)")
    parser.add_argument(
        "--whisper-model", default="medium", help="Whisper modeli (varsayılan: medium)"
    )
    parser.add_argument(
        "--event-threshold",
        type=float,
        default=0.35,
        help="Dış ses algılama eşiği, 0-1 (varsayılan: 0.35)",
    )
    parser.add_argument(
        "--skip-events", action="store_true", help="Dış ses algılamasını çalıştırma"
    )
    parser.add_argument(
        "--recompute-events",
        action="store_true",
        help="Varsa önbelleği kullanmadan dış sesleri yeniden tara",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Uzun videolar için daha küçük Whisper modeli ve seyrek ortam sesi taraması",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        choices=("default", "float16", "int8_float16", "int8"),
        help="Model hesaplama hassasiyeti. CUDA bellek hatası alıyorsanız int8_float16 veya int8 deneyin (varsayılan: default)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="İşlem aygıtı (varsayılan: varsa cuda, yoksa cpu)",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=None,
        help="İsteğe bağlı kesin kişi sayısı; verilmezse tamamen otomatik belirlenir",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Otomatik belirleme için minimum kişi sayısı (örn: 3)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Otomatik belirleme için maksimum kişi sayısı",
    )
    parser.add_argument(
        "--speaker-threshold",
        type=float,
        default=0.40,
        help=(
            "Speaker clustering threshold, 0.3-0.8. Lower values separate "
            "similar voices more readily (default: 0.40)"
        ),
    )
    parser.add_argument(
        "--no-speaker-correction",
        action="store_true",
        help="Embedding tabanlı konservatif doğrulama katmanını kapat",
    )
    parser.add_argument(
        "--disable-intra-speaker-split",
        action="store_true",
        help="Aynı hoparlör etiketindeki farklı sesleri bölmeye çalışmayı devre dışı bırakır",
    )
    parser.add_argument("--speaker-merge-threshold", type=float, default=0.88)
    parser.add_argument("--reassign-similarity", type=float, default=0.72)
    parser.add_argument("--reassign-margin", type=float, default=0.15)
    parser.add_argument(
        "--speaker-overrides",
        type=Path,
        default=None,
        help=(
            "Konuşmacı düzeltme JSON dosyası. Verilmezse "
            "output/<video>/speaker_overrides.json otomatik kullanılır"
        ),
    )
    return parser.parse_args()


def find_ffmpeg() -> Path:
    configured = os.getenv("FFMPEG_PATH")
    if configured and Path(configured).is_file():
        return Path(configured)

    candidates = [
        PROJECT_ROOT / ".venv/Scripts/ffmpeg.exe",
        Path.home()
        / "Downloads/ffmpeg-8.1.2-full_build-shared/ffmpeg-8.1.2-full_build-shared/bin/ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        matches = list(downloads.glob("ffmpeg-*-full_build-shared/**/bin/ffmpeg.exe"))
        if matches:
            return matches[0].resolve()

    executable = shutil.which("ffmpeg")
    if executable:
        return Path(executable)

    raise FileNotFoundError(
        "FFmpeg bulunamadı. FFMPEG_PATH değişkenini ffmpeg.exe yoluna ayarlayın."
    )


def load_audio(video: Path, ffmpeg: Path) -> np.ndarray:
    import numpy as np

    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-",
    ]
    raw_audio = subprocess.run(command, capture_output=True, check=True).stdout
    if not raw_audio:
        raise RuntimeError("Videoda okunabilir bir ses kanalı bulunamadı.")
    return np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0


def diarize(
    audio: np.ndarray,
    video: Path,
    token: str,
    speaker_count: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    clustering_threshold: float = 0.50,
    device: str = "cpu",
    apply_correction: bool = True,
    merge_threshold: float = 0.88,
    reassign_similarity: float = 0.72,
    reassign_margin: float = 0.15,
    enable_intra_speaker_split: bool = False,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    import torch
    from pyannote.audio import Pipeline
    from speaker_correction import correct_diarization

    print("\n[1/4] Diarizing speakers...")
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except Exception as error:
        message = str(error).casefold()
        if any(word in message for word in ("connection", "proxy", "max retries", "offline")):
            raise RuntimeError(
                "Konuşmacı modeli indirilemedi. İnternet bağlantınızı kontrol edip "
                "`hf auth login` komutunu bir kez çalıştırın; ardından tekrar deneyin."
            ) from error
        raise RuntimeError(
            "Konuşmacı modeli yüklenemedi. Hugging Face hesabınızda "
            f"{DIARIZATION_MODEL} modelinin kullanım koşullarını kabul ettiğinizden emin olun."
        ) from error
    pipeline.to(torch.device(device))
    # The model default (0.60) can merge speakers with similar voices.
    # A lower threshold makes creation of separate clusters easier.
    pipeline.instantiate({"clustering": {"threshold": clustering_threshold}})
    audio_input = {
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": SAMPLE_RATE,
        "uri": str(video),
    }

    def collect_annotation(annotation) -> list[dict]:
        collected = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            collected.append(
                {
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker,
                }
            )
        return collected

    if speaker_count is not None:
        result = pipeline(audio_input, num_speakers=speaker_count)
    else:
        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        result = pipeline(audio_input, **kwargs)
    regular = getattr(result, "speaker_diarization", result)
    exclusive = getattr(result, "exclusive_speaker_diarization", regular)
    raw_turns = collect_annotation(exclusive)
    overlaps = find_overlap_regions(collect_annotation(regular))
    if apply_correction:
        corrected_turns, correction_report = correct_diarization(
            audio,
            raw_turns,
            overlaps,
            pipeline._embedding,
            merge_threshold=merge_threshold,
            reassign_similarity=reassign_similarity,
            reassign_margin=reassign_margin,
            enable_intra_speaker_split=enable_intra_speaker_split,
        )
    else:
        corrected_turns = [turn.copy() for turn in raw_turns]
        correction_report = {
            "enabled": False,
            "raw_speaker_count": len({turn["speaker"] for turn in raw_turns}),
            "final_speaker_count": len({turn["speaker"] for turn in raw_turns}),
        }
    return raw_turns, corrected_turns, overlaps, correction_report


def find_overlap_regions(turns: list[dict], min_duration: float = 0.30) -> list[dict]:
    """Build stable overlap regions without assigning transcript words to two people."""
    boundaries = sorted({point for turn in turns for point in (turn["start"], turn["end"])})
    regions: list[dict] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start <= 0:
            continue
        midpoint = (start + end) / 2
        speakers = sorted(
            {
                turn["speaker"]
                for turn in turns
                if turn["start"] <= midpoint < turn["end"]
            }
        )
        if len(speakers) < 2:
            continue
        if regions and regions[-1]["speakers"] == speakers and start <= regions[-1]["end"] + 0.02:
            regions[-1]["end"] = end
        else:
            regions.append(
                {
                    "start": start,
                    "end": end,
                    "type": "overlap",
                    "speakers": speakers,
                }
            )
    return [region for region in regions if region["end"] - region["start"] >= min_duration]


def smooth_turns(turns: list[dict], max_glitch_duration: float = 0.20) -> list[dict]:
    """Remove tiny speaker flips between two turns of the same speaker."""
    if len(turns) < 3:
        return turns

    smoothed = [turn.copy() for turn in sorted(turns, key=lambda item: item["start"])]
    for index in range(1, len(smoothed) - 1):
        previous, current, following = smoothed[index - 1 : index + 2]
        if (
            current["end"] - current["start"] <= max_glitch_duration
            and previous["speaker"] == following["speaker"]
            and current["speaker"] != previous["speaker"]
            and current["start"] - previous["end"] <= 0.05
            and following["start"] - current["end"] <= 0.05
        ):
            current["speaker"] = previous["speaker"]

    merged: list[dict] = []
    for turn in smoothed:
        if (
            merged
            and merged[-1]["speaker"] == turn["speaker"]
            and turn["start"] - merged[-1]["end"] <= 0.05
        ):
            merged[-1]["end"] = turn["end"]
        else:
            merged.append(turn)
    return merged


def transcribe(
    video: Path, language: str, model_name: str, device: str, fast: bool, compute_type_arg: str = "default"
) -> tuple[list[dict], str]:
    from faster_whisper import WhisperModel

    print("[2/4] Transcribing speech...")
    if compute_type_arg == "default":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = compute_type_arg
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(video),
        language=language,
        beam_size=1 if fast else 5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=True,
    )

    results = []
    for segment in segments:
        results.append(
            {
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
        )
    return results, info.language


def interval_overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def find_speaker(item: dict, turns: list[dict]) -> str:
    scores: dict[str, float] = {}
    for turn in turns:
        duration = interval_overlap(item, turn)
        if duration:
            speaker = turn["speaker"]
            scores[speaker] = scores.get(speaker, 0.0) + duration
    if scores:
        return max(scores, key=scores.get)

    if turns:
        midpoint = (item["start"] + item["end"]) / 2
        return min(
            turns,
            key=lambda turn: min(
                abs(midpoint - turn["start"]), abs(midpoint - turn["end"])
            ),
        )["speaker"]
    return "UNKNOWN"


def override_speaker(item: dict, overrides: list[dict]) -> str | None:
    """Return an explicit speaker correction covering the item's midpoint."""
    midpoint = (item["start"] + item["end"]) / 2
    matches = [
        override
        for override in overrides
        if override["start"] <= midpoint <= override["end"]
    ]
    if not matches:
        return None
    return min(matches, key=lambda override: override["end"] - override["start"])[
        "speaker"
    ]


def load_speaker_overrides(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    overrides = data.get("overrides", data) if isinstance(data, dict) else data
    if not isinstance(overrides, list):
        raise ValueError(f"Geçersiz konuşmacı düzeltme dosyası: {path}")
    for item in overrides:
        if not all(key in item for key in ("start", "end", "speaker")):
            raise ValueError(f"Eksik konuşmacı düzeltme alanı: {item}")
        if item["start"] >= item["end"]:
            raise ValueError(f"Geçersiz konuşmacı düzeltme aralığı: {item}")
    return overrides


def assign_word_speakers(
    words: list[dict],
    turns: list[dict],
    overrides: list[dict] | None = None,
    switch_penalty: float = 0.35,
) -> list[str]:
    """Assign a whole Whisper segment while discouraging one-word boundary flips."""
    if not words:
        return []
    overrides = overrides or []
    speakers = sorted(
        {turn["speaker"] for turn in turns}
        | {item["speaker"] for item in overrides}
    )
    if not speakers:
        return ["UNKNOWN"] * len(words)

    emissions: list[dict[str, float]] = []
    for word in words:
        forced = override_speaker(word, overrides)
        duration = max(0.01, word["end"] - word["start"])
        scores = {
            speaker: sum(
                interval_overlap(word, turn)
                for turn in turns
                if turn["speaker"] == speaker
            )
            / duration
            for speaker in speakers
        }
        if forced:
            scores = {
                speaker: 1.0 if speaker == forced else -1_000_000.0
                for speaker in speakers
            }
        elif max(scores.values(), default=0.0) == 0:
            nearest = find_speaker(word, turns)
            if nearest in scores:
                scores[nearest] = 0.1
        emissions.append(scores)

    paths = {speaker: (emissions[0][speaker], [speaker]) for speaker in speakers}
    for scores in emissions[1:]:
        updated = {}
        for speaker in speakers:
            best_score, best_path = max(
                (
                    score + scores[speaker] - (switch_penalty if previous != speaker else 0),
                    path,
                )
                for previous, (score, path) in paths.items()
            )
            updated[speaker] = (best_score, best_path + [speaker])
        paths = updated
    return max(paths.values(), key=lambda item: item[0])[1]


def build_speaker_transcript(
    transcript: list[dict], turns: list[dict], overrides: list[dict] | None = None
) -> list[dict]:
    overrides = overrides or []
    combined: list[dict] = []

    for segment in transcript:
        words = segment.get("words", [])
        if not words:
            # Bazı kısa seslerde kelime zamanı gelmez; segmenti yine koru.
            text = segment.get("text", "").strip()
            if not text:
                continue
            combined.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": override_speaker(segment, overrides)
                    or find_speaker(segment, turns),
                    "text": text,
                    "type": "speech",
                }
            )
            continue

        # Aynı Whisper cümlesi içinde yalnızca konuşmacı değişiminde böl.
        segment_parts: list[dict] = []
        assigned_speakers = assign_word_speakers(words, turns, overrides)
        for word, speaker in zip(words, assigned_speakers):
            if segment_parts and segment_parts[-1]["speaker"] == speaker:
                segment_parts[-1]["end"] = word["end"]
                segment_parts[-1]["text"] += " " + word["text"]
            else:
                segment_parts.append(
                    {
                        "start": word["start"],
                        "end": word["end"],
                        "speaker": speaker,
                        "text": word["text"],
                        "type": "speech",
                    }
                )
        combined.extend(segment_parts)

    # Whisper aynı konuşmayı birkaç cümle/segment halinde döndürebilir.
    # Arada başka bir konuşmacı yoksa bunları tek konuşma bloğu olarak göster.
    merged: list[dict] = []
    for item in combined:
        if merged and merged[-1]["speaker"] == item["speaker"]:
            merged[-1]["end"] = item["end"]
            merged[-1]["text"] += " " + item["text"]
        else:
            merged.append(item.copy())
    return merged


def add_untranscribed_speech(
    speaker_transcript: list[dict], turns: list[dict], min_duration: float = 0.50
) -> list[dict]:
    """Keep diarized speakers visible when Whisper produces no words for a turn.
    Disabled per user preference: we no longer output '[Speech detected, unintelligible]'
    and instead completely ignore untranscribed segments.
    """
    return list(speaker_transcript)


def update_overlap_speakers(overlaps: list[dict], turns: list[dict]) -> list[dict]:
    """Propagate embedding-based speaker splits to overlap labels."""
    updated = []
    for region in overlaps:
        speakers = []
        for original in region["speakers"]:
            candidates = [
                turn
                for turn in turns
                if turn["speaker"] == original
                or turn.get("original_speaker") == original
            ]
            if candidates:
                overlapping = [
                    turn for turn in candidates if interval_overlap(region, turn) > 0
                ]
                if overlapping:
                    best = max(
                        overlapping, key=lambda turn: interval_overlap(region, turn)
                    )
                else:
                    # Exclusive diarization, overlap anında bu kişinin turunu
                    # kaldırabilir. Eski etiketi bırakmak yerine zamansal olarak
                    # en yakın düzeltilmiş turdaki yeni kimliği kullan.
                    midpoint = (region["start"] + region["end"]) / 2
                    best = min(
                        candidates,
                        key=lambda turn: min(
                            abs(midpoint - turn["start"]),
                            abs(midpoint - turn["end"]),
                        ),
                    )
                speaker = best["speaker"]
            else:
                speaker = original
            if speaker not in speakers:
                speakers.append(speaker)
        item = region.copy()
        item["speakers"] = sorted(speakers)
        if len(item["speakers"]) > 1:
            updated.append(item)
    return updated


def event_name(label: str) -> str | None:
    lowered = label.casefold()
    for keyword, translated in EVENT_LABELS.items():
        if keyword in lowered:
            return translated
    return None


def merge_events(events: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for event in sorted(events, key=lambda item: (item["label"], item["start"])):
        if (
            merged
            and merged[-1]["label"] == event["label"]
            and event["start"] <= merged[-1]["end"] + 0.1
        ):
            merged[-1]["end"] = max(merged[-1]["end"], event["end"])
            merged[-1]["score"] = max(merged[-1]["score"], event["score"])
        else:
            merged.append(event.copy())
    return sorted(merged, key=lambda item: item["start"])


def reconcile_overlaps(overlaps: list[dict], events: list[dict]) -> list[dict]:
    """Let confident non-speech vocal events override false speaker overlaps."""
    cleaned = []
    for overlap in overlaps:
        overlap_duration = max(0.01, overlap["end"] - overlap["start"])
        is_vocal_event = any(
            event["label"] in VOCAL_SOUND_EVENTS
            and event["score"] >= 0.30
            and interval_overlap(overlap, event) / overlap_duration >= 0.40
            for event in events
        )
        if not is_vocal_event:
            cleaned.append(overlap)
    return cleaned


def detect_events(
    audio: np.ndarray, threshold: float, device: str = "cpu", fast: bool = False
) -> list[dict]:
    import numpy as np
    import torch

    print("[3/4] Searching for laughter, noise and other sound events...")
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    extractor = AutoFeatureExtractor.from_pretrained(EVENT_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(EVENT_MODEL)
    model.to(device)
    model.eval()

    # Kısa darbeleri (kadeh, tabak, kapı vb.) çevredeki müzikten ayırabilmek
    # için AudioSet'in 10 saniyelik varsayımı yerine yerel pencereler kullan.
    window_seconds = 3.0 if fast else 2.0
    hop_seconds = 2.0 if fast else 0.5
    window_size = int(window_seconds * SAMPLE_RATE)
    hop_size = int(hop_seconds * SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    events = []
    pending: list[tuple[int, np.ndarray]] = []
    batch_size = 16 if device == "cuda" or fast else 8

    def process_batch(batch: list[tuple[int, np.ndarray]]) -> None:
        if not batch:
            return
        inputs = extractor(
            [chunk for _, chunk in batch],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            batch_scores = torch.sigmoid(model(**inputs).logits).cpu()

        for (start_sample, _), scores in zip(batch, batch_scores):
            add_event_candidates(start_sample, scores)

    def add_event_candidates(start_sample: int, scores: torch.Tensor) -> None:
        # Aynı pencere içinde en güçlü olayları al; konuşma etiketlerini dışarıda bırak.
        top_indices = torch.topk(scores, k=min(20, len(scores))).indices.tolist()
        candidates = []
        for index in top_indices:
            score = float(scores[index])
            label = model.config.id2label[index]
            translated = event_name(label)
            if translated in VOCAL_SOUND_EVENTS:
                effective_threshold = max(0.20, threshold - 0.10)
            elif translated == "Music":
                effective_threshold = threshold + 0.10
            else:
                effective_threshold = threshold
            if translated is None or score < effective_threshold:
                continue
            candidates.append((score, translated, label))

        specific = [candidate for candidate in candidates if candidate[1] != "Music"]
        selected = specific[:2] if specific else candidates[:1]
        for score, translated, label in selected:
            events.append(
                {
                    "start": round(start_sample / SAMPLE_RATE, 2),
                    "end": round(min(duration, start_sample / SAMPLE_RATE + window_seconds), 2),
                    "type": "sound_event",
                    "label": translated,
                    "source_label": label,
                    "score": round(score, 3),
                }
            )

    for start_sample in range(0, len(audio), hop_size):
        chunk = audio[start_sample : start_sample + window_size]
        if len(chunk) < SAMPLE_RATE // 2 or float(np.sqrt(np.mean(chunk**2))) < 0.003:
            continue

        pending.append((start_sample, chunk))
        if len(pending) == batch_size:
            process_batch(pending)
            pending.clear()

    process_batch(pending)

    return merge_events(events)


def save_outputs(
    output_dir: Path,
    video: Path,
    language: str,
    turns: list[dict],
    transcript: list[dict],
    speaker_transcript: list[dict],
    events: list[dict],
    overlaps: list[dict],
    correction_report: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_count = len({turn["speaker"] for turn in turns})
    timeline = sorted(speaker_transcript + events + overlaps, key=lambda item: item["start"])

    files = {
        "diarization.json": {
            "speaker_count": speaker_count,
            "segments": turns,
            "overlaps": overlaps,
            "speaker_correction": correction_report,
        },
        "transcript.json": {"language": language, "segments": transcript},
        "result.json": {
            "input_file": str(video),
            "language": language,
            "speaker_count": speaker_count,
            "segments": speaker_transcript,
            "sound_events": events,
            "overlaps": overlaps,
            "speaker_correction": correction_report,
            "timeline": timeline,
        },
    }
    for filename, data in files.items():
        with (output_dir / filename).open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    with (output_dir / "result.txt").open("w", encoding="utf-8") as file:
        for item in timeline:
            if item["type"] == "speech":
                line = (
                    f"[{item['start']:7.2f} - {item['end']:7.2f}] "
                    f"{item['speaker']}: {item['text']}"
                )
            elif item["type"] == "sound_event":
                line = (
                    f"[{item['start']:7.2f} - {item['end']:7.2f}] "
                    f"[SOUND EVENT: {item['label']}] ({item['score']:.2f})"
                )
            else:
                line = (
                    f"[{item['start']:7.2f} - {item['end']:7.2f}] "
                    f"[OVERLAPPING SPEECH: {' + '.join(item['speakers'])}]"
                )
            print(line)
            file.write(line + "\n")


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def event_cache_key(video: Path, threshold: float, fast: bool) -> dict:
    stat = video.stat()
    return {
        "input_size": stat.st_size,
        "input_mtime_ns": stat.st_mtime_ns,
        "threshold": threshold,
        "fast": fast,
        "event_model": EVENT_MODEL,
        "detector_version": 2,
    }


def load_cached_events(path: Path, cache_key: dict) -> list[dict] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if data.get("cache_key") == cache_key:
            return data.get("events", [])
    except (OSError, json.JSONDecodeError):
        pass
    return None


def main() -> None:
    args = parse_args()
    import torch

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA istendi ancak CUDA destekli PyTorch kurulu değil. "
            "--device cpu kullanın veya CUDA sürümünü kurun."
        )
    whisper_model = "small" if args.fast and args.whisper_model == "medium" else args.whisper_model
    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video bulunamadı: {video}")
    if not 0 < args.event_threshold < 1:
        raise ValueError("--event-threshold 0 ile 1 arasında olmalıdır.")
    if args.speakers is not None and args.speakers < 1:
        raise ValueError("--speakers en az 1 olmalıdır.")
    if not 0.3 <= args.speaker_threshold <= 0.8:
        raise ValueError("--speaker-threshold 0.3 ile 0.8 arasında olmalıdır.")
    if not 0.75 <= args.speaker_merge_threshold <= 0.99:
        raise ValueError("--speaker-merge-threshold 0.75 ile 0.99 arasında olmalıdır.")
    if not 0.5 <= args.reassign_similarity <= 0.99:
        raise ValueError("--reassign-similarity 0.5 ile 0.99 arasında olmalıdır.")
    if not 0.05 <= args.reassign_margin <= 0.5:
        raise ValueError("--reassign-margin 0.05 ile 0.5 arasında olmalıdır.")
    if args.min_speakers is not None and args.min_speakers < 1:
        raise ValueError("--min-speakers en az 1 olmalıdır.")
    if args.max_speakers is not None and args.max_speakers < 1:
        raise ValueError("--max-speakers en az 1 olmalıdır.")
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        raise ValueError("--min-speakers, --max-speakers değerinden büyük olamaz.")

    # Önce ortam değişkenini, yoksa daha önce yapılmış `hf auth login` oturumunu kullan.
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face oturumu bulunamadı. Bir kez `hf auth login` çalıştırın."
        )

    output_dir = PROJECT_ROOT / "output" / video.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    print(f"Video: {video}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Device: {device} | Whisper: {whisper_model}")
    print("Loading audio into memory...")
    audio = load_audio(video, ffmpeg)

    raw_turns, turns, overlaps, correction_report = diarize(
        audio,
        video,
        token,
        args.speakers,
        args.min_speakers,
        args.max_speakers,
        args.speaker_threshold,
        device,
        not args.no_speaker_correction,
        args.speaker_merge_threshold,
        args.reassign_similarity,
        args.reassign_margin,
        not args.disable_intra_speaker_split,
    )
    speaker_count = len({turn["speaker"] for turn in turns})
    save_json(
        output_dir / "diarization_raw.json",
        {
            "speaker_count": len({turn["speaker"] for turn in raw_turns}),
            "segments": raw_turns,
            "overlaps": overlaps,
        },
    )
    save_json(
        output_dir / "diarization.json",
        {"speaker_count": speaker_count, "segments": turns, "overlaps": overlaps},
    )
    save_json(output_dir / "speaker_correction_report.json", correction_report)
    print(f"    Saved: {output_dir / 'diarization.json'}")

    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    transcript, detected_language = transcribe(
        video, args.language, whisper_model, device, args.fast, args.compute_type
    )
    save_json(
        output_dir / "transcript.json",
        {"language": detected_language, "segments": transcript},
    )
    print(f"    Saved: {output_dir / 'transcript.json'}")

    overrides_path = (
        args.speaker_overrides.expanduser().resolve()
        if args.speaker_overrides
        else output_dir / "speaker_overrides.json"
    )
    speaker_overrides = load_speaker_overrides(overrides_path)
    if speaker_overrides:
        print(f"    Speaker overrides: {overrides_path}")
    speaker_transcript = build_speaker_transcript(
        transcript, turns, speaker_overrides
    )
    speaker_transcript = add_untranscribed_speech(speaker_transcript, turns)
    events_path = output_dir / "sound_events.json"
    cache_key = event_cache_key(video, args.event_threshold, args.fast)
    if args.skip_events:
        events = []
    else:
        events = None if args.recompute_events else load_cached_events(events_path, cache_key)
        if events is not None:
            print("[3/4] Sound events loaded from cache.")
        else:
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            events = detect_events(audio, args.event_threshold, device, args.fast)
    overlaps = update_overlap_speakers(overlaps, turns)
    overlaps = reconcile_overlaps(overlaps, events)
    # Dış ses analizi sonrasında kahkaha/öksürük kaynaklı sahte overlap'ler
    # temizlendiği için diarization dosyasını nihai haliyle güncelle.
    save_json(
        output_dir / "diarization.json",
        {"speaker_count": speaker_count, "segments": turns, "overlaps": overlaps},
    )
    save_json(events_path, {"cache_key": cache_key, "events": events})
    print(f"    Saved: {output_dir / 'sound_events.json'}")

    print("[4/4] Saving results...\n")
    save_outputs(
        output_dir,
        video,
        detected_language,
        turns,
        transcript,
        speaker_transcript,
        events,
        overlaps,
        correction_report,
    )
    print(
        f"Speaker correction: {correction_report.get('raw_speaker_count')} raw -> "
        f"{correction_report.get('final_speaker_count')} final, "
        f"{correction_report.get('reassigned_segments', 0)} segments reassigned\n"
    )
    print(f"\nFound speaker count: {speaker_count}")
    print(f"Found sound event count: {len(events)}")
    print(f"Completed: {(output_dir / 'result.txt').resolve()}")


def cli() -> None:
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nHata: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    cli()
