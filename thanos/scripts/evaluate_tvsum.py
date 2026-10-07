#!/usr/bin/env python3
"""Evaluate actual Thanos summaries against the official TVSum user annotations.

The TVSum release contains importance ratings, not pre-rendered summaries.  This
script derives one 15%-budget gold summary per annotator, as in the official
MATLAB evaluation example, then compares the source-time intervals selected by
Thanos with those gold summaries at frame level.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
THANOS = ROOT / "thanos"
DEFAULT_TVSUM = ROOT / "gokedniz" / "cinesum" / "dataset"
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov")
BUDGET_PORTION = 0.15
GOLD_SHOT_FRAMES = 60  # The official MATLAB evaluator uses 60-frame gold shots.


def load_catalog(info_path: Path) -> list[dict[str, str]]:
    with info_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 50 or len({row["video_id"] for row in rows}) != 50:
        raise ValueError(f"TVSum katalogu 50 benzersiz video içermeli: {info_path}")
    for index, row in enumerate(rows, 1):
        row["alias"] = f"video{index}"
    return rows


def load_annotation_lines(path: Path) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 3:
                raise ValueError(f"Geçersiz TVSum anotasyon satırı: {row[:2]}")
            annotations[row[0]].append(row[2])
    if len(annotations) != 50 or any(len(lines) != 20 for lines in annotations.values()):
        raise ValueError("TVSum anotasyonları 50 video × 20 değerlendirici olmalı")
    return dict(annotations)


def parse_ratings(lines: list[str]) -> list[list[int]]:
    ratings = [[int(value) for value in line.split(",")] for line in lines]
    lengths = {len(row) for row in ratings}
    if len(ratings) != 20 or len(lengths) != 1 or not all(1 <= v <= 5 for row in ratings for v in row):
        raise ValueError("TVSum kullanıcı puanları 20 eşit uzunlukta 1–5 vektör olmalı")
    return ratings


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Video süresi okunamadı: {path}")
    return duration


def catalog_duration(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    return parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]


def find_video(video_dir: Path, row: dict[str, str]) -> tuple[Path | None, float | None, str | None]:
    """Resolve by YouTube ID or historical alias; reject alias collisions by duration."""
    rejected: list[str] = []
    for stem in (row["video_id"], row["alias"]):
        for extension in VIDEO_EXTENSIONS:
            candidate = video_dir / f"{stem}{extension}"
            if not candidate.is_file():
                continue
            try:
                duration = video_duration(candidate)
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                rejected.append(f"{candidate.name}: ffprobe başarısız ({exc})")
                continue
            expected = catalog_duration(row["length"])
            tolerance = max(8.0, expected * 0.03)
            if abs(duration - expected) <= tolerance:
                return candidate, duration, None
            rejected.append(f"{candidate.name}: {duration:.1f} sn; TVSum beklenen ~{expected} sn")
    return None, None, "; ".join(rejected) if rejected else None


def gold_mask(ratings: list[int], portion: float = BUDGET_PORTION) -> bytearray:
    """Exact 0/1 knapsack for 60-frame equal-weight shots plus the final short shot.

    The official MATLAB code maximizes the *mean* score per shot, not the sum
    over its frames. With equal full-shot weights, sorting is the exact optimum;
    only including/excluding the final shorter shot needs a second comparison.
    """
    nframes = len(ratings)
    budget = int(portion * nframes)
    shots = []
    for start in range(0, nframes, GOLD_SHOT_FRAMES):
        end = min(nframes, start + GOLD_SHOT_FRAMES)
        shots.append((start, end, sum(ratings[start:end]) / (end - start)))
    full = sorted((shot for shot in shots if shot[1] - shot[0] == GOLD_SHOT_FRAMES), key=lambda x: (-x[2], x[0]))
    tail = shots[-1] if shots and shots[-1][1] - shots[-1][0] < GOLD_SHOT_FRAMES else None
    full_count = min(len(full), budget // GOLD_SHOT_FRAMES)
    without = full[:full_count]
    chosen = without
    if tail is not None and budget >= tail[1] - tail[0]:
        with_count = min(len(full), (budget - (tail[1] - tail[0])) // GOLD_SHOT_FRAMES)
        with_tail = full[:with_count] + [tail]
        if sum(shot[2] for shot in with_tail) > sum(shot[2] for shot in without):
            chosen = with_tail
    mask = bytearray(nframes)
    for start, end, _ in chosen:
        mask[start:end] = b"\x01" * (end - start)
    return mask


def intervals_mask(segments: list[dict[str, Any]], nframes: int, duration: float) -> bytearray:
    """Project exported source-time intervals to the annotation's frame axis."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Geçersiz video süresi")
    mask = bytearray(nframes)
    for segment in segments:
        start = float(segment.get("start", segment.get("start_seconds", -1)))
        end = float(segment.get("end", segment.get("end_seconds", -1)))
        if not all(map(math.isfinite, (start, end))) or start < 0 or end <= start or end > duration + 0.5:
            raise ValueError(f"Geçersiz seçilmiş aralık: {segment}")
        first = max(0, min(nframes, round(start * nframes / duration)))
        last = max(first, min(nframes, round(end * nframes / duration)))
        mask[first:last] = b"\x01" * (last - first)
    return mask


def f1(predicted: bytearray, gold: bytearray) -> float:
    if len(predicted) != len(gold):
        raise ValueError("F1 için kare sayıları eşit olmalı")
    pred_count, gold_count = sum(predicted), sum(gold)
    if not pred_count or not gold_count:
        return 0.0
    overlap = sum(a & b for a, b in zip(predicted, gold))
    return 2.0 * overlap / (pred_count + gold_count)


def recall(predicted: bytearray, gold: bytearray) -> float:
    """Fraction of human-reference frames covered by the predicted summary."""
    if len(predicted) != len(gold):
        raise ValueError("Recall için kare sayıları eşit olmalı")
    gold_count = sum(gold)
    return sum(a & b for a, b in zip(predicted, gold)) / gold_count if gold_count else 0.0


def precision(predicted: bytearray, gold: bytearray) -> float:
    """Fraction of predicted frames also present in the human reference."""
    if len(predicted) != len(gold):
        raise ValueError("Precision için kare sayıları eşit olmalı")
    pred_count = sum(predicted)
    return sum(a & b for a, b in zip(predicted, gold)) / pred_count if pred_count else 0.0


def uniform_mask(nframes: int, target_frames: int | None = None) -> bytearray:
    """Non-learned evenly distributed baseline, optionally duration-matched."""
    budget = int(BUDGET_PORTION * nframes) if target_frames is None else min(target_frames, int(BUDGET_PORTION * nframes))
    mask = bytearray(nframes)
    if not budget:
        return mask
    count = math.ceil(budget / GOLD_SHOT_FRAMES)
    for index in range(count):
        length = min(GOLD_SHOT_FRAMES, budget - index * GOLD_SHOT_FRAMES)
        midpoint = (index + 0.5) * nframes / count
        start = max(0, min(nframes - length, round(midpoint - length / 2)))
        mask[start:start + length] = b"\x01" * length
    return mask


def select_scored_shots(shots: list[dict[str, Any]], nframes: int, duration: float) -> list[dict[str, float]]:
    """TVSum-style score-only ablation: predicted shots + 15% knapsack.

    This measures the model's importance ranking separately from Thanos's
    context expansion, sentence guard, and FFmpeg export. It is not the final
    product's F1 and is reported under a distinct mode.
    """
    budget = int(BUDGET_PORTION * nframes)
    candidates = []
    for shot in sorted(shots, key=lambda row: float(row["start_seconds"])):
        start = max(0, min(nframes, round(float(shot["start_seconds"]) * nframes / duration)))
        end = max(start, min(nframes, round(float(shot["end_seconds"]) * nframes / duration)))
        score = float(shot["importance_score"])
        if end > start and end - start <= budget and math.isfinite(score):
            candidates.append((start, end, score))
    dp = [float("-inf")] * (budget + 1)
    dp[0] = 0.0
    decisions = []
    for start, end, score in candidates:
        weight = end - start
        taken = bytearray(budget + 1)
        for capacity in range(budget, weight - 1, -1):
            option = dp[capacity - weight] + score
            if option > dp[capacity]:
                dp[capacity] = option
                taken[capacity] = 1
        decisions.append(taken)
    capacity = max(range(budget + 1), key=lambda value: dp[value])
    selected = []
    for index in range(len(candidates) - 1, -1, -1):
        if decisions[index][capacity]:
            start, end, _ = candidates[index]
            selected.append({"start": start * duration / nframes, "end": end * duration / nframes})
            capacity -= end - start
    return list(reversed(selected))


def generate_thanos_summary(path: Path, row: dict[str, str], target: float, profile: str, narrative_mode: str,
                            *, score_only: bool = False, nframes: int = 0) -> list[dict[str, Any]]:
    """Run the same analysis + exporter used by the web app, including speech guard."""
    sys.path.insert(0, str(THANOS))
    from src.core.analysis_pipeline import analyze_video_features
    from src.scoring.scoring_engine import compute_scene_scores_for_video
    from src.summary.summary_exporter import export_category_summary

    alias = f"tvsum_{row['video_id']}"  # Never overwrite an unrelated video1/video2 cache.
    analyze_video_features(alias, path, THANOS, profile_name=profile)
    scored = compute_scene_scores_for_video(alias, THANOS)
    if score_only:
        return select_scored_shots(scored, nframes, video_duration(path))
    output_dir = THANOS / "outputs" / "tvsum"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{row['video_id']}_importance_{int(target)}s_{narrative_mode}.mp4"
    export_category_summary(
        video_path=str(path), scored_scenes=scored, category="importance",
        target_duration_sec=target, output_mp4_path=str(output),
        narrative_mode=narrative_mode, base_dir=THANOS, video_alias=alias,
    )
    debug_path = output_dir / "debug" / f"{path.stem}_importance_{int(target)}s_segments.json"
    with debug_path.open(encoding="utf-8") as handle:
        return json.load(handle)["segments"]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(args.tvsum_dir / "data" / "ydata-tvsum50-info.tsv")
    annotations = load_annotation_lines(args.tvsum_dir / "data" / "ydata-tvsum50-anno.tsv")
    results: list[dict[str, Any]] = []
    matched: list[str] = []
    missing: list[str] = []
    rejected: list[str] = []
    errors: list[str] = []
    for row in catalog:
        video_id = row["video_id"]
        path, duration, problem = find_video(args.videos_dir, row)
        if path is None:
            missing.append(video_id)
            if problem:
                rejected.append(f"{video_id}: {problem}")
            continue
        matched.append(video_id)
        if args.video_id and video_id not in args.video_id:
            continue
        if not args.run_model and not args.score_only and args.predictions_dir is None:
            continue
        ratings = parse_ratings(annotations[video_id])
        nframes = len(ratings[0])
        gold = [gold_mask(user_ratings) for user_ratings in ratings]
        baseline = uniform_mask(nframes)
        baseline_f1 = round(mean(f1(baseline, user) for user in gold), 5)
        try:
            if args.run_model or args.score_only:
                segments = generate_thanos_summary(
                    path, row, duration * BUDGET_PORTION, args.profile, args.narrative_mode,
                    score_only=args.score_only, nframes=nframes,
                )
            else:
                prediction_path = args.predictions_dir / f"{video_id}.json"
                if not prediction_path.is_file():
                    raise FileNotFoundError(f"Tahmin yok: {prediction_path}")
                with prediction_path.open(encoding="utf-8") as handle:
                    prediction = json.load(handle)
                segments = prediction["segments"] if isinstance(prediction, dict) else prediction
            if not segments:
                raise ValueError("%15 bütçesine seçilebilen sahne yok")
            selected_seconds = sum(
                float(segment.get("end", segment.get("end_seconds", -1)))
                - float(segment.get("start", segment.get("start_seconds", -1)))
                for segment in segments
            )
            if selected_seconds > duration * BUDGET_PORTION + 0.05:
                raise ValueError(f"%15 süre bütçesi aşıldı: {selected_seconds:.2f}/{duration * BUDGET_PORTION:.2f} sn")
            predicted = intervals_mask(segments, nframes, duration)
            matched_baseline = uniform_mask(nframes, sum(predicted))
            results.append({
                "video_id": video_id,
                "category": row["category"],
                "status": "ok",
                "source": str(path),
                "duration_sec": round(duration, 3),
                "selected_sec": round(selected_seconds, 3),
                "duration_ratio": round(sum(predicted) / nframes, 5),
                "f1": round(mean(f1(predicted, user) for user in gold), 5),
                "recall": round(mean(recall(predicted, user) for user in gold), 5),
                "precision": round(mean(precision(predicted, user) for user in gold), 5),
                "uniform_f1": baseline_f1,
                "uniform_recall": round(mean(recall(baseline, user) for user in gold), 5),
                "uniform_precision": round(mean(precision(baseline, user) for user in gold), 5),
                "matched_uniform_f1": round(mean(f1(matched_baseline, user) for user in gold), 5),
                "annotators": len(gold),
            })
            print(f"{video_id}: F1={results[-1]['f1']:.3f}, uniform={results[-1]['uniform_f1']:.3f}", flush=True)
        except Exception as exc:
            errors.append(f"{video_id}: {type(exc).__name__}: {exc}")
            results.append({
                "video_id": video_id, "category": row["category"], "status": "failed",
                "source": str(path), "duration_sec": round(duration, 3),
                "selected_sec": 0.0, "duration_ratio": 0.0, "f1": 0.0,
                "recall": 0.0, "precision": 0.0,
                "uniform_f1": baseline_f1, "annotators": len(gold),
                "uniform_recall": round(mean(recall(baseline, user) for user in gold), 5),
                "uniform_precision": round(mean(precision(baseline, user) for user in gold), 5),
                "matched_uniform_f1": 0.0,
                "error": errors[-1],
            })
            print(f"HATA {errors[-1]}", file=sys.stderr, flush=True)
    categories = sorted({row["category"] for row in results})
    successful = [row for row in results if row["status"] == "ok"]
    report = {
        "protocol": "TVSum 60-frame gold shots, per-user 15% knapsack, frame-level F1, macro average",
        "model": ("Thanos importance end-to-end, " + args.narrative_mode if args.run_model else
                  "Thanos shot scores only, 15% knapsack" if args.score_only else
                  "provided source-time predictions" if args.predictions_dir else
                  "audit-only; no model scored"),
        "evaluated_videos": len(results),
        "successful_videos": sum(row["status"] == "ok" for row in results),
        "matched_video_files": len(matched),
        "expected_videos": 50,
        "benchmark_complete": len(results) == 50,
        "mean_f1": round(mean(row["f1"] for row in results), 5) if results else None,
        "uniform_mean_f1": round(mean(row["uniform_f1"] for row in results), 5) if results else None,
        "mean_f1_success_only": round(mean(row["f1"] for row in successful), 5) if successful else None,
        "mean_recall_success_only": round(mean(row["recall"] for row in successful), 5) if successful else None,
        "mean_precision_success_only": round(mean(row["precision"] for row in successful), 5) if successful else None,
        "uniform_mean_f1_same_successes": round(mean(row["uniform_f1"] for row in successful), 5) if successful else None,
        "uniform_mean_recall_same_successes": round(mean(row["uniform_recall"] for row in successful), 5) if successful else None,
        "uniform_mean_precision_same_successes": round(mean(row["uniform_precision"] for row in successful), 5) if successful else None,
        "matched_uniform_mean_f1_success_only": round(
            mean(row["matched_uniform_f1"] for row in results if row["status"] == "ok"), 5,
        ) if any(row["status"] == "ok" for row in results) else None,
        "by_category": {category: round(mean(row["f1"] for row in results if row["category"] == category), 5) for category in categories},
        "videos": results,
        "missing_video_ids": missing,
        "rejected_files": rejected,
        "errors": errors,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TVSum ile Thanos özetlerinin %15-bütçe F1 değerlendirmesi")
    parser.add_argument("--tvsum-dir", type=Path, default=DEFAULT_TVSUM)
    parser.add_argument("--videos-dir", type=Path, default=THANOS / "dataset" / "video")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run-model", action="store_true", help="Thanos analizi ve gerçek MP4 özet üretimini çalıştır")
    source.add_argument("--score-only", action="store_true", help="Sahne önem puanlarını %15 knapsack ile, MP4 üretmeden ölç")
    source.add_argument("--predictions-dir", type=Path, help="<video_id>.json dosyalarındaki kaynak zaman aralıklarını değerlendir")
    parser.add_argument("--video-id", action="append", help="Yalnızca bu TVSum ID'sini işle; tekrarlanabilir")
    parser.add_argument("--profile", choices=("fast", "balanced", "quality"), default="balanced")
    parser.add_argument("--narrative-mode", choices=("local", "rag_llm"), default="local")
    parser.add_argument("--report", type=Path, default=THANOS / "outputs" / "tvsum" / "evaluation.json")
    args = parser.parse_args()
    if not args.videos_dir.is_dir():
        parser.error(f"Video klasörü bulunamadı: {args.videos_dir}")
    report = evaluate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"TVSum: {report['matched_video_files']}/50 kaynak video, {report['evaluated_videos']}/50 puanlandı; "
          f"ortalama F1={report['mean_f1']}; rapor={args.report}")
    if report["rejected_files"]:
        print("Reddedilenler: " + " | ".join(report["rejected_files"][:5]))
    if report["errors"]:
        return 1
    return 0 if report["evaluated_videos"] or not (args.run_model or args.score_only or args.predictions_dir) else 2


if __name__ == "__main__":
    raise SystemExit(main())
