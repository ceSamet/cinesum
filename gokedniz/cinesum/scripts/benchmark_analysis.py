import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.core.analysis_pipeline import analyze_video_features
from src.core.pipeline_config import ANALYSIS_PROFILES


def parse_args():
    parser = argparse.ArgumentParser(
        description="CineSum v3.1 analiz aşamalarını ölç ve JSON benchmark raporu üret."
    )
    parser.add_argument("video", help="Video alias'ı (video55) veya doğrudan video dosyası yolu")
    parser.add_argument(
        "--profile",
        choices=sorted(ANALYSIS_PROFILES),
        default="balanced",
        help="Analiz profili (varsayılan: balanced)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Geçerli cache olsa bile bütün aşamaları yeniden çalıştır",
    )
    return parser.parse_args()


def resolve_video(value: str) -> tuple[str, Path]:
    candidate = Path(value)
    if candidate.exists():
        return candidate.stem, candidate.resolve()

    alias = candidate.stem
    dataset_video = BASE_DIR / "dataset" / "video" / f"{alias}.mp4"
    if not dataset_video.exists():
        raise FileNotFoundError(f"Video bulunamadı: {value}")
    return alias, dataset_video


def main():
    args = parse_args()
    alias, video_path = resolve_video(args.video)

    def progress(percent, description, detail, _step):
        print(f"[{percent:3d}%] {description} — {detail}")

    started = time.perf_counter()
    result = analyze_video_features(
        video_alias=alias,
        video_path=video_path,
        base_dir=BASE_DIR,
        profile_name=args.profile,
        progress_callback=progress,
        force=args.force,
    )
    wall_time = time.perf_counter() - started

    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = {
        name: {
            "processing_sec": stage.get("processing_sec", 0.0),
            "status": stage.get("status"),
            "cache_hit_this_run": result["cache_hits"].get(name, False),
        }
        for name, stage in manifest.get("stages", {}).items()
    }

    report = {
        "benchmark_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "video_alias": alias,
        "video_path": str(video_path),
        "profile": args.profile,
        "forced": args.force,
        "wall_time_sec": round(wall_time, 4),
        "scenes_found": result["scenes_found"],
        "story_scenes_found": result["story_scenes_found"],
        "cache_hits": result["cache_hits"],
        "stages": stages,
    }

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"benchmark_{alias}_{args.profile}_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nAnaliz benchmark sonucu")
    print(f"  Video: {alias}")
    print(f"  Profil: {args.profile}")
    print(
        f"  Analiz birimleri: {result['scenes_found']} shot / "
        f"{result['story_scenes_found']} StoryScene"
    )
    print(f"  Duvar süresi: {wall_time:.2f} sn ({wall_time / 60:.2f} dk)")
    for stage_name, stage in stages.items():
        cache_label = "cache" if stage["cache_hit_this_run"] else "çalıştı"
        print(f"  - {stage_name}: {stage['processing_sec']:.2f} sn [{cache_label}]")
    print(f"  Rapor: {report_path}")


if __name__ == "__main__":
    main()
