import json
import csv
import numpy as np
from pathlib import Path

def calculate_comparison_metrics():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset" / "data"
    anno_file = data_dir / "ydata-tvsum50-anno.tsv"
    mapping_file = base_dir / "dataset" / "video_mapping.json"
    pyscene_dir = base_dir / "outputs" / "pyscenedetect" / "scene_lists"
    transnet_dir = base_dir / "outputs" / "transnetv2" / "scene_lists"
    reports_dir = base_dir / "reports"

    # 1. Load video mapping
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    video_id_to_alias = {item["video_id"]: item["alias"] for item in mapping}

    # 2. Parse TVSum50 Annotations -> Compute Ground Truth Shot Boundaries
    gt_boundaries = {}  # alias -> list of GT cut timestamps in seconds

    if anno_file.exists():
        raw_annos = {}
        with open(anno_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                vid = parts[0]
                scores = [float(x) for x in parts[2].split(",")]
                if vid not in raw_annos:
                    raw_annos[vid] = []
                raw_annos[vid].append(scores)

        for vid, anno_list in raw_annos.items():
            alias = video_id_to_alias.get(vid, vid)
            avg_scores = np.mean(np.array(anno_list), axis=0)
            
            cuts = []
            for i in range(1, len(avg_scores)):
                delta = abs(avg_scores[i] - avg_scores[i-1])
                if delta >= 0.4:
                    cuts.append(i * 2.0)
            gt_boundaries[alias] = cuts

    # All configurations to evaluate
    pyscene_configs = [("PySceneDetect", "content_t27"), ("PySceneDetect", "content_t15"), ("PySceneDetect", "content_t35"), ("PySceneDetect", "adaptive_t3.0")]
    transnet_configs = [("TransNetV2", "transnetv2_t40"), ("TransNetV2", "transnetv2_t50"), ("TransNetV2", "transnetv2_t60")]

    all_configs = pyscene_configs + transnet_configs
    first_5_aliases = ["video1", "video2", "video3", "video4", "video5"]
    tolerance_sec = 1.0  # Window tolerance for cut matching

    results = []

    for alias in first_5_aliases:
        if alias not in gt_boundaries:
            continue

        gt_cuts = gt_boundaries[alias]
        if not gt_cuts:
            continue

        for model_name, cfg in all_configs:
            if model_name == "PySceneDetect":
                scene_json = pyscene_dir / f"{alias}_{cfg}.json"
            else:
                scene_json = transnet_dir / f"{alias}_{cfg}.json"

            if not scene_json.exists():
                continue

            with open(scene_json, "r", encoding="utf-8") as f:
                scenes = json.load(f)

            pred_cuts = [s["start_seconds"] for s in scenes if s["start_seconds"] > 0.1]

            tp = 0
            matched_gt = set()

            for p_cut in pred_cuts:
                for gt_idx, g_cut in enumerate(gt_cuts):
                    if gt_idx not in matched_gt and abs(p_cut - g_cut) <= tolerance_sec:
                        tp += 1
                        matched_gt.add(gt_idx)
                        break

            precision = tp / len(pred_cuts) if pred_cuts else 0.0
            recall = tp / len(gt_cuts) if gt_cuts else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            results.append({
                "video_alias": alias,
                "model_name": model_name,
                "config": cfg,
                "pred_cuts_count": len(pred_cuts),
                "gt_cuts_count": len(gt_cuts),
                "true_positives": tp,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            })

    # Save summary CSV
    out_csv = reports_dir / "model_comparison_metrics.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Print Summary Table grouped by Model & Config
    print("\n==========================================================================================")
    print("İLK 5 VİDEO İÇİN PYSCENEDETECT VS TRANSNETV2 AKADEMİK DEĞERLENDİRME SKORLARI")
    print("==========================================================================================")

    config_stats = {}
    for r in results:
        key = f"{r['model_name']} ({r['config']})"
        if key not in config_stats:
            config_stats[key] = {"p": [], "r": [], "f1": []}
        config_stats[key]["p"].append(r["precision"])
        config_stats[key]["r"].append(r["recall"])
        config_stats[key]["f1"].append(r["f1_score"])

    for key, vals in config_stats.items():
        avg_p = round(np.mean(vals["p"]), 4)
        avg_r = round(np.mean(vals["r"]), 4)
        avg_f1 = round(np.mean(vals["f1"]), 4)
        print(f"[{key:32s}] -> Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | F1-Score: {avg_f1:.4f}")

    print(f"\n[BAŞARILI] Karşılaştırma raporu kaydedildi: {out_csv}")

if __name__ == "__main__":
    calculate_comparison_metrics()
