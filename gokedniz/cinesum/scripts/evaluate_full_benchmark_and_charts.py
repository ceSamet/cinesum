import os
import json
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set dark theme style for plots
plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="muted")

def evaluate_and_generate_charts():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset" / "data"
    anno_file = data_dir / "ydata-tvsum50-anno.tsv"
    mapping_file = base_dir / "dataset" / "video_mapping.json"
    pyscene_dir = base_dir / "outputs" / "pyscenedetect" / "scene_lists"
    transnet_dir = base_dir / "outputs" / "transnetv2" / "scene_lists"
    scores_dir = base_dir / "outputs" / "scores"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load video mapping
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    video_id_to_alias = {item["video_id"]: item["alias"] for item in mapping}

    # 2. Parse TVSum50 Annotations
    gt_boundaries = {}       # alias -> list of GT cut timestamps in seconds
    gt_frame_scores = {}     # alias -> numpy array of frame importance scores (0-1)

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
            
            # Normalize GT frame scores to [0.0, 1.0]
            norm_scores = (avg_scores - np.min(avg_scores)) / (np.max(avg_scores) - np.min(avg_scores) + 1e-8)
            gt_frame_scores[alias] = norm_scores

            # GT cut boundaries (deltas between frames)
            cuts = []
            for i in range(1, len(avg_scores)):
                delta = abs(avg_scores[i] - avg_scores[i-1])
                if delta >= 0.4:
                    cuts.append(i * 2.0)
            gt_boundaries[alias] = cuts

    # =========================================================================
    # PART A: Sahne Ayırma Başarısı Değerlendirmesi (Scene Boundary Accuracy)
    # =========================================================================
    pyscene_metrics = {"precision": [], "recall": [], "f1": []}
    transnet_metrics = {"precision": [], "recall": [], "f1": []}

    tolerance_sec = 2.0

    for alias in [f"video{i}" for i in range(1, 51)]:
        gt_cuts = gt_boundaries.get(alias, [])
        if not gt_cuts:
            continue

        # PySceneDetect (adaptive_t3.0)
        p_json = pyscene_dir / f"{alias}_scenes.json"
        if p_json.exists():
            with open(p_json, "r", encoding="utf-8") as f:
                p_scenes = json.load(f)
            p_cuts = [s["start_seconds"] for s in p_scenes if s["start_seconds"] > 0.1]
            
            tp = sum(1 for pc in p_cuts if any(abs(pc - gc) <= tolerance_sec for gc in gt_cuts))
            prec = (tp / len(p_cuts)) if p_cuts else 0.0
            rec = (tp / len(gt_cuts)) if gt_cuts else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            pyscene_metrics["precision"].append(prec)
            pyscene_metrics["recall"].append(rec)
            pyscene_metrics["f1"].append(f1)

        # TransNetV2 (transnetv2_t50)
        t_json = transnet_dir / f"{alias}_transnetv2_t50.json"
        if t_json.exists():
            with open(t_json, "r", encoding="utf-8") as f:
                t_scenes = json.load(f)
            t_cuts = [s["start_seconds"] for s in t_scenes if s["start_seconds"] > 0.1]
            
            tp = sum(1 for tc in t_cuts if any(abs(tc - gc) <= tolerance_sec for gc in gt_cuts))
            prec = (tp / len(t_cuts)) if t_cuts else 0.0
            rec = (tp / len(gt_cuts)) if gt_cuts else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            transnet_metrics["precision"].append(prec)
            transnet_metrics["recall"].append(rec)
            transnet_metrics["f1"].append(f1)

    avg_pyscene_prec = np.mean(pyscene_metrics["precision"]) * 100
    avg_pyscene_rec = np.mean(pyscene_metrics["recall"]) * 100
    avg_pyscene_f1 = np.mean(pyscene_metrics["f1"]) * 100

    avg_transnet_prec = np.mean(transnet_metrics["precision"]) * 100 if transnet_metrics["precision"] else 0.0
    avg_transnet_rec = np.mean(transnet_metrics["recall"]) * 100 if transnet_metrics["recall"] else 0.0
    avg_transnet_f1 = np.mean(transnet_metrics["f1"]) * 100 if transnet_metrics["f1"] else 0.0

    # Plot Chart 1: Scene Boundary Detection Comparison with Auto Zoom-In Y-Axis
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    categories = ["Precision (Hassasiyet)", "Recall (Duyarlılık)", "F1-Score (Genel Başarı)"]
    pyscene_vals = [avg_pyscene_prec, avg_pyscene_rec, avg_pyscene_f1]
    transnet_vals = [avg_transnet_prec, avg_transnet_rec, avg_transnet_f1]

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax.bar(x - width/2, pyscene_vals, width, label="PySceneDetect (Adaptive t=3.0)", color="#8b5cf6")
    rects2 = ax.bar(x + width/2, transnet_vals, width, label="TransNetV2 (Deep Neural Net)", color="#3b82f6")

    ax.set_ylabel("Skor Değeri (%)", fontsize=11, fontweight="bold")
    ax.set_title("Sahne Kesim Noktası Tespit Başarısı (TVSum50 Pseudo-Cut Comparison)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    
    max_y = max(max(pyscene_vals), max(transnet_vals)) * 1.35
    ax.set_ylim(0, max(1.5, max_y))
    ax.legend(fontsize=10)

    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"%{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"%{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart1_path = reports_dir / "scene_detection_accuracy.png"
    plt.savefig(str(chart1_path))
    plt.close()

    # =========================================================================
    # PART B: Video Özetleme Başarısı (Multi-Modal CineSum AI vs TVSum50 Human Consensus)
    # =========================================================================
    summarization_results = {"precision": [], "recall": [], "f1": [], "map": []}

    for alias in [f"video{i}" for i in range(1, 51)]:
        gt_scores = gt_frame_scores.get(alias)
        score_file = scores_dir / f"{alias}_scene_scores.json"

        if gt_scores is None or not score_file.exists():
            continue

        with open(score_file, "r", encoding="utf-8") as f:
            scored_scenes = json.load(f)

        if not scored_scenes:
            continue

        # Map scene importance scores back to frame/second timeline
        pred_timeline = []
        for sc in scored_scenes:
            dur = int(np.round(sc["duration_seconds"]))
            sc_score = sc["importance_score"]
            pred_timeline.extend([sc_score] * max(1, dur))

        pred_timeline = np.array(pred_timeline)

        # Match timeline lengths
        min_len = min(len(pred_timeline), len(gt_scores))
        if min_len < 5:
            continue

        pred_timeline = pred_timeline[:min_len]
        gt_arr = gt_scores[:min_len]

        # Top 20% budget summary selection
        top_k = int(min_len * 0.20)
        top_pred_indices = set(np.argsort(pred_timeline)[::-1][:top_k])
        top_gt_indices = set(np.argsort(gt_arr)[::-1][:top_k])

        overlap = len(top_pred_indices.intersection(top_gt_indices))
        prec = overlap / top_k if top_k > 0 else 0.0
        rec = overlap / top_k if top_k > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Mean Average Precision (mAP)
        sorted_gt_rank = np.argsort(pred_timeline)[::-1]
        gt_binary = (gt_arr >= np.percentile(gt_arr, 80)).astype(int)
        
        ap_sum = 0.0
        hits = 0
        for r_idx, idx in enumerate(sorted_gt_rank, 1):
            if gt_binary[idx] == 1:
                hits += 1
                ap_sum += hits / r_idx
        map_score = ap_sum / max(1, np.sum(gt_binary))

        summarization_results["precision"].append(prec)
        summarization_results["recall"].append(rec)
        summarization_results["f1"].append(f1)
        summarization_results["map"].append(map_score)

    avg_sum_prec = np.mean(summarization_results["precision"]) * 100
    avg_sum_rec = np.mean(summarization_results["recall"]) * 100
    avg_sum_f1 = np.mean(summarization_results["f1"]) * 100
    avg_sum_map = np.mean(summarization_results["map"]) * 100

    # Plot Chart 2: Video Summarization Overall Performance
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    metrics_names = ["Precision (Hassasiyet)", "Recall (Duyarlılık)", "F1-Score (Başarı)", "mAP (Sıralama Kalitesi)"]
    metrics_vals = [avg_sum_prec, avg_sum_rec, avg_sum_f1, avg_sum_map]
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"]

    bars = ax.bar(metrics_names, metrics_vals, color=colors, width=0.45)
    ax.set_ylabel("Başarı Oranı (%)", fontsize=11, fontweight="bold")
    ax.set_title("CineSum AI Yapay Zekâ Özetleme Başarısı (TVSum50 Benchmark)", fontsize=12, fontweight="bold", pad=15)
    ax.set_ylim(0, 50)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"%{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    chart2_path = reports_dir / "video_summarization_accuracy.png"
    plt.savefig(str(chart2_path))
    plt.close()

if __name__ == "__main__":
    evaluate_and_generate_charts()
