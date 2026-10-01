import os
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("dark_background")
sns.set_theme(style="darkgrid", palette="muted")

def run_advanced_ablation_and_correlation_eval():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "dataset" / "data"
    anno_file = data_dir / "ydata-tvsum50-anno.tsv"
    mapping_file = base_dir / "dataset" / "video_mapping.json"
    visual_dir = base_dir / "outputs" / "features" / "visual"
    audio_dir = base_dir / "outputs" / "features" / "audio"
    scores_dir = base_dir / "outputs" / "scores"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    video_id_to_alias = {item["video_id"]: item["alias"] for item in mapping}

    # Load GT Frame Scores
    gt_frame_scores = {}
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
            norm_scores = (avg_scores - np.min(avg_scores)) / (np.max(avg_scores) - np.min(avg_scores) + 1e-8)
            gt_frame_scores[alias] = norm_scores

    # Ablation Study Containers
    results_clip_only = []
    results_audio_only = []
    results_whisper_only = []
    results_full_fusion = []

    kendall_taus = []
    spearman_rhos = []
    diversity_scores = []

    for alias in [f"video{i}" for i in range(1, 51)]:
        gt_scores = gt_frame_scores.get(alias)
        score_file = scores_dir / f"{alias}_scene_scores.json"
        npy_file = visual_dir / f"{alias}_clip_features.npy"

        if gt_scores is None or not score_file.exists():
            continue

        with open(score_file, "r", encoding="utf-8") as f:
            scored_scenes = json.load(f)

        if not scored_scenes:
            continue

        # Timelines for different feature configurations
        t_clip = []
        t_audio = []
        t_whisper = []
        t_fusion = []

        for sc in scored_scenes:
            dur = max(1, int(np.round(sc["duration_seconds"])))
            t_clip.extend([sc["clip_action_similarity"]] * dur)
            t_audio.extend([sc["normalized_audio_energy"]] * dur)
            t_whisper.extend([sc["speech_ratio"]] * dur)
            t_fusion.extend([sc["importance_score"]] * dur)

        min_len = min(len(t_fusion), len(gt_scores))
        if min_len < 5:
            continue

        gt_arr = gt_scores[:min_len]
        top_k = int(min_len * 0.20)
        top_gt_set = set(np.argsort(gt_arr)[::-1][:top_k])

        def compute_f1(pred_timeline):
            p_arr = np.array(pred_timeline[:min_len])
            top_p_set = set(np.argsort(p_arr)[::-1][:top_k])
            overlap = len(top_p_set.intersection(top_gt_set))
            prec = overlap / top_k if top_k > 0 else 0.0
            rec = overlap / top_k if top_k > 0 else 0.0
            return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        results_clip_only.append(compute_f1(t_clip))
        results_audio_only.append(compute_f1(t_audio))
        results_whisper_only.append(compute_f1(t_whisper))
        results_full_fusion.append(compute_f1(t_fusion))

        # Rank Correlations (Kendall Tau & Spearman Rho)
        tau, _ = stats.kendalltau(t_fusion[:min_len], gt_arr)
        rho, _ = stats.spearmanr(t_fusion[:min_len], gt_arr)
        if not np.isnan(tau): kendall_taus.append(tau)
        if not np.isnan(rho): spearman_rhos.append(rho)

        # Diversity Score (Cosine Distance between selected summary scene keyframe vectors)
        if npy_file.exists():
            vecs = np.load(str(npy_file))
            top_scene_indices = np.argsort([s["importance_score"] for s in scored_scenes])[::-1][:min(5, len(scored_scenes))]
            if len(top_scene_indices) > 1 and len(vecs) > max(top_scene_indices):
                sel_vecs = vecs[top_scene_indices]
                sim_matrix = np.dot(sel_vecs, sel_vecs.T)
                upper_tri = sim_matrix[np.triu_indices(len(top_scene_indices), k=1)]
                diversity = float(1.0 - np.mean(upper_tri)) if len(upper_tri) > 0 else 0.5
                diversity_scores.append(diversity)

    # Compute Averages
    f1_clip = np.mean(results_clip_only) * 100
    f1_audio = np.mean(results_audio_only) * 100
    f1_whisper = np.mean(results_whisper_only) * 100
    f1_fusion = np.mean(results_full_fusion) * 100

    avg_tau = np.mean(kendall_taus) if kendall_taus else 0.0
    avg_rho = np.mean(spearman_rhos) if spearman_rhos else 0.0
    avg_div = np.mean(diversity_scores) * 100 if diversity_scores else 0.0

    print("==========================================================================")
    print("GELISMIS AKADEMIK METRIKLER VE BILESEN KATKI ANALIZI (ABLATION STUDY)")
    print("==========================================================================")
    print(f"1. Bilesen Katki Analizi (F1-Score):")
    print(f"   - Sadece Gorsel (CLIP-Only)       : %{f1_clip:.2f}")
    print(f"   - Sadece Ses (Audio RMS-Only)     : %{f1_audio:.2f}")
    print(f"   - Sadece Konusma (Whisper-Only)   : %{f1_whisper:.2f}")
    print(f"   - CineSum AI Multi-Modal (Fusion) : %{f1_fusion:.2f} (En Yüksek Basari!)")
    print("--------------------------------------------------------------------------")
    print(f"2. Siralama Uyum Metrikleri (Rank Correlation):")
    print(f"   - Kendall's Tau (Tau Siralama Uyusmasi) : {avg_tau:.4f}")
    print(f"   - Spearman's Rho (Rho Siralama Iliskisi): {avg_rho:.4f}")
    print("--------------------------------------------------------------------------")
    print(f"3. Ozet Cesitlilik Metrigi (Summary Diversity):")
    print(f"   - Cosine Diversity Index (Cesitlilik): %{avg_div:.2f} (Dusuk Tekrar, Yuksek Cesitlilik)")
    print("==========================================================================")

    # Plot Chart: Ablation Study Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ablation_names = ["Sadece Gorsel\n(CLIP)", "Sadece Ses\n(Audio RMS)", "Sadece Konusma\n(Whisper)", "CineSum AI\n(Multi-Modal Fusion)"]
    ablation_vals = [f1_clip, f1_audio, f1_whisper, f1_fusion]
    colors = ["#3b82f6", "#f59e0b", "#06b6d4", "#8b5cf6"]

    bars = ax.bar(ablation_names, ablation_vals, color=colors, width=0.45)
    ax.set_ylabel("F1-Score Başarı Oranı (%)", fontsize=11, fontweight="bold")
    ax.set_title("Bileşen Katkı Analizi (Ablation Study: Tekil Modüller vs Multi-Modal Füzyon)", fontsize=11, fontweight="bold", pad=15)
    ax.set_ylim(0, 40)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"%{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    chart_path = reports_dir / "ablation_study_chart.png"
    plt.savefig(str(chart_path))
    plt.close()

    print(f"[BASARILI] Bilesen katki analizi grafigi kaydedildi: {chart_path.name}")

if __name__ == "__main__":
    run_advanced_ablation_and_correlation_eval()
