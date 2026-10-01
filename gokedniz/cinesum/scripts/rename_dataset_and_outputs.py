import os
import json
import csv
import shutil
from pathlib import Path

def rename_dataset_and_outputs():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    video_dir = dataset_dir / "video"
    info_tsv = dataset_dir / "data" / "ydata-tvsum50-info.tsv"
    outputs_dir = base_dir / "outputs" / "pyscenedetect"
    reports_dir = base_dir / "reports"

    # Read TSV metadata if available, otherwise read video folder
    mapping = []
    video_id_to_alias = {}
    alias_to_info = {}

    if info_tsv.exists():
        with open(info_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader, start=1):
                vid = row["video_id"].strip()
                alias = f"video{idx}"
                item = {
                    "alias": alias,
                    "video_id": vid,
                    "category": row["category"].strip(),
                    "title": row["title"].strip(),
                    "url": row["url"].strip(),
                    "length": row["length"].strip(),
                    "original_filename": f"{vid}.mp4",
                    "new_filename": f"{alias}.mp4"
                }
                mapping.append(item)
                video_id_to_alias[vid] = alias
                alias_to_info[alias] = item
    else:
        # Fallback: scan video folder
        mp4_files = sorted(list(video_dir.glob("*.mp4")))
        for idx, v_file in enumerate(mp4_files, start=1):
            vid = v_file.stem
            alias = f"video{idx}"
            item = {
                "alias": alias,
                "video_id": vid,
                "category": "UNKNOWN",
                "title": vid,
                "url": "",
                "length": "",
                "original_filename": f"{vid}.mp4",
                "new_filename": f"{alias}.mp4"
            }
            mapping.append(item)
            video_id_to_alias[vid] = alias
            alias_to_info[alias] = item

    # Save mapping file
    mapping_file = dataset_dir / "video_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"[BAŞARILI] Haritalama kaydedildi: {mapping_file}")

    # 1. Rename videos in dataset/video and thumbnails in dataset/thumbnail
    renamed_count = 0
    thumb_dir = dataset_dir / "thumbnail"
    for item in mapping:
        old_v_path = video_dir / item["original_filename"]
        new_v_path = video_dir / item["new_filename"]
        if old_v_path.exists() and old_v_path != new_v_path:
            old_v_path.rename(new_v_path)
            renamed_count += 1

        old_t_path = thumb_dir / f"{item['video_id']}.jpg"
        new_t_path = thumb_dir / f"{item['alias']}.jpg"
        if old_t_path.exists() and old_t_path != new_t_path:
            old_t_path.rename(new_t_path)

    print(f"[BAŞARILI] {renamed_count} adet video ve thumbnail dosyası adlandırıldı (ör. video1.mp4/jpg, video2.mp4/jpg...).")

    # 2. Rename outputs/pyscenedetect/scene_lists files
    scene_lists_dir = outputs_dir / "scene_lists"
    if scene_lists_dir.exists():
        for old_file in list(scene_lists_dir.glob("*.*")):
            fname = old_file.name
            for vid, alias in video_id_to_alias.items():
                if vid in fname:
                    new_fname = fname.replace(vid, alias)
                    new_file = scene_lists_dir / new_fname
                    old_file.rename(new_file)
                    break
        print(f"[BAŞARILI] Sahne listesi çıktıları güncellendi.")

    # 3. Rename outputs/pyscenedetect/keyframes directories and files
    keyframes_dir = outputs_dir / "keyframes"
    if keyframes_dir.exists():
        for old_folder in list(keyframes_dir.iterdir()):
            if old_folder.is_dir() and old_folder.name in video_id_to_alias:
                vid = old_folder.name
                alias = video_id_to_alias[vid]
                new_folder = keyframes_dir / alias

                # Rename internal files
                for kf_file in list(old_folder.glob("*.*")):
                    new_kf_name = kf_file.name.replace(vid, alias)
                    kf_file.rename(old_folder / new_kf_name)

                # Rename folder
                old_folder.rename(new_folder)

                # Update keyframes_metadata.json content if exists
                meta_json = new_folder / "keyframes_metadata.json"
                if meta_json.exists():
                    with open(meta_json, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    for entry in meta_data:
                        if entry.get("video_stem") == vid:
                            entry["video_stem"] = alias
                        entry["file_name"] = entry["file_name"].replace(vid, alias)
                        entry["file_path"] = entry["file_path"].replace(vid, alias)
                    with open(meta_json, "w", encoding="utf-8") as f:
                        json.dump(meta_data, f, indent=2, ensure_ascii=False)

        print(f"[BAŞARILI] Keyframe klasörleri ve görselleri güncellendi.")

    # 4. Update reports/pyscenedetect_threshold_comparison.csv
    report_csv = reports_dir / "pyscenedetect_threshold_comparison.csv"
    if report_csv.exists():
        rows = []
        with open(report_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                v_name = row["video_name"]
                v_stem = row["video_stem"]
                if v_stem in video_id_to_alias:
                    alias = video_id_to_alias[v_stem]
                    row["video_name"] = f"{alias}.mp4"
                    row["video_stem"] = alias
                    row["json_output"] = row["json_output"].replace(v_stem, alias)
                    row["csv_output"] = row["csv_output"].replace(v_stem, alias)
                rows.append(row)
        if rows and fieldnames:
            with open(report_csv, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        print(f"[BAŞARILI] Rapor CSV dosyası güncellendi.")

if __name__ == "__main__":
    rename_dataset_and_outputs()
