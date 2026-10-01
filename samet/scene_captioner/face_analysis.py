from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

import cv2
import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def _merge_duplicate_clusters(
    matrix: np.ndarray,
    labels: np.ndarray,
    frame_owners: list[object],
) -> np.ndarray:
    """Conservatively reconnect pose-split identities without merging co-stars.

    Only faces visible in the exact same sampled frame are a hard cannot-link.
    Using the whole scene here used to preserve false identity splits whenever
    the same actor appeared with a different pose or disguise later in a shot.
    """
    labels = np.asarray(labels, dtype=int).copy()
    while True:
        groups = {
            label: np.flatnonzero(labels == label)
            for label in sorted(set(map(int, labels)))
        }
        best_candidate: tuple[float, int, int] | None = None
        group_items = list(groups.items())
        for left_pos, (left_label, left_indices) in enumerate(group_items):
            left_frames = {frame_owners[index] for index in left_indices}
            left_scenes = {
                owner[0] if isinstance(owner, tuple) else owner
                for owner in left_frames
            }
            for right_label, right_indices in group_items[left_pos + 1:]:
                # Two identities visible at the same instant cannot be the same person.
                right_frames = {frame_owners[index] for index in right_indices}
                if left_frames & right_frames:
                    continue
                right_scenes = {
                    owner[0] if isinstance(owner, tuple) else owner
                    for owner in right_frames
                }
                left_center = _unit(matrix[left_indices].mean(axis=0))
                right_center = _unit(matrix[right_indices].mean(axis=0))
                center_distance = 1.0 - float(left_center @ right_center)
                pair_distances = 1.0 - matrix[left_indices] @ matrix[right_indices].T
                best_pair_distance = float(pair_distances.min())
                # Average-link clustering can split profiles from frontal faces.
                # Merge only when the prototypes remain close and at least one
                # cross-cluster observation is a strong match.
                if left_scenes & right_scenes:
                    # Alternating close-ups of co-stars often share one detected
                    # scene. Permit a repair there only for near-duplicate faces.
                    strong_match = center_distance <= 0.38 and best_pair_distance <= 0.27
                    pose_bridge = center_distance <= 0.42 and best_pair_distance <= 0.22
                else:
                    strong_match = center_distance <= 0.46 and best_pair_distance <= 0.40
                    pose_bridge = center_distance <= 0.50 and best_pair_distance <= 0.32
                if not (strong_match or pose_bridge):
                    continue
                score = center_distance * 0.7 + best_pair_distance * 0.3
                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (score, left_label, right_label)
        if best_candidate is None:
            break
        _, keep, remove = best_candidate
        labels[labels == remove] = keep

    remap = {label: index for index, label in enumerate(sorted(set(map(int, labels))))}
    return np.asarray([remap[int(label)] for label in labels], dtype=int)


def _face_quality(image: np.ndarray, face: np.ndarray, aligned: np.ndarray) -> float:
    height, width = image.shape[:2]
    area_ratio = float(face[2] * face[3]) / max(1.0, width * height)
    sharpness = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    # YuNet landmarks: right eye, left eye, nose, mouth corners.
    right_eye = np.asarray(face[4:6], dtype=np.float32)
    left_eye = np.asarray(face[6:8], dtype=np.float32)
    nose = np.asarray(face[8:10], dtype=np.float32)
    eye_distance = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)
    eye_midpoint = (left_eye + right_eye) / 2.0
    frontal = max(0.0, 1.0 - abs(float(nose[0] - eye_midpoint[0])) / eye_distance)
    return (
        float(face[-1]) * 0.45
        + min(1.0, area_ratio / 0.035) * 0.25
        + min(1.0, sharpness / 240.0) * 0.20
        + frontal * 0.10
    )


def _identity_candidates(
    matrix: np.ndarray,
    labels: np.ndarray,
    frame_owners: list[object],
    visible_labels: list[int],
) -> dict[int, tuple[int, float]]:
    """Find reciprocal near-neighbour clusters just below the auto-merge bar."""
    groups = {label: np.flatnonzero(labels == label) for label in visible_labels}
    nearest: dict[int, tuple[int, float]] = {}
    for left, left_indices in groups.items():
        left_center = _unit(matrix[left_indices].mean(axis=0))
        left_frames = {frame_owners[index] for index in left_indices}
        for right, right_indices in groups.items():
            if left == right:
                continue
            right_frames = {frame_owners[index] for index in right_indices}
            if left_frames & right_frames:
                continue
            right_center = _unit(matrix[right_indices].mean(axis=0))
            center_similarity = float(left_center @ right_center)
            best_similarity = float((matrix[left_indices] @ matrix[right_indices].T).max())
            # These are intentionally below the automatic merge threshold.
            if center_similarity < 0.42 or best_similarity < 0.55:
                continue
            score = center_similarity * 0.7 + best_similarity * 0.3
            if left not in nearest or score > nearest[left][1]:
                nearest[left] = (right, score)

    candidates: dict[int, tuple[int, float]] = {}
    for left, (right, score) in nearest.items():
        reverse = nearest.get(right)
        if reverse and reverse[0] == left:
            candidates[left] = (right, score)
    return candidates


def analyze_faces(
    scenes,
    models_dir: Path,
    video_path: Path | None = None,
    portraits_dir: Path | None = None,
) -> tuple[dict[int, list[str]], list[dict], dict[int, list[dict]]]:
    detector_path = models_dir / "face_detection_yunet_2023mar.onnx"
    recognizer_path = models_dir / "face_recognition_sface_2021dec.onnx"
    if not detector_path.exists() or not recognizer_path.exists():
        return {}, [], {}

    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.82, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    observations: list[dict] = []
    sample_totals: dict[int, int] = defaultdict(int)

    capture = cv2.VideoCapture(str(video_path)) if video_path else None
    # Multiple views reduce profile/frontal splits. Keep the extra work bounded
    # on videos containing hundreds of shots.
    extra_samples = 2 if len(scenes) <= 180 else 1 if len(scenes) <= 500 else 0

    def inspect(image: np.ndarray, scene, sample_number: int, primary: bool) -> None:
        if image is None or not image.size:
            return
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        sample_totals[scene.index] += 1
        if faces is None:
            return
        for face in sorted(faces, key=lambda row: row[2] * row[3], reverse=True)[:6]:
            if face[2] * face[3] < width * height * 0.003:
                continue
            try:
                aligned = recognizer.alignCrop(image, face)
                feature = _unit(recognizer.feature(aligned).flatten().astype(np.float32))
            except cv2.error:
                continue
            observations.append({
                "scene": scene.index,
                "sample": (scene.index, sample_number),
                "primary": primary,
                "feature": feature,
                "confidence": float(face[-1]),
                "quality": _face_quality(image, face, aligned),
                "portrait": aligned.copy(),
                "box": (
                    max(0.0, float(face[0]) / width),
                    max(0.0, float(face[1]) / height),
                    min(1.0, float(face[2]) / width),
                    min(1.0, float(face[3]) / height),
                ),
            })

    try:
        for scene in scenes:
            primary = cv2.imread(str(scene.frame_path))
            inspect(primary, scene, 0, True)
            if not capture or not capture.isOpened() or extra_samples == 0 or scene.duration_sec < 1.0:
                continue
            fractions = (0.24, 0.80) if extra_samples == 2 else (0.24,)
            for sample_number, fraction in enumerate(fractions, 1):
                timestamp = scene.start_sec + scene.duration_sec * fraction
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000)
                ok, image = capture.read()
                if ok:
                    inspect(image, scene, sample_number, False)
    finally:
        if capture:
            capture.release()

    if not observations:
        return {}, [], {}

    from sklearn.cluster import AgglomerativeClustering

    matrix = np.asarray([item["feature"] for item in observations], dtype=np.float32)
    # A sample identifies one concrete video frame. Scene ids are intentionally
    # too broad: the same actor can occur in several sampled frames of one shot.
    owners = [item["sample"] for item in observations]
    if len(matrix) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.42, metric="cosine", linkage="average"
        ).fit_predict(matrix)
        labels = _merge_duplicate_clusters(matrix, labels, owners)

    scene_samples: dict[int, dict[int, set[tuple[int, int]]]] = defaultdict(lambda: defaultdict(set))
    raw_presence: dict[int, set[int]] = defaultdict(set)
    for observation, label in zip(observations, labels):
        scene_index = int(observation["scene"])
        label = int(label)
        raw_presence[scene_index].add(label)
        scene_samples[scene_index][label].add(observation["sample"])

    durations = {scene.index: scene.duration_sec for scene in scenes}
    screen_time: dict[int, float] = defaultdict(float)
    appearances: dict[int, int] = defaultdict(int)
    for scene_index, identities in scene_samples.items():
        total_samples = max(1, sample_totals[scene_index])
        for label, hits in identities.items():
            screen_time[label] += durations.get(scene_index, 0.0) * len(hits) / total_samples
            appearances[label] += 1

    minimum_appearances = max(2, math.ceil(len(scenes) * 0.005))
    ordered = [
        label
        for label in sorted(screen_time, key=lambda item: (-screen_time[item], item))
        if appearances[label] >= minimum_appearances
    ][:24]
    if not ordered and screen_time:
        ordered = sorted(screen_time, key=lambda item: (-screen_time[item], item))[:3]
    display_id = {label: rank + 1 for rank, label in enumerate(ordered)}
    remaining = [label for label in sorted(set(map(int, labels))) if label not in display_id]
    all_display_id = dict(display_id)
    for label in remaining:
        all_display_id[label] = len(all_display_id) + 1
    identity_candidates = _identity_candidates(matrix, labels, owners, ordered)

    if portraits_dir:
        portraits_dir.mkdir(parents=True, exist_ok=True)
    portrait_paths: dict[int, str] = {}
    for label in ordered:
        candidates = [
            observation for observation, observation_label in zip(observations, labels)
            if int(observation_label) == label
        ]
        best = max(candidates, key=lambda item: item["quality"])
        if portraits_dir:
            portrait_path = portraits_dir / f"actor_{display_id[label]:02d}.jpg"
            portrait = cv2.resize(best["portrait"], (180, 180), interpolation=cv2.INTER_CUBIC)
            if cv2.imwrite(str(portrait_path), portrait, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
                portrait_paths[label] = str(portrait_path)

    presence = {
        scene_index: [
            f"Oyuncu {display_id[label]}"
            for label in sorted((item for item in scene_labels if item in display_id), key=display_id.get)
        ]
        for scene_index, scene_labels in raw_presence.items()
    }
    cast = []
    for rank, label in enumerate(ordered):
        person = {
            "id": f"Oyuncu {display_id[label]}",
            "screen_time_sec": round(screen_time[label], 1),
            "scene_count": appearances[label],
            "role": "Ana karakter" if rank < min(3, len(ordered)) else "Yan karakter",
            "portrait_path": portrait_paths.get(label),
        }
        candidate = identity_candidates.get(label)
        if candidate and display_id[label] > display_id[candidate[0]]:
            other, score = candidate
            person["possible_same_as"] = f"Oyuncu {display_id[other]}"
            person["identity_similarity"] = round(score, 3)
        cast.append(person)

    face_map: dict[int, list[dict]] = defaultdict(list)
    for observation, label in zip(observations, labels):
        if not observation["primary"]:
            continue
        label = int(label)
        x, y, width, height = observation["box"]
        face_map[int(observation["scene"])].append({
            "actor": f"Oyuncu {all_display_id[label]}",
            "x": round(x, 4), "y": round(y, 4),
            "width": round(width, 4), "height": round(height, 4),
            "confidence": round(float(observation["confidence"]), 3),
            "recurring": label in display_id,
        })
    return presence, cast, dict(face_map)
