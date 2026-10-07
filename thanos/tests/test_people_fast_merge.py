import unittest

import numpy as np

from src.features.samet_face_analysis import _merge_duplicate_clusters, _unit


def original_merge(matrix, labels, frame_owners):
    """Small-input reference for the former full-rescan implementation."""
    labels = np.asarray(labels, dtype=int).copy()
    while True:
        groups = {label: np.flatnonzero(labels == label) for label in sorted(set(map(int, labels)))}
        best = None
        for left_pos, (left, left_indices) in enumerate(groups.items()):
            left_frames = {frame_owners[index] for index in left_indices}
            left_scenes = {owner[0] if isinstance(owner, tuple) else owner for owner in left_frames}
            for right, right_indices in list(groups.items())[left_pos + 1:]:
                right_frames = {frame_owners[index] for index in right_indices}
                if left_frames & right_frames:
                    continue
                right_scenes = {owner[0] if isinstance(owner, tuple) else owner for owner in right_frames}
                center_distance = 1.0 - float(
                    _unit(matrix[left_indices].mean(axis=0))
                    @ _unit(matrix[right_indices].mean(axis=0))
                )
                pair_distance = float((1.0 - matrix[left_indices] @ matrix[right_indices].T).min())
                if left_scenes & right_scenes:
                    match = ((center_distance <= 0.38 and pair_distance <= 0.27)
                             or (center_distance <= 0.42 and pair_distance <= 0.22))
                else:
                    match = ((center_distance <= 0.46 and pair_distance <= 0.40)
                             or (center_distance <= 0.50 and pair_distance <= 0.32))
                if not match:
                    continue
                score = center_distance * 0.7 + pair_distance * 0.3
                if best is None or score < best[0]:
                    best = score, left, right
        if best is None:
            break
        labels[labels == best[2]] = best[1]
    remap = {label: index for index, label in enumerate(sorted(set(map(int, labels))))}
    return np.asarray([remap[int(label)] for label in labels], dtype=int)


class TestFastFaceMerge(unittest.TestCase):
    def test_matches_original_on_pose_splits_and_frame_conflicts(self):
        for seed in range(30):
            rng = np.random.default_rng(seed)
            prototypes = rng.normal(size=(5, 12)).astype(np.float32)
            prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
            source = rng.integers(0, len(prototypes), size=24)
            matrix = prototypes[source] + rng.normal(0, 0.12, size=(24, 12)).astype(np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            labels = rng.integers(0, 15, size=24)
            frame_owners = [(int(rng.integers(0, 9)), int(rng.integers(0, 3))) for _ in labels]
            with self.subTest(seed=seed):
                np.testing.assert_array_equal(
                    _merge_duplicate_clusters(matrix, labels, frame_owners),
                    original_merge(matrix, labels, frame_owners),
                )


if __name__ == "__main__":
    unittest.main()
