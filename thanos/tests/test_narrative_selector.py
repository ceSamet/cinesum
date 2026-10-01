import unittest
from unittest.mock import patch

import src.selection.narrative_selector as narrative_selector

from src.selection.narrative_selector import (
    knapsack_select,
    mmr_rank,
    narrative_region,
    select_narrative_shots,
)


class TestNarrativeSelector(unittest.TestCase):
    def test_narrative_region_boundaries(self):
        self.assertEqual(
            narrative_region({"start_seconds": 5.0, "end_seconds": 10.0}, 100.0),
            "intro",
        )
        self.assertEqual(
            narrative_region({"start_seconds": 45.0, "end_seconds": 50.0}, 100.0),
            "middle",
        )
        self.assertEqual(
            narrative_region({"start_seconds": 80.0, "end_seconds": 85.0}, 100.0),
            "late_climax",
        )

    def test_knapsack_prefers_best_combined_value(self):
        candidates = [
            {"scene_id": 1, "_estimated_duration": 6.0, "_mmr_score": 0.90},
            {"scene_id": 2, "_estimated_duration": 4.0, "_mmr_score": 0.62},
            {"scene_id": 3, "_estimated_duration": 4.0, "_mmr_score": 0.61},
        ]
        selected = knapsack_select(candidates, 8.0)
        self.assertEqual({shot["scene_id"] for shot in selected}, {2, 3})

    def test_mmr_ranks_distinct_scene_before_near_duplicate(self):
        candidates = [
            {
                "scene_id": 1,
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "duration_seconds": 5.0,
                "score": 0.90,
                "action_score": 0.95,
                "dialogue_score": 0.05,
            },
            {
                "scene_id": 2,
                "start_seconds": 5.0,
                "end_seconds": 10.0,
                "duration_seconds": 5.0,
                "score": 0.88,
                "action_score": 0.94,
                "dialogue_score": 0.05,
            },
            {
                "scene_id": 3,
                "start_seconds": 50.0,
                "end_seconds": 55.0,
                "duration_seconds": 5.0,
                "score": 0.82,
                "action_score": 0.05,
                "dialogue_score": 0.95,
                "speech_ratio": 0.9,
            },
        ]
        ranked = mmr_rank(
            candidates,
            score_key="score",
            diversity_weight=0.30,
            video_duration=100.0,
        )
        self.assertEqual([shot["scene_id"] for shot in ranked[:2]], [1, 3])

    def test_mmr_uses_incremental_similarity_cache(self):
        candidates = [
            {
                "scene_id": index + 1,
                "start_seconds": float(index),
                "end_seconds": float(index + 1),
                "duration_seconds": 1.0,
                "score": 1.0 - index / 1000.0,
                "importance_score": (index % 11) / 10.0,
                "normalized_audio_energy": (index % 7) / 6.0,
            }
            for index in range(120)
        ]

        original = narrative_selector.cosine_similarity
        with patch.object(
            narrative_selector,
            "cosine_similarity",
            wraps=original,
        ) as similarity:
            ranked = mmr_rank(
                candidates,
                score_key="score",
                diversity_weight=0.25,
                video_duration=120.0,
                max_results=30,
            )

        self.assertEqual(len(ranked), 30)
        self.assertLess(similarity.call_count, 4000)

    def test_feature_length_video_bounds_mmr_candidate_pools(self):
        shots = []
        for index in range(2470):
            start = index * 4.0
            shots.append({
                "scene_id": index + 1,
                "story_scene_id": index // 4,
                "start_seconds": start,
                "end_seconds": start + 3.0,
                "duration_seconds": 3.0,
                "smoothed_score": 1.0 - (index % 100) / 1000.0,
                "importance_score": 0.8,
                "normalized_audio_energy": (index % 10) / 10.0,
            })

        result = select_narrative_shots(
            shots,
            score_key="smoothed_score",
            category="importance",
            target_duration_sec=60.0,
            video_duration=9880.0,
            segment_config={
                "min_duration": 5.0,
                "max_duration": 15.0,
                "pre_context": 2.0,
                "post_context": 2.5,
            },
        )

        self.assertTrue(result["selected_shot_ids"])
        self.assertTrue(result["candidate_pool_counts"])
        self.assertTrue(all(count <= 400 for count in result["candidate_pool_counts"].values()))

    def test_selector_represents_intro_middle_and_ending(self):
        shots = []
        for index in range(12):
            start = index * 10.0
            shots.append({
                "scene_id": index + 1,
                "start_seconds": start,
                "end_seconds": start + 2.0,
                "duration_seconds": 2.0,
                "smoothed_score": 0.95 - index * 0.01,
                "action_score": 0.95 - index * 0.01,
                "dialogue_score": (index % 3) * 0.2,
            })

        result = select_narrative_shots(
            shots,
            score_key="smoothed_score",
            category="action",
            target_duration_sec=30.0,
            video_duration=120.0,
            segment_config={
                "min_duration": 4.0,
                "max_duration": 12.0,
                "pre_context": 1.5,
                "post_context": 2.0,
            },
        )

        self.assertEqual(result["strategy"], "narrative_mmr_knapsack")
        self.assertGreater(result["region_selected_counts"]["intro"], 0)
        self.assertGreater(result["region_selected_counts"]["middle"], 0)
        self.assertGreater(result["region_selected_counts"]["ending"], 0)
        self.assertTrue(result["decisions"])

    def test_selector_caps_shots_from_same_story_scene(self):
        shots = []
        for index in range(9):
            shots.append({
                "scene_id": index + 1,
                "story_scene_id": 7,
                "start_seconds": index * 10.0,
                "end_seconds": index * 10.0 + 2.0,
                "duration_seconds": 2.0,
                "smoothed_score": 0.95 - index * 0.01,
                "importance_score": 0.95 - index * 0.01,
            })

        result = select_narrative_shots(
            shots,
            score_key="smoothed_score",
            category="importance",
            target_duration_sec=40.0,
            video_duration=100.0,
            segment_config={
                "min_duration": 5.0,
                "max_duration": 15.0,
                "pre_context": 2.0,
                "post_context": 2.5,
            },
        )
        self.assertLessEqual(len(result["selected_shot_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
