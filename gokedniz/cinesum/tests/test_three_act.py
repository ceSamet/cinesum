import unittest

from src.selection.three_act import ACT_QUOTAS, act_for, act_overlap_durations, find_turning_points
from src.selection.narrative_selector import select_narrative_shots
from src.llm.narrative_candidates import build_narrative_candidate_pool
from src.summary.speech_boundaries import enforce_duration_ceiling
from src.summary.temporal_segment_builder import SummarySegment, add_protected_event_chains, select_balanced_narrative_segments


class ThreeActTests(unittest.TestCase):
    def test_time_position_ignores_misleading_shot_index(self):
        self.assertEqual(act_for({"start_seconds": 80, "end_seconds": 90, "story_position_ratio": 0.10}, 100), "act_iii")
        self.assertAlmostEqual(sum(ACT_QUOTAS.values()), 1.0)

    def test_cross_act_segment_seconds_are_split(self):
        self.assertEqual(
            act_overlap_durations([{"start": 20, "end": 30}, {"start": 70, "end": 80}], 100),
            {"act_i": 5.0, "act_ii": 10.0, "act_iii": 5.0},
        )

    def test_turning_point_requires_content_evidence(self):
        shots = [
            {"scene_id": 1, "start_seconds": 20, "end_seconds": 21, "importance_score": 0.99},
            {"scene_id": 2, "start_seconds": 50, "end_seconds": 51, "transcript_text": "We must decide the truth", "narrative_event_score": 0.9},
            {"scene_id": 3, "start_seconds": 88, "end_seconds": 89, "transcript_text": "We win and save everyone", "narrative_event_score": 0.95},
        ]
        result = find_turning_points(shots, 100)
        self.assertEqual(result["inciting_incident"]["status"], "unverified")
        self.assertEqual(result["midpoint"]["scene_id"], 2)
        self.assertEqual(result["climax"]["scene_id"], 3)

    def test_required_low_score_shot_survives_first_and_llm_candidate_selection(self):
        shots = [{"scene_id": i + 1, "start_seconds": i * 10.0, "end_seconds": i * 10.0 + 5,
                  "duration_seconds": 5.0, "importance_score": 0.01 if i == 8 else 0.9,
                  "story_scene_id": i + 1} for i in range(10)]
        result = select_narrative_shots(
            shots, score_key="importance_score", category="importance",
            target_duration_sec=30, video_duration=100,
            segment_config={"min_duration": 3, "max_duration": 10, "pre_context": 0, "post_context": 0},
            required_shot_ids=[9],
        )
        self.assertIn(9, result["selected_shot_ids"])
        pool = build_narrative_candidate_pool(shots, result["selected_shot_ids"], max_candidates=5, required_shot_ids=[9])
        self.assertEqual(pool[0]["scene_id"], 9)

    def test_anchor_survives_post_alignment_and_final_ceiling(self):
        optional = SummarySegment(0, 35, 35, "importance", segment_score=0.99, narrative_region="act_i")
        anchor = SummarySegment(85, 95, 10, "importance", segment_score=0.2, narrative_region="act_iii", hard_anchor_roles=["climax"])
        selected, meta = select_balanced_narrative_segments([optional, anchor], 40)
        self.assertIn(anchor, selected)
        self.assertEqual(meta["mandatory_duration"], 10)
        rows = [item.to_dict() for item in selected]
        final, info = enforce_duration_ceiling(rows, 40, [], category="importance")
        self.assertTrue(any("climax" in row.get("hard_anchor_roles", []) for row in final))
        self.assertLessEqual(info["post_ceiling_duration"], 40)

    def test_impossible_anchor_budget_is_explicit(self):
        anchor = SummarySegment(85, 125, 40, "importance", narrative_region="act_iii", hard_anchor_roles=["climax"])
        with self.assertRaisesRegex(ValueError, "en az"):
            select_balanced_narrative_segments([anchor], 30)

    def test_ceiling_removes_excess_from_overrepresented_act_first(self):
        rows = [
            {"start": 0, "end": 55, "duration": 55, "narrative_region": "act_i", "segment_score": 0.7},
            {"start": 260, "end": 410, "duration": 150, "narrative_region": "act_ii", "segment_score": 0.7},
            {"start": 800, "end": 880, "duration": 80, "narrative_region": "act_iii", "segment_score": 0.7, "protected_narrative_chain": True},
            {"start": 885, "end": 901, "duration": 16, "narrative_region": "act_iii", "segment_score": 0.5},
            {"start": 905, "end": 912, "duration": 7, "narrative_region": "act_iii", "segment_score": 0.5},
        ]
        final, _ = enforce_duration_ceiling(rows, 300, [], category="importance")
        self.assertEqual(sum(row["duration"] for row in final), 292)
        self.assertTrue(any(row.get("protected_narrative_chain") for row in final))
        self.assertTrue(any(row["narrative_region"] == "act_ii" for row in final))

    def test_late_event_chain_includes_visible_consequence(self):
        anchor = {"scene_id": 1, "start_seconds": 850, "end_seconds": 855,
                  "narrative_event_score": 0.97, "importance_score": 0.8}
        later = {"scene_id": 2, "start_seconds": 885, "end_seconds": 895,
                 "narrative_event_score": 0.95, "importance_score": 0.4}
        result, _ = add_protected_event_chains([anchor], [anchor, later], video_duration=1000, target_duration_sec=300)
        self.assertTrue(result[0]["late_resolution_chain"])
        self.assertEqual(result[0]["event_chain_end"], 895)


if __name__ == "__main__":
    unittest.main()
