import sys
import unittest
from unittest.mock import patch
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.summary.temporal_segment_builder import (
    SummarySegment,
    _segment_knapsack,
    add_protected_event_chains,
    build_temporal_segments,
    smooth_shot_scores,
)

class TestTemporalSegmentBuilder(unittest.TestCase):

    def test_smooth_shot_scores(self):
        shots = [
            {"scene_id": 1, "action_score": 0.20},
            {"scene_id": 2, "action_score": 0.30},
            {"scene_id": 3, "action_score": 0.90},
            {"scene_id": 4, "action_score": 0.70},
            {"scene_id": 5, "action_score": 0.20},
        ]
        smoothed = smooth_shot_scores(shots, "action_score", weights=[0.20, 0.60, 0.20])
        self.assertEqual(len(smoothed), 5)
        self.assertIn("raw_score", smoothed[2])
        self.assertIn("smoothed_score", smoothed[2])
        self.assertEqual(smoothed[2]["raw_score"], 0.90)
        # 0.20*0.30 + 0.60*0.90 + 0.20*0.70 = 0.06 + 0.54 + 0.14 = 0.74
        self.assertAlmostEqual(smoothed[2]["smoothed_score"], 0.74, places=2)

    def test_dense_action_merging(self):
        # 20 adjacent short shots (0.8s each) with high scores
        shots = []
        for i in range(1, 21):
            s_time = (i - 1) * 0.8
            e_time = i * 0.8
            shots.append({
                "scene_id": i,
                "start_seconds": s_time,
                "end_seconds": e_time,
                "duration_seconds": 0.8,
                "action_score": 0.85 + (i % 3) * 0.03,
            })

        res = build_temporal_segments(shots, category="action", target_duration_sec=16.0)
        segments = res["segments"]

        # Should merge adjacent short action shots into 1-3 coherent segments instead of 20 rapid cuts!
        self.assertLessEqual(len(segments), 3)
        self.assertGreater(len(segments), 0)
        for seg in segments:
            self.assertGreaterEqual(seg["duration"], 4.0)

    def test_isolated_peak_context_expansion(self):
        # Single isolated high score short shot
        shots = [
            {"scene_id": 1, "start_seconds": 0.0, "end_seconds": 10.0, "duration_seconds": 10.0, "importance_score": 0.20},
            {"scene_id": 2, "start_seconds": 10.0, "end_seconds": 11.2, "duration_seconds": 1.2, "importance_score": 0.95},
            {"scene_id": 3, "start_seconds": 11.2, "end_seconds": 25.0, "duration_seconds": 13.8, "importance_score": 0.25},
        ]
        res = build_temporal_segments(shots, category="importance", target_duration_sec=10.0)
        segments = res["segments"]

        self.assertEqual(len(segments), 1)
        # Context expanded to >= min segment duration (5.0s for importance)
        self.assertGreaterEqual(segments[0]["duration"], 5.0)

    def test_speech_boundary_preservation(self):
        shots = [
            {"scene_id": 1, "start_seconds": 0.0, "end_seconds": 10.0, "duration_seconds": 10.0, "speech_ratio": 0.0, "transcript_text": "", "dialogue_score": 0.10},
            {"scene_id": 2, "start_seconds": 10.0, "end_seconds": 18.0, "duration_seconds": 8.0, "speech_ratio": 0.85, "transcript_text": "Where are you going tonight?", "dialogue_score": 0.90},
            {"scene_id": 3, "start_seconds": 18.0, "end_seconds": 30.0, "duration_seconds": 12.0, "speech_ratio": 0.0, "transcript_text": "", "dialogue_score": 0.15},
        ]
        res = build_temporal_segments(shots, category="dialogue", target_duration_sec=12.0)
        segments = res["segments"]

        self.assertEqual(len(segments), 1)
        self.assertTrue(segments[0]["has_speech"])
        self.assertIn("Where are you going tonight?", segments[0]["transcript_text"])

    def test_word_timestamps_and_boundary_debug_metadata(self):
        words = [
            {"start": 1.0, "end": 1.5, "word": "Stay"},
            {"start": 1.6, "end": 2.2, "word": "here."},
        ]
        shots = [{
            "scene_id": 1,
            "start_seconds": 0.0,
            "end_seconds": 8.0,
            "duration_seconds": 8.0,
            "speech_ratio": 0.9,
            "transcript_text": "Stay here.",
            "words": words,
            "importance_score": 0.95,
        }]
        result = build_temporal_segments(
            shots, category="importance", target_duration_sec=8.0
        )
        segment = result["segments"][0]
        self.assertEqual(result["summary_algorithm_version"], "3.9.0")
        self.assertEqual(segment["word_count"], 2)
        self.assertIn(segment["boundary_mode"], {"shot", "sentence", "word"})
        self.assertIn("complete_utterance", segment)
        self.assertIn("budget_trimmed", segment)

    def test_rag_llm_mode_calls_optional_reranker(self):
        shots = [
            {"scene_id": index + 1, "start_seconds": index * 8.0,
             "end_seconds": index * 8.0 + 6.0, "duration_seconds": 6.0,
             "importance_score": 0.9 - index * 0.03}
            for index in range(8)
        ]

        def keep_local(all_shots, local_selection, **kwargs):
            result = dict(local_selection)
            result["llm_selection"] = {"status": "applied"}
            return result

        with patch(
            "src.summary.temporal_segment_builder.rerank_narrative_shots",
            side_effect=keep_local,
        ) as reranker:
            result = build_temporal_segments(
                shots,
                category="importance",
                target_duration_sec=20.0,
                narrative_mode="rag_llm",
                base_dir=BASE_DIR,
                video_alias="sample",
            )
        reranker.assert_called_once()
        self.assertEqual(result["narrative_mode"], "rag_llm")
        self.assertEqual(result["selection"]["llm_selection"]["status"], "applied")

    def test_segment_overlapping_merge(self):
        shots = [
            {"scene_id": 1, "start_seconds": 20.0, "end_seconds": 24.0, "duration_seconds": 4.0, "action_score": 0.88},
            {"scene_id": 2, "start_seconds": 24.0, "end_seconds": 25.0, "duration_seconds": 1.0, "action_score": 0.85},
            {"scene_id": 3, "start_seconds": 25.0, "end_seconds": 31.0, "duration_seconds": 6.0, "action_score": 0.90},
        ]
        res = build_temporal_segments(shots, category="action", target_duration_sec=15.0)
        segments = res["segments"]

        # Overlapping/adjacent segments within merge_gap merge into a single 20.0 -> 31.0 segment
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["start"], 20.0, places=1)
        self.assertAlmostEqual(segments[0]["end"], 31.0, places=1)

    def test_chronological_ordering(self):
        # Candidate ranking starts out of order, final export must be chronological
        shots = [
            {"scene_id": 1, "start_seconds": 120.0, "end_seconds": 128.0, "duration_seconds": 8.0, "importance_score": 0.95}, # minute 2
            {"scene_id": 2, "start_seconds": 10.0, "end_seconds": 18.0, "duration_seconds": 8.0, "importance_score": 0.90},   # second 10
            {"scene_id": 3, "start_seconds": 300.0, "end_seconds": 308.0, "duration_seconds": 8.0, "importance_score": 0.85}, # minute 5
        ]
        res = build_temporal_segments(shots, category="importance", target_duration_sec=30.0)
        segments = res["segments"]

        starts = [s["start"] for s in segments]
        self.assertEqual(starts, sorted(starts))

    def test_protected_event_chain_extends_to_visible_payoff(self):
        shots = [{
            "scene_id": index + 1,
            "start_seconds": index * 5.0,
            "end_seconds": index * 5.0 + 4.0,
            "duration_seconds": 4.0,
            "importance_score": 0.5,
            "narrative_event_score": 0.2,
            "action_score": 0.2,
        } for index in range(8)]
        shots[3]["hybrid_narrative_score"] = 0.98
        shots[3]["narrative_event_score"] = 0.96
        shots[5]["narrative_event_score"] = 0.99
        shots[5]["action_score"] = 0.95
        protected, ids = add_protected_event_chains(
            [shots[3]], shots, video_duration=40.0, target_duration_sec=180.0
        )
        self.assertEqual(ids, [4])
        self.assertTrue(protected[0]["protected_narrative_chain"])
        self.assertGreaterEqual(protected[0]["event_chain_end"], shots[5]["end_seconds"])

    def test_knapsack_keeps_protected_event_anchor(self):
        protected = SummarySegment(
            start=0, end=10, duration=10, category="importance",
            segment_score=0.1, protected_narrative_chain=True,
            narrative_chain_role="event_anchor",
        )
        ordinary = SummarySegment(
            start=20, end=30, duration=10, category="importance", segment_score=1.0,
        )
        selected = _segment_knapsack([protected, ordinary], 10.0)
        self.assertEqual(selected, [protected])

    def test_high_credit_probability_shots_excluded_from_every_category(self):
        shots = [
            {"scene_id": 1, "start_seconds": 0.0, "end_seconds": 8.0,
             "duration_seconds": 8.0, "action_score": 0.95, "dialogue_score": 0.95,
             "importance_score": 0.95, "credit_probability": 0.90},
            {"scene_id": 2, "start_seconds": 8.0, "end_seconds": 16.0,
             "duration_seconds": 8.0, "action_score": 0.90, "dialogue_score": 0.90,
             "importance_score": 0.90, "credit_probability": 0.05},
        ]
        for category in ("action", "dialogue", "importance", "custom"):
            res = build_temporal_segments(
                [dict(s) for s in shots], category=category, target_duration_sec=20.0,
            )
            self.assertEqual(res["excluded_credit_shots"], 1)
            selected_starts = {
                shot_id
                for seg in res["segments"]
                for shot_id in seg["core_shot_ids"] + seg["context_shot_ids"]
            }
            self.assertNotIn(1, selected_starts)

    def test_budget_trimmed_tail_keeps_acoustic_padding(self):
        words = [
            {"start": 1.0, "end": 1.4, "word": "One."},
            {"start": 2.0, "end": 2.4, "word": "Two."},
            {"start": 5.0, "end": 5.4, "word": "Three."},
            {"start": 5.6, "end": 6.3, "word": "Four."},
        ]
        shots = [{
            "scene_id": 1,
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "duration_seconds": 10.0,
            "speech_ratio": 0.8,
            "transcript_text": "One. Two. Three. Four.",
            "words": words,
            "dialogue_score": 0.95,
        }]
        res = build_temporal_segments(shots, category="dialogue", target_duration_sec=6.5)
        segment = res["segments"][0]
        # The safe sentence boundary within budget is 6.3s ("Four."); a
        # budget-trimmed speech segment must still carry its trailing acoustic pad
        # instead of ending exactly on the word timestamp (which can otherwise
        # clip the tail of the last word).
        self.assertTrue(segment["budget_trimmed"])
        self.assertGreater(segment["end"], 6.3)

if __name__ == "__main__":
    unittest.main()
