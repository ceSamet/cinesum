import unittest

from src.llm.narrative_candidates import (
    build_narrative_candidate_pool,
    narrative_candidate_limit,
    serialize_candidate,
)


class TestNarrativeCandidates(unittest.TestCase):
    def _shots(self, count=100):
        return [{
            "scene_id": index + 1,
            "start_seconds": float(index * 5),
            "end_seconds": float(index * 5 + 4),
            "importance_score": (index % 17) / 17,
            "narrative_event_score": ((index * 7) % 19) / 19,
            "story_scene_id": index // 3,
            "transcript_text": "x" * 2000,
            "words": [{"word": "secretly large"}],
        } for index in range(count)]

    def test_pool_is_bounded_deduplicated_and_keeps_local_first(self):
        pool = build_narrative_candidate_pool(self._shots(), [50, 50, 1])
        ids = [row["scene_id"] for row in pool]
        self.assertLessEqual(len(pool), 32)
        self.assertEqual(ids[:2], [50, 1])
        self.assertEqual(len(ids), len(set(ids)))

    def test_transcript_is_bounded_and_embeddings_are_not_serialized(self):
        pool = build_narrative_candidate_pool(self._shots(10), [1])
        pool[0]["related_context"] = [{
            "document_id": "story:2",
            "score": 0.75,
            "content": "y" * 1000,
            "private_vector": [1, 2, 3],
        }]
        pool[0]["visual_caption"] = "LLaVA: a decisive action " * 50
        pool[0]["visual_caption_source"] = "blip_llava"
        payload = serialize_candidate(pool[0])
        self.assertLessEqual(len(payload["transcript"]), 300)
        self.assertLessEqual(len(payload["related_context"][0]["content"]), 180)
        self.assertNotIn("private_vector", payload["related_context"][0])
        self.assertNotIn("words", payload)
        self.assertLessEqual(len(payload["visual_caption"]), 420)
        self.assertEqual(payload["visual_caption_source"], "blip_llava")

    def test_candidate_limit_scales_with_video_duration(self):
        cases = [
            (10 * 60, 16),
            (10 * 60 + 1, 24),
            (30 * 60, 24),
            (60 * 60, 32),
            (90 * 60, 40),
            (2 * 60 * 60, 48),
            (150 * 60 + 1, 56),
            (165 * 60 + 1, 72),
            (210 * 60 + 1, 72),
        ]
        for duration_seconds, expected in cases:
            with self.subTest(duration_seconds=duration_seconds):
                self.assertEqual(narrative_candidate_limit(duration_seconds), expected)

    def test_pool_accepts_dynamic_long_video_limit(self):
        pool = build_narrative_candidate_pool(
            self._shots(100),
            [1],
            max_candidates=narrative_candidate_limit(2 * 60 * 60),
        )
        self.assertEqual(len(pool), 48)

    def test_long_pool_reserves_temporal_and_story_coverage(self):
        pool = build_narrative_candidate_pool(
            self._shots(150),
            list(range(1, 80)),
            max_candidates=56,
        )
        sources = {row["candidate_source"] for row in pool}
        self.assertIn("local_selected", sources)
        self.assertIn("temporal_bin", sources)
        self.assertIn("event_peak", sources)
        self.assertIn("strong_story_scene", sources)
        self.assertEqual(len(pool), 56)


if __name__ == "__main__":
    unittest.main()
