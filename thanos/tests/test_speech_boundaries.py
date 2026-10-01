import unittest

from src.summary.speech_boundaries import (
    align_segment_boundaries,
    build_speech_context,
    enforce_duration_ceiling,
    find_safe_budget_end,
)


def make_shot(words, start=0.0, end=12.0):
    return {
        "scene_id": 1,
        "start_seconds": start,
        "end_seconds": end,
        "speech_ratio": 0.9 if words else 0.0,
        "transcript_text": " ".join(word["word"] for word in words),
        "words": words,
    }


class TestSpeechBoundaries(unittest.TestCase):
    def test_expands_importance_segment_to_complete_sentence(self):
        words = [
            {"start": 1.0, "end": 1.4, "word": "Hello"},
            {"start": 1.5, "end": 2.0, "word": "there."},
        ]
        result = align_segment_boundaries(
            1.2, 1.8, [make_shot(words)],
            category="importance", max_segment_duration=10.0,
        )
        self.assertEqual((result["start"], result["end"]), (1.0, 2.0))
        self.assertEqual(result["boundary_mode"], "sentence")
        self.assertTrue(result["complete_utterance"])

    def test_silence_gap_closes_utterance(self):
        words = [
            {"start": 0.0, "end": 0.4, "word": "First"},
            {"start": 3.3, "end": 3.7, "word": "Second"},
        ]
        context = build_speech_context([make_shot(words)])
        self.assertEqual(len(context.utterances), 2)
        self.assertEqual(context.utterances[0]["closed_by"], "silence")

    def test_dramatic_pause_does_not_split_an_unfinished_sentence(self):
        shots = [make_shot([
            {"start": 1.0, "end": 1.4, "word": "Bunu"},
            {"start": 1.5, "end": 2.0, "word": "yapmak"},
            {"start": 4.2, "end": 4.7, "word": "zorundayız."},
        ])]
        context = build_speech_context(shots)
        self.assertEqual(len(context.utterances), 1)

    def test_question_expands_through_the_next_response(self):
        shots = [make_shot([
            {"start": 1.0, "end": 1.5, "word": "Gelecek"},
            {"start": 1.6, "end": 2.0, "word": "misin?"},
            {"start": 3.0, "end": 3.4, "word": "Evet,"},
            {"start": 3.5, "end": 4.0, "word": "geleceğim."},
        ])]
        result = align_segment_boundaries(
            0.5,
            2.2,
            shots,
            category="importance",
            max_segment_duration=30.0,
        )
        self.assertEqual(result["boundary_mode"], "exchange")
        self.assertAlmostEqual(result["end"], 4.0)
        self.assertIn("expanded_question_to_response", result["boundary_reason"])

    def test_unanswered_question_trims_the_silent_reaction_tail(self):
        shots = [make_shot([
            {"start": 1.0, "end": 1.5, "word": "Hazır"},
            {"start": 1.6, "end": 2.0, "word": "mısın?"},
        ], end=7.0)]
        result = align_segment_boundaries(
            0.5,
            6.0,
            shots,
            category="importance",
            max_segment_duration=30.0,
        )
        self.assertAlmostEqual(result["end"], 2.0)
        self.assertIn("trimmed_unanswered_question_reaction", result["boundary_reason"])

    def test_long_utterance_is_forced_complete_when_it_fits_maximum(self):
        words = [
            {
                "start": float(second),
                "end": float(second) + 0.5,
                "word": "cümledir." if second == 27 else f"kelime{second}",
            }
            for second in range(1, 28, 2)
        ]
        shots = [make_shot(words, end=30.0)]
        result = align_segment_boundaries(
            13.0,
            16.0,
            shots,
            category="importance",
            max_segment_duration=30.0,
        )
        self.assertEqual((result["start"], result["end"]), (1.0, 27.5))
        self.assertTrue(result["complete_utterance"])
        self.assertIn("forced_complete_utterance_start", result["boundary_reason"])

    def test_action_mode_avoids_mid_word_without_sentence_expansion(self):
        words = [
            {"start": 1.0, "end": 1.5, "word": "Run"},
            {"start": 1.6, "end": 2.2, "word": "now."},
        ]
        result = align_segment_boundaries(
            1.2, 1.9, [make_shot(words)],
            category="action", max_segment_duration=10.0,
        )
        self.assertEqual((result["start"], result["end"]), (1.0, 2.2))
        self.assertEqual(result["boundary_mode"], "word")

    def test_sentence_expansion_respects_category_limit(self):
        words = [
            {"start": 0.0, "end": 0.4, "word": "A"},
            {"start": 4.0, "end": 4.4, "word": "sentence."},
        ]
        result = align_segment_boundaries(
            2.0, 4.2, [make_shot(words)],
            category="importance", max_segment_duration=10.0,
        )
        self.assertGreaterEqual(result["start"], 2.0)
        self.assertEqual(result["end"], 4.4)

    def test_duplicate_words_across_shots_are_removed(self):
        word = {"start": 1.0, "end": 1.4, "word": "same", "probability": 0.8}
        context = build_speech_context([
            make_shot([word], 0.0, 2.0),
            make_shot([{**word, "probability": 0.9}], 1.0, 3.0),
        ])
        self.assertEqual(len(context.words), 1)
        self.assertEqual(context.words[0]["probability"], 0.9)

    def test_missing_words_uses_transcript_fallback(self):
        shot = make_shot([], 4.0, 9.0)
        shot.update({"speech_ratio": 0.8, "transcript_text": "fallback text"})
        result = align_segment_boundaries(
            5.0, 8.0, [shot],
            category="dialogue", max_segment_duration=10.0,
        )
        self.assertEqual(result["boundary_mode"], "fallback")
        self.assertEqual(result["transcript_text"], "fallback text")

    def test_budget_trim_uses_sentence_end(self):
        words = [
            {"start": 0.0, "end": 1.0, "word": "One."},
            {"start": 1.2, "end": 3.0, "word": "Two."},
            {"start": 3.2, "end": 6.0, "word": "Three."},
        ]
        result = find_safe_budget_end(
            0.0, 4.0, [make_shot(words)],
            category="dialogue", min_duration=2.0,
        )
        self.assertEqual(result["end"], 3.0)
        self.assertTrue(result["complete_utterance"])

    def test_budget_trim_returns_none_without_complete_sentence(self):
        words = [
            {"start": 0.0, "end": 1.0, "word": "This"},
            {"start": 5.0, "end": 6.0, "word": "continues."},
        ]
        result = find_safe_budget_end(
            0.0, 4.0, [make_shot(words)],
            category="importance", min_duration=2.0,
        )
        self.assertIsNone(result)

    def test_duration_ceiling_never_exceeds_target(self):
        # Three segments that individually respect budget rules but sum past it
        # (mirrors the real-world case: budget loop tolerance + adjacent re-merge +
        # targeted ASR re-expansion each add a little, none re-check the total).
        shots = [
            {"scene_id": 1, "start_seconds": 200.0, "end_seconds": 260.0},
            {"scene_id": 2, "start_seconds": 260.0, "end_seconds": 320.0},
        ]
        segments = [
            {"start": 0.0, "end": 100.0, "duration": 100.0, "has_speech": False,
             "boundary_reason": "shot_boundary"},
            {"start": 100.0, "end": 200.0, "duration": 100.0, "has_speech": False,
             "boundary_reason": "shot_boundary"},
            {"start": 200.0, "end": 320.0, "duration": 120.0, "has_speech": False,
             "boundary_reason": "shot_boundary"},
        ]
        kept, info = enforce_duration_ceiling(segments, 300.0, shots=shots, category="importance")
        self.assertLessEqual(sum(s["duration"] for s in kept), 300.0)
        self.assertTrue(info["applied"])

    def test_duration_ceiling_trims_at_sentence_boundary_not_mid_word(self):
        words = [
            {"start": 95.0, "end": 96.0, "word": "One."},
            {"start": 96.5, "end": 98.5, "word": "Two."},
            {"start": 99.0, "end": 103.0, "word": "Three."},
        ]
        shot = make_shot(words, start=90.0, end=110.0)
        segments = [
            {"start": 90.0, "end": 103.0, "duration": 13.0, "has_speech": True,
             "boundary_reason": "sentence", "original_end": 103.0},
        ]
        kept, info = enforce_duration_ceiling(
            segments, 8.0, shots=[shot], category="importance",
        )
        self.assertEqual(len(kept), 1)
        # Must land exactly on a sentence end, never inside "Two." or "Three."
        self.assertIn(kept[0]["end"], (96.0, 98.5))
        self.assertTrue(kept[0]["complete_utterance"])
        self.assertTrue(info["trimmed_last_segment"])

    def test_duration_ceiling_drops_segments_that_cannot_fit_safely(self):
        segments = [
            {"start": 0.0, "end": 10.0, "duration": 10.0, "has_speech": False,
             "boundary_reason": "shot_boundary"},
            {"start": 10.0, "end": 20.0, "duration": 10.0, "has_speech": False,
             "boundary_reason": "shot_boundary"},
        ]
        kept, info = enforce_duration_ceiling(segments, 10.0, shots=[], category="importance")
        self.assertEqual(len(kept), 1)
        self.assertEqual(info["dropped_segment_count"], 1)

    def test_duration_ceiling_is_noop_for_unbounded_target(self):
        segments = [
            {"start": 0.0, "end": 10.0, "duration": 10.0, "has_speech": False,
             "boundary_reason": "shot_boundary"},
        ]
        kept, info = enforce_duration_ceiling(segments, 0.0, shots=[])
        self.assertEqual(kept, segments)
        self.assertFalse(info["applied"])


if __name__ == "__main__":
    unittest.main()
