import unittest

import process_video


class TranscriptTests(unittest.TestCase):
    def test_uses_segment_when_word_timestamps_are_missing(self):
        transcript = [{"start": 0.0, "end": 1.0, "text": "Hello", "words": []}]
        turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

        self.assertEqual(
            process_video.build_speaker_transcript(transcript, turns),
            [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "SPEAKER_00",
                    "text": "Hello",
                    "type": "speech",
                }
            ],
        )

    def test_assigns_word_to_largest_overlap(self):
        word = {"start": 0.8, "end": 1.4}
        turns = [
            {"start": 0.0, "end": 1.0, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ]

        self.assertEqual(process_video.find_speaker(word, turns), "B")

    def test_explicit_override_splits_an_incorrect_automatic_segment(self):
        transcript = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "one two",
                "words": [
                    {"start": 0.0, "end": 0.8, "text": "one"},
                    {"start": 1.2, "end": 2.0, "text": "two"},
                ],
            }
        ]
        turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}]
        overrides = [{"start": 1.0, "end": 2.1, "speaker": "B"}]

        result = process_video.build_speaker_transcript(transcript, turns, overrides)

        self.assertEqual([item["speaker"] for item in result], ["A", "B"])

    def test_merges_consecutive_whisper_segments_for_same_speaker(self):
        transcript = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "First.",
                "words": [{"start": 0.0, "end": 1.0, "text": "First."}],
            },
            {
                "start": 1.2,
                "end": 2.0,
                "text": "Second.",
                "words": [{"start": 1.2, "end": 2.0, "text": "Second."}],
            },
        ]
        turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}]

        result = process_video.build_speaker_transcript(transcript, turns)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "First. Second.")

    def test_overlap_uses_nearest_corrected_split_identity(self):
        overlaps = [
            {"start": 10.0, "end": 10.5, "type": "overlap", "speakers": ["A", "B"]}
        ]
        turns = [
            {
                "start": 8.0,
                "end": 9.5,
                "speaker": "A_SPLIT",
                "original_speaker": "A",
            },
            {"start": 10.0, "end": 10.5, "speaker": "B"},
        ]

        result = process_video.update_overlap_speakers(overlaps, turns)

        self.assertEqual(result[0]["speakers"], ["A_SPLIT", "B"])

    def test_small_leading_boundary_drift_follows_rest_of_sentence(self):
        words = [
            {"start": 0.8, "end": 1.2, "text": "But"},
            {"start": 1.2, "end": 1.6, "text": "this"},
            {"start": 1.6, "end": 2.0, "text": "continues"},
        ]
        turns = [
            {"start": 0.0, "end": 1.05, "speaker": "OLD"},
            {"start": 1.05, "end": 2.1, "speaker": "NEW"},
        ]

        result = process_video.assign_word_speakers(words, turns)

        self.assertEqual(result, ["NEW", "NEW", "NEW"])


if __name__ == "__main__":
    unittest.main()
