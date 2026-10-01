import unittest

from src.features.emotion_features import dialogue_cues, emotion_peaks


class EmotionFeaturesTests(unittest.TestCase):
    def test_negative_revelation_is_important_not_discarded(self):
        rows = [
            {"scene_id": 1, "start_seconds": 0, "transcript_text": "We are going home."},
            {"scene_id": 2, "start_seconds": 5, "transcript_text": "I must confess the truth. I betrayed you."},
            {"scene_id": 3, "start_seconds": 10, "transcript_text": "We are going home."},
        ]
        peaks = emotion_peaks(rows)
        self.assertGreater(peaks[2]["emotion_peak"], peaks[1]["emotion_peak"])
        self.assertLess(peaks[2]["polarity"], 0)
        self.assertGreater(peaks[2]["decision"], 0)

    def test_missing_text_has_no_claimed_emotion(self):
        self.assertEqual(dialogue_cues(""), {"polarity": 0, "intensity": 0, "decision": 0, "confidence": 0})


if __name__ == "__main__":
    unittest.main()
