import unittest

from src.summary.transition_manager import get_audio_fade_filter


class TestTransitionManager(unittest.TestCase):
    def test_speech_segments_do_not_fade_first_or_last_syllable(self):
        self.assertEqual(
            get_audio_fade_filter(12.0, preserve_speech=True),
            "",
        )

    def test_non_speech_segment_keeps_boundary_fades(self):
        value = get_audio_fade_filter(12.0, preserve_speech=False)
        self.assertIn("afade=t=in", value)
        self.assertIn("afade=t=out", value)


if __name__ == "__main__":
    unittest.main()
