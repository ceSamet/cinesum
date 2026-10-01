import os
import unittest
from unittest.mock import patch

from src.core.vlm_config import load_vlm_config


class TestVLMConfig(unittest.TestCase):
    def test_defaults_are_bounded_hybrid_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_vlm_config(load_env_file=False)
        self.assertTrue(config.enabled)
        self.assertEqual(config.max_frames, 40)
        self.assertEqual(config.llava_max_frames, 12)
        self.assertEqual(config.time_budget_sec, 60.0)

    def test_invalid_frame_limit_is_rejected(self):
        with patch.dict(os.environ, {"VLM_MAX_FRAMES": "100"}, clear=True):
            with self.assertRaisesRegex(ValueError, "VLM_MAX_FRAMES"):
                load_vlm_config(load_env_file=False)


if __name__ == "__main__":
    unittest.main()
