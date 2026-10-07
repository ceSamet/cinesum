import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.scene_detection.keyframe_extractor import extract_keyframes_for_scenes


class FakeCapture:
    def __init__(self, frame_count=100):
        self.frame_count = frame_count
        self.position = 0
        self.current = None
        self.seek_count = 0

    def isOpened(self):
        return True

    def get(self, property_id):
        return 25.0 if property_id == cv2.CAP_PROP_FPS else 0.0

    def set(self, property_id, value):
        assert property_id == cv2.CAP_PROP_POS_FRAMES
        self.position = int(value)
        self.seek_count += 1
        return True

    def grab(self):
        if self.position >= self.frame_count:
            return False
        self.current = self.position
        self.position += 1
        return True

    def retrieve(self):
        return True, np.full((4, 4, 3), self.current, dtype=np.uint8)

    def read(self):
        if not self.grab():
            return False, None
        return self.retrieve()

    def release(self):
        pass


class TestKeyframeExtractor(unittest.TestCase):
    def test_same_windows_and_best_sample_with_one_seek_per_window(self):
        capture = FakeCapture()
        scenes = [
            {"scene_id": 1, "start_frame": 0, "end_frame": 20, "duration_seconds": 4},
            {"scene_id": 2, "start_frame": 20, "end_frame": 60, "duration_seconds": 12},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("src.scene_detection.keyframe_extractor.cv2.VideoCapture", return_value=capture),
                patch("src.scene_detection.keyframe_extractor.calculate_blur_score", side_effect=lambda frame: float(frame[0, 0, 0])),
            ):
                result = extract_keyframes_for_scenes("sample.mp4", scenes, temp_dir)

            self.assertEqual([item["frame_number"] for item in result], [14, 34, 46, 56])
            self.assertEqual(capture.seek_count, 4)
            self.assertTrue(all(Path(item["file_path"]).exists() for item in result))

    def test_invalid_sample_step_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_keyframes_for_scenes("sample.mp4", [], "unused", sample_step=0)


if __name__ == "__main__":
    unittest.main()
