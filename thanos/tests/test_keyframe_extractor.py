import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.scene_detection.keyframe_extractor import (
    calculate_blur_score,
    extract_keyframes_for_scenes,
)


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
    def test_sequential_sampling_matches_random_seek_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "sample.avi"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48)
            )
            self.assertTrue(writer.isOpened())
            checker = ((np.indices((48, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
            for frame_number in range(60):
                brightness = checker if frame_number in {12, 26, 43} else np.full((48, 64), 100, np.uint8)
                writer.write(cv2.cvtColor(brightness, cv2.COLOR_GRAY2BGR))
            writer.release()

            scenes = [{"scene_id": 1, "start_frame": 0, "end_frame": 20,
                       "duration_seconds": 2.0},
                      {"scene_id": 2, "start_frame": 20, "end_frame": 60,
                       "duration_seconds": 4.0}]
            result = extract_keyframes_for_scenes(str(video), scenes, str(root), sample_step=2)

            capture = cv2.VideoCapture(str(video))
            expected = []
            for start, end in [(6, 14), (32, 48)]:
                scores = []
                for frame_number in range(start, end + 1, 2):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ok, frame = capture.read()
                    if ok:
                        scores.append((calculate_blur_score(frame), frame_number))
                expected.append(max(scores, key=lambda item: item[0])[1])
            capture.release()

            self.assertEqual([row["frame_number"] for row in result], expected)
            self.assertTrue(all(Path(row["file_path"]).exists() for row in result))

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
