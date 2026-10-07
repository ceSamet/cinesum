import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.scene_detection.keyframe_extractor import (
    calculate_blur_score,
    extract_keyframes_for_scenes,
)


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


if __name__ == "__main__":
    unittest.main()
