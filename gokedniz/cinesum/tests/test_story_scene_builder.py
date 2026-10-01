import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.story.story_scene_builder import (
    build_story_scenes,
    build_story_scenes_from_files,
)


class TestStorySceneBuilder(unittest.TestCase):
    def setUp(self):
        self.scenes = [
            {
                "scene_id": index + 1,
                "start_seconds": index * 5.0,
                "end_seconds": (index + 1) * 5.0,
                "duration_seconds": 5.0,
            }
            for index in range(4)
        ]
        self.clip_vectors = np.asarray([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 1.0, 0.0],
            [0.01, 0.99, 0.0],
        ], dtype=np.float32)
        self.clip_metadata = [
            {"scene_id": index + 1, "feature_vector_index": index}
            for index in range(4)
        ]
        self.audio_features = [
            {"scene_id": 1, "transcript_text": "same conversation", "speech_ratio": 0.9},
            {"scene_id": 2, "transcript_text": "same conversation", "speech_ratio": 0.8},
            {"scene_id": 3, "transcript_text": "car driving", "speech_ratio": 0.1},
            {"scene_id": 4, "transcript_text": "car driving", "speech_ratio": 0.1},
        ]

    def test_groups_visually_and_textually_continuous_shots(self):
        result, embeddings = build_story_scenes(
            self.scenes,
            self.clip_vectors,
            self.clip_metadata,
            self.audio_features,
        )

        self.assertEqual(result["story_scene_count"], 2)
        self.assertEqual(result["story_scenes"][0]["shot_ids"], [1, 2])
        self.assertEqual(result["story_scenes"][1]["shot_ids"], [3, 4])
        self.assertEqual(embeddings.shape, (2, 3))
        self.assertEqual(
            result["story_scenes"][0]["transcript_text"],
            "same conversation",
        )

    def test_max_story_duration_forces_a_split(self):
        result, _ = build_story_scenes(
            self.scenes,
            np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
            self.clip_metadata,
            [{"scene_id": index + 1, "transcript_text": "same", "speech_ratio": 0.5} for index in range(4)],
            max_story_duration_sec=9.0,
        )
        self.assertEqual(result["story_scene_count"], 4)

    def test_file_entrypoint_persists_json_and_embeddings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenes_path = root / "scenes.json"
            clip_path = root / "clip.npy"
            metadata_path = root / "clip_metadata.json"
            audio_path = root / "audio.json"
            output_json = root / "story.json"
            output_npy = root / "story.npy"

            scenes_path.write_text(json.dumps(self.scenes), encoding="utf-8")
            np.save(str(clip_path), self.clip_vectors)
            metadata_path.write_text(json.dumps(self.clip_metadata), encoding="utf-8")
            audio_path.write_text(json.dumps(self.audio_features), encoding="utf-8")

            result = build_story_scenes_from_files(
                scenes_path,
                clip_path,
                metadata_path,
                audio_path,
                output_json,
                output_npy,
            )

            self.assertTrue(output_json.exists())
            self.assertTrue(output_npy.exists())
            self.assertEqual(result["story_scene_count"], 2)


if __name__ == "__main__":
    unittest.main()
