import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.rag.scene_rag import ensure_scene_rag, retrieve_related_scenes


class TestSceneRAG(unittest.TestCase):
    def _write_artifacts(self, root: Path) -> None:
        story_dir = root / "outputs" / "features" / "story"
        story_dir.mkdir(parents=True)
        data = {
            "story_scenes": [
                {"story_scene_id": 1, "start_seconds": 0, "end_seconds": 10,
                 "duration_seconds": 10, "shot_ids": [1], "transcript_text": "hero loses the magic stone",
                 "speech_ratio": 0.8, "dominant_mode": "dialogue"},
                {"story_scene_id": 2, "start_seconds": 40, "end_seconds": 50,
                 "duration_seconds": 10, "shot_ids": [2], "transcript_text": "a quiet car drives away",
                 "speech_ratio": 0.0, "dominant_mode": "general"},
                {"story_scene_id": 3, "start_seconds": 90, "end_seconds": 100,
                 "duration_seconds": 10, "shot_ids": [3], "transcript_text": "hero finds the magic stone again",
                 "speech_ratio": 0.8, "dominant_mode": "dialogue"},
            ]
        }
        (story_dir / "sample_story_scenes.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        np.save(
            story_dir / "sample_story_embeddings.npy",
            np.asarray([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
        )

    def test_index_is_built_then_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_artifacts(root)
            first = ensure_scene_rag(root, "sample")
            second = ensure_scene_rag(root, "sample")
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(first["source_hash"], second["source_hash"])
            connection = sqlite3.connect(first["database_path"])
            try:
                count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(count, 3)

    def test_related_distant_story_is_retrieved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_artifacts(root)
            index = ensure_scene_rag(root, "sample")
            results = retrieve_related_scenes(
                index["database_path"],
                "hero magic stone",
                query_visual_vector=np.asarray([1, 0], dtype=np.float32),
                query_time_seconds=5,
                top_k=1,
                exclude_document_ids=["story:1"],
            )
            self.assertEqual(results[0]["document_id"], "story:3")

    def test_source_change_rebuilds_only_rag_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_artifacts(root)
            first = ensure_scene_rag(root, "sample")
            story_path = root / "outputs/features/story/sample_story_scenes.json"
            data = json.loads(story_path.read_text(encoding="utf-8"))
            data["story_scenes"][0]["transcript_text"] += " changed"
            story_path.write_text(json.dumps(data), encoding="utf-8")
            second = ensure_scene_rag(root, "sample")
            self.assertFalse(second["cache_hit"])
            self.assertNotEqual(first["source_hash"], second["source_hash"])


if __name__ == "__main__":
    unittest.main()
