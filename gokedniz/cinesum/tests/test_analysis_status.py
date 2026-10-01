import tempfile
import unittest
from pathlib import Path

from src.core.analysis_status import get_analysis_artifact_status


class TestAnalysisArtifactStatus(unittest.TestCase):
    def test_reports_missing_core_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status = get_analysis_artifact_status(Path(temp_dir), "video1")
            self.assertFalse(status["ready"])
            self.assertEqual(
                set(status["missing"]),
                {"scene_list", "clip_features", "clip_metadata", "audio_features"},
            )

    def test_ready_when_all_core_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            status = get_analysis_artifact_status(root, "video1")
            for path in status["required"].values():
                artifact = Path(path)
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"artifact")

            refreshed = get_analysis_artifact_status(root, "video1")
            self.assertTrue(refreshed["ready"])
            self.assertEqual(refreshed["missing"], [])


if __name__ == "__main__":
    unittest.main()
