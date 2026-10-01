import json
import tempfile
import unittest
from pathlib import Path

from src.core.feature_manifest import FeatureManifest, stable_config_hash
from src.core.pipeline_config import get_analysis_profile


class TestFeatureManifest(unittest.TestCase):
    def test_stage_cache_requires_matching_config_and_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            artifact = root / "scenes.json"
            manifest_path = root / "manifest.json"
            video.write_bytes(b"video-content")
            artifact.write_text("[]", encoding="utf-8")

            manifest = FeatureManifest(manifest_path, video)
            config = {"threshold": 3.0}
            manifest.mark_complete("scenes", config, [artifact], 1.25)

            self.assertTrue(manifest.stage_is_valid("scenes", config, [artifact]))
            self.assertFalse(
                manifest.stage_is_valid("scenes", {"threshold": 4.0}, [artifact])
            )

            artifact.unlink()
            self.assertFalse(manifest.stage_is_valid("scenes", config, [artifact]))

    def test_source_change_invalidates_all_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            artifact = root / "clip.npy"
            manifest_path = root / "manifest.json"
            video.write_bytes(b"version-one")
            artifact.write_bytes(b"features")

            manifest = FeatureManifest(manifest_path, video)
            manifest.mark_complete("clip", {"model": "clip"}, [artifact], 2.0)
            video.write_bytes(b"version-two")

            refreshed = FeatureManifest(manifest_path, video)
            self.assertEqual(refreshed.data["stages"], {})

    def test_manifest_is_json_and_records_timing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            artifact = root / "transcript.json"
            manifest_path = root / "manifest.json"
            video.write_bytes(b"video")
            artifact.write_text("{}", encoding="utf-8")

            manifest = FeatureManifest(manifest_path, video)
            manifest.mark_complete("transcript", {"model": "small"}, [artifact], 3.45678)
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(saved["stages"]["transcript"]["processing_sec"], 3.4568)
            self.assertEqual(
                saved["stages"]["transcript"]["config_hash"],
                stable_config_hash({"model": "small"}),
            )


class TestPipelineProfiles(unittest.TestCase):
    def test_balanced_profile_uses_faster_whisper_small(self):
        profile = get_analysis_profile("balanced")
        self.assertEqual(profile.whisper_model, "small")
        self.assertIsNone(profile.whisper_compute_type)
        self.assertIsNone(profile.clip_batch_size)
        self.assertTrue(profile.word_timestamps)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            get_analysis_profile("turbo-unknown")


if __name__ == "__main__":
    unittest.main()
