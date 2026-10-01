import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web_app


class TestSummaryAutoAnalysis(unittest.TestCase):
    def test_missing_features_trigger_analysis_before_scoring(self):
        scored = [{
            "scene_id": 1,
            "start_timecode": "00:00",
            "end_timecode": "00:05",
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "duration_seconds": 5.0,
            "importance_score": 0.9,
        }]
        missing = {"ready": False, "missing": ["clip_features"]}
        analysis_result = {
            "scenes_found": 1,
            "story_scenes_found": 1,
            "cache_hits": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(web_app, "BASE_DIR", Path(temp_dir)),
                patch.object(web_app, "get_analysis_artifact_status", return_value=missing),
                patch.object(web_app, "analyze_video_features", return_value=analysis_result) as analyze,
                patch.object(web_app, "compute_scene_scores_for_video", return_value=scored) as scoring,
                patch.object(web_app, "export_category_summary", return_value="summary.mp4"),
            ):
                result = web_app._sync_generate_summary(
                    "autotest",
                    "importance",
                    "",
                    30.0,
                    "task-1",
                    "balanced",
                )

        self.assertTrue(result["analysis_performed"])
        self.assertEqual(result["analysis"], analysis_result)
        analyze.assert_called_once()
        scoring.assert_called_once()

    def test_ready_features_skip_analysis(self):
        scored = [{
            "scene_id": 1,
            "start_timecode": "00:00",
            "end_timecode": "00:05",
            "start_seconds": 0.0,
            "end_seconds": 5.0,
            "duration_seconds": 5.0,
            "importance_score": 0.9,
        }]
        ready = {"ready": True, "missing": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(web_app, "BASE_DIR", Path(temp_dir)),
                patch.object(web_app, "get_analysis_artifact_status", return_value=ready),
                patch.object(web_app, "analyze_video_features") as analyze,
                patch.object(web_app, "compute_scene_scores_for_video", return_value=scored),
                patch.object(web_app, "export_category_summary", return_value="summary.mp4"),
            ):
                result = web_app._sync_generate_summary(
                    "autotest",
                    "importance",
                    "",
                    30.0,
                    "task-2",
                    "balanced",
                )

        self.assertFalse(result["analysis_performed"])
        analyze.assert_not_called()


if __name__ == "__main__":
    unittest.main()
