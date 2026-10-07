import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web_app


class TestProgressRecovery(unittest.TestCase):
    def test_disk_status_counts_live_keyframes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            video = base / "dataset/video/video3.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            manifest = base / "outputs/manifests/video3.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({
                "updated_at_unix": 1,
                "stages": {
                    "scene_detection": {"status": "complete"},
                    "keyframes": {"status": "running"},
                },
            }))
            scenes = base / "outputs/pyscenedetect/scene_lists/video3_scenes.json"
            scenes.parent.mkdir(parents=True)
            scenes.write_text(json.dumps([
                {"start_frame": 0, "end_frame": 100, "duration_seconds": 4},
                {"start_frame": 100, "end_frame": 400, "duration_seconds": 12},
            ]))
            frames = base / "outputs/pyscenedetect/keyframes/video3"
            frames.mkdir(parents=True)
            (frames / "video3_scene_001_kf1.jpg").write_bytes(b"jpg")
            with patch.object(web_app, "BASE_DIR", base):
                status = web_app.get_analysis_disk_status("video3")

        self.assertEqual(status["status"], "running")
        self.assertEqual(status["stage"], "keyframes")
        self.assertEqual(status["scene_count"], 2)
        self.assertEqual(status["keyframes_done"], 1)
        self.assertEqual(status["keyframes_total"], 4)

    def test_task_metadata_survives_progress_update(self):
        task_id = "recovery-test"
        web_app.set_task_metadata(task_id, kind="upload", video_alias="video3")
        web_app.update_task_progress(task_id, 30, "Kareler çıkarılıyor")
        self.assertEqual(web_app.progress_store[task_id]["video_alias"], "video3")
        web_app.progress_store.pop(task_id, None)


if __name__ == "__main__":
    unittest.main()
