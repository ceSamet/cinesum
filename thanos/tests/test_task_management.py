import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web_app


class TestTaskManagement(unittest.TestCase):
    def test_cancel_before_upload_handler_starts_is_preserved(self):
        task_id = "task_abcdefg_1234567890123"
        with tempfile.TemporaryDirectory() as directory, patch.object(web_app, "TASK_DIR", Path(directory) / "tasks"):
            web_app.TASK_DIR.mkdir()
            try:
                response = asyncio.run(web_app.cancel_task(task_id))
                self.assertEqual(response["status"], "cancel_requested")
                with web_app.progress_store_lock:
                    web_app.cancel_events.setdefault(task_id, threading.Event())
                    web_app.pending_cancel_times.pop(task_id, None)
                with self.assertRaises(web_app.TaskCancelled):
                    web_app._cancel_checkpoint(task_id)
            finally:
                with web_app.progress_store_lock:
                    web_app.cancel_events.pop(task_id, None)
                    web_app.pending_cancel_times.pop(task_id, None)

    def test_cancel_waits_for_worker_checkpoint_and_history_persists(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(web_app, "TASK_DIR", Path(directory) / "tasks"), patch.object(web_app, "BASE_DIR", Path(directory)):
            web_app.TASK_DIR.mkdir()
            task_id = "cancel-test"
            with web_app.progress_store_lock:
                web_app.cancel_events[task_id] = threading.Event()
            web_app.set_task_metadata(task_id, kind="summary", video_alias="video1", status="running")
            web_app.update_task_progress(task_id, 35, "Kareler çıkarılıyor")

            response = asyncio.run(web_app.cancel_task(task_id))
            self.assertEqual(response["status"], "cancel_requested")
            self.assertEqual(web_app.progress_store[task_id]["status"], "cancel_requested")
            with self.assertRaises(web_app.TaskCancelled):
                web_app._cancel_checkpoint(task_id)

            web_app._finish_task(task_id, "cancelled", "İşlem iptal edildi")
            with web_app.progress_store_lock:
                web_app.progress_store.pop(task_id, None)
            restored = asyncio.run(web_app.get_task_progress(task_id))
            self.assertTrue(restored["found"])
            self.assertEqual(restored["status"], "cancelled")
            history = asyncio.run(web_app.list_tasks())
            self.assertEqual(history["tasks"][0]["id"], task_id)
            self.assertEqual(history["tasks"][0]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
