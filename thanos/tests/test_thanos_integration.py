import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.core.video_registry import store_video
from src.selection.adaptive_diversity import select_adaptive
from src.summary.samet_speech_guard import guard_and_fit_segments, load_turns
from src.summary.summary_exporter import export_summary_segments


class TestThanosIntegration(unittest.TestCase):
    def test_reupload_reuses_same_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, path, reused = store_video(io.BytesIO(b"same movie"), root)
            second, same_path, reused_again = store_video(io.BytesIO(b"same movie"), root)
            self.assertFalse(reused)
            self.assertTrue(reused_again)
            self.assertEqual((first, path), (second, same_path))
            self.assertEqual(len(list(root.glob("*.mp4"))), 1)

    def test_final_cut_recovers_whole_turn_and_respects_budget(self):
        rows = [
            {"start": 2.0, "end": 3.5, "duration": 1.5, "segment_score": 0.9},
            {"start": 12.0, "end": 13.0, "duration": 1.0, "segment_score": 0.2},
        ]
        turns = [(1.0, 4.0), (11.5, 13.5)]
        result, report = guard_and_fit_segments(
            rows, turns=turns, video_duration=20.0, target_duration_sec=4.0
        )
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["start"], result[0]["end"]), (0.78, 4.22))
        self.assertLessEqual(report["duration_after_fit"], 4.0)
        for row in result:
            for start, end in turns:
                self.assertFalse(start < row["start"] < end)
                self.assertFalse(start < row["end"] < end)

    def test_transcript_continuation_is_one_atomic_sentence(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "transcript.json"
            transcript.write_text(json.dumps({"segments": [
                {"start": 1.0, "end": 2.0, "text": "Bu konuşma"},
                {"start": 2.3, "end": 3.0, "text": "devam ediyor."},
                {"start": 5.0, "end": 6.0, "text": "Yeni cümle."},
            ]}), encoding="utf-8")
            self.assertEqual(load_turns(transcript), [(1.0, 3.0), (5.0, 6.0)])

    def test_adaptive_distribution_never_requires_empty_act(self):
        rows = [
            {"score": 0.99 - index * 0.01, "duration": 2.0,
             "position": 0.45 + index * 0.01, "group": index}
            for index in range(5)
        ]
        selected, report = select_adaptive(
            rows, budget_sec=6.0,
            score=lambda row: row["score"],
            duration=lambda row: row["duration"],
            position=lambda row: row["position"],
            event_group=lambda row: row["group"],
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(report["strategy"], "samet_adaptive")

    def test_ffmpeg_export_preserves_source_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "summary.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=10:d=4",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
            ], check=True)
            export_summary_segments(
                str(source),
                [{"start": 0.5, "end": 1.5, "duration": 1.0},
                 {"start": 2.0, "end": 3.0, "duration": 1.0}],
                str(output),
            )
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_name", "-of", "json", str(output),
            ], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(probe.stdout)["streams"][0]["codec_name"], "aac")
            volume = subprocess.run([
                "ffmpeg", "-hide_banner", "-i", str(output),
                "-af", "volumedetect", "-f", "null", "-",
            ], capture_output=True, text=True, check=True)
            self.assertIn("mean_volume:", volume.stderr)
            self.assertNotIn("mean_volume: -inf", volume.stderr)


if __name__ == "__main__":
    unittest.main()
