import subprocess
import tempfile
import unittest
from pathlib import Path

from src.audio.transcript_repair import (
    detect_transcript_anomalies,
    repair_transcript_anomalies,
)


class TestTranscriptRepair(unittest.TestCase):
    def test_detects_extreme_sparse_segment(self):
        transcript = {"segments": [{
            "id": 1, "start": 100.0, "end": 390.0,
            "text": "iki kelime", "words": [{"start": 100.0, "end": 100.5, "word": "iki"}],
        }]}
        anomalies = detect_transcript_anomalies(transcript)
        self.assertEqual(len(anomalies), 1)
        self.assertIn("extreme_duration", anomalies[0]["reasons"])

    def test_replaces_anomaly_with_absolute_chunk_timestamps(self):
        transcript = {
            "language": "tr",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "normal", "words": []},
                {"id": 1, "start": 100.0, "end": 151.0, "text": "bozuk", "words": []},
            ],
        }

        def fake_runner(command, **kwargs):
            Path(command[-1]).touch()
            return subprocess.CompletedProcess(command, 0)

        def fake_transcribe(*args, **kwargs):
            return {"segments": [{
                "start": 2.0, "end": 4.0, "text": "düzeltilmiş cümle.",
                "words": [{"start": 2.0, "end": 4.0, "word": "düzeltilmiş"}],
            }]}

        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "source.wav"
            audio.touch()
            repaired, metadata = repair_transcript_anomalies(
                transcript,
                audio,
                transcribe_fn=fake_transcribe,
                audio_runner=fake_runner,
                device="cuda",
                compute_type="float16",
            )
        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["repaired_count"], 1)
        repaired_rows = [row for row in repaired["segments"] if "düzeltilmiş" in row["text"]]
        self.assertEqual(len(repaired_rows), 2)
        self.assertEqual(repaired_rows[0]["start"], 100.5)
        self.assertGreater(repaired_rows[1]["start"], repaired_rows[0]["start"])


if __name__ == "__main__":
    unittest.main()
