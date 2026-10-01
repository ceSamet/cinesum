import subprocess
import tempfile
import unittest
from pathlib import Path

from src.summary.targeted_asr import (
    refine_critical_silent_segments,
    verify_and_extend_speech_boundaries,
)


class TestTargetedAsr(unittest.TestCase):
    def test_critical_silent_segment_is_extended_to_detected_utterance(self):
        segments = [{
            "start": 100.0,
            "end": 120.0,
            "duration": 20.0,
            "has_speech": False,
            "core_shot_ids": [7],
        }]
        shots = [{"scene_id": 7, "narrative_event_score": 0.97}]

        def fake_runner(command, **kwargs):
            Path(command[-1]).touch()
            return subprocess.CompletedProcess(command, 0)

        def fake_transcribe(*args, **kwargs):
            return {"segments": [{
                "start": 36.5,
                "end": 39.0,
                "text": "Ben de kahramanım.",
                "no_speech_prob": 0.2,
            }]}

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "source.wav"
            wav.touch()
            refined, metadata = refine_critical_silent_segments(
                segments,
                shots,
                audio_wav_path=wav,
                transcribe_fn=fake_transcribe,
                audio_runner=fake_runner,
            )

        self.assertEqual(metadata["status"], "applied")
        self.assertTrue(refined[0]["targeted_asr_applied"])
        self.assertEqual(refined[0]["boundary_mode"], "targeted_asr")
        self.assertEqual(refined[0]["transcript_text"], "Ben de kahramanım.")
        self.assertGreater(refined[0]["end"], 120.0)

    def test_noncritical_silent_segment_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "source.wav"
            wav.touch()
            refined, metadata = refine_critical_silent_segments(
                [{"start": 0.0, "end": 7.0, "has_speech": False, "core_shot_ids": [1]}],
                [{"scene_id": 1, "narrative_event_score": 0.4}],
                audio_wav_path=wav,
            )
        self.assertEqual(metadata["status"], "skipped")
        self.assertFalse(refined[0].get("targeted_asr_applied", False))

    def test_targeted_asr_can_recover_dialogue_before_an_aftermath_shot(self):
        segments = [{
            "start": 100.0, "end": 120.0, "duration": 20.0,
            "has_speech": False, "core_shot_ids": [9],
        }]
        shots = [{"scene_id": 9, "narrative_event_score": 0.99}]

        def fake_runner(command, **kwargs):
            Path(command[-1]).touch()
            return subprocess.CompletedProcess(command, 0)

        def fake_transcribe(*args, **kwargs):
            return {"segments": [{
                "start": 5.0, "end": 8.0,
                "text": "Önemli son replik.", "no_speech_prob": 0.1,
            }]}

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "source.wav"
            wav.touch()
            refined, metadata = refine_critical_silent_segments(
                segments, shots, audio_wav_path=wav,
                transcribe_fn=fake_transcribe, audio_runner=fake_runner,
            )
        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(refined[0]["start"], 89.55)
        self.assertEqual(refined[0]["transcript_text"], "Önemli son replik.")


class TestBoundaryVerification(unittest.TestCase):
    def _fake_runner(self, command, **kwargs):
        Path(command[-1]).touch()
        return subprocess.CompletedProcess(command, 0)

    def test_end_edge_extended_when_real_speech_straddles_boundary(self):
        # Mirrors a confirmed real-video case: the full-video Whisper pass timed
        # a word's end at ~segment.end, but a focused re-check shows the real
        # audio for that word actually continues about a second longer.
        segments = [{"start": 10.0, "end": 20.0, "duration": 10.0, "has_speech": True}]

        def fake_transcribe(path, **kwargs):
            if "end_" in str(path):
                # Window starts at end - 0.5 = 19.5 (relative time base).
                return {"segments": [{
                    "start": 0.3, "end": 1.2, "no_speech_prob": 0.1,
                    "words": [{"start": 0.3, "end": 1.2, "word": "clipped"}],
                }]}
            return {"segments": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "source.wav"
            wav.touch()
            refined, metadata = verify_and_extend_speech_boundaries(
                segments, audio_wav_path=wav,
                transcribe_fn=fake_transcribe, audio_runner=self._fake_runner,
            )

        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["extended_end_count"], 1)
        self.assertTrue(refined[0]["boundary_verified_end"])
        self.assertGreater(refined[0]["end"], 20.0)
        self.assertAlmostEqual(refined[0]["end"], 21.0, places=2)

    def test_start_edge_extended_when_real_speech_straddles_boundary(self):
        segments = [{"start": 20.0, "end": 30.0, "duration": 10.0, "has_speech": True}]

        def fake_transcribe(path, **kwargs):
            if "start_" in str(path):
                # Window starts at start - 1.5 = 18.5 (relative time base).
                return {"segments": [{
                    "start": 1.0, "end": 2.2, "no_speech_prob": 0.1,
                    "words": [{"start": 1.0, "end": 2.2, "word": "clipped"}],
                }]}
            return {"segments": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "source.wav"
            wav.touch()
            refined, metadata = verify_and_extend_speech_boundaries(
                segments, audio_wav_path=wav,
                transcribe_fn=fake_transcribe, audio_runner=self._fake_runner,
            )

        self.assertEqual(metadata["status"], "applied")
        self.assertEqual(metadata["extended_start_count"], 1)
        self.assertTrue(refined[0]["boundary_verified_start"])
        self.assertLess(refined[0]["start"], 20.0)

    def test_no_extension_when_boundary_is_already_clean(self):
        segments = [{"start": 10.0, "end": 20.0, "duration": 10.0, "has_speech": True}]

        def fake_transcribe(path, **kwargs):
            # Real speech is fully inside the segment or fully outside it --
            # nothing straddles the boundary itself.
            return {"segments": [{
                "start": 0.6, "end": 0.9, "no_speech_prob": 0.1,
                "words": [{"start": 0.6, "end": 0.9, "word": "fine"}],
            }]}

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "source.wav"
            wav.touch()
            refined, metadata = verify_and_extend_speech_boundaries(
                segments, audio_wav_path=wav,
                transcribe_fn=fake_transcribe, audio_runner=self._fake_runner,
            )

        self.assertEqual(metadata["status"], "no_extension_needed")
        self.assertEqual(refined[0]["start"], 10.0)
        self.assertEqual(refined[0]["end"], 20.0)

    def test_extension_never_overlaps_the_next_segment(self):
        segments = [
            {"start": 10.0, "end": 20.0, "duration": 10.0, "has_speech": True},
            {"start": 20.3, "end": 25.0, "duration": 4.7, "has_speech": False},
        ]

        def fake_transcribe(path, **kwargs):
            if "end_0" in str(path):
                # Would naturally extend well past the next segment's start.
                return {"segments": [{
                    "start": 0.3, "end": 5.0, "no_speech_prob": 0.1,
                    "words": [{"start": 0.3, "end": 5.0, "word": "long"}],
                }]}
            return {"segments": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            wav = Path(temp_dir) / "source.wav"
            wav.touch()
            refined, _ = verify_and_extend_speech_boundaries(
                segments, audio_wav_path=wav,
                transcribe_fn=fake_transcribe, audio_runner=self._fake_runner,
            )

        self.assertLessEqual(refined[0]["end"], refined[1]["start"])

    def test_skipped_without_audio_file(self):
        refined, metadata = verify_and_extend_speech_boundaries(
            [{"start": 0.0, "end": 5.0, "has_speech": True}],
            audio_wav_path=Path("does_not_exist.wav"),
        )
        self.assertEqual(metadata["status"], "skipped")
        self.assertEqual(refined[0]["end"], 5.0)


if __name__ == "__main__":
    unittest.main()
