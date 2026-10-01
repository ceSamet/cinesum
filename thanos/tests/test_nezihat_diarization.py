import unittest
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.audio.nezihat_diarization import (
    align_transcript,
    assign_word_speakers,
    correct_speakers,
    diarize_transcript,
    overlap_regions,
)
from src.features.samet_enrichment import analyze_people


class NezihatDiarizationTests(unittest.TestCase):
    def test_word_alignment_splits_a_whisper_segment_at_speaker_change(self):
        transcript = {"segments": [{
            "start": 0.0, "end": 2.0, "text": "hello there",
            "words": [
                {"start": 0.0, "end": 0.5, "word": "hello"},
                {"start": 1.2, "end": 1.7, "word": "there"},
            ],
        }]}
        turns = [
            {"start": 0.0, "end": 0.9, "speaker": "A"},
            {"start": 1.0, "end": 2.0, "speaker": "B"},
        ]
        result = align_transcript(transcript, turns)
        self.assertEqual([(row["text"], row["speaker"]) for row in result], [
            ("hello", "A"), ("there", "B"),
        ])

    def test_switch_penalty_ignores_small_leading_diarization_drift(self):
        words = [
            {"start": 0.8, "end": 1.2},
            {"start": 1.2, "end": 1.6},
            {"start": 1.6, "end": 2.0},
        ]
        turns = [
            {"start": 0.0, "end": 1.05, "speaker": "OLD"},
            {"start": 1.05, "end": 2.1, "speaker": "NEW"},
        ]
        self.assertEqual(assign_word_speakers(words, turns), ["NEW"] * 3)

    def test_overlapping_speakers_are_not_merged_despite_similar_embeddings(self):
        regular = [
            {"start": 0.0, "end": 2.0, "speaker": "A"},
            {"start": 1.0, "end": 3.0, "speaker": "B"},
        ]
        regions = overlap_regions(regular)
        self.assertEqual(regions[0]["speakers"], ["A", "B"])
        _, report = correct_speakers(regular, {
            0: np.asarray([1.0, 0.0]),
            1: np.asarray([0.99, 0.01]),
        }, regions)
        self.assertEqual(report["final_speaker_count"], 2)

    def test_people_pipeline_exposes_nezihat_word_turns_and_survives_face_failure(self):
        transcript = {"segments": [{"start": 0.0, "end": 2.0, "text": "hello there"}]}
        diarization = {
            "turns": [{"start": 0.0, "end": 1.0, "speaker": "A"},
                      {"start": 1.0, "end": 2.0, "speaker": "B"}],
            "speaker_transcript": [
                {"start": 0.0, "end": 1.0, "speaker": "A", "text": "hello"},
                {"start": 1.0, "end": 2.0, "speaker": "B", "text": "there"},
            ],
            "overlaps": [], "correction": {"final_speaker_count": 2},
        }
        scenes = [{"scene_id": 1, "start_seconds": 0.0, "end_seconds": 2.0,
                   "duration_seconds": 2.0}]
        with patch("src.features.samet_enrichment.analyze_faces", side_effect=RuntimeError("faces")), patch(
            "src.features.samet_enrichment.pyannote_available", return_value=True
        ), patch("src.features.samet_enrichment.diarize_transcript", return_value=diarization), patch.dict(
            "os.environ", {"THANOS_DIARIZATION_BACKEND": "auto"}
        ):
            result = analyze_people(
                scenes=scenes, keyframes=[], transcript=transcript,
                video_path=Path("movie.mp4"), wav_path=Path(__file__),
                models_dir=Path("models"), portraits_dir=Path("portraits"),
            )
        self.assertEqual(result["backend"], "nezihat_pyannote")
        self.assertEqual(result["speaker_count"], 2)
        self.assertEqual([row["speaker"] for row in result["turns"]], ["A", "B"])
        self.assertEqual(result["face_warning"], "RuntimeError")

    def test_people_pipeline_falls_back_to_samet_when_pyannote_is_unavailable(self):
        transcript = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}
        with patch("src.features.samet_enrichment.analyze_faces", return_value=({}, [], {})), patch(
            "src.features.samet_enrichment.pyannote_available", return_value=False
        ), patch("src.features.samet_enrichment.assign_speakers", return_value=1) as fallback, patch.dict(
            "os.environ", {"THANOS_DIARIZATION_BACKEND": "pyannote"}
        ):
            result = analyze_people(
                scenes=[], keyframes=[], transcript=transcript,
                video_path=Path("movie.mp4"), wav_path=Path(__file__),
                models_dir=Path("models"), portraits_dir=Path("portraits"),
            )
        fallback.assert_called_once()
        self.assertEqual(result["backend"], "samet_mfcc")
        self.assertIn("pyannote.audio", result["fallback_reason"])

    def test_pyannote_adapter_reads_exclusive_turns_and_regular_overlaps(self):
        class Annotation:
            def __init__(self, rows):
                self.rows = rows

            def itertracks(self, yield_label=False):
                for start, end, speaker in self.rows:
                    yield SimpleNamespace(start=start, end=end), None, speaker

        class FakePipeline:
            _embedding = object()

            @classmethod
            def from_pretrained(cls, model, token):
                self = cls()
                self.model, self.token = model, token
                return self

            def to(self, device):
                self.device = device

            def instantiate(self, settings):
                self.settings = settings

            def __call__(self, audio):
                self.audio = audio
                return SimpleNamespace(
                    speaker_diarization=Annotation([(0.0, 2.0, "A"), (1.0, 3.0, "B")]),
                    exclusive_speaker_diarization=Annotation([(0.0, 1.5, "A"), (1.5, 3.0, "B")]),
                )

        parent = ModuleType("pyannote")
        child = ModuleType("pyannote.audio")
        child.Pipeline = FakePipeline
        parent.audio = child
        transcript = {"segments": [{"start": 0.0, "end": 3.0, "text": "one two",
                                   "words": [{"start": 0.2, "end": 0.6, "word": "one"},
                                             {"start": 2.1, "end": 2.5, "word": "two"}]}]}
        with patch.dict(sys.modules, {"pyannote": parent, "pyannote.audio": child}), patch(
            "src.audio.nezihat_diarization._token", return_value="test-token"
        ), patch("src.audio.nezihat_diarization.load_pcm_wav", return_value=np.zeros(48000, dtype=np.float32)), patch(
            "src.audio.nezihat_diarization._embeddings", return_value={}
        ):
            result = diarize_transcript(Path("unused.wav"), transcript)
        self.assertEqual(result["backend"], "nezihat_pyannote")
        self.assertEqual(result["overlaps"][0]["speakers"], ["A", "B"])
        self.assertEqual([row["speaker"] for row in result["speaker_transcript"]], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
