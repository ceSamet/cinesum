import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.audio.whisper_transcriber import (
    map_transcript_to_scenes,
    resolve_whisper_compute_type,
    resolve_whisper_device,
    transcribe_audio_with_whisper,
)


class TestWhisperSceneMapping(unittest.TestCase):
    def test_runtime_precision_follows_device(self):
        self.assertEqual(resolve_whisper_compute_type("cuda"), "float16")
        self.assertEqual(resolve_whisper_compute_type("cpu"), "int8")
        self.assertEqual(resolve_whisper_compute_type("cuda", "int8"), "int8")

    def test_explicit_device_is_preserved(self):
        self.assertEqual(resolve_whisper_device("cpu"), "cpu")

    def test_missing_cublas_selects_cpu_even_if_torch_sees_gpu(self):
        with patch("torch.cuda.is_available", return_value=True), patch(
            "src.audio.whisper_transcriber._cuda_runtime_ready", return_value=False
        ), patch.dict("os.environ", {"THANOS_WHISPER_DEVICE": "auto"}):
            self.assertEqual(resolve_whisper_device(), "cpu")

    def test_windows_cuda_uses_ctranslate2_device_count(self):
        with patch("src.audio.whisper_transcriber.sys.platform", "win32"), patch(
            "ctranslate2.get_cuda_device_count", return_value=1
        ), patch("torch.cuda.is_available", return_value=True), patch.dict(
            "os.environ", {"THANOS_WHISPER_DEVICE": "auto"}
        ):
            self.assertEqual(resolve_whisper_device(), "cuda")

    def test_cuda_iterator_failure_retries_cpu(self):
        class BrokenCuda:
            def transcribe(self, *args, **kwargs):
                def fail_during_iteration():
                    raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
                    yield None
                return fail_during_iteration(), SimpleNamespace(language="tr", language_probability=1.0)

        class WorkingCpu:
            def transcribe(self, *args, **kwargs):
                return iter([]), SimpleNamespace(language="tr", language_probability=1.0)

        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "sample.wav"
            wav.touch()
            with patch("src.audio.whisper_transcriber.get_whisper_model", side_effect=lambda **kw: BrokenCuda() if kw["device"] == "cuda" else WorkingCpu()) as get_model:
                result = transcribe_audio_with_whisper(str(wav), device="cuda")
            self.assertEqual(result["language"], "tr")
            self.assertEqual([call.kwargs["device"] for call in get_model.call_args_list], ["cuda", "cpu"])

    def test_overlapping_segments_do_not_double_count_speech_time(self):
        scenes = [{
            "scene_id": 1,
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "duration_seconds": 10.0,
        }]
        transcript = {
            "segments": [
                {"start": 1.0, "end": 5.0, "text": "one", "words": []},
                {"start": 4.0, "end": 7.0, "text": "two", "words": []},
            ]
        }

        result = map_transcript_to_scenes(transcript, scenes)[0]
        self.assertEqual(result["speech_time_seconds"], 6.0)
        self.assertEqual(result["speech_ratio"], 0.6)

    def test_word_timestamps_are_preserved_per_scene(self):
        scenes = [
            {
                "scene_id": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "duration_seconds": 2.0,
            },
            {
                "scene_id": 2,
                "start_seconds": 2.0,
                "end_seconds": 4.0,
                "duration_seconds": 2.0,
            },
        ]
        transcript = {
            "segments": [{
                "start": 0.5,
                "end": 3.5,
                "text": "hello world",
                "words": [
                    {"start": 0.5, "end": 1.0, "word": "hello", "probability": 0.9},
                    {"start": 2.5, "end": 3.0, "word": "world", "probability": 0.8},
                ],
            }]
        }

        result = map_transcript_to_scenes(transcript, scenes)
        self.assertEqual([word["word"] for word in result[0]["words"]], ["hello"])
        self.assertEqual([word["word"] for word in result[1]["words"]], ["world"])
        self.assertEqual(result[0]["word_count"], 1)
        self.assertEqual(result[1]["word_count"], 1)

    def test_sparse_words_do_not_fill_a_long_whisper_segment(self):
        scenes = [
            {"scene_id": 1, "start_seconds": 0.0, "end_seconds": 10.0, "duration_seconds": 10.0},
            {"scene_id": 2, "start_seconds": 10.0, "end_seconds": 20.0, "duration_seconds": 10.0},
            {"scene_id": 3, "start_seconds": 20.0, "end_seconds": 30.0, "duration_seconds": 10.0},
        ]
        transcript = {"segments": [{
            "start": 1.0,
            "end": 29.0,
            "text": "başla ve bitir",
            "words": [
                {"start": 1.0, "end": 1.5, "word": "başla"},
                {"start": 28.0, "end": 28.5, "word": "bitir"},
            ],
        }]}

        result = map_transcript_to_scenes(transcript, scenes)

        self.assertEqual(result[0]["transcript_text"], "başla")
        self.assertEqual(result[1]["transcript_text"], "")
        self.assertEqual(result[1]["speech_ratio"], 0.0)
        self.assertEqual(result[2]["transcript_text"], "bitir")


if __name__ == "__main__":
    unittest.main()
