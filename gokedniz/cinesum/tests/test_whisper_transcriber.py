import unittest

from src.audio.whisper_transcriber import (
    map_transcript_to_scenes,
    resolve_whisper_compute_type,
    resolve_whisper_device,
)


class TestWhisperSceneMapping(unittest.TestCase):
    def test_runtime_precision_follows_device(self):
        self.assertEqual(resolve_whisper_compute_type("cuda"), "float16")
        self.assertEqual(resolve_whisper_compute_type("cpu"), "int8")
        self.assertEqual(resolve_whisper_compute_type("cuda", "int8"), "int8")

    def test_explicit_device_is_preserved(self):
        self.assertEqual(resolve_whisper_device("cpu"), "cpu")

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
