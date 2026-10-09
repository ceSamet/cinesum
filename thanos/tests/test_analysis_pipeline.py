import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.core.analysis_pipeline import analyze_video_features, find_reusable_keyframes


class TestAnalysisPipelineCache(unittest.TestCase):
    def test_legacy_keyframes_require_exact_scene_coverage(self):
        scenes = [{"scene_id": 1}, {"scene_id": 2}]
        matching = [{"scene_id": 1}, {"scene_id": 2}, {"scene_id": 2}]
        partial = [{"scene_id": 1}]

        with patch(
            "src.core.analysis_pipeline.load_or_reconstruct_keyframes_metadata",
            return_value=matching,
        ):
            self.assertEqual(
                find_reusable_keyframes("video1", Path("keyframes"), scenes),
                matching,
            )

        with patch(
            "src.core.analysis_pipeline.load_or_reconstruct_keyframes_metadata",
            return_value=partial,
        ):
            self.assertEqual(
                find_reusable_keyframes("video1", Path("keyframes"), scenes),
                [],
            )

    def test_second_run_uses_every_cached_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            video_dir = base_dir / "dataset" / "video"
            video_dir.mkdir(parents=True)
            video_path = video_dir / "video1.mp4"
            video_path.write_bytes(b"synthetic-video")

            scenes = [{
                "scene_id": 1,
                "start_frame": 0,
                "end_frame": 50,
                "start_timecode": "00:00:00.000",
                "end_timecode": "00:00:02.000",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "duration_seconds": 2.0,
            }]

            def fake_save_json(_scenes, output_path):
                import json

                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(_scenes), encoding="utf-8")

            def fake_save_csv(_scenes, output_path):
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("scene_id\n1\n", encoding="utf-8")

            def fake_keyframes(_video_path, _scenes, output_dir, **_kwargs):
                frame_dir = Path(output_dir) / "video1"
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / "video1_scene_001_kf1.jpg"
                frame_path.write_bytes(b"jpeg")
                return [{
                    "video_stem": "video1",
                    "scene_id": 1,
                    "kf_index": 1,
                    "file_path": str(frame_path),
                }]

            def fake_clip(_alias, _keyframes, npy_path, json_path, **_kwargs):
                import json

                Path(npy_path).write_bytes(b"npy")
                Path(json_path).write_text(
                    json.dumps([{"scene_id": 1, "feature_vector_index": 0}]),
                    encoding="utf-8",
                )
                return [[0.1, 0.2]], [{"scene_id": 1}]

            def fake_audio(_video_path, output_path, **_kwargs):
                Path(output_path).write_bytes(b"wav")
                return output_path

            def fake_story(_scenes, _clip, _clip_meta, _audio, json_path, npy_path):
                import json

                result = {
                    "story_scene_count": 1,
                    "shot_to_story_scene": {
                        "1": {"story_scene_id": 1, "position": 0, "shot_count": 1}
                    },
                    "story_scenes": [],
                }
                Path(json_path).write_text(json.dumps(result), encoding="utf-8")
                Path(npy_path).write_bytes(b"story-npy")
                return result

            transcript = {
                "backend": "faster-whisper",
                "model": "small",
                "text": "hello",
                "language": "en",
                "segments": [{
                    "start": 0.5,
                    "end": 1.5,
                    "text": "hello",
                    "words": [{"start": 0.5, "end": 1.0, "word": "hello"}],
                }],
            }
            energy = [{
                "scene_id": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "duration_seconds": 2.0,
                "mean_rms_energy": 0.1,
                "max_rms_energy": 0.2,
                "normalized_audio_energy": 0.5,
            }]

            with (
                patch("src.core.analysis_pipeline.detect_scenes_pyscenedetect", return_value=scenes) as detect,
                patch("src.core.analysis_pipeline.save_scenes_to_json", side_effect=fake_save_json),
                patch("src.core.analysis_pipeline.save_scenes_to_csv", side_effect=fake_save_csv),
                patch("src.core.analysis_pipeline.extract_keyframes_for_scenes", side_effect=fake_keyframes) as keyframes,
                patch("src.core.analysis_pipeline.extract_clip_features_for_video", side_effect=fake_clip) as clip,
                patch("src.core.analysis_pipeline.extract_audio_from_video", side_effect=fake_audio) as audio,
                patch("src.core.analysis_pipeline.transcribe_audio_with_whisper", return_value=transcript) as transcribe,
                patch("src.core.analysis_pipeline.compute_scene_audio_energy", return_value=energy) as rms,
                patch("src.core.analysis_pipeline.build_story_scenes_from_files", side_effect=fake_story) as story,
            ):
                first = analyze_video_features("video1", video_path, base_dir)
                second = analyze_video_features("video1", video_path, base_dir)

            self.assertFalse(any(first["cache_hits"].values()))
            self.assertTrue(all(second["cache_hits"].values()))
            detect.assert_called_once()
            keyframes.assert_called_once()
            clip.assert_called_once()
            audio.assert_called_once()
            transcribe.assert_called_once()
            rms.assert_called_once()
            story.assert_called_once()


if __name__ == "__main__":
    unittest.main()
