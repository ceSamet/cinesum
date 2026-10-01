import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.core.vlm_config import VLMConfig
from src.features.visual_captioner import (
    _caption_is_informative,
    enrich_candidates_with_visual_captions,
)


class FakeCaptioner:
    device = "test-gpu"

    def __init__(self):
        self.blip_calls = 0
        self.llava_calls = 0

    def caption_blip(self, image_path):
        self.blip_calls += 1
        return f"basic action {image_path.stem}"

    def caption_llava(self, image_paths, blip_caption):
        self.llava_calls += 1
        return "decisive visible action " + " -> ".join(path.stem for path in image_paths)


class OutOfMemoryError(RuntimeError):
    pass


class OomThenSingleFrameCaptioner(FakeCaptioner):
    def caption_llava(self, image_paths, blip_caption):
        self.llava_calls += 1
        if len(image_paths) > 1:
            raise OutOfMemoryError("simulated")
        return "single frame shows a decisive action"


class TestVisualCaptioner(unittest.TestCase):
    def test_multiframe_oom_retries_single_frame_and_caches_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = self._setup(root)
            config = VLMConfig(max_frames=3, llava_max_frames=1, time_budget_sec=60)
            backend = OomThenSingleFrameCaptioner()
            first = enrich_candidates_with_visual_captions(
                candidates, base_dir=root, video_alias="sample", config=config, captioner=backend
            )
            self.assertEqual(first["llava_generated_count"], 1)
            self.assertGreaterEqual(backend.llava_calls, 2)
            cache = json.loads((root / "outputs/cache/vlm/sample/captions.json").read_text(encoding="utf-8"))
            self.assertTrue(any(record.get("llava_context_mode") == "single_frame_after_oom" for record in cache["records"].values()))
            second_backend = OomThenSingleFrameCaptioner()
            enrich_candidates_with_visual_captions(
                self._setup(root), base_dir=root, video_alias="sample", config=config, captioner=second_backend
            )
            self.assertEqual(second_backend.llava_calls, 0)

    def test_generic_or_repeated_captions_are_not_trusted(self):
        self.assertFalse(_caption_is_informative("the avengers movie trailer"))
        self.assertFalse(_caption_is_informative("thor thor thor thor thor thor"))
        self.assertTrue(_caption_is_informative("a hand catches a flying metal hammer"))

    def _setup(self, root: Path):
        directory = root / "outputs/pyscenedetect/keyframes/sample"
        directory.mkdir(parents=True, exist_ok=True)
        metadata = []
        candidates = []
        for index in range(4):
            scene_id = index + 1
            image_path = directory / f"sample_scene_{scene_id:03d}_kf1.jpg"
            Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(image_path)
            metadata.append({
                "scene_id": scene_id,
                "kf_index": 1,
                "file_name": image_path.name,
                "file_path": str(image_path),
            })
            candidates.append({
                "scene_id": scene_id,
                "story_scene_id": scene_id,
                "candidate_source": "local_selected" if index == 0 else "temporal_bin",
                "importance_score": 1.0 - index * 0.1,
                "narrative_event_score": 0.9 - index * 0.1,
                "speech_ratio": index * 0.1,
            })
        (directory / "keyframes_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return candidates

    def test_hybrid_captions_are_cached_and_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_candidates = self._setup(root)
            config = VLMConfig(max_frames=3, llava_max_frames=2, time_budget_sec=60)
            first_backend = FakeCaptioner()
            first = enrich_candidates_with_visual_captions(
                first_candidates,
                base_dir=root,
                video_alias="sample",
                config=config,
                captioner=first_backend,
            )
            self.assertEqual(first["status"], "applied")
            self.assertEqual(first_backend.blip_calls, 3)
            self.assertEqual(first_backend.llava_calls, 2)
            self.assertEqual(first["captioned_scene_count"], 3)
            self.assertIn("LLaVA:", first_candidates[0]["visual_caption"])
            cache = json.loads(
                (root / "outputs/cache/vlm/sample/captions.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(len(cache["records"]["1"]["llava_context_frames"]), 2)

            second_candidates = self._setup(root)
            second_backend = FakeCaptioner()
            second = enrich_candidates_with_visual_captions(
                second_candidates,
                base_dir=root,
                video_alias="sample",
                config=config,
                captioner=second_backend,
            )
            self.assertEqual(second["status"], "cached")
            self.assertEqual(second_backend.blip_calls, 0)
            self.assertEqual(second_backend.llava_calls, 0)
            self.assertEqual(second_candidates[0]["visual_caption_source"], "blip_llava")

    def test_missing_keyframes_keeps_safe_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = enrich_candidates_with_visual_captions(
                [{"scene_id": 1, "story_scene_id": 1}],
                base_dir=Path(temp_dir),
                video_alias="missing",
                config=VLMConfig(),
                captioner=FakeCaptioner(),
            )
        self.assertEqual(result["status"], "fallback_no_captions")


if __name__ == "__main__":
    unittest.main()
