import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.core.llm_config import LLMConfig
from src.llm.narrative_reranker import rerank_narrative_shots
from src.llm.paralon_client import ParalonError


class FakeClient:
    def __init__(self, fail=False, partial=False):
        self.calls = 0
        self.fail = fail
        self.partial = partial

    def rerank(self, candidates, request_context):
        self.calls += 1
        if self.fail:
            raise ParalonError("provider_timeout")
        return {
            "status": "applied",
            "provider": "paralon",
            "model": "test-model",
            "latency_sec": 0.01,
            "partial": self.partial,
            "summary": "test",
            "recommendations": [{
                "scene_id": row["scene_id"],
                "priority": 100 - index,
                "reason": f"reason {index}",
                "context_with": [],
            } for index, row in enumerate(candidates)],
        }


class TestNarrativeReranker(unittest.TestCase):
    def _setup(self, root: Path):
        story_dir = root / "outputs/features/story"
        story_dir.mkdir(parents=True)
        shots = []
        stories = []
        for index in range(9):
            scene_id = index + 1
            shots.append({
                "scene_id": scene_id,
                "start_seconds": index * 10.0,
                "end_seconds": index * 10.0 + 6.0,
                "duration_seconds": 6.0,
                "importance_score": 0.5 + index * 0.03,
                "story_scene_id": scene_id,
                "transcript_text": f"story event {scene_id}",
                "speech_ratio": 0.5,
            })
            stories.append({
                "story_scene_id": scene_id,
                "start_seconds": index * 10.0,
                "end_seconds": index * 10.0 + 6.0,
                "duration_seconds": 6.0,
                "shot_ids": [scene_id],
                "transcript_text": f"story event {scene_id}",
                "speech_ratio": 0.5,
                "dominant_mode": "dialogue",
            })
        (story_dir / "sample_story_scenes.json").write_text(
            json.dumps({"story_scenes": stories}), encoding="utf-8"
        )
        np.save(story_dir / "sample_story_embeddings.npy", np.eye(9, dtype=np.float32))
        local = {
            "strategy": "local",
            "selected_shots": shots[:3],
            "selected_shot_ids": [1, 2, 3],
        }
        return shots, local

    def _run(self, root, shots, local, client):
        return rerank_narrative_shots(
            shots,
            local,
            base_dir=root,
            video_alias="sample",
            category="importance",
            target_duration_sec=24,
            score_key="importance_score",
            segment_config={
                "min_duration": 5.0, "max_duration": 15.0,
                "pre_context": 2.0, "post_context": 2.5,
            },
            video_duration=86,
            config=LLMConfig(api_key="test", base_url="http://localhost:9000/v1"),
            client=client,
        )

    def test_hybrid_selection_and_cache_prevent_second_call(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shots, local = self._setup(root)
            client = FakeClient()
            first = self._run(root, shots, local, client)
            second = self._run(root, shots, local, client)
            self.assertEqual(first["strategy"], "rag_llm_hybrid_knapsack")
            self.assertEqual(first["llm_selection"]["status"], "applied")
            self.assertEqual(first["llm_selection"]["candidate_limit"], 16)
            self.assertEqual(second["llm_selection"]["status"], "cached")
            self.assertEqual(client.calls, 1)
            self.assertTrue(first["selected_shots"])
            self.assertTrue(all(
                "_selection_duration" in row for row in first["selected_shots"]
            ))

    def test_provider_failure_preserves_local_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shots, local = self._setup(root)
            result = self._run(root, shots, local, FakeClient(fail=True))
            self.assertEqual(result["llm_selection"]["status"], "fallback_local")
            self.assertEqual(result["llm_selection"]["reason"], "provider_timeout")
            self.assertEqual(result["llm_selection"]["candidate_limit"], 16)
            self.assertEqual(result["llm_selection"]["timeout_sec"], 60.0)
            self.assertEqual(result["selected_shot_ids"], [1, 2, 3])

    def test_partial_provider_result_is_not_cached(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shots, local = self._setup(root)
            client = FakeClient(partial=True)
            self._run(root, shots, local, client)
            self._run(root, shots, local, client)
            self.assertEqual(client.calls, 2)

    def test_vlm_failure_does_not_disable_rag_llm(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shots, local = self._setup(root)
            with patch(
                "src.llm.narrative_reranker.enrich_candidates_with_visual_captions",
                side_effect=RuntimeError("gpu oom"),
            ):
                result = self._run(root, shots, local, FakeClient())
            self.assertEqual(result["strategy"], "rag_llm_hybrid_knapsack")
            visual = result["llm_selection"]["visual_captioning"]
            self.assertEqual(visual["status"], "fallback_no_captions")
            self.assertEqual(visual["reason"], "vlm_RuntimeError")


if __name__ == "__main__":
    unittest.main()
