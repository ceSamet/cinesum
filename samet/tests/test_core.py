from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from scene_captioner.scene_detector import extract_scene_keyframes
from scene_captioner.analyzer import (
    SCORE_WEIGHTS,
    _adaptive_diversity_select,
    _knapsack_select,
    _normalize_selection_request,
    _selection_profile,
    _story_context,
    _speech_safe_intervals,
    _temporally_balanced_select,
    _video_has_audio_stream,
)
from scene_captioner.audio_analysis import SpeechTurn, _face_agreement
from scene_captioner.face_analysis import _identity_candidates, _merge_duplicate_clusters
from scene_captioner import web_app as web_module
from scene_captioner.web_app import app
from scene_captioner.scene_rag import SceneVectorStore, _paralon_chat, select_with_paralon_rag


class SceneDetectorTests(unittest.TestCase):
    def test_adaptive_detector_finds_abrupt_color_cut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "cuts.mp4"
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
            )
            for color in ((0, 0, 255), (0, 255, 0), (255, 0, 0)):
                for _ in range(20):
                    writer.write(np.full((180, 320, 3), color, dtype=np.uint8))
            writer.release()

            scenes, duration = extract_scene_keyframes(video, root / "frames")
            self.assertGreaterEqual(len(scenes), 3)
            self.assertAlmostEqual(duration, 6.0, delta=0.2)
            self.assertTrue(all(scene.frame_path.exists() for scene in scenes))
            self.assertFalse(_video_has_audio_stream(video))


class WebTests(unittest.TestCase):
    def test_dashboard_and_model_health_routes(self) -> None:
        client = app.test_client()
        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("SceneMind", page.get_data(as_text=True))
        self.assertIn("Konuşmalı", page.get_data(as_text=True))
        self.assertIn("person-photo", page.get_data(as_text=True))
        self.assertIn("Uygulamayı kapat", page.get_data(as_text=True))
        self.assertIn("Özet süresi", page.get_data(as_text=True))
        self.assertIn("Özet isteği", page.get_data(as_text=True))
        self.assertIn("Sahne türleri", page.get_data(as_text=True))
        self.assertNotIn("Görsel model", page.get_data(as_text=True))
        self.assertNotIn("BLIP · Hızlı", page.get_data(as_text=True))
        self.assertNotIn("Bir videoyu, hikâyesiyle birlikte gör.", page.get_data(as_text=True))
        health = client.get("/api/models")
        self.assertEqual(health.status_code, 200)
        self.assertIn("llava", health.get_json()["models"])
        self.assertEqual(health.get_json()["paralon"]["model"], "qwen3.8-27b")

    def test_same_video_content_finds_completed_analysis_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root / "jobs"
            uploads = root / "uploads"
            source_id = "a" * 32
            source_job = jobs / source_id
            source_upload = uploads / source_id / "old-name.mp4"
            source_job.mkdir(parents=True)
            source_upload.parent.mkdir(parents=True)
            source_upload.write_bytes(b"same-video-content")
            (source_job / "state.json").write_text(json.dumps({
                "job_id": source_id,
                "created_at": "2026-08-27T12:00:00",
                "status": "complete",
                "video_filename": "old-name.mp4",
                "analysis": {"scenes": [{"description": "cached"}]},
            }), encoding="utf-8")
            incoming = root / "new-name.mp4"
            incoming.write_bytes(b"same-video-content")

            with patch.object(web_module, "JOBS_DIR", jobs), patch.object(
                web_module, "UPLOADS_DIR", uploads
            ):
                digest = web_module._sha256_file(incoming)
                match = web_module._find_cached_job(incoming, digest, "b" * 32)

            self.assertEqual(match, source_id)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _profile_scene(
        description: str = "",
        dialogue: bool = False,
        actors: list[str] | None = None,
        visual_change: float = 0.1,
        music: bool = False,
    ):
        return SimpleNamespace(
            description=description,
            transcript=[{"text": "replik"}] if dialogue else [],
            actors=actors or [],
            visual_change_score=visual_change,
            audio_events=[{"label": "müzik"}] if music else [],
        )

    def test_content_profile_is_inferred_without_fixed_acts(self) -> None:
        action = _selection_profile([
            self._profile_scene("A car chase and explosion", visual_change=0.9)
            for _ in range(4)
        ])
        atmospheric = _selection_profile([
            self._profile_scene("Sessiz bir manzara", visual_change=0.5, music=True)
            for _ in range(4)
        ])
        self.assertEqual(action["key"], "action_led")
        self.assertEqual(atmospheric["key"], "atmospheric")
        self.assertEqual(sum(action["weights"].values()), 100)

    def test_prompt_infers_strict_action_selection(self) -> None:
        request = _normalize_selection_request({
            "prompt": "Sadece aksiyon ve kovalamaca sahnelerini kullan",
        })
        self.assertTrue(request["strict"])
        self.assertIn("action", request["scene_types"])

    def test_knapsack_maximizes_value_under_duration_budget(self) -> None:
        selected = _knapsack_select([60.0, 100.0, 120.0], [10, 20, 30], 50)
        self.assertEqual(selected, [1, 2])
        self.assertEqual(sum(SCORE_WEIGHTS.values()), 100)

    def test_adaptive_selection_does_not_force_a_weak_time_region(self) -> None:
        selected = _temporally_balanced_select(
            [100.0, 90.0, 80.0, 1.0],
            [2, 2, 2, 2],
            [0.01, 0.11, 0.21, 0.91],
            budget=6,
        )
        self.assertEqual(selected, [0, 1, 2])

    def test_adaptive_selection_limits_one_event_cluster(self) -> None:
        selected = _adaptive_diversity_select(
            [100.0, 98.0, 96.0, 80.0, 78.0, 76.0],
            [2, 2, 2, 2, 2, 2],
            [0.10, 0.11, 0.12, 0.50, 0.70, 0.90],
            budget=6,
            event_groups=[1, 1, 1, 2, 3, 4],
        )
        self.assertEqual(selected, [0, 3, 4])

    def test_highlight_boundaries_never_split_detected_speech(self) -> None:
        scenes = [
            SimpleNamespace(start_sec=1.0, end_sec=2.0, summary_start_sec=None, summary_end_sec=None),
            SimpleNamespace(start_sec=5.0, end_sec=6.0, summary_start_sec=None, summary_end_sec=None),
        ]
        turns = [
            SpeechTurn(0.8, 2.4, "ilk konuşma"),
            SpeechTurn(5.6, 6.3, "ikinci konuşma"),
        ]
        intervals = _speech_safe_intervals(scenes, turns, duration=8.0)
        for start, end in intervals:
            for turn in turns:
                self.assertFalse(turn.start_sec < start < turn.end_sec)
                self.assertFalse(turn.start_sec < end < turn.end_sec)


class StoryCaptionTests(unittest.TestCase):
    def test_story_context_combines_visuals_dialogue_and_actor_identity(self) -> None:
        scene = SimpleNamespace(
            description="Two people argue beside a car.",
            actors=["Oyuncu 1", "Oyuncu 2"],
            transcript=[
                {
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                    "text": "Arabaya binmeyeceğim.",
                    "speaker": "Konuşmacı 1",
                    "actor": "Oyuncu 1",
                }
            ],
        )
        context = _story_context({"index": 3, "duration_sec": 14.2}, [scene])
        self.assertEqual(context["index"], 3)
        self.assertIn("Two people argue", context["visual_observations"][0])
        self.assertEqual(context["dialogue"], ["Oyuncu 1: Arabaya binmeyeceğim."])
        self.assertEqual(context["actors"], ["Oyuncu 1", "Oyuncu 2"])


class SceneRagTests(unittest.TestCase):
    @staticmethod
    def _scene(index: int, description: str):
        return SimpleNamespace(
            index=index,
            start_sec=float(index * 3),
            end_sec=float(index * 3 + 2),
            description=description,
            actors=["Oyuncu 1"] if index < 2 else ["Oyuncu 2"],
            transcript=[{
                "start_sec": float(index * 3),
                "end_sec": float(index * 3 + 1),
                "text": f"replik {index}",
                "speaker": "Konuşmacı 1",
            }],
            audio_events=[],
            scoring_breakdown={},
        )

    def test_sqlite_vector_store_returns_semantically_nearest_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SceneVectorStore(Path(directory) / "rag.sqlite3")
            store.replace([
                {
                    "id": "cut:0", "kind": "cut", "title": "Araba konuşması",
                    "content": "Arabada tartışırlar", "metadata": {"cut_index": 0},
                    "vector": np.asarray([1.0, 0.0], dtype=np.float32),
                },
                {
                    "id": "cut:1", "kind": "cut", "title": "Ev konuşması",
                    "content": "Evde buluşurlar", "metadata": {"cut_index": 1},
                    "vector": np.asarray([0.0, 1.0], dtype=np.float32),
                },
            ])
            matches = store.search(np.asarray([0.9, 0.1], dtype=np.float32), limit=1)
            store.close()
            self.assertEqual(matches[0]["id"], "cut:0")

    def test_paralon_review_is_repacked_by_local_duration_knapsack(self) -> None:
        scenes = [
            self._scene(0, "İki kişi arabada tartışıyor."),
            self._scene(1, "Tartışmanın nedeni açıklanıyor."),
            self._scene(2, "Başka bir kişi evde bekliyor."),
        ]
        stories = [{
            "index": 1, "label": "Arabadaki tartışma", "description": "Neden ortaya çıkar",
            "start_sec": 0.0, "end_sec": 8.0, "cut_indices": [0, 1, 2],
        }]
        embeddings = np.asarray([
            [1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory, patch(
            "scene_captioner.scene_rag._paralon_chat",
            return_value=(
                {
                    "selections": [
                        {"cut_index": 0, "priority": 90, "reason": "çatışmayı kurar", "context_with": [1]},
                        {"cut_index": 1, "priority": 100, "reason": "nedeni açıklar", "context_with": [0]},
                        {"cut_index": 2, "priority": 20, "reason": "yan olay", "context_with": []},
                    ],
                    "summary": "Kurulum ve açıklama birlikte tutuldu.",
                },
                {"provider": "ParalonCloud", "model": "qwen3.8-27b", "usage": {}},
            ),
        ):
            selected, meta = select_with_paralon_rag(
                scenes, stories, embeddings, [0, 2], [80.0, 70.0, 60.0],
                [2, 2, 2], [0.1, 0.5, 0.9], 4, set(),
                Path(directory) / "scene_rag.sqlite3", _knapsack_select,
            )
            self.assertEqual(selected, [0, 1])
            self.assertEqual(meta["status"], "applied")
            self.assertTrue((Path(directory) / "scene_rag.sqlite3").exists())

    def test_candidate_omitted_by_llm_keeps_its_local_score(self) -> None:
        scenes = [
            self._scene(0, "Açılış."),
            self._scene(1, "Yerel puanı yüksek dönüm noktası."),
            self._scene(2, "Kapanış."),
        ]
        stories = [{
            "index": 1, "label": "Olay", "description": "Olay akışı",
            "start_sec": 0.0, "end_sec": 8.0, "cut_indices": [0, 1, 2],
        }]
        with tempfile.TemporaryDirectory() as directory, patch(
            "scene_captioner.scene_rag._paralon_chat",
            return_value=(
                {"selections": [{
                    "cut_index": 0, "priority": 70, "reason": "açılış",
                    "context_with": [],
                }]},
                {"provider": "ParalonCloud", "model": "qwen3.8-27b", "usage": {}},
            ),
        ):
            selected, meta = select_with_paralon_rag(
                scenes, stories, np.eye(3, dtype=np.float32), [0, 1],
                [60.0, 100.0, 20.0], [2, 2, 2], [0.1, 0.5, 0.9], 4, set(),
                Path(directory) / "scene_rag.sqlite3", _knapsack_select,
            )
        self.assertIn(1, selected)
        self.assertGreater(meta["local_fallback_count"], 0)

    def test_paralon_request_uses_cloudflare_compatible_headers(self) -> None:
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": '{"selections": []}'}}],
            "usage": {},
        }).encode("utf-8")
        with patch.dict("os.environ", {
            "PARALON_API_KEY": "prlc_test",
            "PARALON_BASE_URL": "https://paraloncloud.com/v1/",
            "PARALON_MODEL": "qwen3.8-27b",
        }), patch("scene_captioner.scene_rag.urllib.request.urlopen", return_value=response) as urlopen:
            decision, _ = _paralon_chat("test")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://paraloncloud.com/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer prlc_test")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertIn("LexismAI", request.get_header("User-agent"))
        payload = json.loads(request.data)
        self.assertEqual(payload["max_tokens"], 768)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("/no_think", payload["messages"][0]["content"])
        self.assertTrue(payload["messages"][1]["content"].endswith("/no_think"))
        self.assertEqual(decision, {"selections": []})

    def test_missing_paralon_key_keeps_baseline_selection(self) -> None:
        scenes = [self._scene(0, "İlk olay."), self._scene(1, "İkinci olay.")]
        stories = [{
            "index": 1, "label": "Olay zinciri", "description": "İki olay",
            "start_sec": 0.0, "end_sec": 5.0, "cut_indices": [0, 1],
        }]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"PARALON_API_KEY": ""}, clear=False
        ):
            selected, meta = select_with_paralon_rag(
                scenes, stories, np.eye(2, dtype=np.float32), [1], [50.0, 60.0],
                [2, 2], [0.2, 0.8], 2, set(),
                Path(directory) / "scene_rag.sqlite3", _knapsack_select,
            )
            self.assertEqual(selected, [1])
            self.assertEqual(meta["status"], "fallback_knapsack")
            self.assertIn("PARALON_API_KEY", meta["reason"])


class FaceIdentityTests(unittest.TestCase):
    def test_second_pass_merges_pose_split_identity_but_not_costar(self) -> None:
        matrix = np.asarray([
            [1.0, 0.0, 0.0],
            [0.94, 0.341, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        merged = _merge_duplicate_clusters(
            matrix,
            np.asarray([0, 1, 2]),
            [1, 2, 2],
        )
        self.assertEqual(merged[0], merged[1])
        self.assertNotEqual(merged[1], merged[2])

    def test_second_pass_can_merge_different_frames_of_same_scene(self) -> None:
        matrix = np.asarray([[1.0, 0.0], [0.95, 0.31]], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        merged = _merge_duplicate_clusters(
            matrix,
            np.asarray([0, 1]),
            [(4, 0), (4, 1)],
        )
        self.assertEqual(merged[0], merged[1])

    def test_reciprocal_near_clusters_are_kept_as_identity_candidates(self) -> None:
        matrix = np.asarray([
            [1.0, 0.0, 0.0],
            [0.60, 0.80, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        candidates = _identity_candidates(
            matrix,
            np.asarray([0, 1, 2]),
            [(1, 0), (2, 0), (3, 0)],
            [0, 1, 2],
        )
        self.assertEqual(candidates[0][0], 1)
        self.assertEqual(candidates[1][0], 0)
        self.assertNotIn(2, candidates)


class SpeakerFaceFusionTests(unittest.TestCase):
    def test_face_agreement_rewards_cohesive_voice_clusters(self) -> None:
        cohesive = _face_agreement(
            np.asarray([0, 0, 1, 1]), ["Oyuncu 1", "Oyuncu 1", "Oyuncu 2", "Oyuncu 2"]
        )
        mixed = _face_agreement(
            np.asarray([0, 1, 0, 1]), ["Oyuncu 1", "Oyuncu 1", "Oyuncu 2", "Oyuncu 2"]
        )
        self.assertGreater(cohesive, mixed)


if __name__ == "__main__":
    unittest.main()
