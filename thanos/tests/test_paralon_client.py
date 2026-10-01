import json
import unittest

import httpx

from src.core.llm_config import LLMConfig
from src.llm.paralon_client import (
    ParalonClient,
    ParalonError,
    extract_json_object,
    recommendation_limit_for_request,
)


class TestParalonClient(unittest.TestCase):
    def test_fenced_json_is_extracted(self):
        result = extract_json_object('```json\n{"summary":"ok","recommendations":[]}\n```')
        self.assertEqual(result["summary"], "ok")

    def test_recommendation_limit_scales_with_summary_duration(self):
        self.assertEqual(recommendation_limit_for_request(48, 60), 8)
        self.assertEqual(recommendation_limit_for_request(48, 180), 12)
        self.assertEqual(recommendation_limit_for_request(48, 300), 20)
        self.assertEqual(recommendation_limit_for_request(48, 600), 24)
        self.assertEqual(recommendation_limit_for_request(5, 180), 5)

    def test_single_call_and_strict_candidate_validation(self):
        calls = []

        def handler(request):
            calls.append(request)
            content = json.dumps({
                "summary": "anlatı",
                "recommendations": [
                    {"scene_id": 2, "priority": 88, "reason": "Bağı kuruyor", "context_with": [1, 999]},
                    {"scene_id": 999, "priority": 100, "reason": "invalid"},
                ],
            })
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        secret = "prlc_must_not_leak"
        client = ParalonClient(
            LLMConfig(api_key=secret, base_url="http://127.0.0.1:9000/v1"),
            transport=httpx.MockTransport(handler),
        )
        result = client.rerank(
            [{"scene_id": 1}, {"scene_id": 2}],
            {"category": "importance", "target_duration_sec": 30},
        )
        self.assertEqual(len(calls), 1)
        request_body = json.loads(calls[0].content)
        user_payload = json.loads(request_body["messages"][1]["content"])
        self.assertEqual(user_payload["recommendation_limit"], 2)
        self.assertIn("recommendations boş olamaz", request_body["messages"][0]["content"])
        self.assertEqual([row["scene_id"] for row in result["recommendations"]], [2])
        self.assertEqual(result["recommendations"][0]["context_with"], [1])
        self.assertNotIn(secret, str(result))

    def test_missing_key_is_disabled_without_network(self):
        client = ParalonClient(LLMConfig(api_key=None))
        self.assertEqual(client.rerank([], {})["status"], "disabled")

    def test_http_error_is_sanitized(self):
        def handler(request):
            return httpx.Response(401, json={"error": "key prlc_server_echo"})

        client = ParalonClient(
            LLMConfig(api_key="prlc_local_secret", base_url="http://localhost:9000/v1"),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaisesRegex(ParalonError, "provider_http_401"):
            client.rerank([{"scene_id": 1}], {})

    def test_http_error_exposes_safe_code_only(self):
        def handler(request):
            return httpx.Response(400, json={"error": {"code": "unsupported_response_format", "message": "private request text"}})

        client = ParalonClient(
            LLMConfig(api_key="secret", base_url="http://localhost:9000/v1"),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaisesRegex(ParalonError, "provider_http_400_unsupported_response_format") as caught:
            client.rerank([{"scene_id": 1}], {})
        self.assertNotIn("private request text", str(caught.exception))

    def test_timeout_is_sanitized_for_local_fallback(self):
        def handler(request):
            raise httpx.ReadTimeout("simulated timeout", request=request)

        client = ParalonClient(
            LLMConfig(
                api_key="prlc_local_secret",
                base_url="http://localhost:9000/v1",
                timeout_sec=60,
            ),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaisesRegex(ParalonError, "provider_timeout"):
            client.rerank([{"scene_id": 1}], {})

    def test_candidate_pool_rejects_only_above_global_safety_cap(self):
        client = ParalonClient(
            LLMConfig(api_key="test", base_url="http://localhost:9000/v1")
        )
        with self.assertRaisesRegex(ValueError, "72"):
            client.rerank([{"scene_id": index} for index in range(73)], {})

    def test_empty_candidate_pool_is_rejected_before_network(self):
        client = ParalonClient(
            LLMConfig(api_key="test", base_url="http://localhost:9000/v1")
        )
        with self.assertRaisesRegex(ValueError, "boş olamaz"):
            client.rerank([], {})

    def test_large_pool_is_split_into_temporally_distributed_batches(self):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            user_payload = json.loads(body["messages"][1]["content"])
            batch = user_payload["candidates"]
            calls.append([row["scene_id"] for row in batch])
            content = json.dumps({
                "summary": "batch",
                "recommendations": [{
                    "scene_id": batch[0]["scene_id"],
                    "priority": 80,
                    "reason": "temsilci",
                    "context_with": [],
                }],
            })
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        client = ParalonClient(
            LLMConfig(api_key="test", base_url="http://localhost:9000/v1"),
            transport=httpx.MockTransport(handler),
        )
        candidates = [{
            "scene_id": index + 1,
            "start_seconds": index * 10.0,
        } for index in range(25)]
        result = client.rerank(
            candidates,
            {"category": "importance", "target_duration_sec": 180},
        )
        self.assertEqual(result["batch_count"], 3)
        self.assertEqual(result["successful_batch_count"], 3)
        self.assertFalse(result["partial"])
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(len(batch) <= 12 for batch in calls))
        self.assertEqual(sorted(scene_id for batch in calls for scene_id in batch), list(range(1, 26)))

    def test_one_failed_batch_keeps_valid_partial_result(self):
        def handler(request):
            body = json.loads(request.content)
            batch = json.loads(body["messages"][1]["content"])["candidates"]
            if batch[0]["scene_id"] % 2 == 0:
                return httpx.Response(500, json={"error": "synthetic"})
            content = json.dumps({
                "summary": "partial",
                "recommendations": [{
                    "scene_id": batch[0]["scene_id"],
                    "priority": 90,
                    "reason": "geçerli batch",
                    "context_with": [],
                }],
            })
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        client = ParalonClient(
            LLMConfig(api_key="test", base_url="http://localhost:9000/v1"),
            transport=httpx.MockTransport(handler),
        )
        result = client.rerank(
            [{"scene_id": index + 1, "start_seconds": index * 10.0} for index in range(24)],
            {"target_duration_sec": 180},
        )
        self.assertTrue(result["partial"])
        self.assertEqual(result["successful_batch_count"], 1)
        self.assertEqual(result["failed_batch_count"], 1)
        self.assertTrue(result["recommendations"])

    def test_failed_large_batch_is_retried_as_smaller_batches(self):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            batch = json.loads(body["messages"][1]["content"])["candidates"]
            calls.append(len(batch))
            if len(batch) > 6:
                return httpx.Response(500, json={"error": "prompt_too_dense"})
            content = json.dumps({
                "summary": "recovered",
                "recommendations": [{
                    "scene_id": batch[0]["scene_id"],
                    "priority": 90,
                    "reason": "küçük batch başarılı",
                    "context_with": [],
                }],
            })
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        client = ParalonClient(
            LLMConfig(api_key="test", base_url="http://localhost:9000/v1"),
            transport=httpx.MockTransport(handler),
        )
        result = client.rerank(
            [{"scene_id": index + 1, "start_seconds": index * 10.0} for index in range(12)],
            {"target_duration_sec": 180},
        )
        self.assertFalse(result["partial"])
        self.assertEqual(result["failed_batch_count"], 0)
        self.assertEqual(result["recovered_original_batch_count"], 1)
        self.assertEqual(result["retry_batch_count"], 2)
        self.assertEqual(calls, [12, 6, 6])


if __name__ == "__main__":
    unittest.main()
