import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, Optional

import httpx

from src.core.llm_config import LLMConfig
from src.llm.narrative_candidates import MAX_NARRATIVE_CANDIDATES


PROMPT_VERSION = "1.4-visual-semantics"
MAX_LLM_RECOMMENDATIONS = 24
MIN_LLM_RECOMMENDATIONS = 8
MAX_CANDIDATES_PER_REQUEST = 12
MAX_PARALLEL_REQUESTS = 2


class ParalonError(RuntimeError):
    """Sanitized provider failure safe to expose in local diagnostics."""


def recommendation_limit_for_request(
    candidate_count: int,
    target_duration_sec: float,
) -> int:
    """Request enough narrative anchors without forcing an oversized JSON response."""
    duration_based = max(
        MIN_LLM_RECOMMENDATIONS,
        math.ceil(max(0.0, float(target_duration_sec)) / 15.0),
    )
    return min(max(0, int(candidate_count)), MAX_LLM_RECOMMENDATIONS, duration_based)


def build_rerank_request_body(
    candidates: list[Dict[str, Any]],
    request_context: Dict[str, Any],
    model: str,
) -> tuple[Dict[str, Any], int]:
    recommendation_limit = recommendation_limit_for_request(
        len(candidates),
        float(request_context.get("target_duration_sec", 0.0)),
    )
    system_prompt = (
        "Sen bir video kurgu editörüsün. Yalnız geçerli JSON döndür. "
        "Adaylar dışındaki scene_id değerlerini kullanma. Neden-sonuç, soru-cevap, "
        "çatışma-çözüm ve giriş-gelişme-sonuç sürekliliğini değerlendir. "
        "visual_caption alanı varsa görünür kritik eylem, nesne, ölüm, zafer ve "
        "dönüm noktalarını özellikle hesaba kat; görsel açıklamanın ötesinde kimlik uydurma. "
        f"recommendations boş olamaz ve en fazla {recommendation_limit} kayıt içermelidir. "
        "Yalnız en önemli adayları, önem sırasına göre öner."
    )
    safe_request_context = dict(request_context)
    safe_request_context["custom_prompt"] = str(
        safe_request_context.get("custom_prompt", "")
    )[:500]
    user_payload = {
        "prompt_version": PROMPT_VERSION,
        "request": safe_request_context,
        "candidates": candidates,
        "recommendation_limit": recommendation_limit,
        "response_schema": {
            "summary": "short Turkish assessment",
            "recommendations": [{
                "scene_id": "candidate integer",
                "priority": "1-100",
                "reason": "short Turkish reason",
                "context_with": ["candidate scene ids"],
            }],
        },
    }
    return {
        "model": model,
        "temperature": 0.12,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }, recommendation_limit


def extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text or ""):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ParalonError("invalid_json_response")


def validate_rerank_response(
    raw: Dict[str, Any],
    candidate_ids: Iterable[int],
) -> Dict[str, Any]:
    allowed = {int(value) for value in candidate_ids}
    recommendations = []
    seen = set()
    for item in raw.get("recommendations", []):
        if not isinstance(item, dict):
            continue
        try:
            scene_id = int(item.get("scene_id"))
            priority = float(item.get("priority"))
        except (TypeError, ValueError):
            continue
        reason = " ".join(str(item.get("reason", "")).split())[:300]
        if scene_id not in allowed or scene_id in seen or not 1.0 <= priority <= 100.0 or not reason:
            continue
        context_with = []
        for value in item.get("context_with", []):
            try:
                context_id = int(value)
            except (TypeError, ValueError):
                continue
            if context_id in allowed and context_id != scene_id and context_id not in context_with:
                context_with.append(context_id)
        recommendations.append({
            "scene_id": scene_id,
            "priority": round(priority, 3),
            "reason": reason,
            "context_with": context_with[:4],
        })
        seen.add(scene_id)
    return {
        "summary": " ".join(str(raw.get("summary", "")).split())[:500],
        "recommendations": recommendations,
    }


class ParalonClient:
    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.config = config
        self.transport = transport

    def _rerank_batch(
        self,
        candidates: list[Dict[str, Any]],
        request_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_body, recommendation_limit = build_rerank_request_body(
            candidates,
            request_context,
            self.config.model,
        )
        try:
            with httpx.Client(
                timeout=self.config.timeout_sec,
                transport=self.transport,
            ) as client:
                response = client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    json=request_body,
                )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            validated = validate_rerank_response(
                extract_json_object(content),
                [candidate["scene_id"] for candidate in candidates],
            )
            if not validated["recommendations"]:
                raise ParalonError("no_valid_recommendations")
            validated["recommendations"] = validated["recommendations"][:recommendation_limit]
            return validated
        except httpx.TimeoutException as exc:
            raise ParalonError("provider_timeout") from exc
        except httpx.HTTPStatusError as exc:
            # Keep diagnostics useful without persisting the provider's raw body,
            # which may echo request text or credentials.
            detail = ""
            try:
                error = exc.response.json().get("error", {})
                if isinstance(error, dict):
                    code = str(error.get("code") or error.get("type") or "")
                    if code.replace("_", "").replace("-", "").isalnum():
                        detail = f"_{code[:48]}"
            except (ValueError, AttributeError, TypeError):
                pass
            raise ParalonError(f"provider_http_{exc.response.status_code}{detail}") from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ParalonError("provider_invalid_response") from exc

    def rerank(self, candidates: list[Dict[str, Any]], request_context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"status": "disabled", "reason": "api_key_missing"}
        if len(candidates) > MAX_NARRATIVE_CANDIDATES:
            raise ValueError(
                f"Paralon aday havuzu {MAX_NARRATIVE_CANDIDATES} kaydı aşamaz"
            )
        if not candidates:
            raise ValueError("Paralon aday havuzu boş olamaz")

        global_recommendation_limit = recommendation_limit_for_request(
            len(candidates),
            float(request_context.get("target_duration_sec", 0.0)),
        )
        ordered = sorted(
            candidates,
            key=lambda row: (float(row.get("start_seconds", 0.0)), int(row["scene_id"])),
        )
        batch_count = math.ceil(len(ordered) / MAX_CANDIDATES_PER_REQUEST)
        batches = [ordered[index::batch_count] for index in range(batch_count)]
        started = time.perf_counter()
        batch_results: list[tuple[tuple[int, int], Dict[str, Any]]] = []
        failed_initial: Dict[int, str] = {}
        with ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_REQUESTS, len(batches))
        ) as executor:
            futures = {
                executor.submit(self._rerank_batch, batch, request_context): index
                for index, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    batch_results.append(((futures[future], 0), future.result()))
                except ParalonError as exc:
                    failed_initial[futures[future]] = str(exc)

        # A provider can time out or emit invalid JSON for a dense 12-candidate
        # request while succeeding for a smaller prompt. Retry failed original
        # batches once as two smaller requests. Successful runs pay no retry cost.
        retry_jobs: list[tuple[int, int, list[Dict[str, Any]]]] = []
        for original_index in sorted(failed_initial):
            failed_batch = batches[original_index]
            midpoint = max(1, math.ceil(len(failed_batch) / 2))
            parts = [failed_batch[:midpoint], failed_batch[midpoint:]]
            retry_jobs.extend(
                (original_index, part_index, part)
                for part_index, part in enumerate(parts, start=1)
                if part
            )

        retry_successes: Dict[int, set[int]] = {}
        retry_expected: Dict[int, int] = {}
        retry_errors: Dict[int, list[str]] = {}
        for original_index, _, _ in retry_jobs:
            retry_expected[original_index] = retry_expected.get(original_index, 0) + 1
        if retry_jobs:
            with ThreadPoolExecutor(
                max_workers=min(MAX_PARALLEL_REQUESTS, len(retry_jobs))
            ) as executor:
                futures = {
                    executor.submit(self._rerank_batch, batch, request_context): (
                        original_index,
                        part_index,
                    )
                    for original_index, part_index, batch in retry_jobs
                }
                for future in as_completed(futures):
                    original_index, part_index = futures[future]
                    try:
                        batch_results.append(((original_index, part_index), future.result()))
                        retry_successes.setdefault(original_index, set()).add(part_index)
                    except ParalonError as exc:
                        retry_errors.setdefault(original_index, []).append(str(exc))

        recovered_originals = {
            original_index
            for original_index, expected in retry_expected.items()
            if len(retry_successes.get(original_index, set())) == expected
        }
        unresolved_originals = set(failed_initial) - recovered_originals

        if not batch_results:
            errors = [
                error
                for original_index in sorted(unresolved_originals)
                for error in retry_errors.get(
                    original_index,
                    [failed_initial[original_index]],
                )
            ]
            raise ParalonError(errors[0] if errors else "provider_invalid_response")

        batch_results.sort(key=lambda item: item[0])
        recommendations = [
            recommendation
            for _, result in batch_results
            for recommendation in result["recommendations"]
        ]
        recommendations.sort(key=lambda row: (-float(row["priority"]), int(row["scene_id"])))
        summaries = [result["summary"] for _, result in batch_results if result["summary"]]

        return {
            "status": "applied",
            "provider": "paralon",
            "model": self.config.model,
            "prompt_version": PROMPT_VERSION,
            "latency_sec": round(time.perf_counter() - started, 3),
            "candidate_count": len(candidates),
            "batch_count": len(batches),
            "successful_batch_count": len(batches) - len(unresolved_originals),
            "failed_batch_count": len(unresolved_originals),
            "retry_batch_count": len(retry_jobs),
            "retry_successful_batch_count": sum(len(value) for value in retry_successes.values()),
            "recovered_original_batch_count": len(recovered_originals),
            "partial": bool(unresolved_originals),
            "summary": " | ".join(summaries)[:500],
            "recommendations": recommendations[:global_recommendation_limit],
        }
