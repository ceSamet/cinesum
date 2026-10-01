import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.core.llm_config import LLMConfig, load_llm_config
from src.features.visual_captioner import enrich_candidates_with_visual_captions
from src.llm.narrative_candidates import (
    build_narrative_candidate_pool,
    narrative_candidate_limit,
    serialize_candidate,
)
from src.llm.paralon_client import PROMPT_VERSION, ParalonClient, ParalonError
from src.rag.scene_rag import (
    ensure_scene_rag,
    load_scene_rag_corpus,
    retrieve_related_scenes,
)
from src.selection.narrative_selector import select_narrative_shots
from src.summary.speech_boundaries import align_segment_boundaries, build_speech_context


LOCAL_WEIGHT = 0.45
LLM_WEIGHT = 0.55
UNRECOMMENDED_LLM_FLOOR = 0.15


def _normalized_scores(candidates: List[Dict[str, Any]], score_key: str) -> Dict[int, float]:
    values = [float(row.get(score_key, 0.0)) for row in candidates]
    low, high = min(values, default=0.0), max(values, default=0.0)
    if high - low <= 1e-12:
        return {int(row["scene_id"]): 1.0 for row in candidates}
    return {
        int(row["scene_id"]): (float(row.get(score_key, 0.0)) - low) / (high - low)
        for row in candidates
    }


def _request_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value
            if isinstance(value, dict)
            and value.get("status") == "applied"
            and not value.get("partial", False)
            else None
        )
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def rerank_narrative_shots(
    shots: List[Dict[str, Any]],
    local_selection: Dict[str, Any],
    *,
    base_dir: Path,
    video_alias: str,
    category: str,
    target_duration_sec: float,
    score_key: str,
    segment_config: Dict[str, float],
    video_duration: float,
    custom_prompt: str = "",
    config: Optional[LLMConfig] = None,
    client: Optional[ParalonClient] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    required_shot_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Apply optional RAG+LLM reranking, preserving local selection on every failure."""
    config = config or load_llm_config(Path(base_dir))
    local_shots = local_selection.get("selected_shots", [])
    fallback = dict(local_selection)
    visual_captioning: Dict[str, Any] = {
        "status": "not_started",
        "reason": "reranker_not_started",
    }
    if target_duration_sec <= 0 or target_duration_sec >= 9999:
        fallback["llm_selection"] = {
            "status": "disabled",
            "reason": "unbounded_selection",
        }
        return fallback
    if not config.enabled:
        fallback["llm_selection"] = {"status": "disabled", "reason": "api_key_missing"}
        return fallback

    candidate_limit = narrative_candidate_limit(video_duration)
    try:
        if progress_callback:
            progress_callback(70, "Narrative RAG bağlamı hazırlanıyor...")
        rag_info = ensure_scene_rag(Path(base_dir), video_alias)
        candidates = build_narrative_candidate_pool(
            shots,
            [int(row["scene_id"]) for row in local_shots],
            score_key=score_key,
            max_candidates=candidate_limit,
            required_shot_ids=required_shot_ids or (),
        )
        if progress_callback:
            progress_callback(71, "BLIP/LLaVA görsel anlam katmanı hazırlanıyor...")
        try:
            visual_captioning = enrich_candidates_with_visual_captions(
                candidates,
                base_dir=Path(base_dir),
                video_alias=video_alias,
            )
        except Exception as exc:
            # VLM is an optional enrichment layer. Model/package/GPU failures must
            # never prevent the established RAG + LLM or local selection path.
            visual_captioning = {
                "status": "fallback_no_captions",
                "reason": f"vlm_{type(exc).__name__}"[:120],
            }
        story_embeddings_path = (
            Path(base_dir) / "outputs" / "features" / "story"
            / f"{video_alias}_story_embeddings.npy"
        )
        story_embeddings = np.load(str(story_embeddings_path)).astype(np.float32)
        rag_corpus = load_scene_rag_corpus(rag_info["database_path"])
        for candidate in candidates:
            story_id = int(candidate.get("story_scene_id", 0) or 0)
            visual = (
                story_embeddings[story_id - 1]
                if 1 <= story_id <= len(story_embeddings)
                else None
            )
            midpoint = (
                float(candidate["start_seconds"]) + float(candidate["end_seconds"])
            ) / 2.0
            related = retrieve_related_scenes(
                rag_info["database_path"],
                " ".join(filter(None, [
                    str(candidate.get("transcript_text", "")),
                    str(candidate.get("visual_caption", "")),
                ])),
                query_visual_vector=visual,
                query_time_seconds=midpoint,
                top_k=2,
                exclude_document_ids=[f"story:{story_id}"] if story_id else None,
                corpus=rag_corpus,
            )
            candidate["related_context"] = [{
                "document_id": row["document_id"],
                "score": row["score"],
                "content": row["content"][:600],
            } for row in related]

        serialized = [serialize_candidate(row) for row in candidates]
        cache_key_data = {
            "rag_source_hash": rag_info["source_hash"],
            "category": category,
            "target_duration_sec": target_duration_sec,
            "custom_prompt": custom_prompt,
            "model": config.model,
            "prompt_version": PROMPT_VERSION,
            "candidates": serialized,
        }
        cache_path = (
            Path(base_dir) / "outputs" / "cache" / "llm" / video_alias
            / f"{_request_hash(cache_key_data)}.json"
        )
        llm_result = _read_cache(cache_path)
        cache_hit = llm_result is not None
        if llm_result is None:
            if progress_callback:
                progress_callback(
                    73,
                    "Paralon anlatı değerlendirmesi çalışıyor... "
                    f"({len(candidates)} aday, ağ timeout {config.timeout_sec:g} sn)",
                )
            llm_result = (client or ParalonClient(config)).rerank(
                serialized,
                {
                    "category": category,
                    "target_duration_sec": target_duration_sec,
                    "custom_prompt": custom_prompt,
                },
            )
            if not llm_result.get("partial", False):
                _write_cache(cache_path, llm_result)

        recommendation_by_id = {
            int(row["scene_id"]): row for row in llm_result["recommendations"]
        }
        local_normalized = _normalized_scores(candidates, score_key)
        local_selected_ids = {int(row["scene_id"]) for row in local_shots}
        partial_result = bool(llm_result.get("partial", False))
        speech_context = build_speech_context(shots)
        hybrid_candidates = []
        for row in candidates:
            item = dict(row)
            scene_id = int(item["scene_id"])
            recommendation = recommendation_by_id.get(scene_id)
            llm_score = (
                float(recommendation["priority"]) / 100.0
                if recommendation is not None
                else UNRECOMMENDED_LLM_FLOOR
            )
            if partial_result:
                hybrid = 0.75 * local_normalized[scene_id] + 0.25 * llm_score
                if scene_id in local_selected_ids:
                    hybrid = max(hybrid, 0.82 + 0.18 * local_normalized[scene_id])
            else:
                hybrid = LOCAL_WEIGHT * local_normalized[scene_id] + LLM_WEIGHT * llm_score
            item["local_normalized_score"] = round(local_normalized[scene_id], 6)
            item["llm_narrative_score"] = round(llm_score, 6)
            item["hybrid_narrative_score"] = round(hybrid, 6)
            item["llm_reason"] = recommendation["reason"] if recommendation else "Yerel güvenli havuz"
            safe_window = align_segment_boundaries(
                max(0.0, float(item["start_seconds"]) - segment_config["pre_context"]),
                min(
                    video_duration,
                    float(item["end_seconds"]) + segment_config["post_context"],
                ),
                shots,
                category=category,
                max_segment_duration=segment_config["max_duration"],
                context=speech_context,
            )
            item["_selection_duration"] = round(
                float(safe_window["end"]) - float(safe_window["start"]), 3
            )
            item["speech_safe_complete_utterance"] = safe_window["complete_utterance"]
            hybrid_candidates.append(item)

        if progress_callback:
            progress_callback(78, "Hibrit süre optimizasyonu yapılıyor...")
        second = select_narrative_shots(
            hybrid_candidates,
            score_key="hybrid_narrative_score",
            category=category,
            target_duration_sec=target_duration_sec,
            video_duration=video_duration,
            segment_config=segment_config,
            required_shot_ids=required_shot_ids,
        )
        if not second.get("selected_shots"):
            raise ParalonError("second_selection_empty")
        second["strategy"] = "rag_llm_hybrid_knapsack"
        second["llm_selection"] = {
            **{key: value for key, value in llm_result.items() if key != "recommendations"},
            "status": "cached" if cache_hit else "applied",
            "cache_hit": cache_hit,
            "candidate_count": len(candidates),
            "candidate_limit": candidate_limit,
            "recommendation_count": len(recommendation_by_id),
            "rag_document_count": rag_info["document_count"],
            "rag_cache_hit": rag_info["cache_hit"],
            "visual_captioning": visual_captioning,
            "selected_reasons": [{
                "scene_id": int(row["scene_id"]),
                "reason": row.get("llm_reason", ""),
                "local_score": row.get("local_normalized_score", 0.0),
                "llm_score": row.get("llm_narrative_score", 0.0),
                "hybrid_score": row.get("hybrid_narrative_score", 0.0),
            } for row in second["selected_shots"]],
        }
        return second
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ParalonError) as exc:
        fallback["llm_selection"] = {
            "status": "fallback_local",
            "reason": str(exc)[:120],
            "candidate_limit": candidate_limit,
            "timeout_sec": config.timeout_sec,
            "visual_captioning": visual_captioning,
        }
        return fallback
