from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np


DEFAULT_PARALON_BASE_URL = "https://paraloncloud.com/v1"
DEFAULT_PARALON_MODEL = "qwen3.8-27b"
PARALON_USER_AGENT = "LexismAI-SceneMind/1.0 (OpenAI-compatible client)"


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _dialogue(scene) -> list[str]:
    lines = []
    seen: set[tuple] = set()
    for turn in scene.transcript:
        key = (turn.get("start_sec"), turn.get("end_sec"), turn.get("text"))
        if key in seen:
            continue
        seen.add(key)
        speaker = turn.get("actor") or turn.get("speaker") or "Konuşmacı"
        text = " ".join(str(turn.get("text") or "").split())[:320]
        if text:
            lines.append(f"{speaker}: {text}")
    return lines


def _cut_title(scene) -> str:
    identity = ", ".join(scene.actors[:2])
    event = next((item.get("label") for item in scene.audio_events if item.get("label")), "")
    subject = identity or event or "Anlatı"
    return f"{subject} · {_clock(scene.start_sec)}–{_clock(scene.end_sec)}"


def _document_text(scene, previous_scene=None, next_scene=None) -> str:
    parts = [
        f"Başlık: {_cut_title(scene)}",
        f"Zaman: {_clock(scene.start_sec)}–{_clock(scene.end_sec)}",
        f"Görsel tanım: {scene.description or 'Yok'}",
        f"Karakterler: {', '.join(scene.actors) or 'Yok'}",
        f"Konuşmalar: {' | '.join(_dialogue(scene)) or 'Yok'}",
        "Ses olayları: " + (
            ", ".join(str(item.get("label")) for item in scene.audio_events if item.get("label"))
            or "Yok"
        ),
    ]
    if previous_scene is not None:
        parts.append(f"Önceki cut: {previous_scene.description or 'Yok'}")
    if next_scene is not None:
        parts.append(f"Sonraki cut: {next_scene.description or 'Yok'}")
    return "\n".join(parts)


class SceneVectorStore:
    """Small persistent local vector store backed by one SQLite file per video."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents(kind)")

    def replace(self, documents: list[dict]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM documents")
            self.connection.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item["id"], item["kind"], item["title"], item["content"],
                        json.dumps(item["metadata"], ensure_ascii=False),
                        int(len(item["vector"])),
                        np.asarray(item["vector"], dtype=np.float32).tobytes(),
                    )
                    for item in documents
                ],
            )

    def search(self, vector: np.ndarray, limit: int = 5, kind: str | None = None) -> list[dict]:
        query = "SELECT id, kind, title, content, metadata, dimensions, vector FROM documents"
        parameters: tuple = ()
        if kind:
            query += " WHERE kind = ?"
            parameters = (kind,)
        target = np.asarray(vector, dtype=np.float32)
        target /= max(float(np.linalg.norm(target)), 1e-8)
        matches = []
        for row in self.connection.execute(query, parameters):
            candidate = np.frombuffer(row[6], dtype=np.float32, count=row[5])
            if len(candidate) != len(target):
                continue
            candidate = candidate / max(float(np.linalg.norm(candidate)), 1e-8)
            matches.append({
                "id": row[0], "kind": row[1], "title": row[2], "content": row[3],
                "metadata": json.loads(row[4]), "score": round(float(target @ candidate), 4),
            })
        return sorted(matches, key=lambda item: item["score"], reverse=True)[:limit]

    def close(self) -> None:
        self.connection.close()


def _build_documents(results: list, story_scenes: list[dict], embeddings: np.ndarray) -> list[dict]:
    documents = []
    source_by_cut = {scene.index: index for index, scene in enumerate(results)}
    for source_index, scene in enumerate(results):
        documents.append({
            "id": f"cut:{scene.index}",
            "kind": "cut",
            "title": _cut_title(scene),
            "content": _document_text(
                scene,
                results[source_index - 1] if source_index else None,
                results[source_index + 1] if source_index + 1 < len(results) else None,
            ),
            "metadata": {
                "cut_index": scene.index,
                "start_sec": scene.start_sec,
                "end_sec": scene.end_sec,
                "actors": scene.actors,
            },
            "vector": embeddings[source_index],
        })
    for story in story_scenes:
        member_indices = [
            source_by_cut[index] for index in story.get("cut_indices", []) if index in source_by_cut
        ]
        if not member_indices:
            continue
        member_scenes = [results[index] for index in member_indices]
        story_vector = np.asarray(embeddings[member_indices], dtype=np.float32).mean(axis=0)
        story_vector /= max(float(np.linalg.norm(story_vector)), 1e-8)
        dialogue = []
        for scene in member_scenes:
            dialogue.extend(_dialogue(scene))
        story_title = story.get("label") or f"Anlatı bölümü {story['index']}"
        documents.append({
            "id": f"story:{story['index']}",
            "kind": "story",
            "title": story_title,
            "content": "\n".join([
                f"Başlık: {story_title}",
                f"Tanım: {story.get('description') or 'Yok'}",
                f"Zaman: {_clock(story.get('start_sec', 0))}–{_clock(story.get('end_sec', 0))}",
                f"Karakterler: {', '.join(dict.fromkeys(a for s in member_scenes for a in s.actors)) or 'Yok'}",
                f"Konuşmalar: {' | '.join(dialogue[:24]) or 'Yok'}",
                f"Cutlar: {story.get('cut_indices', [])}",
            ]),
            "metadata": {
                "story_index": story["index"], "cut_indices": story.get("cut_indices", []),
                "start_sec": story.get("start_sec"), "end_sec": story.get("end_sec"),
            },
            "vector": story_vector,
        })
    return documents


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _paralon_chat(
    prompt: str,
    model_override: str | None = None,
    max_tokens: int = 768,
    request_timeout: int = 45,
) -> tuple[dict, dict]:
    api_key = os.getenv("PARALON_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PARALON_API_KEY tanımlı değil")
    base_url = os.getenv("PARALON_BASE_URL", DEFAULT_PARALON_BASE_URL).rstrip("/")
    model = model_override or os.getenv("PARALON_MODEL", DEFAULT_PARALON_MODEL)
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Sen bir video kurgu editörüsün. Yalnız verilen kanıta dayan, anlatı "
                    "bağlamını ve neden-sonuç zincirini koru, süre bütçesini aşma ve "
                    "yalnız JSON döndür. /no_think"
                ),
            },
            {"role": "user", "content": prompt + "\n/no_think"},
        ],
        "temperature": 0.1,
        # The selection JSON is deliberately compact. A large output budget
        # makes the 27B model exceed Paralon's Cloudflare gateway deadline.
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            # Cloudflare rejects Python urllib's default user agent with error
            # 1010 even when the Paralon API key is valid.
            "User-Agent": PARALON_USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace").strip()
        try:
            error_payload = json.loads(detail)
            detail = str(error_payload.get("title") or error_payload.get("detail") or detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        detail = " ".join(detail.split())
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Paralon HTTP {exc.code}{suffix}") from exc
    content = raw["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Paralon boş yanıt döndürdü")
    return _extract_json(content), {
        "provider": "ParalonCloud",
        "model": model,
        "usage": raw.get("usage") or {},
    }


def _candidate_pool(
    results: list,
    story_scenes: list[dict],
    base_selected: list[int],
    scores: list[float],
    excluded: set[int],
    maximum: int = 48,
) -> list[int]:
    pool = set(base_selected)
    for index in base_selected:
        pool.update((index - 1, index + 1))
    pool.update(sorted(range(len(results)), key=lambda index: scores[index], reverse=True)[:24])
    source_by_cut = {scene.index: index for index, scene in enumerate(results)}
    for story in story_scenes:
        members = [source_by_cut[cut] for cut in story.get("cut_indices", []) if cut in source_by_cut]
        if members:
            pool.add(max(members, key=lambda index: scores[index]))
    for bin_index in range(12):
        members = [
            index for index in range(len(results))
            if min(11, int(index / max(1, len(results)) * 12)) == bin_index
        ]
        if members:
            pool.add(max(members, key=lambda index: scores[index]))
    allowed = [index for index in pool if 0 <= index < len(results) and index not in excluded]
    required = set(base_selected)
    ordered = sorted(allowed, key=lambda index: (index not in required, -scores[index]))
    # Every first-pass Knapsack choice must reach the reviewer. The pool may
    # grow beyond the normal cap when a short-cut-heavy video selects many cuts.
    return sorted(ordered[:max(maximum, len(required))])


def select_with_paralon_rag(
    results: list,
    story_scenes: list[dict],
    embeddings: np.ndarray,
    base_selected: list[int],
    scores: list[float],
    costs: list[int],
    positions: list[float],
    budget: int,
    excluded: set[int],
    database_path: Path,
    knapsack_select: Callable[[list[float], list[int], int], list[int]],
    cancel_check: Callable[[], bool] | None = None,
    selection_request: dict | None = None,
    selection_profile: dict | None = None,
    diversity_select: Callable | None = None,
) -> tuple[list[int], dict]:
    store = SceneVectorStore(database_path)
    try:
        documents = _build_documents(results, story_scenes, embeddings)
        store.replace(documents)
        if cancel_check and cancel_check():
            raise RuntimeError("Analiz iptal edildi")
        pool = _candidate_pool(results, story_scenes, base_selected, scores, excluded)
        cut_to_source = {scene.index: index for index, scene in enumerate(results)}
        story_by_cut = {
            cut_index: story.get("index")
            for story in story_scenes
            for cut_index in story.get("cut_indices", [])
        }
        records = []
        for source_index in pool:
            scene = results[source_index]
            related = [
                match for match in store.search(embeddings[source_index], limit=5, kind="cut")
                if match["metadata"].get("cut_index") != scene.index
            ][:2]
            records.append({
                "cut_index": scene.index,
                "time": f"{_clock(scene.start_sec)}-{_clock(scene.end_sec)}",
                "cost_sec": costs[source_index],
                "algorithm_score": round(scores[source_index], 1),
                "baseline_selected": source_index in base_selected,
                "event_group": story_by_cut.get(scene.index),
                "visual": (scene.description or "")[:240],
                "actors": scene.actors[:4],
                "dialogue": [line[:180] for line in _dialogue(scene)[:2]],
                "audio_events": [
                    item.get("label") for item in scene.audio_events if item.get("label")
                ][:3],
                "rag_related": [
                    {
                        "cut_index": item["metadata"].get("cut_index"),
                        "similarity": item["score"],
                    }
                    for item in related
                ],
            })
        def batch_prompt(batch: list[dict]) -> str:
            request_context = selection_request or {}
            return (
                "Video kurgusu için aşağıdaki her aday cut'a içerik önem puanı ver. "
                "Filme üç perde, kahraman yolculuğu veya Hollywood dönüm noktaları dayatma. "
                "İçeriğin kendi olay kümelerini, karakter hatlarını, komedide setup-punchline "
                "birliğini, aksiyonda eylem değişimini, atmosferik içerikte motif ve görsel "
                "çeşitliliği dikkate al. Kullanıcı isteğini en önemli seçim sinyali olarak kullan. "
                "Gerekli neden-sonuç bağlamını koru fakat aynı event_group içindeki semantik "
                "tekrarları düşük puanla. Kronolojik konumu yalnız yığılmayı önleyen zayıf bir "
                "sinyal say; boş bir zaman aralığından zorla aday seçme. "
                "Hiçbir adayı atlama. "
                "Yalnız JSON döndür: {\"selections\":[{\"cut_index\":1,"
                "\"priority\":100,\"reason\":\"en fazla 4 kelime\","
                "\"context_with\":[2]}],\"summary\":\"en fazla 8 kelime\"}. "
                "priority 1-100 olsun.\n"
                + json.dumps({
                    "duration_budget_sec": budget,
                    "user_request": request_context,
                    "observed_content_profile": selection_profile or {},
                    "candidates": batch,
                }, ensure_ascii=False)
            )

        primary_model = os.getenv("PARALON_MODEL", DEFAULT_PARALON_MODEL)
        fallback_model = os.getenv("PARALON_FALLBACK_MODEL", "qwen3-3b").strip()
        batches = [records[index:index + 6] for index in range(0, len(records), 6)]

        def review_batch(batch: list[dict]) -> tuple[dict, dict, bool]:
            prompt = batch_prompt(batch)
            try:
                decision, meta = _paralon_chat(
                    prompt, max_tokens=768, request_timeout=20
                )
                return decision, meta, False
            except Exception as primary_error:
                if not fallback_model or fallback_model == primary_model:
                    raise primary_error
                decision, meta = _paralon_chat(
                    prompt,
                    model_override=fallback_model,
                    max_tokens=768,
                    request_timeout=45,
                )
                return decision, meta, True

        reviewed: list[tuple[dict, dict, bool]] = []
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
            futures = [executor.submit(review_batch, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    reviewed.append(future.result())
                except Exception as exc:
                    failures.append(" ".join(str(exc).split())[:120])
        if not reviewed:
            raise RuntimeError(
                "Paralon grupları yanıtlamadı: " + (failures[0] if failures else "bilinmeyen hata")
            )
        decision = {
            "selections": [
                item
                for batch_decision, _, _ in reviewed
                for item in batch_decision.get("selections", [])
            ],
            "summary": " | ".join(
                str(batch_decision.get("summary") or "").strip()
                for batch_decision, _, _ in reviewed
                if batch_decision.get("summary")
            ),
        }
        fallback_count = sum(used_fallback for _, _, used_fallback in reviewed)
        usage: dict[str, int] = {}
        for _, meta, _ in reviewed:
            for key, value in (meta.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + int(value)
        provider_meta = {
            "provider": "ParalonCloud",
            "model": (
                f"{primary_model} + {fallback_model} fallback"
                if fallback_count else primary_model
            ),
            "usage": usage,
            "batch_count": len(batches),
            "completed_batch_count": len(reviewed),
            "fallback_batch_count": fallback_count,
            "failed_batch_count": len(failures),
        }
        recommendations: dict[int, dict] = {}
        for item in decision.get("selections", []):
            try:
                cut_index = int(item["cut_index"])
                source_index = cut_to_source[cut_index]
                priority = max(1.0, min(100.0, float(item.get("priority", 50))))
            except (KeyError, TypeError, ValueError):
                continue
            if source_index not in pool or source_index in excluded:
                continue
            recommendations[source_index] = {
                "priority": priority,
                "reason": " ".join(str(item.get("reason") or "LLM anlatı tercihi").split())[:240],
                "context_with": [int(value) for value in item.get("context_with", []) if str(value).isdigit()],
            }
        if not recommendations:
            raise ValueError("Paralon geçerli bir seçim döndürmedi")

        candidate_indices = sorted(pool)
        hybrid_values = [
            (
                scores[index] * 0.42 + recommendations[index]["priority"] * 0.58
                if index in recommendations
                else scores[index]
            )
            for index in candidate_indices
        ]
        if diversity_select is not None:
            candidate_groups = [
                int(story_by_cut.get(results[index].index) or -(index + 1))
                for index in candidate_indices
            ]
            local_selected = diversity_select(
                hybrid_values,
                [costs[index] for index in candidate_indices],
                [positions[index] for index in candidate_indices],
                budget,
                set(),
                candidate_groups,
            )
        else:
            local_selected = knapsack_select(
                hybrid_values,
                [costs[index] for index in candidate_indices],
                budget,
            )
        selected = sorted(candidate_indices[index] for index in local_selected)
        if not selected:
            raise ValueError("Paralon önerileri süre bütçesine yerleşmedi")
        for index in selected:
            if index in recommendations:
                results[index].scoring_breakdown["llm_narrative"] = {
                    "label": "Paralon anlatı kararı",
                    "raw": round(recommendations[index]["priority"] / 100.0, 3),
                    "weight": 0,
                    "points": round(recommendations[index]["priority"] * 0.58, 1),
                }
                results[index].selection_reasons = [recommendations[index]["reason"]]
            else:
                results[index].selection_reasons = [
                    "LLM bu adayı atladı; yerel puanı korunarak değerlendirildi"
                ]
        return selected, {
            "status": "applied",
            **provider_meta,
            "candidate_count": len(pool),
            "recommended_count": len(recommendations),
            "local_fallback_count": len(pool) - len(recommendations),
            "summary": " ".join(str(decision.get("summary") or "").split())[:600],
            "vector_database": str(database_path),
            "document_count": len(documents),
        }
    except Exception as exc:
        return base_selected, {
            "status": "fallback_knapsack",
            "provider": "ParalonCloud",
            "model": os.getenv("PARALON_MODEL", DEFAULT_PARALON_MODEL),
            "reason": " ".join(str(exc).split())[:240],
            "vector_database": str(database_path),
        }
    finally:
        store.close()
