import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


RAG_SCHEMA_VERSION = "1.0"
TEXT_VECTOR_SIZE = 512
TEXT_WEIGHT = 0.70
VISUAL_WEIGHT = 0.25
TEMPORAL_WEIGHT = 0.05
_TOKEN_RE = re.compile(r"(?u)\b\w+\b")


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def hashing_text_vector(text: str, size: int = TEXT_VECTOR_SIZE) -> np.ndarray:
    tokens = _TOKEN_RE.findall((text or "").casefold())
    features = tokens + [f"{left}::{right}" for left, right in zip(tokens, tokens[1:])]
    vector = np.zeros(size, dtype=np.float32)
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % size
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    return _normalize(vector)


def _source_hash(story_json_path: Path, story_embeddings_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(RAG_SCHEMA_VERSION.encode("ascii"))
    digest.update(str(TEXT_VECTOR_SIZE).encode("ascii"))
    for path in (story_json_path, story_embeddings_path):
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _vector_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def _blob_vector(blob: Optional[bytes]) -> np.ndarray:
    if not blob:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(blob, dtype="<f4")


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    return float(np.clip(np.dot(_normalize(left), _normalize(right)), -1.0, 1.0))


def _position_label(start: float, video_duration: float) -> str:
    ratio = start / max(video_duration, 0.001)
    if ratio <= 0.20:
        return "intro"
    if ratio >= 0.75:
        return "ending"
    return "middle"


def _document_content(story: Dict[str, Any], video_duration: float) -> str:
    transcript = " ".join(str(story.get("transcript_text", "")).split())
    return (
        f"StoryScene {int(story['story_scene_id'])}. "
        f"Narrative region: {_position_label(float(story['start_seconds']), video_duration)}. "
        f"Mode: {story.get('dominant_mode', 'general')}. "
        f"Time: {float(story['start_seconds']):.1f}-{float(story['end_seconds']):.1f}. "
        f"Shots: {', '.join(str(value) for value in story.get('shot_ids', []))}. "
        f"Transcript: {transcript or '[no speech]'}"
    )


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            video_alias TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            text_vector BLOB NOT NULL,
            visual_vector BLOB,
            schema_version TEXT NOT NULL,
            source_hash TEXT NOT NULL
        );
        CREATE INDEX idx_documents_video ON documents(video_alias);
    """)


def build_scene_rag(
    video_alias: str,
    story_json_path: Path,
    story_embeddings_path: Path,
    database_path: Path,
) -> Dict[str, Any]:
    story_json_path = Path(story_json_path)
    story_embeddings_path = Path(story_embeddings_path)
    database_path = Path(database_path)
    source_hash = _source_hash(story_json_path, story_embeddings_path)
    story_data = json.loads(story_json_path.read_text(encoding="utf-8"))
    story_embeddings = np.load(str(story_embeddings_path)).astype(np.float32)
    stories = story_data.get("story_scenes", [])
    video_duration = max(
        (float(story.get("end_seconds", 0.0)) for story in stories),
        default=0.0,
    )
    if len(story_embeddings) != len(stories):
        raise ValueError("StoryScene kayıtları ile centroid embedding sayısı eşleşmiyor")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    connection = sqlite3.connect(temporary_path)
    try:
        _initialize_schema(connection)
        metadata = {
            "schema_version": RAG_SCHEMA_VERSION,
            "source_hash": source_hash,
            "video_alias": video_alias,
            "document_count": str(len(stories)),
            "created_at_unix": str(round(time.time(), 3)),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            metadata.items(),
        )
        for index, story in enumerate(stories):
            story_id = int(story["story_scene_id"])
            content = _document_content(story, video_duration)
            document_metadata = {
                "story_scene_id": story_id,
                "shot_ids": [int(value) for value in story.get("shot_ids", [])],
                "start_seconds": float(story.get("start_seconds", 0.0)),
                "end_seconds": float(story.get("end_seconds", 0.0)),
                "duration_seconds": float(story.get("duration_seconds", 0.0)),
                "speech_ratio": float(story.get("speech_ratio", 0.0)),
                "dominant_mode": story.get("dominant_mode", "general"),
                "narrative_region": _position_label(
                    float(story.get("start_seconds", 0.0)), video_duration
                ),
                "video_duration": video_duration,
            }
            connection.execute(
                """INSERT INTO documents(
                    document_id, video_alias, title, content, metadata_json,
                    text_vector, visual_vector, schema_version, source_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"story:{story_id}",
                    video_alias,
                    f"StoryScene {story_id}",
                    content,
                    json.dumps(document_metadata, ensure_ascii=False, sort_keys=True),
                    _vector_blob(hashing_text_vector(content)),
                    _vector_blob(_normalize(story_embeddings[index])),
                    RAG_SCHEMA_VERSION,
                    source_hash,
                ),
            )
        connection.commit()
    finally:
        connection.close()
    temporary_path.replace(database_path)
    return {
        "database_path": str(database_path),
        "document_count": len(stories),
        "source_hash": source_hash,
        "cache_hit": False,
    }


def _cached_index_matches(database_path: Path, source_hash: str) -> bool:
    if not database_path.exists():
        return False
    connection = None
    try:
        connection = sqlite3.connect(database_path)
        values = dict(connection.execute("SELECT key, value FROM metadata"))
        return (
            values.get("schema_version") == RAG_SCHEMA_VERSION
            and values.get("source_hash") == source_hash
        )
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            connection.close()


def ensure_scene_rag(base_dir: Path, video_alias: str) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    story_dir = base_dir / "outputs" / "features" / "story"
    story_json_path = story_dir / f"{video_alias}_story_scenes.json"
    story_embeddings_path = story_dir / f"{video_alias}_story_embeddings.npy"
    if not story_json_path.exists() or not story_embeddings_path.exists():
        raise FileNotFoundError(f"StoryScene artifact'leri bulunamadı: {video_alias}")
    database_path = (
        base_dir / "outputs" / "features" / "rag" / f"{video_alias}_scene_rag.sqlite3"
    )
    source_hash = _source_hash(story_json_path, story_embeddings_path)
    if _cached_index_matches(database_path, source_hash):
        connection = sqlite3.connect(database_path)
        try:
            count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        finally:
            connection.close()
        return {
            "database_path": str(database_path),
            "document_count": int(count),
            "source_hash": source_hash,
            "cache_hit": True,
        }
    return build_scene_rag(
        video_alias, story_json_path, story_embeddings_path, database_path
    )


def retrieve_related_scenes(
    database_path: Path,
    query_text: str,
    *,
    query_visual_vector: Optional[np.ndarray] = None,
    query_time_seconds: Optional[float] = None,
    top_k: int = 2,
    exclude_document_ids: Optional[List[str]] = None,
    documents: Optional[List[Dict[str, Any]]] = None,
    corpus: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if top_k <= 0:
        return []
    excluded = set(exclude_document_ids or [])
    query_text_vector = hashing_text_vector(query_text)
    query_visual = (
        _normalize(np.asarray(query_visual_vector, dtype=np.float32))
        if query_visual_vector is not None
        else np.empty(0, dtype=np.float32)
    )
    if corpus is not None:
        loaded_documents = corpus["documents"]
        text_scores = np.maximum(0.0, corpus["text_matrix"] @ query_text_vector)
        if query_visual.size and corpus["visual_matrix"].shape[1] == query_visual.size:
            visual_scores = np.maximum(0.0, corpus["visual_matrix"] @ query_visual)
        else:
            visual_scores = np.zeros(len(loaded_documents), dtype=np.float32)
        if query_time_seconds is None:
            temporal_scores = np.zeros(len(loaded_documents), dtype=np.float32)
        else:
            temporal_scores = np.maximum(
                0.0,
                1.0 - np.abs(corpus["centers"] - float(query_time_seconds))
                / np.maximum(corpus["video_durations"], 0.001),
            )
        combined_scores = (
            TEXT_WEIGHT * text_scores
            + VISUAL_WEIGHT * visual_scores
            + TEMPORAL_WEIGHT * temporal_scores
        )
        ranked_indexes = sorted(
            (
                index for index, document in enumerate(loaded_documents)
                if document["document_id"] not in excluded
            ),
            key=lambda index: (-float(combined_scores[index]), loaded_documents[index]["document_id"]),
        )[:top_k]
        return [{
            "document_id": loaded_documents[index]["document_id"],
            "title": loaded_documents[index]["title"],
            "content": loaded_documents[index]["content"],
            "metadata": loaded_documents[index]["metadata"],
            "score": round(float(combined_scores[index]), 6),
            "text_similarity": round(float(text_scores[index]), 6),
            "visual_similarity": round(float(visual_scores[index]), 6),
            "temporal_score": round(float(temporal_scores[index]), 6),
        } for index in ranked_indexes]

    loaded_documents = documents or load_scene_rag_documents(database_path)

    results = []
    for document in loaded_documents:
        document_id = document["document_id"]
        if document_id in excluded:
            continue
        metadata = document["metadata"]
        text_similarity = max(0.0, _cosine(query_text_vector, document["text_vector"]))
        visual_similarity = max(0.0, _cosine(query_visual, document["visual_vector"]))
        if query_time_seconds is None:
            temporal_score = 0.0
        else:
            center = (
                float(metadata["start_seconds"]) + float(metadata["end_seconds"])
            ) / 2.0
            temporal_score = max(
                0.0,
                1.0 - abs(float(query_time_seconds) - center)
                / max(float(metadata.get("video_duration", 0.0)), 0.001),
            )
        score = (
            TEXT_WEIGHT * text_similarity
            + VISUAL_WEIGHT * visual_similarity
            + TEMPORAL_WEIGHT * temporal_score
        )
        results.append({
            "document_id": document_id,
            "title": document["title"],
            "content": document["content"],
            "metadata": metadata,
            "score": round(score, 6),
            "text_similarity": round(text_similarity, 6),
            "visual_similarity": round(visual_similarity, 6),
            "temporal_score": round(temporal_score, 6),
        })
    results.sort(key=lambda item: (-item["score"], item["document_id"]))
    return results[:top_k]


def load_scene_rag_documents(database_path: Path) -> List[Dict[str, Any]]:
    """Load an index once so a candidate batch does not reopen SQLite repeatedly."""
    connection = sqlite3.connect(Path(database_path))
    try:
        rows = connection.execute(
            """SELECT document_id, title, content, metadata_json,
                      text_vector, visual_vector
               FROM documents"""
        ).fetchall()
    finally:
        connection.close()
    return [{
        "document_id": document_id,
        "title": title,
        "content": content,
        "metadata": json.loads(metadata_json),
        "text_vector": _blob_vector(text_blob).copy(),
        "visual_vector": _blob_vector(visual_blob).copy(),
    } for document_id, title, content, metadata_json, text_blob, visual_blob in rows]


def load_scene_rag_corpus(database_path: Path) -> Dict[str, Any]:
    documents = load_scene_rag_documents(database_path)
    text_matrix = np.stack(
        [document["text_vector"] for document in documents], axis=0
    ) if documents else np.empty((0, TEXT_VECTOR_SIZE), dtype=np.float32)
    visual_size = documents[0]["visual_vector"].size if documents else 0
    visual_matrix = np.stack(
        [document["visual_vector"] for document in documents], axis=0
    ) if documents and visual_size else np.empty((len(documents), 0), dtype=np.float32)
    return {
        "documents": documents,
        "text_matrix": text_matrix,
        "visual_matrix": visual_matrix,
        "centers": np.asarray([
            (float(document["metadata"]["start_seconds"])
             + float(document["metadata"]["end_seconds"])) / 2.0
            for document in documents
        ], dtype=np.float32),
        "video_durations": np.asarray([
            float(document["metadata"].get("video_duration", 0.0))
            for document in documents
        ], dtype=np.float32),
    }
