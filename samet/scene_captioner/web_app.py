from __future__ import annotations

import json
import hashlib
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from .analyzer import (
    AnalysisCancelled,
    MODEL_DIR,
    _project_embeddings,
    analyze_video,
    reanalyze_from_cache,
)
from .cli import load_env_file


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "web"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
JOBS_DIR = OUTPUTS_DIR / "jobs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analyzer")
state_lock = threading.Lock()
job_states: dict[str, dict] = {}
job_cancel_events: dict[str, threading.Event] = {}
job_futures: dict[str, object] = {}


def _stop_process() -> None:
    """Stop the process after Flask has had time to send the HTTP response."""
    os._exit(0)


@app.after_request
def disable_browser_cache(response):
    """The UI is local and changes frequently; never serve stale dashboard code."""
    if response.mimetype in {"text/html", "application/json"}:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _job_id(value: str) -> str:
    try:
        return uuid.UUID(value).hex
    except ValueError as exc:
        raise ValueError("Geçersiz iş kimliği") from exc


def _save_state(job_id: str, state: dict) -> None:
    with state_lock:
        job_states[job_id] = state
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_state(job_id: str) -> dict:
    with state_lock:
        state = job_states.get(job_id)
    if state:
        return state
    path = JOBS_DIR / job_id / "state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("İş bulunamadı")


def _decorate(job_id: str, state: dict) -> dict:
    payload = dict(state)
    analysis = payload.get("analysis")
    if analysis:
        analysis = dict(analysis)
        scenes = []
        for scene in analysis.get("scenes", []):
            item = dict(scene)
            item["frame_url"] = f"/outputs/{job_id}/frames/{Path(item['frame_path']).name}"
            scenes.append(item)
        analysis["scenes"] = scenes
        cast = []
        for person in (analysis.get("cast") or [])[:24]:
            person = dict(person)
            portrait_path = person.get("portrait_path")
            if portrait_path:
                portrait_name = Path(portrait_path).name
                if (JOBS_DIR / job_id / "portraits" / portrait_name).exists():
                    person["portrait_url"] = f"/outputs/{job_id}/portraits/{portrait_name}"
            cast.append(person)
        analysis["cast"] = cast
        if not analysis.get("embedding_map"):
            embedding_path = JOBS_DIR / job_id / "scene_embeddings.npy"
            cache_path = JOBS_DIR / job_id / "embedding_map.json"
            try:
                if cache_path.exists():
                    embedding_map = json.loads(cache_path.read_text(encoding="utf-8"))
                elif embedding_path.exists():
                    import numpy as np

                    coordinates, explained = _project_embeddings(
                        np.load(embedding_path, allow_pickle=False)
                    )
                    embedding_map = {
                        "method": "PCA",
                        "explained_variance": explained,
                        "points": [
                            {
                                "cut_index": scene.get("index"),
                                "x": round(float(coordinates[index, 0]), 5),
                                "y": round(float(coordinates[index, 1]), 5),
                                "start_sec": scene.get("start_sec"),
                                "end_sec": scene.get("end_sec"),
                                "importance": scene.get("importance_score", 0),
                                "selected": scene.get("selected", False),
                                "description": scene.get("description", ""),
                                "actors": scene.get("actors", []),
                            }
                            for index, scene in enumerate(analysis.get("scenes", []))
                            if index < len(coordinates)
                        ],
                        "stories": [],
                    }
                    cache_path.write_text(
                        json.dumps(embedding_map, ensure_ascii=False), encoding="utf-8"
                    )
                else:
                    embedding_map = None
                if embedding_map:
                    analysis["embedding_map"] = embedding_map
            except Exception:
                pass
        cleaned_warnings = []
        for warning in analysis.get("warnings", []):
            lowered = warning.lower()
            if "out of memory" in lowered or "cuda" in lowered:
                warning = "GPU belleği dolduğu için bazı analizler tamamlanamadı."
            elif len(warning) > 180 and not warning.startswith("Paralon seçimi kullanılamadı"):
                warning = "Bir analiz bileşeni tamamlanamadı."
            if warning not in cleaned_warnings:
                cleaned_warnings.append(warning)
        analysis["warnings"] = cleaned_warnings
        if analysis.get("highlight_path"):
            analysis["highlight_url"] = f"/outputs/{job_id}/highlight.mp4"
        payload["analysis"] = analysis
    upload_name = payload.get("video_filename") or payload.get("title")
    if upload_name and (UPLOADS_DIR / job_id / upload_name).exists():
        payload["source_url"] = f"/uploads/{job_id}/{upload_name}"
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_cached_job(video_path: Path, digest: str, current_job_id: str) -> str | None:
    candidates = []
    if not JOBS_DIR.exists():
        return None
    for directory in JOBS_DIR.iterdir():
        if not directory.is_dir() or directory.name == current_job_id:
            continue
        try:
            state = _read_state(directory.name)
            analysis = state.get("analysis") or {}
            if state.get("status") != "complete" or not analysis.get("scenes"):
                continue
            upload = UPLOADS_DIR / directory.name / str(state.get("video_filename") or "")
            if not upload.is_file() or upload.stat().st_size != video_path.stat().st_size:
                continue
            candidates.append((state.get("created_at") or "", directory.name, state, upload))
        except Exception:
            continue
    for _, job_id, state, upload in sorted(candidates, reverse=True):
        cached_digest = state.get("video_sha256") or _sha256_file(upload)
        if cached_digest == digest:
            return job_id
    return None


def _copy_cached_assets(source_dir: Path, target_dir: Path) -> None:
    def link_or_copy(source: str, target: str) -> str:
        try:
            os.link(source, target)
            return target
        except OSError:
            return shutil.copy2(source, target)

    for name in ("frames", "portraits"):
        source = source_dir / name
        if source.is_dir():
            shutil.copytree(source, target_dir / name, copy_function=link_or_copy, dirs_exist_ok=True)


def _run_analysis(
    job_id: str,
    video_path: Path,
    backend: str,
    ratio: float,
    language: str | None,
    events: bool,
    selection_request: dict | None = None,
    cache_job_id: str | None = None,
) -> None:
    state = _read_state(job_id)
    state.update({"status": "running", "phase": "Sahneler, ses ve karakterler analiz ediliyor"})
    _save_state(job_id, state)
    try:
        cancel_event = job_cancel_events[job_id]
        def report(message: str, analysis: dict | None = None, percent: int | None = None) -> None:
            current = _read_state(job_id)
            current.update({"status": "running", "phase": message})
            if analysis is not None:
                current["analysis"] = analysis
            if percent is not None:
                current["progress"] = percent
            _save_state(job_id, current)

        if cache_job_id:
            cached_state = _read_state(cache_job_id)
            _copy_cached_assets(JOBS_DIR / cache_job_id, JOBS_DIR / job_id)
            analysis = reanalyze_from_cache(
                video_path=video_path,
                job_dir=JOBS_DIR / job_id,
                cached_analysis=cached_state["analysis"],
                cached_job_dir=JOBS_DIR / cache_job_id,
                budget_ratio=ratio,
                selection_request=selection_request,
                progress=report,
                cancel_check=cancel_event.is_set,
            )
        else:
            analysis = analyze_video(
                video_path=video_path,
                job_dir=JOBS_DIR / job_id,
                backend=backend,
                budget_ratio=ratio,
                language=language,
                audio_events=events,
                selection_request=selection_request,
                progress=report,
                cancel_check=cancel_event.is_set,
            )
        state.update({"status": "complete", "phase": "Tamamlandı", "analysis": analysis})
    except AnalysisCancelled:
        state = _read_state(job_id)
        state.update({
            "status": "cancelled",
            "phase": "Analiz durduruldu",
            "error": None,
        })
    except Exception as exc:
        (JOBS_DIR / job_id / "errors.log").write_text(str(exc), encoding="utf-8")
        state = _read_state(job_id)
        state.update({
            "status": "error",
            "phase": "Analiz durdu",
            "error": "Analiz beklenmedik şekilde durdu. Ayrıntı errors.log dosyasına kaydedildi.",
        })
    _save_state(job_id, state)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def models():
    names = ["llava", "whisper", "audio-events", "embeddings"]
    return jsonify({
        "models": {name: (MODEL_DIR / name).exists() for name in names},
        "faces": (MODEL_DIR / "opencv" / "face_detection_yunet_2023mar.onnx").exists(),
        "paralon": {
            "configured": bool(os.getenv("PARALON_API_KEY", "").strip()),
            "base_url": os.getenv("PARALON_BASE_URL", "https://paraloncloud.com/v1"),
            "model": os.getenv("PARALON_MODEL", "qwen3.8-27b"),
        },
    })


@app.post("/api/shutdown")
def shutdown():
    if request.remote_addr not in {"127.0.0.1", "::1"}:
        return jsonify({"error": "Uygulama yalnızca yerel arayüzden kapatılabilir."}), 403
    for event in job_cancel_events.values():
        event.set()
    executor.shutdown(wait=False, cancel_futures=True)
    threading.Timer(0.35, _stop_process).start()
    return jsonify({"status": "shutting_down", "message": "SceneMind kapatılıyor."}), 202


@app.post("/api/process")
def process_video():
    try:
        upload = request.files.get("video")
        if upload is None or not upload.filename:
            return jsonify({"error": "Önce bir video seçmelisin."}), 400
        filename = secure_filename(upload.filename)
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Bu video biçimi desteklenmiyor."}), 400
        backend = "llava"
        ratio = float(request.form.get("summary_ratio", "22")) / 100
        if not 0.05 <= ratio <= 0.25:
            return jsonify({"error": "Özet süresi videonun %5 ile %25'i arasında olmalı."}), 400
        # These are intentionally fixed in the compact UI: Whisper detects the
        # language and important audio events are always analyzed.
        language = None
        events = True
        allowed_scene_types = {"action", "dialogue", "silent", "music"}
        allowed_character_roles = {"main", "supporting"}
        selection_request = {
            "prompt": " ".join(request.form.get("summary_prompt", "").split())[:1000],
            "scene_types": [
                value for value in request.form.getlist("scene_types")
                if value in allowed_scene_types
            ],
            "character_roles": [
                value for value in request.form.getlist("character_roles")
                if value in allowed_character_roles
            ],
            "people": " ".join(request.form.get("people", "").split())[:500],
            "strict": request.form.get("strict_filter") == "true",
        }

        job_id = uuid.uuid4().hex
        upload_dir = UPLOADS_DIR / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / filename
        upload.save(video_path)
        video_sha256 = _sha256_file(video_path)
        cache_job_id = _find_cached_job(video_path, video_sha256, job_id)
        state = {
            "job_id": job_id,
            "title": filename,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "status": "queued",
            "phase": "Sırada",
            "progress": 0,
            "backend": backend,
            "video_filename": filename,
            "video_sha256": video_sha256,
            "cache_source_job_id": cache_job_id,
            "selection_request": selection_request,
        }
        _save_state(job_id, state)
        cancel_event = threading.Event()
        job_cancel_events[job_id] = cancel_event
        job_futures[job_id] = executor.submit(
            _run_analysis,
            job_id,
            video_path,
            backend,
            ratio,
            language,
            events,
            selection_request,
            cache_job_id,
        )
        return jsonify(state), 202
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/jobs/<job_id>")
def job(job_id: str):
    try:
        safe_id = _job_id(job_id)
        return jsonify(_decorate(safe_id, _read_state(safe_id)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    try:
        safe_id = _job_id(job_id)
        state = _read_state(safe_id)
        if state.get("status") in {"complete", "error", "cancelled"}:
            return jsonify(_decorate(safe_id, state))
        event = job_cancel_events.get(safe_id)
        if event is not None:
            event.set()
        future = job_futures.get(safe_id)
        if future is not None and future.cancel():
            state.update({"status": "cancelled", "phase": "Analiz durduruldu"})
        else:
            state.update({"status": "cancelling", "phase": "Analiz durduruluyor"})
        _save_state(safe_id, state)
        return jsonify(_decorate(safe_id, state)), 202
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/history")
def history():
    items = []
    if JOBS_DIR.exists():
        for directory in JOBS_DIR.iterdir():
            if not directory.is_dir():
                continue
            try:
                state = _read_state(directory.name)
                analysis = state.get("analysis") or {}
                items.append({
                    "job_id": state["job_id"], "title": state.get("title"),
                    "created_at": state.get("created_at"), "status": state.get("status"),
                    "scene_count": analysis.get("scene_count", 0),
                })
            except Exception:
                continue
    items.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return jsonify({"items": items})


@app.get("/outputs/<job_id>/<path:filename>")
def output(job_id: str, filename: str):
    return send_from_directory(JOBS_DIR / _job_id(job_id), filename)


@app.get("/uploads/<job_id>/<filename>")
def uploaded_video(job_id: str, filename: str):
    return send_from_directory(UPLOADS_DIR / _job_id(job_id), filename)


def main() -> None:
    load_env_file()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=7860, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
