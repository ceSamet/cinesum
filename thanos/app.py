import os
import sys
import shutil
import json
import uuid
import re
import threading
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Disable HuggingFace symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add src to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.core.analysis_pipeline import analyze_video_features
from src.core.analysis_status import get_analysis_artifact_status
from src.core.pipeline_config import ANALYSIS_PROFILES
from src.core.video_registry import store_video
from src.scoring.scoring_engine import compute_scene_scores_for_video
from src.summary.summary_exporter import export_category_summary
from src.summary.custom_query_summary import generate_custom_query_summary

# Initialize FastAPI App
app = FastAPI(title="CineSum AI — Video Summarization Engine", version="1.0.0")

# Mount Static Files & Directories (Including Dataset Video Directory)
(BASE_DIR / "outputs").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "dataset").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(BASE_DIR / "outputs")), name="outputs")
app.mount("/dataset", StaticFiles(directory=str(BASE_DIR / "dataset")), name="dataset")

# Setup Jinja2 Templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global Task Progress Storage
progress_store: Dict[str, Dict[str, Any]] = {}
progress_store_lock = threading.Lock()
analysis_locks: Dict[str, threading.Lock] = {}
analysis_locks_guard = threading.Lock()


async def run_in_threadpool(func, *args, **kwargs):
    """Keep the event loop responsive even in restricted socketless runtimes."""
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    while not task.done():
        # In restricted runtimes the worker completion cannot wake the loop's
        # socketpair; a short timer also keeps /api/progress serviceable.
        await asyncio.sleep(0.1)
    return task.result()


def _people_data(video_alias: str) -> Dict[str, Any]:
    path = BASE_DIR / "outputs" / "features" / "audio" / f"{video_alias}_people.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

def update_task_progress(task_id: str, progress: int, stage_desc: str, detail: str = "", active_step: str = ""):
    if task_id:
        now = time.time()
        with progress_store_lock:
            previous = progress_store.get(task_id, {})
            events = list(previous.get("events", []))
            if not events or (events[-1]["stage_desc"], events[-1]["detail"]) != (stage_desc, detail):
                if events and events[-1]["status"] == "running":
                    events[-1]["status"] = "completed"
                    events[-1]["duration_sec"] = round(now - events[-1]["started_at"], 1)
                events.append({
                    "stage_desc": stage_desc,
                    "detail": detail,
                    "active_step": active_step,
                    "started_at": now,
                    "status": "failed" if progress == 0 else "completed" if progress >= 100 else "running",
                })
            progress_store[task_id] = {
                "progress": progress,
                "stage_desc": stage_desc,
                "detail": detail,
                "active_step": active_step,
                "events": events[-100:],
                "started_at": previous.get("started_at", now),
                "updated_at": now,
            }

# Pydantic Schemas
class SummarizeRequest(BaseModel):
    task_id: Optional[str] = None
    video_alias: str
    category: str = "action"  # action, dialogue, importance, custom
    custom_prompt: Optional[str] = None
    target_duration_sec: float = 30.0
    analysis_profile: str = "balanced"
    narrative_mode: str = "local"


def get_video_analysis_lock(video_alias: str) -> threading.Lock:
    with analysis_locks_guard:
        if video_alias not in analysis_locks:
            analysis_locks[video_alias] = threading.Lock()
        return analysis_locks[video_alias]

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """
    Serve the main Glassmorphism Web Dashboard using modern Starlette TemplateResponse signature.
    """
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/videos")
async def list_available_videos():
    """
    Get list of all available processed videos.
    """
    video_dir = BASE_DIR / "dataset" / "video"
    if not video_dir.exists():
        return {"videos": []}

    videos = sorted(
        [f.stem for f in video_dir.glob("*.mp4")],
        key=lambda x: int(x.replace("video", "")) if x.replace("video", "").isdigit() else 999
    )
    return {"videos": videos}

@app.get("/api/video-info/{video_alias}")
async def get_video_info(video_alias: str):
    """
    Instantly fetch full video shot scores, timeline data, and transcripts as soon as video is selected from dropdown.
    """
    v_mp4 = BASE_DIR / "dataset" / "video" / f"{video_alias}.mp4"
    if not v_mp4.exists():
        raise HTTPException(status_code=404, detail=f"Video bulunamadı: {video_alias}")

    artifact_status = get_analysis_artifact_status(BASE_DIR, video_alias)
    if not artifact_status["ready"]:
        return {
            "success": False,
            "video_alias": video_alias,
            "analysis_status": "missing",
            "missing_artifacts": artifact_status["missing"],
            "message": (
                "Bu videonun ilk analizi tamamlanmamış. Özet oluşturduğunuzda "
                "eksik aşamalar otomatik olarak çalıştırılacak."
            ),
            "all_scenes": [],
        }

    try:
        scored_scenes = compute_scene_scores_for_video(video_alias, BASE_DIR)
        return {
            "success": True,
            "video_alias": video_alias,
            "video_url": f"/dataset/video/{video_alias}.mp4",
            "all_scenes": scored_scenes,
            "cast": _people_data(video_alias).get("cast", []),
        }
    except Exception as e:
        print(f"[UYARI] {video_alias} için henüz analiz verileri üretilmemiş: {e}")
        return {
            "success": False,
            "video_alias": video_alias,
            "message": f"Bu video henüz analiz edilmemiş. 'Yapay Zekâ Özeti Üret' butonuna basarak ilk analizi başlatabilirsiniz.",
            "all_scenes": [],
        }

@app.get("/api/progress/{task_id}")
async def get_task_progress(task_id: str):
    """
    Real-time Progress Polling Endpoint. NON-BLOCKING!
    """
    fallback = {
        "progress": 5,
        "stage_desc": "Hazırlanıyor...",
        "detail": "Analiz başlatıldı",
        "active_step": "stepScene",
        "events": [],
    }
    with progress_store_lock:
        prog = dict(progress_store.get(task_id, fallback))
        prog["events"] = [dict(event) for event in prog["events"]]
    now = time.time()
    for event in prog["events"]:
        if event["status"] == "running":
            event["duration_sec"] = round(now - event["started_at"], 1)
    if "started_at" in prog:
        prog["elapsed_sec"] = round(now - prog["started_at"], 1)
    return prog

def _sync_process_upload(dest_path: Path, new_alias: str, task_id: str, profile: str):
    """
    Synchronous heavy background processing for uploaded video.
    Executed in a separate worker threadpool so asyncio main thread is never blocked.
    """
    with get_video_analysis_lock(new_alias):
        status = get_analysis_artifact_status(BASE_DIR, new_alias)
        if status["ready"]:
            scene_path = BASE_DIR / "outputs" / "pyscenedetect" / "scene_lists" / f"{new_alias}_scenes.json"
            story_path = BASE_DIR / "outputs" / "features" / "story" / f"{new_alias}_story_scenes.json"
            return {
                "scenes_found": len(json.loads(scene_path.read_text(encoding="utf-8"))),
                "story_scenes_found": len(json.loads(story_path.read_text(encoding="utf-8")).get("story_scenes", [])) if story_path.exists() else 0,
                "cache_hits": {"content_hash": True},
            }
        return analyze_video_features(
            video_alias=new_alias,
            video_path=dest_path,
            base_dir=BASE_DIR,
            profile_name=profile,
            progress_callback=lambda pct, desc, detail, step: update_task_progress(
                task_id, pct, desc, detail, step
            ),
        )

@app.post("/api/upload")
async def upload_and_process_video(
    file: UploadFile = File(...),
    task_id: Optional[str] = None,
    profile: str = "balanced",
):
    if not (file.filename or "").lower().endswith((".mp4", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Lütfen geçerli bir MP4, MOV veya AVI video yükleyin.")

    if not task_id:
        task_id = str(uuid.uuid4())

    profile = profile.strip().lower()
    if profile not in ANALYSIS_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz analiz profili. Seçenekler: {', '.join(sorted(ANALYSIS_PROFILES))}",
        )

    video_dir = BASE_DIR / "dataset" / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    update_task_progress(task_id, 10, "Video dosyası kaydediliyor...", file.filename, "stepScene")
    new_alias, dest_path, duplicate = await run_in_threadpool(
        store_video, file.file, video_dir
    )

    try:
        status = get_analysis_artifact_status(BASE_DIR, new_alias)
        if duplicate and status["ready"]:
            scene_path = BASE_DIR / "outputs" / "pyscenedetect" / "scene_lists" / f"{new_alias}_scenes.json"
            story_path = BASE_DIR / "outputs" / "features" / "story" / f"{new_alias}_story_scenes.json"
            scene_rows = json.loads(scene_path.read_text(encoding="utf-8"))
            story_data = json.loads(story_path.read_text(encoding="utf-8")) if story_path.exists() else {}
            analysis = {
                "scenes_found": len(scene_rows),
                "story_scenes_found": len(story_data.get("story_scenes", [])),
                "cache_hits": {"content_hash": True},
            }
            update_task_progress(task_id, 100, "Önceki analiz kullanıldı", new_alias, "stepExport")
        else:
            analysis = await run_in_threadpool(
                _sync_process_upload, dest_path, new_alias, task_id, profile
            )

        return {
            "success": True,
            "video_alias": new_alias,
            "filename": file.filename,
            "profile": profile,
            "scenes_found": analysis["scenes_found"],
            "story_scenes_found": analysis["story_scenes_found"],
            "cache_hits": analysis["cache_hits"],
            "reused_video": duplicate,
        }
    except Exception as e:
        print(f"[HATA] Video analizinde hata: {e}")
        update_task_progress(task_id, 0, "Hata oluştu", str(e), "")
        raise HTTPException(status_code=500, detail=f"Video analizi sırasında hata oluştu: {str(e)}")

def _sync_generate_summary(
    video_alias: str,
    category: str,
    custom_prompt: str,
    target_dur: float,
    task_id: str,
    analysis_profile: str,
    narrative_mode: str = "local",
):
    """
    Synchronous heavy background processing for summary generation.
    Executed in a separate worker threadpool so asyncio main thread is never blocked.
    """
    v_mp4 = BASE_DIR / "dataset" / "video" / f"{video_alias}.mp4"
    summaries_dir = BASE_DIR / "outputs" / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    analysis_performed = False
    analysis_result = None
    artifact_status = get_analysis_artifact_status(BASE_DIR, video_alias)
    if not artifact_status["ready"]:
        missing_label = ", ".join(artifact_status["missing"])
        update_task_progress(
            task_id,
            12,
            "İlk video analizi gerekiyor...",
            f"Eksik: {missing_label}",
            "stepScene",
        )

        # Prevent two requests from writing the same feature artifacts concurrently.
        with get_video_analysis_lock(video_alias):
            artifact_status = get_analysis_artifact_status(BASE_DIR, video_alias)
            if not artifact_status["ready"]:
                def analysis_progress(pct, desc, detail, step):
                    scaled_progress = 12 + int(max(0, min(pct, 100)) * 0.53)
                    update_task_progress(
                        task_id,
                        scaled_progress,
                        f"İlk analiz — {desc}",
                        detail,
                        step,
                    )

                analysis_result = analyze_video_features(
                    video_alias=video_alias,
                    video_path=v_mp4,
                    base_dir=BASE_DIR,
                    profile_name=analysis_profile,
                    progress_callback=analysis_progress,
                )
                analysis_performed = True

    update_task_progress(task_id, 68, "Görsel ve ses matrisleri okunuyor...", f"Video: {video_alias}", "stepClip")

    # Read scored scenes
    scored_scenes = compute_scene_scores_for_video(video_alias, BASE_DIR)
    debug_res: Dict[str, Any] = {}

    if category == "custom":
        if not custom_prompt:
            custom_prompt = "a car chase"

        update_task_progress(task_id, 69, f"CLIP Text Encoder çalıştırılıyor: '{custom_prompt}'", "Cosine Similarity matrisi oluşturuluyor", "stepClip")

        def p_cb(pct, desc, det, step):
            scaled_progress = 68 + int(max(0, min(pct, 100)) * 0.32)
            update_task_progress(task_id, scaled_progress, desc, det, step)

        res = generate_custom_query_summary(
            video_alias=video_alias,
            text_prompt=custom_prompt,
            base_dir=BASE_DIR,
            target_duration_sec=target_dur,
            progress_callback=p_cb,
            narrative_mode=narrative_mode,
            base_scored_scenes=scored_scenes,
        )
        out_mp4_path = Path(res["output_mp4_path"])
        selected_scenes = res["selected_scenes"]
        all_scenes = scored_scenes
        debug_res = res.get("debug_metadata", {})

    else:
        update_task_progress(task_id, 69, f"{category.upper()} puanları hesaplanıyor...", "Görsel + Ses + Konuşma skor füzyonu", "stepAudio")

        duration_label = "all" if target_dur <= 0 or target_dur >= 9999 else f"{int(target_dur)}s"
        out_mp4_path = summaries_dir / f"{video_alias}_{category}_{duration_label}_{narrative_mode}_summary.mp4"

        def p_cb(pct, desc, det, step):
            update_task_progress(task_id, max(70, pct), desc, det, step)

        export_category_summary(
            video_path=str(v_mp4),
            scored_scenes=scored_scenes,
            category=category,
            target_duration_sec=target_dur if target_dur > 0 else 99999.0,
            output_mp4_path=str(out_mp4_path),
            progress_callback=p_cb,
            narrative_mode=narrative_mode,
            base_dir=BASE_DIR,
            video_alias=video_alias,
        )

        # Load generated debug segments JSON for UI timeline mapping
        debug_file = summaries_dir / "debug" / f"{video_alias}_{category}_{int(target_dur if target_dur > 0 else 99999)}s_segments.json"
        if debug_file.exists():
            with open(debug_file, "r", encoding="utf-8") as f:
                debug_res = json.load(f)
            selected_scenes = []
            for idx, seg in enumerate(debug_res.get("segments", []), 1):
                selected_scenes.append({
                    "scene_id": idx,
                    "start_timecode": f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}",
                    "end_timecode": f"{int(seg['end']//60):02d}:{int(seg['end']%60):02d}",
                    "start_seconds": seg["start"],
                    "end_seconds": seg["end"],
                    "duration_seconds": seg["duration"],
                    "action_score": seg["peak_score"],
                    "dialogue_score": seg["peak_score"],
                    "importance_score": seg["segment_score"],
                    "transcript_text": seg.get("transcript_text", ""),
                })
        else:
            score_key = f"{category}_score"
            sorted_scenes = sorted(scored_scenes, key=lambda s: s.get(score_key, 0.0), reverse=True)
            selected_scenes = []
            acc_dur = 0.0
            for sc in sorted_scenes:
                selected_scenes.append(sc)
                acc_dur += sc["duration_seconds"]
                if target_dur > 0 and acc_dur >= target_dur:
                    break
            selected_scenes.sort(key=lambda s: s["start_seconds"])

        all_scenes = scored_scenes

    for segment in selected_scenes:
        overlapping = [
            row for row in scored_scenes
            if float(row["end_seconds"]) > float(segment["start_seconds"])
            and float(row["start_seconds"]) < float(segment["end_seconds"])
        ]
        segment["actors"] = sorted({actor for row in overlapping for actor in row.get("actors", [])})
        segment["speakers"] = sorted({speaker for row in overlapping for speaker in row.get("speakers", [])})

    update_task_progress(task_id, 100, "Özet Video Hazır!", f"{out_mp4_path.name}", "stepExport")

    relative_url = f"/outputs/summaries/{out_mp4_path.name}"
    return {
        "success": True,
        "video_alias": video_alias,
        "category": category,
        "output_video_url": relative_url,
        "selected_scenes": selected_scenes,
        "all_scenes": all_scenes,
        "cast": _people_data(video_alias).get("cast", []),
        "analysis_performed": analysis_performed,
        "analysis_profile": analysis_profile,
        "analysis": analysis_result,
        "narrative_mode": narrative_mode,
        "llm_selection": debug_res.get("selection", {}).get(
            "llm_selection", {"status": "disabled", "reason": "local_mode"}
        ),
    }

@app.post("/api/summarize")
async def generate_summary(req: SummarizeRequest):
    """
    Generate custom video summary. NON-BLOCKING threadpool execution!
    """
    task_id = req.task_id or str(uuid.uuid4())
    video_alias = req.video_alias
    category = req.category.lower()
    custom_prompt = req.custom_prompt
    target_dur = req.target_duration_sec
    analysis_profile = req.analysis_profile.strip().lower()
    narrative_mode = req.narrative_mode.strip().lower()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_alias):
        raise HTTPException(status_code=400, detail="Geçersiz video alias'ı.")
    if category not in {"action", "dialogue", "importance", "custom"}:
        raise HTTPException(status_code=400, detail="Geçersiz özet kategorisi.")
    if analysis_profile not in ANALYSIS_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz analiz profili. Seçenekler: {', '.join(sorted(ANALYSIS_PROFILES))}",
        )
    if target_dur < 0:
        raise HTTPException(status_code=400, detail="Hedef özet süresi negatif olamaz.")
    if narrative_mode not in {"local", "rag_llm"}:
        raise HTTPException(status_code=400, detail="Geçersiz anlatı modu.")

    v_mp4 = BASE_DIR / "dataset" / "video" / f"{video_alias}.mp4"
    if not v_mp4.exists():
        raise HTTPException(status_code=404, detail=f"Video bulunamadı: {video_alias}")

    try:
        update_task_progress(task_id, 10, "Yapay Zekâ Analizi Başlatılıyor...", f"Kategori: {category.upper()}", "stepScene")

        result = await run_in_threadpool(
            _sync_generate_summary,
            video_alias,
            category,
            custom_prompt,
            target_dur,
            task_id,
            analysis_profile,
            narrative_mode,
        )
        return result

    except Exception as e:
        print(f"[HATA] Özetleme API hatası: {e}")
        update_task_progress(task_id, 0, "Özetleme Hatası", str(e), "")
        raise HTTPException(status_code=500, detail=f"Özet üretilemedi: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
