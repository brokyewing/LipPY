"""
LipPY Inference API — FastAPI.

Modeller: video (sadece dudak) + av-fusion (dudak + ses, Whisper).

Endpoint'ler:
  GET  /api/health                 → servis canlı mı + checkpoint durumu
  GET  /api/models                 → kullanılabilir modeller (video + av-fusion)
  GET  /api/samples                → hazır Türkçe tez/veri seti örnekleri
  POST /api/infer_sample           → örnek klip üzerinden karşılaştırmalı çıkarım
  POST /api/infer                  → kısa video → tahmin
  POST /api/infer_long             → uzun video → async job (VAD altyazı)
  POST /api/infer_long_demo        → uzun video örnek altyazı akışı (async SSE)
  GET  /api/jobs/:id               → job durumu
  GET  /api/jobs/:id/stream        → SSE ilerleme + altyazı segmentleri
  GET  /api/jobs/:id/subtitles.vtt → WebVTT altyazı indir
  GET  /api/jobs/:id/subtitles.srt → SubRip (.srt) altyazı indir
  WS   /ws/live                    → canlı webcam altyazı
"""
from __future__ import annotations

import os

# TensorFlow'u devre dışı bırak — MediaPipe ile protobuf çakışmasını önler.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.inference import infer_from_video_bytes, infer_sample_real, list_available_models, _find_checkpoint
from backend.av_fusion import infer_av_fusion, is_fusion_available
from backend.demo_samples import get_samples, get_sample_video_path
from backend import long_video

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"

# Video boyut limitleri
MAX_VIDEO_SIZE = 50 * 1024 * 1024         # kısa video (sync /api/infer)
MAX_LONG_VIDEO_SIZE = 500 * 1024 * 1024   # uzun video (async /api/infer_long)
READ_CHUNK_SIZE = 64 * 1024               # 64 KB akışlı okuma

# CORS origin listesi
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://localhost:18080",
    "http://127.0.0.1:18080",
]


async def _read_upload_bounded(upload: UploadFile, max_bytes: int) -> bytes:
    """Dosyayı RAM'i taşırmadan 64 KB parçalar halinde okur; sınırı aşarsa 413 fırlatır."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                413,
                f"Dosya çok büyük (maksimum {max_bytes // (1024 * 1024)} MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="LipPY Inference API",
    description="Türkçe dudak okuma — video + AV-Fusion (Whisper) modelleri, 31 fonem CTC",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Servis canlılık ve model ağırlık durumu kontrolü."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        device = "cuda" if has_cuda else "cpu"
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else None
    except ImportError:
        device = "cpu"
        gpu_name = None

    video_ckpt = _find_checkpoint("video") is not None
    fusion_ckpt = is_fusion_available()
    return {
        "status": "ok",
        "device": device,
        "gpu": gpu_name,
        "video_checkpoint": video_ckpt,
        "fusion_checkpoint": fusion_ckpt,
        "mode": "checkpoint" if (video_ckpt or fusion_ckpt) else "demo",
    }


@app.get("/api/models")
def models():
    """Kullanılabilir modeller (video + av-fusion)."""
    available = list_available_models()
    fusion_ready = is_fusion_available()
    available.append({
        "id": "av-fusion",
        "name": "LipPY — Dudak + Ses (AV-Fusion)",
        "vocab_size": 31,
        "available": True,
        "checkpoint_ready": fusion_ready,
        "mode": "checkpoint" if fusion_ready else "demo",
        "checkpoint_path": "checkpoints/av-fusion/best_model.pt" if fusion_ready else None,
        "kenlm_path": None,
        "default_decode": "beam_lm",
        "decode_modes": ["greedy", "beam_lm", "attn"],
        "requires_audio": True,
    })
    return {"models": available}


@app.get("/api/samples")
def list_samples():
    """Tez ve veri setinden hazır Türkçe konuşma örnekleri."""
    return {"samples": get_samples()}


@app.get("/api/samples/{sample_id}/video.mp4")
def sample_video(sample_id: str):
    """Hazır örnek klibin gerçek 96x96 dudak ROI .mp4 videosunu döner."""
    vpath = get_sample_video_path(sample_id)
    if vpath is None or not vpath.is_file():
        raise HTTPException(404, "Örnek video dosyası bulunamadı")
    return FileResponse(str(vpath), media_type="video/mp4")


@app.post("/api/infer_sample")
async def infer_sample(
    sample_id: str = Form("tr-hedef"),
    model: str = Form("video"),
    decode_mode: str = Form("beam_lm"),
):
    """Hazır örnek klip (.mp4) üzerinde gerçek PyTorch modeli ve decode modlarını çalıştırır."""
    success, text, details = await asyncio.to_thread(
        infer_sample_real,
        sample_id=sample_id,
        model_id=model,
        decode_mode=decode_mode,
    )
    return {
        "success": success,
        "text": text,
        "processing": details,
    }


@app.post("/api/infer")
async def infer(
    file: UploadFile = File(...),
    audio: UploadFile | None = File(None),
    model: str = Form("video"),
    decode_mode: str = Form("auto"),
):
    """Video dosyası → fonem decode → metin. AV-Fusion modu destekler."""
    content = await _read_upload_bounded(file, MAX_VIDEO_SIZE)
    if not content:
        raise HTTPException(400, "Boş dosya")

    if model == "av-fusion":
        audio_bytes = None
        if audio is not None:
            audio_bytes = await _read_upload_bounded(audio, MAX_VIDEO_SIZE)
        success, text, details = infer_av_fusion(content, audio_bytes, decode_mode=decode_mode)
    else:
        success, text, details = infer_from_video_bytes(
            content,
            model_id=model,
            decode_mode=decode_mode,
        )

    return {
        "success": success,
        "text": text if success else None,
        "processing": details,
    }


# ─── Uzun video (async job + SSE) ───────────────────────────

@app.post("/api/infer_long")
async def infer_long(
    file: UploadFile = File(...),
    model: str = Form("av-fusion"),
):
    """Uzun video → VAD segmentasyon → altyazı segmentleri (async job)."""
    content = await _read_upload_bounded(file, MAX_LONG_VIDEO_SIZE)
    if not content:
        raise HTTPException(400, "Boş dosya")
    job_id = long_video.create_job(content, model)
    return {"job_id": job_id}


@app.post("/api/infer_long_demo")
async def infer_long_demo(
    model: str = Form("av-fusion"),
):
    """Örnek uzun video altyazı senaryosunu SSE üzerinden canlı akıtır."""
    job_id = long_video.create_job(None, model)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    status = long_video.get_job_status(job_id)
    if status is None:
        raise HTTPException(404, "Job bulunamadı")
    return status


@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    if long_video.get_job_status(job_id) is None:
        raise HTTPException(404, "Job bulunamadı")
    return StreamingResponse(
        long_video.stream_events(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/subtitles.vtt")
def job_subtitles_vtt(job_id: str):
    vtt = long_video.get_job_subtitles_vtt(job_id)
    if vtt is None:
        raise HTTPException(404, "Job bulunamadı")
    return PlainTextResponse(vtt, media_type="text/vtt; charset=utf-8")


@app.get("/api/jobs/{job_id}/subtitles.srt")
def job_subtitles_srt(job_id: str):
    srt = long_video.get_job_subtitles_srt(job_id)
    if srt is None:
        raise HTTPException(404, "Job bulunamadı")
    return PlainTextResponse(srt, media_type="text/plain; charset=utf-8")


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    if not long_video.cancel_job(job_id):
        raise HTTPException(404, "Job bulunamadı")
    return {"canceled": True}


# ─── Canlı çekim (WebSocket) ───────────────────────────

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Canlı webcam: her bağımsız WebM segmenti → infer → caption."""
    model = websocket.query_params.get("model", "av-fusion")
    await websocket.accept()
    seq = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data or len(data) < 200:
                continue
            seq += 1
            try:
                text = await asyncio.to_thread(long_video._infer_segment, data, model)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue
            if text:
                await websocket.send_json({
                    "type": "raw",
                    "text": text,
                    "timestamp": time.time(),
                    "seq": seq,
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/live] hata: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


# ─── Statik Frontend Sunumu (frontend/dist mevcutsa) ────────────────────────
if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        target = (FRONTEND_DIST / full_path).resolve()
        if full_path and target.is_file() and str(target).startswith(str(FRONTEND_DIST.resolve())):
            return FileResponse(str(target))
        index_file = FRONTEND_DIST / "index.html"
        if index_file.is_file():
            return FileResponse(str(index_file))
        raise HTTPException(404, "Sayfa bulunamadı")
