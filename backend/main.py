"""
LipPY Inference API — FastAPI.

Modeller: video (sadece dudak) + av-fusion (dudak + ses, Whisper).

Endpoint'ler:
  GET  /api/health         → servis canlı mı
  GET  /api/models         → kullanılabilir modeller (checkpoint bulunanlar)
  POST /api/infer          → kısa video → tahmin
  POST /api/infer_long     → uzun video → async job (VAD altyazı)
  GET  /api/jobs/:id/stream → SSE ilerleme + altyazı segmentleri
  WS   /ws/live            → canlı webcam altyazı
"""
from __future__ import annotations

import os

# TensorFlow'u devre dışı bırak — MediaPipe ile protobuf çakışmasını önler.
# transformers PyTorch'la çalışır, TF gereksiz. Bu satırlar HERHANGI BIR transformers
# / mediapipe import'undan ÖNCE olmalı, o yüzden en üstte.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from contextlib import asynccontextmanager

import asyncio
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.inference import infer_from_video_bytes, list_available_models
from backend.av_fusion import infer_av_fusion, is_fusion_available
from backend import long_video

# Video boyut limitleri
MAX_VIDEO_SIZE = 50 * 1024 * 1024         # kısa video (sync /api/infer)
MAX_LONG_VIDEO_SIZE = 500 * 1024 * 1024   # uzun video (async /api/infer_long)

# CORS origin listesi
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",  # vite preview
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: modelleri önceden yükleme opsiyonel
    yield


app = FastAPI(
    title="LipPY Inference API",
    description="Türkçe dudak okuma — video + AV-Fusion (Whisper) modelleri, 31 fonem CTC",
    version="2.0.0",
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
    """Servis canlılık kontrolü."""
    import torch
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/api/models")
def models():
    """Kullanılabilir modeller (checkpoint'i disk'te bulunanlar).

    Frontend bu listeye göre dropdown oluşturur.
    """
    available = list_available_models()
    # AV-Fusion (sesli — symmetric fusion + hybrid CTC/Attn)
    if is_fusion_available():
        available.append({
            "id": "av-fusion",
            "name": "LipPY — Dudak + Ses (AV-Fusion)",
            "vocab_size": 31,
            "available": True,
            "checkpoint_path": "checkpoints/av-fusion/best_model.pt",
            "kenlm_path": None,
            "default_decode": "beam_lm",
            "requires_audio": True,
        })
    return {"models": available}


@app.post("/api/infer")
async def infer(
    file: UploadFile = File(...),
    audio: UploadFile | None = File(None),
    model: str = Form("v1c"),
    decode_mode: str = Form("auto"),
):
    """Video dosyası → fonem decode → metin. AV-Fusion modu destekler.

    Form alanları:
      - file: video dosyası (.mp4, .webm, vb.)
      - audio: ses dosyası (.wav, opsiyonel — av-fusion'da video'dan da çıkarılır)
      - model: model id ('video' | 'av-fusion')
      - decode_mode: 'auto' | 'greedy' | 'beam' | 'beam_lm'
    """
    content = await file.read()
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(
            413,
            f"Video çok büyük (max {MAX_VIDEO_SIZE // (1024 * 1024)} MB)",
        )

    if not content:
        raise HTTPException(400, "Boş dosya")

    if model == "av-fusion":
        # AV-Fusion (sesli — symmetric + hybrid) — greedy/beam_lm/attn
        audio_bytes = None
        if audio is not None:
            audio_bytes = await audio.read()
        success, text, details = infer_av_fusion(content, audio_bytes, decode_mode=decode_mode)
    else:
        success, text, details = infer_from_video_bytes(
            content,
            model_id=model,
            decode_mode=decode_mode,
        )

    response = {
        "success": success,
        "text": text if success else None,
        "processing": details,
    }
    return response


# ─── Uzun video (async job + SSE) ───────────────────────────

@app.post("/api/infer_long")
async def infer_long(
    file: UploadFile = File(...),
    model: str = Form("av-fusion"),
):
    """Uzun video → VAD segmentasyon → altyazı segmentleri (async job)."""
    content = await file.read()
    if not content:
        raise HTTPException(400, "Boş dosya")
    if len(content) > MAX_LONG_VIDEO_SIZE:
        raise HTTPException(
            413, f"Video çok büyük (max {MAX_LONG_VIDEO_SIZE // (1024 * 1024)} MB)"
        )
    job_id = long_video.create_job(content, model)
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


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str):
    if not long_video.cancel_job(job_id):
        raise HTTPException(404, "Job bulunamadı")
    return {"canceled": True}


# ─── Canlı çekim (WebSocket) ───────────────────────────

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Canlı webcam: her bagimsiz WebM segmenti → infer → caption.

    Query param: model (av-fusion=sesli, video=sessiz)
    Gelen: binary WebM segment (frontend her ~3sn bagimsiz segment gonderir).
    Giden: {type:'raw', text, timestamp, seq} JSON.
    """
    model = websocket.query_params.get("model", "av-fusion")
    await websocket.accept()
    seq = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            if not data or len(data) < 2000:   # cok kucuk segment atla
                continue
            seq += 1
            try:
                # long_video._infer_segment WebM bytes alir, model'e gore infer eder (blocking → thread)
                text = await asyncio.to_thread(long_video._infer_segment, data, model)
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})
                continue
            if text:
                await websocket.send_json({
                    "type": "raw", "text": text,
                    "timestamp": time.time(), "seq": seq,
                })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/live] hata: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
