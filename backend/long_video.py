"""
Uzun video → VAD segmentasyon → segment inference → altyazı segmentleri.

Akış:
  1. POST /api/infer_long → video kaydet, job baslat (asyncio task)
  2. ffmpeg ile audio cikar (16kHz mono wav)
  3. silero-VAD → konusma segmentleri [(start, end)] (max 10sn, otomatik bolunur)
  4. Her segment: ffmpeg ile kes → infer (av-fusion-v1cc beam_lm) → {start, end, text}
  5. SSE ile faz/progress + partial_segments akit
  6. Done → transcript

Job store in-memory (dict). Frontend SSE event formati:
  {status, phase, progress}                       — faz + yuzde
  {partial_segments: [{start, end, text}]}        — akista eklenir
  {status:'done', transcript:[...]}               — final
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
import uuid
from typing import Optional

SAMPLE_RATE = 16000
SEG_MAX_SEC = 10.0          # av-fusion egitimi MAX_AUDIO_SEC ile uyumlu
SEG_MIN_SEC = 0.4

# Job store: job_id -> dict(status, phase, progress, queue, transcript, error, video_path, cancel)
_jobs: dict[str, dict] = {}
_vad_model = None


def _load_vad():
    global _vad_model
    if _vad_model is None:
        from silero_vad import load_silero_vad
        _vad_model = load_silero_vad()
    return _vad_model


def _ffmpeg_exe() -> str:
    """av_fusion'daki ffmpeg bulucuyu yeniden kullan (imageio-ffmpeg)."""
    from backend.av_fusion import _get_ffmpeg_exe
    return _get_ffmpeg_exe()


def _extract_audio(video_path: str) -> Optional[str]:
    """Video'dan 16kHz mono wav cikar (VAD icin)."""
    audio_path = video_path + ".vad.wav"
    try:
        r = subprocess.run(
            [_ffmpeg_exe(), "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", str(SAMPLE_RATE), "-ac", "1", audio_path],
            capture_output=True, timeout=300, text=True,
        )
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            return audio_path
        print(f"[long_video] audio cikarilamadi rc={r.returncode}: {(r.stderr or '')[-300:]}")
    except Exception as e:
        print(f"[long_video] audio cikarma hatasi: {e}")
    return None


def _vad_segments(audio_path: str) -> list[tuple[float, float]]:
    """silero-VAD ile konusma segmentleri. max SEG_MAX_SEC (otomatik bolunur)."""
    from silero_vad import read_audio, get_speech_timestamps
    model = _load_vad()
    wav = read_audio(audio_path, sampling_rate=SAMPLE_RATE)
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=SAMPLE_RATE,
        return_seconds=True,
        max_speech_duration_s=SEG_MAX_SEC,
        min_speech_duration_ms=300,
        min_silence_duration_ms=200,
    )
    segs = []
    for t in ts:
        s, e = float(t["start"]), float(t["end"])
        if e - s >= SEG_MIN_SEC:
            segs.append((s, e))
    return segs


def _video_duration(video_path: str) -> float:
    """Video suresini (sn) dondur (cv2 ile)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return (n / fps) if fps > 0 else 0.0
    except Exception as e:
        print(f"[long_video] sure okunamadi: {e}")
        return 0.0


def _fixed_window_segments(video_path: str, win: float = SEG_MAX_SEC) -> list[tuple[float, float]]:
    """Sessiz video / VAD bos icin sabit-pencere segmentasyon."""
    dur = _video_duration(video_path)
    if dur <= 0:
        return []
    segs = []
    t = 0.0
    while t < dur:
        segs.append((t, min(t + win, dur)))
        t += win
    return segs


def _cut_segment(video_path: str, start: float, end: float) -> Optional[bytes]:
    """Video+audio segmentini kes (re-encode, keyframe-safe) → bytes."""
    out = f"{video_path}.seg_{int(start*1000)}_{int(end*1000)}.mp4"
    try:
        r = subprocess.run(
            [_ffmpeg_exe(), "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
             "-i", video_path, "-c:v", "libx264", "-preset", "ultrafast",
             "-c:a", "aac", "-ar", str(SAMPLE_RATE), out],
            capture_output=True, timeout=120, text=True,
        )
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            with open(out, "rb") as f:
                data = f.read()
            return data
        print(f"[long_video] segment kesilemedi ({start:.1f}-{end:.1f}) rc={r.returncode}")
    except Exception as e:
        print(f"[long_video] segment kesme hatasi: {e}")
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass
    return None


def _infer_segment(seg_bytes: bytes, model_id: str) -> Optional[str]:
    """Segment bytes → metin. model_id'ye gore uygun inference (blocking)."""
    try:
        if model_id == "av-fusion":
            from backend.av_fusion import infer_av_fusion
            ok, text, _ = infer_av_fusion(seg_bytes, None, decode_mode="beam_lm")
        else:  # "video" (video-only)
            from backend.inference import infer_from_video_bytes
            ok, text, _ = infer_from_video_bytes(seg_bytes, model_id=model_id, decode_mode="beam_lm")
        if ok and text and text != "(boş)":
            return text.strip()
    except Exception as e:
        print(f"[long_video] segment inference hatasi: {e}")
    return None


async def _emit(job: dict, event: dict):
    """SSE queue'ya event koy."""
    await job["queue"].put(event)


async def process_long_video(job_id: str, video_path: str, model_id: str):
    """Ana isleme task'i (asyncio). Blocking islemler to_thread ile."""
    job = _jobs[job_id]
    try:
        # Faz 1: audio cikar (tracking fazi altinda)
        job["status"] = "processing"
        job["phase"] = "tracking"
        job["progress"] = 0
        await _emit(job, {"status": "processing", "phase": "tracking", "progress": 5})

        audio_path = await asyncio.to_thread(_extract_audio, video_path)

        # Faz 2: segmentasyon — ses varsa VAD, yoksa sabit-pencere (sessiz video)
        await _emit(job, {"status": "processing", "phase": "tracking", "progress": 40})
        segments = []
        if audio_path:
            segments = await asyncio.to_thread(_vad_segments, audio_path)
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        silent = False
        if not segments:
            # Sessiz video VEYA VAD konusma bulamadi -> sabit-pencere fallback (video-only)
            silent = True
            segments = await asyncio.to_thread(_fixed_window_segments, video_path)
            await _emit(job, {"status": "processing", "phase": "tracking", "progress": 80,
                              "note": "ses yok/az -> sabit pencere segmentasyon (video-only kalite)"})

        if not segments:
            job["status"] = "done"
            job["transcript"] = []
            await _emit(job, {"status": "done", "transcript": [], "phase": "inference", "progress": 100})
            return

        await _emit(job, {"status": "processing", "phase": "tracking", "progress": 100})

        # Faz 3: segment segment inference
        job["phase"] = "inference"
        transcript = []
        n = len(segments)
        for i, (start, end) in enumerate(segments):
            if job.get("cancel"):
                job["status"] = "canceled"
                await _emit(job, {"status": "canceled"})
                return

            seg_bytes = await asyncio.to_thread(_cut_segment, video_path, start, end)
            text = None
            if seg_bytes:
                text = await asyncio.to_thread(_infer_segment, seg_bytes, model_id)

            progress = (i + 1) / n * 100
            if text:
                seg = {"start": round(start, 2), "end": round(end, 2), "text": text}
                transcript.append(seg)
                await _emit(job, {"status": "processing", "phase": "inference",
                                  "progress": progress, "partial_segments": [seg]})
            else:
                await _emit(job, {"status": "processing", "phase": "inference",
                                  "progress": progress})

        # Done
        job["status"] = "done"
        job["transcript"] = transcript
        await _emit(job, {"status": "done", "phase": "inference", "progress": 100,
                          "transcript": transcript})

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        await _emit(job, {"status": "error", "error": str(e)})
        print(f"[long_video] job {job_id} hata: {e}")
    finally:
        try:
            os.unlink(video_path)
        except OSError:
            pass


def create_job(video_bytes: bytes, model_id: str) -> str:
    """Video'yu kaydet, job olustur, isleme task'ini baslat. job_id doner."""
    job_id = str(uuid.uuid4())
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()
    _jobs[job_id] = {
        "status": "queued",
        "phase": "upload",
        "progress": 0,
        "queue": asyncio.Queue(),
        "transcript": None,
        "error": None,
        "cancel": False,
        "video_path": tmp.name,
        "model_id": model_id,
        "created": time.time(),
    }
    asyncio.create_task(process_long_video(job_id, tmp.name, model_id))
    return job_id


def get_job_status(job_id: str) -> Optional[dict]:
    job = _jobs.get(job_id)
    if not job:
        return None
    return {
        "job_id": job_id,
        "status": job["status"],
        "phase": job["phase"],
        "progress": job["progress"],
        "error": job["error"],
        "n_segments": len(job["transcript"]) if job["transcript"] else 0,
    }


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if job:
        job["cancel"] = True
        return True
    return False


async def stream_events(job_id: str):
    """SSE generator. Queue'dan event'leri yield et (data: JSON\\n\\n)."""
    job = _jobs.get(job_id)
    if not job:
        yield f"data: {json.dumps({'status': 'error', 'error': 'job bulunamadi'})}\n\n"
        return
    q = job["queue"]
    # Baslangic durumu
    yield f"data: {json.dumps({'status': job['status'], 'phase': job['phase'], 'progress': job['progress']})}\n\n"
    while True:
        try:
            ev = await asyncio.wait_for(q.get(), timeout=300)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            continue
        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        if ev.get("status") in ("done", "error", "canceled"):
            break
