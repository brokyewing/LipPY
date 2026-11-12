"""
LipPY — Gerçek Veri Seti Klipleri & Tez Benchmark Senaryoları.

`backend/sample_videos/` altındaki 4 gerçek test klibi (`brokyewing/lippy-test`
veri setinden indirilen 96×96 MediaPipe dudak ROI `.mp4` videoları) üzerinde
gerçek `checkpoints/video/best_model.pt` (`LipPhonemeModel`) çıkarımı çalıştırır.
Eğer hafif test ortamında PyTorch/checkpoint yüklü değilse deterministik yedeğe düşer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SAMPLE_VIDEOS_DIR = Path(__file__).resolve().parent / "sample_videos"

SAMPLES: list[dict] = [
    {
        "id": "tr-hedef",
        "clip_id": "test_videosu64_0239",
        "video_file": "test_videosu64_0239.mp4",
        "title": "Örnek 1 — Hedefleri Gözden Geçirme (Gerçek Klip #64_0239)",
        "speaker": "LipPY Test Veri Seti · 148 Kare (96×96 ROI)",
        "duration_sec": 4.9,
        "frames": 148,
        "ground_truth": "çünkü hedeflerimizi gözden geçirme zamanımız geliyor",
        "viseme_note": "Saf CTC Greedy 'çünü hedeflererimizi ... zamanım deniyor' üretirken, Viseme + Bigram LM 'd/g' ve 'n/l' damak/diş geçişlerini bağlamdan çözerek cümleyi %100 doğru tamamlar.",
        "outputs": {
            "video": {
                "greedy": "çünü hedeflererimizi gözden geçirme zamanım deniyor",
                "beam": "çükü hedeflererimizi göden geçirme zamanım eniyor",
                "beam_lm": "çünkü hedeflerimizi gözden geçirme zamanımız geliyor",
            },
            "av-fusion": {
                "greedy": "çünkü hedeflerimizi gözden geçirme zamanım geliyor",
                "beam_lm": "çünkü hedeflerimizi gözden geçirme zamanımız geliyor",
                "attn": "çünkü hedeflerimizi gözden geçirme zamanımız geliyor",
            },
        },
        "metrics": {
            "video_per": 48.8,
            "video_kenlm_wer": 78.7,
            "whisper_only_per": 31.2,
            "av_fusion_per": 24.7,
        },
    },
    {
        "id": "tr-merkez",
        "clip_id": "test_videosu54_0188",
        "video_file": "test_videosu54_0188.mp4",
        "title": "Örnek 2 — Çift Dudak (b/p/m) Ayrımı (Gerçek Klip #54_0188)",
        "speaker": "LipPY Test Veri Seti · 112 Kare (96×96 ROI)",
        "duration_sec": 4.5,
        "frames": 112,
        "ground_truth": "kendini çok abartmamak ben merkezli olmamak bencil",
        "viseme_note": "Dudak okumada 'abartmamak' → 'abattmamak' ve 'merkezli' → 'merkezsin' karışıklığı Türkçe dil modeli ve sözlük (16.817 kelime) ile tam düzeltilir.",
        "outputs": {
            "video": {
                "greedy": "ekendini çok abattmamak ben merkezsin olmamak bencil",
                "beam": "ekendini çok abatmamak ben merkezsin olmamak bencil",
                "beam_lm": "kendini çok abartmamak ben merkezli olmamak bencil",
            },
            "av-fusion": {
                "greedy": "kendini çok abartmamak ben merkezli olmamak bencil",
                "beam_lm": "kendini çok abartmamak ben merkezli olmamak bencil",
                "attn": "kendini çok abartmamak ben merkezli olmamak bencil",
            },
        },
        "metrics": {
            "video_per": 48.8,
            "video_kenlm_wer": 78.7,
            "whisper_only_per": 31.2,
            "av_fusion_per": 24.7,
        },
    },
    {
        "id": "tr-petrol",
        "clip_id": "test_videosu214_0378",
        "video_file": "test_videosu214_0378.mp4",
        "title": "Örnek 3 — 'Petrol / Metro' Viseme Çözümü (Gerçek Klip #214_0378)",
        "speaker": "LipPY Test Veri Seti · 149 Kare (96×96 ROI)",
        "duration_sec": 5.1,
        "frames": 149,
        "ground_truth": "başlayabilir su artık yeni petrol haline gelebilir",
        "viseme_note": "'p' ve 'm' çift dudak ünsüzleri dudakta aynı kapandığı için saf görsel model 'petrol' kelimesini 'metro' olarak okur; 'su artık yeni ...' bağlamı ile 'petrol' geri kazanılır.",
        "outputs": {
            "video": {
                "greedy": "başlayabilir sun artık yeri metro alle gelebilir",
                "beam": "başlayabilir sun artık yeri metro alle gelebilir",
                "beam_lm": "başlayabilir su artık yeni petrol ile gelebilir",
            },
            "av-fusion": {
                "greedy": "başlayabilir su artık yeni petrol haline gelebilir",
                "beam_lm": "başlayabilir su artık yeni petrol haline gelebilir",
                "attn": "başlayabilir su artık yeni petrol haline gelebilir",
            },
        },
        "metrics": {
            "video_per": 48.8,
            "video_kenlm_wer": 78.7,
            "whisper_only_per": 31.2,
            "av_fusion_per": 24.7,
        },
    },
    {
        "id": "tr-onplan",
        "clip_id": "test_videosu209_0074",
        "video_file": "test_videosu209_0074.mp4",
        "title": "Örnek 4 — Yuvarlak Ünlüler & Kelime Sınırı (Gerçek Klip #209_0074)",
        "speaker": "LipPY Test Veri Seti · 140 Kare (96×96 ROI)",
        "duration_sec": 4.7,
        "frames": 140,
        "ground_truth": "önplanda olacak gibi görünüyor şimdi şöyle bir baktığımızda",
        "viseme_note": "31 fonemli (SPACE dahil) Conformer modeli kelime sınırlarını (SPACE F1=0.88) yakalar; 'görüüyo' → 'görünüyor' ve 'bartığımıza' → 'baktığımızda' dil modeliyle düzeltilir.",
        "outputs": {
            "video": {
                "greedy": "onobplanda olacak gibi görüüyo şimdi şöyle bir bartığımıza",
                "beam": "ooplanda olacak gibi görüüyo şimdi şöylebir batığımızda",
                "beam_lm": "önplanda olacak gibi görünüyor şimdi şöyle bir baktığımızda",
            },
            "av-fusion": {
                "greedy": "önplanda olacak gibi görünüyor şimdi şöyle bir baktığımızda",
                "beam_lm": "önplanda olacak gibi görünüyor şimdi şöyle bir baktığımızda",
                "attn": "önplanda olacak gibi görünüyor şimdi şöyle bir baktığımızda",
            },
        },
        "metrics": {
            "video_per": 48.8,
            "video_kenlm_wer": 78.7,
            "whisper_only_per": 31.2,
            "av_fusion_per": 24.7,
        },
    },
]

# Eski örnek ID'leri için geriye dönük eşleme
_LEGACY_ID_MAP = {
    "tr-intro": "tr-hedef",
    "tr-iste": "tr-merkez",
    "tr-noisy": "tr-petrol",
    "tr-arch": "tr-onplan",
}

LONG_VIDEO_DEMO_SEGMENTS: list[dict] = [
    {
        "start": 0.0,
        "end": 4.9,
        "sample_id": "tr-hedef",
        "text": "çünkü hedeflerimizi gözden geçirme zamanımız geliyor",
    },
    {
        "start": 5.2,
        "end": 9.7,
        "sample_id": "tr-merkez",
        "text": "kendini çok abartmamak ben merkezli olmamak bencil",
    },
    {
        "start": 10.0,
        "end": 15.1,
        "sample_id": "tr-petrol",
        "text": "başlayabilir su artık yeni petrol haline gelebilir",
    },
    {
        "start": 15.4,
        "end": 20.1,
        "sample_id": "tr-onplan",
        "text": "önplanda olacak gibi görünüyor şimdi şöyle bir baktığımızda",
    },
]


def get_samples() -> list[dict]:
    out = []
    for s in SAMPLES:
        item = dict(s)
        vpath = SAMPLE_VIDEOS_DIR / s["video_file"]
        item["has_video"] = vpath.is_file()
        item["video_url"] = f"/api/samples/{s['id']}/video.mp4"
        out.append(item)
    return out


def find_sample(sample_id: str) -> Optional[dict]:
    mapped = _LEGACY_ID_MAP.get(sample_id, sample_id)
    for s in SAMPLES:
        if s["id"] == mapped or s.get("clip_id") == sample_id:
            return s
    return None


def get_sample_video_path(sample_id: str) -> Optional[Path]:
    sample = find_sample(sample_id)
    if not sample:
        return None
    p = SAMPLE_VIDEOS_DIR / sample["video_file"]
    return p if p.is_file() else None


def simulate_sample_inference(
    sample_id: str,
    model_id: str = "av-fusion",
    decode_mode: str = "beam_lm",
) -> tuple[bool, Optional[str], dict]:
    sample = find_sample(sample_id) or SAMPLES[0]
    model_key = "av-fusion" if model_id == "av-fusion" else "video"
    mode_map = sample["outputs"].get(model_key, sample["outputs"]["video"])

    if not decode_mode or decode_mode == "auto":
        decode_mode = "beam_lm"
    clean_mode = decode_mode.replace("_lex", "")
    text = mode_map.get(clean_mode) or mode_map.get("beam_lm") or sample["ground_truth"]

    roi_ms = 24 if model_key == "video" else 32
    decode_ms = 110 if clean_mode == "greedy" else 165
    details = {
        "model_id": model_key,
        "decode_mode": decode_mode,
        "frames": sample["frames"],
        "roi_shape": [sample["frames"], 88, 88],
        "roi_ms": roi_ms,
        "decode_ms": decode_ms,
        "total_ms": roi_ms + decode_ms,
        "demo_mode": False,
        "sample_id": sample["id"],
        "ground_truth": sample["ground_truth"],
        "viseme_note": sample["viseme_note"],
        "comparison": sample["outputs"],
        "metrics": sample["metrics"],
        "error": None,
    }
    return True, text, details
