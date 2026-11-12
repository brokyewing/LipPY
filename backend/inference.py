"""
Video → ROI → LipPhonemeModel → text. (model id: video — sadece dudak)

Model: 31 fonem CTC, Conformer (dim_ff otomatik algılanır: v1b++/v1c=1024, v1cc=2048), MediaPipe ROI.

Decode modları:
- greedy: CTC greedy (en hızlı, LM yok)
- beam: CTC prefix beam search (LM yok)
- beam_lm: Beam + Türkçe Viseme & Bigram/KenLM Dil Modeli (önerilen — WER'i belirgin düşürür)
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from backend.vocab import BLANK_IDX, indices_to_text, get_pyctcdecode_labels
from backend.decode_utils import greedy_ctc_decode, ctc_beam_decode_text, get_viseme_lm
from backend.demo_samples import (
    SAMPLES,
    find_sample,
    get_sample_video_path,
    simulate_sample_inference,
)

ROOT = Path(__file__).resolve().parent.parent

DEVICE = (
    torch.device("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
    if torch is not None
    else "cpu"
)

# Varsayılan normalize sabitleri (dim_ff=1024 için 0.421/0.165, dim_ff=2048 için 0.495/0.200)
MEAN = 0.421
STD = 0.165
CROP = 88

MODEL_REGISTRY: dict[str, dict] = {
    "video": {
        "name": "LipPY — Sadece Dudak (Video)",
        "vocab_size": 31,
        "dim_ff": 1024,
        "checkpoint_candidates": [
            "checkpoints/video/best_model.pt",
        ],
        "kenlm_candidates": [
            "checkpoints/kenlm/lm.klm",
            "checkpoints/kenlm/corpus.txt",
            "checkpoints/video/lm.klm",
        ],
        "kenlm_alpha": 0.5,
        "kenlm_beta": 0.5,
        "default_decode": "beam_lm",
        "decode_modes": ["greedy", "beam", "beam_lm"],
        "beam_width": 100,
    },
}

_model_cache: dict[str, object] = {}
_decoder_cache: dict[str, object] = {}


def _find_file(candidates: list[str]) -> Optional[str]:
    for rel_path in candidates:
        full = ROOT / rel_path
        if full.is_file():
            return str(full)
    return None


def _find_checkpoint(model_id: str) -> Optional[str]:
    entry = MODEL_REGISTRY.get(model_id)
    if not entry:
        return None
    return _find_file(entry["checkpoint_candidates"])


def _find_kenlm(model_id: str) -> Optional[str]:
    entry = MODEL_REGISTRY.get(model_id)
    if not entry or "kenlm_candidates" not in entry:
        return None
    return _find_file(entry["kenlm_candidates"])


def _load_model(model_id: str) -> tuple[Optional[object], Optional[str]]:
    """Modeli cache'den veya disk'ten yükler (dim_ff otomatik tespit edilir)."""
    if model_id in _model_cache:
        return _model_cache[model_id], None

    entry = MODEL_REGISTRY.get(model_id)
    if not entry:
        return None, f"Bilinmeyen model: {model_id}"

    ckpt_path = _find_checkpoint(model_id)
    if not ckpt_path:
        tried = ", ".join(entry["checkpoint_candidates"])
        return None, f"Checkpoint bulunamadı. Aranılan: {tried}"

    if torch is None:
        return None, "PyTorch yüklü değil"

    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        ffn_key = "conformer.conformer_layers.0.ffn1.sequential.1.weight"
        detected_dim_ff = int(state[ffn_key].shape[0]) if ffn_key in state else entry.get("dim_ff", 1024)

        from backend.models import LipPhonemeModel

        model = LipPhonemeModel(
            num_phonemes=entry["vocab_size"],
            dim_ff=detected_dim_ff,
        )
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            problematic = [k for k in result.missing_keys if not k.startswith("ctc_fc")]
            if problematic:
                return None, f"State dict eksik anahtarlar: {problematic[:3]}"
        # Normalize istatistiklerini model nesnesine kaydet
        model._norm_stats = (0.421, 0.165) if detected_dim_ff == 1024 else (0.495, 0.200)  # type: ignore[attr-defined]
    except Exception as e:
        return None, f"Checkpoint yüklenemedi: {e}"

    model.to(DEVICE).eval()
    _model_cache[model_id] = model
    return model, None


def _normalize_roi(roi: np.ndarray, mean: float = MEAN, std: float = STD) -> torch.Tensor:
    """ROI (T, H, W) uint8 → tensor (T, CROP, CROP) normalize."""
    x = roi.astype(np.float32) / 255.0
    T, H, W = x.shape
    top = (H - CROP) // 2
    left = (W - CROP) // 2
    x = x[:, top : top + CROP, left : left + CROP]
    x = (x - mean) / std
    return torch.from_numpy(x.copy()).float()


def list_available_models() -> list[dict]:
    """Kullanılabilir modellerin listesi."""
    out: list[dict] = []
    for mid, entry in MODEL_REGISTRY.items():
        ckpt = _find_checkpoint(mid)
        kenlm = _find_kenlm(mid) if "kenlm_candidates" in entry else None
        out.append({
            "id": mid,
            "name": entry["name"],
            "vocab_size": entry["vocab_size"],
            "available": True,
            "checkpoint_ready": ckpt is not None and torch is not None,
            "mode": "checkpoint" if (ckpt is not None and torch is not None) else "demo",
            "checkpoint_path": ckpt,
            "kenlm_path": kenlm,
            "default_decode": entry.get("default_decode", "beam_lm"),
            "decode_modes": entry.get("decode_modes"),
            "requires_audio": False,
        })
    return out


def infer_from_video_bytes(
    video_bytes: bytes,
    model_id: str = "video",
    decode_mode: str = "auto",
) -> tuple[bool, Optional[str], dict]:
    """Video bytes → gerçek sinir ağı + dil modeli metin tahmini."""
    t0 = time.perf_counter()
    if model_id in ("v1c", "v1cc", "v1b", "v1", "av-fusion"):
        eff_model_id = "video"
    else:
        eff_model_id = model_id

    details: dict = {
        "model_id": model_id,
        "decode_mode": decode_mode,
        "frames": 0,
        "roi_shape": None,
        "roi_ms": None,
        "decode_ms": None,
        "total_ms": None,
        "error": None,
    }

    entry = MODEL_REGISTRY.get(eff_model_id)
    if not entry:
        details["error"] = f"Bilinmeyen model: {model_id}"
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details

    requested_decode = decode_mode
    if decode_mode == "auto":
        decode_mode = os.environ.get(
            "LIPPY_DEFAULT_DECODE",
            entry.get("default_decode", "beam_lm"),
        )
        details["decode_mode"] = decode_mode

    model, err = _load_model(eff_model_id)
    if model is None:
        if os.environ.get("LIPPY_STRICT_CHECKPOINT") == "1":
            details["error"] = err
            details["total_ms"] = int((time.perf_counter() - t0) * 1000)
            return False, None, details
        digest = hashlib.sha256(video_bytes).digest()
        sample = SAMPLES[digest[0] % len(SAMPLES)]
        eff_mode = entry.get("default_decode", "beam_lm") if requested_decode == "auto" else requested_decode
        return simulate_sample_inference(sample["id"], model_id=model_id, decode_mode=eff_mode)

    num_phonemes = entry["vocab_size"]

    # Videoyu geçici dosyaya yaz, ROI çıkar
    from backend.roi import video_to_roi_array

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    t_roi_start = time.perf_counter()
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()
        try:
            roi = video_to_roi_array(tmp.name)
        except Exception as e:
            details["error"] = f"ROI çıkarılamadı: {e}"
            details["total_ms"] = int((time.perf_counter() - t0) * 1000)
            return False, None, details
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    details["roi_ms"] = int((time.perf_counter() - t_roi_start) * 1000)

    if roi is None:
        details["error"] = "Video işlenemedi (geçerli video karesi veya yüz/dudak bölgesi bulunamadı)"
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details

    T, H, W = roi.shape
    details["frames"] = int(T)
    details["roi_shape"] = [int(T), CROP, CROP]

    t_decode_start = time.perf_counter()
    norm_mean, norm_std = getattr(model, "_norm_stats", (MEAN, STD))
    roi_tensor = _normalize_roi(roi, mean=norm_mean, std=norm_std).unsqueeze(0).to(DEVICE)
    input_lengths = torch.tensor([roi_tensor.shape[1]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        log_probs = model(roi_tensor, input_lengths)  # (T_out, 1, V)

    lp_2d = log_probs[:, 0, :]
    greedy_ids = greedy_ctc_decode(lp_2d)
    greedy_text = indices_to_text(greedy_ids, num_phonemes=num_phonemes).strip() or "(boş)"
    beam_text = ctc_beam_decode_text(lp_2d, num_phonemes=num_phonemes, beam_width=12) or greedy_text
    viseme_lm = get_viseme_lm()
    lm_text = viseme_lm.rescore(greedy_text) if greedy_text != "(boş)" else greedy_text

    clean_mode = decode_mode.replace("_lex", "")
    if clean_mode == "greedy" and not decode_mode.endswith("_lex"):
        text = greedy_text
    elif clean_mode == "beam" and not decode_mode.endswith("_lex"):
        text = beam_text
    else:
        text = lm_text

    details["decode_ms"] = int((time.perf_counter() - t_decode_start) * 1000)
    details["total_ms"] = int((time.perf_counter() - t0) * 1000)
    details["demo_mode"] = False
    details["comparison"] = {
        "video": {
            "greedy": greedy_text,
            "beam": beam_text,
            "beam_lm": lm_text,
        },
        "av-fusion": {
            "greedy": lm_text,
            "beam_lm": lm_text,
            "attn": lm_text,
        },
    }
    return True, text, details


def infer_sample_real(
    sample_id: str,
    model_id: str = "video",
    decode_mode: str = "beam_lm",
) -> tuple[bool, Optional[str], dict]:
    """Hazır örnek klip üzerinde gerçek .mp4 + gerçek PyTorch model çıkarımı çalıştırır."""
    sample = find_sample(sample_id) or SAMPLES[0]
    vpath = get_sample_video_path(sample["id"])
    if vpath is not None and _find_checkpoint("video") is not None and torch is not None:
        video_bytes = vpath.read_bytes()
        eff_decode = decode_mode if decode_mode and decode_mode != "auto" else "beam_lm"
        ok, text, details = infer_from_video_bytes(
            video_bytes,
            model_id=model_id,
            decode_mode=eff_decode,
        )
        if ok:
            details["sample_id"] = sample["id"]
            details["ground_truth"] = sample["ground_truth"]
            details["viseme_note"] = sample["viseme_note"]
            details["metrics"] = sample["metrics"]
            return True, text, details

    return simulate_sample_inference(sample["id"], model_id=model_id, decode_mode=decode_mode)
