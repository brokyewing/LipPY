"""
Video → ROI → LipPhonemeModel → text. (model id: video — sadece dudak)

Model: 31 fonem CTC, Conformer (dim_ff=2048), MediaPipe ROI.

Decode modları:
- greedy: CTC greedy (en hızlı, LM yok)
- beam: pyctcdecode beam (LM yok)
- beam_lm: pyctcdecode beam + KenLM (önerilen — video-only'de WER'i belirgin düşürür)

Hafızada çalışır, kalıcı dosya yazmaz.
Normalize: (x/255 - 0.495) / 0.200. 88x88 center crop (eğitim ile aynı).
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from backend.models import LipPhonemeModel
from backend.vocab import BLANK_IDX, indices_to_text, get_pyctcdecode_labels
from backend.roi import video_to_roi_array
from backend.decode_utils import greedy_ctc_decode

ROOT = Path(__file__).resolve().parent.parent


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eğitim ile aynı normalize sabitleri (video modeli = v1cc, 500 ROI örnekleminden)
# DIKKAT: eski v1c 0.421/0.165 idi — video modeli 0.495/0.200 ile eğitildi.
MEAN = 0.495
STD = 0.200
CROP = 88

# Model registry — tek video-only model: "video" (eski v1cc, dim_ff=2048, 31 fonem CTC)
# Eski v1, v1b, v1b_plus, v1c kaldırıldı. KenLM ortak checkpoints/kenlm/'de.
MODEL_REGISTRY: dict[str, dict] = {
    "video": {
        "name": "LipPY — Sadece Dudak (Video)",
        "vocab_size": 31,
        "dim_ff": 2048,
        "checkpoint_candidates": [
            "checkpoints/video/best_model.pt",
        ],
        "kenlm_candidates": [
            # Ortak domain KenLM (konuşma dili, 31 fonem)
            "checkpoints/kenlm/lm.klm",
            "checkpoints/video/lm.klm",
        ],
        "kenlm_alpha": 0.5,
        "kenlm_beta": 0.5,
        "default_decode": "beam_lm",
        "decode_modes": ["greedy", "beam_lm"],
        "beam_width": 100,
    },
}

# Yüklü model cache
_model_cache: dict[str, LipPhonemeModel] = {}
# KenLM beam decoder cache (model_id → decoder)
_decoder_cache: dict[str, object] = {}


def _find_file(candidates: list[str]) -> Optional[str]:
    """Aday yollardan ilk mevcut olanı döndür."""
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


def _load_model(model_id: str) -> tuple[Optional[LipPhonemeModel], Optional[str]]:
    """Modeli cache'den veya disk'ten yükler."""
    if model_id in _model_cache:
        return _model_cache[model_id], None

    entry = MODEL_REGISTRY.get(model_id)
    if not entry:
        return None, f"Bilinmeyen model: {model_id}"

    ckpt_path = _find_checkpoint(model_id)
    if not ckpt_path:
        tried = ", ".join(entry["checkpoint_candidates"])
        return None, f"Checkpoint bulunamadı. Aranılan: {tried}"

    model = LipPhonemeModel(
        num_phonemes=entry["vocab_size"],
        dim_ff=entry.get("dim_ff", 1024),   # v1c=1024, v1cc=2048
    )
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys:
            problematic = [k for k in result.missing_keys if not k.startswith("ctc_fc")]
            if problematic:
                return None, f"State dict eksik anahtarlar: {problematic[:3]}"
    except Exception as e:
        return None, f"Checkpoint yüklenemedi: {e}"

    model.to(DEVICE).eval()
    _model_cache[model_id] = model
    return model, None


def _get_beam_decoder(model_id: str, with_lm: bool = True) -> tuple[Optional[object], Optional[str]]:
    """pyctcdecode beam decoder üret (cache'li).

    with_lm=True: KenLM ile (eğer lm.klm varsa)
    with_lm=False: LM'siz beam
    """
    cache_key = f"{model_id}__{'lm' if with_lm else 'nolm'}"
    if cache_key in _decoder_cache:
        return _decoder_cache[cache_key], None

    entry = MODEL_REGISTRY.get(model_id)
    if not entry:
        return None, f"Bilinmeyen model: {model_id}"

    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        return None, "pyctcdecode kurulu değil (pip install pyctcdecode)"

    labels = get_pyctcdecode_labels(entry["vocab_size"])

    if with_lm:
        lm_path = _find_kenlm(model_id)
        if not lm_path:
            return None, f"KenLM bulunamadı ({model_id})"

        # Unigrams: 3 yol (decode_v1c notebook ile aynı mantık)
        # 1. unigrams*.txt direkt
        # 2. corpus.txt'den türet (filter: len>=2, freq>=3) — plan dosyası standardı
        # 3. Hiçbiri yoksa unigrams=None (KenLM-only mode, kalite düşük!)
        unigrams = None
        lm_dir = Path(lm_path).parent

        # Yol 1: unigrams.txt direkt
        for ug_name in ["unigrams.txt", "unigrams_filtered.txt"]:
            ug_path = lm_dir / ug_name
            if ug_path.is_file():
                try:
                    with open(ug_path, "r", encoding="utf-8") as f:
                        unigrams = [l.strip() for l in f if l.strip()]
                    print(f"[Decoder] Unigrams (direkt): {ug_path} ({len(unigrams)})")
                    break
                except Exception:
                    pass

        # Yol 2: corpus.txt'den türet (cache: aynı path için bir kere)
        if unigrams is None:
            corpus_candidates = [
                lm_dir / "corpus.txt",
                ROOT / "checkpoints" / "v1c" / "corpus.txt",
                ROOT / "checkpoints" / "v1b-plus" / "corpus.txt",
                ROOT / "checkpoints" / "kenlm" / "corpus.txt",
            ]
            for cp in corpus_candidates:
                if cp.is_file():
                    try:
                        from collections import Counter
                        print(f"[Decoder] Corpus.txt'den unigrams türetiliyor: {cp} ...")
                        word_counter: Counter = Counter()
                        with open(cp, "r", encoding="utf-8") as f:
                            for line in f:
                                for w in line.strip().lower().split():
                                    if len(w) >= 2:
                                        word_counter[w] += 1
                        unigrams = [w for w, c in word_counter.items() if c >= 3]
                        print(f"[Decoder] Unigrams türetildi: {len(unigrams)} kelime "
                              f"(toplam {len(word_counter)}, filter: len>=2 freq>=3)")
                        # Cache: bir sonraki request'te tekrar oluşturma
                        ug_cache_path = lm_dir / "unigrams.txt"
                        try:
                            with open(ug_cache_path, "w", encoding="utf-8") as fw:
                                fw.write("\n".join(unigrams))
                            print(f"[Decoder] Cache: {ug_cache_path}")
                        except Exception:
                            pass
                        break
                    except Exception as e:
                        print(f"[Decoder] Corpus oku HATA: {e}")

        # Yol 3 fallback: vocab.txt (label vocab, unigram değil — atla)
        if unigrams is None:
            vocab_path = lm_dir / "vocab.txt"
            if vocab_path.is_file():
                try:
                    with open(vocab_path, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    if all(len(l) <= 3 for l in lines[:50]):
                        # Label vocab (fonem karakterleri) — gerçek unigram değil
                        print(f"[Decoder] vocab.txt label vocab içeriyor (unigram değil), atlandı")
                    else:
                        unigrams = lines
                        print(f"[Decoder] Unigrams (vocab.txt'den, fallback): {len(unigrams)}")
                except Exception:
                    pass

        if unigrams is None:
            print("[Decoder] !!! UYARI: Unigrams bulunamadi, KenLM kalitesi DUSUK olacak")
            print("[Decoder]     Cozum: corpus.txt'yi checkpoints/v1c/ icine koy")

        # Alpha/beta: env variable override (LIPPY_KENLM_ALPHA, LIPPY_KENLM_BETA)
        # Yeni speaker'da düşük alpha (0.2-0.3) daha iyi sonuç verebilir
        alpha = float(os.environ.get("LIPPY_KENLM_ALPHA", entry.get("kenlm_alpha", 0.5)))
        beta = float(os.environ.get("LIPPY_KENLM_BETA", entry.get("kenlm_beta", 0.5)))
        print(f"[Decoder] KenLM hyperparametreleri: alpha={alpha}, beta={beta}")
        try:
            decoder = build_ctcdecoder(
                labels,
                kenlm_model_path=lm_path,
                alpha=alpha,
                beta=beta,
                unigrams=unigrams,
            )
        except Exception as e:
            return None, f"KenLM decoder oluşturulamadı: {e}"
    else:
        try:
            decoder = build_ctcdecoder(labels)
        except Exception as e:
            return None, f"Beam decoder oluşturulamadı: {e}"

    _decoder_cache[cache_key] = decoder
    return decoder, None


def _normalize_roi(roi: np.ndarray) -> torch.Tensor:
    """ROI (T, H, W) uint8 → tensor (T, CROP, CROP) normalize."""
    x = roi.astype(np.float32) / 255.0
    T, H, W = x.shape
    top = (H - CROP) // 2
    left = (W - CROP) // 2
    x = x[:, top : top + CROP, left : left + CROP]
    x = (x - MEAN) / STD
    return torch.from_numpy(x.copy()).float()


def list_available_models() -> list[dict]:
    """Checkpoint'i disk'te bulunan modellerin listesi."""
    out: list[dict] = []
    for mid, entry in MODEL_REGISTRY.items():
        ckpt = _find_checkpoint(mid)
        kenlm = _find_kenlm(mid) if "kenlm_candidates" in entry else None
        out.append({
            "id": mid,
            "name": entry["name"],
            "vocab_size": entry["vocab_size"],
            "available": ckpt is not None,
            "checkpoint_path": ckpt,
            "kenlm_path": kenlm,
            "default_decode": entry.get("default_decode", "greedy"),
            "decode_modes": entry.get("decode_modes"),
        })
    return out


def infer_from_video_bytes(
    video_bytes: bytes,
    model_id: str = "v1c",
    decode_mode: str = "auto",
) -> tuple[bool, Optional[str], dict]:
    """Video bytes → metin tahmini.

    Args:
        video_bytes: .mp4 / .webm / vb. video dosyası ham bayt
        model_id: MODEL_REGISTRY içinde bir anahtar (varsayılan v1c — en iyi WER %78.7)
        decode_mode: 'auto' | 'greedy' | 'beam' | 'beam_lm'
                    'auto' ise registry'deki default_decode kullanılır.

    Returns:
        (success, text, details)
    """
    t0 = time.perf_counter()
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

    entry = MODEL_REGISTRY.get(model_id)
    if not entry:
        details["error"] = f"Bilinmeyen model: {model_id}"
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details

    # decode mode auto → registry default
    # LIPPY_DEFAULT_DECODE env var registry default'unu override eder
    # Örn: LIPPY_DEFAULT_DECODE=beam → KenLM atla, ham CTC beam search
    if decode_mode == "auto":
        decode_mode = os.environ.get(
            "LIPPY_DEFAULT_DECODE",
            entry.get("default_decode", "greedy")
        )
        details["decode_mode"] = decode_mode

    # KenLM istendi ama dosya yok → otomatik 'beam' (no LM) fallback
    if decode_mode == "beam_lm" and "kenlm_candidates" in entry:
        if not _find_kenlm(model_id):
            details["decode_fallback"] = "KenLM dosyası yok, beam (no LM) kullanılıyor"
            decode_mode = "beam"
            details["decode_mode"] = decode_mode

    # v1b++ için SPACE dahil vocab
    num_phonemes = entry["vocab_size"]

    model, err = _load_model(model_id)
    if model is None:
        details["error"] = err
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details

    # Videoyu geçici dosyaya yaz, ROI çıkar
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
        details["error"] = "Video işlenemedi (yüz bulunamadı veya affine align başarısız)"
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details

    T, H, W = roi.shape
    details["frames"] = int(T)
    details["roi_shape"] = [int(T), int(H), int(W)]

    # Normalize + model forward
    t_decode_start = time.perf_counter()
    roi_tensor = _normalize_roi(roi).unsqueeze(0).to(DEVICE)
    input_lengths = torch.tensor([roi_tensor.shape[1]], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        log_probs = model(roi_tensor, input_lengths)  # (T_out, 1, V)

    # Decode
    if decode_mode == "greedy":
        pred_indices = greedy_ctc_decode(log_probs[:, 0, :])
        text = indices_to_text(pred_indices, num_phonemes=num_phonemes).strip() or "(boş)"
    elif decode_mode in ("beam", "beam_lm"):
        with_lm = (decode_mode == "beam_lm")
        decoder, derr = _get_beam_decoder(model_id, with_lm=with_lm)
        if decoder is None:
            # Fallback greedy
            details["decode_fallback"] = f"beam decoder alınamadı ({derr}), greedy fallback"
            pred_indices = greedy_ctc_decode(log_probs[:, 0, :])
            text = indices_to_text(pred_indices, num_phonemes=num_phonemes).strip() or "(boş)"
        else:
            lp_np = log_probs[:, 0, :].float().cpu().numpy()
            beam_width = entry.get("beam_width", 100)
            text = decoder.decode(lp_np, beam_width=beam_width).strip() or "(boş)"
    else:
        details["error"] = f"Bilinmeyen decode_mode: {decode_mode}"
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details

    details["decode_ms"] = int((time.perf_counter() - t_decode_start) * 1000)
    details["total_ms"] = int((time.perf_counter() - t0) * 1000)
    return True, text, details
