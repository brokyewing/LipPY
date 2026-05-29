"""
LipPY fonem vocab — iki versiyonu destekler:

- **30 fonem** (v1, v1b): SPACE YOK. Kelime sınırı LLM'e bırakılır.
- **31 fonem** (v1b++): SPACE dahil (IDX=1). Model kelime sınırını öğrenir.

BLANK_IDX = 0 (CTC blank)
Her iki vocab için eğitim notebook'larıyla birebir aynıdır.
"""
from __future__ import annotations

# ============ 30 FONEM (v1, v1b) ============
PHONEMES_30: list[str] = [
    "SIL",   # 0: CTC blank
    "a", "e", "i", "o", "u",
    "ı", "ö", "ü",
    "b", "p", "m",
    "f", "v",
    "d", "t", "n", "l",
    "s", "z",
    "c", "ç",
    "ş", "j",
    "k", "g", "ğ",
    "r", "h", "y",
]
assert len(PHONEMES_30) == 30

# ============ 31 FONEM (v1b++, SPACE DAHIL) ============
PHONEMES_31: list[str] = [
    "SIL",     # 0: CTC blank
    "SPACE",   # 1: kelime ayıracı (YENI v1b++)
    "a", "e", "i", "o", "u",
    "ı", "ö", "ü",
    "b", "p", "m",
    "f", "v",
    "d", "t", "n", "l",
    "s", "z",
    "c", "ç",
    "ş", "j",
    "k", "g", "ğ",
    "r", "h", "y",
]
assert len(PHONEMES_31) == 31

BLANK_IDX: int = 0

# Geriye dönük uyumluluk — v1b default (30 fonem)
PHONEMES: list[str] = PHONEMES_30
NUM_PHONEMES: int = len(PHONEMES_30)
PH_TO_IDX: dict[str, int] = {p: i for i, p in enumerate(PHONEMES_30)}
IDX_TO_PH: dict[int, str] = {i: p for i, p in enumerate(PHONEMES_30)}


def get_vocab(num_phonemes: int) -> tuple[list[str], int, int | None]:
    """Vocab size'a göre fonem listesi, blank idx ve space idx döndür.

    Returns:
        (phonemes, blank_idx, space_idx)
        space_idx None ise SPACE yoktur.
    """
    if num_phonemes == 30:
        return PHONEMES_30, 0, None
    elif num_phonemes == 31:
        return PHONEMES_31, 0, 1
    else:
        raise ValueError(f"Desteklenmeyen vocab size: {num_phonemes}. Sadece 30/31.")


def indices_to_text(indices: list[int], num_phonemes: int = 30) -> str:
    """Fonem indices → text.

    - 30 fonem: SPACE yok, fonemler birleştirilir (boşluksuz).
    - 31 fonem: SPACE_IDX (1) → ' ', diğerleri birleştirilir.

    CTC blank (SIL=0) her durumda atılır.
    """
    phonemes, blank_idx, space_idx = get_vocab(num_phonemes)

    out: list[str] = []
    for i in indices:
        if i == blank_idx:
            continue
        if space_idx is not None and i == space_idx:
            out.append(" ")
        elif 0 <= i < len(phonemes):
            out.append(phonemes[i])
    return "".join(out)


def text_to_phonemes(text: str, num_phonemes: int = 30) -> list[int]:
    """Türkçe text → fonem indices (validasyon için).

    - 30 fonem: Boşluklar atılır.
    - 31 fonem: Boşluklar SPACE_IDX'e çevrilir.
    """
    phonemes, blank_idx, space_idx = get_vocab(num_phonemes)
    ph_to_idx = {p: i for i, p in enumerate(phonemes)}

    out: list[int] = []
    for ch in text.lower().strip():
        if ch == " ":
            if space_idx is not None:
                if not out or out[-1] != space_idx:
                    out.append(space_idx)
        elif ch in ph_to_idx and ph_to_idx[ch] > 0:
            out.append(ph_to_idx[ch])

    # 31 fonem: baş/son SPACE temizle
    if space_idx is not None:
        while out and out[0] == space_idx:
            out.pop(0)
        while out and out[-1] == space_idx:
            out.pop()
    return out


def get_pyctcdecode_labels(num_phonemes: int) -> list[str]:
    """pyctcdecode için label listesi üret.

    pyctcdecode format: ilk eleman '' (blank), sonrası karakterler.
    SPACE için ' ' kullanılır.
    """
    phonemes, _, space_idx = get_vocab(num_phonemes)
    labels = [""]  # blank
    for i, ph in enumerate(phonemes):
        if i == 0:  # SIL skip (zaten blank)
            continue
        if ph == "SPACE":
            labels.append(" ")
        else:
            labels.append(ph)
    return labels
