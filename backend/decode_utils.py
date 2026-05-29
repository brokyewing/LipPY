"""
Ortak CTC decode yardımcıları — inference.py (video) ve av_fusion.py paylaşır.
"""
from __future__ import annotations

import torch

from backend.vocab import BLANK_IDX


def greedy_ctc_decode(log_probs: torch.Tensor) -> list[int]:
    """CTC greedy decode. log_probs: (T, V) → fonem id listesi.

    Ardışık tekrarları ve BLANK'ı atar (standart CTC collapse).
    """
    ids = log_probs.argmax(dim=-1).tolist()
    decoded: list[int] = []
    prev = -1
    for i in ids:
        if i != prev and i != BLANK_IDX:
            decoded.append(i)
        prev = i
    return decoded
