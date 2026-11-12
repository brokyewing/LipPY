"""
Ortak CTC decode ve Türkçe Viseme + N-Gram Dil Modeli (LM) yardımcıları.
inference.py (video) ve av_fusion.py paylaşır.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from backend.vocab import BLANK_IDX, indices_to_text

ROOT = Path(__file__).resolve().parent.parent

# Türkçe dudak okuma (viseme) eşdeğerlik grupları:
# Aynı gruptaki harfler dudakta neredeyse aynı görünür (düşük ikame maliyeti)
_VISEME_GROUPS = [
    set("bpm"),          # çift dudak (bilabial)
    set("fv"),           # diş-dudak (labiodental)
    set("dtnlr"),        # diş/damak (alveolar)
    set("szşcçj"),       # ıslıklı (sibilant)
    set("kgğhy"),        # art damak / gırtlak (velar/glottal)
    set("oöuü"),         # yuvarlak ünlüler
    set("aeıi"),         # düz ünlüler
]

_CHAR_GROUP: dict[str, int] = {}
for _gid, _grp in enumerate(_VISEME_GROUPS):
    for _ch in _grp:
        _CHAR_GROUP[_ch] = _gid


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    ga = _CHAR_GROUP.get(a, -1)
    gb = _CHAR_GROUP.get(b, -2)
    if ga == gb:
        return 0.45  # Aynı viseme grubu (ör. b↔p↔m, d↔t↔n↔l)
    return 1.0


def viseme_edit_distance(s1: str, s2: str, max_dist: float = 3.0) -> float:
    """Viseme ağırlıklı Levenshtein mesafesi (erken çıkışlı)."""
    n, m = len(s1), len(s2)
    if abs(n - m) > max_dist:
        return max_dist + 1.0
    dp = [float(j) for j in range(m + 1)]
    for i in range(1, n + 1):
        c1 = s1[i - 1]
        new_dp = [float(i)] + [0.0] * m
        row_min = new_dp[0]
        for j in range(1, m + 1):
            cost = _sub_cost(c1, s2[j - 1])
            val = min(
                dp[j] + 0.85,          # silme
                new_dp[j - 1] + 0.85,  # ekleme
                dp[j - 1] + cost,      # ikame
            )
            new_dp[j] = val
            if val < row_min:
                row_min = val
        if row_min > max_dist:
            return max_dist + 1.0
        dp = new_dp
    return dp[m]


def greedy_ctc_decode(log_probs) -> list[int]:
    """CTC greedy decode. log_probs: (T, V) → fonem id listesi."""
    ids = log_probs.argmax(dim=-1).tolist()
    decoded: list[int] = []
    prev = -1
    for i in ids:
        if i != prev and i != BLANK_IDX:
            decoded.append(i)
        prev = i
    return decoded


def ctc_beam_decode_text(log_probs, num_phonemes: int = 31, beam_width: int = 12) -> str:
    """Saf Python/PyTorch CTC prefix beam search decode.
    log_probs: (T, V) log-olasılık tensörü.
    """
    if torch is None:
        return ""
    lp = log_probs.detach().cpu()
    T, V = lp.shape
    # Her adımda sadece en yüksek 6 fonemi genişlet (hız için)
    top_vals, top_ids = torch.topk(lp, k=min(6, V), dim=-1)
    top_vals_list = top_vals.tolist()
    top_ids_list = top_ids.tolist()

    # beam: {prefix_tuple: (p_blank, p_non_blank)} log-uzayında
    NEG_INF = -1e9
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}

    def _log_add(a: float, b: float) -> float:
        if a <= NEG_INF:
            return b
        if b <= NEG_INF:
            return a
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))

    for t in range(T):
        next_beams: dict[tuple[int, ...], tuple[float, float]] = defaultdict(lambda: (NEG_INF, NEG_INF))
        vals_t = top_vals_list[t]
        ids_t = top_ids_list[t]

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)
            for c, log_p in zip(ids_t, vals_t):
                if c == BLANK_IDX:
                    nb_b, nb_nb = next_beams[prefix]
                    next_beams[prefix] = (_log_add(nb_b, p_total + log_p), nb_nb)
                else:
                    end_t = prefix[-1] if prefix else -1
                    new_prefix = prefix + (c,)
                    if c == end_t:
                        # Aynı karakter tekrarı: blank üzerinden geçmeli
                        nb_b, nb_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (nb_b, _log_add(nb_nb, p_b + log_p))
                        # Collapse (uzatılan aynı karakter)
                        cb_b, cb_nb = next_beams[prefix]
                        next_beams[prefix] = (cb_b, _log_add(cb_nb, p_nb + log_p))
                    else:
                        nb_b, nb_nb = next_beams[new_prefix]
                        next_beams[new_prefix] = (nb_b, _log_add(nb_nb, p_total + log_p))

        # En iyi beam_width prefix'i tut
        scored = sorted(
            next_beams.items(),
            key=lambda kv: _log_add(kv[1][0], kv[1][1]),
            reverse=True,
        )[:beam_width]
        beams = dict(scored)

    best_prefix = max(beams.items(), key=lambda kv: _log_add(kv[1][0], kv[1][1]))[0]
    return indices_to_text(list(best_prefix), num_phonemes=num_phonemes).strip()


# ─── Türkçe N-Gram + Viseme Sözlük Dil Modeli ────────────

class TurkishVisemeLM:
    """checkpoints/kenlm/corpus.txt ve unigrams.txt üzerinden çalışan
    Viterbi Bigram + Viseme-Ağırlıklı Türkçe Düzeltici.
    """

    def __init__(self) -> None:
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_len: dict[int, list[str]] = defaultdict(list)
        self.total_words: int = 1
        self.loaded: bool = False
        self._load()

    def _load(self) -> None:
        corpus_candidates = [
            ROOT / "checkpoints" / "kenlm" / "corpus.txt",
            ROOT / "backend" / "sample_videos" / "corpus.txt",
        ]
        unigrams_candidates = [
            ROOT / "checkpoints" / "kenlm" / "unigrams.txt",
            ROOT / "backend" / "sample_videos" / "unigrams.txt",
        ]
        corpus_path = next((p for p in corpus_candidates if p.is_file()), corpus_candidates[0])
        unigrams_path = next((p for p in unigrams_candidates if p.is_file()), unigrams_candidates[0])

        if corpus_path.is_file():
            try:
                with open(corpus_path, "r", encoding="utf-8") as f:
                    for line in f:
                        words = [w for w in line.strip().lower().split() if len(w) >= 2]
                        prev = "<bos>"
                        for w in words:
                            self.unigram_counts[w] += 1
                            self.bigram_counts[prev][w] += 1
                            prev = w
            except Exception:
                pass

        if unigrams_path.is_file():
            try:
                with open(unigrams_path, "r", encoding="utf-8") as f:
                    for line in f:
                        w = line.strip().lower()
                        if len(w) >= 2 and w not in self.unigram_counts:
                            self.unigram_counts[w] = 1
            except Exception:
                pass

        self.total_words = max(1, sum(self.unigram_counts.values()))
        # Her uzunluk için en sık 1500 kelimeyi indeksle (hızlı aday arama)
        for w, _ in self.unigram_counts.most_common():
            bucket = self.by_len[len(w)]
            if len(bucket) < 1500:
                bucket.append(w)
        self.loaded = len(self.unigram_counts) > 0

    def _premerge_split_tokens(self, tokens: list[str]) -> list[str]:
        """CTC'nin yanlışlıkla böldüğü kelimeleri birleştirir veya bitişik kelimeleri ayırır."""
        if not tokens:
            return tokens
        # 1. Bitişik yazılmış iki yaygın kelimeyi ayır (ör. 'şöylebir' -> 'şöyle bir')
        split_pass: list[str] = []
        for tok in tokens:
            if tok not in self.unigram_counts and len(tok) >= 6:
                found_split = None
                for k in range(3, len(tok) - 2):
                    w1, w2 = tok[:k], tok[k:]
                    if self.unigram_counts.get(w1, 0) >= 5 and self.unigram_counts.get(w2, 0) >= 5:
                        found_split = (w1, w2)
                        break
                if found_split:
                    split_pass.extend(found_split)
                    continue
            split_pass.append(tok)

        if len(split_pass) < 2:
            return split_pass
        # 2. Yanlışlıkla bölünmüş kelimeleri birleştir (ör. 'başla yabilir' -> 'başlayabilir')
        out: list[str] = []
        i = 0
        while i < len(split_pass):
            if i + 1 < len(split_pass):
                merged = split_pass[i] + split_pass[i + 1]
                if merged in self.unigram_counts and (
                    len(split_pass[i]) <= 3 or len(split_pass[i + 1]) <= 3 or self.unigram_counts[merged] >= 2
                ):
                    out.append(merged)
                    i += 2
                    continue
            out.append(split_pass[i])
            i += 1
        return out

    def _candidates(self, token: str, prev_words: list[str], max_dist: float = 2.4) -> list[tuple[str, float]]:
        """Bir token için (aday_kelime, viseme_maliyeti) listesi döndürür."""
        # Başta sık görülen sahte ünlü/ünsüz tekrarı temizliği (ör. 'ekendini'→'kendini', 'bbaşla'→'başla')
        clean_variants = {token}
        if len(token) >= 4 and token[0] == token[1]:
            clean_variants.add(token[1:])
        if len(token) >= 5 and token[0] in "aeıioöuü" and token[1:] in self.unigram_counts:
            clean_variants.add(token[1:])

        cand_map: dict[str, float] = {}
        for var in clean_variants:
            base_pen = 0.0 if var == token else 0.35
            if var in self.unigram_counts:
                cand_map[var] = min(cand_map.get(var, 99.0), base_pen)

        # Önceki kelimelerin bigram takipçilerini öncelikli tara (bağlam adayları)
        context_Followers: set[str] = set()
        for pw in prev_words:
            for fw in self.bigram_counts.get(pw, {}):
                context_Followers.add(fw)

        L = len(token)
        search_pool: set[str] = set(context_Followers)
        for dl in range(max(2, L - 2), L + 3):
            for w in self.by_len.get(dl, [])[:600]:
                search_pool.add(w)

        for w in search_pool:
            if abs(len(w) - L) > 2:
                continue
            d = viseme_edit_distance(token, w, max_dist=max_dist)
            if d <= max_dist:
                if w not in cand_map or d < cand_map[w]:
                    cand_map[w] = d

        if not cand_map:
            return [(token, 0.0)]

        # En düşük mesafeli 15 adayı döndür
        return sorted(cand_map.items(), key=lambda x: (x[1], -self.unigram_counts.get(x[0], 0)))[:15]

    def rescore(self, raw_text: str, lm_weight: float = 1.35) -> str:
        """Ham CTC metnini Viterbi Bigram + Viseme modeliyle düzeltir."""
        if not self.loaded or not raw_text:
            return raw_text
        raw_tokens = [t for t in raw_text.strip().lower().split() if t]
        if not raw_tokens:
            return raw_text

        tokens = self._premerge_split_tokens(raw_tokens)

        # Viterbi: state = (son_kelime, toplam_skor, yol)
        beams: list[tuple[str, float, list[str]]] = [("<bos>", 0.0, [])]

        for tok in tokens:
            prev_words = [b[0] for b in beams]
            cands = self._candidates(tok, prev_words)
            next_beams: dict[str, tuple[float, list[str]]] = {}

            for prev_w, prev_score, path in beams:
                prev_cnt = self.unigram_counts.get(prev_w, 0) if prev_w != "<bos>" else 6461
                for cand_w, dist in cands:
                    # Akustik/görsel skor (düşük mesafe = yüksek skor)
                    vis_score = -2.4 * dist
                    # Dil modeli skoru: log P(cand_w | prev_w) (add-k smoothing)
                    bi_cnt = self.bigram_counts.get(prev_w, {}).get(cand_w, 0)
                    uni_prob = (self.unigram_counts.get(cand_w, 0) + 0.5) / (self.total_words + 10000.0)
                    if bi_cnt > 0 and prev_cnt > 0:
                        lm_prob = 0.75 * (bi_cnt / prev_cnt) + 0.25 * uni_prob
                    else:
                        lm_prob = 0.12 * uni_prob
                    total = prev_score + vis_score + lm_weight * math.log(max(lm_prob, 1e-9))

                    if cand_w not in next_beams or total > next_beams[cand_w][0]:
                        next_beams[cand_w] = (total, path + [cand_w])

            beams = sorted(
                [(w, sc, p) for w, (sc, p) in next_beams.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:12]

        return " ".join(beams[0][2]) if beams else raw_text


_viseme_lm: Optional[TurkishVisemeLM] = None


def get_viseme_lm() -> TurkishVisemeLM:
    global _viseme_lm
    if _viseme_lm is None:
        _viseme_lm = TurkishVisemeLM()
    return _viseme_lm
