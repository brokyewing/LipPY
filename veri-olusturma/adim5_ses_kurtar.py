"""
Adim 5: Audio kurtarma scripti

adim4 ile gosterildi ki yeni Whisper SEGMENT sinirlarini farkli koyuyor,
ama ICERIK %75-90 ortusuyor. Bu script:

1. Her eski text dosyasi icin yeni Whisper segmentlerinde EN IYI eslesmeyi bul:
   - Once tek segment match (drift +/-100)
   - Sonra adjacent segment concat (1, 2, 3 segment birlesimi)
   - Word-level Jaccard overlap + char-level edit distance kombinasyonu
2. Match bulundu ise: m4a'dan o (start, end) araliginda audio kes
   -> dataset/audios/<video>_<old_idx>.wav (eski text/ROI ile EŞLEŞMIŞ)
3. Match bulunmadi ise: missing.txt'e yaz, kullanici karar versin

Kullanim:
    python adim5_ses_kurtar.py --video test_videosu181 --dry-run
    python adim5_ses_kurtar.py --video test_videosu181
    python adim5_ses_kurtar.py --all  # tum videolar (uzun is)
"""

import os
import sys
import argparse
import subprocess
import glob
import wave
import time
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adim2_segmentle import read_whisper_cache, words_to_segments, normalize_text


# ============================================================
# FAST: In-memory audio + multiprocessing
# ============================================================

def load_audio_ffmpeg(m4a_path, sr=16000):
    """m4a/webm -> numpy float32 array (mono, sr Hz). ONE ffmpeg call per video.

    Tam audio'yu RAM'e yukler. ~10 dk video = ~40 MB float32 array.
    """
    proc = subprocess.run([
        "ffmpeg", "-loglevel", "error",
        "-i", m4a_path,
        "-f", "f32le",       # 32-bit float little-endian raw
        "-ar", str(sr),
        "-ac", "1",          # mono
        "-",                  # stdout
    ], capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32)


def write_wav_int16(path, audio_float, sr=16000):
    """numpy float32 [-1, 1] -> int16 PCM WAV via stdlib wave module."""
    audio_int16 = np.clip(audio_float * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(audio_int16.tobytes())


def cut_audio_inmemory(audio_array, sr, start_sec, end_sec, out_wav):
    """Tam audio array'inden segment kes ve wav yaz. NO ffmpeg call."""
    start_sample = int(start_sec * sr)
    end_sample = min(int(end_sec * sr), len(audio_array))
    if start_sample >= end_sample:
        return False
    slice_audio = audio_array[start_sample:end_sample]
    if len(slice_audio) < sr * 0.1:  # min 100ms
        return False
    try:
        write_wav_int16(out_wav, slice_audio, sr)
        return True
    except Exception:
        return False


def char_match(a, b):
    """Char-level edit distance similarity (0.0-1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for ai in a:
        r = [dp[0] + 1]
        for j, bj in enumerate(b):
            r.append(min(r[-1] + 1, dp[j + 1] + 1, dp[j] + (0 if ai == bj else 1)))
        dp = r
    return 1.0 - dp[-1] / max(n, m)


def word_jaccard(a, b):
    """Kelime-level Jaccard overlap (0.0-1.0)."""
    aw = set(a.split())
    bw = set(b.split())
    if not aw and not bw:
        return 1.0
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def word_lcs(a_words, b_words):
    """Longest Common Subsequence (sirayi koruyarak)."""
    n, m = len(a_words), len(b_words)
    if n == 0 or m == 0:
        return 0
    # 1D DP (memory tasarrufu)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(n):
        for j in range(m):
            if a_words[i] == b_words[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev, curr = curr, [0] * (m + 1)
    return prev[m]


def containment_score(target_norm, container_norm):
    """
    Asimetrik: TARGET'in kelimeleri CONTAINER'da SIRALI olarak ne kadar var?
    Eger target tamamen container'in icindeyse 1.0.

    Ornek:
      target="merhaba dunya nasilsin"
      container="iyi merhaba dunya nasilsin sen iyi misin"
      -> 1.0 (tum target kelimeleri sirayla container'da var)
    """
    tw = target_norm.split()
    cw = container_norm.split()
    if not tw:
        return 0.0
    return word_lcs(tw, cw) / len(tw)


def combined_match(a, b):
    """
    A ve B icin BEST score:
      - Char-level edit distance (simetrik)
      - Word Jaccard (simetrik)
      - Containment A-in-B (asimetrik, A daha kisaysa)
      - Containment B-in-A (asimetrik, B daha kisaysa)

    En iyisini al -> containment basariliysa diger skorlar dusuk olsa bile yakaliyoruz.
    """
    c = char_match(a, b)
    w = word_jaccard(a, b)
    # Asimetrik: hangisi daha kisa, onun digerinin icindeki orani
    sym = 0.5 * c + 0.5 * w
    cont_a_in_b = containment_score(a, b)
    cont_b_in_a = containment_score(b, a)
    # En iyisini al
    return max(sym, cont_a_in_b, cont_b_in_a)


def find_best_match(old_norm, new_segments, new_norms,
                    expected_idx, search_window=None, max_concat=4,
                    min_threshold=0.6, return_best_always=False):
    """
    old_norm icin en iyi eslesmeyi bul.

    Strateji:
      1. Tek segment match (window: expected_idx +/- search_window)
         search_window=None ise TUM segmentleri tara
      2. Adjacent concat (2, 3, 4 segment birlesimi)

    Return: (start_idx, end_idx, match_score, segment_count)
            None if no match found above threshold (unless return_best_always=True).
    """
    if search_window is None:
        lo, hi = 0, len(new_segments)
    else:
        lo = max(0, expected_idx - search_window)
        hi = min(len(new_segments), expected_idx + search_window + 1)

    best = None
    best_score = 0.0

    # 1. Tek segment
    for j in range(lo, hi):
        score = combined_match(new_norms[j], old_norm)
        if score > best_score:
            best_score = score
            best = (j, j, score, 1)

    # 2. 2-segment concat
    for j in range(lo, hi - 1):
        concat = (new_norms[j] + " " + new_norms[j + 1]).strip()
        score = combined_match(concat, old_norm)
        if score > best_score:
            best_score = score
            best = (j, j + 1, score, 2)

    # 3. 3-segment concat
    if max_concat >= 3:
        for j in range(lo, hi - 2):
            concat = " ".join(new_norms[j:j + 3]).strip()
            score = combined_match(concat, old_norm)
            if score > best_score:
                best_score = score
                best = (j, j + 2, score, 3)

    # 4. 4-segment concat (cok uzun cumleler icin)
    if max_concat >= 4:
        for j in range(lo, hi - 3):
            concat = " ".join(new_norms[j:j + 4]).strip()
            score = combined_match(concat, old_norm)
            if score > best_score:
                best_score = score
                best = (j, j + 3, score, 4)

    if return_best_always:
        return best
    if best_score >= min_threshold:
        return best
    return None


def find_word_level_alignment(old_norm, words, min_coverage=0.7):
    """
    Eski text'in kelimelerini new Whisper'in kelime listesinde SIRALI olarak ara.

    words: read_whisper_cache cikti - list of {start, end, text}
    old_norm: normalize edilmis eski text

    Sliding window: her i pozisyonu icin words[i:i+len(old_words)+padding]'da
    eski kelimelerin kac tanesi var, hangi siralayla?

    Return: (start_time, end_time, coverage, word_count) or None.
    """
    old_words = old_norm.split()
    n_old = len(old_words)
    if n_old < 2:
        return None

    new_word_texts = [normalize_text(w['text']) for w in words]

    best = None
    best_cov = 0.0

    # Pencere uzunlugu: eski kelime sayisi + %50 marj
    win_size = int(n_old * 1.5) + 2

    for i in range(len(words) - n_old + 1):
        j_end = min(i + win_size, len(words))
        window = new_word_texts[i:j_end]
        # LCS / n_old: eski kelimelerin ne kadari sirali olarak burada?
        lcs = word_lcs(old_words, window)
        coverage = lcs / n_old

        if coverage > best_cov:
            # Tam kelime konumlarini bul: ilk eski kelime ile son eski kelimenin yeri
            first_match = None
            last_match = None
            old_i = 0
            for k, w_text in enumerate(window):
                if old_i < n_old and w_text == old_words[old_i]:
                    if first_match is None:
                        first_match = i + k
                    last_match = i + k
                    old_i += 1
            if first_match is not None and last_match is not None and last_match > first_match:
                start_time = words[first_match]['start']
                end_time = words[last_match]['end']
                best_cov = coverage
                best = (start_time, end_time, coverage, last_match - first_match + 1)

    if best is None or best[2] < min_coverage:
        return None
    return best


def cut_audio(m4a_path, start, end, out_wav):
    """16kHz mono PCM WAV (Wav2Vec2 input format)."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start),
        "-i", m4a_path,
        "-t", str(end - start),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        out_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def kurtar_video(video_name, whisper_dir, texts_dir, audio_dir,
                  out_audio_dir, search_window=200, threshold=0.6,
                  high_threshold=0.85, dry_run=False):
    """
    Bir video icin tum eski text'lere karsilik audio segmentleri kurtar.

    HIZ OPTIMIZASYONU: m4a tam audio'yu BIR KEZ RAM'e yukler (load_audio_ffmpeg),
    sonra her segment icin numpy slice + wav yazar. Eski ffmpeg-per-segment'ten
    ~5x hizli.

    Return: dict with stats.
    """
    cache_path = os.path.join(whisper_dir, f"_whisper_{video_name}.txt")
    m4a_path = os.path.join(audio_dir, f"{video_name}.m4a")

    if not os.path.exists(cache_path):
        return {"video": video_name, "error": "no_whisper_cache"}
    if not dry_run and not os.path.exists(m4a_path):
        return {"video": video_name, "error": "no_m4a"}

    # Whisper -> segments + words (her ikisi de kullanilacak)
    words = read_whisper_cache(cache_path)
    new_segments = words_to_segments(
        words, min_sure=1.5, max_sure=5.0, max_bosluk=0.8,
        min_kelime=2, fps=25.0, min_ctc_ratio=1.5,
    )
    new_norms = [normalize_text(s[2]) for s in new_segments]

    # Eski text dosyalari
    prefix = f"{video_name}_"
    old_files = []
    for f in sorted(os.listdir(texts_dir)):
        if f.startswith(prefix) and f.endswith(".txt"):
            try:
                idx = int(f[len(prefix):-4])
                old_files.append((idx, os.path.join(texts_dir, f)))
            except ValueError:
                pass

    if not old_files:
        return {"video": video_name, "error": "no_old_texts"}

    if not dry_run:
        os.makedirs(out_audio_dir, exist_ok=True)

    # FAST: Audio'yu BIR KEZ RAM'e yukle (one ffmpeg call total)
    SR = 16000
    audio_array = None
    if not dry_run:
        audio_array = load_audio_ffmpeg(m4a_path, sr=SR)
        if audio_array is None:
            return {"video": video_name, "error": "ffmpeg_load_failed"}

    stats = {
        "video": video_name,
        "old_count": len(old_files),
        "new_seg_count": len(new_segments),
        "high_match": 0,   # >= high_threshold
        "low_match": 0,    # >= threshold ama < high
        "no_match": 0,
        "cut_ok": 0,
        "cut_fail": 0,
        "concat_used": 0,
        "word_align_used": 0,  # word-level fallback kullanildi
        "details": [],
    }

    for old_idx, old_path in old_files:
        with open(old_path, "r", encoding="utf-8") as f:
            old_txt = f.read().strip()
        old_norm = normalize_text(old_txt)

        # ALL segments araz, en iyi match'i her zaman al
        match = find_best_match(old_norm, new_segments, new_norms,
                                 expected_idx=old_idx,
                                 search_window=None,   # TUM segmentleri tara
                                 max_concat=4,
                                 min_threshold=threshold,
                                 return_best_always=True)

        if match is None or match[2] < threshold:
            # Segment-level basarisiz: WORD-LEVEL alignment dene
            word_align = find_word_level_alignment(old_norm, words,
                                                     min_coverage=0.7)
            if word_align is not None:
                # Word-level basarili: bunu kullan
                start_time, end_time, coverage, n_matched = word_align
                stats["word_align_used"] += 1
                if coverage >= high_threshold:
                    stats["high_match"] += 1
                else:
                    stats["low_match"] += 1

                out_wav = os.path.join(out_audio_dir,
                                        f"{video_name}_{old_idx:04d}.wav")
                stats["details"].append({
                    "old_idx": old_idx,
                    "new_start_idx": -1,  # word-level kullanildi
                    "new_end_idx": -1,
                    "drift": 0,
                    "concat": 0,
                    "score": coverage,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "method": "word_align",
                    "matched_words": n_matched,
                })

                if not dry_run:
                    if cut_audio_inmemory(audio_array, SR, start_time, end_time, out_wav):
                        stats["cut_ok"] += 1
                    else:
                        stats["cut_fail"] += 1
                continue

            # Hem segment hem word level basarisiz: gercek NO match
            stats["no_match"] += 1
            best_attempt = match if match is not None else (None, None, 0.0, 0)
            stats["details"].append({
                "old_idx": old_idx,
                "match": None,
                "score": best_attempt[2],
                "best_new_start": best_attempt[0],
                "best_new_end": best_attempt[1],
                "best_concat": best_attempt[3],
            })
            continue

        start_seg_idx, end_seg_idx, score, seg_count = match
        if seg_count > 1:
            stats["concat_used"] += 1

        if score >= high_threshold:
            stats["high_match"] += 1
        else:
            stats["low_match"] += 1

        # Audio kes
        start_time = new_segments[start_seg_idx][0]
        end_time = new_segments[end_seg_idx][1]
        drift = start_seg_idx - old_idx

        out_wav = os.path.join(out_audio_dir, f"{video_name}_{old_idx:04d}.wav")
        stats["details"].append({
            "old_idx": old_idx,
            "new_start_idx": start_seg_idx,
            "new_end_idx": end_seg_idx,
            "drift": drift,
            "concat": seg_count,
            "score": score,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
        })

        if dry_run:
            continue

        if cut_audio_inmemory(audio_array, SR, start_time, end_time, out_wav):
            stats["cut_ok"] += 1
        else:
            stats["cut_fail"] += 1

    return stats


# ============================================================
# Multiprocessing worker (must be at module level)
# ============================================================

def _worker_process_video(task_args):
    """Single video processor for multiprocessing.Pool.

    task_args: (video_name, whisper_dir, texts_dir, audio_dir, out_audio_dir,
                search_window, threshold, high_threshold, dry_run)
    """
    (video_name, whisper_dir, texts_dir, audio_dir, out_audio_dir,
     search_window, threshold, high_threshold, dry_run) = task_args
    try:
        stats = kurtar_video(
            video_name, whisper_dir, texts_dir, audio_dir, out_audio_dir,
            search_window=search_window,
            threshold=threshold,
            high_threshold=high_threshold,
            dry_run=dry_run,
        )
        return stats
    except Exception as e:
        return {"video": video_name, "error": f"exception: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Adim 5: Audio kurtarma")
    parser.add_argument("--video", default=None,
                        help="Tek video adi (test_videosu181). --all yoksa gerekli.")
    parser.add_argument("--all", action="store_true",
                        help="Tum video'lar icin (Whisper cache olanlar)")
    parser.add_argument("--whisper-dir", default="dataset")
    parser.add_argument("--texts-dir", default="dataset/texts")
    parser.add_argument("--audio-dir", default="audios_raw")
    parser.add_argument("--out-audio-dir", default="dataset/audios")
    parser.add_argument("--search-window", type=int, default=200,
                        help="Drift arama penceresi (+/-)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="En dusuk match esigi (altinda 'no match')")
    parser.add_argument("--high-threshold", type=float, default=0.85,
                        help="Yuksek guven esigi (BASARILI sayim icin)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Audio kesme, sadece istatistik raporla")
    parser.add_argument("--workers", type=int, default=8,
                        help="--all icin paralel worker sayisi (varsayilan 8). "
                             "Tek video icin yoksayilir.")
    args = parser.parse_args()

    if not args.video and not args.all:
        print("HATA: --video <name> veya --all gerekli", flush=True)
        sys.exit(1)

    if args.all:
        # Whisper cache'i olan tum video'lari bul
        pattern = os.path.join(args.whisper_dir, "_whisper_*.txt")
        videos = []
        for p in sorted(glob.glob(pattern)):
            name = os.path.basename(p)[len("_whisper_"):-4]
            videos.append(name)
        print(f"Toplam {len(videos)} video kurtarilacak", flush=True)
    else:
        videos = [args.video]

    toplam = {
        "old": 0, "high": 0, "low": 0, "no": 0,
        "cut_ok": 0, "cut_fail": 0, "concat": 0,
    }

    # ======================================================
    # PARALLEL PATH: --all + workers > 1 + olusturan video sayisi > 1
    # ======================================================
    use_parallel = args.all and args.workers > 1 and len(videos) > 1
    if use_parallel:
        n_workers = min(args.workers, len(videos), cpu_count())
        print(f"\nPARALEL MODE: {n_workers} worker (multiprocessing)",
              flush=True)
        task_args_list = [
            (v, args.whisper_dir, args.texts_dir, args.audio_dir,
             args.out_audio_dir, args.search_window, args.threshold,
             args.high_threshold, args.dry_run)
            for v in videos
        ]
        t_start = time.time()
        with Pool(processes=n_workers) as pool:
            completed = 0
            for stats in pool.imap_unordered(_worker_process_video,
                                              task_args_list):
                completed += 1
                v = stats.get("video", "?")

                if "error" in stats:
                    print(f"[{completed}/{len(videos)}] {v}: HATA {stats['error']}",
                          flush=True)
                    continue

                # Toplam guncelle
                toplam["old"] += stats.get("old_count", 0)
                toplam["high"] += stats.get("high_match", 0)
                toplam["low"] += stats.get("low_match", 0)
                toplam["no"] += stats.get("no_match", 0)
                toplam["cut_ok"] += stats.get("cut_ok", 0)
                toplam["cut_fail"] += stats.get("cut_fail", 0)
                toplam["concat"] += stats.get("concat_used", 0)

                # Hizli rapor (tek satir per video)
                total = stats.get("old_count", 0)
                elapsed = time.time() - t_start
                eta = (elapsed / completed) * (len(videos) - completed)
                print(
                    f"[{completed}/{len(videos)}] {v}: "
                    f"H={stats.get('high_match',0)} "
                    f"L={stats.get('low_match',0)} "
                    f"N={stats.get('no_match',0)} | "
                    f"cut={stats.get('cut_ok',0)} OK / "
                    f"{stats.get('cut_fail',0)} fail | "
                    f"eta {eta/60:.1f}dk",
                    flush=True,
                )

        # Final ozet
        print(f"\n{'='*60}")
        print(f"TOPLAM OZET (paralel mode)")
        print(f"{'='*60}")
        print(f"  Eski text dosyasi:  {toplam['old']:,}")
        print(f"  HIGH match:         {toplam['high']:,} "
              f"({100.0 * toplam['high'] / max(toplam['old'], 1):.1f}%)")
        print(f"  LOW match:          {toplam['low']:,} "
              f"({100.0 * toplam['low'] / max(toplam['old'], 1):.1f}%)")
        print(f"  NO match:           {toplam['no']:,} "
              f"({100.0 * toplam['no'] / max(toplam['old'], 1):.1f}%)")
        print(f"  Concat kullanildi:  {toplam['concat']:,}")
        if not args.dry_run:
            print(f"  Audio kesildi:      {toplam['cut_ok']:,} OK, "
                  f"{toplam['cut_fail']:,} fail")
        print(f"  Sure:               {(time.time()-t_start)/60:.1f} dakika")
        return

    # ======================================================
    # SERIAL PATH: tek video veya workers=1
    # ======================================================
    for v in videos:
        print(f"\n--- {v} ---", flush=True)
        stats = kurtar_video(
            v,
            args.whisper_dir, args.texts_dir, args.audio_dir,
            args.out_audio_dir,
            search_window=args.search_window,
            threshold=args.threshold,
            high_threshold=args.high_threshold,
            dry_run=args.dry_run,
        )

        if "error" in stats:
            print(f"  HATA: {stats['error']}", flush=True)
            continue

        toplam["old"] += stats["old_count"]
        toplam["high"] += stats["high_match"]
        toplam["low"] += stats["low_match"]
        toplam["no"] += stats["no_match"]
        toplam["cut_ok"] += stats["cut_ok"]
        toplam["cut_fail"] += stats["cut_fail"]
        toplam["concat"] += stats["concat_used"]

        total = stats["old_count"]
        print(f"  Eski text: {total} | Yeni seg: {stats['new_seg_count']}", flush=True)
        print(f"  HIGH match (>={args.high_threshold:.0%}): {stats['high_match']} "
              f"({100.0 * stats['high_match'] / max(total, 1):.1f}%)", flush=True)
        print(f"  LOW match  (>={args.threshold:.0%}):     {stats['low_match']} "
              f"({100.0 * stats['low_match'] / max(total, 1):.1f}%)", flush=True)
        print(f"  NO match   (<{args.threshold:.0%}):     {stats['no_match']} "
              f"({100.0 * stats['no_match'] / max(total, 1):.1f}%)", flush=True)
        print(f"  Concat kullanildi: {stats['concat_used']}", flush=True)
        print(f"  Word-level align:  {stats['word_align_used']}", flush=True)
        if not args.dry_run:
            print(f"  Audio kesildi: {stats['cut_ok']} OK, {stats['cut_fail']} fail",
                  flush=True)

        # Ilk 5 detay
        if args.video and stats["details"]:
            print(f"\n  Ilk 5 detay:", flush=True)
            for d in stats["details"][:5]:
                if "new_start_idx" in d:
                    # Match basarili
                    print(
                        f"    idx {d['old_idx']:04d}: "
                        f"new[{d['new_start_idx']}-{d['new_end_idx']}] "
                        f"drift={d['drift']:+3d} "
                        f"concat={d['concat']} "
                        f"score={d['score']:.1%} "
                        f"({d['start_time']:.1f}-{d['end_time']:.1f}s)",
                        flush=True,
                    )
                else:
                    # NO match
                    best_s = d.get("best_new_start")
                    best_e = d.get("best_new_end")
                    if best_s is not None:
                        drift = best_s - d["old_idx"]
                        print(
                            f"    idx {d['old_idx']:04d}: NO MATCH "
                            f"(en iyi: new[{best_s}-{best_e}] drift={drift:+3d} "
                            f"score={d['score']:.1%})",
                            flush=True,
                        )
                    else:
                        print(f"    idx {d['old_idx']:04d}: NO MATCH", flush=True)

            # NO match ornekleri — FILTER: success ise new_start_idx var, NO match'te yok
            no_match = [d for d in stats["details"] if "new_start_idx" not in d]
            if no_match:
                # new_segments'i tekrar yukle (NO match raporu icin)
                cache_p = os.path.join(args.whisper_dir, f"_whisper_{v}.txt")
                words = read_whisper_cache(cache_p)
                ns = words_to_segments(
                    words, min_sure=1.5, max_sure=5.0, max_bosluk=0.8,
                    min_kelime=2, fps=25.0, min_ctc_ratio=1.5,
                )
                print(f"\n  NO match olan {len(no_match)} kayit (en iyi deneme dahil):",
                      flush=True)
                for d in no_match:
                    old_idx = d["old_idx"]
                    try:
                        with open(os.path.join(args.texts_dir,
                                                f"{v}_{old_idx:04d}.txt"),
                                  encoding="utf-8") as f:
                            old = f.read().strip()
                    except Exception:
                        old = "?"
                    best_score = d.get("score", 0.0)
                    best_start = d.get("best_new_start")
                    best_end = d.get("best_new_end", best_start)
                    best_concat = d.get("best_concat", 0)
                    drift = (best_start - old_idx) if best_start is not None else None
                    drift_str = f"{drift:+4d}" if drift is not None else " ?"
                    print(f"    {old_idx:04d} | en iyi: new[{best_start}-{best_end}] "
                          f"drift={drift_str} concat={best_concat} "
                          f"score={best_score:.1%}", flush=True)
                    print(f"           old: '{old[:70]}'", flush=True)
                    if best_start is not None and best_start < len(ns):
                        try:
                            new_concat_text = " ".join(
                                normalize_text(ns[k][2])
                                for k in range(best_start, min(best_end + 1, len(ns)))
                            )
                            print(f"           new: '{new_concat_text[:70]}'",
                                  flush=True)
                        except Exception as e:
                            print(f"           new: (okuma hatasi: {e})", flush=True)

    # Toplam ozet
    if args.all:
        print(f"\n{'='*60}")
        print(f"TOPLAM OZET")
        print(f"{'='*60}")
        print(f"  Eski text dosyasi:  {toplam['old']:,}")
        print(f"  HIGH match:         {toplam['high']:,} "
              f"({100.0 * toplam['high'] / max(toplam['old'], 1):.1f}%)")
        print(f"  LOW match:          {toplam['low']:,} "
              f"({100.0 * toplam['low'] / max(toplam['old'], 1):.1f}%)")
        print(f"  NO match:           {toplam['no']:,} "
              f"({100.0 * toplam['no'] / max(toplam['old'], 1):.1f}%)")
        print(f"  Concat kullanildi:  {toplam['concat']:,}")
        if not args.dry_run:
            print(f"  Audio kesildi:      {toplam['cut_ok']:,} OK, "
                  f"{toplam['cut_fail']:,} fail")


if __name__ == "__main__":
    main()
