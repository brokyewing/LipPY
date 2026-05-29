"""
Adim 4: Yeni Whisper cache + eski text alignment dogrulamasi

Plan B icin kritik test:
  - adim3_audio.py ile uretilen YENI Whisper cache'i oku
  - words_to_segments uygula (adim2_segmentle ile birebir ayni)
  - Eski dataset/texts/<video>_<idx>.txt dosyalariyla KARSILASTIR
  - Alignment kalitesini olc:
      * Segment sayisi ayni mi?
      * Her segment'in metni eski text dosyasiyla eslesyior mu?

EK olarak bir ornek audio segment kes -> kullanici dinleyip dogrulayabilir.

Kullanim:
    python adim4_ses_dogrula.py --video test_videosu181
    python adim4_ses_dogrula.py --video test_videosu181 --kes-ornek 0
    python adim4_ses_dogrula.py --video test_videosu181 --kes-hepsi
"""

import os
import sys
import argparse
import subprocess

# adim2_segmentle.py'den fonksiyonlari import et (birebir ayni segmentleme icin)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adim2_segmentle import read_whisper_cache, words_to_segments, normalize_text


def read_old_text(texts_dir, video_name, idx):
    """Eski text dosyasini oku. Bulunmazsa None."""
    path = os.path.join(texts_dir, f"{video_name}_{idx:04d}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def count_old_segments(texts_dir, video_name):
    """video_name icin eski text dosyalarini say + idx listesi."""
    if not os.path.isdir(texts_dir):
        return []
    prefix = f"{video_name}_"
    idxs = []
    for f in os.listdir(texts_dir):
        if f.startswith(prefix) and f.endswith(".txt"):
            try:
                idx = int(f[len(prefix):-4])
                idxs.append(idx)
            except ValueError:
                pass
    return sorted(idxs)


def char_level_match(a, b):
    """Iki normalize edilmis string ne kadar benziyor (0.0-1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Edit distance / max length
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i, ai in enumerate(a):
        r = [dp[0] + 1]
        for j, bj in enumerate(b):
            r.append(min(r[-1] + 1, dp[j + 1] + 1, dp[j] + (0 if ai == bj else 1)))
        dp = r
    dist = dp[-1]
    return 1.0 - dist / max(n, m)


def cut_audio_segment(m4a_path, start, end, out_wav):
    """
    m4a'dan ses segmenti kes (16 kHz mono PCM WAV).
    Wav2Vec2 input format'ina uygun.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", m4a_path,
        "-t", str(end - start),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out_wav,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Adim 4: Audio-text alignment dogrulama")
    parser.add_argument("--video", required=True,
                        help="Video adi (ornek: test_videosu181)")
    parser.add_argument("--whisper-dir", default="dataset",
                        help="Whisper cache klasoru")
    parser.add_argument("--texts-dir", default="dataset/texts",
                        help="Eski text dosyalari klasoru")
    parser.add_argument("--audio-dir", default="audios_raw",
                        help="Indirilen m4a klasoru")
    parser.add_argument("--out-audio-dir", default="dataset/audios",
                        help="Kesilen audio segmentlerinin yazilacagi klasor")
    parser.add_argument("--kes-ornek", type=int, default=None,
                        help="Belirtilen idx'li audio segmenti kes (ornek olarak)")
    parser.add_argument("--kes-hepsi", action="store_true",
                        help="Tum segmentleri kes (.wav cikar)")
    parser.add_argument("--match-threshold", type=float, default=0.85,
                        help="Char-level match esigi (>= ise OK sayilir)")
    args = parser.parse_args()

    # 1. Yeni Whisper cache'i oku
    cache_path = os.path.join(args.whisper_dir, f"_whisper_{args.video}.txt")
    if not os.path.exists(cache_path):
        print(f"HATA: Whisper cache yok: {cache_path}", flush=True)
        print(f"  Once adim3_audio.py --test {args.video} koshtur.", flush=True)
        sys.exit(1)

    print(f"Whisper cache okunuyor: {cache_path}", flush=True)
    words = read_whisper_cache(cache_path)
    print(f"  Kelime sayisi: {len(words)}", flush=True)

    # 2. words_to_segments (adim2 ile birebir ayni parametreler)
    new_segments = words_to_segments(
        words,
        min_sure=1.5,
        max_sure=5.0,
        max_bosluk=0.8,
        min_kelime=2,
        fps=25.0,
        min_ctc_ratio=1.5,
    )
    print(f"  Yeni segment sayisi: {len(new_segments)}", flush=True)

    # 3. Eski text dosyalarini oku
    old_idxs = count_old_segments(args.texts_dir, args.video)
    print(f"  Eski text dosya sayisi: {len(old_idxs)}", flush=True)

    if not old_idxs:
        print(f"\nUYARI: Eski text yok ({args.texts_dir}/{args.video}_*.txt)", flush=True)
        print(f"  Bu video dataset'te yok veya farkli yerde.", flush=True)

    # 4. Alignment karsilastirmasi (DUZELTILMIS)
    print(f"\n{'='*70}")
    print(f"ALIGNMENT KARSILASTIRMASI (index-bazli)")
    print(f"{'='*70}")
    print(f"NOT: Eski text dosyalari SPARSE (temizleme sonrasi {len(old_idxs)} kaldi),")
    print(f"     indeksleri orijinal Whisper segment sirasini koruyor.")

    # DOGRU karsilastirma: her surviving old_idx icin new_segments[old_idx] ile karsilastir
    sayim = {"exact": 0, "high": 0, "medium": 0, "low": 0, "out_of_range": 0}
    detayli = []
    fuzzy_drift = []  # her old icin en iyi NEW match'in idx offset'i

    new_norms = [normalize_text(s[2]) for s in new_segments]

    for old_idx in old_idxs:
        old_txt = read_old_text(args.texts_dir, args.video, old_idx)
        if old_txt is None:
            continue
        old_norm = normalize_text(old_txt)

        # Index-bazli direkt esleme
        if old_idx >= len(new_segments):
            sayim["out_of_range"] += 1
            detayli.append((old_idx, "OOR", "", old_norm, 0.0, None))
            continue

        new_norm = new_norms[old_idx]
        match = char_level_match(new_norm, old_norm)

        if match >= 0.99:
            sayim["exact"] += 1; tag = "EXACT"
        elif match >= args.match_threshold:
            sayim["high"] += 1; tag = "HIGH"
        elif match >= 0.6:
            sayim["medium"] += 1; tag = "MED"
        else:
            sayim["low"] += 1; tag = "LOW"

        # FUZZY: en iyi match nerede? (drift araysir)
        best_match = match
        best_offset = 0
        # +/-50 segment penceresinde ara
        search_lo = max(0, old_idx - 50)
        search_hi = min(len(new_segments), old_idx + 51)
        for j in range(search_lo, search_hi):
            m = char_level_match(new_norms[j], old_norm)
            if m > best_match:
                best_match = m
                best_offset = j - old_idx
        fuzzy_drift.append((old_idx, best_offset, best_match))

        detayli.append((old_idx, tag, new_norm, old_norm, match, best_offset))

    # Ozet
    total_compared = sum(sayim.values())
    print(f"\nOzet (eski text {len(old_idxs)} kaydi karsilastirildi):")
    for k, v in sayim.items():
        pct = 100.0 * v / max(total_compared, 1)
        print(f"  {k:13s}: {v:4d} ({pct:5.1f}%)")

    # Ilk 10 ornek (index-bazli, dogru karsilastirma)
    print(f"\nIlk 10 ornek (new_segments[old_idx] vs text_<old_idx>.txt):")
    print(f"  {'old_idx':>7} {'tag':>7} {'match':>7} {'drift':>5}  text karsilastirmasi")
    for old_idx, tag, new_t, old_t, match, drift in detayli[:10]:
        marker = "OK" if tag in ("EXACT", "HIGH") else ("?" if tag == "MED" else "!")
        drift_str = f"{drift:+3d}" if drift is not None else " ?"
        print(f"  {old_idx:7d} {tag:>7} {match:>6.1%} {drift_str:>5}  "
              f"{marker} new[{old_idx}]: '{new_t[:55]}'")
        print(f"  {' '*7} {' '*7} {' '*6} {' '*5}    old_{old_idx:04d}.txt: '{old_t[:55]}'")

    # Fuzzy drift analizi
    if fuzzy_drift:
        print(f"\nFUZZY DRIFT ANALIZI (her eski text icin en iyi yeni segment nerede?):")
        offsets = [d[1] for d in fuzzy_drift]
        from collections import Counter
        offset_counter = Counter(offsets)
        print(f"  Drift dagilimi (en yaygin 10):")
        for off, cnt in offset_counter.most_common(10):
            pct = 100.0 * cnt / len(offsets)
            print(f"    offset {off:+4d}: {cnt:3d} kayit ({pct:5.1f}%)")

        # Fuzzy match ile gecerli ornek sayisi
        good_fuzzy = sum(1 for _, _, m in fuzzy_drift if m >= args.match_threshold)
        print(f"\n  Fuzzy match >= {args.match_threshold:.0%}: "
              f"{good_fuzzy}/{len(fuzzy_drift)} "
              f"({100.0 * good_fuzzy / max(len(fuzzy_drift), 1):.1f}%)")
        print(f"  Yani: ICERIK olarak {good_fuzzy} kayit yeni Whisper'da bulunuyor,")
        print(f"        sadece SEGMENT INDEKSI kaymis durumda.")

    # 5. Verdict
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if not old_idxs:
        print("Eski veri ile karsilastirma yapilamadi.")
        return

    direct_ok = sayim["exact"] + sayim["high"]
    direct_total = direct_ok + sayim["medium"] + sayim["low"]
    direct_pct = 100.0 * direct_ok / max(direct_total, 1)

    fuzzy_ok = sum(1 for _, _, m in fuzzy_drift if m >= args.match_threshold) if fuzzy_drift else 0
    fuzzy_pct = 100.0 * fuzzy_ok / max(len(fuzzy_drift), 1) if fuzzy_drift else 0

    print(f"\nDirect match (index-bazli): {direct_pct:.1f}% (>= {args.match_threshold:.0%})")
    print(f"Fuzzy match (icerik bazli):  {fuzzy_pct:.1f}% (icerik var, idx kaymis)")

    if direct_pct >= 85:
        print(f"\nBASARILI: Index-bazli %85+ uyum.")
        print(f"  -> Yeni audio segmentleri eski text/ROI ile DOGRUDAN eslesir.")
        print(f"  -> Video indirmeye gerek YOK, sadece ses kes yeter.")
    elif fuzzy_pct >= 85:
        print(f"\nKURTARMA MUMKUN: Index kaymis ama icerik %85+ uyumlu.")
        print(f"  -> Yeni Whisper FARKLI segmentlere boldu (segment sayisi {len(new_segments)} vs eski {len(old_idxs)} suvived).")
        print(f"  -> Cozum: Her eski text icin yeni Whisper'da BENZER icerikli segmenti bul,")
        print(f"           o segmentin (start, end) timestamp'lerini kullan -> audio kes.")
        print(f"  -> Yine VIDEO indirmeye gerek yok!")
    elif fuzzy_pct >= 50:
        print(f"\nPARSIYEL: %50-85 icerik uyumu.")
        print(f"  -> Yeni Whisper bazi segmentleri kacirmis veya birlestirmis.")
        print(f"  -> Cozum 1: Kurtarilabilen segmentler icin audio kes,")
        print(f"             diger ROI'leri C planinda video-only kullan.")
        print(f"  -> Cozum 2: Whisper large-v3 dene (daha tutarli olabilir).")
    else:
        print(f"\nBASARISIZ: Whisper output cok farkli.")
        print(f"  -> Cozum: large-v3 dene, veya:")
        print(f"  -> Tam yeni dataset (video indir + adim2 + ROI re-extract).")

    # 6. Ses segmenti kes (ornek)
    m4a_path = os.path.join(args.audio_dir, f"{args.video}.m4a")

    if args.kes_ornek is not None:
        if not os.path.exists(m4a_path):
            print(f"\nHATA: m4a yok: {m4a_path}", flush=True)
            sys.exit(1)
        if args.kes_ornek >= len(new_segments):
            print(f"\nHATA: idx {args.kes_ornek} >= segment sayisi {len(new_segments)}",
                  flush=True)
            sys.exit(1)
        start, end, text = new_segments[args.kes_ornek]
        os.makedirs(args.out_audio_dir, exist_ok=True)
        out_wav = os.path.join(args.out_audio_dir,
                                f"{args.video}_{args.kes_ornek:04d}.wav")
        print(f"\nOrnek audio kes: idx={args.kes_ornek}")
        print(f"  Start: {start:.2f}s  End: {end:.2f}s  Sure: {end - start:.2f}s")
        print(f"  Yeni text: '{text}'")
        old_t = read_old_text(args.texts_dir, args.video, args.kes_ornek)
        if old_t:
            print(f"  Eski text: '{old_t}'")
        if cut_audio_segment(m4a_path, start, end, out_wav):
            size_kb = os.path.getsize(out_wav) / 1024
            print(f"  YAZILDI: {out_wav} ({size_kb:.0f} KB)")
            print(f"  Dinle ve dogrula: yeni text + ses ayni icerik mi?")
        else:
            print(f"  HATA: ffmpeg basarisiz")

    if args.kes_hepsi:
        if not os.path.exists(m4a_path):
            print(f"\nHATA: m4a yok: {m4a_path}", flush=True)
            sys.exit(1)
        os.makedirs(args.out_audio_dir, exist_ok=True)
        print(f"\nTum {len(new_segments)} segment kesiliyor -> {args.out_audio_dir}/")
        ok = 0
        fail = 0
        for i, (start, end, text) in enumerate(new_segments):
            out_wav = os.path.join(args.out_audio_dir,
                                    f"{args.video}_{i:04d}.wav")
            if os.path.exists(out_wav):
                ok += 1
                continue
            if cut_audio_segment(m4a_path, start, end, out_wav):
                ok += 1
            else:
                fail += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(new_segments)} kesildi", flush=True)
        print(f"  Sonuc: {ok} OK, {fail} fail")


if __name__ == "__main__":
    main()
