"""
Adim 2: Whisper transkriptlerinden segment olustur + klip kes
dataset/_whisper_*.txt -> dataset/clips/*.mp4 + dataset/texts/*.txt + dataset/splits/

Kullanim:
    python adim2_segmentle.py
    python adim2_segmentle.py --min-sure 1.5 --max-sure 5.0 --min-ctc-ratio 1.5
"""

import os
import sys
import glob
import argparse
import subprocess
import random
from typing import List, Tuple

# Turkce alfabe
TURKISH_ALPHABET = " abc\u00e7defg\u011fh\u0131ijklmno\u00f6prs\u015ftu\u00fcvyz"

def normalize_text(raw_text: str) -> str:
    TURKISH_UPPER_MAP = str.maketrans(
        "\u0130I\u00c7\u011e\u00d6\u015e\u00dc",
        "i\u0131\u00e7\u011f\u00f6\u015f\u00fc"
    )
    text = raw_text.strip().translate(TURKISH_UPPER_MAP).lower()
    text = " ".join(text.split())
    return "".join(ch for ch in text if ch in TURKISH_ALPHABET).strip()


def read_whisper_cache(txt_path: str) -> List[dict]:
    words = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                try:
                    words.append({
                        'start': float(parts[0]),
                        'end': float(parts[1]),
                        'text': parts[2],
                    })
                except ValueError:
                    continue
    return words


def words_to_segments(
    words: List[dict],
    min_sure: float = 1.5,
    max_sure: float = 5.0,
    max_bosluk: float = 0.8,
    min_kelime: int = 2,
    fps: float = 25.0,
    min_ctc_ratio: float = 1.5,
) -> List[Tuple[float, float, str]]:
    if not words:
        return []

    segments = []
    gs, ge, gw = words[0]['start'], words[0]['end'], [words[0]['text']]

    def flush():
        text = " ".join(gw)
        norm = normalize_text(text)
        sure = ge - gs
        if sure < min_sure or len(gw) < min_kelime or not norm:
            return
        n_frames = sure * fps
        n_chars = len(norm)
        if n_chars > 0 and n_frames / n_chars >= min_ctc_ratio:
            segments.append((gs, ge, norm))

    for w in words[1:]:
        bosluk = w['start'] - ge
        bsure = w['end'] - gs
        if bosluk > max_bosluk or bsure > max_sure:
            flush()
            gs, ge, gw = w['start'], w['end'], [w['text']]
        else:
            ge = w['end']
            gw.append(w['text'])
    flush()
    return segments


def cut_clip(video_path: str, start: float, end: float, output_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(end - start),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def create_splits(output_dir: str, train_ratio: float = 0.85):
    """
    Splits olusturur/gunceller.
    - Eski splits'teki ID'leri KORUR
    - Yeni clip+text eslesen ID'leri EKLER
    - Sadece clip VE text'i IKISI DE silinmis olanlari cikarir
    """
    clips_dir = os.path.join(output_dir, "clips")
    texts_dir = os.path.join(output_dir, "texts")
    rois_dir = os.path.join(output_dir, "rois_mp4")
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    train_path = os.path.join(splits_dir, "train.txt")
    val_path = os.path.join(splits_dir, "val.txt")

    # 1. Eski splits oku
    eski_train = set()
    eski_val = set()
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            eski_train = set(l.strip() for l in f if l.strip())
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            eski_val = set(l.strip() for l in f if l.strip())

    eski_toplam = len(eski_train) + len(eski_val)
    print(f"  Eski splits: train={len(eski_train)} val={len(eski_val)} toplam={eski_toplam}")

    # 2. Disk'teki text ID'leri
    text_ids = set(f[:-4] for f in os.listdir(texts_dir) if f.endswith('.txt')) if os.path.isdir(texts_dir) else set()
    disk_valid = text_ids

    # 3. Eski splits'ten SADECE text'i silinmis olanlari cikar
    tamamen_silinmis = (eski_train | eski_val) - text_ids
    if tamamen_silinmis:
        print(f"  Disk'te yok, splits'ten cikariliyor: {len(tamamen_silinmis)}")
        eski_train -= tamamen_silinmis
        eski_val -= tamamen_silinmis

    # 4. Yeni ID'ler (disk'te var ama splits'te yok)
    yeni_ids = sorted(disk_valid - eski_train - eski_val)

    if yeni_ids:
        random.seed(42)
        random.shuffle(yeni_ids)
        split_idx = int(len(yeni_ids) * train_ratio)
        yeni_train = set(yeni_ids[:split_idx])
        yeni_val = set(yeni_ids[split_idx:])

        eski_train |= yeni_train
        eski_val |= yeni_val
        print(f"  Yeni eklenen: {len(yeni_ids)} ({len(yeni_train)} train + {len(yeni_val)} val)")
    else:
        print(f"  Yeni ID yok")

    # 5. Yaz
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(eski_train)) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(eski_val)) + "\n")

    print(f"  Sonuc: train={len(eski_train)} | val={len(eski_val)} | toplam={len(eski_train) + len(eski_val)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adim 2: Segmentasyon + Klip Kesme")
    parser.add_argument("--video-dir", default="videolar")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--min-sure", type=float, default=1.5)
    parser.add_argument("--max-sure", type=float, default=5.0)
    parser.add_argument("--min-ctc-ratio", type=float, default=1.5)
    args = parser.parse_args()

    clips_dir = os.path.join(args.output_dir, "clips")
    texts_dir = os.path.join(args.output_dir, "texts")
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)

    # Whisper cache dosyalarini bul
    whisper_files = sorted(glob.glob(os.path.join(args.output_dir, "_whisper_*.txt")))
    if not whisper_files:
        print("Hic whisper transkript bulunamadi!")
        print("Once calistir: python adim1_transkript.py")
        sys.exit(1)

    print(f"Segment: {args.min_sure}-{args.max_sure}s | CTC ratio >= {args.min_ctc_ratio}")
    print(f"Transkript sayisi: {len(whisper_files)}\n")

    toplam_segment = 0
    toplam_klip = 0

    for wf in whisper_files:
        # Video adini bul: _whisper_test_videosu211.txt -> test_videosu211
        video_name = os.path.basename(wf).replace("_whisper_", "").replace(".txt", "")
        video_path = os.path.join(args.video_dir, f"{video_name}.mp4")

        if not os.path.exists(video_path):
            print(f"{video_name}: Video bulunamadi ({video_path}), ATLANIYOR")
            continue

        print(f"{video_name}:")

        # Whisper cache oku
        words = read_whisper_cache(wf)
        print(f"  {len(words)} kelime")

        # Segmentasyon
        segments = words_to_segments(
            words, min_sure=args.min_sure, max_sure=args.max_sure,
            min_ctc_ratio=args.min_ctc_ratio,
        )
        print(f"  {len(segments)} segment")
        toplam_segment += len(segments)

        # Klip kes + text kaydet
        video_klip = 0
        for i, (start, end, text) in enumerate(segments):
            sample_id = f"{video_name}_{i:04d}"
            clip_path = os.path.join(clips_dir, f"{sample_id}.mp4")
            text_path = os.path.join(texts_dir, f"{sample_id}.txt")

            # Zaten varsa atla
            if os.path.exists(clip_path) and os.path.exists(text_path):
                video_klip += 1
                continue

            if cut_clip(video_path, start, end, clip_path):
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)
                video_klip += 1
            else:
                print(f"    HATA: {sample_id}")

            # Ilerleme
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(segments)} klip kesildi...", flush=True)

        print(f"  {video_klip} klip kaydedildi")
        toplam_klip += video_klip

    # Splits olustur
    print(f"\nToplam: {toplam_segment} segment, {toplam_klip} klip")
    create_splits(args.output_dir)

    print("\nAdim 2 tamamlandi!")
    print("Sonraki: python NEW/extract_rois.py")
