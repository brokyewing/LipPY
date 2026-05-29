"""
Veri tutarlilik kontrolu + bozuk MP4 + CTC ratio temizleme.

Kontroller:
  1. Text + ROI + Splits eslesmesi
  2. Bozuk MP4 tespiti (moov atom, acilamayan, 0 frame)
  3. Bos text dosyalari
  4. CTC ratio kontrolu (frame/karakter < 1.5 → egitime uygun degil)

Kullanim:
  python veri_temizleyici1.py                     # Sadece rapor
  python veri_temizleyici1.py --uygula            # Sorunlu ornekleri sil
  python veri_temizleyici1.py --splits-olustur    # text+roi eslesenlerden splits olustur
"""

import os
import argparse
import random
import cv2
import numpy as np
from typing import Set
from tqdm import tqdm

BASE = "dataset"
TEXT_DIR = os.path.join(BASE, "texts")
ROIS_DIR = os.path.join(BASE, "rois_mp4")
SPLIT_DIR = os.path.join(BASE, "splits")
TRAIN_FILE = os.path.join(SPLIT_DIR, "train.txt")
VAL_FILE = os.path.join(SPLIT_DIR, "val.txt")

# Turkce alfabe (CTC ratio hesabi icin)
CHARS = " abc\u00e7defg\u011fh\u0131ijklmno\u00f6prs\u015ftu\u00fcvyz"
CHAR_SET = set(CHARS)


def text_to_len(text):
    """Turkce karaktere donusen karakter sayisi."""
    return sum(1 for c in text.lower().strip() if c in CHAR_SET)


def dosyalardan_id_oku(klasor, uzanti):
    if not os.path.isdir(klasor):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(klasor) if f.endswith(uzanti)}


def splits_id_oku():
    ids = set()
    for dosya in (TRAIN_FILE, VAL_FILE):
        if not os.path.exists(dosya): continue
        with open(dosya, "r", encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid: ids.add(sid)
    return ids


def bozuk_mp4_bul():
    if not os.path.isdir(ROIS_DIR):
        return set()
    bozuk = set()
    mp4s = [f for f in os.listdir(ROIS_DIR) if f.endswith('.mp4')]
    print(f"\n  Bozuk MP4 taraniyor ({len(mp4s)} dosya)...")
    for f in tqdm(mp4s, desc="  MP4 kontrol"):
        sid = f[:-4]
        path = os.path.join(ROIS_DIR, f)
        try:
            if os.path.getsize(path) < 100:
                bozuk.add(sid); continue
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                bozuk.add(sid); continue
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fc < 1:
                cap.release(); bozuk.add(sid); continue
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                bozuk.add(sid)
        except:
            bozuk.add(sid)
    return bozuk


def bos_text_bul():
    if not os.path.isdir(TEXT_DIR):
        return set()
    bos = set()
    for f in os.listdir(TEXT_DIR):
        if not f.endswith('.txt'): continue
        sid = f[:-4]
        try:
            with open(os.path.join(TEXT_DIR, f), 'r', encoding='utf-8') as fh:
                text = fh.read().strip()
            if len(text) < 2:
                bos.add(sid)
        except:
            bos.add(sid)
    return bos


def titreme_ve_bozuk_roi_bul(min_std=3.0, max_frame_diff=80.0):
    """
    ROI kalite kontrolu:
    1. Titreme: ardisik frame'ler arasi piksel farki cok yuksek (kamera/yuz atlama)
    2. Bozuk ROI: std cok dusuk (siyah/gri/sabit goruntu)
    3. Eksik dudak: frame'lerin cogu cok karanlik veya cok acik
    """
    if not os.path.isdir(ROIS_DIR):
        return set()

    sorunlu = set()
    mp4s = [f for f in os.listdir(ROIS_DIR) if f.endswith('.mp4')]
    print(f"\n  ROI kalite taraniyor ({len(mp4s)} dosya)...")

    for f in tqdm(mp4s, desc="  ROI kalite"):
        sid = f[:-4]
        path = os.path.join(ROIS_DIR, f)
        try:
            cap = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            cap.release()

            if len(frames) < 5:
                sorunlu.add(sid); continue

            arr = [f.astype(float) for f in frames]

            # Genel std: cok dusukse sabit goruntu (dudak yok)
            all_pixels = [f.mean() for f in arr]
            overall_std = float(np.std(all_pixels)) if len(all_pixels) > 1 else 0
            if overall_std < 1.0:
                sorunlu.add(sid); continue

            # Frame-arasi fark: titreme tespiti
            diffs = []
            for i in range(1, min(len(arr), 50)):  # ilk 50 frame yeterli
                diff = float(np.mean(np.abs(np.array(arr[i]) - np.array(arr[i-1]))))
                diffs.append(diff)

            if diffs:
                avg_diff = sum(diffs) / len(diffs)
                max_diff = max(diffs)
                # Ortalama fark cok yuksek = titreme/atlama
                if avg_diff > max_frame_diff:
                    sorunlu.add(sid); continue
                # Tek frame'de buyuk sıçrama = sahne degisimi
                if max_diff > max_frame_diff * 3:
                    sorunlu.add(sid); continue

            # Cok karanlik veya cok acik frame orani
            dark_count = sum(1 for f in arr if np.mean(f) < 30)
            bright_count = sum(1 for f in arr if np.mean(f) > 230)
            bad_ratio = (dark_count + bright_count) / len(arr)
            if bad_ratio > 0.3:
                sorunlu.add(sid)

        except:
            sorunlu.add(sid)

    return sorunlu


def ctc_ratio_dusuk_bul(min_ratio=1.5):
    """ROI frame sayisi / text karakter sayisi < min_ratio olanlari bul."""
    if not os.path.isdir(ROIS_DIR) or not os.path.isdir(TEXT_DIR):
        return set()

    roi_ids = dosyalardan_id_oku(ROIS_DIR, ".mp4")
    text_ids = dosyalardan_id_oku(TEXT_DIR, ".txt")
    ortak = sorted(roi_ids & text_ids)

    dusuk = set()
    print(f"\n  CTC ratio taraniyor ({len(ortak)} dosya, esik={min_ratio})...")
    for sid in tqdm(ortak, desc="  CTC ratio"):
        roi_path = os.path.join(ROIS_DIR, f"{sid}.mp4")
        txt_path = os.path.join(TEXT_DIR, f"{sid}.txt")
        try:
            cap = cv2.VideoCapture(roi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n_frames < 5:
                dusuk.add(sid); continue

            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            n_chars = text_to_len(text)
            if n_chars == 0:
                dusuk.add(sid); continue

            ratio = n_frames / n_chars
            if ratio < min_ratio:
                dusuk.add(sid)
        except:
            dusuk.add(sid)
    return dusuk


def tutarsiz_idleri_bul():
    text_ids = dosyalardan_id_oku(TEXT_DIR, ".txt")
    roi_ids = dosyalardan_id_oku(ROIS_DIR, ".mp4")
    split_ids = splits_id_oku()
    tum = text_ids | roi_ids | split_ids
    tam = text_ids & roi_ids & split_ids
    return tum - tam


def rapor_yazdir(tutarsiz, bozuk_mp4, bos_txt, ctc_dusuk, roi_bozuk=None):
    if roi_bozuk is None: roi_bozuk = set()
    text_ids = dosyalardan_id_oku(TEXT_DIR, ".txt")
    roi_ids = dosyalardan_id_oku(ROIS_DIR, ".mp4")
    split_ids = splits_id_oku()
    tam = text_ids & roi_ids & split_ids
    tum_sorunlu = tutarsiz | bozuk_mp4 | bos_txt | ctc_dusuk | roi_bozuk

    print(f"\n{'='*60}")
    print(f"  DOSYA SAYILARI")
    print(f"{'='*60}")
    print(f"  texts/     (.txt)  : {len(text_ids)}")
    print(f"  rois_mp4/  (.mp4)  : {len(roi_ids)}")
    print(f"  splits (train+val) : {len(split_ids)}")
    print(f"\n  Tam tutarli (3/3)  : {len(tam)}")
    print(f"  Tutarsiz           : {len(tutarsiz)}")
    print(f"  Bozuk MP4          : {len(bozuk_mp4)}")
    print(f"  Bos text           : {len(bos_txt)}")
    print(f"  CTC ratio dusuk    : {len(ctc_dusuk)}")
    print(f"  ROI bozuk/titreme  : {len(roi_bozuk)}")
    print(f"  ─────────────────────")
    print(f"  TOPLAM SORUNLU     : {len(tum_sorunlu)}")
    print(f"  TEMIZ              : {len(tam) - len(tam & tum_sorunlu)}")


def sorunlu_ornekleri_sil(silinecek):
    silinen = 0
    for sid in silinecek:
        for dir_path, ext in [(TEXT_DIR, ".txt"), (ROIS_DIR, ".mp4")]:
            path = os.path.join(dir_path, f"{sid}{ext}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    silinen += 1
                except Exception as e:
                    print(f"  Hata: {path} -> {e}")

    for split_name in ("train", "val"):
        dosya = os.path.join(SPLIT_DIR, f"{split_name}.txt")
        if not os.path.exists(dosya): continue
        with open(dosya, "r", encoding="utf-8") as f:
            satirlar = [line.strip() for line in f if line.strip()]
        gorulen = set()
        yeni = []
        for s in satirlar:
            if s not in silinecek and s not in gorulen:
                yeni.append(s); gorulen.add(s)
        with open(dosya, "w", encoding="utf-8") as f:
            for s in yeni: f.write(s + "\n")
        print(f"  {split_name}.txt: {len(satirlar)} -> {len(yeni)}")

    print(f"  Toplam {silinen} dosya silindi")


def splits_olustur(train_ratio=0.85):
    text_ids = dosyalardan_id_oku(TEXT_DIR, ".txt")
    roi_ids = dosyalardan_id_oku(ROIS_DIR, ".mp4")
    valid = sorted(text_ids & roi_ids)
    if not valid:
        print("Gecerli ornek yok!")
        return
    random.seed(42)
    random.shuffle(valid)
    split = int(len(valid) * train_ratio)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(valid[:split])) + "\n")
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(valid[split:])) + "\n")
    print(f"Splits: train={split} | val={len(valid)-split} | toplam={len(valid)}")


def main():
    global BASE, TEXT_DIR, ROIS_DIR, SPLIT_DIR, TRAIN_FILE, VAL_FILE

    parser = argparse.ArgumentParser(description="Veri temizleme (tutarlilik + bozuk + CTC ratio)")
    parser.add_argument("--uygula", action="store_true", help="Sorunlu ornekleri sil")
    parser.add_argument("--splits-olustur", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--base-data-dir", type=str, default=BASE)
    parser.add_argument("--skip-mp4-check", action="store_true")
    parser.add_argument("--skip-ctc-check", action="store_true")
    parser.add_argument("--skip-roi-check", action="store_true", help="ROI titreme/kalite kontrol atlama")
    parser.add_argument("--min-ctc-ratio", type=float, default=1.5)
    args = parser.parse_args()

    BASE = args.base_data_dir
    TEXT_DIR = os.path.join(BASE, "texts")
    ROIS_DIR = os.path.join(BASE, "rois_mp4")
    SPLIT_DIR = os.path.join(BASE, "splits")
    TRAIN_FILE = os.path.join(SPLIT_DIR, "train.txt")
    VAL_FILE = os.path.join(SPLIT_DIR, "val.txt")

    print(f"Dataset: {BASE}")

    if args.splits_olustur:
        splits_olustur(args.train_ratio)
        return

    total_steps = 2 + (0 if args.skip_mp4_check else 1) + (0 if args.skip_ctc_check else 1) + (0 if args.skip_roi_check else 1)
    step = 0

    # 1. Eslesmeme
    step += 1
    print(f"\n[{step}/{total_steps}] Eslesmeme kontrolu...")
    tutarsiz = tutarsiz_idleri_bul()

    # 2. Bozuk MP4
    bozuk_mp4 = set()
    if not args.skip_mp4_check:
        step += 1
        print(f"\n[{step}/{total_steps}] Bozuk MP4 kontrolu...")
        bozuk_mp4 = bozuk_mp4_bul()

    # 3. Bos text
    step += 1
    print(f"\n[{step}/{total_steps}] Bos text kontrolu...")
    bos_txt = bos_text_bul()

    # 4. CTC ratio
    ctc_dusuk = set()
    if not args.skip_ctc_check:
        step += 1
        print(f"\n[{step}/{total_steps}] CTC ratio kontrolu...")
        ctc_dusuk = ctc_ratio_dusuk_bul(args.min_ctc_ratio)

    # 5. ROI kalite (titreme + bozuk dudak)
    roi_bozuk = set()
    if not args.skip_roi_check:
        step += 1
        print(f"\n[{step}/{total_steps}] ROI kalite kontrolu (titreme/eksik dudak)...")
        roi_bozuk = titreme_ve_bozuk_roi_bul()

    rapor_yazdir(tutarsiz, bozuk_mp4, bos_txt, ctc_dusuk, roi_bozuk)

    tum_sorunlu = tutarsiz | bozuk_mp4 | bos_txt | ctc_dusuk | roi_bozuk

    if tum_sorunlu and args.uygula:
        print(f"\n{len(tum_sorunlu)} sorunlu ornek temizleniyor...")
        sorunlu_ornekleri_sil(tum_sorunlu)
        print("Tamamlandi.")
    elif tum_sorunlu:
        print(f"\nSilmek icin: python veri_temizleyici1.py --uygula")


if __name__ == "__main__":
    main()
