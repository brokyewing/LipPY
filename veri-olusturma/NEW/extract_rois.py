"""
ROI Extraction Pipeline - Auto-AVSR Uyumlu (MAVI EKRAN FIX)
Her videoyu AYRI PROCESS'te isler — bellek sizintisi olmaz.

Kullanim:
    python NEW/extract_rois.py
    python NEW/extract_rois.py --limit 100
    python NEW/extract_rois.py --skip-frontal
    python NEW/extract_rois.py --single dataset/clips/video.mp4 dataset/rois_mp4/video.mp4
"""

import os
import sys
import argparse
import glob
import subprocess
import json
from tqdm import tqdm


def process_single_video(source_path, target_path, max_angle_ratio=1.25, skip_frontal=False):
    """
    Tek bir videoyu AYRI PYTHON PROCESS'te isle.
    Process bitince TUM bellek OS tarafindan geri alinir — leak IMKANSIZ.
    """
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single", source_path, target_path,
        "--max-angle-ratio", str(max_angle_ratio),
    ]
    if skip_frontal:
        cmd.append("--skip-frontal")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            encoding='utf-8', errors='replace'
        )
        # Sonucu parse et
        output = result.stdout.strip()
        if result.returncode == 0 and "BASARILI" in output:
            return "basarili", ""
        elif "YANDAN" in output:
            return "yandan", output.split("YANDAN:")[-1].strip() if "YANDAN:" in output else ""
        elif "ATLANDI" in output:
            return "atlandi", output.split("ATLANDI:")[-1].strip() if "ATLANDI:" in output else ""
        else:
            err = result.stderr.strip()[-200:] if result.stderr else output[-200:]
            return "hata", err
    except subprocess.TimeoutExpired:
        return "hata", "timeout (120s)"
    except Exception as e:
        return "hata", str(e)


def run_single_mode(source_path, target_path, max_angle_ratio, skip_frontal):
    """
    Tek video isleme modu — subprocess olarak cagirilir.
    Bu fonksiyon kendi process'inde calisir, bitince bellek tamamen serbest.
    """
    import gc
    import cv2
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["LIPPY_SUBPROCESS"] = "1"  # FAN GPU subprocess'te calismaz, MediaPipe kullan

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detector import LandmarksDetector
    from video_process import VideoProcess

    # Frontal check
    if not skip_frontal:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(source_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release(); face_mesh.close()
            print("ATLANDI: bos video"); return

        ratios = []
        for idx in [0, total // 2, max(0, total - 1)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks: continue
            lm = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            nose = np.array([lm[1].x * w, lm[1].y * h])
            le = np.array([lm[133].x * w, lm[133].y * h])
            re = np.array([lm[362].x * w, lm[362].y * h])
            dl, dr = np.linalg.norm(nose - le), np.linalg.norm(nose - re)
            if dl > 1e-6 and dr > 1e-6: ratios.append(dl / dr)
        cap.release(); face_mesh.close()

        if ratios:
            avg = float(np.mean(ratios))
            min_r = 1.0 / max_angle_ratio
            if not (min_r <= avg <= max_angle_ratio):
                yon = "SOL" if avg < 1.0 else "SAG"
                print(f"YANDAN: {yon} oran={avg:.2f}"); return

    # Video oku
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        print("ATLANDI: bos video"); return

    video_numpy = np.array(frames, dtype=np.uint8)
    del frames

    # Landmark tespit
    detector = LandmarksDetector()
    landmarks = detector.detect(video_numpy)

    if landmarks is None or all(lm is None for lm in landmarks):
        print("ATLANDI: yuz bulunamadi"); return

    # ROI cikar
    video_processor = VideoProcess(convert_gray=True)
    roi_sequence = video_processor(video_numpy, landmarks)
    del video_numpy, landmarks

    if roi_sequence is None:
        print("ATLANDI: roi bos"); return

    # Kaydet (grayscale -> 3 kanal MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = roi_sequence.shape[1:3]
    out = cv2.VideoWriter(target_path, fourcc, int(fps), (w, h))
    if not out.isOpened():
        print("HATA: videowriter acilamadi"); return
    for frame in roi_sequence:
        rgb = cv2.merge([frame, frame, frame])
        out.write(np.ascontiguousarray(rgb, dtype=np.uint8))
    out.release()

    print("BASARILI")


def main():
    parser = argparse.ArgumentParser(description="ROI cikarma (bellek guvenli)")
    parser.add_argument("--source", default="dataset/clips")
    parser.add_argument("--target", default="dataset/rois_mp4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-angle-ratio", type=float, default=1.25)
    parser.add_argument("--skip-frontal", action="store_true")
    parser.add_argument("--single", nargs=2, metavar=("SRC", "DST"),
                        help="Tek video isle (subprocess modu)")
    args = parser.parse_args()

    # Subprocess modu — tek video isle ve cik
    if args.single:
        run_single_mode(args.single[0], args.single[1],
                        args.max_angle_ratio, args.skip_frontal)
        return

    # Ana mod — tum videolari isle (her biri ayri subprocess)
    os.makedirs(args.target, exist_ok=True)
    mp4_files = sorted(glob.glob(os.path.join(args.source, "*.mp4")))
    if args.limit:
        mp4_files = mp4_files[:args.limit]

    # Yandan reddedilenler listesi (tekrar kontrol etmemek icin)
    skip_file = os.path.join(args.target, "_skipped.txt")
    skipped_set = set()
    if os.path.exists(skip_file):
        with open(skip_file, "r") as f:
            skipped_set = set(l.strip() for l in f if l.strip())

    print(f"[{len(mp4_files)}] video bulundu.")
    print(f"Onceden reddedilen: {len(skipped_set)}")
    if not args.skip_frontal:
        print(f"Yuz acisi filtresi: {1/args.max_angle_ratio:.2f} - {args.max_angle_ratio:.2f}")
    print(f"Her video AYRI PROCESS'te islenecek (bellek guvenli)\n")

    basarili, atlanan, hata, yandan = 0, 0, 0, 0

    for file_path in tqdm(mp4_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(args.target, filename)

        # Zaten ROI varsa atla
        if os.path.exists(out_path):
            atlanan += 1
            continue

        # Daha once reddedilmisse atla (yandan/bos/yuz yok)
        if filename in skipped_set:
            atlanan += 1
            continue

        status, msg = process_single_video(
            file_path, out_path, args.max_angle_ratio, args.skip_frontal)

        if status == "basarili":
            basarili += 1
        elif status == "yandan":
            yandan += 1
            skipped_set.add(filename)
            with open(skip_file, "a") as f: f.write(filename + "\n")
            tqdm.write(f"YANDAN: {filename} {msg}")
        elif status == "atlandi":
            atlanan += 1
            skipped_set.add(filename)
            with open(skip_file, "a") as f: f.write(filename + "\n")
        else:
            hata += 1
            tqdm.write(f"HATA: {filename} {msg}")

    print(f"\nTamamlandi: {basarili} basarili, {atlanan} atlandi, {yandan} yandan, {hata} hata")
    print(f"Reddedilenler kaydedildi: {skip_file} ({len(skipped_set)} dosya)")


if __name__ == '__main__':
    main()
