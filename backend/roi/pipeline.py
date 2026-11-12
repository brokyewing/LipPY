"""Video → ROI array wrapper.

Eğitim pipeline'ı (veri-olusturma/NEW/extract_rois.py) ile birebir aynı sonucu üretir:
    MediaPipe FaceMesh (4 landmark) → VideoProcess affine align
    → 256×256 warped frame → 96×96 grayscale dudak patch

Not: Eğer yüklenen video zaten eğitim veri setindeki gibi kırpılmış bir dudak ROI
videosu ise (örn. 88×88 veya 96×96 kare klip), doğrudan gri tonlamalı (T, 96, 96)
diziye dönüştürülür (FPS yeniden örneklemesi yapılmaz). Tam yüz videolarında ise
MediaPipe FaceMesh (veya OpenCV fallback) + affine align çalışır.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detector import LandmarksDetector
from .video_process import VideoProcess


_detector: Optional[LandmarksDetector] = None
_processor: Optional[VideoProcess] = None


def get_pipeline() -> tuple[LandmarksDetector, VideoProcess]:
    """Detector ve processor'ı lazy-load ile döndür."""
    global _detector, _processor
    if _detector is None:
        _detector = LandmarksDetector()
    if _processor is None:
        _processor = VideoProcess(convert_gray=True)
    return _detector, _processor


TARGET_FPS = 25.0
MAX_NATIVE_FPS = 35.0  # 24-30 fps videolar olduğu gibi korunur; >35 fps (60fps) indirgenir


def _read_video_rgb(video_path: str) -> Optional[np.ndarray]:
    """Video dosyasını RGB frame dizisine oku."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or not np.isfinite(src_fps):
        src_fps = TARGET_FPS

    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not frames:
        return None

    arr = np.asarray(frames, dtype=np.uint8)
    _, H, W, _ = arr.shape
    # Önceden kırpılmış ROI videolarında veya 24-35 fps standart videolarda tüm kareleri koru
    if (H <= 128 and W <= 128) or src_fps <= MAX_NATIVE_FPS:
        return arr

    # Yüksek FPS (örn. 60 fps) videoları 25 fps'e indirge
    src_total = len(frames)
    target_total = max(1, int(round(src_total * TARGET_FPS / src_fps)))
    target_indices = np.linspace(0, src_total - 1, target_total).astype(int)
    return arr[target_indices]


def _frames_to_direct_roi(video_rgb: np.ndarray) -> np.ndarray:
    """Önceden kırpılmış dudak ROI videolarını (örn. 96×96) doğrudan (T, 96, 96) uint8'e çevirir."""
    gray_frames = []
    for frame in video_rgb:
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if g.shape[0] != 96 or g.shape[1] != 96:
            g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_AREA)
        gray_frames.append(g)
    return np.asarray(gray_frames, dtype=np.uint8)


def video_to_roi_array(video_path: str) -> Optional[np.ndarray]:
    """Video dosyasından dudak ROI dizisi çıkar.

    Returns:
        (T, 96, 96) uint8 grayscale ROI dizisi, veya None
    """
    video_rgb = _read_video_rgb(video_path)
    if video_rgb is None:
        return None

    _, H, W, _ = video_rgb.shape
    if H <= 128 and W <= 128 and abs(H - W) <= 16:
        return _frames_to_direct_roi(video_rgb)

    try:
        detector, processor = get_pipeline()
        landmarks = detector.detect(video_rgb)
        if landmarks is not None and not all(lm is None for lm in landmarks):
            roi_sequence = processor(video_rgb, landmarks)
            if roi_sequence is not None:
                return roi_sequence
    except Exception as e:
        print(f"[Pipeline] ROI çıkarma uyarısı: {e}")

    return None
