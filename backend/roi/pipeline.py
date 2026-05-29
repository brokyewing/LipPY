"""Video → ROI array wrapper.

Eğitim pipeline'ı (veri-olusturma/NEW/extract_rois.py) ile birebir aynı sonucu üretir:
    MediaPipe FaceMesh (4 landmark) → VideoProcess affine align
    → 256×256 warped frame → 96×96 grayscale dudak patch

MediaPipe varsayılandır (eğitim verisi onunla çıkarıldı). FAN (68 landmark)
sadece deneysel: LIPPY_USE_FAN=1 — ROI hizalaması farklı olur, uyumsuzluk riski.

Singleton pattern: detector ve processor bir kere yüklenir, cache'lenir.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .detector import LandmarksDetector
from .video_process import VideoProcess


# Singleton cache — ilk inference'ta yükle, sonra yeniden kullan
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


# Eğitim videoları 25 fps. Farklı FPS'li videolar otomatik resample edilir
# (60 fps video → 25 fps temporal pattern, model'in beklediği gibi).
TARGET_FPS = 25.0
FPS_TOLERANCE = 0.5  # 24.5-25.5 arasını "uyumlu" say


def _read_video_rgb(video_path: str) -> Optional[np.ndarray]:
    """Video dosyasını RGB frame dizisine oku — 25 FPS'e otomatik resample.

    Eğitim ile uyumluluk için tüm videolar 25 fps'e dönüştürülür.
    60 fps bir video → her ~2.4 frame'den 1'i seçilir (uniform sampling).

    Returns:
        (T, H, W, 3) uint8 veya None (dosya açılamazsa / boşsa)
    """
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

    # FPS uyumlu mu?
    if abs(src_fps - TARGET_FPS) <= FPS_TOLERANCE:
        print(f"[Pipeline] Video FPS={src_fps:.1f} (uyumlu, {len(frames)} frame)")
        return np.asarray(frames, dtype=np.uint8)

    # FPS resample — uniform sampling
    src_total = len(frames)
    target_total = max(1, int(round(src_total * TARGET_FPS / src_fps)))
    target_indices = np.linspace(0, src_total - 1, target_total).astype(int)
    resampled = [frames[i] for i in target_indices]
    print(f"[Pipeline] Video FPS={src_fps:.1f} → resample {src_total} → {target_total} frame ({TARGET_FPS} fps)")
    return np.asarray(resampled, dtype=np.uint8)


def video_to_roi_array(video_path: str) -> Optional[np.ndarray]:
    """Video dosyasından dudak ROI dizisi çıkar.

    Args:
        video_path: .mp4/.webm/.avi/... video dosya yolu

    Returns:
        (T, 96, 96) uint8 grayscale ROI dizisi, veya None:
            - video okunamazsa
            - hiçbir karede yüz bulunamazsa
            - affine align başarısızsa (landmark > %30 kayıp)
    """
    video_rgb = _read_video_rgb(video_path)
    if video_rgb is None:
        return None

    detector, processor = get_pipeline()

    # detector.detect(frames: np.ndarray (T,H,W,3)) → List[np.ndarray(68,2) | None]
    landmarks = detector.detect(video_rgb)
    if landmarks is None or all(lm is None for lm in landmarks):
        return None

    # processor(video, landmarks) → (T, 96, 96) uint8 (convert_gray=True) veya None
    roi_sequence = processor(video_rgb, landmarks)
    return roi_sequence  # zaten None olabilir processor tarafından
