"""
MediaPipe FaceMesh Landmark Detector — TRAINING ILE UYUMLU.

ÖNEMLİ — Training-Inference Uyumluluğu:
    Bizim training data'sı MediaPipe FaceMesh (4 landmark: göz×2, burun, ağız) ile
    çıkarıldı. Inference'da FAN (68 landmark) kullanırsak ROI hizalaması farklı
    olur (farklı affine matrix), model kötü tahmin yapar.

    KESINLIKLE MediaPipe kullanılmalı — FAN deneysel olarak env değişkeniyle
    açılabilir ama varsayılan değildir.

Kurulum:
    pip install mediapipe==0.10.14

Opsiyonel (deneysel, FAN ile farklı eğitim için):
    git clone https://github.com/hhj1897/face_detection.git
    cd face_detection && git lfs pull && pip install -e . && cd ..
    git clone https://github.com/hhj1897/face_alignment.git
    cd face_alignment && pip install -e . && cd ..

    Env variable: LIPPY_USE_FAN=1 → FAN kullanır (training UYUMSUZLUK riski!)
"""

import os
import time
import cv2
import numpy as np

# Hangi backend kullanilacak
_BACKEND = None  # 'mediapipe' (default) veya 'fan' (deneysel)


def _detect_backend():
    """Default: MediaPipe (training ile uyumlu).
    FAN sadece LIPPY_USE_FAN=1 env variable ile aktiflesir.
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    # FAN sadece bilincli secim (training UYUMSUZLUK riski)
    if os.environ.get("LIPPY_USE_FAN") == "1":
        try:
            from ibug.face_detection import RetinaFacePredictor
            from ibug.face_alignment import FANPredictor
            _BACKEND = 'fan'
            print("[Detector] !!! UYARI: FAN aktif (LIPPY_USE_FAN=1) — training MediaPipe ile yapildi, ROI uyumsuzluk olabilir")
            return _BACKEND
        except ImportError:
            print("[Detector] FAN istendi ama ibug yuklu degil, MediaPipe'a duser")

    # Default: MediaPipe (training pipeline ile uyumlu)
    try:
        import mediapipe as mp
        _BACKEND = 'mediapipe'
        print("[Detector] MediaPipe FaceMesh (training pipeline ile UYUMLU, 4 landmark)")
        return _BACKEND
    except ImportError:
        # Mediapipe yoksa son care FAN
        try:
            from ibug.face_detection import RetinaFacePredictor
            from ibug.face_alignment import FANPredictor
            _BACKEND = 'fan'
            print("[Detector] !!! UYARI: MediaPipe yuklu degil, FAN fallback — UYUMSUZLUK riski!")
            print("[Detector]     pip install mediapipe==0.10.14 yapip backend'i tekrar baslat")
            return _BACKEND
        except ImportError:
            raise ImportError(
                "Hicbir landmark detector yuklu degil!\n"
                "Tavsiye edilen: pip install mediapipe==0.10.14"
            )


class LandmarksDetector:
    def __init__(self, device="cuda:0"):
        backend = _detect_backend()

        if backend == 'mediapipe':
            os.environ["GLOG_minloglevel"] = "2"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.detect = self._detect_mediapipe
        else:  # fan (deneysel)
            from ibug.face_detection import RetinaFacePredictor
            from ibug.face_alignment import FANPredictor
            # MobileNet 0.25 — ResNet50'den 5x hizli (~30ms vs 150ms/frame)
            self.face_detector = RetinaFacePredictor(
                device=device, threshold=0.8,
                model=RetinaFacePredictor.get_model('mobilenet0.25')
            )
            self.landmark_detector = FANPredictor(device=device, model=None)
            self.detect = self._detect_fan

    def __call__(self, filename):
        """Eski API uyumlu (filename veya np.ndarray)."""
        if isinstance(filename, str):
            cap = cv2.VideoCapture(filename)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames: return None
            return self.detect(np.array(frames, dtype=np.uint8))
        else:
            # np.ndarray (T, H, W, 3) RGB
            return self.detect(filename)

    def _detect_mediapipe(self, video_frames):
        """MediaPipe FaceMesh: 4 landmark, CPU.
        TRAINING ILE UYUMLU PIPELINE — kullanilmasi sart.

        Returns: list[np.ndarray(4, 2) | None] — her frame icin 4 landmark
            [right_eye, left_eye, nose, mouth] sirasinda.
        """
        landmarks = []
        t0 = time.perf_counter()
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.5
        ) as face_mesh:
            for idx, frame in enumerate(video_frames):
                frame_contig = np.ascontiguousarray(frame, dtype=np.uint8)
                ih, iw, _ = frame_contig.shape
                results = face_mesh.process(frame_contig)

                if not results or not results.multi_face_landmarks:
                    landmarks.append(None)
                    continue

                lm = results.multi_face_landmarks[0]
                # 4 landmark: right_eye, left_eye, nose, mouth
                right_eye_x = (lm.landmark[33].x + lm.landmark[133].x) / 2 * iw
                right_eye_y = (lm.landmark[33].y + lm.landmark[133].y) / 2 * ih
                left_eye_x = (lm.landmark[362].x + lm.landmark[263].x) / 2 * iw
                left_eye_y = (lm.landmark[362].y + lm.landmark[263].y) / 2 * ih
                nose_x = lm.landmark[4].x * iw
                nose_y = lm.landmark[4].y * ih
                mouth_x = (lm.landmark[13].x + lm.landmark[14].x) / 2 * iw
                mouth_y = (lm.landmark[13].y + lm.landmark[14].y) / 2 * ih

                landmarks.append(np.array([
                    [right_eye_x, right_eye_y],
                    [left_eye_x, left_eye_y],
                    [nose_x, nose_y],
                    [mouth_x, mouth_y]
                ]))

        valid = sum(1 for lm in landmarks if lm is not None)
        print(f"[Detector] MediaPipe {len(video_frames)} frame icin: {(time.perf_counter() - t0):.1f}s ({valid} valid)")
        return landmarks

    def _detect_fan(self, video_frames):
        """RetinaFace + FAN: 68 landmark, GPU. DENEYSEL — training uyumsuz!

        OPTIMIZE: Yuz tespit sadece ilk frame'de (cache).
        """
        if len(video_frames) == 0:
            return []

        # Ilk frame: yuz tespit
        first_bgr = cv2.cvtColor(video_frames[0], cv2.COLOR_RGB2BGR)
        detected_faces = self.face_detector(first_bgr, rgb=False)

        if len(detected_faces) == 0:
            print(f"[Detector] HATA: ilk frame'de yuz bulunamadi")
            return [None] * len(video_frames)

        # En buyuk yuzu sec
        max_id, max_size = 0, 0
        for i, bbox in enumerate(detected_faces):
            size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
            if size > max_size:
                max_id, max_size = i, size

        cached_bbox = detected_faces[max_id:max_id + 1]
        print(f"[Detector] Yuz cache'lendi (ilk frame), FAN her frame")

        landmarks = []
        t0 = time.perf_counter()
        for idx, frame in enumerate(video_frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            try:
                face_points, _ = self.landmark_detector(frame_bgr, cached_bbox, rgb=False)
                landmarks.append(face_points[0])  # (68, 2)
            except Exception:
                landmarks.append(None)
        print(f"[Detector] FAN {len(video_frames)} frame icin: {(time.perf_counter() - t0):.1f}s")

        return landmarks
