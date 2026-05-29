"""
RetinaFace + FAN Landmark Detector (Auto-AVSR Orijinal)
GPU uzerinde calisir, 68 landmark doner.

Kurulum:
    git clone https://github.com/hhj1897/face_detection.git
    cd face_detection && git lfs pull && pip install -e . && cd ..
    git clone https://github.com/hhj1897/face_alignment.git
    cd face_alignment && pip install -e . && cd ..

MediaPipe yedek olarak kalir — ibug yuklu degilse otomatik MediaPipe'a duser.
"""

import os
import cv2
import numpy as np

# Hangi backend kullanilacak
_BACKEND = None  # 'fan' veya 'mediapipe'


def _detect_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    # Subprocess modunda GPU paylasimi sorunlu — FAN sadece tek process'te calisir
    # SUBPROCESS_MODE env degiskeni varsa MediaPipe kullan
    if os.environ.get("LIPPY_SUBPROCESS") == "1":
        _BACKEND = 'mediapipe'
        return _BACKEND
    try:
        from ibug.face_detection import RetinaFacePredictor
        from ibug.face_alignment import FANPredictor
        _BACKEND = 'fan'
        print("[Detector] RetinaFace + FAN (GPU, 68 landmark)")
    except ImportError:
        _BACKEND = 'mediapipe'
        print("[Detector] MediaPipe FaceMesh (CPU, 4 landmark)")
    return _BACKEND


class LandmarksDetector:
    def __init__(self, device="cuda:0"):
        backend = _detect_backend()

        if backend == 'fan':
            from ibug.face_detection import RetinaFacePredictor
            from ibug.face_alignment import FANPredictor
            self.face_detector = RetinaFacePredictor(
                device=device, threshold=0.8,
                model=RetinaFacePredictor.get_model('resnet50')
            )
            self.landmark_detector = FANPredictor(device=device, model=None)
            self.detect = self._detect_fan
        else:
            os.environ["GLOG_minloglevel"] = "2"
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.detect = self._detect_mediapipe

    def __call__(self, filename):
        cap = cv2.VideoCapture(filename)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames: return None
        return self.detect(np.array(frames, dtype=np.uint8))

    def _detect_fan(self, video_frames):
        """RetinaFace + FAN: 68 landmark, GPU."""
        landmarks = []
        for idx, frame in enumerate(video_frames):
            # RetinaFace BGR bekler
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            detected_faces = self.face_detector(frame_bgr, rgb=False)

            if len(detected_faces) == 0:
                landmarks.append(None)
                continue

            # En buyuk yuzu sec
            max_id, max_size = 0, 0
            for i, bbox in enumerate(detected_faces):
                size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if size > max_size:
                    max_id, max_size = i, size

            # FAN landmark (68 nokta)
            face_points, _ = self.landmark_detector(frame_bgr, detected_faces, rgb=False)
            landmarks.append(face_points[max_id])  # (68, 2)

        return landmarks

    def _detect_mediapipe(self, video_frames):
        """MediaPipe FaceMesh: 4 landmark, CPU (yedek)."""
        landmarks = []
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

        return landmarks
