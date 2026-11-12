#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# FAN (68 landmark) + MediaPipe (4 landmark) uyumlu

import os
import cv2
import numpy as np

try:
    from skimage import transform as tf
except ImportError:
    tf = None



def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx-start_idx):
        landmarks[start_idx+idx] = start_landmarks + idx/float(stop_idx-start_idx) * delta
    return landmarks


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)
    warped = (warped * 255).astype('uint8')
    return warped, tform


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = (warped * 255).astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)
    if abs(center_y - img.shape[0] / 2) > height + threshold:
        raise Exception('too much bias in height')
    if abs(center_x - img.shape[1] / 2) > width + threshold:
        raise Exception('too much bias in width')
    y_min = int(round(np.clip(center_y - height, 0, img.shape[0])))
    y_max = int(round(np.clip(center_y + height, 0, img.shape[0])))
    x_min = int(round(np.clip(center_x - width, 0, img.shape[1])))
    x_max = int(round(np.clip(center_x + width, 0, img.shape[1])))
    return np.copy(img[y_min:y_max, x_min:x_max])


def _detect_landmark_format(landmarks):
    """Landmark formatini otomatik tespit et: 68 (FAN) veya 4 (MediaPipe)."""
    for lm in landmarks:
        if lm is not None:
            n_points = lm.shape[0]
            if n_points >= 68:
                return 'fan'
            else:
                return 'mediapipe'
    return 'mediapipe'


class VideoProcess:
    def __init__(self, mean_face_path="20words_mean_face.npy", crop_width=96, crop_height=96,
                 start_idx=None, stop_idx=None, window_margin=12, convert_gray=True):
        self.reference = np.load(os.path.join(os.path.dirname(__file__), mean_face_path))
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.window_margin = window_margin
        self.convert_gray = convert_gray
        # start_idx/stop_idx otomatik ayarlanacak (landmark formatina gore)
        self._start_idx = start_idx
        self._stop_idx = stop_idx

    def __call__(self, video, landmarks):
        # Landmark formatini tespit et
        fmt = _detect_landmark_format(landmarks)
        if fmt == 'fan':
            # FAN 68 landmark: dudak indeksleri 48-68
            self.start_idx = self._start_idx if self._start_idx is not None else 48
            self.stop_idx = self._stop_idx if self._stop_idx is not None else 68
            self.stable_points = (28, 33, 36, 39, 42, 45, 48, 54)  # Auto-AVSR orijinal
        else:
            # MediaPipe 4 landmark: index 3 = agiz merkezi
            self.start_idx = self._start_idx if self._start_idx is not None else 3
            self.stop_idx = self._stop_idx if self._stop_idx is not None else 4
            self.stable_points = (0, 1, 2, 3)

        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        if not preprocessed_landmarks:
            return None
        if len(preprocessed_landmarks) < self.window_margin:
            return None
        sequence = self.crop_patch(video, preprocessed_landmarks)
        return sequence


    def crop_patch(self, video, landmarks):
        sequence = []
        failed_frames = 0
        last_good_patch = None

        for frame_idx, frame in enumerate(video):
            window_margin = min(self.window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = np.mean(
                [landmarks[x] for x in range(frame_idx - window_margin, frame_idx + window_margin + 1)], axis=0)
            smoothed_landmarks += landmarks[frame_idx].mean(axis=0) - smoothed_landmarks.mean(axis=0)
            transformed_frame, transformed_landmarks = self.affine_transform(
                frame, smoothed_landmarks, self.reference, grayscale=self.convert_gray)

            try:
                patch = cut_patch(
                    transformed_frame,
                    transformed_landmarks[self.start_idx:self.stop_idx],
                    self.crop_height // 2, self.crop_width // 2)
                last_good_patch = patch
                sequence.append(patch)
            except Exception:
                failed_frames += 1
                if last_good_patch is not None:
                    sequence.append(last_good_patch)
                else:
                    h, w = self.crop_height, self.crop_width
                    sequence.append(np.zeros((h, w), dtype=np.uint8) if self.convert_gray
                                    else np.zeros((h, w, 3), dtype=np.uint8))

        total_frames = len(video)
        fail_ratio = failed_frames / max(total_frames, 1)

        if fail_ratio > 0.3:
            return None

        return np.array(sequence)


    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if valid_frames_idx:
            landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks


    def affine_transform(self, frame, landmarks, reference, grayscale=False,
                         target_size=(256, 256), reference_size=(256, 256), stable_points=None,
                         interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=0):
        if stable_points is None:
            stable_points = self.stable_points
        if grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stable_reference = self.get_stable_reference(reference, reference_size, target_size, stable_points)
        transform = self.estimate_affine_transform(landmarks, stable_points, stable_reference)
        transformed_frame, transformed_landmarks = self.apply_affine_transform(
            frame, landmarks, transform, target_size, interpolation, border_mode, border_value)
        return transformed_frame, transformed_landmarks


    def get_stable_reference(self, reference, reference_size, target_size, stable_points=None):
        if stable_points is None:
            stable_points = self.stable_points

        if len(stable_points) == 8:
            # FAN 68 landmark modu — Auto-AVSR orijinal 8 stabil nokta
            stable_reference = np.vstack([
                np.mean(reference[36:42], axis=0),   # sag goz
                np.mean(reference[42:48], axis=0),   # sol goz
                np.mean(reference[31:36], axis=0),   # burun
                np.mean(reference[48:68], axis=0),   # agiz
                reference[28],                        # burun koprusu
                reference[33],                        # burun ucu
                reference[36],                        # sag goz kose
                reference[45],                        # sol goz kose
            ])
        elif len(stable_points) == 4:
            # MediaPipe 4 landmark modu
            stable_reference = np.vstack([
                np.mean(reference[36:42], axis=0),
                np.mean(reference[42:48], axis=0),
                np.mean(reference[31:36], axis=0),
                np.mean(reference[48:68], axis=0)
            ])
        else:
            # Genel: stable_points'taki indeksleri reference'dan al
            stable_reference = np.vstack([reference[x] for x in stable_points])

        stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
        stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0
        return stable_reference


    def estimate_affine_transform(self, landmarks, stable_points, stable_reference):
        src = np.vstack([landmarks[x] for x in stable_points])
        return cv2.estimateAffinePartial2D(src, stable_reference, method=cv2.LMEDS)[0]


    def apply_affine_transform(self, frame, landmarks, transform, target_size, interpolation, border_mode, border_value):
        transformed_frame = cv2.warpAffine(frame, transform, dsize=(target_size[0], target_size[1]),
                                           flags=interpolation, borderMode=border_mode, borderValue=border_value)
        transformed_landmarks = np.matmul(landmarks, transform[:, :2].transpose()) + transform[:, 2].transpose()
        return transformed_frame, transformed_landmarks
