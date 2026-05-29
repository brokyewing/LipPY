"""ROI çıkarma pipeline — eğitim (NEW/extract_rois.py) ile birebir aynı.

RetinaFace + FAN (GPU, 68 landmark) → affine align (mean face template)
 → 256×256 warp → merkez 96×96 grayscale patch.

Kurulum (ibug paketleri):
    git clone https://github.com/hhj1897/face_detection.git
    cd face_detection && git lfs pull && pip install -e . && cd ..
    git clone https://github.com/hhj1897/face_alignment.git
    cd face_alignment && pip install -e . && cd ..

ibug kurulu değilse MediaPipe fallback'e düşer (4 landmark, daha düşük kalite).
"""
from .pipeline import video_to_roi_array, get_pipeline

__all__ = ["video_to_roi_array", "get_pipeline"]
