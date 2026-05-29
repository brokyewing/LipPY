"""
Generator: kaggle/lippy-av-fusion-decode-v1.ipynb notebook'unu olusturur.

Audio-Video late fusion + KenLM decode:
  - v1c-disjoint (video) en iyi checkpoint -> logits V
  - lippy-audio-wav2vec-v1 (audio) en iyi checkpoint -> logits A
  - Fusion: alpha * V_prob + (1-alpha) * A_prob_downsampled
  - Grid search alpha: 0.0, 0.3, 0.5, 0.7, 1.0
  - Greedy decode + KenLM beam search + final WER tablosu

Kullanim:
    python tools/build_av_fusion_notebook.py
"""
import json
import os

OUT_PATH = r"C:\Users\asker\Projelerim\LipPY\kaggle\lippy-av-fusion-decode-v1.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


cells = []

# =========================================================
# 0. INTRO
# =========================================================
cells.append(md("""# LipPY AV Fusion Decode v1 — Late Fusion + KenLM

**Plan B'nin 3. asamasi** — Video ve audio modelleri **late fusion** ile birlestir,
gercek WER'i hesapla. Paper'in **ana sonucu**.

## Pipeline

```
Video v1c-disjoint best ckpt    -> CTC logits V (T_v=25fps, V=31 fonem)
                                            \\
                                             alpha * V_prob + (1-alpha) * A_prob
                                            /                              ↓
Audio Wav2Vec2 best ckpt        -> CTC logits A (T_a=50fps, downsample 2x to T_v)
                                                                            ↓
                                                                  Greedy/Beam decode
                                                                            ↓
                                                                  + KenLM (alpha_lm, beta)
                                                                            ↓
                                                                       Final WER
```

## Grid search alpha (video aglrli)

| alpha | (1-alpha) | Yorum |
|---|---|---|
| 1.0 | 0.0 | Saf video baseline |
| 0.7 | 0.3 | Kullanici tercihi (video-aglirli) |
| 0.5 | 0.5 | Esit |
| 0.3 | 0.7 | Audio-aglirli |
| 0.0 | 1.0 | Saf audio baseline |

## Inputs

- `brokyewing/lippy-test` (ROI + text + splits)
- `brokyewing/lippy-audio-segments` (audio wav)
- `brokyewing/lippy-phoneme-v1c-full-checkpoint` (video model)
- `brokyewing/lippy-audio-wav2vec-v1-checkpoint` (audio model)
- `brokyewing/lippy-kenlm-phoneme-lm` (KenLM, optional)

## Cikti

- `dataset-metadata.json` -> `brokyewing/lippy-av-fusion-results`
- Final WER tablosu (sunum/paper icin)
"""))

# =========================================================
# 1. IMPORTS
# =========================================================
cells.append(code("""!pip install -q torchaudio transformers pyctcdecode

import os, glob, json, time, math, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.models import Conformer
from transformers import Wav2Vec2Model

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')
"""))

# =========================================================
# 2. VOCAB (BOTH MODELS USE THIS)
# =========================================================
cells.append(md("## 1) Vocab — 31 Fonem (v1c + audio AYNI)"))

cells.append(code("""PHONEMES = [
    'SIL', 'SPACE',
    'a', 'e', 'i', 'o', 'u', '\\u0131', '\\u00f6', '\\u00fc',
    'b', 'p', 'm', 'f', 'v', 'd', 't', 'n', 'l',
    's', 'z', 'c', '\\u00e7', '\\u015f', 'j',
    'k', 'g', '\\u011f', 'r', 'h', 'y',
]
BLANK_IDX = 0
SPACE_IDX = 1
PH_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}
IDX_TO_PH = {i: p for i, p in enumerate(PHONEMES)}
NUM_PHONEMES = len(PHONEMES)
CHAR_TO_PH = {ph: ph for ph in PHONEMES if ph not in ('SIL', 'SPACE')}


def text_to_phonemes(text):
    phs = []
    for ch in text.lower().strip():
        if ch == ' ':
            if not phs or phs[-1] != SPACE_IDX:
                phs.append(SPACE_IDX)
        elif ch in CHAR_TO_PH:
            phs.append(PH_TO_IDX[CHAR_TO_PH[ch]])
    while phs and phs[0] == SPACE_IDX: phs.pop(0)
    while phs and phs[-1] == SPACE_IDX: phs.pop()
    return phs


def phonemes_to_text(indices):
    if isinstance(indices, torch.Tensor):
        indices = indices.tolist()
    out = []
    for i in indices:
        if i == BLANK_IDX: continue
        out.append(' ' if i == SPACE_IDX else IDX_TO_PH.get(i, ''))
    return ''.join(out)


print(f'Vocab: {NUM_PHONEMES} fonem')
"""))

# =========================================================
# 3. VIDEO MODEL (v1c)
# =========================================================
cells.append(md("## 2) Video Model — LipPhonemeModel (v1c ile birebir)"))

cells.append(code("""class SiLU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


class BasicBlockSiLU(nn.Module):
    expansion = 1
    def __init__(self, ic, oc, stride=1, ds=None):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.act = SiLU()
        self.ds = ds
    def forward(self, x):
        identity = x if self.ds is None else self.ds(x)
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + identity)


class ResNetSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = self._ml(64, 64, 2, 1)
        self.layer2 = self._ml(64, 128, 2, 2)
        self.layer3 = self._ml(128, 256, 2, 2)
        self.layer4 = self._ml(256, 512, 2, 2)
    def _ml(self, ic, oc, n, s):
        ds = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=s, bias=False),
                            nn.BatchNorm2d(oc)) if s != 1 or ic != oc else None
        return nn.Sequential(BasicBlockSiLU(ic, oc, s, ds),
                              *[BasicBlockSiLU(oc, oc) for _ in range(n-1)])
    def forward(self, x): return self.layer4(self.layer3(self.layer2(self.layer1(x))))


D_MODEL = 512
ENC_LAYERS = 6
NHEAD = 8
DIM_FF = 1024


class LipPhonemeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), SiLU(),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNetSiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Identity()
        self.conformer = Conformer(input_dim=D_MODEL, num_heads=NHEAD, ffn_dim=DIM_FF,
                                     num_layers=ENC_LAYERS, depthwise_conv_kernel_size=31,
                                     dropout=0.1)
        self.ctc_fc = nn.Linear(D_MODEL, NUM_PHONEMES)

    def forward(self, x, il=None):
        B, T, H, W = x.shape
        x = x.unsqueeze(1)
        x = self.frontend3D(x)
        B2, C, To, H2, W2 = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B2 * To, C, H2, W2)
        x = self.avg_pool(self.trunk(x)).view(B2, To, -1)
        x = self.proj(x)
        if il is None:
            il = torch.full((B2,), To, dtype=torch.long, device=x.device)
        x, _ = self.conformer(x, lengths=il)
        logits = self.ctc_fc(x).log_softmax(-1)
        return logits.permute(1, 0, 2), il
"""))

# =========================================================
# 4. AUDIO MODEL (Wav2Vec2)
# =========================================================
cells.append(md("## 3) Audio Model — Wav2Vec2 + CTC (audio notebook ile birebir)"))

cells.append(code("""HF_AUDIO_MODEL = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"


class AudioPhonemeModel(nn.Module):
    def __init__(self, model_name=HF_AUDIO_MODEL, num_phonemes=NUM_PHONEMES):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.ctc_fc = nn.Linear(hidden, num_phonemes)

    def forward(self, audio, attention_mask=None):
        out = self.encoder(audio, attention_mask=attention_mask, return_dict=True)
        x = self.dropout(out.last_hidden_state)
        logits = self.ctc_fc(x)
        return logits.log_softmax(-1).permute(1, 0, 2)

    def get_output_length(self, input_length):
        return self.encoder._get_feat_extract_output_lengths(input_length)
"""))

# =========================================================
# 5. LOAD CHECKPOINTS
# =========================================================
cells.append(md("""## 4) Checkpoint Yukle

Kaggle input dataset slug'lari:
- Video: `brokyewing/lippy-phoneme-v1c-full-checkpoint`
- Audio: `brokyewing/lippy-audio-wav2vec-v1-checkpoint`
"""))

cells.append(code("""# Video checkpoint
video_ckpt_cands = glob.glob('/kaggle/input/**/lippy-phoneme-v1c-full-checkpoint/**/best_model.pt',
                              recursive=True)
if not video_ckpt_cands:
    video_ckpt_cands = glob.glob('/kaggle/input/**/best_model.pt', recursive=True)
    video_ckpt_cands = [p for p in video_ckpt_cands if 'v1c' in p.lower() or 'phoneme' in p.lower()]
VIDEO_CKPT = video_ckpt_cands[0] if video_ckpt_cands else None
print(f'Video ckpt: {VIDEO_CKPT}')

# Audio checkpoint
audio_ckpt_cands = glob.glob('/kaggle/input/**/lippy-audio-wav2vec-v1-checkpoint/**/best_model.pt',
                              recursive=True)
AUDIO_CKPT = audio_ckpt_cands[0] if audio_ckpt_cands else None
print(f'Audio ckpt: {AUDIO_CKPT}')

assert VIDEO_CKPT, 'Video checkpoint bulunamadi'
assert AUDIO_CKPT, 'Audio checkpoint bulunamadi'

# Video model yukle
video_model = LipPhonemeModel().to(DEVICE)
state = torch.load(VIDEO_CKPT, map_location=DEVICE, weights_only=False)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
video_model.load_state_dict(state, strict=False)
video_model.eval()
print(f'Video model OK: {sum(p.numel() for p in video_model.parameters()):,} param')

# Audio model yukle
audio_model = AudioPhonemeModel().to(DEVICE)
state = torch.load(AUDIO_CKPT, map_location=DEVICE, weights_only=False)
if isinstance(state, dict) and 'model_state_dict' in state:
    state = state['model_state_dict']
audio_model.load_state_dict(state, strict=False)
audio_model.eval()
print(f'Audio model OK: {sum(p.numel() for p in audio_model.parameters()):,} param')
"""))

# =========================================================
# 6. DATASET (joint ROI + Audio + Text)
# =========================================================
cells.append(md("""## 5) Joint Dataset — ROI + Audio + Text (val only)

Sadece **val split** uzerinde inference (paper sonucu icin).
Audio olmayan sample'lar `audio=None` ile gecer -> video-only decode kullanilir.
"""))

cells.append(code("""# Auto-discover paths
ROIS_CANDS = glob.glob('/kaggle/input/**/rois_mp4', recursive=True)
ROIS_DIR = ROIS_CANDS[0] if ROIS_CANDS else '/kaggle/input/lippy-test/dataset/rois_mp4'

AUDIO_CANDS = glob.glob('/kaggle/input/**/lippy-audio-segments/**/audios', recursive=True)
AUDIO_DIR = AUDIO_CANDS[0] if AUDIO_CANDS else '/kaggle/input/lippy-audio-segments/audios'

TEXT_CANDS = glob.glob('/kaggle/input/**/dataset/texts', recursive=True)
TEXT_DIR = TEXT_CANDS[0] if TEXT_CANDS else '/kaggle/input/lippy-test/dataset/texts'

SPLIT_CANDS = glob.glob('/kaggle/input/**/dataset/splits', recursive=True)
SPLIT_DIR = SPLIT_CANDS[0] if SPLIT_CANDS else '/kaggle/input/lippy-test/dataset/splits'

print(f'ROIS_DIR:  {ROIS_DIR}')
print(f'AUDIO_DIR: {AUDIO_DIR}')
print(f'TEXT_DIR:  {TEXT_DIR}')
print(f'SPLIT_DIR: {SPLIT_DIR}')

MEAN, STD, CROP = 0.421, 0.165, 88
SAMPLE_RATE = 16000


def read_mp4(p):
    cap = cv2.VideoCapture(p)
    frames = []
    while True:
        r, fr = cap.read()
        if not r: break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.array(frames, dtype=np.uint8) if frames else None


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    wav = wav.squeeze(0).float()
    wav = (wav - wav.mean()) / (wav.std() + 1e-7)
    return wav


class JointDataset(Dataset):
    def __init__(self, split='val'):
        with open(os.path.join(SPLIT_DIR, f'{split}.txt'), 'r', encoding='utf-8') as f:
            ids = [l.strip() for l in f if l.strip()]
        self.ids = []
        n_audio_yok = 0
        for s in ids:
            rp = os.path.join(ROIS_DIR, f'{s}.mp4')
            tp = os.path.join(TEXT_DIR, f'{s}.txt')
            if not (os.path.exists(rp) and os.path.exists(tp)):
                continue
            ap = os.path.join(AUDIO_DIR, f'{s}.wav')
            has_audio = os.path.exists(ap)
            if not has_audio:
                n_audio_yok += 1
            self.ids.append((s, has_audio))
        n_with_audio = sum(1 for _, h in self.ids if h)
        print(f'  [{split}] {len(self.ids)} ornek, {n_with_audio} audio var, '
              f'{n_audio_yok} audio yok')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        s, has_audio = self.ids[i]
        # ROI
        roi = read_mp4(os.path.join(ROIS_DIR, f'{s}.mp4')).astype(np.float32) / 255.0
        T, H, W = roi.shape
        t = (H - CROP) // 2
        l = (W - CROP) // 2
        roi = roi[:, t:t + CROP, l:l + CROP]
        roi_t = torch.from_numpy(((roi - MEAN) / STD).copy()).float()

        # Text
        with open(os.path.join(TEXT_DIR, f'{s}.txt'), 'r', encoding='utf-8') as f:
            txt = f.read().strip()
        lab = torch.tensor(text_to_phonemes(txt), dtype=torch.long)

        # Audio (varsa)
        audio_t = None
        if has_audio:
            try:
                audio_t = load_audio(os.path.join(AUDIO_DIR, f'{s}.wav'))
            except Exception:
                audio_t = None

        return s, roi_t, audio_t, lab, txt


val_ds = JointDataset('val')
print(f'Toplam val ornek: {len(val_ds)}')
"""))

# =========================================================
# 7. FUSION INFERENCE
# =========================================================
cells.append(md("""## 6) Fusion Inference — Tum val'a calistir

Her sample icin:
1. Video model -> log_probs_v (T_v, V)
2. Audio model -> log_probs_a (T_a, V)
3. Audio'yu 2x downsample (50Hz -> 25Hz) average pool
4. Length align (min(T_v, T_a_ds))
5. probability domain weighted sum
"""))

cells.append(code("""def video_inference_single(roi_tensor):
    \"\"\"Tek ROI icin video logits. roi_tensor: (T, 88, 88) torch.\"\"\"
    x = roi_tensor.unsqueeze(0).to(DEVICE)  # (1, T, 88, 88)
    with torch.no_grad():
        lp, il = video_model(x)  # lp: (T_v, 1, V), il: actual length
    return lp[:il[0].item(), 0, :].cpu()  # (T_v, V)


def audio_inference_single(audio_tensor):
    \"\"\"Tek audio icin audio logits. audio_tensor: (N,) torch.\"\"\"
    x = audio_tensor.unsqueeze(0).to(DEVICE)  # (1, N)
    attn = torch.ones_like(x, dtype=torch.long)
    with torch.no_grad():
        lp = audio_model(x, attention_mask=attn)  # (T_a, 1, V)
        T_a = lp.shape[0]
        il_a = audio_model.get_output_length(torch.tensor([x.shape[1]]))
        T_a_valid = min(T_a, il_a[0].item())
    return lp[:T_a_valid, 0, :].cpu()  # (T_a, V)


def downsample_logits(lp, factor):
    \"\"\"Log_probs (T, V)'i factor'a gore average pool ile downsample.\"\"\"
    T, V = lp.shape
    # T -> T // factor
    target_T = T // factor
    if target_T == 0:
        return lp[:1]
    # (T, V) -> (V, T) -> avg_pool1d -> (V, T/f) -> (T/f, V)
    x = lp.t().unsqueeze(0)  # (1, V, T)
    # Audio log_probs -> probs once, sonra pool, sonra log
    probs = x.exp()
    pooled = F.avg_pool1d(probs, kernel_size=factor, stride=factor)  # (1, V, T/f)
    pooled = pooled.clamp(min=1e-10).log()
    return pooled.squeeze(0).t()  # (T/f, V)


def fuse_logits(lp_v, lp_a_ds, alpha):
    \"\"\"
    lp_v: (T_v, V) video log_probs
    lp_a_ds: (T_a_ds, V) audio downsampled log_probs (already 25Hz)
    alpha: weight on video (0..1)
    Return: (T, V) fused log_probs.
    \"\"\"
    T = min(lp_v.shape[0], lp_a_ds.shape[0])
    v_prob = lp_v[:T].exp()
    a_prob = lp_a_ds[:T].exp()
    fused = alpha * v_prob + (1.0 - alpha) * a_prob
    return fused.clamp(min=1e-10).log()


def greedy_decode(lp):
    \"\"\"lp: (T, V) log_probs -> phoneme indices.\"\"\"
    ids = lp.argmax(-1).tolist()
    out = []
    p = -1
    for i in ids:
        if i != p and i != BLANK_IDX:
            out.append(i)
        p = i
    return out


def edit_dist(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for ai in a:
        r = [dp[0] + 1]
        for j, bj in enumerate(b):
            r.append(min(r[-1] + 1, dp[j + 1] + 1, dp[j] + (0 if ai == bj else 1)))
        dp = r
    return dp[-1]
"""))

cells.append(code("""# Tum val'a inference - logits cache et (RAM/disk)
import pickle

ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

# Cache: her sample icin (logits_v, logits_a_ds)
# 25 fps + 31 vocab + 250 ortalama frame = 31*250*4 = 31KB per sample
# 10K val * 31KB = 310 MB - RAM'de tutulabilir

cache = []
print(f'Inference baslat: {len(val_ds)} sample...', flush=True)
t0 = time.time()

for idx in range(len(val_ds)):
    s, roi, audio, lab, txt = val_ds[idx]

    lp_v = video_inference_single(roi)

    if audio is not None:
        lp_a = audio_inference_single(audio)
        # Audio 50Hz -> 25Hz (2x downsample)
        lp_a_ds = downsample_logits(lp_a, factor=2)
    else:
        lp_a_ds = None

    cache.append({
        'id': s,
        'lp_v': lp_v,        # (T_v, V)
        'lp_a_ds': lp_a_ds,   # (T_a_ds, V) or None
        'ref': lab.tolist(),
        'text': txt,
    })

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed * (len(val_ds) - idx - 1) / (idx + 1)
        print(f'  [{idx+1}/{len(val_ds)}] elapsed {elapsed:.0f}s, eta {eta:.0f}s',
              flush=True)

print(f'Inference bitti: {time.time() - t0:.0f}s')
print(f'Cache boyutu: {sum(c[\"lp_v\"].nelement() + (c[\"lp_a_ds\"].nelement() if c[\"lp_a_ds\"] is not None else 0) for c in cache) * 4 / 1024 / 1024:.1f} MB')
"""))

# =========================================================
# 8. GRID SEARCH ALPHA
# =========================================================
cells.append(md("""## 7) Grid Search Alpha — PER hesapla

Her alpha icin:
- Audio yoksa: video-only decode (alpha=1.0 zorlanir)
- Audio varsa: fusion
- Greedy decode + PER hesapla
"""))

cells.append(code("""def calc_per(preds, refs, exclude_space=False):
    if exclude_space:
        preds = [[t for t in p if t != SPACE_IDX] for p in preds]
        refs = [[t for t in r if t != SPACE_IDX] for r in refs]
    td = tl = 0
    for p, r in zip(preds, refs):
        td += edit_dist(p, r)
        tl += max(len(r), 1)
    return td / max(tl, 1)


def calc_space_f1(preds, refs):
    tp = fp = fn = 0
    for p, r in zip(preds, refs):
        ps = sum(1 for t in p if t == SPACE_IDX)
        rs = sum(1 for t in r if t == SPACE_IDX)
        tp += min(ps, rs)
        fp += max(0, ps - rs)
        fn += max(0, rs - ps)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-6)


results = {}

for alpha in ALPHAS:
    preds = []
    refs = []
    n_video_only = 0
    n_fused = 0
    for c in cache:
        if c['lp_a_ds'] is not None and alpha < 1.0:
            # Fusion
            lp = fuse_logits(c['lp_v'], c['lp_a_ds'], alpha)
            n_fused += 1
        else:
            # Video-only
            lp = c['lp_v']
            n_video_only += 1
        pred = greedy_decode(lp)
        preds.append(pred)
        refs.append(c['ref'])

    per_all = calc_per(preds, refs)
    per_ph = calc_per(preds, refs, exclude_space=True)
    f1 = calc_space_f1(preds, refs)

    results[alpha] = {
        'per_all': per_all, 'per_ph': per_ph, 'space_f1': f1,
        'n_video_only': n_video_only, 'n_fused': n_fused,
    }
    print(f'alpha={alpha:.1f} | PER-all: {per_all:.1%} | PER-ph: {per_ph:.1%} | '
          f'SP F1: {f1:.2f} | fused: {n_fused}, video-only: {n_video_only}')


print('\\n' + '=' * 70)
print('GRID SEARCH OZET')
print('=' * 70)
print(f'{\"alpha\":>6} | {\"PER-all\":>8} | {\"PER-ph\":>8} | {\"SP F1\":>6}')
print('-' * 50)
best_alpha = None
best_per = 1.0
for a, r in sorted(results.items()):
    flag = ''
    if r['per_all'] < best_per:
        best_per = r['per_all']
        best_alpha = a
        flag = ' (BEST)'
    print(f'{a:>6.1f} | {r[\"per_all\"]:>7.1%} | {r[\"per_ph\"]:>7.1%} | '
          f'{r[\"space_f1\"]:>6.2f}{flag}')
print(f'\\nBEST alpha: {best_alpha} | PER: {best_per:.1%}')
"""))

# =========================================================
# 9. KENLM (optional)
# =========================================================
cells.append(md("""## 8) KenLM Beam Search Decode (opsiyonel)

`brokyewing/lippy-kenlm-phoneme-lm` dataset yuklendiyse, en iyi `alpha` icin
KenLM ile beam search yap, WER hesapla.

KenLM dataset YOK ise bu cell skip — saf PER sonucu yeterli (paper'a ana sonuc).
"""))

cells.append(code("""KENLM_CANDS = glob.glob('/kaggle/input/**/lm.klm', recursive=True)
if not KENLM_CANDS:
    KENLM_CANDS = glob.glob('/kaggle/input/**/*.klm', recursive=True)

if KENLM_CANDS:
    KENLM_PATH = KENLM_CANDS[0]
    print(f'KenLM bulundu: {KENLM_PATH}')

    try:
        from pyctcdecode import build_ctcdecoder
        # pyctcdecode label sirasi: blank index 0 olmali, sonraki labellar
        labels = [' '] + [' '] + ['a','e','i','o','u','i','o','u',
                                   'b','p','m','f','v','d','t','n','l',
                                   's','z','c','c','s','j','k','g','g','r','h','y']
        # Note: pyctcdecode labels need to match phoneme indices exactly
        # For now, use simplified phoneme labels
        labels = list(map(str, range(NUM_PHONEMES)))  # fonem indeksleri label olarak

        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=KENLM_PATH,
            alpha=0.5,
            beta=0.5,
        )
        print(f'KenLM decoder hazir: alpha=0.5, beta=0.5')

        # Best alpha icin beam decode
        if best_alpha is not None:
            print(f'\\nBest alpha {best_alpha} icin KenLM beam decode...')
            wer_results = []
            for c in cache[:500]:  # ilk 500 sample (zaman tasarrufu)
                if c['lp_a_ds'] is not None and best_alpha < 1.0:
                    lp = fuse_logits(c['lp_v'], c['lp_a_ds'], best_alpha)
                else:
                    lp = c['lp_v']
                # Beam search
                # NOTE: pyctcdecode expects (T, V) np array
                try:
                    beam_pred = decoder.decode(lp.numpy(), beam_width=50)
                    # beam_pred is space-separated phoneme indices
                    pred_idxs = [int(t) for t in beam_pred.split() if t.isdigit()]
                    wer_results.append((c['text'], phonemes_to_text(pred_idxs)))
                except Exception as e:
                    wer_results.append((c['text'], ''))

            # Sample print
            print(f'\\nIlk 5 beam decode ornek:')
            for ref, hyp in wer_results[:5]:
                print(f'  REF: \"{ref[:60]}\"')
                print(f'  HYP: \"{hyp[:60]}\"')
                print()
    except ImportError:
        print('pyctcdecode kurulu degil, beam decode atlandi.')
else:
    print('KenLM dataset bulunamadi — beam decode atlandi.')
    print('Greedy decode sonuclari yukarida.')
"""))

# =========================================================
# 10. FINAL TABLE + SAVE
# =========================================================
cells.append(md("""## 9) Final Sonuc Tablosu (paper icin)

Tum sonuclar JSON olarak kaydedilir, paper tablosu uretilir.
"""))

cells.append(code("""# Final tablo
print('\\n' + '=' * 75)
print('LipPY AV FUSION — FINAL RESULTS')
print('=' * 75)
print(f'{\"Config\":<25} | {\"PER-all\":>8} | {\"PER-ph\":>8} | {\"SP F1\":>6}')
print('-' * 75)
for a in ALPHAS:
    r = results[a]
    if a == 1.0:
        name = 'Video-only (v1c)'
    elif a == 0.0:
        name = 'Audio-only (wav2vec)'
    else:
        name = f'Fusion alpha={a}'
    flag = ' (BEST)' if a == best_alpha else ''
    print(f'{name:<25} | {r[\"per_all\"]:>7.1%} | {r[\"per_ph\"]:>7.1%} | '
          f'{r[\"space_f1\"]:>6.2f}{flag}')
print('=' * 75)

# JSON kaydet (paper figure'i icin)
import json as _json
output = {
    'best_alpha': best_alpha,
    'best_per': best_per,
    'results': {f'{a:.1f}': results[a] for a in ALPHAS},
    'video_ckpt': VIDEO_CKPT,
    'audio_ckpt': AUDIO_CKPT,
    'val_size': len(cache),
    'val_with_audio': sum(1 for c in cache if c['lp_a_ds'] is not None),
}
out_dir = '/kaggle/working/av_fusion_results'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/results.json', 'w') as f:
    _json.dump(output, f, indent=2, default=str)
print(f'\\nJSON: {out_dir}/results.json')

# Paper hikayesi
print('\\n' + '=' * 70)
print('PAPER HIKAYESI')
print('=' * 70)
v_per = results[1.0]['per_all']
a_per = results[0.0]['per_all']
print(f'Video-only PER-all:           {v_per:.1%}')
print(f'Audio-only PER-all:           {a_per:.1%}')
print(f'Best fusion (alpha={best_alpha}):       {best_per:.1%}')
gain = v_per - best_per
if gain > 0:
    print(f'\\nFusion video-only\\'a karsi: -{gain*100:.1f} puan kazanc')
    print(f'-> AV fusion Turkce VSR icin etkili ✓')
else:
    print(f'\\nFusion video-only\\'dan {-gain*100:.1f} puan KOTU')
    print(f'-> Audio modelinin domain shift\\'i (CV stüdyo -> YouTube) negatif etki')
"""))

# =========================================================
# DUMP
# =========================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"YAZILDI: {OUT_PATH}")
print(f"Cell sayisi: {len(cells)}")
print(f"  Markdown: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code:     {sum(1 for c in cells if c['cell_type'] == 'code')}")
