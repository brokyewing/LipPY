"""
Generator: kaggle/lippy-av-fusion-cross-attn.ipynb

Cross-Attention Fusion: video features + audio features -> joint CTC.

Mimari:
  Video encoder (v1c, DONMUS) -> v_feat (T_v=25Hz, 512)
                                                       \\
                                                        Cross-attn fusion (3 layer)
                                                       /
  Audio encoder (best, DONMUS) -> a_feat (T_a, 1024)
                                                        \\
                                                         -> fused (T_v, 512)
                                                            -> CTC head (31)
"""
import json
import os

OUT_PATH = r"C:\Users\asker\Projelerim\LipPY\kaggle\lippy-av-fusion-cross-attn.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


cells = []

# 0. INTRO
cells.append(md("""# LipPY AV Cross-Attention Fusion (Max Effort Path 3B)

**Hedef:** Late fusion'in (~%40-50 WER) ustune cikamadigi yere ulasmak: **WER ~30-35**.

## Niye cross-attention?

Late fusion = logit toplama (audio ve video birbirini gormez).
Cross-attention = **audio video'ya zaman dilimi bazli "donerek dinler"**:
- Frame T'deki dudak hareketi <-> ayni anki ses fonem'ine baglar
- Long-range temporal alignment yapar (drift toleransi)
- Spelling-confusion'i ses ile cozer

## Mimari

```
Video (B,T_v=25Hz,512) [frozen encoder]
        \\
         -> Cross-attention layer 1: V <- attends_to <- A
            Cross-attention layer 2: V <- attends_to <- A
            Cross-attention layer 3: V <- attends_to <- A
         /
Audio (B,T_a,D_a) [frozen encoder]
         -> Linear proj (D_a -> 512)

Fused (B,T_v,512) -> Linear(512, 31) -> CTC log_softmax
```

**TRAINABLE:** Sadece fusion layer'lar + CTC head (~5M param)
**FROZEN:** Video encoder (~37M) + Audio encoder (~300-770M)
**Egitim suresi:** ~5-8 saat (donmus encoder'lar sayesinde hizli)

## Inputs

- `brokyewing/lippy-test` (rois_mp4 + texts + splits)
- `brokyewing/lippy-audio-segments` (audio wav)
- `brokyewing/lippy-phoneme-v1c-full-checkpoint` (video model, best ckpt)
- En iyi audio: `brokyewing/lippy-audio-whisper-ctc-checkpoint`
  veya `brokyewing/lippy-audio-wav2vec-v1-checkpoint`
"""))

# 1. IMPORTS + CONFIG
cells.append(code("""!pip install -q transformers torchaudio soundfile

import os, glob, json, shutil, time, math, random, subprocess
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchaudio.models import Conformer
from transformers import Wav2Vec2Model, WhisperModel, WhisperFeatureExtractor

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')

# ============ CONFIG ============
EPOCHS = 12
LR = 5e-5                                 # Sadece fusion + CTC head egitiliyor
WARMUP = 2
GCLIP = 2.0
ACCUM = 4
BS = 2                                    # Iki buyuk encoder VRAM'de -> kucuk BS
UPLOAD_EVERY = 3

# Fusion mimari
D_MODEL = 512
NHEAD = 8
N_FUSION_LAYERS = 3                       # 3 cross-attn block
FUSION_DROPOUT = 0.1

# Audio modeli secimi (hangisi best ckpt'i varsa)
USE_WHISPER = True                        # False ise Wav2Vec2
WHISPER_MODEL = "openai/whisper-medium"
WAV2VEC_MODEL = "mpoyraz/wav2vec2-xls-r-300m-cv8-turkish"

# Audio constraints
SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 0.5
MAX_AUDIO_SEC = 10.0

FORCE_SCRATCH = False
"""))

# 2. VOCAB
cells.append(md("## 1) Vocab — 31 fonem (v1c + audio ile birebir)"))
cells.append(code("""PHONEMES = [
    'SIL', 'SPACE',
    'a','e','i','o','u','\\u0131','\\u00f6','\\u00fc',
    'b','p','m','f','v','d','t','n','l',
    's','z','c','\\u00e7','\\u015f','j',
    'k','g','\\u011f','r','h','y',
]
BLANK_IDX = 0; SPACE_IDX = 1
PH_TO_IDX = {p:i for i,p in enumerate(PHONEMES)}
IDX_TO_PH = {i:p for i,p in enumerate(PHONEMES)}
NUM_PHONEMES = len(PHONEMES)
CHAR_TO_PH = {ph:ph for ph in PHONEMES if ph not in ('SIL','SPACE')}


def text_to_phonemes(text):
    phs = []
    for ch in text.lower().strip():
        if ch == ' ':
            if not phs or phs[-1] != SPACE_IDX:
                phs.append(SPACE_IDX)
        elif ch in CHAR_TO_PH:
            phs.append(PH_TO_IDX[CHAR_TO_PH[ch]])
    while phs and phs[0]==SPACE_IDX: phs.pop(0)
    while phs and phs[-1]==SPACE_IDX: phs.pop()
    return phs


def phonemes_to_text(idxs):
    if isinstance(idxs, torch.Tensor): idxs = idxs.tolist()
    out = []
    for i in idxs:
        if i==BLANK_IDX: continue
        out.append(' ' if i==SPACE_IDX else IDX_TO_PH.get(i,''))
    return ''.join(out)
"""))

# 3. VIDEO ENCODER (v1c, frozen)
cells.append(md("""## 2) Video Encoder — LipPhonemeModel (v1c, DONMUS)

v1c mimari birebir kopyalanir. `get_features` metodu eklenir → CTC head ONCESI feature'lar.
"""))

cells.append(code("""class SiLU(nn.Module):
    def forward(self,x): return x * torch.sigmoid(x)


class BasicBlockSiLU(nn.Module):
    expansion=1
    def __init__(self,ic,oc,stride=1,ds=None):
        super().__init__()
        self.conv1=nn.Conv2d(ic,oc,3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(oc)
        self.conv2=nn.Conv2d(oc,oc,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(oc)
        self.act=SiLU(); self.ds=ds
    def forward(self,x):
        identity = x if self.ds is None else self.ds(x)
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + identity)


class ResNetSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=self._ml(64,64,2,1)
        self.layer2=self._ml(64,128,2,2)
        self.layer3=self._ml(128,256,2,2)
        self.layer4=self._ml(256,512,2,2)
    def _ml(self,ic,oc,n,s):
        ds = nn.Sequential(nn.Conv2d(ic,oc,1,stride=s,bias=False),nn.BatchNorm2d(oc)) if s!=1 or ic!=oc else None
        return nn.Sequential(BasicBlockSiLU(ic,oc,s,ds), *[BasicBlockSiLU(oc,oc) for _ in range(n-1)])
    def forward(self,x): return self.layer4(self.layer3(self.layer2(self.layer1(x))))


V_D_MODEL = 512
V_ENC_LAYERS = 6
V_NHEAD = 8
V_DIM_FF = 1024


class VideoEncoder(nn.Module):
    \"\"\"v1c LipPhonemeModel'in encoder kismi. CTC head VAR ama sadece get_features kullanilir.\"\"\"
    def __init__(self):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1,64,(5,7,7),stride=(1,2,2),padding=(2,3,3),bias=False),
            nn.BatchNorm3d(64), SiLU(),
            nn.MaxPool3d((1,3,3),stride=(1,2,2),padding=(0,1,1)),
        )
        self.trunk = ResNetSiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Identity()
        self.conformer = Conformer(input_dim=V_D_MODEL, num_heads=V_NHEAD, ffn_dim=V_DIM_FF,
                                     num_layers=V_ENC_LAYERS, depthwise_conv_kernel_size=31,
                                     dropout=0.1)
        self.ctc_fc = nn.Linear(V_D_MODEL, NUM_PHONEMES)  # v1c'den state load icin var

    def get_features(self, x, il=None):
        \"\"\"
        x: (B, T, H, W) ROI tensor
        Return: (B, T_v, 512) - conformer cikti (CTC head oncesi)
        \"\"\"
        B,T,H,W = x.shape
        x = x.unsqueeze(1)
        x = self.frontend3D(x)
        B2,C,To,H2,W2 = x.shape
        x = x.permute(0,2,1,3,4).reshape(B2*To, C, H2, W2)
        x = self.avg_pool(self.trunk(x)).view(B2, To, -1)
        x = self.proj(x)
        if il is None:
            il = torch.full((B2,), To, dtype=torch.long, device=x.device)
        x, lengths = self.conformer(x, lengths=il)
        return x, lengths  # (B, T_v, 512), (B,) actual lengths
"""))

# 4. AUDIO ENCODER (best, frozen)
cells.append(md("""## 3) Audio Encoder — Whisper veya Wav2Vec2 (best, DONMUS)

Hangi audio modeli daha iyiyse onu kullan. Bu notebook iki seçenegi destekler:
- Whisper-medium (PER hedef ~%12, USE_WHISPER=True)
- Wav2Vec2-Turkish (PER hedef ~%15-20, USE_WHISPER=False)

`get_features` -> CTC head ONCESI (B, T_a, hidden_dim).
"""))

cells.append(code("""class AudioEncoderWhisper(nn.Module):
    def __init__(self, model_name=WHISPER_MODEL):
        super().__init__()
        full = WhisperModel.from_pretrained(model_name)
        self.encoder = full.encoder
        del full
        torch.cuda.empty_cache()
        self.hidden = self.encoder.config.d_model
        # CTC head v1 model'den state load icin
        self.ctc_fc = nn.Linear(self.hidden, NUM_PHONEMES)

    def get_features(self, input_features):
        \"\"\"
        input_features: (B, 80, 3000) - log-mel
        Return: (B, 1500, 1024)
        \"\"\"
        out = self.encoder(input_features, return_dict=True)
        return out.last_hidden_state


class AudioEncoderWav2Vec(nn.Module):
    def __init__(self, model_name=WAV2VEC_MODEL):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden = self.encoder.config.hidden_size
        self.ctc_fc = nn.Linear(self.hidden, NUM_PHONEMES)

    def get_features(self, audio, attention_mask=None):
        \"\"\"
        audio: (B, T_samples)
        Return: (B, T_a, 1024)
        \"\"\"
        out = self.encoder(audio, attention_mask=attention_mask, return_dict=True)
        return out.last_hidden_state

    def get_output_length(self, input_length):
        return self.encoder._get_feat_extract_output_lengths(input_length)


# Audio modeli sec
if USE_WHISPER:
    AudioEncoder = AudioEncoderWhisper
    AUDIO_DIM = 1024
    print('Audio encoder: Whisper-medium')
else:
    AudioEncoder = AudioEncoderWav2Vec
    AUDIO_DIM = 1024
    print('Audio encoder: Wav2Vec2-Turkish')
"""))

# 5. CROSS-ATTENTION FUSION MODEL
cells.append(md("""## 4) Cross-Attention Fusion Module

Standart Transformer cross-attn block:
- Self-attention (video features kendine bakar)
- Cross-attention (video features audio'ya bakar)
- FFN

3 layer derin. Query=video, Key/Value=audio. Output: fused video features.
"""))

cells.append(code("""class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv, v_mask=None, kv_mask=None):
        \"\"\"
        x: (B, T_v, D) - video features (query)
        kv: (B, T_a, D) - audio features (key+value, ayni dim'e proj edilmis)
        v_mask: (B, T_v) - True = padding (ignore)
        kv_mask: (B, T_a) - True = padding (ignore)
        \"\"\"
        # Self-attn (video kendine)
        x2, _ = self.self_attn(x, x, x, key_padding_mask=v_mask, need_weights=False)
        x = self.norm1(x + self.dropout(x2))

        # Cross-attn (video <- audio)
        x2, _ = self.cross_attn(x, kv, kv, key_padding_mask=kv_mask, need_weights=False)
        x = self.norm2(x + self.dropout(x2))

        # FFN
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x


class AVFusionModel(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, n_layers=N_FUSION_LAYERS,
                 dropout=FUSION_DROPOUT):
        super().__init__()
        # Encoders (yuklendikten sonra dondurulacak)
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Audio dimension projection (audio_dim -> d_model)
        self.audio_proj = nn.Linear(self.audio_encoder.hidden, d_model)

        # Cross-attention fusion stack
        self.fusion_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, nhead, dim_ff=2048, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final CTC head
        self.ctc_fc = nn.Linear(d_model, NUM_PHONEMES)
        self.dropout = nn.Dropout(dropout)

    def freeze_encoders(self):
        \"\"\"Video + audio encoder donduruldu, sadece fusion + CTC head egitilir.\"\"\"
        for p in self.video_encoder.parameters():
            p.requires_grad = False
        for p in self.audio_encoder.parameters():
            p.requires_grad = False
        # Bat normalization layerlari eval modunda kalsin
        self.video_encoder.eval()
        self.audio_encoder.eval()

        v_total = sum(p.numel() for p in self.video_encoder.parameters())
        a_total = sum(p.numel() for p in self.audio_encoder.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'Video encoder (frozen):  {v_total:,}')
        print(f'Audio encoder (frozen):  {a_total:,}')
        print(f'Trainable (fusion+CTC):  {trainable:,}')
        print(f'Total:                   {total:,} ({100*trainable/total:.2f}% trainable)')

    def forward(self, video_input, audio_input, video_il=None, audio_attn=None):
        \"\"\"
        video_input: (B, T_video, H, W) ROI
        audio_input: ya (B, T_samples) wav2vec ya (B, 80, 3000) whisper
        video_il: (B,) actual video frame counts
        audio_attn: (B, T_samples) Wav2Vec2 attention mask veya None
        \"\"\"
        # Video features (donmus encoder)
        with torch.no_grad():
            v_feat, v_lens = self.video_encoder.get_features(video_input, il=video_il)

        # Audio features (donmus encoder)
        with torch.no_grad():
            if USE_WHISPER:
                a_feat = self.audio_encoder.get_features(audio_input)
            else:
                a_feat = self.audio_encoder.get_features(audio_input, attention_mask=audio_attn)

        # Audio'yu D_MODEL'a proje
        a_feat = self.audio_proj(a_feat)  # (B, T_a, D_MODEL)
        a_feat = self.dropout(a_feat)

        # Padding masks (True = padded)
        B, T_v, _ = v_feat.shape
        T_a = a_feat.shape[1]
        v_mask = torch.arange(T_v, device=v_feat.device).unsqueeze(0) >= v_lens.unsqueeze(1)

        # Cross-attention fusion
        x = v_feat
        for layer in self.fusion_layers:
            x = layer(x, a_feat, v_mask=v_mask, kv_mask=None)

        # CTC head
        logits = self.ctc_fc(x)  # (B, T_v, V)
        log_probs = logits.log_softmax(-1)
        return log_probs.permute(1, 0, 2), v_lens  # (T_v, B, V), (B,)
"""))

# 6. LOAD PRETRAINED CHECKPOINTS
cells.append(md("""## 5) Pretrained Checkpoint Yukle

Video encoder + audio encoder pretrained ckpt'lerini state_dict ile yukle.
strict=False kullanilir (sadece eslesen key'ler atanir).
"""))

cells.append(code("""model = AVFusionModel().to(DEVICE)

# Video encoder pretrained yukle
video_cands = (
    glob.glob('/kaggle/input/**/lippy-phoneme-v1c-full-checkpoint/**/best_model.pt', recursive=True)
    + glob.glob('/kaggle/input/**/best_model.pt', recursive=True)
)
video_cands = [p for p in video_cands if 'v1c' in p.lower() or 'phoneme' in p.lower()]
if video_cands:
    print(f'Video ckpt: {video_cands[0]}')
    state = torch.load(video_cands[0], map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    missing, unexpected = model.video_encoder.load_state_dict(state, strict=False)
    print(f'  Video matched: {len([k for k in state if k not in missing])} / {len(state)}')
else:
    print('UYARI: Video ckpt yok, scratch')

# Audio encoder pretrained yukle
if USE_WHISPER:
    audio_cands = glob.glob('/kaggle/input/**/lippy-audio-whisper-ctc-checkpoint/**/best_model.pt',
                             recursive=True)
else:
    audio_cands = glob.glob('/kaggle/input/**/lippy-audio-wav2vec-v1-checkpoint/**/best_model.pt',
                             recursive=True)

if audio_cands:
    print(f'Audio ckpt: {audio_cands[0]}')
    state = torch.load(audio_cands[0], map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    missing, unexpected = model.audio_encoder.load_state_dict(state, strict=False)
    print(f'  Audio matched (yaklasik)')
else:
    print('UYARI: Audio ckpt yok, HF default kullanilacak')

# Encoderlari dondur
model.freeze_encoders()
"""))

# 7. DATA (joint dataset, ROI + audio + text)
cells.append(md("""## 6) Joint Dataset — ROI + Audio + Text (speaker-disjoint)

av_fusion_decode notebook'undakine benzer. Audio yoksa sample skip.
"""))

cells.append(code("""ROIS_CANDS = glob.glob('/kaggle/input/**/rois_mp4', recursive=True)
ROIS_DIR = ROIS_CANDS[0] if ROIS_CANDS else '/kaggle/input/lippy-test/dataset/rois_mp4'

AUDIO_CANDS = glob.glob('/kaggle/input/**/lippy-audio-segments/**/audios', recursive=True)
AUDIO_DIR = AUDIO_CANDS[0] if AUDIO_CANDS else '/kaggle/input/lippy-audio-segments/audios'

TEXT_CANDS = glob.glob('/kaggle/input/**/dataset/texts', recursive=True)
TEXT_DIR = TEXT_CANDS[0] if TEXT_CANDS else '/kaggle/input/lippy-test/dataset/texts'

SPLIT_CANDS = glob.glob('/kaggle/input/**/dataset/splits', recursive=True)
SPLIT_DIR = SPLIT_CANDS[0] if SPLIT_CANDS else '/kaggle/input/lippy-test/dataset/splits'

print(f'ROIS:  {ROIS_DIR}')
print(f'AUDIO: {AUDIO_DIR}')
print(f'TEXT:  {TEXT_DIR}')
print(f'SPLIT: {SPLIT_DIR}')

MEAN, STD, CROP = 0.421, 0.165, 88

if USE_WHISPER:
    whisper_fe = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)


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
    def __init__(self, split):
        self.train = split == 'train'
        with open(os.path.join(SPLIT_DIR, f'{split}.txt'), encoding='utf-8') as f:
            all_ids = [l.strip() for l in f if l.strip()]
        self.ids = []
        skipped = 0
        for s in all_ids:
            rp = os.path.join(ROIS_DIR, f'{s}.mp4')
            ap = os.path.join(AUDIO_DIR, f'{s}.wav')
            tp = os.path.join(TEXT_DIR, f'{s}.txt')
            if not (os.path.exists(rp) and os.path.exists(ap) and os.path.exists(tp)):
                skipped += 1; continue
            self.ids.append(s)
        print(f'  [{split}] {len(self.ids)}/{len(all_ids)} (audio+video+text var, skipped={skipped})')

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        for off in range(len(self.ids)):
            s = self.ids[(i + off) % len(self.ids)]
            try:
                # ROI
                roi = read_mp4(os.path.join(ROIS_DIR, f'{s}.mp4'))
                if roi is None or len(roi) < 5: continue
                roi = roi.astype(np.float32) / 255.0
                T, H, W = roi.shape
                if self.train:
                    t = random.randint(0, H - CROP); l = random.randint(0, W - CROP)
                    if random.random() > 0.5:
                        roi = roi[:, :, ::-1].copy()
                else:
                    t = (H - CROP)//2; l = (W - CROP)//2
                roi = roi[:, t:t+CROP, l:l+CROP]
                roi_t = torch.from_numpy(((roi - MEAN) / STD).copy()).float()

                # Audio
                audio = load_audio(os.path.join(AUDIO_DIR, f'{s}.wav'))
                n_samples = audio.shape[0]
                if n_samples < SAMPLE_RATE * MIN_AUDIO_SEC: continue
                if n_samples > SAMPLE_RATE * MAX_AUDIO_SEC:
                    target = int(SAMPLE_RATE * MAX_AUDIO_SEC)
                    start = (n_samples - target) // 2
                    audio = audio[start:start + target]
                    n_samples = target

                # Whisper feature ya da raw wav
                if USE_WHISPER:
                    feats = whisper_fe(audio.numpy(), sampling_rate=SAMPLE_RATE,
                                        return_tensors="pt").input_features.squeeze(0).float()
                    audio_out = feats  # (80, 3000)
                    audio_real_len = min(n_samples // 320, 1500)
                else:
                    audio_out = audio  # (T_samples,)
                    audio_real_len = n_samples

                # Text
                with open(os.path.join(TEXT_DIR, f'{s}.txt'), encoding='utf-8') as f:
                    txt = f.read().strip()
                lab = torch.tensor(text_to_phonemes(txt), dtype=torch.long)
                if len(lab) == 0: continue

                # CTC safety: video frame count >= 2 * text length
                if roi_t.shape[0] < 2 * len(lab): continue

                return roi_t, audio_out, lab, audio_real_len
            except Exception as e:
                continue
        raise RuntimeError('Sample yuklenemedi')


def collate(batch):
    rois, audios, labs, audio_lens = zip(*batch)
    # Video padding
    il_v = torch.tensor([r.shape[0] for r in rois], dtype=torch.long)
    pad_v = torch.zeros(len(rois), il_v.max().item(), CROP, CROP)
    for i, r in enumerate(rois): pad_v[i, :r.shape[0]] = r

    # Audio padding (whisper veya wav2vec)
    if USE_WHISPER:
        # (B, 80, 3000) - hepsi ayni boyutta
        pad_a = torch.stack(audios, dim=0)
        audio_attn = None
    else:
        il_a = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
        pad_a = torch.zeros(len(audios), il_a.max().item())
        audio_attn = torch.zeros(len(audios), il_a.max().item(), dtype=torch.long)
        for i, a in enumerate(audios):
            pad_a[i, :a.shape[0]] = a
            audio_attn[i, :a.shape[0]] = 1

    tl = torch.tensor([l.shape[0] for l in labs], dtype=torch.long)
    return pad_v, pad_a, torch.cat(labs), il_v, tl, audio_attn


trl = DataLoader(JointDataset('train'), batch_size=BS, shuffle=True,
                  collate_fn=collate, num_workers=1, pin_memory=True)
vll = DataLoader(JointDataset('val'), batch_size=BS, shuffle=False,
                  collate_fn=collate, num_workers=1, pin_memory=True)
print(f'Train: {len(trl)} batch | Val: {len(vll)} batch')
"""))

# 8. TRAINING + METRICS
cells.append(md("""## 7) Egitim — Sadece fusion + CTC head egitilir"""))

cells.append(code("""MDIR = '/kaggle/working/ckpt_av_cross_attn'
UDIR = '/kaggle/working/upload_av_cross_attn'
os.makedirs(MDIR, exist_ok=True); os.makedirs(UDIR, exist_ok=True)
CP = f'{MDIR}/checkpoint.pt'; BP = f'{MDIR}/best_model.pt'

KAGGLE_USER = os.environ.get('KAGGLE_USERNAME', 'UNKNOWN')
try:
    from kaggle_secrets import UserSecretsClient
    _us = UserSecretsClient()
    KAGGLE_USER = _us.get_secret('KAGGLE_USERNAME')
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USER
    os.environ['KAGGLE_KEY'] = _us.get_secret('KAGGLE_KEY')
except Exception: pass

DATASET_SLUG = 'lippy-av-cross-attn-checkpoint'
DATASET_ID = f'{KAGGLE_USER}/{DATASET_SLUG}'
with open(f'{UDIR}/dataset-metadata.json', 'w') as f:
    json.dump({'title':'LipPY AV Cross-Attn Fusion','id':DATASET_ID,
                'licenses':[{'name':'CC0-1.0'}]}, f)


def greedy_ctc(lp):
    ids = lp.argmax(-1).tolist()
    d=[]; p=-1
    for i in ids:
        if i != p and i != BLANK_IDX:
            d.append(i)
        p = i
    return d


def edit_dist(a,b):
    n,m=len(a),len(b); dp=list(range(m+1))
    for ai in a:
        r=[dp[0]+1]
        for j,bj in enumerate(b):
            r.append(min(r[-1]+1, dp[j+1]+1, dp[j]+(0 if ai==bj else 1)))
        dp=r
    return dp[-1]


def per_calc(preds, refs):
    td=tl=0
    for p,r in zip(preds,refs):
        td+=edit_dist(p,r); tl+=max(len(r),1)
    return td/max(tl,1)


def cosine_warmup(ep, warm, total, min_r=0.01):
    if ep < warm: return max(0.01, ep/warm)
    p=(ep-warm)/max(1,total-warm)
    return max(min_r, 0.5*(1+math.cos(math.pi*p)))


# Optimizer: sadece trainable param'lar
trainable_params = [p for p in model.parameters() if p.requires_grad]
opt = AdamW(trainable_params, lr=LR, betas=(0.9, 0.98), weight_decay=0.01)
sched = LambdaLR(opt, lambda e: cosine_warmup(e, WARMUP, EPOCHS))
ctc_crit = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
scaler = torch.amp.GradScaler('cuda')

# Resume
start_ep = 1; best_per = 1.0
if not FORCE_SCRATCH:
    ckpts = ([CP] if os.path.exists(CP) else []) + glob.glob(
        f'/kaggle/input/**/{DATASET_SLUG}/**/checkpoint.pt', recursive=True)
    ckpts = sorted(set(ckpts), key=lambda p: -os.path.getmtime(p) if os.path.exists(p) else 0)
    if ckpts:
        try:
            ck = torch.load(ckpts[0], map_location=DEVICE, weights_only=False)
            model.load_state_dict(ck['model_state_dict'], strict=False)
            start_ep = ck.get('epoch', 0) + 1
            best_per = ck.get('best_per', 1.0)
            print(f'Resume: ep {ck.get(\"epoch\",0)} (best_per={best_per:.1%})')
        except Exception as e:
            print(f'Resume HATA: {e}')
"""))

cells.append(code("""print(f'\\n--- AV Cross-Attn Fusion Egitim (start_ep={start_ep}) ---\\n')
history = []

for epoch in range(start_ep, EPOCHS + 1):
    t0 = time.time()
    model.train()
    # Encoders eval modunda kalsin (BN stats degisecek olmasin)
    model.video_encoder.eval()
    model.audio_encoder.eval()

    tl_s = tc = nan = 0
    opt.zero_grad(set_to_none=True)

    for step, (vid, aud, tgts, il_v, tl, audio_attn) in enumerate(trl):
        vid = vid.to(DEVICE); aud = aud.to(DEVICE); tgts = tgts.to(DEVICE)
        il_v = il_v.to(DEVICE)
        audio_attn_dev = audio_attn.to(DEVICE) if audio_attn is not None else None

        try:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                lp, v_lens = model(vid, aud, video_il=il_v, audio_attn=audio_attn_dev)
            T_out = lp.shape[0]
            il_clamp = v_lens.clamp(max=T_out)
            loss = ctc_crit(lp.float(), tgts, il_clamp, tl) / ACCUM
        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache(); nan += 1; continue
            raise

        if torch.isnan(loss) or torch.isinf(loss):
            nan += 1; continue

        scaler.scale(loss).backward()
        if (step + 1) % ACCUM == 0 or (step + 1) == len(trl):
            scaler.unscale_(opt)
            gn = nn.utils.clip_grad_norm_(trainable_params, GCLIP)
            if torch.isfinite(gn):
                scaler.step(opt)
                tl_s += loss.item() * ACCUM
                tc += 1
            else:
                nan += 1
            scaler.update()
            opt.zero_grad(set_to_none=True)

    # Val
    model.eval()
    preds = []; refs = []; vl_s = vc = 0
    with torch.no_grad():
        for vid, aud, tgts, il_v, tl, audio_attn in vll:
            vid = vid.to(DEVICE); aud = aud.to(DEVICE)
            il_v = il_v.to(DEVICE)
            audio_attn_dev = audio_attn.to(DEVICE) if audio_attn is not None else None
            lp, v_lens = model(vid, aud, video_il=il_v, audio_attn=audio_attn_dev)
            T_out = lp.shape[0]
            il_clamp = v_lens.clamp(max=T_out)
            try:
                vl = ctc_crit(lp.float(), tgts, il_clamp, tl)
                if not (torch.isnan(vl) or torch.isinf(vl)):
                    vl_s += vl.item(); vc += 1
            except Exception: pass
            for i in range(vid.shape[0]):
                preds.append(greedy_ctc(lp[:il_clamp[i], i]))
            off = 0
            for i in range(vid.shape[0]):
                refs.append(tgts[off:off + tl[i]].tolist())
                off += tl[i].item()

    train_loss = tl_s / max(tc, 1); val_loss = vl_s / max(vc, 1)
    per_all = per_calc(preds, refs)

    if per_all < best_per:
        best_per = per_all
        torch.save(model.state_dict(), BP)
        print(f'  [BEST] PER {best_per:.1%}')

    sched.step()
    ns = f' NaN:{nan}' if nan else ''
    print(f'Ep {epoch:2d}/{EPOCHS} | TrL:{train_loss:.3f} | VaL:{val_loss:.3f} | '
          f'gap:{(train_loss-val_loss):+.2f} | PER-all:{per_all:.1%} | '
          f'{int(time.time()-t0)}s{ns}')

    if epoch % 3 == 0 or epoch == EPOCHS:
        for i in range(min(2, len(preds))):
            print(f'  H: \"{phonemes_to_text(refs[i])[:80]}\"')
            print(f'  T: \"{phonemes_to_text(preds[i])[:80]}\"')

    history.append({'epoch':epoch, 'per_all':per_all, 'val_loss':val_loss})

    torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(),
                'best_per':best_per}, CP)

    if epoch % UPLOAD_EVERY == 0:
        # Upload
        for f in glob.glob(f'{MDIR}/*.pt'):
            shutil.copy(f, f'{UDIR}/{os.path.basename(f)}')
        r = subprocess.run(['kaggle','datasets','version','-p',UDIR,
                             '-m',f'cross-attn ep{epoch}','--dir-mode','zip'],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 and ('not found' in (r.stderr or '').lower() or '404' in (r.stderr or '').lower()):
            subprocess.run(['kaggle','datasets','create','-p',UDIR,'--dir-mode','zip'],
                            capture_output=True, text=True)

print('\\n' + '='*70)
print('AV CROSS-ATTN FUSION SONUC')
print('='*70)
for h in history:
    print(f'ep {h[\"epoch\"]:>2}: PER {h[\"per_all\"]:.1%} | VaL {h[\"val_loss\"]:.3f}')
print(f'BEST: {best_per:.1%}')

print(f'\\nKarsilastirmalar:')
print(f'  v1c-disjoint (video-only):  PER ~%45')
print(f'  Late fusion alpha=0.5:        PER ~%35-40')
print(f'  Cross-attn fusion (BU):       PER {best_per:.1%} ⭐')
print(f'\\nSonraki: lippy-kenlm-6gram.ipynb -> beam search + KenLM')
"""))

# DUMP
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"YAZILDI: {OUT_PATH}")
print(f"Cell sayisi: {len(cells)}")
