"""
Generator: kaggle/lippy-audio-whisper-ctc.ipynb

Whisper-medium encoder + custom CTC head (decoder kullanilmaz).

Onemli teknik nokta:
- Whisper feature extractor SABIT 30 sec input bekler (480K sample @ 16kHz)
- Log-mel: 80 bin x 3000 frame (10ms hop)
- Encoder: 1500 frame cikti (50Hz)
- Kisa audio (5 sec) padding ile 30 sec olur, encoder hala 1500 frame uretir
- CTC icin: input_lengths = original_audio_sec * 50 (real frame count)
- Padded frame'ler CTC tarafindan zaten yoksayilir

Beklenen PER: %10-15 (Whisper YouTube domain match avantaji)

Kullanim:
    python tools/build_audio_whisper_ctc_notebook.py
"""
import json
import os

OUT_PATH = r"C:\Users\asker\Projelerim\LipPY\kaggle\lippy-audio-whisper-ctc.ipynb"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


cells = []

# =========================================================
# 0. INTRO
# =========================================================
cells.append(md("""# LipPY Audio CTC — Whisper Encoder + Custom CTC Head

**Max Effort Path 2B** — En yuksek hedef PER (%10-15) icin Whisper-medium encoder kullanildi.

## Niye Whisper?

- **Domain match:** Whisper YouTube dahil 680K saat veride egitildi -> bizim YouTube veriyle birebir uyumlu
- **Turkce kalitesi:** Whisper TR'de mevcut Wav2Vec2'den daha guclu (CV8 sadece 30+ saat, Whisper milyonlarca)
- **Audio fonem extraction:** Encoder'in son layer'i fonem-zengini features uretir

## Mimari

```
Audio (16kHz, var len 1-10s)
  -> WhisperFeatureExtractor (log-mel 80 x 3000, padded to 30s)
  -> WhisperEncoder.from_pretrained('openai/whisper-medium')
     - 24 transformer layer, hidden=1024
     - cikti: (B, 1500, 1024)
     - DECODER KULLANILMIYOR (sadece encoder + CTC)
  -> Dropout + Linear (1024 -> 31 fonem)
  -> CTC loss (input_length = real_frames, padded'ler atlanır)
```

## Onemli detay: Padding ve attention mask

Whisper SABIT 30sec input bekler. Kisa audio (5 sec) -> 25 sec sessizlik ile padded.
- Mel cikti: 3000 frame (ilk 500 real, 2500 padded)
- Encoder cikti: 1500 frame (ilk 250 real, 1250 padded)
- CTC `input_lengths = real_frames` -> padded kisim zaten yoksayilir

## Kaggle Input

- `brokyewing/lippy-test` (texts + splits)
- `brokyewing/lippy-audio-segments` (audio wav, adim5 ciktisi)
- Internet ACIK (Whisper-medium ~1.5 GB indirilir)

## Cikti

- `KULLANICI/lippy-audio-whisper-ctc-checkpoint`
- Beklenen PER: ~%10-15
"""))

# =========================================================
# 1. IMPORTS + CONFIG
# =========================================================
cells.append(code("""!pip install -q transformers torchaudio soundfile

import os, glob, json, shutil, time, math, random, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import WhisperModel, WhisperFeatureExtractor

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CPU')

# =================== CONFIG ===================
HF_MODEL = "openai/whisper-medium"     # 769M params, hidden=1024
# Alternatif: "openai/whisper-small" (244M, hidden=768) - VRAM kisitliysa

EPOCHS = 15
LR_FE = 5e-6                              # Whisper encoder (low to preserve pretrained)
LR_BE = 5e-5                              # CTC head (faster)
WARMUP = 2
GCLIP = 2.0
ACCUM = 8
BS = 2                                    # Whisper medium buyuk - kucuk BS
UPLOAD_EVERY = 5

# Whisper sabit input
WHISPER_INPUT_SEC = 30                    # Whisper SABIT 30 sec input bekler
WHISPER_FRAMES_PER_SEC = 50               # encoder output rate
WHISPER_OUTPUT_FRAMES = 1500              # 30 sec * 50 fps

# Audio constraints
SAMPLE_RATE = 16000
MIN_AUDIO_SEC = 0.5
MAX_AUDIO_SEC = 10.0                      # 30 sec'e padded, fakat real cap 10 sec

# Freeze
FREEZE_FE = True                          # Conv layer (feature_projection) donmus
FREEZE_FIRST_N = 12                       # Ilk 12 transformer layer donmus

FORCE_SCRATCH = False
MAX_RESUME_PER = 0.95

print(f'Model: {HF_MODEL}')
print(f'Egitim: EPOCHS={EPOCHS}, BS={BS}xACCUM={ACCUM} = efektif {BS*ACCUM}')
print(f'LR: FE={LR_FE}, BE={LR_BE}')
print(f'Freeze: feature_extractor={FREEZE_FE}, ilk {FREEZE_FIRST_N} encoder layer')
print(f'Audio: {SAMPLE_RATE}Hz, {MIN_AUDIO_SEC}-{MAX_AUDIO_SEC}s -> padded to {WHISPER_INPUT_SEC}s')
"""))

# =========================================================
# 2. VOCAB (V1C ile AYNI - kritik fusion icin)
# =========================================================
cells.append(md("## 1) Vocab — 31 Fonem (v1c + audio-wav2vec ile AYNI)"))

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
assert NUM_PHONEMES == 31

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
# 3. MODEL
# =========================================================
cells.append(md("""## 2) Model — Whisper Encoder + CTC Head

- WhisperModel.encoder kullanilir, decoder atilir
- Feature extractor (conv_in_1 + conv_in_2 + positional_embedding) donmus
- Ilk 12 transformer layer donmus (Whisper-medium toplam 24 layer)
- Ust 12 transformer layer fine-tune
- CTC head: Linear 1024 -> 31
"""))

cells.append(code("""class WhisperAudioModel(nn.Module):
    def __init__(self, model_name=HF_MODEL, num_phonemes=NUM_PHONEMES,
                 freeze_fe=FREEZE_FE, freeze_first_n=FREEZE_FIRST_N,
                 dropout=0.1):
        super().__init__()
        full_model = WhisperModel.from_pretrained(model_name)
        self.encoder = full_model.encoder
        del full_model  # decoder atildi, RAM tasarruflu
        torch.cuda.empty_cache()

        hidden = self.encoder.config.d_model
        n_layers = len(self.encoder.layers)
        print(f'Whisper encoder: hidden={hidden}, layers={n_layers}')

        # Freeze feature extractor (conv1 + conv2 + positional)
        if freeze_fe:
            for p in self.encoder.conv1.parameters():
                p.requires_grad = False
            for p in self.encoder.conv2.parameters():
                p.requires_grad = False
            self.encoder.embed_positions.requires_grad = False
            print(f'Conv1 + Conv2 + positional embedding DONMUS')

        # Ilk N transformer layer donmus
        for i, layer in enumerate(self.encoder.layers):
            if i < freeze_first_n:
                for p in layer.parameters():
                    p.requires_grad = False
        print(f'Ilk {freeze_first_n}/{n_layers} transformer layer DONMUS')

        # CTC head (yeni, scratch)
        self.dropout = nn.Dropout(dropout)
        self.ctc_fc = nn.Linear(hidden, num_phonemes)

        # Trainable parametre
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f'Parametre: trainable={trainable:,} / total={total:,} '
              f'({100*trainable/total:.1f}%)')

    def forward(self, input_features):
        \"\"\"
        input_features: (B, 80, 3000) - log-mel spectrogram (Whisper standart format)

        Return: log_probs (T_frames=1500, B, V) for CTC
        \"\"\"
        # Whisper encoder cikti: (B, 1500, 1024)
        out = self.encoder(input_features, return_dict=True)
        x = self.dropout(out.last_hidden_state)
        logits = self.ctc_fc(x)
        log_probs = logits.log_softmax(-1)
        return log_probs.permute(1, 0, 2)  # (T, B, V)

    @staticmethod
    def get_output_length(audio_samples):
        \"\"\"
        Real audio sample sayisindan Whisper encoder cikti uzunlugunu hesapla.
        Whisper: 30 sec input -> 1500 frame cikti (50Hz).
        Real audio: audio_samples / 16000 sec -> audio_samples / 320 frame.
        Cap at 1500.
        \"\"\"
        return min(audio_samples // 320, WHISPER_OUTPUT_FRAMES)
"""))

# =========================================================
# 4. DATA
# =========================================================
cells.append(md("""## 3) Veri — Audio + Text, Speaker-Disjoint Split

WhisperFeatureExtractor 16kHz audio'yu log-mel spectrogram'a cevirir (80 bin x 3000 frame).
30 sec'den kisa audio otomatik padded.
"""))

cells.append(code("""# Auto-discovery (audio v1 ile ayni)
AUDIO_CANDS = (
    glob.glob('/kaggle/input/**/lippy-audio-segments/**/audios', recursive=True)
    + glob.glob('/kaggle/input/**/audios', recursive=True)
)
AUDIO_DIR = AUDIO_CANDS[0] if AUDIO_CANDS else '/kaggle/input/lippy-audio-segments/audios'

TEXT_CANDS = glob.glob('/kaggle/input/**/dataset/texts', recursive=True)
TEXT_DIR = TEXT_CANDS[0] if TEXT_CANDS else '/kaggle/input/lippy-test/dataset/texts'

SPLIT_CANDS = glob.glob('/kaggle/input/**/dataset/splits', recursive=True)
SPLIT_DIR = SPLIT_CANDS[0] if SPLIT_CANDS else '/kaggle/input/lippy-test/dataset/splits'

print(f'AUDIO_DIR: {AUDIO_DIR}')
print(f'TEXT_DIR:  {TEXT_DIR}')
print(f'SPLIT_DIR: {SPLIT_DIR}')

# Whisper feature extractor (her sample icin run edilecek)
feature_extractor = WhisperFeatureExtractor.from_pretrained(HF_MODEL)
print(f'Whisper feature extractor: sr={feature_extractor.sampling_rate}, '
      f'n_mels={feature_extractor.feature_size}')
"""))

cells.append(code("""class WhisperAudioDataset(Dataset):
    def __init__(self, split, fe):
        self.train = split == 'train'
        self.fe = fe
        split_path = os.path.join(SPLIT_DIR, f'{split}.txt')
        with open(split_path, 'r', encoding='utf-8') as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self.ids = []
        skipped_no_audio = 0
        skipped_no_text = 0
        for s in all_ids:
            wav_p = os.path.join(AUDIO_DIR, f'{s}.wav')
            txt_p = os.path.join(TEXT_DIR, f'{s}.txt')
            if not os.path.exists(wav_p):
                skipped_no_audio += 1; continue
            if not os.path.exists(txt_p):
                skipped_no_text += 1; continue
            self.ids.append(s)

        kept = 100.0 * len(self.ids) / max(len(all_ids), 1)
        print(f'  [{split}] {len(self.ids)}/{len(all_ids)} ({kept:.1f}%) | '
              f'audio_yok={skipped_no_audio}, text_yok={skipped_no_text}')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        for offset in range(len(self.ids)):
            s = self.ids[(i + offset) % len(self.ids)]
            try:
                wav_p = os.path.join(AUDIO_DIR, f'{s}.wav')
                wav, sr = torchaudio.load(wav_p)
                if sr != SAMPLE_RATE:
                    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                wav = wav.squeeze(0).float().numpy()

                # Length filter
                n_samples = len(wav)
                if n_samples < SAMPLE_RATE * MIN_AUDIO_SEC:
                    continue
                if n_samples > SAMPLE_RATE * MAX_AUDIO_SEC:
                    target = int(SAMPLE_RATE * MAX_AUDIO_SEC)
                    if self.train:
                        start = random.randint(0, n_samples - target)
                    else:
                        start = (n_samples - target) // 2
                    wav = wav[start:start + target]
                    n_samples = target

                # Whisper feature extraction (otomatik padding to 30s)
                # input_features: (1, 80, 3000)
                features = self.fe(wav, sampling_rate=SAMPLE_RATE,
                                    return_tensors="pt").input_features
                features = features.squeeze(0).float()  # (80, 3000)

                # Real output frame count (encoder downsamples 2x more)
                real_frames = WhisperAudioModel.get_output_length(n_samples)

                # Text
                txt_p = os.path.join(TEXT_DIR, f'{s}.txt')
                with open(txt_p, 'r', encoding='utf-8') as f:
                    txt = f.read().strip()
                lab = torch.tensor(text_to_phonemes(txt), dtype=torch.long)
                if len(lab) == 0:
                    continue

                # CTC safety: real_frames >= 2 * text_length
                if real_frames < 2 * len(lab):
                    continue

                return features, lab, real_frames
            except Exception as e:
                continue
        raise RuntimeError('Audio yuklenemedi')


def collate(batch):
    feats, labs, real_lens = zip(*batch)
    feats = torch.stack(feats, dim=0)        # (B, 80, 3000)
    il = torch.tensor(real_lens, dtype=torch.long)
    tl = torch.tensor([l.shape[0] for l in labs], dtype=torch.long)
    return feats, torch.cat(labs), il, tl


trl = DataLoader(WhisperAudioDataset('train', feature_extractor),
                  batch_size=BS, shuffle=True, collate_fn=collate,
                  num_workers=2, pin_memory=True)
vll = DataLoader(WhisperAudioDataset('val', feature_extractor),
                  batch_size=BS, shuffle=False, collate_fn=collate,
                  num_workers=2, pin_memory=True)
print(f'Train: {len(trl)} batch | Val: {len(vll)} batch')
"""))

# =========================================================
# 5. TRAINING SETUP
# =========================================================
cells.append(md("## 4) Egitim setup — Optimizer, scheduler, metrics, resume"))

cells.append(code("""MDIR = '/kaggle/working/ckpt_audio_whisper_ctc'
UDIR = '/kaggle/working/upload_audio_whisper_ctc'
os.makedirs(MDIR, exist_ok=True)
os.makedirs(UDIR, exist_ok=True)
CP = f'{MDIR}/checkpoint.pt'
BP = f'{MDIR}/best_model.pt'

# Kaggle credentials
KAGGLE_USER = os.environ.get('KAGGLE_USERNAME', 'UNKNOWN')
try:
    from kaggle_secrets import UserSecretsClient
    _us = UserSecretsClient()
    KAGGLE_USER = _us.get_secret('KAGGLE_USERNAME')
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USER
    os.environ['KAGGLE_KEY'] = _us.get_secret('KAGGLE_KEY')
except Exception as e:
    print(f'UYARI: Secrets okunamadi: {e}')

DATASET_SLUG = 'lippy-audio-whisper-ctc-checkpoint'
DATASET_ID = f'{KAGGLE_USER}/{DATASET_SLUG}'
with open(f'{UDIR}/dataset-metadata.json', 'w') as f:
    json.dump({
        'title': 'LipPY Audio Whisper CTC v1 (Phoneme)',
        'id': DATASET_ID,
        'licenses': [{'name': 'CC0-1.0'}],
    }, f)
print(f'Upload: {DATASET_ID}')


def greedy_ctc(lp):
    ids = lp.argmax(-1).tolist()
    d = []
    p = -1
    for i in ids:
        if i != p and i != BLANK_IDX:
            d.append(i)
        p = i
    return d


def edit_dist(a, b):
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for ai in a:
        r = [dp[0] + 1]
        for j, bj in enumerate(b):
            r.append(min(r[-1] + 1, dp[j + 1] + 1, dp[j] + (0 if ai == bj else 1)))
        dp = r
    return dp[-1]


def per_calc(preds, refs):
    td = tl = 0
    for p, r in zip(preds, refs):
        td += edit_dist(p, r)
        tl += max(len(r), 1)
    return td / max(tl, 1)


def metrics(preds, refs):
    per_all = per_calc(preds, refs)
    preds_ph = [[t for t in p if t != SPACE_IDX] for p in preds]
    refs_ph = [[t for t in r if t != SPACE_IDX] for r in refs]
    per_phoneme = per_calc(preds_ph, refs_ph)
    tp = fp = fn = 0
    for p, r in zip(preds, refs):
        ps = sum(1 for t in p if t == SPACE_IDX)
        rs = sum(1 for t in r if t == SPACE_IDX)
        tp += min(ps, rs); fp += max(0, ps - rs); fn += max(0, rs - ps)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-6)
    return {'per_all': per_all, 'per_phoneme': per_phoneme, 'space_f1': f1}


def cosine_warmup(ep, warm, total, min_r=0.01):
    if ep < warm:
        return max(0.01, ep / warm)
    p = (ep - warm) / max(1, total - warm)
    return max(min_r, 0.5 * (1 + math.cos(math.pi * p)))


def upload(ep):
    for f in glob.glob(f'{MDIR}/*.pt'):
        shutil.copy(f, f'{UDIR}/{os.path.basename(f)}')
    r = subprocess.run(['kaggle', 'datasets', 'version', '-p', UDIR,
                         '-m', f'whisper-ctc ep{ep}', '--dir-mode', 'zip'],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode == 0:
        print(f'[upload ep{ep}] VERSION OK')
    else:
        stderr = (r.stderr or '').lower()
        if 'not found' in stderr or '404' in stderr:
            cr = subprocess.run(['kaggle', 'datasets', 'create', '-p', UDIR,
                                  '--dir-mode', 'zip'],
                                 capture_output=True, text=True, timeout=1800)
            print(f'[upload ep{ep}] CREATE rc={cr.returncode}')
        else:
            print(f'[upload ep{ep}] FAIL rc={r.returncode}')
"""))

cells.append(code("""model = WhisperAudioModel().to(DEVICE)

# Optimizer 2 grup
fe_p = [p for n, p in model.named_parameters()
        if p.requires_grad and n.startswith('encoder.')]
be_p = [p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith('encoder.')]

opt = AdamW([
    {'params': fe_p, 'lr': LR_FE},
    {'params': be_p, 'lr': LR_BE},
], betas=(0.9, 0.98), weight_decay=0.01)

sched = LambdaLR(opt, [
    lambda e: cosine_warmup(e, WARMUP, EPOCHS),
    lambda e: cosine_warmup(e, WARMUP, EPOCHS),
])

ctc_crit = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
scaler = torch.amp.GradScaler('cuda')

print(f'FE params (unfrozen encoder): {sum(p.numel() for p in fe_p):,}')
print(f'BE params (CTC head):         {sum(p.numel() for p in be_p):,}')

# Resume
start_ep = 1
best_per = 1.0
best_f1 = 0.0

if not FORCE_SCRATCH:
    ckpts = ([CP] if os.path.exists(CP) else []) + glob.glob(
        f'/kaggle/input/**/{DATASET_SLUG}/**/checkpoint.pt', recursive=True)
    ckpts = sorted(set(ckpts), key=lambda p: -os.path.getmtime(p) if os.path.exists(p) else 0)
    if ckpts:
        print(f'Resume candidate: {ckpts[0]}')
        try:
            ck = torch.load(ckpts[0], map_location=DEVICE, weights_only=False)
            cand_per = ck.get('best_per', 1.0)
            cand_ep = ck.get('epoch', 0)
            if cand_per > MAX_RESUME_PER:
                print(f'UYARI: best_per={cand_per:.1%} > MAX, ATLANIYOR')
            else:
                model.load_state_dict(ck['model_state_dict'])
                start_ep = cand_ep + 1
                best_per = cand_per
                best_f1 = ck.get('best_f1', 0.0)
                try:
                    opt.load_state_dict(ck['opt'])
                    sched.load_state_dict(ck['sched'])
                except Exception:
                    pass
                if 'scaler' in ck:
                    scaler.load_state_dict(ck['scaler'])
                print(f'Resume OK: epoch {cand_ep} (best_per={best_per:.1%})')
        except Exception as e:
            print(f'Resume HATA: {e}')
"""))

# =========================================================
# 6. TRAINING LOOP
# =========================================================
cells.append(md("""## 5) Egitim Loop

AMP fp16, gradient accumulation, her epoch best ckpt save, her 5 ep upload.

NOTE: Whisper-medium VRAM intensive (1.5GB model + activations). BS=2 onerilen,
T4 GPU'da ~12GB free RAM kullanir. Eger OOM olursa BS=1, ACCUM=16 dene.
"""))

cells.append(code("""print(f'\\n--- Audio Whisper-CTC v1 Egitim (start_ep={start_ep}) ---\\n')
epoch_history = []

for epoch in range(start_ep, EPOCHS + 1):
    t0 = time.time()

    # Train
    model.train()
    tl_s = tc = nan = 0
    opt.zero_grad(set_to_none=True)

    for step, (feats, tgts, il, tl) in enumerate(trl):
        feats = feats.to(DEVICE)
        tgts = tgts.to(DEVICE)
        il_dev = il.to(DEVICE)

        try:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                lp = model(feats)  # (T=1500, B, V)
            T_out = lp.shape[0]
            il_clamp = il_dev.clamp(max=T_out)
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
            gn = nn.utils.clip_grad_norm_(model.parameters(), GCLIP)
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
    preds = []
    refs = []
    vl_s = vc = 0
    with torch.no_grad():
        for feats, tgts, il, tl in vll:
            feats = feats.to(DEVICE)
            il_dev = il.to(DEVICE)
            lp = model(feats)
            T_out = lp.shape[0]
            il_clamp = il_dev.clamp(max=T_out)
            try:
                vl = ctc_crit(lp.float(), tgts, il_clamp, tl)
                if not (torch.isnan(vl) or torch.isinf(vl)):
                    vl_s += vl.item(); vc += 1
            except Exception:
                pass
            for i in range(feats.shape[0]):
                preds.append(greedy_ctc(lp[:il_clamp[i], i]))
            off = 0
            for i in range(feats.shape[0]):
                refs.append(tgts[off:off + tl[i]].tolist())
                off += tl[i].item()

    train_loss = tl_s / max(tc, 1)
    val_loss = vl_s / max(vc, 1)
    m = metrics(preds, refs)

    if m['per_all'] < best_per:
        best_per = m['per_all']
        torch.save(model.state_dict(), BP)
        print(f'  [BEST] PER {best_per:.1%} kaydedildi')
    if m['space_f1'] > best_f1:
        best_f1 = m['space_f1']

    sched.step()
    ns = f' NaN:{nan}' if nan else ''
    gap = train_loss - val_loss
    print(f'Ep {epoch:2d}/{EPOCHS} | TrL:{train_loss:.3f} | VaL:{val_loss:.3f} | '
          f'gap:{gap:+.2f} | PER-all:{m[\"per_all\"]:.1%} ph:{m[\"per_phoneme\"]:.1%} | '
          f'SP F1:{m[\"space_f1\"]:.2f} | {int(time.time() - t0)}s{ns}')

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        for i in range(min(2, len(preds))):
            print(f'  H: \"{phonemes_to_text(refs[i])[:80]}\"')
            print(f'  T: \"{phonemes_to_text(preds[i])[:80]}\"')

    epoch_history.append({
        'epoch': epoch, 'per_all': m['per_all'], 'per_phoneme': m['per_phoneme'],
        'space_f1': m['space_f1'], 'val_loss': val_loss,
    })

    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'opt': opt.state_dict(), 'sched': sched.state_dict(),
        'scaler': scaler.state_dict(),
        'best_per': best_per, 'best_f1': best_f1,
    }, CP)

    if epoch % UPLOAD_EVERY == 0:
        upload(epoch)

upload(EPOCHS)

print('\\n' + '=' * 70)
print('AUDIO WHISPER-CTC v1 EGITIM SONUC')
print('=' * 70)
print(f'{\"Epoch\":>6} | {\"PER-all\":>8} | {\"PER-ph\":>8} | {\"SP F1\":>6} | {\"VaL\":>6}')
print('-' * 50)
for h in epoch_history:
    print(f'{h[\"epoch\"]:>6} | {h[\"per_all\"]:>7.1%} | {h[\"per_phoneme\"]:>7.1%} | '
          f'{h[\"space_f1\"]:>6.2f} | {h[\"val_loss\"]:>6.3f}')
print('-' * 50)
print(f'{\"BEST\":>6} | {best_per:>7.1%}')

print(f'\\nKarsilastir:')
print(f'  Wav2Vec2-Turkish freeze-12: PER ~%20')
print(f'  Wav2Vec2-Turkish full:      PER ~%15')
print(f'  Whisper-medium CTC (BU):    PER {best_per:.1%}')
print(f'\\n  Fusion icin en iyi audio modeli sec ve cross-attn notebook\\'una geç.')
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
