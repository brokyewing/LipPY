"""
AV-Fusion Inference (model id: av-fusion) — Video + Audio → Türkçe metin.

Mimari (kaggle/train-av-fusion.ipynb ile birebir):
  VideoEncoder (dim_ff=2048, frozen feature extractor)
  AudioEncoderWhisper (Whisper-medium, frozen feature extractor)
  SymmetricFusion (concat + 8-layer shared Transformer + audio downsample)
  HybridHead (CTC branch + 4-layer attention decoder)
Checkpoint: checkpoints/av-fusion/best_model.pt  (PER %24.7)

Ses yoksa: modality dropout sayesinde sıfır-mel ile video-only çalışır (kalite düşer).

Decode modlari:
  greedy  : CTC argmax (LM yok, en hizli — baseline)
  beam_lm : CTC + pyctcdecode beam + KenLM (ortak domain LM, 31 fonem) — varsayılan
  attn    : attention decoder seq2seq (autoregressive greedy)

Kullanım:
  from backend.av_fusion import infer_av_fusion, is_fusion_available

  if is_fusion_available():
      success, text, details = infer_av_fusion(video_bytes, audio_bytes, decode_mode="beam_lm")
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.vocab import BLANK_IDX, indices_to_text, get_pyctcdecode_labels
from backend.roi import video_to_roi_array
from backend.decode_utils import greedy_ctc_decode

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ v1cc video normalize (v1c'den FARKLI!) ============
# v1c: 0.421 / 0.165, v1cc: 0.495 / 0.200 — v3 video encoder v1cc'dir.
MEAN_V = 0.495
STD_V = 0.200
CROP = 88

SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = "openai/whisper-medium"

# ============ Vocab / mimari sabitleri (notebook cell-1, cell-3 ile birebir) ============
NUM_PHONEMES = 31
SPACE_IDX = 1
BOS_IDX = NUM_PHONEMES          # 31 — decoder <bos>/<eos> ortak token
EOS_IDX = NUM_PHONEMES          # 31
PAD_IDX = BLANK_IDX             # 0
DECODER_VOCAB = NUM_PHONEMES + 1  # 32

D_MODEL = 512
NHEAD = 8
N_FUSION_LAYERS = 8
N_DECODER_LAYERS = 4
FUSION_DROPOUT = 0.1

V_D_MODEL = 512
V_ENC_LAYERS = 6
V_NHEAD = 8
V_DIM_FF = 2048                 # v1cc (v1c=1024)

FUSION_CHECKPOINT = ROOT / "checkpoints" / "av-fusion" / "best_model.pt"

# KenLM (ortak domain LM — ayni 31 fonem vocab)
FUSION_KENLM_CANDIDATES = [
    ROOT / "checkpoints" / "kenlm" / "lm.klm",
    ROOT / "checkpoints" / "av-fusion" / "lm.klm",
]

ATTN_MAX_LEN = 200              # autoregressive decode uzunluk siniri

_fusion_model = None
_whisper_fe = None
_beam_decoders: dict[str, object] = {}   # cache: "lm" / "nolm"

# Lexicon post-correction (opsiyonel ekstra katman) — ortak unigrams
LEXICON_CANDIDATES = [
    ROOT / "checkpoints" / "kenlm" / "unigrams.txt",
    ROOT / "checkpoints" / "av-fusion" / "unigrams.txt",
]
_lexicon_set = None


# ====================================================================
# Mimari — notebook cell-5, cell-7, cell-9 ile birebir
# ====================================================================
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


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
        ds = nn.Sequential(nn.Conv2d(ic, oc, 1, stride=s, bias=False), nn.BatchNorm2d(oc)) if s != 1 or ic != oc else None
        return nn.Sequential(BasicBlockSiLU(ic, oc, s, ds), *[BasicBlockSiLU(oc, oc) for _ in range(n - 1)])

    def forward(self, x):
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class VideoEncoder(nn.Module):
    """v1cc encoder. dim_ff=2048. get_features (B, T_v, 512) dondurur."""
    def __init__(self):
        super().__init__()
        from torchaudio.models import Conformer
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64), SiLU(),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNetSiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Identity()
        self.conformer = Conformer(
            input_dim=V_D_MODEL, num_heads=V_NHEAD, ffn_dim=V_DIM_FF,
            num_layers=V_ENC_LAYERS, depthwise_conv_kernel_size=31, dropout=0.1)
        self.ctc_fc = nn.Linear(V_D_MODEL, NUM_PHONEMES)   # ckpt key uyumu (inference'ta kullanilmaz)

    def get_features(self, x, il=None):
        B, T, H, W = x.shape
        x = x.unsqueeze(1)
        x = self.frontend3D(x)
        B2, C, To, H2, W2 = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B2 * To, C, H2, W2)
        x = self.avg_pool(self.trunk(x)).view(B2, To, -1)
        x = self.proj(x)
        if il is None:
            il = torch.full((B2,), To, dtype=torch.long, device=x.device)
        x, lengths = self.conformer(x, lengths=il)
        return x, lengths


class AudioEncoderWhisper(nn.Module):
    """Whisper-medium encoder, dynamic-length forward (pos embed dilimlenir)."""
    def __init__(self, model_name=WHISPER_MODEL_NAME):
        super().__init__()
        from transformers import WhisperModel
        full = WhisperModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.encoder = full.encoder
        self.encoder_dropout_p = 0.1
        del full
        self.hidden = self.encoder.config.d_model   # medium=1024
        self.n_layers = len(self.encoder.layers)     # medium=24
        self.ctc_fc = nn.Linear(self.hidden, NUM_PHONEMES)  # ckpt key uyumu

    def get_features(self, input_features):
        x = F.gelu(self.encoder.conv1(input_features))
        x = F.gelu(self.encoder.conv2(x))
        x = x.permute(0, 2, 1)
        T_out = x.shape[1]
        embed_pos = self.encoder.embed_positions.weight[:T_out]
        x = x + embed_pos
        x = F.dropout(x, p=self.encoder_dropout_p, training=self.training)
        for layer in self.encoder.layers:
            x = layer(x, None, None)[0]
        x = self.encoder.layer_norm(x)
        return x


class SymmetricFusion(nn.Module):
    """Concat (V + downsampled A) -> Linear(2D->D) -> 8-layer shared Transformer.
    Audio 50Hz -> downsample 2x -> video temporal uzunluguna interp.
    Modality dropout train-only (eval'de no-op)."""
    def __init__(self, d=D_MODEL, audio_dim=1024, video_dim=V_D_MODEL,
                 n_layers=N_FUSION_LAYERS, nhead=NHEAD, dropout=FUSION_DROPOUT):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, d)
        self.audio_proj = nn.Linear(audio_dim, d)
        self.audio_downsample = nn.Conv1d(d, d, kernel_size=2, stride=2)
        self.combine = nn.Linear(2 * d, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=2048, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.shared = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def _align_audio(self, a_feat, target_T):
        a = a_feat.transpose(1, 2)              # (B, D, T_a)
        a = self.audio_downsample(a)            # (B, D, T_a/2)
        if a.shape[-1] != target_T:
            a = F.interpolate(a, size=target_T, mode='linear', align_corners=False)
        return a.transpose(1, 2)                # (B, target_T, D)

    def forward(self, v_feat, a_feat, v_mask=None):
        v = self.video_proj(v_feat)             # (B, T_v, D)
        a_proj = self.audio_proj(a_feat)        # (B, T_a, D)
        a_aligned = self._align_audio(a_proj, v.shape[1])
        # eval'de modality dropout YOK (self.training=False)
        fused = torch.cat([v, a_aligned], dim=-1)
        fused = self.combine(fused)
        fused = self.dropout(fused)
        return self.shared(fused, src_key_padding_mask=v_mask)


class HybridHead(nn.Module):
    """CTC branch + Attention seq2seq decoder."""
    def __init__(self, d=D_MODEL, vocab=NUM_PHONEMES, decoder_vocab=DECODER_VOCAB,
                 n_decoder_layers=N_DECODER_LAYERS, nhead=NHEAD, dropout=FUSION_DROPOUT):
        super().__init__()
        self.ctc_proj = nn.Linear(d, vocab)
        self.tgt_embed = nn.Embedding(decoder_vocab, d, padding_idx=PAD_IDX)
        self.pos_embed = nn.Embedding(512, d)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=2048, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)
        self.attn_out = nn.Linear(d, decoder_vocab)

    def forward_ctc(self, encoder_out):
        """(B, T, D) -> (T, B, vocab) log_probs."""
        return self.ctc_proj(encoder_out).log_softmax(-1).permute(1, 0, 2)

    def forward_attn(self, encoder_out, tgt_in, memory_key_padding_mask=None):
        """Teacher-forcing / autoregressive decoder forward.
        encoder_out (B,T_v,D), tgt_in (B,L) -> (B, L, decoder_vocab)."""
        B, L = tgt_in.shape
        positions = torch.arange(L, device=tgt_in.device).unsqueeze(0).expand(B, -1)
        tgt = self.tgt_embed(tgt_in) + self.pos_embed(positions)
        causal = torch.triu(torch.ones(L, L, device=tgt_in.device), diagonal=1).bool()
        out = self.decoder(tgt, encoder_out, tgt_mask=causal,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_key_padding_mask=(tgt_in == PAD_IDX))
        return self.attn_out(out)


class AVFusionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoderWhisper()
        self.fusion = SymmetricFusion(
            d=D_MODEL, audio_dim=self.audio_encoder.hidden, video_dim=V_D_MODEL)
        self.head = HybridHead(d=D_MODEL)

    def encode(self, video_input, audio_input, video_il=None):
        """Fusion ciktisini + v_mask + v_lens dondurur (CTC ve attn ortak)."""
        v_feat, v_lens = self.video_encoder.get_features(video_input, il=video_il)
        a_feat = self.audio_encoder.get_features(audio_input)
        B, T_v, _ = v_feat.shape
        v_mask = torch.arange(T_v, device=v_feat.device).unsqueeze(0) >= v_lens.unsqueeze(1)
        fused = self.fusion(v_feat, a_feat, v_mask=v_mask)
        return fused, v_mask, v_lens


# ====================================================================
# Yukleme
# ====================================================================
def is_fusion_available() -> bool:
    return FUSION_CHECKPOINT.is_file()


def _get_whisper_fe():
    global _whisper_fe
    if _whisper_fe is None:
        from transformers import WhisperFeatureExtractor
        _whisper_fe = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL_NAME)
    return _whisper_fe


def _load_fusion_model() -> Optional[AVFusionV3]:
    global _fusion_model
    if _fusion_model is not None:
        return _fusion_model
    if not FUSION_CHECKPOINT.is_file():
        return None

    model = AVFusionV3().to(DEVICE)
    state = torch.load(str(FUSION_CHECKPOINT), map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    result = model.load_state_dict(state, strict=False)
    # Sessiz basarisizligi yakala: fusion/head yuklenmediyse cikti cop olur
    critical_missing = [k for k in result.missing_keys
                        if k.startswith(("fusion.", "head."))]
    if critical_missing:
        print(f"[AV-Fusion] !!! UYARI: kritik agirlik eksik ({len(critical_missing)} key): "
              f"{critical_missing[:3]} ... — checkpoint v3 mimarisiyle uyumsuz olabilir")
    if result.unexpected_keys:
        print(f"[AV-Fusion] beklenmeyen {len(result.unexpected_keys)} key atlandi "
              f"(ornek: {result.unexpected_keys[:2]})")
    model.eval()
    _fusion_model = model
    return model


def _get_fusion_beam_decoder(with_lm: bool = True):
    """pyctcdecode beam decoder (v3 CTC ciktisi icin, v1c KenLM ile)."""
    cache_key = "lm" if with_lm else "nolm"
    if cache_key in _beam_decoders:
        return _beam_decoders[cache_key], None
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        return None, "pyctcdecode kurulu degil (pip install pyctcdecode)"

    labels = get_pyctcdecode_labels(NUM_PHONEMES)

    if not with_lm:
        try:
            decoder = build_ctcdecoder(labels)
        except Exception as e:
            return None, f"Beam decoder olusturulamadi: {e}"
        _beam_decoders[cache_key] = decoder
        return decoder, None

    # KenLM dosyasini bul
    lm_path = next((str(p) for p in FUSION_KENLM_CANDIDATES if p.is_file()), None)
    if not lm_path:
        return None, "KenLM bulunamadi (checkpoints/v1c/lm.klm veya av-fusion/lm.klm)"

    # Unigrams: lm.klm yaninda unigrams.txt ara (v1c paylasimi)
    unigrams = None
    lm_dir = Path(lm_path).parent
    for ug_name in ["unigrams.txt", "unigrams_filtered.txt"]:
        ug = lm_dir / ug_name
        if ug.is_file():
            try:
                with open(ug, "r", encoding="utf-8") as f:
                    unigrams = [l.strip() for l in f if l.strip()]
                print(f"[AV-Fusion Decoder] Unigrams: {ug} ({len(unigrams)})")
                break
            except Exception:
                pass
    if unigrams is None:
        print("[AV-Fusion Decoder] UYARI: unigrams.txt yok, KenLM kalitesi dusuk olabilir")

    alpha = float(os.environ.get("LIPPY_KENLM_ALPHA", 0.5))
    beta = float(os.environ.get("LIPPY_KENLM_BETA", 0.5))
    try:
        decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path,
                                   alpha=alpha, beta=beta, unigrams=unigrams)
    except Exception as e:
        return None, f"KenLM decoder olusturulamadi: {e}"
    _beam_decoders[cache_key] = decoder
    return decoder, None


# ====================================================================
# Decode
# ====================================================================
def attn_greedy_decode(model: AVFusionV3, fused: torch.Tensor,
                       v_mask: torch.Tensor, max_len: int = ATTN_MAX_LEN) -> list[int]:
    """Attention decoder autoregressive greedy decode (batch=1).
    fused (1, T_v, D), v_mask (1, T_v). BOS/EOS/SPACE handle eder."""
    device = fused.device
    tgt = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)
    out_ids: list[int] = []
    for _ in range(max_len):
        logits = model.head.forward_attn(fused, tgt, memory_key_padding_mask=v_mask)
        next_tok = int(logits[0, -1].argmax(-1).item())
        if next_tok == EOS_IDX:
            break
        out_ids.append(next_tok)
        tgt = torch.cat([tgt, torch.tensor([[next_tok]], device=device)], dim=1)
    return out_ids


# ---- Lexicon post-correction (opsiyonel ekstra katman) ----
def _load_lexicon():
    """v1c unigrams.txt'yi set olarak yukle (cache'li). Yoksa bos set (no-op)."""
    global _lexicon_set
    if _lexicon_set is not None:
        return _lexicon_set
    for p in LEXICON_CANDIDATES:
        if p.is_file():
            try:
                with open(p, encoding="utf-8") as f:
                    _lexicon_set = set(l.strip().lower() for l in f if l.strip())
                print(f"[Lexicon] {len(_lexicon_set)} kelime yuklendi: {p}")
                return _lexicon_set
            except Exception as e:
                print(f"[Lexicon] okuma hatasi: {e}")
    _lexicon_set = set()
    print("[Lexicon] UYARI: unigrams.txt yok -> lexicon correction no-op")
    return _lexicon_set


def _edit_distance(a, b, max_d=2):
    """Levenshtein, erken cikisli (max_d asarsa max_d+1 doner)."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    m = len(b)
    dp = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        row_min = dp[0]
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ca == cb else 1))
            prev = cur
            if dp[j] < row_min:
                row_min = dp[j]
        if row_min > max_d:
            return max_d + 1
    return dp[m]


def _nearest_word(w, vocab, max_edit=2):
    """Sozlukte en yakin kelime (edit<=max_edit). Belirsiz/uzaksa None."""
    best_d, best_w = max_edit + 1, None
    n = len(w)
    for v in vocab:
        if abs(len(v) - n) > max_edit:
            continue
        d = _edit_distance(w, v, max_edit)
        if d < best_d:
            best_d, best_w = d, v
            if best_d == 1:   # 1'den iyisi yok (0 = zaten sozlukte)
                break
    return best_w if best_d <= max_edit else None


def lexicon_correct(text, max_edit=2, min_len=3):
    """Sozluk tabanli post-correction.
    Kurallar: kelime sozlukte VARSA veya kisa (<min_len) ise DOKUNMA.
    Sozlukte yok + en yakin (edit<=max_edit) varsa duzelt; yoksa DOKUNMA.
    Returns (corrected_text, n_fixed)."""
    vocab = _load_lexicon()
    if not vocab:
        return text, 0
    out, n_fixed = [], 0
    for w in text.split():
        if len(w) < min_len or w in vocab:
            out.append(w)
        else:
            nw = _nearest_word(w, vocab, max_edit)
            if nw and nw != w:
                out.append(nw)
                n_fixed += 1
            else:
                out.append(w)
    return " ".join(out), n_fixed


# ====================================================================
# Audio I/O (av_fusion.py ile ayni mantik — bagimsiz kopyalar)
# ====================================================================
_last_ffmpeg_error: str = ""
_ffmpeg_exe: Optional[str] = None


def _get_ffmpeg_exe() -> str:
    global _ffmpeg_exe
    if _ffmpeg_exe is not None:
        return _ffmpeg_exe
    try:
        import imageio_ffmpeg
        _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[ffmpeg] imageio-ffmpeg binary: {_ffmpeg_exe}")
        return _ffmpeg_exe
    except ImportError:
        pass
    _ffmpeg_exe = "ffmpeg"
    return _ffmpeg_exe


def _extract_audio_from_video(video_path: str) -> Optional[str]:
    import subprocess
    global _last_ffmpeg_error
    _last_ffmpeg_error = ""
    ffmpeg_exe = _get_ffmpeg_exe()
    audio_path = video_path + ".audio.wav"
    try:
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", str(SAMPLE_RATE), "-ac", "1", audio_path],
            capture_output=True, timeout=30, text=True,
        )
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            return audio_path
        stderr_tail = (result.stderr or "")[-500:].strip()
        _last_ffmpeg_error = f"rc={result.returncode}, stderr_tail: {stderr_tail}"
        print(f"[ffmpeg HATA] {_last_ffmpeg_error}")
    except FileNotFoundError:
        _last_ffmpeg_error = (f"ffmpeg binary bulunamadi: {ffmpeg_exe} "
                              "(pip install imageio-ffmpeg ile yukleyin)")
        print(f"[ffmpeg HATA] {_last_ffmpeg_error}")
    except subprocess.TimeoutExpired:
        _last_ffmpeg_error = "ffmpeg 30sn'de tamamlanmadi (timeout)"
        print(f"[ffmpeg HATA] {_last_ffmpeg_error}")
    except Exception as e:
        _last_ffmpeg_error = f"subprocess hatasi: {type(e).__name__}: {e}"
        print(f"[ffmpeg HATA] {_last_ffmpeg_error}")
    return None


def _load_wav_any(audio_path: str) -> tuple[np.ndarray, int]:
    try:
        import torchaudio
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav.squeeze(0).float().numpy(), int(sr)
    except Exception as e_ta:
        print(f"[audio] torchaudio.load basarisiz: {e_ta}. soundfile deneniyor...")
    try:
        import soundfile as sf
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr)
    except Exception as e_sf:
        raise RuntimeError(f"Ses dosyasi okunamadi (torchaudio + soundfile basarisiz). "
                           f"Path: {audio_path}, soundfile hata: {e_sf}")


def _extract_audio_features_from_file(audio_path: str) -> Optional[torch.Tensor]:
    fe = _get_whisper_fe()
    wav, sr = _load_wav_any(audio_path)
    if sr != SAMPLE_RATE:
        import torchaudio
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, SAMPLE_RATE)
        wav = wav_t.squeeze(0).numpy()
    if len(wav) < SAMPLE_RATE * 0.5:
        return None
    features = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                  padding=False, return_attention_mask=False).input_features
    return features.float()


def _extract_audio_features(audio_bytes: bytes) -> Optional[torch.Tensor]:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        return _extract_audio_features_from_file(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _normalize_roi(roi: np.ndarray) -> torch.Tensor:
    x = roi.astype(np.float32) / 255.0
    T, H, W = x.shape
    top = (H - CROP) // 2
    left = (W - CROP) // 2
    x = x[:, top:top + CROP, left:left + CROP]
    x = (x - MEAN_V) / STD_V
    return torch.from_numpy(x.copy()).float()


# ====================================================================
# Inference
# ====================================================================
def infer_av_fusion(
    video_bytes: bytes,
    audio_bytes: Optional[bytes] = None,
    decode_mode: str = "greedy",
) -> tuple[bool, Optional[str], dict]:
    """AV-Fusion inference (video + ses → metin).

    Args:
        video_bytes: video ham bayt
        audio_bytes: ses ham bayt (None ise video'dan ffmpeg ile cikarilir)
        decode_mode: 'greedy' | 'beam_lm' | 'beam' | 'attn' | 'auto'
    """
    t0 = time.perf_counter()
    if decode_mode == "auto":
        decode_mode = "beam_lm"   # av-fusion en iyi mod (test edildi)
    # "_lex" suffix -> base mode + lexicon post-correction (opsiyonel ekstra katman)
    apply_lex = decode_mode.endswith("_lex")
    base_mode = decode_mode[:-4] if apply_lex else decode_mode
    details = {
        "mode": "av-fusion",
        "decode_mode": decode_mode,
        "frames": 0,
        "roi_shape": None,
        "total_ms": None,
        "error": None,
    }

    model = _load_fusion_model()
    if model is None:
        details["error"] = "AV-Fusion checkpoint bulunamadı (checkpoints/av-fusion/best_model.pt)"
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_audio = None
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()

        roi = video_to_roi_array(tmp.name)
        if roi is None:
            details["error"] = "Video işlenemedi (yüz bulunamadı)"
            details["total_ms"] = int((time.perf_counter() - t0) * 1000)
            return False, None, details

        T, H, W = roi.shape
        details["frames"] = int(T)
        details["roi_shape"] = [int(T), int(H), int(W)]

        if audio_bytes is not None:
            audio_features = _extract_audio_features(audio_bytes)
        else:
            tmp_audio = _extract_audio_from_video(tmp.name)
            if tmp_audio is None:
                err_detail = f" ({_last_ffmpeg_error})" if _last_ffmpeg_error else ""
                details["error"] = f"Video'dan ses çıkarılamadı{err_detail}"
                details["ffmpeg_error"] = _last_ffmpeg_error
                details["total_ms"] = int((time.perf_counter() - t0) * 1000)
                return False, None, details
            audio_features = _extract_audio_features_from_file(tmp_audio)

        if audio_features is None:
            # Sessiz video (ses yok/cikarilamadi): sifir mel ile devam.
            # AV-Fusion modality dropout (0.5) ile egitildi -> audio olmadan video-only calisir.
            # NOT: audio avantaji yok -> kalite video-only (v1cc) seviyesine duser.
            T_frames = roi.shape[0]
            T_mel = max(4, (T_frames * 4 // 2) * 2)
            audio_features = torch.zeros(1, 80, T_mel)
            details["audio_mode"] = "silent_zero_mel"

    except Exception as e:
        details["error"] = f"İşleme hatası: {e}"
        details["total_ms"] = int((time.perf_counter() - t0) * 1000)
        return False, None, details
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        if tmp_audio:
            try:
                os.unlink(tmp_audio)
            except OSError:
                pass

    # Whisper mel: (1, 80, T_mel) — T_mel cift olmali (downsample conv stride=2)
    if audio_features.dim() == 2:
        audio_features = audio_features.unsqueeze(0)
    if audio_features.shape[-1] % 2 != 0:
        audio_features = audio_features[..., :-1]

    roi_tensor = _normalize_roi(roi).unsqueeze(0).to(DEVICE)
    audio_features = audio_features.to(DEVICE)
    input_lengths = torch.tensor([roi_tensor.shape[1]], dtype=torch.long, device=DEVICE)

    t_dec = time.perf_counter()
    with torch.no_grad():
        fused, v_mask, v_lens = model.encode(roi_tensor, audio_features, video_il=input_lengths)

        if base_mode == "attn":
            pred_indices = attn_greedy_decode(model, fused, v_mask)
            text = indices_to_text(pred_indices, num_phonemes=NUM_PHONEMES).strip() or "(boş)"
        else:
            ctc_log_probs = model.head.forward_ctc(fused)   # (T, B, V)
            T_out = ctc_log_probs.shape[0]
            il_clamp = int(v_lens.clamp(max=T_out)[0].item())
            lp = ctc_log_probs[:il_clamp, 0, :]             # (T_valid, V)

            if base_mode in ("beam", "beam_lm"):
                with_lm = (base_mode == "beam_lm")
                decoder, derr = _get_fusion_beam_decoder(with_lm=with_lm)
                if decoder is None:
                    details["decode_fallback"] = f"beam decoder alinamadi ({derr}), greedy fallback"
                    pred_indices = greedy_ctc_decode(lp)
                    text = indices_to_text(pred_indices, num_phonemes=NUM_PHONEMES).strip() or "(boş)"
                else:
                    lp_np = lp.float().cpu().numpy()
                    text = decoder.decode(lp_np, beam_width=100).strip() or "(boş)"
            else:  # greedy (default)
                pred_indices = greedy_ctc_decode(lp)
                text = indices_to_text(pred_indices, num_phonemes=NUM_PHONEMES).strip() or "(boş)"

    # Opsiyonel ekstra katman: lexicon post-correction ("_lex" suffix verildiyse)
    if apply_lex and text and text != "(boş)":
        text, n_fixed = lexicon_correct(text)
        details["lexicon_fixed"] = n_fixed

    details["decode_ms"] = int((time.perf_counter() - t_dec) * 1000)
    details["total_ms"] = int((time.perf_counter() - t0) * 1000)
    return True, text, details
