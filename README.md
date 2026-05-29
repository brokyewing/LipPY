# LipPY — Türkçe Dudak Okuma (Audio-Visual Speech Recognition)

Konuşan bir kişinin videosundan **Türkçe metin** üreten sistem. Dudak hareketlerini (görsel) ve — varsa — ses sinyalini (Whisper) birleştirerek konuşmayı yazıya çevirir. Türkçe için açık kaynak AV-VSR (Audio-Visual Visual Speech Recognition).

## İki Model

| Model (id) | Girdi | PER | Mimari |
|---|---|---|---|
| **video** | Sadece dudak (sessiz) | ~%48 | Conv3D + ResNet-SiLU + Conformer (6 layer, 512d) → CTC |
| **av-fusion** | Dudak + Ses | **%24.7** | video encoder + Whisper-medium → symmetric fusion (8-layer Transformer) + hybrid CTC/Attention |

> **Multimodal kazanım:** AV-Fusion, sadece-audio Whisper'ı (%31.2) geçer — ses sinyali, dudak okumanın belirsizliğini (b/p/m aynı görünür, sesli harfler ayırt edilemez) çözer. Projenin ana yeniliği budur.

## Üç Mod (Frontend)

- **Kısa Video** — video (+ opsiyonel ses) yükle → anlık tahmin. Video modelinde decode modu seçilebilir (greedy / beam+KenLM).
- **Uzun Video** — silero-VAD ile konuşma segmentlerine böl → segment segment işle → zaman damgalı altyazı (VTT / SRT / TXT indir).
- **Canlı Çekim** — webcam'den gerçek zamanlı altyazı (WebSocket). Sesli (av-fusion) veya sessiz (video) mod.

## Mimari

```
Video (.mp4 / webcam)
  → MediaPipe FaceMesh → affine align → 88×88 grayscale dudak patch (normalize)
  → Video Encoder (Conv3D + ResNet-SiLU + Conformer)
                                              ↘
(opsiyonel ses → Whisper-medium encoder)  →  Symmetric Fusion + Hybrid Head
                                              ↗
  → CTC log-probs → decode (greedy / beam / beam+KenLM)
  → Türkçe metin
```

ROI pipeline eğitim ile birebir aynıdır (**MediaPipe**; eğitim verisi MediaPipe ile çıkarıldı).

## Decode Modları

| Mod | Açıklama |
|---|---|
| `greedy` | CTC argmax — en hızlı, dil modeli yok |
| `beam` | pyctcdecode beam search (LM yok) |
| `beam_lm` | beam + **KenLM** (Türkçe domain dil modeli) — özellikle WER'i belirgin düşürür |

Video modelinde KenLM'in etkisi dramatiktir (ham çıktı anlamsız → Türkçe kelimeler); AV-Fusion'da ses zaten çözdüğü için `beam_lm` sabittir.

## Kurulum

### Backend (WSL / Linux önerilir)

> **Önemli:** `kenlm` (dil modeli) Windows'ta C++ derleme nedeniyle kurulamaz. Backend'i **WSL Ubuntu** veya Linux'ta çalıştırın.

```bash
# venv
python3 -m venv lippy-env && source lippy-env/bin/activate

pip install -r requirements.txt
pip install -r backend/requirements.txt

# KenLM (beam_lm için — Linux/WSL):
pip install https://github.com/kpu/kenlm/archive/master.zip
```

`backend/requirements.txt` şunları kurar: `transformers==4.47.1` (Whisper — sürüm **pinli**, 5.x API'yi bozar), `imageio-ffmpeg`, `soundfile`, `silero-vad`, `pyctcdecode`.

### Frontend (React 19 + Vite)

```bash
cd frontend && npm install
```

## Çalıştırma

**Backend** (WSL):
```bash
source lippy-env/bin/activate
# Whisper modelini tekrar indirmemek için (varsa) HF cache'i göster:
export HF_HOME=/mnt/c/Users/<kullanici>/.cache/huggingface
uvicorn backend.main:app --reload --port 8000
```

**Frontend** (ayrı terminal):
```bash
cd frontend && npm run dev   # http://localhost:5173
```

## API Endpoint'leri

| Method | Endpoint | Açıklama |
|---|---|---|
| GET | `/api/health` | Servis + GPU durumu |
| GET | `/api/models` | Kullanılabilir modeller (checkpoint bulunanlar) |
| POST | `/api/infer` | Kısa video → tahmin (`file`, opsiyonel `audio`, `model`, `decode_mode`) |
| POST | `/api/infer_long` | Uzun video → async job (`file`, `model`) |
| GET | `/api/jobs/:id/stream` | SSE — ilerleme + altyazı segmentleri |
| WS | `/ws/live?model=av-fusion` | Canlı webcam stream |

Örnek:
```bash
# Kısa video, sesli (av-fusion)
curl -X POST http://localhost:8000/api/infer \
  -F "file=@test.mp4" -F "audio=@test.wav" -F "model=av-fusion"

# Kısa video, sessiz (video, KenLM ile)
curl -X POST http://localhost:8000/api/infer \
  -F "file=@test.mp4" -F "model=video" -F "decode_mode=beam_lm"
```

## Proje Yapısı

```
LipPY/
├── backend/
│   ├── main.py              FastAPI (infer / infer_long / ws-live)
│   ├── inference.py         Video modeli (id: video) orkestrasyon + decode
│   ├── av_fusion_v1cc.py    AV-Fusion (id: av-fusion) inference
│   ├── long_video.py        Uzun video: VAD segmentasyon + altyazı
│   ├── models.py            Video model mimari (LipPhonemeModel)
│   ├── vocab.py             31 fonem (SPACE dahil)
│   └── roi/                 ROI çıkarma (MediaPipe, eğitim ile aynı)
├── frontend/                React 19 + Vite — 3 mod
├── checkpoints/             (git'e dahil değil — büyük dosyalar)
│   ├── video/best_model.pt       video modeli
│   ├── av-fusion/best_model.pt   av-fusion modeli
│   └── kenlm/                     lm.klm + unigrams.txt (ortak)
├── kaggle/                  Eğitim notebook'ları
│   ├── train-video-model.ipynb
│   ├── train-audio-encoder.ipynb
│   ├── train-av-fusion.ipynb
│   ├── precompute-mel-cache.ipynb
│   └── train-kenlm.ipynb
└── veri-olusturma/          Veri pipeline (indir → transkript → segment → ROI → temizle)
```

## Eğitim Pipeline'ı

**Veri hazırlama** (`veri-olusturma/`): `youtubevideo.py` (indir) → `adim1_transkript.py` (Whisper) → `adim2_segmentle.py` → `NEW/extract_rois.py` (MediaPipe ROI) → `veri_temizleyici.py`.

**Model eğitimi** (`kaggle/`, sırayla): mel cache → audio encoder → video model → av-fusion → KenLM.

## Notlar

- ROI eğitim ile birebir **MediaPipe** kullanır (FAN deneyseldir: `LIPPY_USE_FAN=1`).
- İlk inference cold-start: model + Whisper yüklenir (~10-30 s). Sonraki istekler hızlı (singleton cache).
- Canlı çekimde ~5 sn gecikme normaldir (Whisper inference + WebM decode + ROI).
- Checkpoint dosyaları git'e dahil değildir; Kaggle Output'tan indirilip `checkpoints/<model>/` altına konur.
