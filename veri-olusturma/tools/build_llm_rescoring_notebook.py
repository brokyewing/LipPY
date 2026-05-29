"""
Generator: kaggle/lippy-llm-rescoring-api.ipynb

API tabanli LLM rescoring (stretch goal). Local fine-tuning yok (onceki 3 fail).
Beam top-50 hipotezleri LLM'e ver, en akici Turkce'yi sec.

API secenekleri:
  - GPT-4o-mini ($0.15/1M input, $0.6/1M output) - ucuz, hizli
  - Claude haiku ($0.25/1M input, $1.25/1M output) - kaliteli
  - Llama 3.1 70B via Groq (ucretsiz tier var)
"""
import json
import os

OUT_PATH = r"C:\Users\asker\Projelerim\LipPY\kaggle\lippy-llm-rescoring-api.ipynb"


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


cells = []

cells.append(md("""# LipPY LLM Rescoring (API) — Max Effort Stretch Goal

**Onceki 3 LLM fail:**
- Zero-shot Turkish-Llama: CER 78 (hallucinate)
- Synthetic LoRA: CER 50 (distribution mismatch)
- Real LoRA: CER 70 (overfit)

**Yeni yaklasim:** API tabanli, fine-tuning YOK. Beam search'un top-50 hipotezini
LLM'e gosterip "en akici Turkce'yi sec" diye sor.

## Niye bu kez calisma sansi yuksek?

| Onceki | Bu |
|---|---|
| Fine-tune (overfit riski) | No fine-tune (pretrained kalite korunur) |
| Tek hipotez gor (hallucinate) | Top-50 hipotez gor (secim) |
| Whisper output verme | Hem audio output hem video output |
| Generic prompt | Domain-aware prompt |

## Pipeline

```
Cross-attn fusion logits
  -> Beam search (KenLM 6-gram), beam_width=200
  -> Top-50 hipotez listesi
       ["bir gun gokyuzunde", "bir gun gokyuzu nde", "bir gun gok yuzunde", ...]
  -> LLM (API): "Bu listeden en dogru/akici Turkce cumleyi sec"
  -> Final tahmin
```

## Maliyet tahmini

10K val sample x ~500 token (50 hipotez ortalama) x $0.15/1M = **~$0.75** GPT-4o-mini ile.
Cok ucuz. Anthropic API icin ~$1.25.

## Inputs

- Cross-attn fusion best ckpt + logits cache
- KenLM 6-gram (lippy-kenlm-6gram)
- OPENAI_API_KEY veya ANTHROPIC_API_KEY (Kaggle Secrets)

## Beklenen sonuc

| Pipeline | WER |
|---|---|
| Cross-attn + 6-gram KenLM | ~%28 |
| **+ LLM API rescoring** | **~%22-25** ⭐ (-3-8 puan) |
"""))

cells.append(code("""!pip install -q anthropic openai

import os, glob, json, time, subprocess
from typing import List

# Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    _us = UserSecretsClient()
    try:
        os.environ['OPENAI_API_KEY'] = _us.get_secret('OPENAI_API_KEY')
        print('OpenAI API key yuklendi')
    except Exception:
        pass
    try:
        os.environ['ANTHROPIC_API_KEY'] = _us.get_secret('ANTHROPIC_API_KEY')
        print('Anthropic API key yuklendi')
    except Exception:
        pass
except Exception as e:
    print(f'Kaggle secrets erisilemedi: {e}')

USE_PROVIDER = 'anthropic'  # 'openai' or 'anthropic'

if USE_PROVIDER == 'openai':
    from openai import OpenAI
    client = OpenAI()
    MODEL = 'gpt-4o-mini'
elif USE_PROVIDER == 'anthropic':
    import anthropic
    client = anthropic.Anthropic()
    MODEL = 'claude-3-5-haiku-20241022'

print(f'LLM: {USE_PROVIDER} / {MODEL}')
"""))

cells.append(md("""## 1) Beam search ile top-50 hipotez uret

Cross-attn fusion notebook'undaki logits_cache + 6-gram KenLM ile beam=200 search.
Her sample icin top-50 hipotez doner.
"""))

cells.append(code("""# pyctcdecode beam search top-N
from pyctcdecode import build_ctcdecoder

KENLM_CANDS = glob.glob('/kaggle/input/**/lm_6gram.klm', recursive=True)
if not KENLM_CANDS:
    KENLM_CANDS = glob.glob('/kaggle/input/**/*.klm', recursive=True)
KENLM_PATH = KENLM_CANDS[0] if KENLM_CANDS else None
print(f'KenLM: {KENLM_PATH}')

PHONEME_LABELS = ['SIL','SPACE',
                   'a','e','i','o','u','\\u0131','\\u00f6','\\u00fc',
                   'b','p','m','f','v','d','t','n','l',
                   's','z','c','\\u00e7','\\u015f','j',
                   'k','g','\\u011f','r','h','y']

decoder = build_ctcdecoder(
    labels=PHONEME_LABELS,
    kenlm_model_path=KENLM_PATH,
    alpha=0.5,
    beta=0.5,
)


def get_top_n_beams(log_probs_np, n=50, beam_width=200):
    \"\"\"
    log_probs_np: (T, V) numpy
    Return: list of (text, score)
    \"\"\"
    beams = decoder.decode_beams(log_probs_np, beam_width=beam_width)
    # beams: list of (text, lm_state, last_label, last_lm_state, score)
    results = []
    for b in beams[:n]:
        # pyctcdecode beam tuple: (text, lm_state, last_label, last_state, logit_score)
        text = b[0] if isinstance(b, tuple) else b
        score = b[-1] if isinstance(b, tuple) and len(b) >= 5 else 0.0
        results.append((text, score))
    return results


# Phoneme indices'i text'e cevir
def phoneme_str_to_text(phoneme_str):
    \"\"\"pyctcdecode label-bazli cikti -> Turkish text.\"\"\"
    # Labels: 'a','e',...,'SPACE',...
    out = []
    for tok in phoneme_str.split():
        if tok == 'SPACE':
            out.append(' ')
        elif tok == 'SIL':
            continue
        else:
            out.append(tok)
    return ''.join(out)


# Logits cache yukle (cross-attn fusion notebook'tan)
LOGITS_CACHE = '/kaggle/input/lippy-av-cross-attn-checkpoint/logits_cache.pt'
# veya cross-attn fusion notebook'unu BURADA inference yapip cache'i RAM'de tut

print('Logits cache hazirlanmasi gerekiyor (cross-attn fusion ciktis\\'i)')
"""))

cells.append(md("""## 2) LLM rescoring prompt'u

**Kritik:** prompt LLM'in halüsinasyondan kaçınmasını sağlamalı. Listede olmayan kelime ekleyemez.
"""))

cells.append(code("""RESCORE_PROMPT = \"\"\"Asagida bir Turkce konusmanin dudak okuma + ses tanima sisteminden cikan top-50 hipotezi var.
Her hipotez biraz farkli. Senin gorevin: bu listeden EN AKICI ve EN DOGRU TURKCE cumleyi sec.

KURALLAR:
1. SADECE listeden bir secenegi sec, yeni kelime EKLEME
2. Dilbilgisi acisindan en saglam olani sec
3. Kelimeleri yeniden DUZENLEME, sirayi koruma
4. Cevabin SADECE secilen cumle olsun, aciklama yok

Hipotezler:
{hypotheses}

Secilen cumle:\"\"\"


def call_llm_rescorer(hypotheses: List[str]) -> str:
    \"\"\"Top-50 hipotezi LLM'e ver, en iyiyi al.\"\"\"
    numbered = '\\n'.join(f'{i+1}. {h}' for i, h in enumerate(hypotheses))
    prompt = RESCORE_PROMPT.format(hypotheses=numbered)

    try:
        if USE_PROVIDER == 'openai':
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        else:  # anthropic
            resp = client.messages.create(
                model=MODEL,
                max_tokens=200,
                temperature=0.0,
                messages=[{'role':'user', 'content': prompt}],
            )
            return resp.content[0].text.strip()
    except Exception as e:
        print(f'API error: {e}')
        return hypotheses[0]  # fallback: best beam


def validate_response(response, hypotheses):
    \"\"\"LLM cevap listedeki birine yakinsa onu sec, yoksa top-1 dondur.\"\"\"
    # Tam match
    for h in hypotheses:
        if h.strip() == response.strip():
            return h
    # Numara match (LLM bazen "5." ile basliyor)
    for i, h in enumerate(hypotheses):
        if response.strip().startswith(f'{i+1}.'):
            return h
    # En yakin match (edit distance)
    def ed(a, b):
        n,m=len(a),len(b); dp=list(range(m+1))
        for ai in a:
            r=[dp[0]+1]
            for j,bj in enumerate(b):
                r.append(min(r[-1]+1, dp[j+1]+1, dp[j]+(0 if ai==bj else 1)))
            dp=r
        return dp[-1]
    best = hypotheses[0]; best_d = float('inf')
    for h in hypotheses:
        d = ed(response, h)
        if d < best_d:
            best_d = d; best = h
    return best
"""))

cells.append(md("""## 3) Tum val'a rescoring + WER hesapla

Logits cache'inden beam top-50 al, LLM'e gonder, secimi al. Final WER hesapla.
"""))

cells.append(code("""# Pseudocode — cross-attn cache hazir oldugunda calistir
# import torch
# cache = torch.load('/kaggle/input/lippy-av-cross-attn-checkpoint/logits_cache.pt')

# results = {'baseline_top1': [], 'llm_rescored': [], 'refs': []}
# n_examples = len(cache)

# for idx, item in enumerate(cache):
#     log_probs = item['log_probs'].numpy()  # (T, V)
#     ref_text = item['text']
#
#     # Beam top-50
#     beams = get_top_n_beams(log_probs, n=50, beam_width=200)
#     hypotheses = [phoneme_str_to_text(b[0]) for b in beams]
#
#     # Baseline: top-1
#     top1 = hypotheses[0]
#     results['baseline_top1'].append(top1)
#     results['refs'].append(ref_text)
#
#     # LLM rescoring
#     llm_pred = call_llm_rescorer(hypotheses)
#     llm_final = validate_response(llm_pred, hypotheses)
#     results['llm_rescored'].append(llm_final)
#
#     if idx % 100 == 0:
#         print(f'[{idx}/{n_examples}] LLM: \"{llm_final[:60]}\"')


# # WER hesapla
# def wer(ref, hyp):
#     ref_w = ref.split(); hyp_w = hyp.split()
#     n,m=len(ref_w),len(hyp_w); dp=list(range(m+1))
#     for ai in ref_w:
#         r=[dp[0]+1]
#         for j,bj in enumerate(hyp_w):
#             r.append(min(r[-1]+1, dp[j+1]+1, dp[j]+(0 if ai==bj else 1)))
#         dp=r
#     return dp[-1] / max(n,1)
#
# baseline_wer = sum(wer(r,h) for r,h in zip(results['refs'], results['baseline_top1'])) / len(results['refs'])
# llm_wer = sum(wer(r,h) for r,h in zip(results['refs'], results['llm_rescored'])) / len(results['refs'])
#
# print(f'Baseline WER (top-1):   {baseline_wer:.1%}')
# print(f'LLM rescored WER:        {llm_wer:.1%}')
# print(f'Kazanc:                  {(baseline_wer-llm_wer)*100:+.1f} puan')

print('NOT: Bu notebook cross-attn fusion logits cache hazir oldugunda calistirilir.')
print('Once cross-attn fusion bitirilmeli (output: logits_cache.pt).')
"""))

cells.append(md("""## 4) Maliyet izleme + final tablo

Eger LLM rescoring -3+ puan WER kazandirsa: paper'a ekle.
Eger marjinal/negatif ise: terk et, paper'a "negative result" olarak yaz.
"""))

cells.append(code("""# Final tablo + JSON save
print('=' * 70)
print('LipPY MAX EFFORT FINAL PIPELINE')
print('=' * 70)
print(f'{\"Konfig\":<35} | {\"WER\":>8}')
print('-' * 50)
configs = [
    ('Video-only (v1c)',                  None),   # Cross-attn notebook'dan al
    ('Audio-only (best of 3)',            None),
    ('Late fusion alpha=0.5',             None),
    ('Cross-attn fusion',                 None),
    ('+ 4-gram KenLM (mevcut)',           None),
    ('+ 6-gram KenLM',                    None),
    ('+ LLM rescoring (BU)',              None),
]
for name, wer in configs:
    wer_str = f'{wer:.1%}' if wer is not None else '?'
    print(f'{name:<35} | {wer_str:>8}')
print('=' * 70)

print('\\nNOT: Bu hucredeki sayilar diger notebook ciktilariyla doldurulur.')
"""))

# DUMP
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.11.0"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"YAZILDI: {OUT_PATH}")
print(f"Cell sayisi: {len(cells)}")
