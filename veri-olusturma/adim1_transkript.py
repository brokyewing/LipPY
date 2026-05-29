"""
Adim 1: Whisper ile transkript cikar
videolar/*.mp4 -> dataset/_whisper_*.txt

Kullanim:
    python adim1_transkript.py
    python adim1_transkript.py --model medium
"""

import os
import sys
import glob
import argparse
import subprocess
import gc

def extract_audio(video_path, audio_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", audio_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def transcribe_and_save(audio_path, out_txt, model):
    """Whisper ile transkript al ve ANINDA dosyaya yaz."""
    segments, info = model.transcribe(
        audio_path, language="tr", word_timestamps=True,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
    )
    duration = info.duration
    print(f"  Sure: {duration:.0f}s ({duration/60:.1f} dk)", flush=True)

    word_count = 0
    seg_count = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for segment in segments:
            seg_count += 1
            pct = min(100, int(segment.end / max(duration, 1) * 100))
            print(f"\r  Transkript: {pct}% ({segment.end:.0f}/{duration:.0f}s) [{seg_count} seg]", end="", flush=True)
            if segment.words:
                for w in segment.words:
                    f.write(f"{w.start:.2f}\t{w.end:.2f}\t{w.word.strip()}\n")
                    word_count += 1
            else:
                f.write(f"{segment.start:.2f}\t{segment.end:.2f}\t{segment.text.strip()}\n")
                word_count += 1
            if seg_count % 10 == 0:
                f.flush()

    print(f"\n  {word_count} kelime -> {out_txt}", flush=True)
    return word_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adim 1: Whisper transkript")
    parser.add_argument("--video-dir", default="videolar")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", default=None, help="Tek video isle")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.video:
        videos = [args.video]
    else:
        videos = sorted(glob.glob(os.path.join(args.video_dir, "*.mp4")))

    print(f"Model: {args.model} | Device: {args.device}")
    print(f"Video sayisi: {len(videos)}\n")

    # Model BIR KERE yukle, tum videolarda kullan
    from faster_whisper import WhisperModel
    print("Whisper model yukleniyor...", flush=True)
    whisper_model = WhisperModel(args.model, device=args.device, compute_type="int8_float16")
    print("Model hazir.\n", flush=True)

    for video_path in videos:
        name = os.path.splitext(os.path.basename(video_path))[0]
        out_txt = os.path.join(args.output_dir, f"_whisper_{name}.txt")

        if os.path.exists(out_txt) and os.path.getsize(out_txt) > 100:
            print(f"{name}: Cache var, ATLANIYOR")
            continue

        print(f"{name}: Isleniyor...", flush=True)
        audio_path = os.path.join(args.output_dir, f"_temp_{name}.wav")

        print("  Ses cikariliyor...", flush=True)
        if not extract_audio(video_path, audio_path):
            print("  HATA: Ses cikarilamadi!")
            continue

        print(f"  Whisper ({args.model})...", flush=True)
        word_count = transcribe_and_save(audio_path, out_txt, whisper_model)

        # Temp ses sil (RAM tasarrufu)
        if os.path.exists(audio_path):
            os.remove(audio_path)

        # RAM temizle (Whisper model kalir, sadece gecici veriler silinir)
        gc.collect()

        print(f"  Tamamlandi!\n", flush=True)

    print("Adim 1 tamamlandi!")
    print("Sonraki: python adim2_segmentle.py")
