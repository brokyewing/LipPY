"""
Adim 3: Audio-only indirme + Whisper re-run + cache regenerate

Plan B (C planinin temeli) icin:
- video_listesi.txt'den 432+ video URL'lerini oku
- Her video icin SADECE ses indir (yt-dlp -x --audio-format m4a)
  -> audios_raw/<name>.m4a  (~5 GB toplam, video'dan ~10x kucuk)
- Indirilen m4a uzerinde Whisper medium calistir
  -> dataset/_whisper_<name>.txt  (word-level timestamp cache)

NOT: Bu script SADECE Whisper cache uretir.
Sonraki adimlar (yeni clip + yeni ROI + ses segment) icin:
  1. python adim2_segmentle.py --video-dir audios_raw --output-dir dataset_v2
     (NOT: adim2 video bekliyor; clip kesimi icin VIDEO download da gerekecek.
      Bu script ses uzerinden TRANSKRIPT cikartmak icin, alignment icin sadece
      timestamp degerleri lazim, ses dosyasinin kendisi clip kesmek icin
      kullanilamaz cunku ROI cikartma video gerektirir.)

Kullanim:
    python adim3_audio.py                                # video_listesi.txt'den tum
    python adim3_audio.py --test test_videosu181         # tek video test
    python adim3_audio.py --skip-download                # cache'i regenerate, ses var
    python adim3_audio.py --skip-whisper                 # sadece indir, transcribe etme
    python adim3_audio.py --slow                         # bot detection icin yavas mod
"""

import os
import sys
import time
import random
import argparse
import subprocess
import gc


# ============================================================
# yt-dlp ile sadece ses indirme
# ============================================================

def update_ytdlp():
    """yt-dlp'yi guncelle."""
    print("yt-dlp guncelleniyor...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], check=False)
    r = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    print(f"yt-dlp versiyon: {r.stdout.strip()}", flush=True)


def export_cookies(browser="brave"):
    """Tarayicidan cookie cikar (bot tespiti icin)."""
    cookie_file = "cookies.txt"
    print(f"Cookie cikariliyor ({browser})...", flush=True)
    subprocess.run(
        ["yt-dlp", "--cookies-from-browser", browser, "--cookies", cookie_file,
         "--skip-download", "https://www.youtube.com"],
        capture_output=True, text=True
    )
    if os.path.exists(cookie_file) and os.path.getsize(cookie_file) > 100:
        print(f"  Cookie kaydedildi: {cookie_file}", flush=True)
        return cookie_file
    print(f"  Cookie cikarilamadi. Diger tarayicilar: chrome, firefox, edge", flush=True)
    return None


def download_audio_only(name, url, audio_dir, cookie_file=None, slow=False):
    """
    Tek video icin SADECE ses indir (m4a).
    Output: audio_dir/<name>.m4a

    Return: True if downloaded or already exists, False on failure.
    """
    out_template = os.path.join(audio_dir, f"{name}.%(ext)s")
    out_m4a = os.path.join(audio_dir, f"{name}.m4a")

    if os.path.exists(out_m4a) and os.path.getsize(out_m4a) > 1000:
        return True

    cmd = [
        "yt-dlp",
        "-x",                                  # sadece ses extract
        "--audio-format", "m4a",               # m4a (AAC, kompakt)
        "--audio-quality", "0",                # en yuksek kalite
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "3",
        "--extractor-retries", "3",
        # Bot tespiti onlemleri
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "--referer", "https://www.youtube.com/",
        "--add-header", "Accept-Language: tr-TR,tr;q=0.9,en;q=0.8",
        "-o", out_template,
    ]

    if cookie_file and os.path.exists(cookie_file):
        cmd.extend(["--cookies", cookie_file])

    if slow:
        cmd.extend([
            "--sleep-interval", "2",
            "--max-sleep-interval", "5",
            "--sleep-requests", "1",
        ])

    cmd.append(url)

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and os.path.exists(out_m4a):
            size_mb = os.path.getsize(out_m4a) / (1024 * 1024)
            print(f"  Indirildi: {size_mb:.1f} MB", flush=True)
            return True
        # Bazi durumlarda m4a yerine webm vs cikabilir
        cand = [
            os.path.join(audio_dir, f"{name}.webm"),
            os.path.join(audio_dir, f"{name}.opus"),
            os.path.join(audio_dir, f"{name}.aac"),
        ]
        for c in cand:
            if os.path.exists(c):
                # m4a'ya cevir
                subprocess.run([
                    "ffmpeg", "-y", "-i", c, "-c:a", "aac", "-b:a", "128k",
                    out_m4a
                ], capture_output=True)
                if os.path.exists(out_m4a):
                    os.remove(c)
                    return True
        print(f"  HATA: yt-dlp rc={r.returncode}", flush=True)
        if r.stderr:
            print(f"    {r.stderr.strip()[-300:]}", flush=True)
        return False
    except subprocess.TimeoutExpired:
        print(f"  HATA: Timeout (10 dk asildi)", flush=True)
        return False
    except Exception as e:
        print(f"  HATA: {e}", flush=True)
        return False


# ============================================================
# Whisper transkripsiyon (adim1_transkript.py'den uyarli)
# ============================================================

def transcribe_audio(audio_path, out_txt, model):
    """
    m4a/wav/herhangi bir audio dosyasini Whisper ile transcribe et,
    word-level timestamp'leri tab-separated kaydet.

    Format (adim2_segmentle.py ile uyumlu):
        start\tend\ttext
    """
    segments, info = model.transcribe(
        audio_path,
        language="tr",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    duration = info.duration
    print(f"  Sure: {duration:.0f}s ({duration / 60:.1f} dk)", flush=True)

    word_count = 0
    seg_count = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for segment in segments:
            seg_count += 1
            pct = min(100, int(segment.end / max(duration, 1) * 100))
            print(
                f"\r  Transkript: {pct}% ({segment.end:.0f}/{duration:.0f}s) "
                f"[{seg_count} seg]",
                end="", flush=True,
            )
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


# ============================================================
# Ana akis
# ============================================================

def load_video_list(path):
    """video_listesi.txt'den (name, url) ciftlerini oku."""
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Whitespace (space veya tab) ile bol
            parts = line.split()
            if len(parts) < 2:
                continue
            name, url = parts[0], parts[1]
            items.append((name, url))
    return items


def main():
    parser = argparse.ArgumentParser(description="Adim 3: Audio download + Whisper re-run")
    parser.add_argument("--video-list", default="video_listesi.txt",
                        help="Video URL listesi (varsayilan: video_listesi.txt)")
    parser.add_argument("--audio-dir", default="audios_raw",
                        help="Indirilen m4a dosyalarinin klasoru")
    parser.add_argument("--whisper-dir", default="dataset",
                        help="Whisper cache'in yazilacagi klasor (_whisper_<name>.txt)")
    parser.add_argument("--whisper-model", default="medium",
                        help="Whisper model (tiny, base, small, medium, large-v2, large-v3)")
    parser.add_argument("--device", default="cuda",
                        help="cuda veya cpu")
    parser.add_argument("--test", type=str, default=None,
                        help="Sadece belirtilen video adi icin calistir (debug)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Indirme aslama, sadece Whisper calistir")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Whisper atla, sadece indir")
    parser.add_argument("--update", action="store_true",
                        help="yt-dlp'yi guncelle ve cik")
    parser.add_argument("--cookies", action="store_true",
                        help="Brave'den cookie cikar (bot tespiti onlemi)")
    parser.add_argument("--cookie-file", default=None,
                        help="Hazir cookie dosyasi yolu")
    parser.add_argument("--browser", default="brave",
                        help="Cookie tarayicisi (brave, chrome, firefox, edge)")
    parser.add_argument("--slow", action="store_true",
                        help="Yavas mod (bot tespiti onlemi)")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Listenin n'inci video'sundan basla (resume icin)")
    args = parser.parse_args()

    if args.update:
        update_ytdlp()
        return

    # Klasorleri olustur
    os.makedirs(args.audio_dir, exist_ok=True)
    os.makedirs(args.whisper_dir, exist_ok=True)

    # Cookie
    cookie_file = args.cookie_file
    if args.cookies and not cookie_file:
        cookie_file = export_cookies(args.browser)
    if not cookie_file and os.path.exists("cookies.txt"):
        cookie_file = "cookies.txt"
        print(f"Cookie bulundu: {cookie_file}", flush=True)

    # Video listesi
    items = load_video_list(args.video_list)
    if not items:
        print(f"Hata: {args.video_list} bos veya bulunamadi.", flush=True)
        sys.exit(1)

    if args.test:
        items = [(n, u) for n, u in items if n == args.test]
        if not items:
            print(f"Hata: --test {args.test} listede yok.", flush=True)
            sys.exit(1)
        print(f"Test modu: sadece {args.test}", flush=True)

    items = items[args.start_from - 1:]
    print(f"Toplam islem: {len(items)} video (start_from={args.start_from})", flush=True)
    print(f"Audio dir:   {args.audio_dir}", flush=True)
    print(f"Whisper dir: {args.whisper_dir}", flush=True)

    # Whisper model bir kez yukle (varsa)
    whisper_model = None
    if not args.skip_whisper:
        print(f"\nWhisper model yukleniyor: {args.whisper_model} (device={args.device})...",
              flush=True)
        try:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel(
                args.whisper_model,
                device=args.device,
                compute_type="int8_float16" if args.device == "cuda" else "int8",
            )
            print("Whisper hazir.\n", flush=True)
        except ImportError:
            print("HATA: faster-whisper kurulu degil. Kur:\n"
                  "  pip install faster-whisper", flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"HATA: Whisper yuklenemedi: {e}", flush=True)
            sys.exit(1)

    indirilen = 0
    transkript = 0
    atlanan = 0
    hatali = 0

    for i, (name, url) in enumerate(items, start=args.start_from):
        print(f"\n[{i}/{len(items) + args.start_from - 1}] {name}", flush=True)

        whisper_path = os.path.join(args.whisper_dir, f"_whisper_{name}.txt")
        audio_path = os.path.join(args.audio_dir, f"{name}.m4a")

        # Hizli atlama: whisper cache zaten varsa
        if os.path.exists(whisper_path) and os.path.getsize(whisper_path) > 100:
            print(f"  Whisper cache zaten var, ATLANIYOR", flush=True)
            atlanan += 1
            continue

        # 1. Ses indir
        if not args.skip_download:
            if not os.path.exists(audio_path):
                print(f"  Ses indiriliyor: {url}", flush=True)
                ok = download_audio_only(name, url, args.audio_dir,
                                          cookie_file=cookie_file, slow=args.slow)
                if not ok:
                    hatali += 1
                    continue
                indirilen += 1
            else:
                print(f"  Ses dosyasi mevcut: {audio_path}", flush=True)

        # 2. Whisper transkript
        if not args.skip_whisper:
            if not os.path.exists(audio_path):
                print(f"  HATA: Ses dosyasi yok, transkript atlandi: {audio_path}",
                      flush=True)
                hatali += 1
                continue

            print(f"  Whisper transkript...", flush=True)
            try:
                transcribe_audio(audio_path, whisper_path, whisper_model)
                transkript += 1
                gc.collect()
            except Exception as e:
                print(f"  HATA: Whisper - {e}", flush=True)
                # Yarim kalmis cache dosyasini sil (eksik)
                if os.path.exists(whisper_path) and os.path.getsize(whisper_path) < 100:
                    os.remove(whisper_path)
                hatali += 1
                continue

        # Yavas mod: videolar arasi bekleme
        if args.slow and i < len(items) + args.start_from - 1:
            wait = random.randint(3, 8)
            print(f"  Bekleniyor ({wait}s)...", flush=True)
            time.sleep(wait)

    # Ozet
    print(f"\n{'='*60}")
    print(f"ADIM 3 TAMAMLANDI")
    print(f"  Indirilen:  {indirilen}")
    print(f"  Transkript: {transkript}")
    print(f"  Atlanan:    {atlanan} (zaten cache var)")
    print(f"  Hatali:     {hatali}")
    print(f"\nWhisper cache: {args.whisper_dir}/_whisper_*.txt")
    print(f"Ham ses:       {args.audio_dir}/*.m4a")
    print(f"\nSONRAKI ADIM:")
    print(f"  Yeni clip + ROI uretmek icin VIDEO indirilmesi gerekiyor:")
    print(f"    python youtubevideo.py --list video_listesi.txt --slow")
    print(f"  Sonra:")
    print(f"    python adim2_segmentle.py")
    print(f"    python NEW/extract_rois.py")


if __name__ == "__main__":
    main()
