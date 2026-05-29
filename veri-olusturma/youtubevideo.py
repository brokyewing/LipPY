"""
YouTube video indirici - Bot tespiti onlemli
720p ve ustu videolari indirir.

Kullanim:
    python youtubevideo.py                           # video_listesi.txt'den toplu
    python youtubevideo.py --url "https://..."       # tek video
    python youtubevideo.py --update                  # yt-dlp guncelle

Bot hatasi aliyorsan:
    1. yt-dlp guncelle: python youtubevideo.py --update
    2. Cookie kullan: python youtubevideo.py --cookies
    3. Yavas mod: python youtubevideo.py --slow
"""

import os
import subprocess
import sys
import time
import random
import argparse


def update_ytdlp():
    """yt-dlp'yi guncelle (bot tespiti icin kritik)."""
    print("yt-dlp guncelleniyor...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp"], check=False)
    r = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
    print(f"yt-dlp versiyon: {r.stdout.strip()}")


def export_cookies(browser="brave"):
    """Tarayicidan cookie cikar (bot tespiti icin)."""
    cookie_file = "cookies.txt"
    print(f"Cookie cikariliyor ({browser})...")
    r = subprocess.run(
        ["yt-dlp", "--cookies-from-browser", browser, "--cookies", cookie_file,
         "--skip-download", "https://www.youtube.com"],
        capture_output=True, text=True
    )
    if os.path.exists(cookie_file) and os.path.getsize(cookie_file) > 100:
        print(f"Cookie kaydedildi: {cookie_file}")
        return cookie_file
    else:
        print(f"Cookie cikarilamadi. Diger tarayicilar: chrome, firefox, edge")
        return None


def download_video(name, url, output_dir="videolar", cookie_file=None, slow=False):
    """Tek video indir."""
    output_template = os.path.join(output_dir, f"{name}.%(ext)s")
    output_path = os.path.join(output_dir, f"{name}.mp4")

    # Zaten varsa atla
    if os.path.exists(output_path):
        print(f"  {name}: Zaten var, ATLANIYOR")
        return True

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height>=720][ext=mp4]+bestaudio[ext=m4a]/best[height>=720][ext=mp4]/best[height>=720]",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--retries", "3",
        "--fragment-retries", "3",
        "--extractor-retries", "3",
        # Bot tespiti onlemleri
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "--referer", "https://www.youtube.com/",
        "--add-header", "Accept-Language: tr-TR,tr;q=0.9,en;q=0.8",
        "-o", output_template,
    ]

    # Cookie kullan (varsa)
    if cookie_file and os.path.exists(cookie_file):
        cmd.extend(["--cookies", cookie_file])

    # Yavas mod
    if slow:
        cmd.extend(["--sleep-interval", "2", "--max-sleep-interval", "5",
                     "--sleep-requests", "1"])

    cmd.append(url)

    try:
        subprocess.run(cmd, check=True)
        print(f"  {name}: OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  {name}: HATA - {e}")
        return False


def download_from_list(list_file, output_dir="videolar", cookie_file=None, slow=False):
    """video_listesi.txt'den toplu indir."""
    if not os.path.exists(list_file):
        print(f"Hata: {list_file} bulunamadi.")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(list_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    basarili = 0
    hatali = 0

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            continue

        name, url = parts[0], parts[1]
        print(f"\n[{i+1}/{len(lines)}] {name}: {url}")

        if download_video(name, url, output_dir, cookie_file, slow):
            basarili += 1
        else:
            hatali += 1

        # Videolar arasi bekleme (bot tespiti onlemi)
        if slow and i < len(lines) - 1:
            wait = random.randint(3, 8)
            print(f"  Bekleniyor ({wait}s)...")
            time.sleep(wait)

    print(f"\nToplam: {basarili} basarili, {hatali} hatali")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube Video Indirici")
    parser.add_argument("--url", type=str, help="Tek video URL")
    parser.add_argument("--name", type=str, default="video", help="Video adi (--url ile)")
    parser.add_argument("--list", type=str, default="video_listesi.txt", help="Video listesi dosyasi")
    parser.add_argument("--output", type=str, default="videolar", help="Cikti klasoru")
    parser.add_argument("--update", action="store_true", help="yt-dlp guncelle")
    parser.add_argument("--cookies", action="store_true", help="Tarayicidan cookie cikar ve kullan")
    parser.add_argument("--cookie-file", type=str, default=None, help="Cookie dosyasi yolu")
    parser.add_argument("--slow", action="store_true", help="Yavas mod (bot tespiti onlemi)")
    parser.add_argument("--browser", type=str, default="brave", help="Cookie tarayicisi (varsayilan: brave)")
    args = parser.parse_args()

    # yt-dlp guncelle
    if args.update:
        update_ytdlp()
        sys.exit(0)

    # Cookie
    cookie_file = args.cookie_file
    if args.cookies:
        cookie_file = export_cookies(args.browser)

    # Mevcut cookies.txt varsa otomatik kullan
    if not cookie_file and os.path.exists("cookies.txt"):
        cookie_file = "cookies.txt"
        print(f"Cookie bulundu: {cookie_file}")

    # Tek video
    if args.url:
        os.makedirs(args.output, exist_ok=True)
        download_video(args.name, args.url, args.output, cookie_file, args.slow)
    else:
        download_from_list(args.list, args.output, cookie_file, args.slow)
