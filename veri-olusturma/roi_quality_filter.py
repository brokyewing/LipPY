"""
roi_quality_filter.py — LipPY ROI dataset kalite filtresi.

Parlaklık/kontrast eşiklerine göre düşük kaliteli ROI videolarını tespit eder.
DOSYA SİLMEZ — `bad_rois/` klasörüne TAŞIR (hem video hem text dosyası).
Yanlış atılırsa geri taşıyabilirsin.

Kullanım:
    # 1) Önce dry-run (sadece rapor, dosya taşıma)
    python roi_quality_filter.py --dry-run

    # 2) Raporu incele, sonra gerçek taşımayı yap
    python roi_quality_filter.py

    # Custom path
    python roi_quality_filter.py --rois-dir "G:/LipPY/dataset/rois_mp4"

    # Test: sadece ilk 500 dosyayı kontrol et
    python roi_quality_filter.py --dry-run --limit 500

Eşikler (script üstünde değiştirilebilir):
    MIN_MEAN         = 63.75  (0.25*255) → bu değerden karanlık: ATILIR
    MIN_STD          = 20.4   (0.08*255) → bu değerden az kontrast: ATILIR
    MIN_MAX          = 100    → video boyunca max piksel <100: ATILIR
    MIN_BRIGHT_RATIO = 0.01   → >=150 piksel oranı %1'den az: ATILIR
"""
import argparse
import cv2
import glob
import json
import numpy as np
import os
import shutil
import sys
import time
from collections import defaultdict

# ─── Eşik değerleri (gerekirse değiştir) ───
MIN_MEAN = 63.75          # 0.25 * 255
MIN_STD = 20.4            # 0.08 * 255
MIN_MAX = 100
MIN_BRIGHT_RATIO = 0.01
MIN_FRAMES = 5
# ────────────────────────────────────────


def analyze_video(path):
    """Tek videoyu oku, grayscale frame istatistiklerini döndür."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    grays = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if len(f.shape) == 3:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        else:
            g = f
        grays.append(g)
    cap.release()

    if not grays:
        return None

    arr = np.stack(grays).astype(np.float32)
    return {
        'n_frames': len(grays),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'max': float(arr.max()),
        'min': float(arr.min()),
        'bright_ratio': float((arr >= 150).mean()),
        'dark_ratio': float((arr < 20).mean()),
    }


def classify(stats):
    """Videonun iyi mi kötü mü olduğuna karar ver."""
    if stats is None:
        return False, 'unreadable'
    if stats['n_frames'] < MIN_FRAMES:
        return False, 'too_short'
    if stats['mean'] < MIN_MEAN:
        return False, 'dark'
    if stats['std'] < MIN_STD:
        return False, 'lowcont'
    if stats['max'] < MIN_MAX:
        return False, 'nohigh'
    if stats['bright_ratio'] < MIN_BRIGHT_RATIO:
        return False, 'nobright'
    return True, 'ok'


def main():
    parser = argparse.ArgumentParser(
        description='LipPY ROI kalite filtresi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--rois-dir', default=r'G:\LipPY\dataset\rois_mp4',
                        help='ROI videolarinin oldugu klasor')
    parser.add_argument('--texts-dir', default=None,
                        help='Text dosyalari (default: parent_of_rois/texts)')
    parser.add_argument('--bad-dir', default=None,
                        help='Kotu videolarin tasinacagi klasor')
    parser.add_argument('--bad-texts-dir', default=None,
                        help='Kotu text dosyalarinin tasinacagi klasor')
    parser.add_argument('--dry-run', action='store_true',
                        help='Sadece tara, dosya tasima')
    parser.add_argument('--delete', action='store_true',
                        help='Tasima yerine kalici SIL (GERI ALMA YOK!)')
    parser.add_argument('--report', default=None,
                        help='JSON rapor cikti dosyasi (default: parent/quality_report.json)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Sadece ilk N dosyayi kontrol et (test icin)')
    parser.add_argument('--show-bad', type=int, default=30,
                        help='Konsolda en kotu N orneği goster (default 30)')
    args = parser.parse_args()

    rois_dir = os.path.abspath(args.rois_dir)
    parent = os.path.dirname(rois_dir)
    texts_dir = args.texts_dir or os.path.join(parent, 'texts')
    bad_dir = args.bad_dir or os.path.join(parent, 'bad_rois')
    bad_texts_dir = args.bad_texts_dir or os.path.join(parent, 'bad_texts')
    report_path = args.report or os.path.join(parent, 'quality_report.json')

    if not os.path.isdir(rois_dir):
        sys.exit(f'HATA: rois_dir bulunamadi: {rois_dir}')
    if not os.path.isdir(texts_dir):
        print(f'UYARI: texts_dir yok: {texts_dir} (text tasima yapilamayacak)')

    if not args.dry_run and not args.delete:
        os.makedirs(bad_dir, exist_ok=True)
        os.makedirs(bad_texts_dir, exist_ok=True)

    if args.delete and not args.dry_run:
        print('\n!!!!! UYARI: --delete modu aktif !!!!!')
        print('Kotu dosyalar BIRER BIRER KALICI OLARAK SILINECEK.')
        print('Bu islem GERI ALINAMAZ. Bad_rois/ klasoru olusturulmayacak.')
        print('Devam etmek icin 5 saniye icinde Ctrl+C ile iptal edebilirsin.\n')
        time.sleep(5)

    videos = sorted(glob.glob(os.path.join(rois_dir, '*.mp4')))
    if args.limit:
        videos = videos[:args.limit]
        print(f'[LIMIT] Sadece ilk {args.limit} dosya kontrol edilecek')

    total = len(videos)
    if total == 0:
        sys.exit(f'HATA: {rois_dir} icinde .mp4 bulunamadi')

    print(f'Rois dir: {rois_dir}')
    print(f'Texts dir: {texts_dir}')
    print(f'Bad dir: {bad_dir}')
    print(f'Dry run: {args.dry_run}')
    print(f'Toplam video: {total}')
    print(f'Esikler: MEAN>={MIN_MEAN} STD>={MIN_STD} MAX>={MIN_MAX} BRIGHT>={MIN_BRIGHT_RATIO}')
    print(f'{"=" * 60}')
    print('Basliyor...\n')

    reasons = defaultdict(int)
    bad_items = []
    t0 = time.time()

    for i, vpath in enumerate(videos, 1):
        vid = os.path.basename(vpath).replace('.mp4', '')
        stats = analyze_video(vpath)
        ok, reason = classify(stats)

        if ok:
            reasons['ok'] += 1
        else:
            reasons[reason] += 1
            bad_items.append({'id': vid, 'reason': reason, 'stats': stats})

            # Tasi veya sil
            if not args.dry_run:
                try:
                    tpath = os.path.join(texts_dir, f'{vid}.txt')
                    if args.delete:
                        os.remove(vpath)
                        if os.path.exists(tpath):
                            os.remove(tpath)
                    else:
                        dst_mp4 = os.path.join(bad_dir, os.path.basename(vpath))
                        shutil.move(vpath, dst_mp4)
                        if os.path.exists(tpath):
                            dst_txt = os.path.join(bad_texts_dir, f'{vid}.txt')
                            shutil.move(tpath, dst_txt)
                except Exception as e:
                    action = 'SILME' if args.delete else 'TASIMA'
                    print(f'  {action} HATASI {vid}: {e}')

        # Progress
        if i % 500 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.01)
            eta = (total - i) / max(rate, 0.01)
            bad_count = i - reasons['ok']
            pct = bad_count / i * 100
            print(f'  {i}/{total} ({rate:.1f}/s, ETA {int(eta)}s) — kotu: {bad_count} (%{pct:.1f})')

    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'TAMAMLANDI ({int(elapsed)}s)')
    print(f'{"=" * 60}')
    print(f'Toplam:         {total}')
    print(f'  OK:           {reasons["ok"]:>6}  (%{reasons["ok"]/total*100:.1f})')
    print(f'  Dark:         {reasons["dark"]:>6}  (%{reasons["dark"]/total*100:.1f})')
    print(f'  Low contrast: {reasons["lowcont"]:>6}  (%{reasons["lowcont"]/total*100:.1f})')
    print(f'  No highlight: {reasons["nohigh"]:>6}  (%{reasons["nohigh"]/total*100:.1f})')
    print(f'  No bright:    {reasons["nobright"]:>6}  (%{reasons["nobright"]/total*100:.1f})')
    print(f'  Too short:    {reasons["too_short"]:>6}  (%{reasons["too_short"]/total*100:.1f})')
    print(f'  Unreadable:   {reasons["unreadable"]:>6}  (%{reasons["unreadable"]/total*100:.1f})')
    total_bad = total - reasons['ok']
    print(f'  ─────────────')
    print(f'  Toplam kotu:  {total_bad:>6}  (%{total_bad/total*100:.1f})')

    # En kotu ornekler (mean'e gore sirala)
    if bad_items and args.show_bad > 0:
        print(f'\n--- En karanlik {args.show_bad} ornek ---')
        dark_sorted = sorted(
            [b for b in bad_items if b['stats'] and 'mean' in b['stats']],
            key=lambda b: b['stats']['mean']
        )
        for b in dark_sorted[:args.show_bad]:
            s = b['stats']
            print(f'  {b["id"]:<40} {b["reason"]:<10} mean={s["mean"]:5.1f} std={s["std"]:5.1f} max={s["max"]:4.0f}')

    # Kaynak video gruplama — ayni kaynak videodan kac tane kotu?
    source_bad = defaultdict(int)
    for b in bad_items:
        src = b['id'].rsplit('_', 1)[0]  # test_videosu322_0424 → test_videosu322
        source_bad[src] += 1
    if source_bad:
        top_sources = sorted(source_bad.items(), key=lambda x: -x[1])[:15]
        print(f'\n--- En cok kotu segment ureten kaynak videolar ---')
        for src, cnt in top_sources:
            print(f'  {src}: {cnt} kotu segment')

    # Rapor JSON kaydet
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'thresholds': {
                'min_mean': MIN_MEAN, 'min_std': MIN_STD,
                'min_max': MIN_MAX, 'min_bright_ratio': MIN_BRIGHT_RATIO,
                'min_frames': MIN_FRAMES,
            },
            'summary': {
                **dict(reasons),
                'total': total,
                'total_bad': total_bad,
                'bad_pct': round(total_bad / total * 100, 2),
            },
            'source_bad_counts': dict(sorted(source_bad.items(), key=lambda x: -x[1])),
            'bad_items': bad_items[:2000],
            'dry_run': args.dry_run,
            'elapsed_sec': round(elapsed, 1),
        }, f, ensure_ascii=False, indent=2)
    print(f'\nRapor kaydedildi: {report_path}')

    if args.dry_run:
        print('\n[DRY RUN] Hicbir dosya tasinmadi/silinmedi.')
        print('Gercek tasima icin: python roi_quality_filter.py')
        print('Kalici silme icin:  python roi_quality_filter.py --delete')
    elif args.delete:
        print(f'\n{total_bad} dosya kalici olarak SILINDI (video + text).')
        print('Geri alma yok. Yeni dataset boyutu: {} video'.format(total - total_bad))
    else:
        print(f'\nKotu dosyalar tasindi: {bad_dir}')
        print(f'Kotu text dosyalari:   {bad_texts_dir}')
        print('Yanlis atildigini dusundugun varsa manuel geri tasiyabilirsin.')


if __name__ == '__main__':
    main()
