"""
Veri kalite kontrol scripti (Zeyrek tabanlı).
dataset/texts/*.txt dosyalarını tarar, her kelimeyi Türkçe morfolojik
analizden geçirir ve şüpheli örnekleri raporlar:
- Türkçe olmayan kelimeler (yabancı oran >= %30 ise şüpheli)
- Anlamsız sesler (hımm, hmm, hahaha vb.)
- Çok kısa / çok uzun metinler
- Tek kelimelik metinler
- Dolgu kelime oranı yüksek (>= %40)
- Yarım cümle (bağlaç/edatla biten)
- Ardışık kelime tekrarı (aynı kelime 2'den fazla art arda)

Kullanım:
  pip install zeyrek
  python veri_temizleyici.py              # Sadece rapor
  python veri_temizleyici.py --uygula     # Şüpheli + içerik tekrarı sil; splits tekilleştir
"""

import os
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache

import zeyrek

TEXT_DIR = os.path.join("dataset", "texts")
SPLIT_DIR = os.path.join("dataset", "splits")
CLIPS_DIR = os.path.join("dataset", "clips")
ROIS_DIR = os.path.join("dataset", "rois")

MIN_METIN_UZUNLUK = 3
MAX_METIN_UZUNLUK = 150
MIN_KELIME_SAYISI = 2
MAX_YABANCI_ORAN = 0.01  # %40'tan fazla yabanci kelime varsa sil (Zeyrek her kelimeyi tanimaz)
MAX_DOLGU_ORAN = 0.40
MAX_ARDISIK_TEKRAR = 2

# Zeyrek'in tanıyamayacağı ama Türkçe'de sorun olmayan kelimeler
BEYAZ_LISTE: Set[str] = {
    "merhaba", "selam", "evet", "hayır", "tamam", "peki", "yani",
    "şey", "hani", "acaba", "aslında", "mesela", "falan", "filan",
    "hmm", "aa", "ee",
}

# Anlamsız sesler — bunlar eğitimde gürültü yapar
ANLAMSIZ_SESLER: Set[str] = {
    "hımm", "hmm", "hahaha", "haha", "eee", "aaa", "ooo",
    "ııı", "mmm", "şşş", "uff", "tüh", "aah", "ooh",
}

DOLGU_KELIMELER: Set[str] = {
    "yani", "işte", "şey", "hani", "böyle", "şöyle",
    "falan", "filan", "mesela", "aslında", "acaba",
    "ee", "aa", "hım", "evet",
}

YARIM_CUMLE_SONU: Set[str] = {
    "ve", "ama", "fakat", "çünkü", "veya", "hem", "ya",
}

YABANCI_BIGRAMLAR: Set[str] = {
    "th", "ph", "ck", "gh", "dg",
    "oo", "ee",
    "tion", "sion",
}


def split_ids_oku(split: str) -> List[str]:
    dosya = os.path.join(SPLIT_DIR, f"{split}.txt")
    if not os.path.exists(dosya):
        return []
    with open(dosya, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def metin_oku(sample_id: str) -> str:
    dosya = os.path.join(TEXT_DIR, f"{sample_id}.txt")
    if not os.path.exists(dosya):
        return ""
    with open(dosya, "r", encoding="utf-8") as f:
        return f.read().strip()


def _metin_normalize(metin: str) -> str:
    """İçerik karşılaştırması için: boşlukları birleştir, strip."""
    return " ".join(metin.split()).strip()


def splits_tekillestir() -> Tuple[int, int]:
    """
    train.txt ve val.txt içinde aynı ID birden fazla geçiyorsa tekilleştirir (sıra korunur).
    Kaldırılan tekrar sayısını (train, val) döndürür.
    """
    kaldirilan_train, kaldirilan_val = 0, 0
    for split_name in ("train", "val"):
        dosya = os.path.join(SPLIT_DIR, f"{split_name}.txt")
        if not os.path.exists(dosya):
            continue
        with open(dosya, "r", encoding="utf-8") as f:
            satirlar = [line.strip() for line in f if line.strip()]
        eski_sayi = len(satirlar)
        # Sırayı koruyarak tekilleştir (ilk geçen kalsın)
        tekillestirilmis = list(dict.fromkeys(satirlar))
        yeni_sayi = len(tekillestirilmis)
        kaldirilan = eski_sayi - yeni_sayi
        if split_name == "train":
            kaldirilan_train = kaldirilan
        else:
            kaldirilan_val = kaldirilan
        if kaldirilan > 0:
            with open(dosya, "w", encoding="utf-8") as f:
                for s in tekillestirilmis:
                    f.write(s + "\n")
            print(f"  Splits: {dosya} -> {kaldirilan} tekrarlı satır kaldırıldı ({eski_sayi} -> {yeni_sayi})")
    return kaldirilan_train, kaldirilan_val


def icerik_tekrari_bul(tum_idler: List[str]) -> Tuple[Set[str], int]:
    """
    Aynı metin içeriğine sahip örnekleri bulur. Her içerik için bir ID bırakır, diğerlerini silinecek döndürür.
    (silinecek_id_seti, tekrarlı_icerik_sayisi) döner.
    """
    icerikten_idler: Dict[str, List[str]] = defaultdict(list)
    for sid in tum_idler:
        metin = metin_oku(sid)
        if not metin:
            continue
        anahtar = _metin_normalize(metin)
        icerikten_idler[anahtar].append(sid)

    silinecek: Set[str] = set()
    tekrarlı_grup = 0
    for _icerik, id_list in icerikten_idler.items():
        if len(id_list) > 1:
            tekrarlı_grup += 1
            # İlkini bırak, diğerlerini sil
            silinecek.update(id_list[1:])
    return silinecek, tekrarlı_grup

_analyzer: Optional[zeyrek.MorphAnalyzer] = None

def _get_analyzer() -> zeyrek.MorphAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = zeyrek.MorphAnalyzer()
    return _analyzer


@lru_cache(maxsize=50000)
def _zeyrek_analiz_cache(kelime: str) -> bool:
    try:
        sonuclar = _get_analyzer().analyze(kelime)
        return len(sonuclar) > 0
    except Exception:
        return False  # Zeyrek çökerse/okuyamazsa Türkçe DEĞİLDİR olarak fişle.

def kelime_turkce_mi(analyzer: zeyrek.MorphAnalyzer, kelime: str) -> bool:
    """Zeyrek ile kelimenin Türkçe olup olmadığını kontrol eder."""
    if kelime in BEYAZ_LISTE:
        return True
    if len(kelime) == 1:
        # Tek harfli "o" (O geldi) anlamlıdır. Ama "e", "a" gibi harfler kesilmiş cümle gürültüsüdür.
        if kelime == "o":
            return True
        return False
    return _zeyrek_analiz_cache(kelime)

def yabanci_kelimeleri_bul(
    analyzer: zeyrek.MorphAnalyzer, metin: str
) -> List[str]:
    """Metindeki Türkçe olmayan kelimeleri döndürür."""
    kelimeler = metin.split()
    yabanci = []
    for kelime in kelimeler:
        if not kelime_turkce_mi(analyzer, kelime):
            yabanci.append(kelime)
    return yabanci


def anlamsiz_ses_mi(metin: str) -> bool:
    """Metnin anlamsız bir ses/ünlem olup olmadığını kontrol eder."""
    kelimeler = metin.split()
    if len(kelimeler) != 1:
        return False
    return kelimeler[0] in ANLAMSIZ_SESLER


def analiz_et(
    analyzer: zeyrek.MorphAnalyzer,
    sample_ids: List[str],
    split_adi: str,
) -> Tuple[Dict[str, List], Set[str]]:
    """Analiz yapar, rapor yazar; şüpheli sample ID'lerini set olarak döndürür."""
    sonuclar: Dict[str, List] = {
        "yabanci_kelime": [],
        "anlamsiz_ses": [],
        "cok_kisa": [],
        "cok_uzun": [],
        "tek_kelime": [],
        "dolgu_fazla": [],
        "yarim_cumle": [],
        "ardisik_tekrar": [],
        "yabanci_bigram": [],
    }
    supheli_ids: Set[str] = set()

    toplam = 0
    temiz = 0

    for i, sid in enumerate(sample_ids):
        metin = metin_oku(sid)
        if not metin:
            continue

        toplam += 1
        sorunlu = False

        if (i + 1) % 500 == 0:
            print(f"  [{split_adi}] {i + 1}/{len(sample_ids)} analiz edildi...")

        # Anlamsız ses kontrolü
        if anlamsiz_ses_mi(metin):
            sonuclar["anlamsiz_ses"].append((sid, metin))
            sorunlu = True

        # Zeyrek ile yabancı kelime kontrolü (her zaman raporla, oran eşiği üstünde ise şüpheli)
        kelimeler = metin.split()
        yabanci = yabanci_kelimeleri_bul(analyzer, metin)
        if yabanci:
            oran = len(yabanci) / len(kelimeler) if kelimeler else 0
            sonuclar["yabanci_kelime"].append(
                (sid, metin, yabanci, f"{oran:.0%}")
            )
            if oran >= MAX_YABANCI_ORAN:
                sorunlu = True

        # Çok kısa metin
        if len(metin) < MIN_METIN_UZUNLUK:
            sonuclar["cok_kisa"].append((sid, metin, f"{len(metin)} karakter"))
            sorunlu = True

        # Çok uzun metin
        if len(metin) > MAX_METIN_UZUNLUK:
            sonuclar["cok_uzun"].append(
                (sid, metin[:80] + "...", f"{len(metin)} karakter")
            )
            sorunlu = True

        # Tek kelimelik (anlamsız ses olmayanlar)
        if len(kelimeler) < MIN_KELIME_SAYISI and not anlamsiz_ses_mi(metin):
            sonuclar["tek_kelime"].append((sid, metin, f"{len(kelimeler)} kelime"))
            sorunlu = True

        # Dolgu kelime oranı
        if len(kelimeler) >= 2:
            dolgu_sayi = sum(1 for k in kelimeler if k in DOLGU_KELIMELER)
            dolgu_oran = dolgu_sayi / len(kelimeler)
            if dolgu_oran >= MAX_DOLGU_ORAN:
                sonuclar["dolgu_fazla"].append(
                    (sid, metin, f"{dolgu_oran:.0%}")
                )
                sorunlu = True

        # Yarım cümle (bağlaç/edatla biten)
        if kelimeler and kelimeler[-1] in YARIM_CUMLE_SONU:
            sonuclar["yarim_cumle"].append((sid, metin, kelimeler[-1]))
            sorunlu = True

        # Ardışık kelime tekrarı
        ardisik = 1
        for j in range(1, len(kelimeler)):
            if kelimeler[j] == kelimeler[j - 1]:
                ardisik += 1
                if ardisik > MAX_ARDISIK_TEKRAR:
                    sonuclar["ardisik_tekrar"].append(
                        (sid, metin, f"'{kelimeler[j]}' {ardisik}x")
                    )
                    sorunlu = True
                    break
            else:
                ardisik = 1

        # Yabancı bigram kontrolü (Türkçe'de bulunmayan harf kombinasyonları)
        bigram_bulunan = []
        for kelime in kelimeler:
            for bg in YABANCI_BIGRAMLAR:
                if bg in kelime:
                    bigram_bulunan.append((kelime, bg))
                    break
        if bigram_bulunan:
            sonuclar["yabanci_bigram"].append((sid, metin, bigram_bulunan))
            sorunlu = True

        if sorunlu:
            supheli_ids.add(sid)
        else:
            temiz += 1

    # Rapor
    print(f"\n{'='*60}")
    print(f"  {split_adi.upper()} ANALİZİ (Zeyrek)")
    print(f"{'='*60}")
    print(f"Toplam örnek  : {toplam}")
    print(f"Temiz örnek   : {temiz}")
    print(f"Şüpheli örnek : {toplam - temiz}")

    if sonuclar["yabanci_kelime"]:
        print(f"\n--- Yabancı kelime içerenler ({len(sonuclar['yabanci_kelime'])} adet) ---")
        for sid, metin, yabanci, oran in sonuclar["yabanci_kelime"][:30]:
            print(f"  {sid}  (yabancı oran: {oran})")
            print(f"    Metin    : '{metin}'")
            print(f"    Yabancı  : {yabanci}")
        if len(sonuclar["yabanci_kelime"]) > 30:
            print(f"  ... ve {len(sonuclar['yabanci_kelime']) - 30} tane daha")

    if sonuclar["anlamsiz_ses"]:
        print(f"\n--- Anlamsız sesler ({len(sonuclar['anlamsiz_ses'])} adet) ---")
        for sid, metin in sonuclar["anlamsiz_ses"]:
            print(f"  {sid} -> '{metin}'")

    if sonuclar["cok_kisa"]:
        print(f"\n--- Çok kısa metinler ({len(sonuclar['cok_kisa'])} adet) ---")
        for sid, metin, detay in sonuclar["cok_kisa"]:
            print(f"  {sid} -> '{metin}' ({detay})")

    if sonuclar["cok_uzun"]:
        print(f"\n--- Çok uzun metinler ({len(sonuclar['cok_uzun'])} adet) ---")
        for sid, metin, detay in sonuclar["cok_uzun"]:
            print(f"  {sid} -> '{metin}' ({detay})")

    if sonuclar["tek_kelime"]:
        print(f"\n--- Tek kelimelik metinler ({len(sonuclar['tek_kelime'])} adet) ---")
        for sid, metin, detay in sonuclar["tek_kelime"][:30]:
            print(f"  {sid} -> '{metin}' ({detay})")
        if len(sonuclar["tek_kelime"]) > 30:
            print(f"  ... ve {len(sonuclar['tek_kelime']) - 30} tane daha")

    if sonuclar["dolgu_fazla"]:
        print(f"\n--- Dolgu kelime oranı yüksek ({len(sonuclar['dolgu_fazla'])} adet, eşik: {MAX_DOLGU_ORAN:.0%}) ---")
        for sid, metin, oran in sonuclar["dolgu_fazla"][:30]:
            print(f"  {sid}  (dolgu oran: {oran})")
            print(f"    Metin: '{metin}'")
        if len(sonuclar["dolgu_fazla"]) > 30:
            print(f"  ... ve {len(sonuclar['dolgu_fazla']) - 30} tane daha")

    if sonuclar["yarim_cumle"]:
        print(f"\n--- Yarım cümle / bağlaçla biten ({len(sonuclar['yarim_cumle'])} adet) ---")
        for sid, metin, son_kelime in sonuclar["yarim_cumle"][:30]:
            print(f"  {sid}  (son kelime: '{son_kelime}')")
            print(f"    Metin: '{metin}'")
        if len(sonuclar["yarim_cumle"]) > 30:
            print(f"  ... ve {len(sonuclar['yarim_cumle']) - 30} tane daha")

    if sonuclar["ardisik_tekrar"]:
        print(f"\n--- Ardışık kelime tekrarı ({len(sonuclar['ardisik_tekrar'])} adet, eşik: >{MAX_ARDISIK_TEKRAR}) ---")
        for sid, metin, detay in sonuclar["ardisik_tekrar"][:30]:
            print(f"  {sid}  ({detay})")
            print(f"    Metin: '{metin}'")
        if len(sonuclar["ardisik_tekrar"]) > 30:
            print(f"  ... ve {len(sonuclar['ardisik_tekrar']) - 30} tane daha")

    if sonuclar["yabanci_bigram"]:
        print(f"\n--- Yabancı bigram içerenler ({len(sonuclar['yabanci_bigram'])} adet) ---")
        for sid, metin, eslesme in sonuclar["yabanci_bigram"][:30]:
            kelime_detay = ", ".join(f"'{k}' ({bg})" for k, bg in eslesme)
            print(f"  {sid}")
            print(f"    Metin   : '{metin}'")
            print(f"    Eşleşme : {kelime_detay}")
        if len(sonuclar["yabanci_bigram"]) > 30:
            print(f"  ... ve {len(sonuclar['yabanci_bigram']) - 30} tane daha")

    return sonuclar, supheli_ids


def supheli_ornekleri_sil(silinecek_ids: Set[str]) -> None:
    """Şüpheli ID'lerin clips, texts, rois dosyalarını siler; train/val'dan çıkarır."""
    for sid in silinecek_ids:
        for dir_path, ext in [(CLIPS_DIR, ".mp4"), (TEXT_DIR, ".txt"), (ROIS_DIR, ".npy")]:
            path = os.path.join(dir_path, f"{sid}{ext}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"  Silindi: {path}")
                except Exception as e:
                    print(f"  Hata: {path} -> {e}")

    for split_name in ("train", "val"):
        dosya = os.path.join(SPLIT_DIR, f"{split_name}.txt")
        if not os.path.exists(dosya):
            continue
        with open(dosya, "r", encoding="utf-8") as f:
            satirlar = [line.strip() for line in f if line.strip()]
        yeni_satirlar = [s for s in satirlar if s not in silinecek_ids]
        with open(dosya, "w", encoding="utf-8") as f:
            for s in yeni_satirlar:
                f.write(s + "\n")
        print(f"  Güncellendi: {dosya} ({len(yeni_satirlar)} satır kaldı)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Veri kalite kontrolü (Zeyrek). Şüpheli örnekleri raporla veya sil.")
    parser.add_argument(
        "--uygula",
        action="store_true",
        help="Şüpheli örnekleri sil: clips, texts ve splits (train/val) güncelle",
    )
    args = parser.parse_args()

    # --uygula ise önce splits içinde aynı ID birden fazla varsa tekilleştir
    if args.uygula:
        print("Splits tekilleştiriliyor (aynı ID birden fazla satırda ise 1'e düşürülüyor)...")
        splits_tekillestir()
        print()

    print("Zeyrek morfolojik analiz yükleniyor...")
    analyzer = zeyrek.MorphAnalyzer()
    print("Veri kalite kontrolü başlatılıyor...\n")

    train_ids = split_ids_oku("train")
    val_ids = split_ids_oku("val")

    tum_supheli: Set[str] = set()

    if train_ids:
        _, supheli_train = analiz_et(analyzer, train_ids, "train")
        tum_supheli |= supheli_train
    else:
        print("train.txt bulunamadı veya boş.")

    if val_ids:
        _, supheli_val = analiz_et(analyzer, val_ids, "val")
        tum_supheli |= supheli_val
    else:
        print("val.txt bulunamadı veya boş.")

    # Aynı içeriğe sahip kliplerin silinmesi Görsel Dudak Okuma (Lip-Reading) için 
    # modelin görsel zenginliğini (variance) düşüreceğinden İPTAL EDİLMİŞTİR!
    # (Aynı cümlenin farklı yüzlerle/ışıklarla söylenmesi SOTA modeller için çok değerli bir veridir).

    print(f"\n{'='*60}")
    print("Analiz tamamlandı.")
    print(f"Toplam şüpheli örnek (silinecek): {len(tum_supheli)}")

    if args.uygula and tum_supheli:
        print(f"\nŞüpheli {len(tum_supheli)} örnek siliniyor (clips, texts, rois, splits)...")
        supheli_ornekleri_sil(tum_supheli)
        print("Temizlik tamamlandı.")
    elif args.uygula and not tum_supheli:
        print("Silinecek şüpheli örnek yok.")
    else:
        print("Silmek için: python veri_temizleyici.py --uygula")
