"""Adım 3 – USGS ComCat'ten deprem kataloğunu indirir.

Kapsam : dünya geneli, 1980-01-01 – 2026-06-30, M >= 4,5
Kaynak : USGS FDSN Event Web Service (açık, kayıt gerektirmez)
Çıktı  : veri/ham/yillar/katalog_YYYY.csv  (her yıl ayrı)
         veri/ham/katalog_1980_2026.csv    (hepsi birleşik)

USGS tek sorguda en fazla 20.000 kayıt verdiği için veri yıl yıl çekilir.
Daha önce indirilmiş ve sayısı doğru olan yıllar tekrar indirilmez.
Çalıştırma:  .venv\\Scripts\\python.exe kod\\sude_veri\\veri_cekme.py
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

KOK = Path(__file__).resolve().parents[2]
HAM = KOK / "veri" / "ham"
YIL_KLASORU = HAM / "yillar"
BIRLESIK = HAM / "katalog_1980_2026.csv"

URL = "https://earthquake.usgs.gov/fdsnws/event/1/"
MIN_BUYUKLUK = 4.5
ILK_YIL, SON_YIL = 1980, 2026
SON_TARIH = "2026-07-01"   # bitiş hariç tutulur → 30 Haziran 2026 dahil


def yil_araligi(yil):
    bas = f"{yil}-01-01"
    bit = SON_TARIH if yil == SON_YIL else f"{yil + 1}-01-01"
    return bas, bit


def istek(uc, parametre, deneme=5):
    """Bağlantı hatasında birkaç kez yeniden dener."""
    for i in range(deneme):
        try:
            r = requests.get(URL + uc, params=parametre, timeout=180)
            r.raise_for_status()
            return r.text
        except requests.RequestException as hata:
            bekle = 5 * (i + 1)
            print(f"   ! hata ({hata.__class__.__name__}), {bekle} sn sonra tekrar denenecek")
            time.sleep(bekle)
    raise RuntimeError(f"{uc} isteği {deneme} denemede başarısız: {parametre}")


def usgs_sayi(bas, bit):
    p = dict(starttime=bas, endtime=bit, minmagnitude=MIN_BUYUKLUK, format="text")
    return int(istek("count", p).strip())


def yil_indir(yil):
    bas, bit = yil_araligi(yil)
    dosya = YIL_KLASORU / f"katalog_{yil}.csv"
    beklenen = usgs_sayi(bas, bit)
    if beklenen >= 20000:
        raise RuntimeError(f"{yil}: {beklenen} kayıt, 20.000 sınırını aşıyor")

    if dosya.exists() and len(pd.read_csv(dosya)) == beklenen:
        print(f"{yil}: zaten var ({beklenen} kayıt)")
        return beklenen, beklenen

    p = dict(starttime=bas, endtime=bit, minmagnitude=MIN_BUYUKLUK,
             format="csv", orderby="time-asc")
    df = pd.read_csv(io.StringIO(istek("query", p)))
    df.to_csv(dosya, index=False)
    durum = "tamam" if len(df) == beklenen else "SAYI TUTMUYOR"
    print(f"{yil}: {len(df)} kayıt indirildi (USGS: {beklenen}) → {durum}")
    time.sleep(1)   # USGS sunucusunu yormamak için
    return len(df), beklenen


def main():
    YIL_KLASORU.mkdir(parents=True, exist_ok=True)
    sorunlu = []
    for yil in range(ILK_YIL, SON_YIL + 1):
        inen, beklenen = yil_indir(yil)
        if inen != beklenen:
            sorunlu.append(yil)

    parcalar = [pd.read_csv(YIL_KLASORU / f"katalog_{y}.csv") for y in range(ILK_YIL, SON_YIL + 1)]
    df = pd.concat(parcalar, ignore_index=True)
    df.to_csv(BIRLESIK, index=False)

    print("\n--- ÖZET ---")
    print("Toplam kayıt      :", len(df))
    print("Tekrar eden id    :", df["id"].duplicated().sum())
    print("Tarih aralığı     :", df["time"].min(), "→", df["time"].max())
    print("En küçük büyüklük :", df["mag"].min())
    print("M >= 6 olay sayısı:", (df["mag"] >= 6).sum())
    print("Olay türleri      :", df["type"].value_counts().to_dict())
    print("Sayısı tutmayan yıllar:", sorunlu or "yok")
    print("Kaydedildi:", BIRLESIK)


if __name__ == "__main__":
    main()
