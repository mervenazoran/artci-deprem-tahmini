"""Adım 5 – Dış faktörler: ay evresi + hava durumu (sıcaklık, basınç, yağış).

Girdi : veri/islenmis/artcilar.csv  (Adım 4)
Çıktı : veri/islenmis/artcilar_dis_faktorlu.csv
Önbellek: veri/ham/hava/<dizi_id>.csv  (her dizi için Open-Meteo yanıtı; tekrar indirilmez)

- ay_evresi: 0 = yeni ay, 0,5 = dolunay. Tarihten hesaplanır (sinodik ay = 29,530588853 gün,
  referans yeni ay: 6 Ocak 2000 18:14 UTC). İnternet gerekmez.
- sicaklik (°C, 2 m), basinc (hPa, deniz seviyesine indirgenmiş basınç; yükseklik etkisi
  olmasın diye yüzey basıncı yerine bu seçildi), yagis (mm, o saatteki toplam):
  Open-Meteo Historical Weather API (ERA5 yeniden analizi, açık, kayıt gerektirmez).
  Her dizi için ANA DEPREM konumunda, ana deprem gününden son artçı gününe kadar
  saatlik veri tek sorguda çekilir; her artçıya kendi saatindeki değer eşlenir.
  (Artçı saatindeki hava o anda bilinen bilgidir → sızıntı yok.)
- ay_uzaklik_km: Ay'ın Dünya'ya uzaklığı (km), artçı anında.
- gelgit_ugal: Ay + Güneş'in, artçı anında ana deprem konumunda yarattığı dikey gelgit ivmesi
  (µGal = 1e-8 m/s²; pozitif = yukarı çekme). Formül: g = GM·R/d³ · (3·cos²θ − 1), θ = cismin
  başucu açısı. Ay ve Güneş'in konumu PyEphem (açık kaynak) ile hesaplanır; internet gerekmez.
  Basit model: Dünya'nın esnekliği (Love sayıları) ve okyanus gelgiti hesaba katılmaz.
  Dayanak: Ide, Yabe ve Tanaka (2016), Nature Geoscience (literatür Excel'inde).
  (9–12 Kasım 2026 ekip kararı ile eklendi.)
Çalıştırma:  .venv\\Scripts\\python.exe kod\\sude_veri\\dis_faktorler.py
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

KOK = Path(__file__).resolve().parents[2]
GIRDI = KOK / "veri" / "islenmis" / "artcilar.csv"
CIKTI = KOK / "veri" / "islenmis" / "artcilar_dis_faktorlu.csv"
ONBELLEK = KOK / "veri" / "ham" / "hava"

URL = "https://archive-api.open-meteo.com/v1/archive"
DEGISKENLER = "temperature_2m,pressure_msl,precipitation"   # basınç: deniz seviyesine indirgenmiş (9 Kasım ekip kararı)
SINODIK_AY = 29.530588853
REF_YENI_AY = pd.Timestamp("2000-01-06 18:14", tz="UTC")
BEKLEME_SN = 1.0       # Open-Meteo ücretsiz sınırlarına (dakikada 600, saatte 5.000) uymak için


def ay_evresi(zaman):
    gun = (zaman - REF_YENI_AY).dt.total_seconds() / 86400
    return np.mod(gun / SINODIK_AY, 1.0)


GM_AY, GM_GUNES = 4.9028e12, 1.32712440018e20     # m³/s²
DUNYA_R = 6.371e6                                   # m
AU_M = 149597870700.0                               # m


def gelgit(zaman, enlem, boylam):
    """Tek an ve konum için (Ay uzaklığı km, dikey gelgit ivmesi µGal)."""
    import ephem
    gozlem = ephem.Observer()
    gozlem.lat, gozlem.lon = str(enlem), str(boylam)
    gozlem.elevation, gozlem.pressure = 0, 0            # pressure=0: atmosferik kırılma hesaba katılmaz
    gozlem.date = ephem.Date(zaman.floor("s").to_pydatetime().replace(tzinfo=None))
    toplam, ay_km = 0.0, np.nan
    for cisim, gm in [(ephem.Moon(), GM_AY), (ephem.Sun(), GM_GUNES)]:
        cisim.compute(gozlem.date)                       # Dünya MERKEZİNDEN uzaklık (yer merkezli)
        d = cisim.earth_distance * AU_M
        cisim.compute(gozlem)                            # gözlem noktasından görünen yükseklik açısı
        cos_t = np.sin(float(cisim.alt))                 # başucu açısının kosinüsü = yükseklik açısının sinüsü
        toplam += gm * DUNYA_R / d ** 3 * (3 * cos_t ** 2 - 1)
        if gm == GM_AY:
            ay_km = d / 1000
    return ay_km, toplam / 1e-8


def hava_indir(dizi_id, enlem, boylam, bas, bit, deneme=8):
    dosya = ONBELLEK / f"{dizi_id}.csv"
    if dosya.exists():
        return pd.read_csv(dosya, parse_dates=["saat"])
    p = dict(latitude=round(enlem, 4), longitude=round(boylam, 4), start_date=bas, end_date=bit,
             hourly=DEGISKENLER, timezone="GMT", format="csv")
    for i in range(deneme):
        try:
            r = requests.get(URL, params=p, timeout=120)
            if r.status_code == 429:                       # sınır aşıldı → bekle
                print("   ! istek sınırı, 65 sn bekleniyor"); time.sleep(65); continue
            r.raise_for_status()
            metin = r.text
            # CSV başında konum bilgisi satırları var; saatlik tablo "time," ile başlar
            tablo = metin[metin.index("time,"):]
            h = pd.read_csv(io.StringIO(tablo))
            h.columns = ["saat", "sicaklik", "basinc", "yagis"]
            h["saat"] = pd.to_datetime(h["saat"], utc=True)
            h.to_csv(dosya, index=False)
            time.sleep(BEKLEME_SN)
            return h
        except (requests.RequestException, ValueError) as hata:
            bekle = 10 * (i + 1)
            print(f"   ! {dizi_id}: {hata.__class__.__name__}, {bekle} sn sonra tekrar")
            time.sleep(bekle)
    raise RuntimeError(f"{dizi_id} için hava verisi alınamadı")


def main():
    ONBELLEK.mkdir(parents=True, exist_ok=True)
    art = pd.read_csv(GIRDI, low_memory=False)
    art["artci_zaman"] = pd.to_datetime(art["artci_zaman"], utc=True, format="ISO8601")
    art["ay_evresi"] = ay_evresi(art["artci_zaman"])
    t0 = time.time()
    g = [gelgit(z, e, b) for z, e, b in zip(art["artci_zaman"], art["ana_enlem"], art["ana_boylam"])]
    art["ay_uzaklik_km"] = [x[0] for x in g]
    art["gelgit_ugal"] = [x[1] for x in g]
    print(f"Gelgit ve Ay uzaklığı hesaplandı ({time.time() - t0:.0f} sn)")

    diziler = art.groupby("dizi_id").agg(enlem=("ana_enlem", "first"), boylam=("ana_boylam", "first"),
                                          ilk=("artci_zaman", "min"), son=("artci_zaman", "max"))
    # ana deprem günü = ilk artçı gününden önce olabilir; zaman_farki ile geri hesapla
    ana_zaman = art.groupby("dizi_id").apply(
        lambda g: (g["artci_zaman"] - pd.to_timedelta(g["zaman_farki_saat"], unit="h")).iloc[0], include_groups=False)
    diziler["bas"] = ana_zaman.dt.strftime("%Y-%m-%d")
    diziler["bit"] = diziler["son"].dt.strftime("%Y-%m-%d")

    sinir = int(sys.argv[1]) if len(sys.argv) > 1 else None   # deneme için: ilk N dizi
    if sinir:
        diziler = diziler.head(sinir)
    parcalar = []
    t0 = time.time()
    for n, (dizi_id, d) in enumerate(diziler.iterrows(), 1):
        h = hava_indir(dizi_id, d.enlem, d.boylam, d.bas, d.bit)
        h["dizi_id"] = dizi_id
        parcalar.append(h)
        if n % 100 == 0:
            print(f"{n}/{len(diziler)} dizi tamam ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    hava = pd.concat(parcalar, ignore_index=True)

    art = art[art["dizi_id"].isin(diziler.index)].copy()
    art["saat"] = art["artci_zaman"].dt.floor("h")
    art = art.merge(hava, on=["dizi_id", "saat"], how="left").drop(columns="saat")

    if not sinir:
        art.to_csv(CIKTI, index=False)
    print("\n--- ÖZET ---")
    print("Artçı satırı:", len(art), "| dizi:", len(diziler))
    for c in ["ay_evresi", "ay_uzaklik_km", "gelgit_ugal", "sicaklik", "basinc", "yagis"]:
        print(f"{c:10s} boş: {art[c].isna().sum():5d} | min {art[c].min():8.2f} | ort {art[c].mean():8.2f} | max {art[c].max():8.2f}")
    print("Kaydedildi:" if not sinir else "(deneme çalıştırması, dosya yazılmadı)", CIKTI if not sinir else "")


if __name__ == "__main__":
    main()
