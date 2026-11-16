"""Adım 4 – Ana deprem / artçı ayırma ve sismik girdilerin hesaplanması.

Girdi : veri/ham/katalog_1980_2026.csv  (Adım 3)
Çıktı : veri/islenmis/ana_depremler.csv  (her satır bir ana deprem = bir dizi)
        veri/islenmis/artcilar.csv       (her satır bir artçı; model tablosu)

Kurallar (Proje_Tasarimi.md, bölüm 3–4):
- Sadece "earthquake" türündeki ve M >= 4,5 olaylar kullanılır.
- Mekânsal pencere (Gardner–Knopoff 1974): L(M) = 10^(0,1238·M + 0,983) km
- Zaman penceresi: ana depremden sonraki 30 gün
- Ana deprem: M >= 6, en geç 31 Mayıs 2026; kendinden önceki 30 gün ve L km
  içinde daha büyük bir deprem yok ve başka bir dizinin penceresine düşmüyor.
- Çakışma kuralı: bir dizinin penceresine düşen olay ayrı dizi başlatmaz,
  o dizinin artçısı sayılır (ör. Kahramanmaraş'ta M7,5 Elbistan → M7,8'in artçısı).
- En büyük ana deprem kuralı (9 Kasım 2026, ekip kararı): pencereye ana depremden
  BÜYÜK bir deprem düşerse dizinin ana depremi o olur; ondan öncekiler öncü
  deprem sayılıp çıkarılır ve pencere yeni ana depreme göre yeniden başlar
  (ör. 2011 Tohoku M9,1 → 2 gün önceki M7,3 öncü olur).
- Sızıntı kuralı: "önceki_*" girdileri sadece o artçıdan önceki artçılardan hesaplanır.
"""
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
HAM = KOK / "veri" / "ham" / "katalog_1980_2026.csv"
CIKTI = KOK / "veri" / "islenmis"

ANA_ESIK, ARTCI_ESIK = 6.0, 4.5
PENCERE_GUN = 30
SON_ANA_TARIH = pd.Timestamp("2026-06-01", tz="UTC")   # 31 Mayıs 2026 dahil

# Türkiye test vakaları (eğitimden ayrı tutulacak): tarih ve yaklaşık merkez
TURKIYE = {
    "Izmit_1999":          ("1999-08-17", 40.75, 29.86),
    "Sisam_Izmir_2020":    ("2020-10-30", 37.90, 26.79),
    "Kahramanmaras_2023":  ("2023-02-06", 37.23, 37.02),
}


def L_km(m):
    return 10 ** (0.1238 * m + 0.983)


def uzaklik_km(enlem1, boylam1, enlem2, boylam2):
    """Haversine (büyük daire) uzaklığı, km."""
    e1, b1, e2, b2 = map(np.radians, (enlem1, boylam1, enlem2, boylam2))
    a = np.sin((e2 - e1) / 2) ** 2 + np.cos(e1) * np.cos(e2) * np.sin((b2 - b1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def katalog_oku():
    df = pd.read_csv(HAM, usecols=["time", "latitude", "longitude", "depth", "mag", "magType", "id", "place", "type"])
    n0 = len(df)
    df = df[df["type"] == "earthquake"]
    n1 = len(df)
    df = df[df["mag"] >= ARTCI_ESIK]
    n2 = len(df)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    df = df.sort_values("time").reset_index(drop=True)
    print(f"Katalog: {n0} kayıt → deprem olmayan {n0 - n1} çıkarıldı → M<4,5 olan {n1 - n2} çıkarıldı → {n2} kayıt")
    return df


def dizileri_bul(df):
    """Zaman sırasıyla tek geçiş. Her olay için dizi_id (ana deprem id'si) atanır."""
    zaman = df["time"].values.astype("datetime64[s]").astype(np.int64)  # saniye
    enlem, boylam, mag = df["latitude"].values, df["longitude"].values, df["mag"].values
    pencere = PENCERE_GUN * 86400
    son_ana = SON_ANA_TARIH.value // 10**9

    dizi_id = np.full(len(df), -1)        # -1: hiçbir diziye ait değil
    rol = np.array([""] * len(df), dtype=object)
    aktif = []        # açık diziler: [ana_indeks, zaman, enlem, boylam, L, mag, üyeler]
    buyukler = []     # son 30 gündeki M>=6 olaylar: (indeks, zaman, enlem, boylam, mag)
    engellenen = yer_degisen = kesilen = 0

    def diziyi_birak(a, yeni_rol):
        for j in a[6]:
            dizi_id[j], rol[j] = -1, yeni_rol

    for i in range(len(df)):
        t = zaman[i]
        aktif = [a for a in aktif if t - a[1] <= pencere]
        buyukler = [b for b in buyukler if t - b[1] <= pencere]

        # 1) Açık bir dizinin penceresine düşüyor mu? (ilk açılan dizi öncelikli)
        sahip = None
        for a in aktif:
            if uzaklik_km(enlem[i], boylam[i], a[2], a[3]) <= a[4]:
                sahip = a
                break
        if sahip is not None and mag[i] > sahip[5]:
            # Dizide ana depremden büyük deprem geldi → en büyüğü ana deprem olur.
            # Öncekiler (eski ana + o ana kadarki artçılar) öncü sayılıp çıkarılır.
            aktif.remove(sahip)
            if t < son_ana:
                diziyi_birak(sahip, "oncu")
                yer_degisen += 1
                dizi_id[i], rol[i] = i, "ana"
                aktif.append([i, t, enlem[i], boylam[i], L_km(mag[i]), mag[i], [i]])
            else:   # 31 Mayıs'tan sonra: 30 günlük takip yapılamaz → dizi tamamen çıkarılır
                diziyi_birak(sahip, "kesik")
                rol[i] = "kesik"
                kesilen += 1
        elif sahip is not None:
            dizi_id[i], rol[i] = sahip[0], "artci"
            sahip[6].append(i)
        elif mag[i] >= ANA_ESIK and t < son_ana:
            # 2) Önceki 30 gün ve L km içinde daha büyük deprem var mı?
            daha_buyuk = any(b[4] > mag[i] and uzaklik_km(enlem[i], boylam[i], b[2], b[3]) <= L_km(b[4])
                             for b in buyukler)
            if daha_buyuk:
                engellenen += 1
            else:
                dizi_id[i], rol[i] = i, "ana"
                aktif.append([i, t, enlem[i], boylam[i], L_km(mag[i]), mag[i], [i]])
        if mag[i] >= ANA_ESIK:
            buyukler.append((i, t, enlem[i], boylam[i], mag[i]))

    print(f"Başka bir büyük depremin etkisinde olduğu için ana deprem sayılmayan M>=6 olay: {engellenen}")
    print(f"Daha büyük deprem geldiği için ana depremi değişen dizi: {yer_degisen} "
          f"(öncü sayılıp çıkarılan olay: {(rol == 'oncu').sum()})")
    print(f"31 Mayıs 2026 sonrası büyüdüğü için çıkarılan dizi: {kesilen}")
    return dizi_id, rol


def tablolari_kur(df, dizi_id, rol):
    df = df.assign(dizi_idx=dizi_id, rol=rol)
    ana = df[df["rol"] == "ana"].copy()
    ana["dizi_id"] = ana["id"]
    idx2id = dict(zip(ana.index, ana["id"]))

    # Türkiye vakalarını işaretle
    ana["turkiye_vakasi"] = ""
    for ad, (tarih, e, b) in TURKIYE.items():
        aday = ana[(ana["time"].dt.strftime("%Y-%m-%d") == tarih)]
        aday = aday[uzaklik_km(aday["latitude"], aday["longitude"], e, b) < 150]
        if len(aday):
            ana.loc[aday["mag"].idxmax(), "turkiye_vakasi"] = ad
        else:
            print(f"  ! {ad}: ana deprem olarak bulunamadı")

    art = df[df["rol"] == "artci"].copy()
    art["dizi_id"] = art["dizi_idx"].map(idx2id)
    a = ana.set_index("dizi_id")
    art = art.join(a[["time", "mag", "depth", "latitude", "longitude", "turkiye_vakasi"]]
                   .rename(columns={"time": "ana_zaman", "mag": "ana_buyukluk", "depth": "ana_derinlik",
                                    "latitude": "ana_enlem", "longitude": "ana_boylam"}), on="dizi_id")

    art["artci_buyukluk"] = art["mag"]
    art["artci_derinlik"] = art["depth"]
    art["zaman_farki_saat"] = (art["time"] - art["ana_zaman"]).dt.total_seconds() / 3600
    art["uzaklik_km"] = uzaklik_km(art["latitude"], art["longitude"], art["ana_enlem"], art["ana_boylam"])

    # Önceki artçılar (sadece geçmiş bilgi; shift ile kendisi hariç)
    art = art.sort_values(["dizi_id", "time"])
    g = art.groupby("dizi_id")
    art["onceki_sayi"] = g.cumcount()
    gecmis = g["mag"].shift(1)
    art["onceki_ort"] = gecmis.groupby(art["dizi_id"]).expanding().mean().reset_index(level=0, drop=True)
    art["onceki_max"] = gecmis.groupby(art["dizi_id"]).cummax()
    art["onceki_artcidan_saat"] = g["time"].diff().dt.total_seconds() / 3600

    sutunlar = ["dizi_id", "id", "time", "turkiye_vakasi", "artci_buyukluk",
                "ana_buyukluk", "ana_derinlik", "ana_enlem", "ana_boylam",
                "artci_derinlik", "zaman_farki_saat", "uzaklik_km",
                "onceki_sayi", "onceki_ort", "onceki_max", "onceki_artcidan_saat",
                "latitude", "longitude", "magType", "place"]
    art = art[sutunlar].rename(columns={"id": "artci_id", "time": "artci_zaman",
                                        "latitude": "artci_enlem", "longitude": "artci_boylam"})

    sayilar = art.groupby("dizi_id").size()
    ana["artci_sayisi"] = ana["dizi_id"].map(sayilar).fillna(0).astype(int)
    ana = ana[["dizi_id", "time", "latitude", "longitude", "depth", "mag", "magType", "place",
               "turkiye_vakasi", "artci_sayisi"]]
    return ana, art.reset_index(drop=True)


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)
    df = katalog_oku()
    dizi_id, rol = dizileri_bul(df)
    ana, art = tablolari_kur(df, dizi_id, rol)
    ana.to_csv(CIKTI / "ana_depremler.csv", index=False)
    art.to_csv(CIKTI / "artcilar.csv", index=False)

    print("\n--- ÖZET ---")
    print("Ana deprem (dizi) sayısı :", len(ana))
    print("  artçısı olan dizi      :", (ana["artci_sayisi"] > 0).sum())
    print("  artçısı olmayan dizi   :", (ana["artci_sayisi"] == 0).sum())
    print("Artçı (model satırı)     :", len(art))
    print("Dizi başına artçı (artçısı olanlarda): medyan",
          ana.loc[ana.artci_sayisi > 0, "artci_sayisi"].median(), "| en çok", ana["artci_sayisi"].max())
    print("Artçı büyüklüğü: ort %.2f, std %.2f, min %.2f, max %.2f" % (
        art.artci_buyukluk.mean(), art.artci_buyukluk.std(), art.artci_buyukluk.min(), art.artci_buyukluk.max()))
    print("Ana depremden büyük artçı:", (art.artci_buyukluk > art.ana_buyukluk).sum())
    print("\nTürkiye vakaları:")
    print(ana[ana.turkiye_vakasi != ""][["turkiye_vakasi", "time", "mag", "artci_sayisi"]].to_string(index=False))


if __name__ == "__main__":
    main()
