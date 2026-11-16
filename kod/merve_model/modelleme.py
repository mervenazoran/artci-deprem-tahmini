"""Adım 6 – Modelleme ve değerlendirme (vize kapsamı).

Girdi : veri/islenmis/artcilar_dis_faktorlu.csv  (Adım 5)
Çıktı : sonuclar/tablolar/model_sonuclari.csv          (her model × yaklaşım: ortalama ± std)
        sonuclar/tablolar/model_sonuclari_katlar.csv   (her katın ayrı sonucu)
        sonuclar/grafikler/model_karsilastirma.png
        modeller/en_iyi_model_A.joblib, en_iyi_model_B.joblib  (demo için; tüm eğitim verisiyle)

Tasarım (Proje_Tasarimi.md):
- Türkiye vakaları (Kahramanmaraş 2023, İzmit 1999, Sisam–İzmir 2020) eğitimden tamamen ayrı.
- Yaklaşım A: sismik girdiler.  Yaklaşım B: sismik + dış faktörler.
- Basit karşılaştırma: önceki artçıların ortalaması (ilk artçıda eğitim ortalaması), doğrusal regresyon.
- ML: Random Forest, XGBoost, LightGBM. Ayarlar: sonuclar/tablolar/optuna_ayarlar.json varsa Optuna ile
  bulunan ayarlar (kod/merve_model/optuna_ayar.py, iç içe doğrulama: her dış katın ayarı sadece o katın eğitim kısmında
  arandı), yoksa sabit başlangıç ayarları. (13 Kasım 2026 ekip kararı: Optuna vizeye alındı.)
  Komut satırına --sabit eklenirse ayar dosyası yok sayılır.
- 5 katlı, dizi bazlı çapraz doğrulama (GroupKFold): bir dizinin tüm artçıları aynı katta.
- Ölçütler: MAE, RMSE, R².

Çalıştırma:
  .venv\\Scripts\\python.exe kod\\merve_model\\modelleme.py                       → gerçek çalıştırma, dosyaları yazar
  .venv\\Scripts\\python.exe kod\\merve_model\\modelleme.py <veri.csv> --deneme   → kod denemesi, hiçbir şey kaydetmez
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

KOK = Path(__file__).resolve().parents[2]
VERI = KOK / "veri" / "islenmis" / "artcilar_dis_faktorlu.csv"
TABLO = KOK / "sonuclar" / "tablolar"
GRAFIK = KOK / "sonuclar" / "grafikler"
MODEL = KOK / "modeller"

HEDEF = "artci_buyukluk"
SISMIK = ["ana_buyukluk", "ana_derinlik", "ana_enlem", "ana_boylam", "artci_derinlik",
          "zaman_farki_saat", "uzaklik_km", "onceki_sayi", "onceki_ort", "onceki_max",
          "onceki_artcidan_saat"]
DIS = ["ay_evresi", "ay_uzaklik_km", "gelgit_ugal", "sicaklik", "basinc", "yagis"]   # gelgit + Ay uzaklığı: 12 Kasım
YAKLASIMLAR = {"A (sismik)": SISMIK, "B (sismik + dış faktör)": SISMIK + DIS}
ACIKLAMA = {
    "ana_buyukluk": "Ana deprem büyüklüğü",
    "ana_derinlik": "Ana deprem derinliği (km)",
    "ana_enlem": "Ana deprem enlemi",
    "ana_boylam": "Ana deprem boylamı",
    "artci_derinlik": "Artçı derinliği (km)",
    "zaman_farki_saat": "Ana şoktan geçen süre (saat)",
    "uzaklik_km": "Ana şoka uzaklık (km)",
    "onceki_sayi": "Önceki artçı sayısı",
    "onceki_ort": "Önceki artçıların ortalaması",
    "onceki_max": "Önceki en büyük artçı",
    "onceki_artcidan_saat": "Önceki artçıdan geçen süre",
    "ay_evresi": "Ay evresi (0 = yeni ay, 0,5 = dolunay)",
    "ay_uzaklik_km": "Ay'ın Dünya'ya uzaklığı (km)",
    "gelgit_ugal": "Gelgit kuvveti (µGal)",
    "sicaklik": "Sıcaklık (°C)",
    "basinc": "Deniz seviyesi basıncı (hPa)",
    "yagis": "Yağış (mm)",
}


def girdileri_yaz():
    print("--- KULLANILAN GİRDİLER ---")
    print(f"Yaklaşım A (sismik) – {len(SISMIK)} girdi:")
    for i, g in enumerate(SISMIK, 1):
        print(f" {i:2d}. {ACIKLAMA[g]:30s} ({g})")
    print(f"\nYaklaşım B (sismik + dış faktör) – {len(SISMIK) + len(DIS)} girdi:")
    print(f"  A'daki {len(SISMIK)} girdi +")
    for i, g in enumerate(DIS, len(SISMIK) + 1):
        print(f" {i:2d}. {ACIKLAMA[g]:40s} ({g})")
    print(f"\nHedef: Artçı büyüklüğü ({HEDEF})\n")
KAT = 5
TOHUM = 42


class OncekiOrtalama:
    """Basit karşılaştırma: tahmin = önceki artçıların ortalaması; ilk artçıda eğitim ortalaması."""
    def fit(self, X, y):
        self.ort_ = float(np.mean(y))
        return self

    def predict(self, X):
        return X["onceki_ort"].fillna(self.ort_).to_numpy()


def modeller():
    return {
        "Önceki artçı ortalaması": lambda: OncekiOrtalama(),
        "Doğrusal regresyon": lambda: make_pipeline(SimpleImputer(strategy="mean", add_indicator=True),
                                                    StandardScaler(), LinearRegression()),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                       n_jobs=-1, random_state=TOHUM),
        "XGBoost": lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                            colsample_bytree=0.8, n_jobs=-1, random_state=TOHUM),
        "LightGBM": lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                              n_jobs=-1, random_state=TOHUM, verbose=-1),
    }


AYAR_DOSYASI = TABLO / "optuna_ayarlar.json"


def kur(ad, p):
    """Ayarları verilen modeli kurar (Optuna'nın bulduğu ayarlar + sabit kısımlar)."""
    if ad == "Random Forest":
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=TOHUM, **p)
    if ad == "XGBoost":
        return xgb.XGBRegressor(n_jobs=-1, random_state=TOHUM, **p)
    if ad == "LightGBM":
        return lgb.LGBMRegressor(subsample_freq=1, n_jobs=-1, random_state=TOHUM, verbose=-1, **p)
    raise ValueError(ad)


def ayarlari_oku():
    """Optuna ayar dosyası varsa {yaklaşım: {model: {katlar: [...], son: {...}}}} döner, yoksa None."""
    if "--sabit" in sys.argv or not AYAR_DOSYASI.exists():
        return None
    return json.loads(AYAR_DOSYASI.read_text(encoding="utf-8"))


def model_kur(ad, yak, ayar, kat=None):
    """CV'de kat'ın (1–5) ayarıyla, son modelde (kat=None) 'son' ayarla kurar; ayar yoksa sabit ayar."""
    if ayar and ad in ayar.get(yak, {}):
        a = ayar[yak][ad]
        return kur(ad, a["katlar"][kat - 1] if kat else a["son"])
    return modeller()[ad]()


def olc(y, p):
    return dict(MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=r2_score(y, p))


def main():
    deneme = "--deneme" in sys.argv
    ayar = ayarlari_oku()
    print("Model ayarları:", "Optuna (iç içe doğrulama)" if ayar else "sabit başlangıç ayarları")
    girdileri_yaz()
    yol =Path(next((a for a in sys.argv[1:] if not a.startswith("--")), VERI))
    df = pd.read_csv(yol, low_memory=False)
    turkiye = df["turkiye_vakasi"].notna()
    egitim = df[~turkiye].reset_index(drop=True)
    print(f"Veri: {len(df)} artçı | Türkiye vakaları ayrıldı: {turkiye.sum()} artçı "
          f"| eğitim/doğrulama: {len(egitim)} artçı, {egitim.dizi_id.nunique()} dizi")

    y, gruplar = egitim[HEDEF].to_numpy(), egitim["dizi_id"].to_numpy()
    katlar = list(GroupKFold(n_splits=KAT).split(egitim, y, gruplar))
    # Güvenlik kontrolü: hiçbir dizi hem eğitimde hem testte olmamalı
    assert all(not set(gruplar[tr]) & set(gruplar[te]) for tr, te in katlar)

    satirlar = []
    for yak, girdiler in YAKLASIMLAR.items():
        X = egitim[girdiler]
        for ad, kur in modeller().items():
            if ad == "Önceki artçı ortalaması" and yak.startswith("B"):
                continue   # bu yöntem girdi kullanmaz; A ve B'de aynı sonuç verir
            t0 = time.time()
            for k, (tr, te) in enumerate(katlar, 1):
                m = (model_kur(ad, yak, ayar, k) if ad in ("Random Forest", "XGBoost", "LightGBM")
                     else kur()).fit(X.iloc[tr], y[tr])
                satirlar.append(dict(yaklasim=yak, model=ad, kat=k, **olc(y[te], m.predict(X.iloc[te]))))
            s = pd.DataFrame([r for r in satirlar if r["yaklasim"] == yak and r["model"] == ad])
            print(f"{yak:24s} {ad:24s} MAE {s.MAE.mean():.3f} ± {s.MAE.std():.3f} | "
                  f"RMSE {s.RMSE.mean():.3f} | R² {s.R2.mean():.3f}   ({time.time() - t0:.0f} sn)")

    katlar_df = pd.DataFrame(satirlar)
    ozet = katlar_df.groupby(["yaklasim", "model"], sort=False).agg(
        MAE=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2=("R2", "mean"), R2_std=("R2", "std")).reset_index().round(4)

    taban = ozet.loc[ozet.model == "Önceki artçı ortalaması", "MAE"].iloc[0]
    ozet["taban_gore_MAE_iyilesme_%"] = ((taban - ozet["MAE"]) / taban * 100).round(1)
    print("\n--- ÖZET (5 kat ortalaması) ---")
    print(ozet.to_string(index=False))

    if deneme:
        print("\n(deneme çalıştırması: dosya kaydedilmedi)")
        return

    TABLO.mkdir(parents=True, exist_ok=True); GRAFIK.mkdir(parents=True, exist_ok=True); MODEL.mkdir(exist_ok=True)
    ozet.to_csv(TABLO / "model_sonuclari.csv", index=False, encoding="utf-8-sig")
    katlar_df.round(4).to_csv(TABLO / "model_sonuclari_katlar.csv", index=False, encoding="utf-8-sig")

    # Grafik: MAE karşılaştırması (düşük daha iyi)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    adlar = list(modeller())
    gen = 0.38
    for i, (yak, renk) in enumerate(zip(YAKLASIMLAR, ["#4C78A8", "#F58518"])):
        d = ozet[ozet.yaklasim == yak].set_index("model").reindex(adlar)
        if yak.startswith("B"):
            d.loc["Önceki artçı ortalaması"] = np.nan
        x = np.arange(len(adlar)) + (i - 0.5) * gen
        ax.bar(x, d["MAE"], gen, yerr=d["MAE_std"], capsize=3, label=yak, color=renk)
    ax.set_xticks(np.arange(len(adlar)), adlar, rotation=12)
    ax.set_ylabel("MAE (büyüklük birimi) — düşük daha iyi")
    ax.set_title("Artçı büyüklüğü tahmini: 5 katlı dizi bazlı çapraz doğrulama")
    ax.set_ylim(ozet.MAE.min() * 0.85, ozet.MAE.max() * 1.05)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(GRAFIK / "model_karsilastirma.png", dpi=150)

    # B'nin en iyi ML modelini seç; A için de AYNI model kaydedilir ki demodaki A–B karşılaştırması
    # sadece dış faktörlerin etkisini göstersin (model farkı karışmasın). Tüm eğitim verisiyle eğitilir.
    yak_B = list(YAKLASIMLAR)[1]
    ml = ozet[(ozet.yaklasim == yak_B) & ozet.model.isin(["Random Forest", "XGBoost", "LightGBM"])]
    en_iyi = ml.sort_values("MAE").iloc[0]["model"]
    for yak, girdiler in YAKLASIMLAR.items():
        m = model_kur(en_iyi, yak, ayar).fit(egitim[girdiler], y)
        harf = yak[0]
        joblib.dump(dict(model=m, model_adi=en_iyi, girdiler=girdiler, yaklasim=yak,
                         ayar="Optuna" if ayar else "sabit", parametreler=m.get_params()),
                    MODEL / f"en_iyi_model_{harf}.joblib")
        print(f"Kaydedildi: en_iyi_model_{harf}.joblib ({en_iyi})")
    print("Tablolar:", TABLO, "| Grafik:", GRAFIK / "model_karsilastirma.png")


if __name__ == "__main__":
    main()
