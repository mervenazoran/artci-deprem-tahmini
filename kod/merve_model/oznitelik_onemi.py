"""Adım 6b – Öznitelik (girdi) önemi: hangi girdi tahmine ne kadar katkı yapıyor?

Yöntem: Permütasyon önemi. Yaklaşım B'nin en iyi modeli (model_sonuclari.csv'den; Optuna ayarı varsa
her katta o katın ayarıyla) 5 katlı dizi bazlı çapraz doğrulamada
her katın eğitim kısmıyla eğitilir; test kısmında bir girdinin değerleri rastgele karıştırılır
ve hatanın (MAE) ne kadar arttığına bakılır. Hata çok artıyorsa girdi önemlidir; artmıyorsa
model o girdiden fayda görmüyordur. Ölçüm modelin görmediği verilerde yapılır.

Girdi : veri/islenmis/artcilar_dis_faktorlu.csv
Çıktı : sonuclar/tablolar/oznitelik_onemi.csv
        sonuclar/grafikler/oznitelik_onemi.png
Çalıştırma: .venv\\Scripts\\python.exe kod\\merve_model\\oznitelik_onemi.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelleme as M  # aynı veri, girdiler, model ayarları ve katlar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

TEKRAR = 5   # her girdi her katta 5 kez karıştırılır


def main():
    df = pd.read_csv(M.VERI, low_memory=False)
    egitim = df[df["turkiye_vakasi"].isna()].reset_index(drop=True)
    girdiler = M.SISMIK + M.DIS                     # Yaklaşım B: dış faktörler de görünsün
    X, y, g = egitim[girdiler], egitim[M.HEDEF].to_numpy(), egitim["dizi_id"].to_numpy()

    yak = "B (sismik + dış faktör)"
    ozet = pd.read_csv(M.TABLO / "model_sonuclari.csv")
    ml = ozet[(ozet.yaklasim == yak) & ozet.model.isin(["Random Forest", "XGBoost", "LightGBM"])]
    ad = ml.sort_values("MAE").iloc[0]["model"]
    ayar = M.ayarlari_oku()
    print(f"Model: {ad} ({'Optuna ayarı' if ayar else 'sabit ayar'})")

    sonuc = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=M.KAT).split(X, y, g), 1):
        m = M.model_kur(ad, yak, ayar, k).fit(X.iloc[tr], y[tr])
        p = permutation_importance(m, X.iloc[te], y[te], scoring="neg_mean_absolute_error",
                                   n_repeats=TEKRAR, random_state=M.TOHUM, n_jobs=1)
        sonuc.append(p.importances_mean)
        print(f"Kat {k} tamam")
    s = np.array(sonuc)                               # kat × girdi; değer = MAE artışı

    tablo = pd.DataFrame({
        "girdi": girdiler,
        "aciklama": [M.ACIKLAMA[c] for c in girdiler],
        "grup": ["Dış faktör" if c in M.DIS else "Sismik" for c in girdiler],
        "MAE_artisi": s.mean(0), "MAE_artisi_std": s.std(0, ddof=1),
    }).sort_values("MAE_artisi", ascending=False).reset_index(drop=True)
    tablo["onem_%"] = (tablo.MAE_artisi.clip(lower=0) / tablo.MAE_artisi.clip(lower=0).sum() * 100).round(1)
    tablo.insert(0, "sira", range(1, len(tablo) + 1))

    M.TABLO.mkdir(parents=True, exist_ok=True); M.GRAFIK.mkdir(parents=True, exist_ok=True)
    tablo.round(5).to_csv(M.TABLO / "oznitelik_onemi.csv", index=False, encoding="utf-8-sig")

    t = tablo.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    renk = ["#F58518" if gr == "Dış faktör" else "#4C78A8" for gr in t.grup]
    ax.barh(t.aciklama, t.MAE_artisi, xerr=t.MAE_artisi_std, color=renk, capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Girdi karıştırılınca ortalama hatadaki (MAE) artış — büyük = daha önemli")
    ax.set_title(f"Hangi girdi daha önemli? ({ad}, 5 katlı dizi bazlı doğrulama)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C78A8", label="Sismik"), Patch(color="#F58518", label="Dış faktör")],
              loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(M.GRAFIK / "oznitelik_onemi.png", dpi=150)

    print("\n--- ÖZNİTELİK ÖNEMİ (büyükten küçüğe) ---")
    print(tablo[["sira", "aciklama", "grup", "MAE_artisi", "MAE_artisi_std", "onem_%"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nKaydedildi:", M.GRAFIK / "oznitelik_onemi.png")


if __name__ == "__main__":
    main()
