"""Adım 6c – Optuna ile model ayarı (iç içe çapraz doğrulama).

Neden iç içe? Ayarları, sonucu ölçtüğümüz verinin üstünde seçersek sonuç olduğundan iyi görünür
(test verisi ayar seçimine "sızar"). Bu yüzden:
  Dış döngü : modelleme.py ile AYNI 5 katlı, dizi bazlı bölme (GroupKFold).
  İç döngü  : her dış katın SADECE eğitim kısmında, 3 katlı dizi bazlı bölmeyle Optuna ayar arar.
  → Dış katın test kısmı ayar seçimine hiç karışmaz; raporlanan sonuç dürüsttür.
Son model (demo) için ayar, bütün eğitim verisinde (yine 3 katlı iç bölmeyle) bir kez daha aranır.
Türkiye vakaları hiçbir aşamada kullanılmaz.

Her aramada ilk deneme, vizedeki sabit ayarlardır (enqueue): Optuna en kötü ihtimalle onları seçer.

Girdi : veri/islenmis/artcilar_dis_faktorlu.csv
Çıktı : sonuclar/tablolar/optuna_ayarlar.json   (her yaklaşım × model: 5 dış katın ayarı + son ayar)
        sonuclar/tablolar/optuna_kayit.csv       (her denemenin kaydı: ayarlar ve iç MAE)
Çalıştırma: .venv\\Scripts\\python.exe kod\\merve_model\\optuna_ayar.py
Sonra:      .venv\\Scripts\\python.exe kod\\merve_model\\modelleme.py   (ayar dosyası varsa onu kullanır)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelleme as M

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

IC_KAT = 3
DENEME = {"LightGBM": 40, "XGBoost": 40, "Random Forest": 20}
CIKTI = M.TABLO / "optuna_ayarlar.json"
KAYIT = M.TABLO / "optuna_kayit.csv"


def arama_uzayi(ad, t):
    if ad == "LightGBM":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 1500, step=100),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    num_leaves=t.suggest_int("num_leaves", 8, 128, log=True),
                    min_child_samples=t.suggest_int("min_child_samples", 10, 200, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    if ad == "XGBoost":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 1500, step=100),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    max_depth=t.suggest_int("max_depth", 3, 10),
                    min_child_weight=t.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                    subsample=t.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True))
    return dict(min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 50, log=True),
                max_features=t.suggest_float("max_features", 0.2, 1.0),
                max_depth=t.suggest_int("max_depth", 6, 40, log=True))


VARSAYILAN = {   # vizedeki sabit ayarlar (modelleme.py ile aynı) – ilk deneme olarak verilir
    "LightGBM": dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=20, subsample=0.8,
                     colsample_bytree=0.8, reg_lambda=1e-3),
    "XGBoost": dict(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=1.0, subsample=0.8,
                    colsample_bytree=0.8, reg_lambda=1.0),
    "Random Forest": dict(min_samples_leaf=5, max_features=0.5, max_depth=40),
}


def ara(ad, X, y, g, etiket, kayit):
    ic = list(GroupKFold(n_splits=IC_KAT).split(X, y, g))

    def hedef(t):
        p = arama_uzayi(ad, t)
        hatalar = []
        for tr, te in ic:
            m = M.kur(ad, p).fit(X.iloc[tr], y[tr])
            hatalar.append(mean_absolute_error(y[te], m.predict(X.iloc[te])))
        return float(np.mean(hatalar))

    cal = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=M.TOHUM))
    cal.enqueue_trial(VARSAYILAN[ad])
    cal.optimize(hedef, n_trials=DENEME[ad])
    for tr in cal.trials:
        kayit.append(dict(etiket=etiket, model=ad, deneme=tr.number, ic_MAE=tr.value, **tr.params))
    return cal.best_params, float(cal.best_value), float(cal.trials[0].value)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df = pd.read_csv(M.VERI, low_memory=False)
    egitim = df[df["turkiye_vakasi"].isna()].reset_index(drop=True)
    y, g = egitim[M.HEDEF].to_numpy(), egitim["dizi_id"].to_numpy()
    dis_katlar = list(GroupKFold(n_splits=M.KAT).split(egitim, y, g))   # modelleme.py ile aynı bölme

    sonuc, kayit, t_bas = {}, [], time.time()
    for yak, girdiler in M.YAKLASIMLAR.items():
        X = egitim[girdiler]
        sonuc[yak] = {}
        for ad in DENEME:
            t0 = time.time()
            katlar = []
            for k, (tr, _) in enumerate(dis_katlar, 1):   # test kısmı (_) burada HİÇ kullanılmaz
                p, iyi, vars_ = ara(ad, X.iloc[tr].reset_index(drop=True), y[tr], g[tr], f"{yak} | dış kat {k}", kayit)
                katlar.append(p)
                print(f"{yak[:1]} {ad:13s} dış kat {k}: iç MAE {vars_:.4f} → {iyi:.4f}", flush=True)
            p, iyi, vars_ = ara(ad, X, y, g, f"{yak} | son model", kayit)
            print(f"{yak[:1]} {ad:13s} SON MODEL : iç MAE {vars_:.4f} → {iyi:.4f}   "
                  f"({(time.time() - t0) / 60:.1f} dk, toplam {(time.time() - t_bas) / 60:.1f} dk)", flush=True)
            sonuc[yak][ad] = dict(katlar=katlar, son=p)
            # her modelden sonra kaydet: yarıda kesilirse emek kaybolmasın
            CIKTI.write_text(json.dumps(sonuc, ensure_ascii=False, indent=1), encoding="utf-8")
            pd.DataFrame(kayit).to_csv(KAYIT, index=False, encoding="utf-8-sig")
    print("Kaydedildi:", CIKTI, "|", KAYIT)


if __name__ == "__main__":
    main()
