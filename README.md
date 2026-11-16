# Artçı Deprem Büyüklüğünün Makine Öğrenmesi ile Tahmini

MFK420 Mezuniyet Projesi · İstanbul Gedik Üniversitesi
Merve Naz Oran · Sude Anlaş · Hasan Erkeş · Danışman: Dr. Öğr. Üyesi Turgut PURA

Bütün kararlar için: **[Proje_Tasarimi.md](Proje_Tasarimi.md)**

## Kurulum (her bilgisayarda bir kez)
1. Python 3.11 kurun (python.org).
2. Proje klasöründe bir terminal açın:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma sırası
```
python kod/sude_veri/veri_cekme.py         # USGS'den veriyi indirir        → veri/ham/
python kod/sude_veri/on_isleme.py          # ana deprem ve artçıları ayırır → veri/islenmis/
python kod/sude_veri/dis_faktorler.py      # ay evresi + hava durumu ekler
python kod/merve_model/optuna_ayar.py      # (isteğe bağlı) model ayarlarını Optuna ile arar
python kod/merve_model/modelleme.py        # modelleri eğitir ve değerlendirir → sonuclar/, modeller/
python kod/merve_model/oznitelik_onemi.py  # hangi girdi ne kadar önemli → sonuclar/
streamlit run kod/hasan_arayuz/app.py      # demo arayüzü (ya da demo_baslat.bat'a çift tıkla)
```

## Klasörler
| Klasör | İçerik | Sorumlu |
|---|---|---|
| `kod/sude_veri/` | Veri indirme, ayıklama, dış faktör ekleme | Sude |
| `kod/merve_model/` | Model eğitimi, Optuna ayarı, öznitelik önemi | Merve |
| `kod/hasan_arayuz/` | Streamlit demo arayüzü (`eski_surumler/` içinde önceki sürümler) | Hasan |
| `veri/ham/`, `veri/islenmis/` | İndirilen ve işlenmiş veri (GitHub'a yüklenmez) | — |
| `sonuclar/` | Sonuç tabloları ve grafikler | — |
| `modeller/` | Eğitilmiş modeller | — |
| `sunumlar/`, `literatur/`, `ekip/`, `ders_belgeleri/` | Sunumlar ve belgeler | — |
