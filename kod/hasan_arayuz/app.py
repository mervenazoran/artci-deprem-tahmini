"""Adım 7 – Vize demosu (Streamlit arayüzü).

Başlatma: proje klasöründeki demo_baslat.bat dosyasına çift tıkla
      ya da: .venv\\Scripts\\python.exe -m streamlit run kod\\hasan_arayuz\\app.py
Tarayıcıda http://localhost:8501 açılır.
Önceki sürümler: kod/hasan_arayuz/eski_surumler/ (app_eski.py … app_v14_yedek.py)

Sekmeler:
  1) Tahmin            : deprem ve an seçimi · dış koşul kartları · modelin kullandığı bilgiler · tahmin ve gerçek
  2) Türkiye örnekleri : 3 deprem yan yana · deprem anındaki dış koşullar · harita · grafik · tablo
  3) Sonuçlar          : 3 soru-cevap kartı · model karşılaştırma çubukları · sınırlar · teknik ayrıntılar
Arayüz düzeni 13 Kasım 2026'da jüri bakışıyla sadeleştirildi; önceki hali: yedek/arayuz_oncesi_2026-11-13/app.py
Renk/görsel tema 14 Kasım 2026'da USGS + kurumsal mavi esinli laciverte çevrildi (emoji ikonlar korundu);
önceki (mürdüm) hali: yedek/tasarim_oncesi_2026-11-14b/kod/app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sude_veri"))
from dis_faktorler import gelgit   # Ay uzaklığı ve gelgit kuvveti: veri hazırlamadaki aynı fonksiyon (Sude, Hasan)

KOK = Path(__file__).resolve().parents[2]
VERI = KOK / "veri" / "islenmis" / "artcilar_dis_faktorlu.csv"
MODEL = KOK / "modeller"
TABLO = KOK / "sonuclar" / "tablolar"
GRAFIK = KOK / "sonuclar" / "grafikler"

ANA_RENK, GRI = "#1F5FA8", "#AEB7C0"     # 14 Kasım: USGS + kurumsal mavi esinli lacivert; koyu temada dolu/mavi şekiller
RENK_VURGU = "#6CB2F2"                   # yazı olarak kullanılan vurgu mavisi (koyu zeminde daha okunur)
RENK_ACIK, RENK_ACIK2 = "#182430", "#202D3B"    # koyu lacivert/gri kart zeminleri (önceki: açık mavi pastel)
METIN_ACIK, METIN_ORTA, METIN_SOLUK = "#E7ECF1", "#9AA7B4", "#76828F"   # koyu zeminde açık/orta/soluk yazı
CIZGI, TABAN_CUBUK = "#2A3642", "#232E3A"       # ince ayraç çizgisi ve nötr çubuk zemini
SINODIK_AY, REF_YENI_AY = 29.530588853, pd.Timestamp("2000-01-06 18:14", tz="UTC")  # dis_faktorler.py ile aynı
TR = {  # Türkiye ana depremleri (USGS ComCat değerleri)
    "Kahramanmaras_2023": dict(ad="Kahramanmaraş 2023", uzun="Kahramanmaraş – 6 Şubat 2023 (M 7,8)",
                               m=7.8, d=10.0, en=37.2256, bo=37.0143),
    "Izmit_1999": dict(ad="İzmit 1999", uzun="İzmit (Gölcük) – 17 Ağustos 1999 (M 7,6)",
                       m=7.6, d=17.0, en=40.7480, bo=29.8640),
    "Sisam_Izmir_2020": dict(ad="Sisam–İzmir 2020", uzun="Sisam–İzmir – 30 Ekim 2020 (M 7,0)",
                             m=7.0, d=21.0, en=37.8973, bo=26.7838),
}

st.set_page_config(page_title="Artçı Deprem Büyüklüğü Tahmini", page_icon="🌍", layout="wide")
# Sayfa Türkçe; tarayıcının otomatik çevirisi kapatılır (çeviri, güncellenen sayıları donduruyordu).
components.html("""<script>
const h = window.parent.document.documentElement;
h.lang = "tr"; h.setAttribute("translate", "no"); h.classList.add("notranslate");
if (!window.parent.document.querySelector('meta[name="google"]')) {
  const m = window.parent.document.createElement("meta");
  m.name = "google"; m.content = "notranslate";
  window.parent.document.head.appendChild(m);
}
</script>""", height=0)
st.markdown(f"""<style>
.sonuc-etiket {{font-size:0.9rem; color:{METIN_ORTA}; margin:0}}
.sonuc-sayi {{font-size:3.2rem; font-weight:600; color:{RENK_VURGU}; line-height:1.1; margin:0.2rem 0 0.8rem}}
.cubuk {{position:relative; height:10px; background:{TABAN_CUBUK}; border-radius:5px; margin-top:0.4rem}}
.cubuk-aralik {{position:absolute; height:10px; background:{ANA_RENK}; border-radius:5px}}
.cubuk-nokta {{position:absolute; top:-4px; width:4px; height:18px; background:{RENK_VURGU}; border-radius:2px}}
.cubuk-yazi {{display:flex; justify-content:space-between; font-size:0.75rem; color:{METIN_SOLUK}; margin-top:0.3rem}}
.kutu {{background:{RENK_ACIK}; color:{METIN_ACIK}; border-radius:8px; padding:0.6rem 0.9rem; font-size:0.9rem; margin-top:1rem}}
.kutu div {{display:flex; justify-content:space-between; gap:0.6rem; padding:0.15rem 0}}
.kutu b {{white-space:nowrap}}
.not {{font-size:0.78rem; color:{METIN_SOLUK}; margin-top:0.8rem}}
.kart-etiket {{font-size:0.85rem; color:{METIN_ORTA}; margin:0}}
.kart-deger {{font-size:1.6rem; font-weight:600; color:{RENK_VURGU}; margin:0.1rem 0}}
.kart-alt {{font-size:0.8rem; color:{METIN_SOLUK}; margin:0}}
.dis-bant {{display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;
            border-left:5px solid {RENK_VURGU}; padding:0.2rem 0 0.2rem 0.8rem; margin-bottom:0.6rem}}
.dis-bant h4 {{margin:0; padding:0; font-size:1.15rem}}
.rozet {{background:{RENK_ACIK}; color:{RENK_VURGU}; font-size:0.8rem; padding:0.25rem 0.7rem; border-radius:6px}}
.faktor {{background:{RENK_ACIK}; color:{METIN_ACIK}; border-radius:10px; padding:0.7rem 0.5rem 0.3rem; text-align:center}}
.faktor .ad {{font-size:0.85rem; color:{METIN_ORTA}; margin:0}}
.faktor .simge {{line-height:1.15; margin:0.3rem 0; color:{RENK_VURGU}}}
.faktor .simge svg {{display:block; margin:0 auto}}
.faktor .deger {{font-size:1.05rem; font-weight:600; margin:0}}
.faktor .olcek {{font-size:0.72rem; color:{METIN_ORTA}; margin:0.1rem 0 0}}
.karsi {{border-radius:8px; padding:0.6rem 0.9rem; background:{RENK_ACIK}; color:{METIN_ACIK}}}
.karsi.vurgu {{background:{RENK_ACIK2}; color:{RENK_VURGU}}}
.karsi p {{margin:0}}
.karsi .b {{font-size:1.5rem; font-weight:600}}
.kucuk {{font-size:0.85rem; color:{METIN_ORTA}; margin:0}}
.bolum-baslik {{font-weight:600; margin:0 0 0.2rem}}
.ayrac {{margin:0.8rem 0; border:none; border-top:1px solid {CIZGI}}}
h5 {{margin-top:0.6rem !important}}
.block-container {{padding-top:2.2rem !important}}
h1 {{font-size:1.9rem !important; padding-bottom:0.2rem !important}}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{display:none}}
.ikili {{display:flex; flex-wrap:wrap; background:{ANA_RENK}; border-radius:10px; padding:0.9rem 1.1rem}}
.ikili > div {{flex:1}}
.ikili > div:last-child {{border-left:1px solid rgba(255,255,255,0.25); margin-left:1rem; padding-left:1rem}}
.ikili .sonuc-etiket {{color:#AFC8E0; font-size:0.78rem; text-transform:uppercase; letter-spacing:0.03em}}
.ikili-sayi {{font-size:2.3rem; font-weight:600; line-height:1.1; margin:0.15rem 0 0; color:#FFFFFF}}
.ikili-sayi.kisa {{font-size:1.15rem; margin-top:0.6rem}}
.ikili > div.ikili-not {{flex:0 0 100% !important; width:100%; margin-top:0.7rem; padding-top:0.6rem;
             border-top:1px solid rgba(255,255,255,0.2); font-size:0.82rem; color:#D7E6F5}}
.ikili-sayi.vurgu {{color:#FFFFFF}}
.ust-baslik {{display:flex; align-items:center; gap:0.7rem; padding-bottom:0.9rem;
              border-bottom:1px solid {CIZGI}; margin-bottom:0.3rem}}
.ust-rozet {{width:2.4rem; height:2.4rem; border-radius:10px; background:{ANA_RENK};
             display:flex; align-items:center; justify-content:center; font-size:1.2rem; flex:none}}
.ust-baslik h1 {{font-size:1.3rem !important; font-weight:600 !important; margin:0 !important;
                  padding:0 !important; line-height:1.3; color:{METIN_ACIK}}}
.ust-alt {{font-size:0.76rem; color:{METIN_SOLUK}; margin:0.1rem 0 0}}
.sec {{font-size:0.95rem; font-weight:600; color:{METIN_ACIK}; margin:0.9rem 0 0.6rem}}
.sec .no {{color:{RENK_VURGU}}}
table.bilgi {{width:100%; border-collapse:collapse; font-size:0.9rem; margin:0.2rem 0 0.8rem; border:none !important}}
table.bilgi td {{padding:0.35rem 0.2rem; border:none !important; border-bottom:1px solid {CIZGI} !important; vertical-align:top}}
table.bilgi td:first-child {{color:{METIN_ORTA}; white-space:nowrap; width:1%; padding-right:1rem}}
table.bilgi tr.vurgu td {{color:{RENK_VURGU}; font-weight:600}}
table.bilgi tr:last-child td {{border-bottom:none !important}}
.vaka {{border:1px solid {CIZGI}; border-radius:12px; padding:0.7rem 0.9rem; margin-bottom:0.8rem}}
.vaka.secili {{border:2px solid {RENK_VURGU}}}
.vaka-yatay {{display:grid; grid-template-columns:1fr 1.3fr 1fr; gap:1.5rem; align-items:center}}
.vaka-ad {{font-weight:600; margin:0}}
.vaka-satir {{display:flex; justify-content:space-between; max-width:15rem; font-size:0.95rem; padding:0.1rem 0}}
.vaka-hukum {{font-weight:600; font-size:0.9rem; margin:0.4rem 0 0}}
.renkli {{color:{RENK_VURGU}}}
.ozet-kutu {{background:{RENK_ACIK}; color:{METIN_ACIK}; border-radius:10px; padding:0.7rem 1rem; margin:0.4rem 0 0.8rem}}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def modelleri_yukle(degisim):           # degisim: dosya tarihi; model dosyası değişince önbellek yenilenir
    return {h: joblib.load(MODEL / f"en_iyi_model_{h}.joblib") for h in "AB"}


@st.cache_data
def veriyi_yukle(degisim):              # degisim: dosya tarihi; veri dosyası değişince önbellek yenilenir
    df = pd.read_csv(VERI, low_memory=False)
    df["artci_zaman"] = pd.to_datetime(df["artci_zaman"], utc=True, format="ISO8601")
    egitim_ort = df.loc[df.turkiye_vakasi.isna(), "artci_buyukluk"].mean()
    return df[df.turkiye_vakasi.notna()].copy(), float(egitim_ort)


def tahmin(paket, X):
    return paket["model"].predict(X[paket["girdiler"]])


def sayi(v, n=2):
    return f"{v:.{n}f}".replace(".", ",")


def sure_sonra(saat):
    """Süreyi okunur hale getirir: 24 saate kadar saat, sonrası gün olarak gösterilir."""
    return f"{sayi(saat, 1)} saat sonra" if saat < 24 else f"{sayi(saat / 24, 1)} gün sonra"


def ay_evresi(zaman):
    return ((zaman - REF_YENI_AY).total_seconds() / 86400 / SINODIK_AY) % 1.0


_SVG = ('<svg viewBox="0 0 24 24" width="30" height="30" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{}</svg>')
ICON_AY = _SVG.format('<path d="M21 12.8A9 9 0 1111.2 3 7 7 0 0021 12.8z"/>')
ICON_TELESKOP = _SVG.format('<path d="M4 15l13-7 2 4-13 7z"/><circle cx="6" cy="14" r="2.3"/>'
                            '<path d="M8.5 12.7l2 3.8M17 8.5l1.5 3"/>')
ICON_GELGIT = _SVG.format('<path d="M2.5 15c1.5-2.2 3-2.2 4.5 0s3 2.2 4.5 0 3-2.2 4.5 0 3 2.2 4.5 0"/>'
                          '<path d="M2.5 10c1.5-2.2 3-2.2 4.5 0s3 2.2 4.5 0 3-2.2 4.5 0 3 2.2 4.5 0" opacity="0.5"/>')
ICON_SICAKLIK = _SVG.format('<path d="M11 14.8V5a1.5 1.5 0 013 0v9.8a4 4 0 11-3 0z"/><circle cx="12.5" cy="16.5" r="1.2" fill="currentColor"/>')
ICON_BASINC = _SVG.format('<circle cx="12" cy="12" r="8.5"/><path d="M12 12l3.5-3.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>')
ICON_YAGMUR = _SVG.format('<path d="M12 3c3 4 6 7.5 6 11a6 6 0 01-12 0c0-3.5 3-7 6-11z"/>')
ICON_GUNES = _SVG.format('<circle cx="12" cy="12" r="4.2"/>'
                         '<path d="M12 2.5v2.3M12 19.2v2.3M4.4 12H2M22 12h-2.4M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6'
                         'M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6"/>')


def ay_adi(e, simge=True):
    """Ay evresi (0–1) → simge ve adı. 0 = yeni ay, 0,5 = dolunay."""
    for sinir, s, ad in [(0.03, "🌑", "Yeni ay"), (0.22, "🌒", "Hilal (büyüyen)"), (0.28, "🌓", "İlk dördün"),
                         (0.47, "🌔", "Şişkin ay (büyüyen)"), (0.53, "🌕", "Dolunay"),
                         (0.72, "🌖", "Şişkin ay (küçülen)"), (0.78, "🌗", "Son dördün"),
                         (0.97, "🌘", "Hilal (küçülen)"), (1.01, "🌑", "Yeni ay")]:
        if e < sinir:
            return (s, ad) if simge else ad


def kart(kol, etiket, deger, alt_yazi):
    with kol.container(border=True):
        st.markdown(f"<p class='kart-etiket'>{etiket}</p><p class='kart-deger'>{deger}</p>"
                    f"<p class='kart-alt'>{alt_yazi}</p>", unsafe_allow_html=True)


def adim(no, baslik, rozet=None):
    """Bölüm başlığını numaralı olarak gösterir (emoji aynen korunur, sadece 1/2/3 hafif vurgulu)."""
    r = f" <span class='rozet'>{rozet}</span>" if rozet else ""
    st.markdown(f"<p class='sec'><span class='no'>{no}</span>&ensp;{baslik}{r}</p>", unsafe_allow_html=True)


def faktor_kart(kol, ad, simge, deger, olcek=""):
    alt = f"<p class='olcek'>{olcek or '&nbsp;'}</p>"   # boş olsa da satır kalır → kartlar eşit boyda
    kol.markdown(f"<div class='faktor'><p class='ad'>{ad}</p><p class='simge'>{simge}</p>"
                 f"<p class='deger'>{deger}</p>{alt}</div>", unsafe_allow_html=True)


def ay_uzak_olcek(km):
    """Ay uzaklığının yorumu. Ay, Dünya'ya 356 bin ile 407 bin km arasında yaklaşıp uzaklaşır."""
    durum = "yakın" if km < 370000 else "uzak" if km > 395000 else "ortalama"
    return f"{durum} (356–407 bin km)"


def gelgit_olcek(g):
    """Gelgit kuvvetinin yorumu. Verimizde −95 ile +185 µGal arasında."""
    durum = "zayıf" if g < 0 else "güçlü" if g > 80 else "orta"
    return f"{durum} (−95 … +185)"


modeller = modelleri_yukle(max((MODEL / f"en_iyi_model_{h}.joblib").stat().st_mtime for h in "AB"))
turkiye, egitim_ort = veriyi_yukle(VERI.stat().st_mtime)
sonuc = pd.read_csv(TABLO / "model_sonuclari.csv")
onem = pd.read_csv(TABLO / "oznitelik_onemi.csv")
EN_IYI = modeller["B"]["model_adi"]                              # demoda kullanılan model (A ve B aynı)
AYAR = "Optuna ile ayarlandı" if modeller["B"].get("ayar") == "Optuna" else "sabit ayarlar"
satir_A = sonuc[(sonuc.yaklasim.str.startswith("A")) & (sonuc.model == EN_IYI)].iloc[0]
satir_B = sonuc[(sonuc.yaklasim.str.startswith("B")) & (sonuc.model == EN_IYI)].iloc[0]
ml_B = sonuc[sonuc.yaklasim.str.startswith("B") & sonuc.model.isin(["Random Forest", "XGBoost", "LightGBM"])] \
    .sort_values("MAE").reset_index(drop=True)                   # B'deki makine öğrenmesi modelleri, iyiden kötüye
MAE = float(satir_B.MAE)          # ana tahmin B modelinden (Ay, gelgit ve hava dahil)

DEPREMLER = list(TR) + ["kendi"]
ZAMAN_AN = {"İlk 1 saat": "30. dakika", "İlk gün": "12. saat", "İlk hafta": "84. saat (3,5. gün)",
            "İlk ay": "10. gün"}                                                    # modele giden anın okunur hali
ZAMAN = {"İlk 1 saat": (0.5, 0.0, 1.0), "İlk gün": (12.0, 1.0, 24.0),
         "İlk hafta": (84.0, 24.0, 168.0), "İlk ay": (240.0, 168.0, 720.01)}          # (modele giden saat, aralık)
# Not: "İlk ay" için temsili an 10. gün (240 saat); Sisam–İzmir'in saatlik hava verisi 12. güne kadar var.
UZAK = {"Yakın (<10 km)": (5.0, 0.0, 10.0), "Orta (10–50 km)": (30.0, 10.0, 50.0),
        "Uzak (>50 km)": (75.0, 50.0, 1e9)}                                            # (modele giden km, aralık)
AY = {"🌑 Yeni ay": 0.0, "🌓 İlk dördün": 0.25, "🌕 Dolunay": 0.5, "🌗 Son dördün": 0.75}
AY_UZAK = {"Yakın – süper ay (357.000 km)": 357000.0, "Ortalama (384.400 km)": 384400.0,
           "Uzak (406.000 km)": 406000.0}
GELGIT = {"Düşük (−60 µGal)": -60.0, "Orta (10 µGal)": 10.0, "Yüksek (+120 µGal)": 120.0}
HAVA = {"☀️ Açık": (15.0, 1015.0, 0.0), "🌧️ Yağışlı": (12.0, 1005.0, 5.0),
        "❄️ Soğuk": (-5.0, 1025.0, 0.0), "🔥 Sıcak": (35.0, 1008.0, 0.0)}             # (°C, hPa, mm)


def secim(etiket, secenekler, anahtar, varsayilan, fmt=None):
    """Tıklanır seçenek grubu. Seçili düğmeye tekrar basılırsa seçim kalkmaz, önceki seçim geri gelir."""
    onceki = f"_{anahtar}_onceki"
    st.session_state.setdefault(onceki, varsayilan)
    if anahtar in st.session_state and st.session_state[anahtar] is None:
        st.session_state[anahtar] = st.session_state[onceki]
    v = st.pills(etiket, secenekler, key=anahtar, format_func=fmt or (lambda x: x),
                 default=None if anahtar in st.session_state else varsayilan)
    v = v if v is not None else st.session_state[onceki]
    st.session_state[onceki] = v
    return v


@st.cache_data
def gercek_hava(dizi_id, zaman):
    """İndirdiğimiz saatlik Open-Meteo / ERA5 verisinden o saatin gerçek sıcaklık, basınç ve yağışı."""
    h = pd.read_csv(KOK / "veri" / "ham" / "hava" / f"{dizi_id}.csv", parse_dates=["saat"])
    r = h.iloc[(h.saat - zaman.floor("h")).abs().argmin()]
    return float(r.sicaklik), float(r.basinc), float(r.yagis)


def hava_etiketi(sicaklik, yagis):
    if yagis > 0:
        return "🌧️ Yağışlı"
    if sicaklik < 5:
        return "❄️ Soğuk"
    if sicaklik > 30:
        return "🔥 Sıcak"
    return "☀️ Açık"


st.markdown("""<div class="ust-baslik">
  <div class="ust-rozet">🌍</div>
  <div>
    <h1>Artçı Deprem Büyüklüğünün Makine Öğrenmesi ile Tahmini</h1>
  </div>
</div>""", unsafe_allow_html=True)

sekme1, sekme2, sekme3 = st.tabs(["Tahmin", "Türkiye örnekleri", "Sonuçlar"])


def bilgi_tablosu(satirlar):
    """[(etiket, değer, vurgu_mu)] → sade HTML tablo."""
    h = "".join(f"<tr class='{'vurgu' if v else ''}'><td>{e}</td><td>{d}</td></tr>" for e, d, v in satirlar)
    return f"<table class='bilgi'>{h}</table>"


# ================================================================== 1) Tahmin
with sekme1:
    sol, sag = st.columns([1.35, 1], gap="large")
    with sol:
        with st.container(border=True):
            adim(1, "Deprem ve an seç")
            dep = secim("Deprem", DEPREMLER, "sec_deprem", DEPREMLER[0],
                        lambda k: f"{TR[k]['ad']} · M {sayi(TR[k]['m'], 1)}" if k in TR else "Kendi değerlerim")
            kendi = dep == "kendi"
            if kendi:
                ana_m = st.slider("Ana deprem büyüklüğü", 6.0, 9.1, 7.0, 0.1, key="k_ana_m")
            z1, z2 = st.columns([1.15, 1])
            with z1:
                zam = secim("Ana depremden sonra", list(ZAMAN), "sec_zaman", "İlk 1 saat")
            with z2:
                uz = secim("Ana depreme uzaklık", list(UZAK), "sec_uzak", "Yakın (<10 km)")
            saat, z_alt, z_ust = ZAMAN[zam]
            km, u_alt, u_ust = UZAK[uz]

            adim(2, "🌙 O andaki dış koşullar", "projenin özgün kısmı")
            if kendi:
                # Gerçek veri yok: koşulları kullanıcı seçer
                st.caption("Kendi değerlerinde gerçek veri yok; koşulları sen seç.")
                ay = secim("Ay evresi", list(AY), "sec_ay", "🌕 Dolunay")
                ayu = secim("Ay'ın uzaklığı", list(AY_UZAK), "sec_ayu", "Ortalama (384.400 km)")
                gl = secim("Gelgit kuvveti", list(GELGIT), "sec_gelgit", "Orta (10 µGal)")
                hv = secim("Hava", list(HAVA), "sec_hava", "☀️ Açık")
                ay_deger, ay_km, gel = AY[ay], AY_UZAK[ayu], GELGIT[gl]
                sic, bas, yag = HAVA[hv]
            else:
                # Türkiye depremi: koşullar GERÇEK VERİDEN gelir (Ay astronomik hesap, hava Open-Meteo / ERA5)
                t = TR[dep]
                seq = turkiye[turkiye.turkiye_vakasi == dep]
                ilk_s = seq.sort_values("zaman_farki_saat").iloc[0]
                ana_zaman = ilk_s.artci_zaman - pd.Timedelta(hours=float(ilk_s.zaman_farki_saat))
                an = ana_zaman + pd.Timedelta(hours=saat)
                ay_deger = ay_evresi(an)
                sic, bas, yag = gercek_hava(str(seq.dizi_id.iloc[0]), an)
                ay_km, gel = gelgit(an, t["en"], t["bo"])
                ay_s, ay_a = ay_adi(ay_deger)
                c = st.columns(3)
                faktor_kart(c[0], "Ay evresi", ICON_AY, ay_a)
                faktor_kart(c[1], "Ay'ın uzaklığı", ICON_TELESKOP, f"{ay_km:,.0f} km".replace(",", "."), ay_uzak_olcek(ay_km))
                faktor_kart(c[2], "Gelgit kuvveti", ICON_GELGIT, f"{sayi(gel, 0)} µGal", gelgit_olcek(gel))
                c = st.columns(3)
                faktor_kart(c[0], "Sıcaklık", ICON_SICAKLIK, f"{sayi(sic, 1)} °C")
                faktor_kart(c[1], "Basınç", ICON_BASINC, f"{sayi(bas, 0)} hPa")
                faktor_kart(c[2], "Yağış", ICON_YAGMUR if yag > 0 else ICON_GUNES, f"{sayi(yag, 1)} mm")
                st.caption(f"Gerçek veri · {ZAMAN_AN[zam]} ({an.strftime('%d.%m.%Y %H:%M')} UTC)")

                once = seq[seq.zaman_farki_saat < saat]          # sadece seçilen andan ÖNCEKİ gerçek artçılar
                n = len(once)
                girdi = dict(ana_buyukluk=t["m"], ana_derinlik=t["d"], ana_enlem=t["en"], ana_boylam=t["bo"],
                             artci_derinlik=t["d"], onceki_sayi=n,
                             onceki_ort=once.artci_buyukluk.mean() if n else np.nan,
                             onceki_max=once.artci_buyukluk.max() if n else np.nan,
                             onceki_artcidan_saat=saat - once.zaman_farki_saat.max() if n else np.nan)

            if kendi:
                with st.expander("Kendi değerlerin: derinlik, konum ve önceki artçılar", expanded=True):
                    e1, e2 = st.columns(2)
                    a_d = e1.number_input("Ana deprem derinliği (km)", 0.0, 700.0, 10.0, 1.0, key="k_ana_d")
                    r_d = e2.number_input("Artçı derinliği (km)", 0.0, 700.0, 10.0, 1.0, key="k_art_d")
                    a_en = e1.number_input("Enlem", -90.0, 90.0, 38.0, 0.1, key="k_en")
                    a_bo = e2.number_input("Boylam", -180.0, 180.0, 37.0, 0.1, key="k_bo")
                    n = e1.number_input("Önceki artçı sayısı", 0, 700, 3, 1, key="k_n")
                    yok = n == 0   # ilk artçı: önceki artçı bilgisi kullanılmaz
                    o_ort = e2.number_input("Önceki artçıların ortalaması", 4.5, 8.0, 4.9, 0.1, key="k_ort", disabled=yok)
                    o_max = e1.number_input("Önceki en büyük artçı", 4.5, 8.0, 5.2, 0.1, key="k_max", disabled=yok)
                    o_s = e2.number_input("Bir öncekinden geçen süre (saat)", 0.0, 720.0, 1.0, 0.5, key="k_os",
                                          disabled=yok)
                    if yok:
                        st.caption("Artçı sayısı 0: bu ilk artçı, önceki artçı bilgileri kullanılmıyor.")
                    elif o_max < o_ort:
                        st.warning("Önceki en büyük artçı, ortalamadan küçük olamaz.")
                girdi = dict(ana_buyukluk=ana_m, ana_derinlik=a_d, ana_enlem=a_en, ana_boylam=a_bo,
                             artci_derinlik=r_d, onceki_sayi=n,
                             onceki_ort=o_ort if n else np.nan, onceki_max=o_max if n else np.nan,
                             onceki_artcidan_saat=o_s if n else np.nan)
            girdi.update(zaman_farki_saat=saat, uzaklik_km=km, ay_evresi=ay_deger, ay_uzaklik_km=ay_km,
                         gelgit_ugal=gel, sicaklik=sic, basinc=bas, yagis=yag)

            # 3. Modelin kullandığı bilgiler – proje konusundaki girdiler sırasıyla
            adim(3, "Modelin kullandığı bilgiler")
            en_y = "K" if girdi["ana_enlem"] >= 0 else "G"
            bo_y = "D" if girdi["ana_boylam"] >= 0 else "B"
            nn = int(girdi["onceki_sayi"])
            st.markdown(bilgi_tablosu([
                ("Ana şok", f"M {sayi(girdi['ana_buyukluk'], 1)} · {sayi(girdi['ana_derinlik'], 0)} km derinlik · "
                            f"{sayi(abs(girdi['ana_enlem']), 1)}° {en_y}, {sayi(abs(girdi['ana_boylam']), 1)}° {bo_y}", False),
                ("Zaman farkı", f"ana depremden {ZAMAN_AN[zam]} sonra", False),
                ("Uzaklık", f"ana depreme {sayi(km, 0)} km", False),
                ("Önceki artçılar", f"{nn} tane · ortalama {sayi(girdi['onceki_ort'], 1)} · en büyüğü "
                                    f"{sayi(girdi['onceki_max'], 1)}" if nn else "henüz yok (bu ilk artçı)", False),
                ("Dış faktörler", "yukarıda seçtiğin Ay, gelgit ve hava durumu" if kendi else "yukarıdaki 6 kart (Ay, gelgit ve hava durumu)", True),
            ]), unsafe_allow_html=True)
            st.caption("İlk dört satır proje konusunda istenen girdiler; son satır projenin özgün katkısı.")

            with st.expander("Teknik ayrıntılar"):
                bos = lambda v, n=2: "–" if pd.isna(v) else sayi(v, n)
                st.dataframe(pd.DataFrame({
                    "Girdi": ["Ana deprem büyüklüğü", "Ana deprem derinliği (km)", "Enlem / boylam",
                              "Artçı derinliği (km)", "Ana depremden sonra geçen süre (saat)",
                              "Ana depreme uzaklık (km)", "Önceki artçı sayısı", "Önceki artçıların ortalaması",
                              "Önceki en büyük artçı", "Bir öncekinden geçen süre (saat)",
                              "Ay evresi (0–1)", "Ay'ın uzaklığı (km)", "Gelgit kuvveti (µGal)",
                              "Sıcaklık (°C)", "Basınç (hPa)", "Yağış (mm)"],
                    "Değer": [bos(girdi["ana_buyukluk"], 1), bos(girdi["ana_derinlik"], 0),
                              f"{bos(girdi['ana_enlem'])} / {bos(girdi['ana_boylam'])}",
                              bos(girdi["artci_derinlik"], 0), bos(saat, 1), bos(km, 0), str(nn),
                              bos(girdi["onceki_ort"]), bos(girdi["onceki_max"], 1),
                              bos(girdi["onceki_artcidan_saat"], 1), bos(ay_deger), bos(ay_km, 0), bos(gel, 1),
                              bos(sic, 1), bos(bas, 0), bos(yag, 1)]}), hide_index=True, use_container_width=True)
                st.caption(f"Model: {EN_IYI} ({AYAR}), Yaklaşım B (sismik + dış faktör). Seçilen aralık için temsilî an "
                           f"({ZAMAN_AN[zam]}) ve uzaklık ({sayi(km, 0)} km) kullanılır. Türkiye depremlerinde artçı "
                           "derinliği ana deprem derinliği alınır; önceki artçı bilgileri seçilen andan önceki gerçek "
                           "artçılardan hesaplanır. Ay evresi, Ay'ın uzaklığı ve gelgit kuvveti astronomik olarak "
                           "hesaplanır (PyEphem); sıcaklık, basınç (deniz seviyesi) ve yağış Open-Meteo / ERA5 "
                           "arşivinden, ana deprem konumu için alınır.")

    X = pd.DataFrame([girdi])
    pA, pB = float(tahmin(modeller["A"], X)[0]), float(tahmin(modeller["B"], X)[0])
    fark = pB - pA
    ZAMAN_KISA = {"İlk 1 saat": "0–1 saat", "İlk gün": "1–24 saat", "İlk hafta": "1–7 gün", "İlk ay": "7–30 gün"}
    UZAK_KISA = {"Yakın (<10 km)": "10 km'den yakın", "Orta (10–50 km)": "10–50 km",
                 "Uzak (>50 km)": "50 km'den uzak"}

    with sag:
        with st.container(border=True):
            gercek_etiket = "Gerçekte olan"
            gercek_sinif = ""   # "–" yerine kısa metin gösterilince küçük puntoya geçilir
            if kendi:
                gercek_deger, gercek_sinif = "Veri yok", " kisa"
                gercek_cumle = "Karşılaştırma için bir Türkiye depremi seç."
            else:
                g = seq[(seq.zaman_farki_saat >= z_alt) & (seq.zaman_farki_saat < z_ust)
                        & (seq.uzaklik_km >= u_alt) & (seq.uzaklik_km < u_ust)]
                nerede = f"{ZAMAN_KISA[zam]}, {UZAK_KISA[uz]}"
                if len(g):
                    ort = g.artci_buyukluk.mean()
                    sapma = pB - ort
                    yon = "yüksek" if sapma > 0 else "düşük"
                    if len(g) == 1:
                        gercek_deger = sayi(ort, 1)
                        gercek_cumle = (f"Seçilen aralıkta ({nerede}) <b>1 artçı</b> oldu. Model "
                                        f"<b>{sayi(abs(sapma))} {yon}</b> tahmin etti.")
                    else:
                        gercek_deger = sayi(ort)
                        gercek_etiket = "Gerçekte olan (ortalama)"
                        gercek_cumle = (f"Seçilen aralıkta ({nerede}) <b>{len(g)} artçı</b> oldu; gerçek değer onların "
                                        f"ortalaması (en büyüğü {sayi(g.artci_buyukluk.max(), 1)}). Model "
                                        f"<b>{sayi(abs(sapma))} {yon}</b> tahmin etti.")
                else:
                    gercek_deger, gercek_sinif = "Artçı olmadı", " kisa"
                    gercek_cumle = f"Seçilen aralıkta ({nerede}) karşılaştırma yok."
            etki = ("Yani etkisi çok küçük." if abs(fark) < 0.05
                    else f"Bu fark, modelin ortalama sapmasından (± {sayi(MAE)}) küçük.")
            ikili_not = f"<div class='ikili-not'>{gercek_cumle}</div>" if gercek_sinif else ""
            cumle_disarda = "" if gercek_sinif else f"<p style='margin:0.8rem 0 0.2rem'>{gercek_cumle}</p>"
            st.markdown(f"""
<div class="ikili">
  <div><p class="sonuc-etiket">Model tahmini</p><p class="ikili-sayi vurgu">{sayi(pB)}</p></div>
  <div><p class="sonuc-etiket">{gercek_etiket}</p><p class="ikili-sayi{gercek_sinif}">{gercek_deger}</p></div>
  {ikili_not}
</div>
{cumle_disarda}
<p class="kucuk">Modelin ortalama sapması ± {sayi(MAE)}.</p>
<hr class="ayrac">
<p style="margin:0">Dış faktörler olmasaydı tahmin <b>{sayi(pA)}</b> olurdu (fark {sayi(abs(fark))}). {etki}</p>
<p class="not">Araştırma demosudur; kesin tahmin veya erken uyarı değildir.</p>
""", unsafe_allow_html=True)


# ================================================================== 2) Türkiye örnekleri
def vaka_ozeti(v):
    """Bir Türkiye dizisinin artçıları + tahminler ve sapmalar."""
    d = turkiye[turkiye.turkiye_vakasi == v].sort_values("artci_zaman").copy()
    d["tahmin"] = tahmin(modeller["B"], d)
    d["tahmin_A"] = tahmin(modeller["A"], d)
    d["basit"] = d["onceki_ort"].fillna(egitim_ort)
    d["hata"] = d.tahmin - d.artci_buyukluk
    hB = d.hata.abs().mean()
    hA = (d.tahmin_A - d.artci_buyukluk).abs().mean()
    hb = (d.basit - d.artci_buyukluk).abs().mean()
    return d, hB, hA, hb


with sekme2:
    st.markdown("Model bu üç depremi **eğitimde hiç görmedi**. **Basit yöntem:** bir sonraki artçının, önceki "
                "artçıların ortalaması kadar olacağını varsaymak. **Sapma** küçük olan daha iyi.")
    vaka = secim("Deprem", list(TR), "sec_vaka", list(TR)[0], lambda k: TR[k]["ad"])
    ozet = {v: vaka_ozeti(v) for v in TR}

    # Sadece seçilen depremin özet kartı (yatay düzen)
    _d, _hB, _hA, _hb = ozet[vaka]
    iyi = (_hb - _hB) / _hb * 100
    hukum = ("= İkisi aynı derecede yanıldı" if abs(iyi) < 1 else
             f"✓ Modelimiz %{iyi:.0f} daha az yanıldı" if iyi > 0 else
             f"✗ Basit yöntem %{-iyi:.0f} daha az yanıldı")
    st.markdown(f"<div class='vaka secili vaka-yatay'>"
                f"<div><p class='vaka-ad'>{TR[vaka]['ad']}</p>"
                f"<p class='kucuk'>M {sayi(TR[vaka]['m'], 1)} · {len(_d)} artçı</p></div>"
                f"<div><p class='kucuk'>Tahminler gerçekten ortalama ne kadar saptı?</p>"
                f"<div class='vaka-satir'><span>Modelimiz</span><b class='renkli'>{sayi(_hB)}</b></div>"
                f"<div class='vaka-satir'><span>Basit yöntem</span><b>{sayi(_hb)}</b></div></div>"
                f"<div><p class='vaka-hukum' style='margin:0'>{hukum}</p></div></div>", unsafe_allow_html=True)

    t = TR[vaka]
    d, hB, hA, hb = ozet[vaka]
    en_buyuk = d.loc[d.artci_buyukluk.idxmax()]

    # Deprem anındaki dış koşullar
    ilk_satir = d.iloc[0]
    ana_zaman = ilk_satir.artci_zaman - pd.Timedelta(hours=float(ilk_satir.zaman_farki_saat))
    ay_s, ay_a = ay_adi(ay_evresi(ana_zaman))
    sic0, bas0, yag0 = gercek_hava(str(d.dizi_id.iloc[0]), ana_zaman)
    with st.container(border=True):
        st.markdown(f"<div class='dis-bant'><h4>🌙 {t['ad']}: deprem anındaki dış koşullar</h4>"
                    f"<span class='rozet'>{ana_zaman.strftime('%d.%m.%Y %H:%M')} UTC "
                    f"(TSİ {(ana_zaman + pd.Timedelta(hours=3)).strftime('%H:%M')})</span></div>",
                    unsafe_allow_html=True)
        ana_km, ana_gel = gelgit(ana_zaman, t["en"], t["bo"])
        c = st.columns(6, gap="small")
        faktor_kart(c[0], "Ay evresi", ICON_AY, ay_a)
        faktor_kart(c[1], "Ay'ın uzaklığı", ICON_TELESKOP, f"{ana_km:,.0f} km".replace(",", "."), ay_uzak_olcek(ana_km))
        faktor_kart(c[2], "Gelgit kuvveti", ICON_GELGIT, f"{sayi(ana_gel, 0)} µGal", gelgit_olcek(ana_gel))
        faktor_kart(c[3], "Sıcaklık", ICON_SICAKLIK, f"{sayi(sic0, 1)} °C")
        faktor_kart(c[4], "Basınç", ICON_BASINC, f"{sayi(bas0, 0)} hPa")
        faktor_kart(c[5], "Yağış", ICON_YAGMUR if yag0 > 0 else ICON_GUNES, f"{sayi(yag0, 1)} mm")
        st.caption("Ana depremin anındaki gerçek değerler. Her artçının kendi değerleri en alttaki tabloda.")

    h1, h2 = st.columns([1, 1.2], gap="large")
    with h1:
        st.markdown("**Harita:** artçıların yeri ve sapması")
        harita = go.Figure()
        harita.add_trace(go.Scattermapbox(lat=[t["en"]], lon=[t["bo"]], mode="markers", hoverinfo="skip",
                                          marker=dict(size=24, color="white")))      # beyaz halka: ana deprem belirgin olsun
        harita.add_trace(go.Scattermapbox(lat=[t["en"]], lon=[t["bo"]], mode="markers+text", name="Ana deprem",
                                          marker=dict(size=15, color=ANA_RENK),
                                          text=[f"Ana deprem M {sayi(t['m'], 1)}"],
                                          textposition="top center", hoverinfo="text"))
        harita.add_trace(go.Scattermapbox(
            lat=d.artci_enlem, lon=d.artci_boylam, mode="markers", name="Artçı",
            marker=dict(size=7 + (d.artci_buyukluk - 4.5) * 8, color=d.hata.abs(), cmin=0, cmax=0.8,
                        colorscale=[[0, "#639922"], [0.4, "#EF9F27"], [1, "#E24B4A"]],
                        colorbar=dict(title="Sapma", thickness=10)),
            text=[f"Gerçek {sayi(a)} · Tahmin {sayi(b)} · {sure_sonra(c)}"
                  for a, b, c in zip(d.artci_buyukluk, d.tahmin, d.zaman_farki_saat)],
            hoverinfo="text"))
        # Yakınlaşma: harita, ana deprem + bütün artçıları sığdıracak kadar yakından baksın
        enl = list(d.artci_enlem) + [t["en"]]
        boy = list(d.artci_boylam) + [t["bo"]]
        orta_en, orta_bo = (min(enl) + max(enl)) / 2, (min(boy) + max(boy)) / 2
        acik = max(max(enl) - min(enl), (max(boy) - min(boy)) * np.cos(np.radians(orta_en)), 0.3) * 1.4
        yakin = float(np.clip(np.log2(400 / acik), 5, 10))
        harita.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=orta_en, lon=orta_bo), zoom=yakin),
                             height=400, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                             paper_bgcolor="rgba(0,0,0,0)", font=dict(color=METIN_ORTA))
        st.plotly_chart(harita, use_container_width=True)
        st.caption("Lacivert nokta: ana deprem · Büyük nokta: büyük artçı · Renk: yeşil az, kırmızı çok sapma. "
                   "Harita için internet gerekir.")
    with h2:
        st.markdown("**Zaman içinde:** gerçek büyüklük ve tahmin")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.zaman_farki_saat, y=d.artci_buyukluk, mode="markers", name="Gerçek",
                                 marker=dict(size=9, color=GRI)))
        fig.add_trace(go.Scatter(x=d.zaman_farki_saat, y=d.tahmin, mode="markers", name="Model tahmini",
                                 marker=dict(size=9, symbol="x", color=ANA_RENK)))
        fig.add_trace(go.Scatter(x=d.zaman_farki_saat, y=d.basit, mode="lines", name="Basit yöntem",
                                 line=dict(color="#B4B2A9", dash="dot")))
        fig.add_annotation(x=np.log10(en_buyuk.zaman_farki_saat), y=en_buyuk.artci_buyukluk,
                           text=f"En büyük artçı M {sayi(en_buyuk.artci_buyukluk, 1)}<br>tahmin {sayi(en_buyuk.tahmin)}",
                           showarrow=True, arrowhead=0, ay=10, font=dict(size=11, color=METIN_ACIK),
                           arrowcolor=METIN_ORTA,
                           # nokta grafiğin sağ yarısındaysa yazı sola, değilse sağa yazılır (kenarda kesilmesin)
                           **(dict(ax=-60, xanchor="right", align="right")
                              if np.log10(en_buyuk.zaman_farki_saat) > np.log10(d.zaman_farki_saat).mean()
                              else dict(ax=60, xanchor="left", align="left")))
        fig.update_xaxes(type="log", tickvals=[0.1, 1, 10, 100, 720],
                         ticktext=["6 dk", "1 sa", "10 sa", "100 sa", "30 gün"])
        fig.update_layout(xaxis_title="Ana depremden sonra geçen süre (logaritmik ölçek)",
                          yaxis_title="Artçı büyüklüğü", height=400, margin=dict(t=10, l=10, r=10),
                          legend=dict(orientation="h", y=1.08),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=METIN_ORTA),
                          xaxis=dict(gridcolor=CIZGI, zerolinecolor=CIZGI, linecolor=CIZGI),
                          yaxis=dict(gridcolor=CIZGI, zerolinecolor=CIZGI, linecolor=CIZGI))
        st.plotly_chart(fig, use_container_width=True)

    # Ne görüyoruz? – iki cümle
    if hB > hA + 0.005:
        ab = (f"Ay, gelgit ve hava durumu bu depremde tahmini iyileştirmedi (onlarsız sapma {sayi(hA)}, "
              f"onlarla {sayi(hB)}).")
    elif hB < hA - 0.005:
        ab = (f"Ay, gelgit ve hava durumu bu depremde sapmayı çok az azalttı ({sayi(hA)} → {sayi(hB)}).")
    else:
        ab = "Ay, gelgit ve hava durumu bu depremde tahmini değiştirmedi."
    st.markdown(f"<div class='ozet-kutu'><b>Ne görüyoruz?</b> Model ortalama büyüklükteki artçıları yakalıyor, "
                f"büyükleri kaçırıyor: en büyük artçı {sayi(en_buyuk.artci_buyukluk, 1)} iken tahmin "
                f"{sayi(en_buyuk.tahmin)} oldu. {ab}</div>", unsafe_allow_html=True)

    with st.expander("Tüm artçıların tablosu (dış koşullar dahil)"):
        tablo = d[["artci_zaman", "zaman_farki_saat", "uzaklik_km", "artci_buyukluk", "tahmin", "hata",
                   "ay_evresi", "ay_uzaklik_km", "gelgit_ugal", "sicaklik", "basinc", "yagis"]].copy()
        tablo.insert(1, "tsi", (tablo.artci_zaman + pd.Timedelta(hours=3)).dt.strftime("%Y-%m-%d %H:%M"))  # Türkiye saati = UTC+3 (üç depremde de)
        tablo["artci_zaman"] = tablo.artci_zaman.dt.strftime("%Y-%m-%d %H:%M")
        tablo["ay_evresi"] = [" ".join(ay_adi(e)) for e in tablo.ay_evresi]
        tablo["ay_uzaklik_km"] = tablo.ay_uzaklik_km.round(0)
        tablo.columns = ["Zaman (UTC)", "Türkiye saati (TSİ)", "Geçen süre (saat)", "Uzaklık (km)", "Gerçek büyüklük", "Model tahmini",
                         "Sapma (tahmin − gerçek)", "Ay evresi", "Ay'ın uzaklığı (km)", "Gelgit (µGal)",
                         "Sıcaklık (°C)", "Basınç (hPa)", "Yağış (mm)"]
        st.dataframe(tablo.round(2), use_container_width=True, hide_index=True)

    with st.expander("Teknik ayrıntılar"):
        st.dataframe(pd.DataFrame({
            "Deprem": [TR[v]["ad"] for v in TR],
            "Artçı sayısı": [len(ozet[v][0]) for v in TR],
            "Sapma – Ay, gelgit ve hava ile (B)": [round(ozet[v][1], 3) for v in TR],
            "Sapma – Ay, gelgit ve hava olmadan (A)": [round(ozet[v][2], 3) for v in TR],
            "Sapma – basit yöntem": [round(ozet[v][3], 3) for v in TR]}), hide_index=True, use_container_width=True)
        st.caption(f"Model: {EN_IYI} ({AYAR}). Sapma = ortalama mutlak hata (MAE). Basit yöntem: önceki artçıların ortalaması "
                   "(ilk artçıda eğitim verisinin ortalaması). Her artçı, sadece o ana kadar bilinen bilgilerle "
                   "tahmin edildi. Ay değerleri astronomik hesap (PyEphem), hava Open-Meteo / ERA5.")

# ================================================================== 3) Sonuçlar
with sekme3:
    taban_mae = float(sonuc.loc[sonuc.model == "Önceki artçı ortalaması", "MAE"].iloc[0])
    iyilesme = float(satir_B["taban_gore_MAE_iyilesme_%"])
    r = st.columns(3)
    kart(r[0], "Model basit yöntemden iyi mi?", f"Evet, %{iyilesme:.0f}",
         f"sapma {sayi(taban_mae, 3)} → {sayi(MAE, 3)} · hedef %10 idi")
    m1, m2, m3 = ml_B.model                                    # m1 en iyi
    if ml_B.MAE[1] - ml_B.MAE[0] < ml_B.MAE_std[0]:          # ikinci, ölçüm oynamasından daha yakın → berabere
        EN_IYI_YAZI = f"{m1} ve {m2} berabere"
        kart(r[1], "En iyi model hangisi?", f"{m1} ve {m2}", f"berabere; {m3} biraz geride")
    else:
        EN_IYI_YAZI = f"en iyisi {m1}"
        kart(r[1], "En iyi model hangisi?", m1, f"{m2} ve {m3} biraz geride")
    kart(r[2], "Ay, gelgit ve hava durumu işe yaradı mı?", "Hayır",
         f"fark {sayi(abs(satir_A.MAE - satir_B.MAE), 4)}; ölçüm oynaması (± {sayi(satir_B.MAE_std, 3)}) bundan büyük")

    st.caption("5 katlı çapraz doğrulama · 40.264 artçı, 3.627 deprem dizisi · Türkiye depremleri hariç")
    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("**Modellerin ortalama sapması** (küçük olan daha iyi)")
        sira = ["Önceki artçı ortalaması", "Doğrusal regresyon", "Random Forest", "XGBoost", "LightGBM"]
        etiket = ["Basit yöntem", "Doğrusal<br>regresyon", "Random<br>Forest", "XGBoost", "LightGBM"]
        sA = sonuc[sonuc.yaklasim.str.startswith("A")].set_index("model")
        sB = sonuc[sonuc.yaklasim.str.startswith("B")].set_index("model")
        cub = go.Figure()
        cub.add_trace(go.Bar(name="Ay, gelgit ve hava durumu olmadan (A)", x=etiket, y=[sA.MAE[m] for m in sira],
                             marker_color="#B4B2A9",
                             error_y=dict(type="data", array=[sA.MAE_std[m] for m in sira], color="#FFFFFF"),
                             hovertemplate="%{x}<br>sapma %{y:.4f}<extra>A</extra>"))
        # basit yöntem dış faktör kullanmaz → B'de yok (boş bırakılır)
        cub.add_trace(go.Bar(name="Ay, gelgit ve hava durumu ile (B)", x=etiket,
                             y=[sB.MAE[m] if m in sB.index else None for m in sira], marker_color=ANA_RENK,
                             error_y=dict(type="data", array=[sB.MAE_std[m] if m in sB.index else 0 for m in sira],
                                          color="#FFFFFF"),
                             hovertemplate="%{x}<br>sapma %{y:.4f}<extra>B</extra>"))
        cub.update_layout(barmode="group", height=400, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(range=[0.24, 0.31], title="Ortalama sapma", tickformat=".2f",
                                     gridcolor="#3A4552", zeroline=False, linecolor=CIZGI),
                          xaxis=dict(showgrid=False, linecolor=CIZGI),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=METIN_ORTA),
                          legend=dict(orientation="h", y=1.1, x=0), separators=",.")
        st.plotly_chart(cub, use_container_width=True)
        st.markdown("<p class='kucuk'>Gri ve lacivert çubuklar neredeyse aynı boyda: Ay, gelgit ve hava durumunu eklemek "
                    "sapmayı değiştirmedi. Bütün makine öğrenmesi modelleri basit yöntemden daha az yanılıyor; "
                    f"{EN_IYI_YAZI}. Modellerin ayarları Optuna ile seçildi. Çubukların üstündeki beyaz çizgiler, ölçümün kendi oynamasını (±) gösterir. "
                    "Farklar görünsün diye dikey eksen 0,24'ten başlıyor.</p>", unsafe_allow_html=True)
    with g2:
        st.markdown("**Model neye bakıyor?** Girdilerin önemi")
        # 6 dış faktör tek tek görünürse çok küçük kalıp okunamıyor; tek "toplam" çubukta birleştirilir
        dis_toplam = onem[onem.grup == "Dış faktör"]["onem_%"].sum()
        sismik = onem[onem.grup == "Sismik"][["aciklama", "onem_%", "MAE_artisi_std", "sira"]]
        birlesik = pd.concat([sismik, pd.DataFrame([{"aciklama": "Dış faktörler (6 tanesi birlikte)",
                                                      "onem_%": dis_toplam, "MAE_artisi_std": 0,
                                                      "sira": onem[onem.grup == "Dış faktör"].sira.min()}])])
        o = birlesik.sort_values("onem_%")
        dis = o.aciklama == "Dış faktörler (6 tanesi birlikte)"
        olcek = float(onem["onem_%"].iloc[0] / onem["MAE_artisi"].iloc[0])  # MAE artışı → önem payı (%)
        onm = go.Figure(go.Bar(
            x=o["onem_%"], y=o.aciklama, orientation="h",
            marker_color=[ANA_RENK if x else "#B4B2A9" for x in dis],
            error_x=dict(type="data", array=o.MAE_artisi_std * olcek, color="#FFFFFF"),
            hovertemplate="%{y}<br>önem payı %{x:.1f}%<extra></extra>"))
        onm.add_trace(go.Bar(x=[None], y=[None], name="Sismik girdi", marker_color="#B4B2A9"))
        onm.add_trace(go.Bar(x=[None], y=[None], name="Dış faktör (Ay, gelgit ve hava)", marker_color=ANA_RENK))
        onm.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                          legend=dict(orientation="h", y=1.1, x=0), separators=",.",
                          xaxis=dict(title="Önem payı (%)", gridcolor="#3A4552", linecolor=CIZGI),
                          yaxis=dict(tickfont=dict(size=11), showgrid=False, linecolor=CIZGI),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=METIN_ORTA))
        onm.data[0].showlegend = False
        st.plotly_chart(onm, use_container_width=True)
        dis_sira = onem[onem.grup == "Dış faktör"].sira.min()
        st.markdown(f"<p class='kucuk'>Model en çok önceki artçıların ortalamasına, artçının derinliğine ve ana "
                    f"depremin büyüklüğüne bakıyor. 6 dış faktörün hepsi birlikte bile toplam sismik girdilerin "
                    f"çok gerisinde kalıyor (toplam %{sayi(dis_toplam, 1)}); tek tek en önemlisi {dis_sira}. sırada "
                    f"({len(onem)} girdi içinde). Önem şöyle ölçüldü: bir girdi karıştırılınca modelin sapması ne "
                    f"kadar artıyor. Çubukların ucundaki beyaz çizgiler, bu ölçümün kendi oynamasını (±) gösterir; "
                    f"çizgi ne kadar uzunsa o girdinin öneminden o kadar az eminiz.</p>", unsafe_allow_html=True)
        with st.expander("6 dış faktörün kendi aralarındaki sırası"):
            dtab = onem[onem.grup == "Dış faktör"][["sira", "aciklama", "onem_%"]].reset_index(drop=True)
            dtab.columns = ["Sıra (17 içinde)", "Dış faktör", "Önem payı (%)"]
            st.dataframe(dtab, hide_index=True, use_container_width=True)

    with st.container(border=True):
        st.markdown(f"<p class='bolum-baslik'>Sınırlar ve finale kalanlar</p><p class='kucuk'>Model büyük "
                    f"artçıları kaçırıyor; tahminleri ortalamaya yakın kalıyor (R² ≈ {sayi(satir_B.R2)}). "
                    "Optuna ile ayar yapıldı ama sapma sadece biraz azaldı; verinin kendisi bir tavan koyuyor. "
                    "Finalde: SHAP açıklamaları, güven aralığı, Türkiye vakalarının ayrıntılı analizi.</p>",
                    unsafe_allow_html=True)

    with st.expander("Teknik ayrıntılar"):
        g = sonuc[["yaklasim", "model", "MAE", "MAE_std", "RMSE", "R2", "taban_gore_MAE_iyilesme_%"]].copy()
        g.columns = ["Yaklaşım", "Model", "MAE", "MAE ±", "RMSE", "R²", "Basit yönteme göre iyileşme (%)"]
        st.dataframe(g, use_container_width=True, hide_index=True)
        st.caption("Yaklaşım A: sadece sismik girdiler · Yaklaşım B: sismik + dış faktörler (Ay evresi, Ay'ın "
                   "uzaklığı, gelgit kuvveti, sıcaklık, basınç, yağış). 5 katlı, dizi bazlı çapraz doğrulama: bir "
                   "dizinin bütün artçıları aynı parçada. A ile B arasındaki fark, katlar arası oynamadan (MAE ±) "
                   "küçük; yani dış faktörler belirgin bir iyileşme sağlamadı. Random Forest, XGBoost ve LightGBM "
                   "ayarları Optuna ile iç içe doğrulamayla seçildi: her dış katta ayar, sadece o katın eğitim "
                   "kısmında 3 katlı iç bölmeyle arandı; test kısmı ayar seçimine karışmadı.")
        dis = onem[onem.grup == "Dış faktör"][["sira", "aciklama", "onem_%"]]
        dis.columns = ["Sıra", "Dış faktör", "Önem payı (%)"]
        st.dataframe(dis, use_container_width=True, hide_index=True)
