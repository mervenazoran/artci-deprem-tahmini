# Proje Tasarım Belgesi

**Proje:** Artçı Deprem Büyüklüğünün Makine Öğrenmesi ile Tahmini (MFK420)
**Ekip:** Merve Naz Oran (model) · Sude Anlaş (veri) · Hasan Erkeş (karşılaştırma, açıklama, arayüz)
**Danışman:** Dr. Öğr. Üyesi Turgut PURA
**Sürüm:** 1 (9 Kasım 2026) · Bu belge değişmeden kod yazılmaz; her değişiklik buraya işlenir.

---

## 1. Hocanın konu metni (dayanak)
> USGS (ya da online ulaşılabilen) açık deprem verileri kullanılarak ana deprem sonrasında meydana gelen artçı depremlerin büyüklüğü makine öğrenmesi yöntemleriyle tahmin edilecektir. Ana şok büyüklüğü, derinlik, zaman farkı, konum ve önceki artçı depremlere ait bilgiler giriş değişkenleri olarak kullanılacak; Random Forest, XGBoost ve LightGBM gibi algoritmaların performansları karşılaştırılacaktır. Amaç, artçı deprem davranışlarının modellenmesi ve deprem sonrası risk analizine katkı sağlamaktır.

**Hocanın ek istekleri (çizelge ve sözlü notlar):** sadece açık kaynak veri · 2026 Haziran'a kadar veri · "zaman farkı" = ana şoktan geçen süre · mekânsal uzaklık da girdi · özgünlük için dış faktör · en az 2 yöntem · literatür Excel'i en az 30 kaynak · vizede çalışan demo, eksikler ve çözüm planı · kesin tahmin veya erken uyarı iddiası yok.

---

## 2. Veri
| Karar | Değer |
|---|---|
| Kaynak | **USGS ComCat** (açık API, kayıt veya dilekçe gerekmez) |
| Bölge | **Dünya geneli** |
| Tarih aralığı | **1 Ocak 1980 – 30 Haziran 2026** |
| Ana deprem | **M ≥ 6,0** (artçıların 30 gün takibi için en geç 31 Mayıs 2026 tarihli olanlar) |
| Artçı eşiği | **M ≥ 4,5** |
| Türkiye test vakaları | Kahramanmaraş 2023, İzmit 1999, Sisam–İzmir 2020 → **eğitimden tamamen ayrı** |

## 3. Ana deprem ve artçı tanımı
- **Mekânsal pencere (Gardner–Knopoff, 1974):** L(M) = 10^(0,1238·M + 0,983) km. Örnek: M 6 → ~53 km, M 7 → ~71 km.
- **Zaman penceresi:** Ana depremden sonraki **30 gün.**
- **Artçı:** Ana depremden sonraki 30 gün içinde ve L(M) km içinde olan, M ≥ 4,5 olay.
- **Ana deprem:** M ≥ 6 olup kendisi daha büyük bir depremin artçısı olmayan olay (yani kendinden önceki 30 gün ve L(M) km içinde daha büyük bir deprem yok).
- **Çakışma kuralı:** Bir dizinin penceresine düşen deprem ayrı bir dizi başlatmaz; o dizinin artçısı sayılır (ör. Kahramanmaraş'taki M 7,5 Elbistan depremi, M 7,8'in artçısı).
- **En büyük ana deprem kuralı (9 Kasım 2026 ekip kararı):** Pencereye ana depremden **daha büyük** bir deprem düşerse dizinin ana depremi o olur; ondan önceki depremler öncü sayılıp veriden çıkarılır ve pencere yeni ana depreme göre yeniden başlar (ör. 2011 Japonya M 9,1 → 2 gün önceki M 7,3 öncü olur). Gardner–Knopoff'un tanımıyla uyumludur. Etkilenen dizi: 240; çıkarılan öncü olay: 2.029.

## 4. Veri tablosu (her satır bir artçı)
| Sütun | Tür | Açıklama |
|---|---|---|
| `dizi_id` | Kimlik | Ana deprem kimliği; doğrulama gruplaması için |
| **Hedef:** `artci_buyukluk` | Hedef | Artçının büyüklüğü |
| `ana_buyukluk`, `ana_derinlik`, `ana_enlem`, `ana_boylam` | Sismik | Ana şok bilgileri |
| `artci_derinlik` | Sismik | Artçının derinliği |
| `zaman_farki_saat` | Sismik | Ana şoktan geçen süre (hocanın "zaman farkı") |
| `uzaklik_km` | Sismik | Artçının ana şoka mekânsal uzaklığı |
| `onceki_sayi` | Sismik | O ana kadar gelen artçı sayısı |
| `onceki_ort`, `onceki_max` | Sismik | Önceki artçıların ortalama ve en büyük büyüklüğü (ilk artçıda boş) |
| `onceki_artcidan_saat` | Sismik | Bir önceki artçıdan geçen süre |
| `ay_evresi` | **Dış faktör** | Ay evresi (0 = yeni ay, 0,5 = dolunay); tarihten hesaplanır |
| `ay_uzaklik_km`, `gelgit_ugal` | **Dış faktör** | Ay'ın Dünya'ya uzaklığı (km) ve Ay + Güneş'in dikey gelgit ivmesi (µGal), artçı anında ana deprem konumunda; PyEphem ile astronomik hesap, internet gerekmez. Dayanak: Ide vd. (2016). (12 Kasım 2026 ekip kararı) |
| `sicaklik`, `basinc`, `yagis` | **Dış faktör** | Artçı saatindeki hava durumu (Open-Meteo / ERA5 açık arşivi, ana deprem konumu için). Basınç = deniz seviyesine indirgenmiş basınç (yükseklik etkisi olmasın diye; 9 Kasım ekip kararı) |

**Sızıntı kuralı:** Bir satırdaki her bilgi, o artçıdan **önce** bilinen bilgi olmalı (artçının kendi zamanı, yeri ve derinliği hariç; bunlar olay anında bilinir).

## 5. Modeller ve "2 yöntem"
**Karşılaştırma modelleri (basit):**
1. **Önceki artçıların ortalaması** (ilk artçıda eğitim verisinin ortalaması)
2. **Doğrusal regresyon** (hocanın çizelgesinde önerilen basit referans)

**Makine öğrenmesi modelleri:** Random Forest · XGBoost · LightGBM (SVR finalde isteğe bağlı)

**İki yaklaşım (hocanın "2 yöntem" ve özgünlük isteği):**
- **Yaklaşım A:** Sadece sismik girdiler
- **Yaklaşım B:** Sismik + dış faktörler (ay evresi, Ay'ın uzaklığı, gelgit kuvveti, sıcaklık, basınç, yağış)
- **Soru:** Dış faktörler tahmini iyileştiriyor mu?

## 6. Değerlendirme
- **5 katlı, dizi bazlı çapraz doğrulama** (bir dizinin bütün artçıları aynı parçada)
- **Metrikler:** MAE, RMSE, R² (katlar arası ortalama ± sapma)
- **Başarı kriteri 3:** En iyi model, basit modelden daha az hata (MAE); hedef %10
- **Hiperparametre ayarı (Optuna, iç içe doğrulama):** dış döngü aynı 5 katlı bölme; her dış katta ayar, sadece o katın eğitim kısmında 3 katlı iç bölmeyle aranır (LightGBM ve XGBoost 40, Random Forest 20 deneme; ilk deneme eski sabit ayarlar). Test kısmı ayar seçimine karışmaz. *(13 Kasım kararı: finalden vizeye alındı.)*
  - Sonuç (B): LightGBM 0,268 → 0,266 · XGBoost 0,271 → 0,266 · Random Forest 0,271 → 0,268. Küçük ama tutarlı iyileşme; basit yönteme göre %12.
  - Demo modeli: B'nin en iyisi (XGBoost; LightGBM ile berabere). A için de aynı model kaydedilir ki A–B farkı sadece dış faktörleri göstersin.
  - Eski sabit ayarlarla sonuç: `kod\merve_model\modelleme.py --sabit`
- **Türkiye vakaları:** en sonda "tahmin vs gerçek" tablosu (başarı kanıtı değil, örnek)

## 7. Vize (19 Kasım) ve final kapsamı
| Vizede hazır olacak | Finale kalan |
|---|---|
| Veri seti (USGS, 2026'ya kadar) + Optuna ile ayar (iç içe doğrulama) | Güven aralığı |
| Artçı ayırma ve tüm girdiler | SHAP açıklamaları |
| Dış faktörler | Türkiye vakaları analizi |
| Basit modeller + RF, XGBoost, LightGBM; A ve B karşılaştırması | (İsteğe bağlı) SVR |
| İlk sonuç tablosu ve öznitelik önem grafiği | Tez yazımı |
| Basit Streamlit demo | Arayüzün son hali |

**Demo arayüzü geçmişi (Hasan, `kod/hasan_arayuz/`):**
- 13 Kasım: 3 sekmeli (Tahmin / Türkiye örnekleri / Sonuçlar) sade düzene geçildi; önceki hali `yedek/arayuz_oncesi_2026-11-13/`.
- 14 Kasım: renk teması USGS ve kurumsal tasarımlardan esinlenerek mürdümden (#A6236F) laciverte (#1F5FA8) çevrildi; ardından arka plan koyu temaya alındı. Pastel pembe kart zeminleri koyu lacivert/gri tonlarla değiştirildi.
- 16 Kasım: dış koşul kartlarındaki emoji simgeler (🌙🔭🌊 vb.) yerine offline çalışan çizgi tarzı SVG ikonlar kullanıldı; "veri yok" / "artçı olmadı" gibi durumlarda tek bir tire (–) yerine kutunun içinde kısa açıklama gösterilecek şekilde düzeltildi; harita üzerindeki süreler 24 saati geçince gün cinsine çevriliyor.
- 16 Kasım: `kod/` kişiye göre alt klasörlere ayrıldı (`sude_veri/`, `merve_model/`, `hasan_arayuz/`) ve proje GitHub'a yüklenmeye hazırlandı (`.gitignore`, `README.md` güncellendi).
- 16 Kasım: Sonuçlar sekmesindeki iki grafikte hata çubukları (± ölçüm oynaması) gri zeminde seçilmiyordu; rengi beyaza çevrildi ve her iki grafiğin altına "beyaz çizgiler ölçümün kendi oynamasını gösterir" açıklaması eklendi.
- Eski sürümler `kod/hasan_arayuz/eski_surumler/` içinde tutuluyor (GitHub'a atılmaz).

## 8. Klasör ve kod yapısı
*(16 Kasım 2026: kod/ klasörü GitHub'a atılmadan önce kişiye göre alt klasörlere ayrıldı.)*
```
Artci_Deprem_Projesi/
├── Proje_Tasarimi.md             ← bu belge
├── veri/ham/                     ← USGS ve hava durumu ham verisi
├── veri/islenmis/                ← modelin kullandığı tablo
├── kod/
│   ├── sude_veri/
│   │   ├── veri_cekme.py         (Sude)         USGS'den indirme
│   │   ├── on_isleme.py          (Sude)         ana deprem/artçı ayırma
│   │   └── dis_faktorler.py      (Sude, Hasan)  ay evresi + hava durumu
│   ├── merve_model/
│   │   ├── modelleme.py          (Merve)        basit yöntem, RF, XGBoost, LightGBM, 5 katlı CV
│   │   ├── optuna_ayar.py        (Merve)        hiperparametre ayarı (iç içe doğrulama)
│   │   └── oznitelik_onemi.py    (Merve)        permütasyon önemi
│   └── hasan_arayuz/
│       ├── app.py                (Hasan)        Streamlit demo
│       └── eski_surumler/                       önceki sürümler (GitHub'a atılmaz)
├── sonuclar/                     ← tablolar ve grafikler
├── sunumlar/  literatur/  ekip/  ders_belgeleri/
```

## 9. Riskler
| Risk | Önlem |
|---|---|
| Bazı dizilerde artçı az | Veriye bak; gerekirse artçı eşiği M ≥ 4 (danışmanla) |
| Model basit modeli geçemez | Olumsuz sonuç da raporlanır; erken test |
| Hava durumu API'si yavaş veya sınırlı | Ana deprem konumu için tek sorgu; sonuçlar diske kaydedilir |
| Zaman (vize yüklemesi 16 Kasım) | Önce çalışan basit sürüm, sonra iyileştirme |
