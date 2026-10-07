# Thanos son sürüm × TVSum: başarılı 5 videonun karşılaştırması

**Sürüm:** `a9770f3` (2026-10-07) · **Ayar:** `balanced`, `importance`, `local` · **Bütçe:** kaynak video süresinin en çok %15'i.

## Kısa sonuç — başarısızlar ortalamaya dahil değil

| MP4 üretilen aynı 5 TVSum videosu | Ortalama kare-düzeyi F1 | Yorum |
|---|---:|---|
| Eşit aralıklı, %15 süreli basit yöntem | **0,15545** | AI kullanmayan karşılaştırma çizgisi. |
| Thanos yalnız sahne puanları + %15 knapsack | **0,18864** | Seçim aşaması tanısı; gerçek MP4 sonucu değil. |
| **Thanos gerçek özet MP4** | **0,14359** | Asıl ürün sonucu. |

Buradaki **üç ortalama da birebir aynı 5 video** üzerinden hesaplandı; başarısız videoların hiçbirine 0 verilmedi. Thanos bu videolarda ortalama yalnızca **%11,5 kaynak süre** seçti. Basit yöntemin süresi Thanos'un fiilen kullandığı süreye eşitlenince F1'i **0,13113**; bu daha dar bütçede Thanos biraz daha iyi. Başarılı beş videoda insan referansının önemli karelerini yakalama oranı (recall) **%12,83**, seçilen karelerin referansla örtüşme oranı (precision) **%16,49**.

**Hüküm:** Başarılı videolarda gerçek Thanos özeti, tam %15 kullanan basit yöntemin **biraz altında** (0,14359 < 0,15545); yalnız sahne seçimi daha yüksek (0,18864) ama bu gerçek MP4 sonucu değil. Koşu 10 videoyla yapıldı, **5/10 çıktı**; kalan beşi bu kalite ortalamalarının dışında. F1 bir “doğruluk yüzdesi” veya özeti izleme kalitesinin tamamı değildir.

## Video bazında

| TVSum ID | Tür | %15 sınır | Çıkan özet | Basit F1 | Sahne seçimi F1 | Gerçek MP4 F1 |
|---|---|---:|---:|---:|---:|---:|
| `akI8YFjEmUw` | Araç kurtarma | 20,00 sn | 15,24 sn | 0,147 | 0,150 | **0,102** |
| `37rzWOQsNIw` | Sandviç yapımı | 28,74 sn | 18,94 sn | 0,151 | 0,162 | **0,081** |
| `GsAD1KT1xo8` | Parkur | 21,80 sn | 13,98 sn | 0,130 | 0,151 | **0,216** |
| `91IHQYk1IQM` | Geçit töreni | 16,58 sn | 14,05 sn | 0,165 | 0,202 | **0,151** |
| `_xMr-HKMfVA` | Flash mob | 22,34 sn | 20,72 sn | 0,183 | 0,278 | **0,168** |

Diğer beş deneme (`AwmHb44_ouw`, `0tmA_C6XwfM`, `EE-bNr36nyA`, `iVt07TCkFM0`, `kLxoNp-UchI`) `Hedef süreye cümleyi bölmeden sığan sahne yok; süreyi artırın` hatasıyla bitti. **Bu beş video yukarıdaki tablonun ve tüm kalite ortalamalarının dışında.** Yalnızca çıktı oranını dürüstçe göstermek için **5/10** bilgisini koruduk. Üretilen beş MP4'ün tamamında `ffprobe` ile **hem video hem ses akışı** doğrulandı. Donanım kodlayıcısı bu makinede açılmadığında CPU kodlayıcısı devreye girdi.

## Önceki koşuyla fark

Önceki [11 videoluk koşuda](TVSUM_11_VIDEO_RAPORU.md) **6 video** başarılıydı; şimdi bu 10 videoluk seçimde **5 video** başarılı. Önceden başarılı olan `AwmHb44_ouw` (F1 **0,23613**) artık cümle/süre kuralına takılıyor. **İki koşunun ortak başarılı beş videosunun F1 değerleri aynı**; dolayısıyla bu beşli üzerinde karşılaştırılan kalite değişmedi. Eski altılı ile yeni beşlinin ortalamalarını doğrudan kıyaslamak doğru olmaz. Bu rapor tek koşudan `AwmHb44_ouw` gerilemesini hangi kod/önbellek değişikliğinin tetiklediğini kesinleştirmez.

## Ölçüm ve sınırlar

- TVSum'un her video için 20 kişinin 1–5 önem puanı kullanıldı. Her kişi için 60 karelik parçalardan en çok %15 bütçeli ayrı referans özet oluşturuldu; Thanos'un değişken uzunluklu kaynak-zaman kesitleri aynı kare eksenine taşınarak F1, recall ve precision hesaplandı. Basit yöntem aynı videodan %15 süreyi eşit aralıklarla seçiyor. [Hesaplama kodu](scripts/evaluate_tvsum.py) ve [protokol açıklaması](TVSUM_DEGERLENDIRME.md).
- Koşudaki 10 video, önceki değerlendirmedeki 11 videodan **her TVSum türüne bir video** düşecek şekilde, önceki sonuca göre yeniden seçilmeden alındı. Kalite tablosunda yalnızca MP4 üretilebilen 5 video var; bu **başarılı koşullara göre filtrelenmiş**, iyimser bir sonuçtur. TVSum dosyalarının **50/50'si yerelde eşleşti**, ama koşuda yalnız 10'u denendi; bu tam TVSum benchmark sonucu veya filmler için genelleme değildir.
- Analiz önbelleği, geçerli aşamalar için yeniden kullanıldı; sürüm/ayar değişince geçersizleşen aşamalar güncel kodla yeniden hesaplandı. RAG/Paralon modu çalıştırılmadı. Sahne puanları testi, konuşma korumasını ve gerçek MP4 üretimini **içermez**.

Ham sonuçlar: [uçtan uca JSON](outputs/tvsum/latest_10_a9770f3/evaluation_end_to_end.json), [yalnız sahne puanları JSON](outputs/tvsum/latest_10_a9770f3/evaluation_score_only.json). JSON tüm 10 denemeyi saklar; **bu raporun ortalamaları yalnızca `status=ok` olan aynı 5 satırdan** hesaplandı. Üretilen MP4'ler aynı `outputs/tvsum/latest_10_a9770f3/` klasöründe. Eski MP4'ler ve eski raporlar üzerine yazılmadı.
