# Thanos × TVSum: 11 videoluk karşılaştırma

## Net sonuç

| **Özet üretilebilen aynı 6 videonun ortalaması** | F1 puanı | Ne ölçüyor? |
|---|---:|---|
| **Basit yöntem (referans çizgisi)** | **0,155** | Videodan %15 süreyi eşit aralıklarla seçer; yapay zekâ kullanmaz. |
| **Bizim gerçek sistem — Thanos** | **0,159** | Analiz, sahne seçimi, konuşma koruması ve MP4 üretiminin tamamı. **Asıl ürün sonucu budur.** |
| **Bizim yalnız sahne seçimi** | **0,191** | Thanos'un önem puanlarını %15 bütçeyle seçer; konuşma korumasını ve gerçek MP4 üretimini ölçmez. Yalnızca sorunun nerede olduğunu anlamaya yarar. |

**Hüküm:** Başarılı 6 videoda gerçek Thanos özeti **0,159**, aynı 6 videodaki basit yöntem **0,155**: **hemen hemen aynı düzey**. Kullanıcının isteğiyle başarısız 5 video **bu ortalamalara katılmadı**; onlar için sonuç ayrıca **5/11 başarısız, 6/11 başarılı** olarak gösteriliyor. Bu koşullu ortalama, sistemin her videoda çalıştığı anlamına gelmez. Yalnız sahne seçimi 0,191; bu gerçek MP4 kalitesi değildir.

Üç ortalama da **aynı 6 video** üzerinden alındı. Ham JSON'da bütün 11 denemeyi gösteren ayrı ortalamalar da vardır; buradaki karşılaştırma tablosunda onlar kullanılmıyor.

**F1, “doğruluk yüzdesi” değildir.** 0 ile 1 arasında, seçilen video karelerinin insanların önem verdiği karelerle örtüşmesini ölçer; yüksek olması iyidir. Örneğin 0,159, “%15,9 doğru” demek değildir. Buradaki **0,155 basit yöntem puanı bizim aynı 6 videoda hesapladığımız karşılaştırma çizgisidir**, yayımlanmış bir TVSum liderlik tablosu skoru değildir.

### “TVSum'un önemli anlarının yüzde kaçını yakaladık?”

| Aynı başarılı 6 videoda | TVSum referansındaki önemli kareleri yakalama (recall) | Seçtiği karelerin referansa denk gelmesi (precision) |
|---|---:|---:|
| **Bizim gerçek Thanos özeti** | **%14,1** | **%18,4** |
| **Basit eşit aralıklı seçim** | **%15,6** | **%15,3** |

Yani Thanos, TVSum referansındaki önemli anların yaklaşık **%14'ünü seçiyor, %86'sını kaçırıyor**. Basit yöntem biraz daha çok önemli an yakalıyor; Thanos ise seçtiği karelerde biraz daha isabetli. Bunun bir nedeni Thanos'un %15 süre hakkının tamamını kullanmaması. “Yalnız sahne seçimi” tanı testinin aynı 6 videodaki yakalama oranı **%19,0**; gerçek MP4 için geçerli sayı **%14,1**. Bu yüzdeler tek bir “kesin doğru özet”e göre değil, her videodaki **20 kişinin puanlarından ayrı ayrı türetilen referans özetlerin ortalamasına** göre hesaplandı.

## Video video sonuçlar

| TVSum video ID | Tür | Basit yöntem F1 | Bizim gerçek sistem F1 | TVSum anlarını yakalama | Yalnız sahne seçimi F1 | Gerçek çıktı |
|---|---|---:|---:|---:|---:|---|
| `AwmHb44_ouw` | VT | 0,151 | **0,236** | %20,5 | 0,201 | Var |
| `XzYM3PfTM4w` | VT | 0,167 | **—** | — | 0,148 | Yok |
| `akI8YFjEmUw` | VU | 0,147 | **0,102** | %9,1 | 0,150 | Var |
| `0tmA_C6XwfM` | GA | 0,139 | **—** | — | 0,173 | Yok |
| `37rzWOQsNIw` | MS | 0,151 | **0,081** | %6,8 | 0,162 | Var |
| `GsAD1KT1xo8` | PK | 0,130 | **0,216** | %17,9 | 0,151 | Var |
| `91IHQYk1IQM` | PR | 0,165 | **0,151** | %14,0 | 0,202 | Var |
| `_xMr-HKMfVA` | FM | 0,183 | **0,168** | %16,3 | 0,278 | Var |
| `EE-bNr36nyA` | BK | 0,213 | **—** | — | 0,127 | Yok |
| `iVt07TCkFM0` | BT | 0,186 | **—** | — | 0,211 | Yok |
| `kLxoNp-UchI` | DS | 0,118 | **—** | — | 0,058 | Yok |
| **Ortalama (yalnız başarılı aynı 6 video)** | | **0,155** | **0,159** | **%14,1** | **0,191** | **6/11 var** |

Tür kodları: VT lastik değiştirme, VU aracı kurtarma, GA hayvan tımarı, MS sandviç yapımı, PK parkur, PR geçit töreni, FM flash mob, BK arıcılık, BT bisiklet numarası, DS köpek gösterisi. Böylece TVSum'un 10 türünün hepsinden en az bir video var; VT'den iki video var.

## Karşılaştırma neden makul?

TVSum videolarına 20'şer insan önem puanı verir; hazır insan özeti videoları yoktur. [TVSum açıklamasındaki](https://github.com/yalesong/tvsum/blob/master/README.md) ve [resmî MATLAB örneğindeki](https://github.com/yalesong/tvsum/blob/master/matlab/script_evaluate_result.m) yaklaşımla, her kişinin puanından en çok **%15 süre** kullanan ayrı referans özet çıkarıldı. Bizim basit yöntem de aynı videodan tam **%15** süreyi eşit aralıklarla alır.

TVSum referansları **sabit 60 karelik** parçalardan, Thanos özetleri ise **değişken süreli** sahnelerden oluşuyor. Thanos kesitlerini 60 kareye zorlayıp ürünü değiştirmedik: her iki seçimi de kaynak videonun aynı kare zaman çizelgesine çevirdik ve 20 referansa karşı kare düzeyinde F1 ölçtük. Yani karşılaştırılan şey “kaç sahne seçildiği” değil, **hangi video anlarının seçildiği**. Kesit sınırlarında küçük farklar F1'i etkileyebilir; bu ölçüm hikâye akıcılığı veya ses kalitesinin tamamını temsil etmez.

Thanos başarılı videolarda bile %15 bütçenin tamamını kullanmadı (yaklaşık **%9,6–13,9**). Bu yüzden ayrıca, yalnız başarılı 6 videoda **Thanos'un fiilen kullandığı süreyle eşit** bir basit yöntem hesaplandı: Thanos'un ortalama F1'i **0,159**, aynı süreli basit yöntemin **0,128**. Süre eşitlenince Thanos daha iyi; tam %15 kullanan basit yöntemle ise başa baş (**0,159'a karşı 0,155**). Her iki karşılaştırmada da yalnız **aynı başarılı 6 video** var.

## İyi ve kötü tarafı ne?

- **İyi örnekler:** `AwmHb44_ouw` (bizim 0,236; basit 0,151) ve `GsAD1KT1xo8` (0,216; basit 0,130). Ürün bazı videolarda gerçekten daha iyi anları seçiyor.
- **Ana sorun:** Çıktı alınamayan 5 videonun hepsinde aynı hata var: “Hedef süreye cümleyi bölmeden sığan sahne yok.” TVSum'daki kısa videolarda %15 hedef genellikle yalnızca 15–25 saniye; tam cümle/bağlam koruması bu bütçeyle çatışıyor. Alttaki teşhis hangi aşamada olduğunu gösterir.
- **Seçim puanları:** Aynı başarılı 6 videoda gerçek video üretimi devre dışı bırakılınca 0,191'e çıkıyor; basit yöntem 0,155. Bu fark sahne puanlamasında potansiyel gösteriyor ama gerçek çıktıya bütünüyle yansımıyor.

### Beş videoda neden çıktı yok?

| Video | %15 süre sınırı | Seçiciden gelen kesit | Sonuç |
|---|---:|---:|---|
| `XzYM3PfTM4w` | 16,7 sn | 52,1 sn | Bütçeyi aşıyor; konuşmayı kesmeden kısaltılamıyor. |
| `0tmA_C6XwfM` | 21,2 sn | 26,6 sn | Bütçeyi aşıyor; konuşmayı kesmeden kısaltılamıyor. |
| `EE-bNr36nyA` | 14,7 sn | 46,3 sn | Bütçeyi aşıyor; konuşmayı kesmeden kısaltılamıyor. |
| `iVt07TCkFM0` | 15,6 sn | 19,2 sn | Bütçeyi aşıyor; konuşmayı kesmeden kısaltılamıyor. |
| `kLxoNp-UchI` | 19,5 sn | Geçerli kesit yok | Bir konuşma-adayı eksik cümle diye eleniyor; geriye kesit kalmıyor. |

İlk dört videoda süreye sığan aday bulunamayınca `temporal_segment_builder.py` bütçeyi aşan **“en iyi tek kesit”** yedeğini döndürüyor. `samet_speech_guard.py` içindeki son kontrol bu kesiti süreye sığdırmak için cümleyi kesmiyor; tüm kesit düşünce açık hata veriyor. Beşinci videoda seçiciden boş liste geliyor. Dolayısıyla bu beş hata **FFmpeg kodlama veya model anahtarı hatası değil**; sahne seçimiyle konuşma/süre kuralının uyumsuzluğu. Yukarıdaki kesit süreleri seçim aşamasının çıktısıdır; son konuşma kontrolünün öncesidir.

## Sınırlar ve tekrar üretim

Bu, **11/50 videoluk** hızlı ve tür çeşitliliği amaçlı bir alt örnek; yeni eklenenler çoğunlukla kısa videolar. Rastgele seçilmiş veya tam TVSum benchmark sonucu değildir. Başarısız 5'i ortalamadan çıkarmak **yalnızca çalıştığı videolardaki kaliteyi** gösterir; bu yüzden **6/11 çıktı oranı** mutlaka beraber okunmalıdır. Thanos'un `balanced` analiz + **yerel (`local`)** modu ölçüldü; Paralon/RAG modu ölçülmedi. TVSum web kliplerinden oluşur; filmlerdeki hikâye kalitesini tek başına belirlemez.

Ham video bazlı sayılar: [gerçek sistem JSON](outputs/tvsum/comparison_eleven_end_to_end.json), [yalnız seçim JSON](outputs/tvsum/comparison_eleven_score_only.json). Ölçümün komutları ve teknik ayrıntıları [değerlendirme rehberinde](TVSUM_DEGERLENDIRME.md); hesaplayan kod [evaluate_tvsum.py](scripts/evaluate_tvsum.py) dosyasında.
