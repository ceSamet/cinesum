# Thanos × TVSum: 6 videoluk hızlı karşılaştırma

> Bu önceki, 6 videoluk ara rapordur ve başarısızlıkları sıfır sayan ortalama kullanır. Kullanıcının istediği **başarısızlıkları ortalamadan çıkaran güncel karşılaştırma** için [11 videoluk rapora](TVSUM_11_VIDEO_RAPORU.md) bakın.

## Kısa sonuç

Resmî TVSum paketindeki **50 kaynak videonun tamamı** indirildi ve katalog süreleriyle doğrulandı; hızlı karşılaştırmada bunların **6'sı** işlendi. Thanos `balanced` analiz ve **yerel (`local`) önem özeti** modunda 3/6 videoda özet üretti. Altı videonun, çıktı alınamayanlara **F1 = 0** verilerek hesaplanan ortalaması **0,082**; aynı videolarda %15 bütçeli eşit aralıklı seçim **0,171**. Thanos'un yalnız sahne önem puanlarına %15 knapsack uygulanınca ortalama **0,173** çıktı. Bu, küçük örnekte puanlayıcının temel çizgiye yakın; uçtan uca ürünün ise özellikle kısa hedefteki özet üretim başarısızlıkları nedeniyle geride olduğunu gösterir. **Bu, 50 videoluk genel TVSum skoru değildir.**

## Adil karşılaştırma nasıl yapıldı?

TVSum'un [resmî açıklamasına](https://github.com/yalesong/tvsum/blob/master/README.md) göre 50 video için 20'şer kişinin önem puanı vardır; hazır “insan özeti MP4” yoktur. [Resmî MATLAB değerlendirme örneği](https://github.com/yalesong/tvsum/blob/master/matlab/script_evaluate_result.m) gibi, her kişinin puanından **60 karelik sabit parçalarda, video uzunluğunun en çok %15'i** seçilerek ayrı referans özet üretildi.

Thanos'un sahneleri eşit uzunlukta değil. Bu yüzden çıktı kesitleri 60 kareye zorlanmadı: gerçek kaynak başlangıç/bitiş saniyeleri anotasyonların **aynı kare ekseninde 0/1 maskeye** çevrildi. Böylece sabit ve değişken kesitler seçilen **kareler** üzerinden, 20 referansla ayrı ayrı F1 hesaplanıp ortalanarak karşılaştırıldı. Thanos'un ses/konuşma koruması sonrası gerçek MP4'e giden kesitleri esas alındı. Kesit sınırlarının birebir uyuşmaması kare-F1'i etkileyebilir; bu metrik anlam kalitesinin tamamını ölçmez.

İki temel çizgi kullanıldı: (1) videoya yayılmış pencerelerle **tam %15** süre kullanan eşit aralıklı özet; (2) yalnız başarılı videolarda **Thanos'un fiilen kullandığı kare sayısıyla eşleştirilmiş** eşit aralıklı özet. İkincisi önemlidir, çünkü Thanos başarılı videolarda bile %15 bütçeyi tam doldurmadı. Çıktı veremeyen videolar dışlanmadı, sıfır puanlandı. “Yalnız puan” sütunu ise Thanos'un değişken PySceneDetect sahnelerine verdiği önem puanlarına %15 knapsack uygular; bu **ayrı bir tanı testi**, gerçek MP4 özetinin sonucu değildir.

## Sonuçlar

| TVSum video ID | Tür | Gerçek özet / kaynak | Thanos MP4 F1 | Eşit aralıklı %15 F1 | Aynı süreli eşit F1 | Yalnız puan F1 |
|---|---|---:|---:|---:|---:|---:|
| `AwmHb44_ouw` | VT | 38,17 / 353,59 sn (%10,8) | **0,236** | 0,151 | 0,111 | 0,201 |
| `XzYM3PfTM4w` | VT | üretilemedi | **0,000** | 0,167 | — | 0,148 |
| `akI8YFjEmUw` | VU | 15,24 / 133,31 sn (%11,4) | **0,102** | 0,147 | 0,095 | 0,150 |
| `91IHQYk1IQM` | PR | 14,05 / 110,53 sn (%12,7) | **0,151** | 0,165 | 0,100 | 0,202 |
| `EE-bNr36nyA` | BK | üretilemedi | **0,000** | 0,213 | — | 0,127 |
| `iVt07TCkFM0` | BT | üretilemedi | **0,000** | 0,186 | — | 0,211 |
| **6 video ortalaması** | | **3/6 çıktı** | **0,082** | **0,171** | — | **0,173** |

Başarılı 3 videonun kendi içinde Thanos F1 ortalaması **0,163**, aynı süreli eşit seçim ortalaması **0,102**. Ancak başarısızları çıkaran bu sayı ürünün genel performansı değildir. Üç başarısızlıkta aynı hata alındı: “Hedef süreye cümleyi bölmeden sığan sahne yok.” Özellikle 1,5–2 dakikalık videolarda %15 hedef 15–17 saniye olduğundan, konuşma bütünlüğü kuralı özet üretimini engelledi.

## Yorum ve sınır

- En açık darboğaz **kısa bütçede çıktı üretme oranı**: 3/6. Bunu düzeltmeden 50-video ortalamasını yükseltmek zor.
- Puanlayıcı tek başına **0,173**, eşit aralıklı seçim **0,171**: bu altı örnekte belirgin üstünlük gösterilmiyor. Bazı videolarda iyi, bazılarında kötü.
- Uçtan uca başarılı videolarda süre eşitlenince seçimin yararı var; fakat bunun genellenip genellenmediği 6 örnekten söylenemez.
- Örnek küçük ve kolaylık amaçlı seçildi; kategori başına eşit/rasgele örnekleme, güven aralığı ve 50-video sonucu yok. Paralon/RAG modu değil **yerel mod** ölçüldü. TVSum F1, hikâye akışı, konuşma doğallığı veya ses kalitesi gibi kullanıcı deneyimini doğrudan puanlamaz.

Tekrarlanabilir ham sonuçlar: [uçtan uca JSON](outputs/tvsum/comparison_six_end_to_end.json), [yalnız puan JSON](outputs/tvsum/comparison_six_score_only.json). Ölçüm kodu ve komutlar için [TVSum değerlendirme rehberi](TVSUM_DEGERLENDIRME.md).
