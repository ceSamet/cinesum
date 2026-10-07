# Thanos × TVSum değerlendirmesi

TVSum, 50 farklı türde web videosu ve video başına 20 kişinin 1–5 arası önem puanlarını içerir. Hazır insan özeti MP4'leri yoktur; referans özetler bu puanlardan türetilir. [Veri kümesinin açıklaması](https://github.com/yalesong/tvsum/blob/master/README.md) ve [resmî değerlendirme örneği](https://github.com/yalesong/tvsum/blob/master/matlab/script_evaluate_result.m) bu protokolün kaynaklarıdır.

## Ölçüm

1. Kaynak video, YouTube ID'si (`AwmHb44_ouw.mp4`) veya Gökdeniz'in tarihî adı (`video1.mp4`) ile eşleştirilir. İsim eşleşmesi yetmez: süre de TVSum kataloğundaki süreyle uyuşmalıdır. Böylece Thanos'un alakasız `video1.mp4` dosyası yanlışlıkla TVSum olarak puanlanmaz.
2. Thanos `importance` kategorisinde, videonun **%15'i kadar** hedef süreyle gerçek özet MP4'ünü üretir. Varsayılan `balanced` analiz + `local` anlatı modu kullanılır. `rag_llm` ayrıca ölçülebilir fakat ağ/anahtar gerektirir; bunların skoru ayrı raporlanmalıdır.
3. Çıktıdaki kaynak zaman aralıkları anotasyon karelerine dönüştürülür; Thanos'un değişken uzunluklu kesitleri 60 kareye yuvarlanmaz. Her kullanıcı için önem puanlarından 60 karelik parçalarda %15 bütçeli 0/1 knapsack referansı çıkarılır. F1, recall (referans kareleri yakalama oranı) ve precision (seçilen karelerin referansa denk gelme oranı) önce 20 kullanıcı üzerinde, sonra videolar üzerinde ortalanır. Aynı referanslarla bütçeyi tam dolduran eşit aralıklı %15 temel çizgisi ve başarılı çıktılarda Thanos'un kullandığı süreye eşit ikinci temel çizgi hesaplanır.
4. %15 bütçesini aşan sonuç geçersiz sayılır. Model özet üretemezse hata ve **çıktı başarısızlığı** ayrıca kaydedilir. Ham JSON'da başarısızlıkları sıfır sayan tüm-denemeler ortalaması ile yalnız başarılı videoların ortalaması **ayrı alanlardır**; [11 videoluk raporda](TVSUM_11_VIDEO_RAPORU.md) kullanıcının istediği gibi yalnız başarılı videolar ortalamaya alınır ve 6/11 çıktı oranı ayrıca gösterilir. 50 videonun tamamı işlenmezse sonuç **kısmi** olarak işaretlenir; tam TVSum benchmark skoru sayılmaz. Bu tek-mod/tek-veri-kümesi ölçümüdür, eğitim/test çapraz doğrulaması veya insan değerlendirmesi değildir.

## Çalıştırma

Proje kökünden:

```bash
.venv/bin/python thanos/scripts/evaluate_tvsum.py --videos-dir /TVSUM_VIDEO_KLASORU
.venv/bin/python thanos/scripts/evaluate_tvsum.py --videos-dir /TVSUM_VIDEO_KLASORU --run-model
.venv/bin/python thanos/scripts/evaluate_tvsum.py --videos-dir /TVSUM_VIDEO_KLASORU --score-only
```

İlk komut yalnızca 50 video dosyasının eşleşme/süre denetimini yapar. İkinci komut tam Thanos analizini, özet üretimini ve F1 ölçümünü çalıştırır. Üçüncü komut aynı analizden gelen önem puanlarını %15 knapsack ile ölçer; konuşma sınırı ve MP4 dışa aktarımını **ölçmez**, bu yüzden ayrı bir tanısal metrik olarak okunmalıdır. Komutlar videoları kendi başına indirmez. İlk deneme için `--video-id AwmHb44_ouw` eklenebilir. Rapor `thanos/outputs/tvsum/evaluation.json` dosyasına yazılır; üretilen özetler aynı klasörde tutulur. Tüm 50 videonun işlenmesi donanıma bağlı olarak uzun sürebilir.

Önceden seçilmiş aralıklar da ölçülebilir:

```bash
.venv/bin/python thanos/scripts/evaluate_tvsum.py --videos-dir /TVSUM_VIDEO_KLASORU --predictions-dir /TAHMIN_KLASORU
```

Her `<video_id>.json` dosyası `[{"start": 12.4, "end": 18.6}]` veya `{"segments": [...]}` biçiminde kaynak video saniyelerini içermelidir. Bu kip yalnızca verilen tahminleri puanlar, Thanos'u çalıştırmaz.

Anotasyonlar `gokedniz/cinesum/dataset/data/` içindedir. Kaynak TVSum videoları bu repoda bulunmuyorsa gerçek model F1 değeri hesaplanamaz; anotasyon veya önceki projenin eski skorları bu açığı kapatmaz. Videoları doğrudan [resmî TVSum indirme sayfasından](http://people.csail.mit.edu/yalesong/tvsum) edinin ve yerel klasörünüzü `--videos-dir` ile belirtin. Veri lisansını gözeterek video dosyalarını repoya commit etmeyin.
