# Thanos — son değişiklikler ve çalışma durumu

Güncelleme: 7 Ekim 2026. Bu belge, son keyframe hızlandırmasını ve sayfa yenilendiğinde işlem durumunu geri getiren arayüz çalışmasını özetler.

## 1. Keyframe çıkarma hızlandırıldı

`src/scene_detection/keyframe_extractor.py` içindeki yöntem Gökdeniz projesindeki kodla aynıydı. Eski sürüm, her aday kare için `VideoCapture.set(CAP_PROP_POS_FRAMES, ...)` ile videoda yeniden konumlanıyordu. MP4 gibi sıkıştırılmış videolarda bu çok sayıda pahalı seek işlemine dönüşüyordu.

Yeni sürüm her aralığın başına **bir kez** gider; sonraki kareleri `grab()` ile sırayla ilerletir ve yalnız aday kareleri `retrieve()` ile alır. Kısa sahnede bir, uzun sahnede üç keyframe seçme kuralı; aday kareler ve Laplacian netlik puanı korunmuştur. `sample_step` için geçersiz değer kontrolü de eklendi.

Aynı gerçek video parçasındaki karşılaştırmada eski yöntem **1,923 sn**, yeni yöntem **0,313 sn** sürdü. Seçilen altı karenin numarası ve netlik puanı birebir aynıydı. Bu yaklaşık 6 kat fark yalnız ölçülen parça içindir; her videoda aynı oran garanti edilmez. İlgili test: `tests/test_keyframe_extractor.py`.

Uzun koşular arasında doğrudan karşılaştırma yapılamaz: eski `video2` (3 sa 1 dk) için keyframe aşaması 57 dk 32 sn; yeni `video3` (2 sa 45 dk) için 12 dk 55 sn sürdü. Video içerikleri ve sahne sayıları farklıdır.

## 2. Sayfa yenilenince işlem durumu geri geliyor

Değişen dosyalar: `app.py`, `static/app.js`, `static/style.css`, `templates/index.html`; yeni test: `tests/test_progress_recovery.py`.

- Tarayıcı aktif işlem kimliğini ve son seçilen videoyu `localStorage` içinde saklıyor. Sayfa yeniden açıldığında `/api/progress/{task_id}` üzerinden aynı işe bağlanmayı deniyor.
- İşlem kimliği bulunamazsa videoların analiz kayıtlarına bakıyor. Yeni `/api/analysis-status/{video_alias}` endpoint'i tamamlanan/çalışan aşamaları, sahne sayısını ve çıkarılan/toplam keyframe sayısını döndürüyor.
- Sunucu henüz yeni endpoint ile yeniden başlatılmamışsa arayüz mevcut statik manifestten aşamayı geri buluyor. Keyframe sırasında, kaydedilmiş kareleri kontrol ederek **yaklaşık** kare ilerlemesini gösterebiliyor.
- Yükleme ve özetleme sonucu, istek sırasında sayfa kapansa bile çalışan işin sonuç verisinden geri alınabilecek şekilde tutuluyor. “Arka planda devam et” ve “İşlem durumunu aç” düğmeleri eklendi.
- İlerleme sorgusu artık 300 ms yerine 1 saniyede bir yapılıyor.

Önemli sınır: İşlem geçmişi sunucu yeniden başlatıldığında tamamen kalıcı değildir. Analiz manifesti diskte kaldığından yükleme/analiz aşaması geri bulunabilir; fakat çalışan sunucuyu yeniden başlatmak o anda işlenen işi kesebilir. **Devam eden analiz bitmeden sunucuyu yeniden başlatmayın.**

## 3. Son gözlenen durum

7 Ekim 2026, 14:17 (Türkiye saati) kontrolünde `video3` için sahne tespiti, 3.906 keyframe, CLIP, ses çıkarma, Whisper ve transkript düzeltme tamamlanmıştı. Manifest `samet_people` (kişi/konuşmacı eşleştirme) aşamasını `running` gösteriyordu; bu aşama yaklaşık 25 dakikadır sürüyordu. `video3_people.json` ve özet MP4 henüz yoktu.

Bu aşama ara ilerleme kaydetmediğinden manifestteki `running` kaydı tek başına işlemin gerçekten ilerlediğini kanıtlamaz. Kalan süre güvenilir biçimde hesaplanamıyor. Kişi aşamasından sonra ses özellikleri ve hikâye gruplama kalır; özet MP4 ayrıca üretilir.

## 4. Doğrulama ve sonraki ihtiyaç

Thanos test paketinde **139 test ve 9 alt test** geçti; son değişikliklerden sonra ilgili dört test yeniden geçti. `static/app.js` sözdizimi `node --check` ile doğrulandı.

Bir sonraki faydalı iyileştirme, özellikle kişi analizi, ek Whisper ve FFmpeg özet dışa aktarması için **kalıcı aşama süreleri ve ara heartbeat** kaydetmek. Şu an bu aşamaların iç ilerlemesi olmadığı için uzun beklemelerde kesin ETA veya takılma teşhisi verilemiyor.
