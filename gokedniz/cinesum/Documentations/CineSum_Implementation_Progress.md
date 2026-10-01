# CineSum Birleşik Mimari — Implementasyon Günlüğü

> Başlangıç tarihi: 25 Ağustos 2026  
> Güncel özet algoritması: CineSum v3.9.0; analiz pipeline sürümü: 3.1.0
> Durum: Aktif geliştirme  
> Bakım kuralı: Bu dosya her geliştirme aşamasında kod ve testlerle birlikte güncellenir.

> 1 Ekim 2026 giriş noktaları: [Güncel mimari ve ajan devri](CineSum_Current_State_and_Handoff.md), [Üç perde planı ve görev takibi](CineSum_Three_Act_Implementation_Plan.md). Aşağıdaki tarihli bölümler tarihsel kayıtlardır; en son gözlemler bölüm 19'dadır.

## 1. Bu çalışmanın amacı

Mevcut CineSum projesinin cache-first mimarisini koruyarak:

- Samet'in hızlı, anlatı dengeli ve açıklanabilir özet seçim yaklaşımını,
- Nezihat'ın gelişmiş konuşmacı ve kelime hizalama yaklaşımını,
- CineSum'ın bir kere analiz edip aynı videodan hızlıca farklı özetler üretme avantajını

tek ve modüler bir sistem altında birleştirmek.

## 2. Alınan temel mimari kararlar

1. Mevcut CineSum ana uygulama ve feature-store omurgası olarak korunacak.
2. Video analizi ile özet üretimi birbirinden ayrılacak.
3. Ağır modeller her özet isteğinde tekrar çalıştırılmayacak.
4. Samet'in anlatı kotası, Knapsack, çeşitlilik ve StoryScene yaklaşımı temel seçim motoruna eklenecek.
5. Nezihat'ın Pyannote diarization yaklaşımı varsayılan hattı yavaşlatmayacak; isteğe bağlı kalite katmanı olacak.
6. Her analiz aşaması bağımsız cache, sürüm, ayar hash'i ve süre bilgisine sahip olacak.
7. `fast`, `balanced` ve `quality` olmak üzere üç analiz profili desteklenecek.

## 3. Tamamlanan çalışma — CineSum v3.1 altyapısı

### 3.1. Sürümlü feature manifest

Eklendi:

- `src/core/feature_manifest.py`
- `src/core/pipeline_config.py`
- `src/core/analysis_pipeline.py`
- `src/core/timing.py`

Her video için aşağıdaki bilgiler tutuluyor:

- Kaynak video SHA-256 fingerprint'i
- Pipeline ve JSON şema sürümü
- Aşama konfigürasyonu ve config hash'i
- Aşama durumu: `running`, `complete`, `failed`
- Aşama çalışma süresi
- Üretilen artifact yolları
- Hata türü ve hata mesajı

Cache aşamaları:

1. `scene_detection`
2. `keyframes`
3. `clip`
4. `audio_extraction`
5. `transcript`
6. `audio_features`

Kaynak video, model veya aşama ayarı değiştiğinde yalnızca etkilenen aşamalar yeniden çalışır.

### 3.2. Faster-Whisper geçişi

`src/audio/whisper_transcriber.py` yeniden düzenlendi.

Eklenen davranışlar:

- `faster-whisper` backend'i
- CPU `int8` ve GPU `float16` desteği
- VAD filtresi
- Otomatik dil tespiti
- Kelime zaman damgaları
- Kelime güven değerleri
- Model adı, cihaz ve compute type bazlı model cache'i
- Çakışan Whisper segmentlerinin konuşma süresini iki kez saymama

Analiz profilleri:

| Profil | Whisper | Beam | Compute |
|---|---|---:|---|
| Fast | small | 1 | GPU float16 / CPU int8 |
| Balanced | small | 3 | GPU float16 / CPU int8 |
| Quality | medium | 5 | GPU float16 / CPU int8 |

Varsayılan profil `balanced` olarak belirlendi.

### 3.3. CLIP ve scoring düzeltmeleri

- CLIP keyframe'leri artık tek dev batch yerine CPU'da 8, GPU'da 32 karelik batch'lerle işleniyor.
- Bütün görsellerin aynı anda RAM'de tutulması kaldırıldı.
- Scoring ve custom query içindeki hatalı `text_vecs = text_vecs` dalları düzeltildi.
- CLIP modeli embedding, scoring ve özel sorgu yolları arasında yeniden kullanılıyor.
- Yerel model önce çevrimdışı cache'ten yükleniyor; eksikse Hugging Face deneniyor.
- FFmpeg ses çıkarma hataları artık sessizce geçilmiyor.

### 3.4. Web ve CLI entegrasyonu

- Upload API merkezi `analyze_video_features` hattına bağlandı.
- Upload sırasında `fast`, `balanced`, `quality` profil doğrulaması eklendi.
- Video alias üretimi dosya sayısına değil mevcut en büyük numaraya göre yapılıyor.
- `scripts/run_full_pipeline.py` yeni manifest/cache sistemine bağlandı.
- `scripts/benchmark_analysis.py` eklendi.

Benchmark örneği:

```powershell
.\venv\Scripts\python.exe scripts\benchmark_analysis.py video55 --profile balanced --force
.\venv\Scripts\python.exe scripts\benchmark_analysis.py video55 --profile balanced
```

İlk komut gerçek işleme süresini, ikinci komut cache performansını ölçer.

### 3.5. Doğrulama durumu

- Başlangıçtaki test sayısı: 6
- Güncel test sayısı: 14
- Son sonuç: 14/14 başarılı
- Gerçek `video1` scoring doğrulaması: 24 sahne başarıyla skorlandı
- CLIP çevrimdışı cache doğrulaması: başarılı
- İlk sahte pipeline çalışması: 0/6 cache hit
- İkinci sahte pipeline çalışması: 6/6 cache hit

## 4. Tamamlanan çalışma — CineSum v3.2 seçim motoru

Amaç, Samet'in en değerli seçim fikirlerini CineSum temporal coherence hattına eklemek.

Uygulanan işlem sırası:

1. Videoyu başlangıç, orta ve final bölgelerine ayırmak.
2. Hedef özet süresini anlatı kotalarına bölmek.
3. Benzer çekimleri MMR ile cezalandırmak.
4. Her bölgede süre bütçesi altında en yüksek değeri Knapsack ile seçmek.
5. Kullanılmayan bölgesel süreyi genel aday havuzuna geri vermek.
6. Seçilen shot'ları mevcut temporal context/merge/speech alignment hattına göndermek.
7. Seçim kararlarını debug JSON içinde açıklanabilir biçimde saklamak.

Eklenen dosyalar:

- `src/selection/__init__.py`
- `src/selection/narrative_selector.py`
- `tests/test_narrative_selector.py`

Temporal entegrasyon:

- `src/summary/temporal_segment_builder.py` önce göreli tepe adaylarını çıkarır.
- Altı veya daha fazla aday varsa `narrative_mmr_knapsack` stratejisi çalışır.
- Çok küçük aday havuzlarında önceki davranışı korumak için `legacy_small_pool` kullanılır.
- Seçilen shot'lar mevcut context expansion, shot snapping, merge ve speech alignment hattına gönderilir.
- Debug segment JSON'u artık seçim stratejisini, bölgesel bütçeleri, seçilen shot kimliklerini, MMR skorlarını ve seçim gerekçelerini içerir.

Başlangıç kotaları:

| Bölge | Video aralığı | Özet bütçesi |
|---|---:|---:|
| Intro | İlk %20 | %20 |
| Middle | %20–%75 | %55 |
| Ending | Son %25 | %25 |

Kategoriye göre başlangıç MMR çeşitlilik ağırlıkları:

| Kategori | Çeşitlilik ağırlığı |
|---|---:|
| Action | 0.20 |
| Dialogue | 0.12 |
| Importance | 0.25 |
| Custom | 0.18 |

### 4.1. v3.2 doğrulama sonucu

- Güncel otomatik test sayısı: 18
- Sonuç: 18/18 başarılı
- Knapsack birleşik değer optimizasyonu: başarılı
- MMR yakın tekrar cezası: başarılı
- Intro/middle/ending temsili: başarılı
- Eski temporal coherence regresyonları: 6/6 başarılı
- Gerçek `video1`, 30 saniyelik importance özeti:
  - Strateji: `narrative_mmr_knapsack`
  - Seçilen shot'lar: 3, 19, 22
  - Bölgesel dağılım: intro 1, middle 1, ending 1
  - Oluşan segment sayısı: 3
  - Tahmini gerçek özet süresi: 25 saniye
  - Kesim yoğunluğu: 0.12 segment/saniye

## 5. Tamamlanan çalışma — CineSum v3.3 StoryScene

Shot ile SummarySegment arasına yeni bir `StoryScene` analiz katmanı eklendi.

Eklenen dosyalar:

- `src/story/__init__.py`
- `src/story/story_scene_builder.py`
- `tests/test_story_scene_builder.py`

Gruplama sinyalleri:

| Sinyal | Ağırlık |
|---|---:|
| Gerçek CLIP görsel embedding benzerliği | 0.52 |
| TF-IDF transkript embedding benzerliği | 0.28 |
| Konuşma yoğunluğu devamlılığı | 0.12 |
| Zamansal yakınlık | 0.08 |

Uygulanan davranışlar:

- Çok keyframe'li shot'ların CLIP vektörleri ortalanıp normalize edilir.
- Komşu shot transkriptleri çok dilli unigram/bigram TF-IDF vektörlerine dönüştürülür.
- Komşu sınırların continuity skoru hesaplanır.
- Videoya özgü adaptif continuity eşiği üretilir.
- En fazla 45 saniye veya 12 shot içeren StoryScene grupları oluşturulur.
- Tekrarlanan transkript metinleri StoryScene metninde tekilleştirilir.
- Her StoryScene için görsel centroid embedding'i `.npy` olarak saklanır.
- Her shot için `story_scene_id`, StoryScene içi konum ve grup shot sayısı kaydedilir.
- StoryScene üretimi manifest'e yedinci bağımsız cache aşaması olarak eklendi.
- Scoring sonuçlarına StoryScene kimliği ve baskın mod bilgisi eklendi.
- MMR aynı StoryScene içindeki shot'ları maksimum benzer kabul eder.
- Seçim motoru aynı StoryScene'den en fazla iki çekim seçer.

Üretilen artifact'ler:

```text
outputs/features/story/<video>_story_scenes.json
outputs/features/story/<video>_story_embeddings.npy
```

Bu çalışma zamanı artifact'leri ve manifest dosyaları `.gitignore` kapsamına alındı.

### 5.1. v3.3 doğrulama sonucu

- Güncel otomatik test sayısı: 22
- Sonuç: 22/22 başarılı
- Görsel ve metinsel devamlılık gruplama testi: başarılı
- Maksimum StoryScene süresi testi: başarılı
- JSON ve embedding artifact testi: başarılı
- Aynı StoryScene'den en fazla iki shot seçme testi: başarılı
- Gerçek `video1` sonucu:
  - Shot sayısı: 24
  - StoryScene sayısı: 15
  - Adaptif continuity eşiği: 0.68
  - Çoklu shot grupları: 3, 4, 3, 2 ve 2 shot içeren gruplar
  - 30 saniyelik importance seçimindeki StoryScene kimlikleri: 1, 11, 13
  - Intro/middle/ending dengesi korunmaya devam etti
  - Nihai süre: 25 saniye, 3 segment

### 5.2. Yarım kalmış analiz için otomatik kurtarma düzeltmesi

`video58avengers` özetlenirken alınan `Visual CLIP özellikleri bulunamadı` hatası incelendi.

Tespit edilen durum:

- Kaynak video: yaklaşık 2.1 GB
- Hazır keyframe: 2.840 dosya / yaklaşık 318 MB
- Keyframe'lerin kapsadığı sahne kimliği: 1–2.470
- Eksik artifact'ler: scene list, CLIP `.npy`, CLIP metadata, audio features, StoryScene ve manifest

Uygulanan düzeltmeler:

- `src/core/analysis_status.py` ile minimum scoring artifact durumu merkezi olarak denetlenir.
- Özet endpoint'i eksik artifact gördüğünde hata vermek yerine ilk analizi otomatik başlatır.
- İlk analiz ilerlemesi özet progress penceresinde gösterilir.
- Aynı videoya gelen eş zamanlı analiz istekleri video bazlı lock ile tekilleştirilir.
- Analiz bittikten sonra scoring ve özet export aynı istek içinde devam eder.
- Hazır videolarda ağır analiz çağrılmaz.
- Eski keyframe'ler ancak yeni scene detection sonucu ile sahne kimlikleri tam eşleşirse yeniden kullanılır.
- Tam eşleşme yoksa yanlış shot/embedding eşlemesini önlemek için keyframe'ler güvenli biçimde yeniden üretilir.
- Video alias, kategori, profil ve negatif süre girdileri API seviyesinde doğrulanır.
- Video bilgi endpoint'i hangi temel artifact'lerin eksik olduğunu raporlar.

Bu düzeltmelerden sonra otomatik test sayısı 27'ye çıktı ve 27/27 test başarılıdır.

### 5.3. RTX 4050 GPU hızlandırması

`video58avengers` ilk analizinin CPU'da uzun sürmesi üzerine çalışma zamanı incelendi.

Tespit edilen durum:

- Sanal ortamda `torch 2.13.0+cpu` ve `torchvision 0.28.0+cpu` kuruluydu.
- RTX 4050 sistem tarafından görülmesine rağmen CLIP ve Whisper CPU'ya düşüyordu.
- Faster-Whisper'ın otomatik cihaz seçimi de Torch CUDA durumuna bağlıydı.

Uygulanan değişiklikler:

- PyTorch `2.11.0+cu128` ve Torchvision `0.26.0+cu128` kuruldu.
- CUDA görünürlüğü ve RTX 4050 üzerinde gerçek tensor işlemi doğrulandı.
- CLIP cihazı runtime sırasında otomatik seçilir.
- CLIP batch boyutu GPU'da 32, CPU'da 8 olarak otomatik belirlenir.
- Faster-Whisper GPU'da `float16`, CPU fallback durumunda `int8` kullanır.
- CLIP ve transcript manifest config'lerine gerçek cihaz, batch ve compute type yazılır.
- NumPy, Numba uyumluluğu için `2.4.6` sürümünde tutuldu.
- Gerçek bir keyframe ile CLIP CUDA inference testi `(1, 512)` çıktı üretti.
- 12 saniyelik gerçek ses ile Faster-Whisper CUDA inference testi 3 segment üretti.
- GPU geçişi sonrasında otomatik test sonucu 29/29 başarılıdır.

CUDA 12.8 seçilmesinin nedeni CTranslate2 `4.8.1` Windows paketinin çalışma
zamanında `cublas64_12.dll` beklemesidir. CUDA 13.0 Torch CLIP için çalışsa da
Faster-Whisper ilk gerçek inference sırasında bu DLL'i bulamadığı için 12.8 ortak
ve doğrulanmış çalışma zamanı olarak seçildi.

Not: PySceneDetect sahne tespiti ağırlıklı olarak CPU işidir. GPU hızlandırmasının ana
kazanımı CLIP embedding ve Whisper transkripsiyon aşamalarındadır.

### 5.4. Uzun videoda `%69` MMR performans düzeltmesi

`video58avengers` ilk analizi tamamlandıktan sonra arayüz yaklaşık 28 dakika boyunca
`IMPORTANCE puanları hesaplanıyor — %69` durumunda kaldı.

Kök neden:

- Scoring değil, `build_temporal_segments` içindeki MMR sıralaması bloklanıyordu.
- Video 2.470 sahne içerdiği için eski MMR her turda bütün kalan adayları bütün seçilmiş
  adaylarla yeniden karşılaştırıyordu.
- Benzerlikler yeniden hesaplandığından toplam maliyet yaklaşık kübik büyüyordu.
- Süreç donmamıştı fakat milyarlarca Python düzeyi benzerlik işlemi yapıyordu.

Uygulanan çözüm:

- Her adayın seçilmiş kümeye maksimum benzerliği cache'lenir.
- Yeni bir shot seçildiğinde yalnızca o shot ile kalan adayların benzerliği hesaplanır.
- Böylece MMR çekirdeği kübik maliyetten karesel maliyete indirildi.
- MMR öncesi havuz relevance skoruyla ön filtrelenir.
- Aday havuzu hedef süreye göre dinamik belirlenir ve bölge başına en fazla 400 shot olur.
- Sıralanacak aday sayısı hedef süre/minimum segment süresine göre sınırlandırılır.
- Debug seçim verisine `candidate_pool_counts` ve `ranked_candidate_counts` eklendi.
- 2.470 sahnelik uzun-video regresyon testi eklendi.
- Artımlı similarity cache çağrı sınırı testi eklendi.

Gerçek `video58avengers` doğrulaması:

| İşlem | Süre |
|---|---:|
| 2.470 sahne scoring | 0,971 saniye |
| MMR + Knapsack + temporal segment seçimi | 0,036 saniye |
| Toplam `%69` hesaplama yolu | Yaklaşık 1,01 saniye |
| 12 saniyelik, 2 segmentli FFmpeg smoke export | 4,121 saniye |

60 saniyelik importance benchmark sonucu 7 segment ve 48,9 saniye gerçek seçili süre
üretti. Bölgesel ve remainder MMR havuzlarının her biri 80 aday, sıralanan alt kümelerin
her biri 16 aday oldu. Güncel otomatik test sonucu 31/31 başarılıdır.

## 6. Aktif sonraki aşama

### CineSum v3.4 — Kelime/cümle sınırı hizalama

> Durum: Tamamlandı — 26 Ağustos 2026

#### 6.1. Mevcut akış incelemesi

Devam eden `video58avengers` analizi kesilmesin diye bu aşamada yalnızca okuma ve
tasarım çalışması yapıldı; çalışan `.py` dosyaları değiştirilmedi.

Tespitler:

- Faster-Whisper kelime başlangıç/bitiş zamanlarını ve güven değerlerini zaten üretiyor.
- `map_transcript_to_scenes` kelimeleri ilgili shot kayıtlarına ekliyor.
- Scoring katmanı `transcript_text`, `speech_ratio` ve yoğunluğu taşıyor fakat `words`
  alanını temporal builder'a aktarmıyor.
- Mevcut speech alignment, gerçek cümle sınırı yerine konuşmalı shot'ın tamamına genişliyor.
- `max_segment_dur` kontrolü segment sonunu doğrudan saniye ile kesebiliyor.
- Bütçe optimizasyonundaki son kırpma hizalamadan sonra yapıldığı için daha önce düzgün
  hizalanmış bir cümleyi yeniden ortadan kesebiliyor.
- Aynı Whisper segmenti birden fazla shot ile çakıştığında `transcript_text` tekrarları
  oluşabiliyor; zaman damgalı kelimeler üzerinden tekilleştirme daha güvenli olacak.

#### 6.2. Uygulama sırası

1. `words` alanını audio feature'dan scoring sonucuna kayıpsız taşımak.
2. Zaman damgası temelli kelime tekilleştirme yardımcı fonksiyonunu eklemek.
3. Cümle sonlarını `.`, `?`, `!`, `…` işaretleri ve kelimeler arası sessizlik boşluğu
   ile belirlemek.
4. Noktalama eksik olduğunda yaklaşık `0.75` saniye ve üzeri konuşma boşluğunu güvenli
   utterance sınırı kabul etmek.
5. Segment başlangıç/bitişini en yakın güvenli kelime veya cümle sınırına hizalamak.
6. Doğrudan `seg.end = seg.start + remaining` kırpmasını kaldırıp güvenli bitiş adayı
   seçmek; aday yoksa segmenti kırpmak yerine atlamak veya süre toleransını kullanmak.
7. Son segment merge işleminden sonra transcript ve boundary metadata'sını yeniden
   hesaplamak.
8. Özet algoritması sürümünü debug çıktısında saklamak; feature cache'i gereksiz yere
   geçersiz kılmamak.

#### 6.3. Kategori bazlı sınır politikası

| Kategori | Birinci öncelik | İzin verilen hizalama |
|---|---|---|
| Dialogue | Cümle/utterance bütünlüğü | Başlangıç ve bitişi yakın cümle sınırına genişlet |
| Action | Görsel shot bütünlüğü | Yalnızca kelime ortasında kesimi engelle; shot sınırını koru |
| Importance | Anlatı + konuşma dengesi | Sınırlı cümle genişletmesi, aksi halde güvenli kelime sınırı |
| Custom | Sorgu sahnesi + konuşma dengesi | Importance ile aynı hibrit politika |

Başlangıç güvenlik limitleri:

- Dialogue için sınır başına en fazla `3.0` saniye genişleme.
- Importance/custom için sınır başına en fazla `1.5` saniye genişleme.
- Action için en yakın güvenli kelime/sessizlik noktasına en fazla `0.6` saniye kayma.
- Nihai hedef süre toleransı normalde `%10`; eksiksiz dialogue cümlesi için en fazla
  `%15` ve yalnızca debug gerekçesi kaydedilerek.
- Hiç kelime zaman damgası yoksa mevcut shot tabanlı davranış fallback olarak korunur.

Bu değerler ilk gerçek sonuçlar ölçüldükten sonra ayarlanacak; kod içinde dağınık sabitler
yerine kategori konfigürasyonunun parçası olacak.

#### 6.4. Yeni debug alanları

Her SummarySegment için aşağıdaki açıklanabilir alanların eklenmesi planlandı:

- `boundary_mode`: `sentence`, `silence`, `word`, `shot` veya `fallback`
- `original_start` / `original_end`
- `aligned_start` / `aligned_end`
- `start_delta` / `end_delta`
- `boundary_reason`
- `complete_utterance`
- `budget_trimmed`
- `word_count`
- Tekilleştirilmiş `transcript_text`

#### 6.5. Tamamlanan uygulama ve doğrulama

- Audio feature içindeki Whisper `words` kayıtları scoring sonucuna kayıpsız taşındı.
- `src/summary/speech_boundaries.py` ile zaman damgası doğrulama, kelime
  tekilleştirme, noktalama/sessizlik tabanlı utterance üretimi ve kategori bazlı sınır
  politikaları eklendi.
- Bütçeye sığdırmak için kullanılan kör saniye kesimi kaldırıldı. Güvenli cümle,
  kelime veya shot sonu bulunamazsa segment artık konuşmanın ortasından kesilmiyor.
- Eski, kelime zaman damgası olmayan analizler için konuşmalı shot aralığını koruyan
  geriye dönük fallback bırakıldı.
- Merge sonrasında transcript ve konuşma bütünlüğü metadata'sı yeniden hesaplanıyor.
- Debug çıktısına `summary_algorithm_version: 3.4.0` ve planlanan sınır alanları eklendi.
- Sekiz yeni speech-boundary birim testi ve bir temporal entegrasyon testi eklendi.
- Proje sanal ortamında tam test paketi **40/40 başarılı** tamamlandı.

Gerçek `video58avengers` 60 saniyelik importance doğrulaması:

| Ölçüm | Sonuç |
|---|---:|
| Sahne sayısı | 2.470 |
| Scoring | 1,453 saniye |
| v3.4 temporal seçim | 0,115 saniye |
| Seçilen segment | 7 |
| Gerçek seçili süre | 48,9 saniye |
| Bütün utterance kontrolü geçen | 7/7 |
| Bütçe nedeniyle kör kırpılan | 0 |

Not: Eski analiz artifact'leri korunmuştur. Mevcut Avengers analizindeki kelime zamanları
yalnız ilgili shot aralıklarında kullanılmış; zaman damgası bulunmayan aralıklarda fallback
devreye girmiştir. Yeniden ağır video analizi yapılmamıştır.

#### 6.5. Kabul ve regresyon testleri

Uygulama sırasında en az aşağıdaki testler eklenecek:

1. Noktalama bulunan cümlenin ortadan kesilmemesi.
2. Noktalama bulunmadığında sessizlik boşluğunun cümle sınırı sayılması.
3. Segment sınırının bir kelimenin başlangıç ve bitişi arasına düşmemesi.
4. Dialogue kategorisinin cümle bütünlüğünü shot sınırına tercih etmesi.
5. Action kategorisinin uzun konuşma nedeniyle shot bütünlüğünü bozmaması.
6. Importance/custom kategorilerinde maksimum genişleme limitinin korunması.
7. Bütçe kırpmasından sonra cümlenin yeniden ortadan kesilmemesi.
8. Süre toleransının yalnızca izin verilen oranda aşılması.
9. Çakışan shot'lardaki aynı kelimelerin transcript içinde tekrar etmemesi.
10. Kelime zaman damgası olmayan eski artifact'lerde fallback davranışının çalışması.
11. Merge sonrasında sınır ve transcript metadata'sının doğru yeniden hesaplanması.
12. Nihai segmentlerin kronolojik kalması ve negatif/çakışan süre üretmemesi.

#### 6.6. Uygulama sırasında dokunulacak dosyalar

```text
src/scoring/scoring_engine.py
src/summary/temporal_segment_builder.py
tests/test_temporal_segment_builder.py
tests/test_whisper_transcriber.py
Documentations/CineSum_Implementation_Progress.md
```

Gerekirse cümle sınırı mantığı ayrı ve saf bir yardımcı modüle taşınacak; ağır model
ve yeni analiz cache aşaması eklenmeyecek.

### 6.7. `video58avengers` canlı benchmark kaydı

Kaynak özellikleri:

| Alan | Değer |
|---|---:|
| Video süresi | 10.871,428 saniye / yaklaşık 181,2 dakika |
| Çözünürlük | 1920×808 |
| Kare hızı | 24000/1001, yaklaşık 23,976 FPS |
| Dosya boyutu | 2.110.322.946 byte |
| Mevcut eski keyframe | 2.840 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB |
| Analiz profili | balanced |
| Yeni analiz başlangıcı | 25 Ağustos 2026 16:05:26 |

Canlı durum kaydı:

| Aşama | Durum | Süre | Cihaz/ayar |
|---|---|---:|---|
| Scene detection | Tamamlandı | 1.046,872 sn / 17 dk 26,9 sn | CPU, PySceneDetect adaptive 3.0 |
| Keyframe | Cache/reuse tamamlandı | 0,668 sn | 2.840 eski keyframe exact coverage ile kullanıldı |
| CLIP | Tamamlandı | 74,867 sn | RTX 4050, batch 32 |
| Audio extraction | Tamamlandı | 18,807 sn | FFmpeg, 16 kHz mono PCM |
| Faster-Whisper | Tamamlandı | 213,382 sn / 3 dk 33,4 sn | RTX 4050, small/float16, beam 3 |
| Audio features | Tamamlandı | 12,854 sn | CPU |
| StoryScene | Tamamlandı | 0,640 sn | CLIP + transcript + temporal continuity |
| İlk analiz toplamı | Tamamlandı | 1.368,094 sn / 22 dk 48,1 sn | Cache öncesi tam analiz |
| Scoring + MMR seçimi | Tamamlandı | 1,007 sn | 2.470 sahne, optimize edilmiş seçim |
| FFmpeg smoke export | Tamamlandı | 4,121 sn | 12 saniye, 2 segment |

İlk özet isteği eski MMR davranışı nedeniyle export'a ulaşmadan durduruldu. Analiz cache'i
korundu. Optimize edilmiş seçim ve kısa gerçek export ayrı ayrı başarıyla doğrulandı.
Sunucu yeniden başladıktan sonra kullanıcının seçtiği asıl hedef süreyle özet isteği tekrar
çalıştırılacak; cache-hit toplam süresi o çalışmadan sonra ayrıca kaydedilecek.

## 7. Tamamlanan çalışma — CineSum v3.5 Opsiyonel Narrative RAG + LLM

> Durum: Kod, entegrasyon, yerel benchmark ve minimal canlı LLM testi tamamlandı
> Gerçek video kalite A/B testi: proje verisinin dış servise aktarımı için açık onay bekliyor

### 7.1. Amaç ve kapsam kararı

Samet'in 26 Ağustos 2026 tarihli SceneMind raporu incelendi. CineSum'a bütün SceneMind
analiz hattını taşımak yerine mevcut cache-first mimarinin eksik anlatı katmanı alınacak.

Alınacak parçalar:

1. Videoya özel, yerel ve kalıcı SQLite RAG deposu.
2. Mevcut yerel seçimi güvenli taban kabul eden tek çağrılık LLM anlatı incelemesi.
3. LLM ve yerel puanı birleştiren hibrit değer.
4. Hibrit değeri süre ve zamansal dağılım garantisine sokan ikinci yerel seçim.
5. Servis/anahtar/JSON/timeout sorunlarında yerel sonucu koruyan fallback.
6. Aynı aday kümesi için uzak çağrıyı tekrarlamayan LLM sonuç cache'i.
7. Kullanıcıya gösterilebilen kısa Türkçe seçim gerekçeleri.

Şimdilik alınmayacak parçalar:

- Her cut için LLaVA/BLIP caption üretimi
- AudioSet AST
- YuNet/SFace yüz ve karakter analizi
- MFCC konuşmacı kümeleme
- Bütün embedding'leri ikinci kez üretme
- Her cut için ayrı LLM çağrısı
- Zorunlu bulut bağımlılığı

Bu kararın nedeni CineSum'ın ilk analizden sonraki hızlı tekrar özetleme avantajını
korumaktır.

### 7.2. Sürüm sırası ve bağımlılık

Uygulama sırası:

```text
v3.4 kelime/cümle sınırı hizalama
          |
          | konuşma güvenli gerçek segment maliyeti
          v
v3.5 yerel RAG + Paralon anlatı reranking
          |
          v
v3.6 opsiyonel diarization
```

v3.4 önce tamamlanacak. İkinci Knapsack ham shot süresini değil konuşma güvenli segment
süresini maliyet olarak kullanacağı için cümle sınırı düzeltmesi v3.5'in ön koşuludur.

### 7.3. Çalışma modları

| Mod | Yerel seçim | RAG | LLM | Hedef |
|---|---|---|---|---|
| `local` | Evet | Hayır | Hayır | En düşük gecikme, mevcut davranış |
| `rag_llm` | Evet | Evet | Tek çağrı | Daha güçlü anlatı ilişkileri |

İlk yayında varsayılan mod `local` kalacak. Arayüzde `Anlatıyı AI ile iyileştir`
seçeneği açıldığında `rag_llm` çalışacak. LLM kullanılamazsa sonuç otomatik olarak
`local` moda düşecek.

### 7.4. Faz 1 — Yapılandırma ve secret güvenliği

> Durum: Tamamlandı — 26 Ağustos 2026

Planlanan dosyalar:

```text
.env.example
requirements-llm.txt
src/core/llm_config.py
```

Davranışlar:

- `.env` yalnız sunucu tarafında `python-dotenv` ile yüklenir.
- `PARALON_API_KEY`, `PARALON_BASE_URL`, `PARALON_MODEL` merkezi config'ten okunur.
- Varsayılan endpoint `https://paraloncloud.com/v1` olur.
- API anahtarı log, debug JSON, HTTP response veya hata mesajına yazılmaz.
- `.env` ve `.env.*` Git tarafından ignore edilir; yalnız `.env.example` commit edilir.
- Anahtar yoksa uygulama açılmaya devam eder ve LLM modu `disabled` olur.
- Başlangıçta yapılandırılan model canlı `/v1/models` kataloğunda kontrol edilir.
- Model bulunamazsa küçük ve hızlı bir fallback model seçilir; katalog sonucu kısa süreli
  process cache'inde tutulur.

Mevcut kontrol sonucu: proje kökünde `.env` var ve Git tarafından ignore ediliyor.

Uygulama sonucu:

- Merkezi `LLMConfig`, `.env.example` ve ayrı `requirements-llm.txt` eklendi.
- `python-dotenv` proje sanal ortamına kuruldu; mevcut `.env` anahtarı yalnız
  `enabled: true` public metadata'sıyla doğrulandı.
- Uzak bağlantıda HTTPS zorunlu, timeout 1–120 saniye aralığında ve varsayılan 60
  saniyedir.
- API key dataclass repr, public config ve test çıktılarından çıkarıldı.
- Beş config/güvenlik testi eklendi.

### 7.5. Faz 2 — Hafif ve lazy SQLite RAG

> Durum: Çekirdek uygulama tamamlandı — 26 Ağustos 2026

Planlanan dosyalar:

```text
src/rag/__init__.py
src/rag/scene_rag.py
tests/test_scene_rag.py
```

Artifact:

```text
outputs/features/rag/<video>_scene_rag.sqlite3
```

İlk sürümde cut başına yeni ağır model çalıştırılmayacak. İndeks şu hazır kaynaklardan
üretilecek:

- StoryScene JSON ve centroid CLIP embedding'leri
- Shot/StoryScene transkriptleri
- Konuşma ve ses yoğunluğu
- Başlangıç, orta ve final konumu
- StoryScene içindeki shot kimlikleri ve baskın mod

Metin retrieval vektörü deterministik ve fit gerektirmeyen 512 boyutlu hashing
unigram/bigram embedding ile üretilecek. Görsel benzerlik mevcut StoryScene CLIP centroid'i
üzerinden hesaplanacak. Başlangıç birleşimi:

```text
RAG benzerliği = metin benzerliği × 0.70
               + görsel benzerlik × 0.25
               + zamansal bağ skoru × 0.05
```

Bu ağırlıklar benchmark sonrasında ayarlanacak.

SQLite şeması en az şu bilgileri tutacak:

| Alan | İçerik |
|---|---|
| `document_id` | `story:<id>` |
| `video_alias` | Kaynak video kimliği |
| `title` | Deterministik kısa başlık |
| `content` | Transkript, mod, zaman ve shot özeti |
| `metadata_json` | StoryScene/shot kimlikleri ve puanlar |
| `text_vector` | Normalize Float32 BLOB |
| `visual_vector` | Mevcut CLIP centroid BLOB |
| `schema_version` | RAG şema sürümü |
| `source_hash` | Video/StoryScene artifact hash'i |

RAG opsiyonel artifact olacak; `analysis_status.ready` sonucunu false yapmayacak. Yeni
videoda StoryScene sonrasında birkaç saniyede üretilecek, eski cache'li videoda ilk
`rag_llm` isteğinde lazy oluşturulacak. Mevcut yedi analiz aşaması yeniden çalıştırılmayacak.

Uygulama sonucu:

- Standart kütüphane SQLite ile atomik indeks üretimi, schema/source hash kontrolü ve
  lazy reuse eklendi.
- Deterministik 512 boyutlu hashing unigram/bigram metin vektörü ile mevcut StoryScene
  görsel centroid'i birlikte kullanılıyor.
- Retrieval sonucu metin, görsel ve zamansal skor kırılımını ayrı ayrı döndürüyor.
- SQLite dosyaları runtime artifact olarak Git dışında tutuluyor.
- Üç indeks/retrieval/cache testi eklendi.

`video58avengers` ölçümü:

| Ölçüm | Sonuç | Hedef |
|---|---:|---:|
| 1.474 StoryScene lazy indeks build | 0,172 sn | ≤5 sn |
| Cache doğrulama/reuse | 0,006 sn | ≤1 sn |
| Tek retrieval, ilk ölçüm | 36,6 ms | ≤100 ms |

### 7.6. Faz 3 — Sınırlı anlatı aday havuzu

> Durum: Çekirdek tamamlandı — 26 Ağustos 2026

Planlanan dosya:

```text
src/llm/narrative_candidates.py
tests/test_narrative_candidates.py
```

LLM'e bütün 2.470 shot gönderilmeyecek. En fazla 32 adaylık havuz şu sırayla oluşturulacak:

1. Yerel MMR + Knapsack sonucunun core shot'ları.
2. Seçili shot'ların hemen önceki ve sonraki komşuları.
3. Yerel puanı en yüksek shot'lar.
4. Temsil edilmeyen güçlü StoryScene'lerin en iyi shot'ları.
5. Videonun sekiz zamansal diliminin temsilcileri.

Adaylar scene id ile tekilleştirilecek. Her aday için RAG'den en fazla iki uzak ilişkili
StoryScene getirilecek. Prompt'a gönderilen transcript ve context uzunluğu kesin karakter
limitleriyle kırpılacak.

Uygulama sonucu: deterministik öncelik sırası, scene-id tekilleştirme, 32 kayıt üst
sınırı, aday başına 900 karakter transcript sınırı ve embedding/word dizilerini prompt
payload'ından çıkaran serializer eklendi. İki birim test başarılıdır. RAG related-context
bağlantısı Faz 5 orkestrasyonunda yapılacaktır.

### 7.7. Faz 4 — Paralon istemcisi ve tek çağrılık reranking

> Durum: İstemci çekirdeği tamamlandı, canlı çağrı bekliyor — 26 Ağustos 2026

Planlanan dosyalar:

```text
src/llm/__init__.py
src/llm/paralon_client.py
src/llm/narrative_reranker.py
tests/test_paralon_client.py
tests/test_narrative_reranker.py
```

İstemci OpenAI uyumlu `POST /chat/completions` çağrısını yapacak. Başlangıç limitleri:

| Ayar | Değer |
|---|---:|
| Uzak çağrı sayısı | Özet isteği başına en fazla 1 |
| Aday sayısı | En fazla 32 |
| Related context | Aday başına en fazla 2 |
| Temperature | 0.10–0.15 |
| Yanıt token limiti | Yaklaşık 1.500–2.000 |
| Toplam timeout | 60 saniye |

Modelden yalnız JSON istenecek:

```json
{
  "summary": "Kısa anlatı değerlendirmesi",
  "recommendations": [
    {
      "scene_id": 42,
      "priority": 87,
      "reason": "Kurulum ile sonraki çatışma arasında bağ kuruyor",
      "context_with": [105]
    }
  ]
}
```

Fenced JSON ve etrafında kısa açıklama bulunan yanıt ayıklanacak; scene id, priority,
reason ve aday havuzu üyeliği katı biçimde doğrulanacak. Anahtar veya prompt hiçbir hata
mesajında tam olarak gösterilmeyecek.

Uygulama sonucu: OpenAI uyumlu tek chat çağrısı, `temperature=0.12`, 1.800 token yanıt
limiti ve config'teki 60 saniye timeout ile eklendi. Fenced/açıklamalı JSON ayrıştırma,
priority ve aday üyeliği doğrulama, `context_with` süzme ve sanitize edilmiş timeout/HTTP/
yanıt hata kodları hazırdır. Mock HTTP testleri tek çağrı ve secret sızıntısı olmadığını
doğrulamaktadır. Henüz kullanıcının token'ını harcayan canlı istek yapılmamıştır.

Bu noktadaki tam otomatik test sonucu **54/54 başarılıdır**.

### 7.8. Faz 5 — Hibrit puan ve ikinci yerel seçim

> Durum: Tamamlandı — 26 Ağustos 2026

Başlangıç formülü:

```text
hibrit puan = normalize yerel CineSum puanı × 0.45
            + normalize LLM anlatı önceliği × 0.55
```

LLM doğrudan segment seçmeyecek. Hibrit puanlı adaylar mevcut anlatı kotası + MMR +
Knapsack hattına yeniden gönderilecek. Böylece ikinci seçim:

- Hedef süreyi aşmayacak.
- Intro/middle/ending dağılımını koruyacak.
- Aynı StoryScene tekrar sınırını koruyacak.
- v3.4'ten gelen konuşma güvenli maliyeti kullanacak.
- Nihai segmentleri kronolojik sıraya koyacak.

Geçerli LLM önerisi olmayan adaylar tamamen silinmek yerine düşük LLM önceliğiyle havuzda
kalacak. Bu, Samet'in raporundaki “ikinci Knapsack yalnız LLM önerileri üzerinde” riskini
azaltacak.

Uygulama sonucu:

- RAG related-context aday başına iki StoryScene ile prompt payload'ına bağlandı.
- `%45` normalize yerel skor + `%55` LLM önceliği hibrit skoru uygulanıyor.
- LLM'in önermediği yerel adaylar `0.15` anlatı taban puanıyla güvenli havuzda kalıyor.
- İkinci seçim mevcut narrative quota, MMR, StoryScene tekrar sınırı ve Knapsack
  fonksiyonlarını yeniden kullanıyor.
- Maliyet, v3.4 speech-boundary hizalamasından türetilen `_selection_duration` alanıdır.
- Sağlayıcı hatasında ilk yerel `selected_shots` ve `selected_shot_ids` değiştirilmeden
  dönüyor.

### 7.9. Faz 6 — LLM sonuç cache'i

> Durum: Tamamlandı — 26 Ağustos 2026

Planlanan artifact:

```text
outputs/cache/llm/<video>/<request_hash>.json
```

Cache anahtarı:

```text
video source hash
+ pipeline/StoryScene/RAG schema version
+ kategori ve hedef süre
+ custom prompt
+ model adı ve prompt sürümü
+ aday scene id + skor hash'i
```

Aynı aday kümesi ve istek yeniden çalıştırıldığında uzak API çağrısı yapılmayacak. Cache
dosyası model özetini, doğrulanmış önerileri, latency'yi ve oluşturulma zamanını tutacak;
API anahtarını veya Authorization header'ını asla tutmayacak.

Uygulama sonucu: deterministik SHA-256 request hash, atomik JSON yazımı ve cache-hit
durumunda sıfır ağ çağrısı eklendi. Cache dosyaları Git dışında tutuluyor. Mock testte iki
aynı isteğin yalnız bir kere istemci çağırdığı doğrulandı.

### 7.10. Faz 7 — API, arayüz ve ilerleme entegrasyonu

> Durum: Tamamlandı — 26 Ağustos 2026

Planlanan değişiklikler:

```text
app.py
src/summary/summary_exporter.py
src/summary/temporal_segment_builder.py
templates/index.html
```

Özet isteğine `narrative_mode: local | rag_llm` alanı eklenecek. Arayüzde opsiyonel
`Anlatıyı AI ile iyileştir` seçeneği bulunacak.

İlerleme örneği:

```text
%68 Yerel puanlar hazırlanıyor
%70 Narrative RAG bağlamı getiriliyor
%73 Paralon anlatı değerlendirmesi
%78 İkinci süre optimizasyonu
%80–95 FFmpeg export
```

Debug/response metadata:

- `llm_selection.status`: `disabled`, `applied`, `cached`, `fallback_local`
- Sağlayıcı ve gerçek kullanılan model
- Aday, öneri ve RAG doküman sayısı
- LLM latency ve cache hit bilgisi
- Fallback sebebi
- Seçilen her sahne için Türkçe LLM gerekçesi
- Yerel, LLM ve hibrit puan kırılımı

Uygulama sonucu:

- `/api/summarize` şemasına doğrulanan `narrative_mode: local | rag_llm` eklendi.
- Varsayılan `local` bırakıldı; eski istemciler aynı davranışı sürdürüyor.
- Arayüze `Anlatıyı AI ile iyileştir` kutusu ve `uygulandı/cache/yerel fallback` durum
  rozeti eklendi.
- Category ve custom özet hatları aynı opsiyonel orkestrasyonu kullanıyor; custom hatta
  mevcut scored-scene transcript, words ve StoryScene metadata'sı taşınıyor.
- RAG, Paralon ve ikinci seçim ilerleme mesajları mevcut modal yüzdelerine bağlandı.
- API response içinde sanitize edilmiş `llm_selection` metadata'sı dönüyor.

### 7.10.1. Güncel gerçek servis doğrulaması

- Paralon `/v1/models` endpoint'i mevcut anahtarla HTTP 200 döndürdü.
- Katalog: `qwen3-3b`, `qwen3.8-27b`, `gemma3-4b-mac`.
- Yapılandırılmış `qwen3.8-27b` katalogda geçerli.
- İki sahnelik `qwen3.8-27b` ve `qwen3-3b` smoke çağrıları servis tarafında 25 saniye
  içinde yanıt başlığı üretmedi; ikisinde de `provider_timeout` güvenli fallback'i doğru
  çalıştı.
- Kullanıcının LLM'i kritik görmesi üzerine 27 Ağustos 2026'da varsayılan timeout 60
  saniyeye çıkarıldı. Aynı sentetik iki-sahne smoke isteğinde `qwen3.8-27b`, **54,659
  saniyede** `status=applied`, iki geçerli öneri ve dolu bir anlatı özeti döndürdü.
- Otomatik ikinci inference retry eklenmedi. Servis 60 saniyede yanıt vermezse ilk yerel
  seçim korunmaya devam ediyor.
- Gerçek `video1` RAG→Paralon→ikinci seçim testi, video transkript/sahne içeriğini dış
  sağlayıcıya göndereceği için açık veri aktarım onayı alınana kadar çalıştırılmadı.

### 7.11. Fallback matrisi

| Durum | Davranış |
|---|---|
| API anahtarı yok | Yerel seçim, `disabled` |
| Model katalogda yok | Uygun fallback model, yoksa yerel seçim |
| Ağ/timeout | Yerel seçim, `fallback_local` |
| HTTP 4xx/5xx | Yerel seçim, sanitize edilmiş hata kodu |
| JSON parse hatası | Yerel seçim |
| Geçersiz scene id | İlgili kayıt elenir |
| Öneri yok | Yerel seçim |
| İkinci seçim bütçeye sığmıyor | Yerel temel seçim |

Hiçbir fallback video analizini veya export'u başarısız kılmayacak.

### 7.12. Test ve kabul kriterleri

Güvenlik testleri:

1. API anahtarı log, JSON ve exception metninde görünmez.
2. `.env` Git tarafından ignore edilir.
3. İstemci tarafı HTML/JS içine secret yazılmaz.

RAG testleri:

4. SQLite indeks deterministik oluşturulur ve tekrar kullanılır.
5. Source/config hash değiştiğinde yalnız RAG indeksi yenilenir.
6. Uzak ama metinsel olarak ilişkili StoryScene doğru retrieval edilir.
7. 2.470 shot/StoryScene ölçeğinde indeks ve arama bellek sınırını aşmaz.

LLM/fallback testleri:

8. Tek özet isteğinde en fazla bir uzak çağrı yapılır.
9. Aday havuzu 32'yi aşmaz.
10. Timeout, HTTP hatası ve bozuk JSON yerel sonuca düşer.
11. Havuz dışı scene id kabul edilmez.
12. Cache hit durumunda ağ çağrısı yapılmaz.
13. Model bulunamadığında katalog fallback'i çalışır.

Seçim testleri:

14. İkinci seçim hedef süre toleransını korur.
15. Intro/middle/ending temsili korunur.
16. StoryScene tekrar limiti korunur.
17. Yerel öneri LLM tarafından hiç listelenmese bile güvenli havuzdan tamamen kaybolmaz.
18. Nihai çıktı kronolojik kalır.
19. v3.4 konuşma güvenli sınırları ikinci seçim sonrasında bozulmaz.

Performans kabul kriterleri:

| Ölçüm | Hedef |
|---|---:|
| `local` mod regresyonu | En fazla +0,5 sn veya +%5 |
| `video58avengers` lazy RAG build | Hedef ≤5 sn |
| Tek RAG retrieval | Hedef ≤100 ms |
| Yerel ikinci seçim | Hedef ≤250 ms |
| Paralon maksimum bekleme | 60 sn timeout |
| Aynı istekte cache hit | Ağ çağrısı yok, hedef ≤1 sn ek süre |

26 Ağustos uygulama sonrası ölçümleri:

| Ölçüm | Sonuç |
|---|---:|
| `video58avengers` local temporal seçim | 0,104 sn |
| Local sonuç | 7 segment / 48,9 sn |
| RAG corpus belleğe yükleme | 28,1 ms |
| 32 adet vektörize retrieval | 33,4 ms |
| 32 aday RAG + mock LLM + ikinci seçim orkestrasyonu | İlk ölçüm 1,164 sn |
| Paralon servis timeout sınırı | 60 sn, ardından local fallback |
| Tam otomatik test paketi | 57/57 başarılı |

### 7.13. Kalite değerlendirme planı

En az iki video üzerinde üç varyant karşılaştırılacak:

```text
A) Mevcut local seçim
B) Local + RAG bağlamı, LLM kapalı
C) Local + RAG + Paralon + ikinci seçim
```

Ölçümler:

- Hedef/gerçek süre farkı
- Intro/middle/ending dağılımı
- StoryScene tekrar oranı
- Soru-cevap ve çatışma-çözüm çiftlerinin birlikte seçilme oranı
- Konuşma ortasında kesim sayısı
- Ek seçim latency'si
- API cache hit oranı
- Manuel kör A/B anlatı tercihi

İlk olarak kısa `video1`, ardından gerçek uzun-video sınaması için `video58avengers`
kullanılacak. LLM ağırlığı ve aday sayısı bu sonuçlara göre ayarlanacak.

### 7.14. Uygulama teslim sırası

1. v3.4 speech-safe boundary ve testleri.
2. Config + `.env.example` + secret testleri.
3. SQLite RAG builder/retrieval ve lazy cache.
4. Aday havuzu üretimi.
5. Paralon istemcisi ve katı JSON doğrulama.
6. Hibrit puan ve ikinci yerel seçim.
7. LLM response cache ve fallback matrisi.
8. API/UI entegrasyonu.
9. `video1` birim/entegrasyon doğrulaması.
10. `video58avengers` performans ve cache benchmark'ı.
11. A/B kalite raporu ve varsayılan ayar kararı.
12. Her alt faz sonunda bu yaşayan MD dosyasının güncellenmesi.

## 8. Daha sonraki aşamalar

### CineSum v3.6 — Opsiyonel diarization

- İsteğe bağlı Pyannote diarization
- Speaker embedding correction
- Kelime-konuşmacı Viterbi eşleme
- Overlap düzeltme
- Manuel speaker override

## 9. Bilinen geçiş notları

- Eski analiz çıktılarında manifest bulunmadığı için v3.1 ile ilk çalışma ağır aşamaları bir kez yeniden çalıştırır.
- Sonraki çalışmalarda geçerli aşamalar cache'ten kullanılır.
- Pyannote, AudioSet, yüz analizi ve LLaVA henüz varsayılan hatta eklenmemiştir.
- Temporal builder v3.4 kelime/cümle hizalamasını ve güvenli bütçe sonlarını kullanmaktadır.
- Narrative RAG ve Paralon v3.5'te opsiyoneldir; mevcut local sonuç her durumda güvenli fallback'tir.
- Eski analiz cache'leri v3.5 nedeniyle geçersiz olmaz; eksik RAG artifact'i lazy üretilir.
- Paralon model kataloğu ve anahtar doğrulandı. 60 saniyelik pencerede `qwen3.8-27b`
  minimal canlı testi başarılıdır; gerçek video kalite A/B ölçümü açık veri aktarım onayı
  sonrasında yapılmalıdır.

## 10. 27 Ağustos 2026 — Dinamik LLM aday havuzu

Sabit 32 aday sınırı, kısa videolarda gereksiz prompt yükü oluştururken iki saatlik
filmlerde temsil çeşitliliğini daralttığı için video süresine bağlı hale getirildi:

| Video süresi | LLM aday üst sınırı |
|---|---:|
| 0–10 dakika | 16 |
| 10–30 dakika | 24 |
| 30–60 dakika | 32 |
| 60–90 dakika | 40 |
| 90–150 dakika | 48 |
| 150–210 dakika | 56 |
| 210 dakika üzeri | 64 |

İki saatlik bir film böylece en fazla 48 aday kullanır. Artış video süresiyle doğrusal
değildir; bütün shot'lar yerelde analiz edilmeye devam ederken uzak LLM prompt'u,
token kullanımı ve timeout riski kontrollü tutulur. Paralon istemcisindeki güvenlik üst
sınırı 64'e çıkarıldı ve sonuç metadata'sına seçilen `candidate_limit` eklendi.

Timeout davranışı değişmedi: tek Paralon çağrısı için varsayılan üst sınır 60 saniyedir.
HTTP timeout, `provider_timeout` olarak temizlenir ve orkestrasyon mevcut yerel seçime
geri döner. Dinamik süre sınırları, iki saatlik 48 aday havuzu, 64 üst sınırı ve timeout
çevirisi otomatik test kapsamına alındı.

Gerçek servis kontrolünde video verisi içermeyen 48 sentetik aday kullanıldı. Paralon
60 saniyelik HTTP penceresinde yanıtı tamamlayamadı; istemci `provider_timeout` üretti
ve toplam duvar süresi bağlantı/istemci kapanış ek yüküyle 67,452 saniye ölçüldü.
Bu sonuç nedeniyle timeout otomatik olarak yükseltilmedi: 60 saniye sonunda yerel
fallback korunuyor. Arayüz ilerleme mesajı artık aday sayısını ve ağ timeout değerini,
fallback metadata'sı da `candidate_limit` ile `timeout_sec` değerlerini gösteriyor.
48 adaylı uzun-film LLM kalitesi isteniyorsa sonraki A/B adımında 90 saniye ayrı bir
konfigürasyon olarak denenmeli; varsayılan değiştirilmeden önce gerçek başarı oranı
ölçülmelidir.

Takip eden aynı-yük testinde 90 saniye de `provider_timeout` ile sonuçlandı
(93,495 sn toplam duvar süresi). 120 saniyelik istemci penceresinde servis yaklaşık
66,306 saniyede yanıt üretti fakat doğrulama sonrasında kullanılabilir öneri kalmadı
(`no_valid_recommendations`). Bunun üzerine prompt sürümü `1.1` yapıldı: model artık
boş olmayan, önem sırasına dizilmiş ve en fazla 24 kayıt içeren bir `recommendations`
listesi üretmek zorunda. Çıktı sınırı uzun JSON üretimini azaltırken önerilmeyen adaylar
hibrit seçicideki yerel güvenli havuzda kalmaya devam eder.

48 benzersiz sentetik adayla prompt v1.1 ve 120 saniyelik test 152,927 saniye
duvar süresinden sonra `invalid_json_response` verdi. `qwen3-3b` aynı yükte
70,105 saniyede, `gemma3-4b-mac` ise 27,341 saniyede aynı hatayı verdi. Tek adaylı
ham Gemma kontrolü 10,302 saniyede HTTP 200 ve ayrıştırılabilir fenced JSON üretti;
böylece endpoint/alan uyumsuzluğu elendi, büyük yanıtta token kesilmesi ana şüphe
haline geldi.

Bu nedenle prompt sürümü `1.2` oldu ve öneri sayısı hedef özet süresine bağlandı:
yaklaşık her 15 saniyelik özet için bir anlatı çapası, en az 8, en fazla 24 ve hiçbir
zaman aday sayısından fazla değil. Örneğin 180 saniyelik özet 48 adayı inceleyebilir
fakat modelden yalnızca en iyi 12 öneriyi ister. Diğer adaylar yerel güvenli havuz ve
ikinci Knapsack tarafından değerlendirilmeye devam eder.

Ham yanıt teşhisinde gerçekçi 12-aday yükü `prompt_tokens=4096` sınırına çarptı;
başlangıçtaki JSON talimatı bağlam kırpılmasında kaybolduğu için model 2.340 karakter
İngilizce düz metin döndürdü. Çözüm olarak uzak payload aday başına 300 karakter
transkript ve iki adet 180 karakter RAG bağlamıyla sınırlandı. Her Paralon isteği en
fazla 12 aday taşıyor; daha büyük dinamik havuzlar zamana yayılmış 12'lik batch'lere
bölünüp en fazla iki eşzamanlı çağrıyla işleniyor. Sonuçlar yerelde birleştiriliyor,
global öneri sınırı tekrar uygulanıyor ve kısmi batch hataları metadata'da tutuluyor.

Kompakt 12-aday Gemma doğrulaması 74,660 saniyede 12/12 geçerli öneri üretti.
`gemma3-4b-mac`, kısa testte Qwen seçeneklerinden daha hızlı ve JSON çıktısında
başarılı olduğu için varsayılan model yapıldı. Prompt v1.3 ile çıktı üst sınırı 1.200
token, özel kullanıcı prompt'u 500 karakter ve batch boyutu 12 olarak belirlendi.

Canlı paralellik ölçümleri:

| Senaryo | Sonuç |
|---|---|
| 48 aday / 4 eşzamanlı batch | 63,677 sn; 4/4 geçersiz |
| 24 aday / 2 eşzamanlı batch | 24,238 sn; 1 başarılı, 1 geçersiz, 11 öneri |
| 48 aday / 2 eşzamanlı, 4 toplam batch | 44,768 sn; 2 başarılı, 2 geçersiz, 12 öneri |

Bu nedenle eşzamanlılık ikiyle sınırlandı. En az bir batch başarılıysa geçerli öneriler
yerel güvenli havuzla birlikte kullanılıyor ve sonuç `partial=true` olarak işaretleniyor.
Bütün batch'ler başarısızsa tam yerel fallback korunuyor. Kısmi uzak sonuçlar kalıcı
cache'e yazılmıyor; aynı istek daha sonra eksik batch'leri yeniden deneyebiliyor.
Bu faz sonunda projenin kendi sanal ortamında tam otomatik paket 66/66 başarılıdır.

## 11. 27 Ağustos 2026 — Türkçe Avengers önem özeti kalite düzeltmesi

### 11.1. Gözlenen sorun ve gerçek kök neden

175 saniyelik `video58avengers` önem özeti açılıştan birkaç sahne ile final
savaşının büyük bölümüne yığılmış; Captain America'nın çekici kullandığı an, Iron
Man'in belirleyici hareketi ve yeterli diyalog temsili kaçmıştı. Film Türkçe
dublajlıdır; sorun transkripsiyon dili uyuşmazlığı değildir.

Asıl hata mevcut Whisper çıktısındaki 8097,77–8387,34 saniyelerini kapsayan tek bir
anomali segmentiydi. Bu yaklaşık 4 dakika 50 saniyelik segmentte yalnızca yedi
gerçek kelime zaman damgası bulunmasına rağmen eski eşleme kodu bütün aralığı
konuşma kabul ediyordu. Bunun sonucunda yaklaşık elli savaş shot'ı aynı Türkçe
cümleyi, `speech_ratio=1.0` değerini ve yapay biçimde yüksek önem puanını aldı.

### 11.2. Uygulanan düzeltmeler

- Kelime zaman damgası varsa sahne metni ve konuşma süresi artık segmentin kaba
  başlangıç/bitişinden değil, yalnızca sahneyle gerçekten kesişen kelimelerden
  üretiliyor. Kelime damgası bulunmayan modeller için segment fallback'i korundu.
- `transcript_scene_mapping_version=2.0-word-boundaries` eklendi. Böylece eski
  Whisper ve CLIP sonuçları korunurken yalnızca audio-feature eşlemesi ile
  StoryScene katmanı otomatik olarak yenilenebiliyor.
- Mevcut Avengers cache'i pahalı analizler tekrarlanmadan onarıldı: scene,
  keyframe, CLIP, WAV ve Whisper cache hit oldu; audio features ve StoryScenes
  yeniden üretildi. İşlem yaklaşık 18 saniye sürdü.
- Hatalı savaş kümesindeki örnek sahneler `speech_ratio=1.0` yerine `0.0` aldı;
  gerçek kelimeler yalnızca zaman damgasıyla örtüşen sahneye taşındı.
- Önem segmentleri için minimum/hedef/maksimum süre 7/11/20 saniyeye çıkarıldı;
  ön bağlam 5 saniye, merge aralığı 3 saniye ve cümle sonu genişleme hakkı 3
  saniye oldu. Çok kısa cut ve yarım cümle riski azaltıldı.
- Importance seçimi film boyunca daha ince hikâye dilimlerine ayrıldı. Sahne sırası
  esas alındığı için uzun jenerik, çalışma süresinin gerçek finali gibi davranmıyor.
  Bir StoryScene'in benzer shot'ları önem aday listesinde tek temsilciye indiriliyor.
- CLIP'e filme/karaktere özel olmayan güç kazanımı, belirleyici hareket, fedakârlık
  ve dönüm noktası prompt bankası eklendi. Mutlak CLIP değerleri yerine film içi
  yüzdelik görsel olay skoru kullanılıyor.
- Yüksek görsel olay puanlı kısa bir hazırlık shot'ının hemen ardından gelen yüksek
  olay shot'ları aynı segmentte tutuluyor.
- LLM aday havuzu local shot ve komşularıyla dolmadan önce zamansal dilim ve farklı
  StoryScene temsiline yer ayırıyor. Uzun filmlerde geniş anlatı kapsamı garanti
  altına alındı.

### 11.3. Yerel doğrulama

Onarılmış veride 175 saniyelik local önem denemesi yaklaşık 176,05 saniye üretti.
Eski çıktının 122,08 saniyede kalmasına kıyasla hedef bütçe kullanımı düzeldi. Iron
Man sekansındaki 2307 numaralı shot, 9008,1–9028,1 saniyelik devamlı segmentin içine
girdi. Seçilen son örnek segmentlerin tamamı `complete_utterance=true` oldu; bütün
çıktıda tek bir eski/fallback transkript penceresi hâlâ eksik cümle olarak
işaretlenebiliyor.

Captain America/çekiç anı için genel CLIP olay sinyali belirgin biçimde yükseldi,
fakat tam sahnenin otomatik final seçimine girdiği henüz garanti edilemedi. Bunun
nedeni Paralon'a görüntü veya görsel açıklama gönderilmemesi; uzak LLM şu anda yalnızca
zaman, local puan, Türkçe transkript ve RAG metni görüyor. Sessiz bir görüntüden nesne
ve olay kimliğini kesin adlandırmak için sonraki kalite fazında şu iki katman önerilir:

1. Anında ve ucuz çözüm: kullanıcı tarafından verilen `mutlaka dahil et` metnini
   mevcut CLIP embedding'leri üzerinde arayıp bulunan zamanları zorunlu anchor yapmak.
2. Otomatik çözüm: yalnızca çeşitlendirilmiş kısa listedeki keyframe'leri küçük bir
   vision-language modele gönderip kısa görsel açıklamalar üretmek; LLM reranker'a
   transkriptle birlikte bu açıklamaları vermek. Bütün filmi yeniden analiz etmek
   yerine yaklaşık 40–80 aday işlendiği için yavaşlama kontrollü kalır.

Avengers'a özel sahne ID'si, karakter adı veya sabit timestamp üretim koduna
yazılmadı; bütün değişiklikler diğer uzun metrajlara uygulanabilen genel kurallardır.

### 11.4. Test durumu

- Whisper eşleme, anlatı seçimi, aday havuzu ve temporal builder hedef testleri:
  başarılı.
- İnternet gerektirmeyen pytest paketi: 64 başarılı.
- `analysis_pipeline` ve `summary_auto_analysis`, proje sanal ortamında unittest ile
  ayrıca 4/4 başarılı.
- `scripts/test_clip_prompts.py` doğrudan Hugging Face ağına çıkmaya çalıştığı ve
  sandbox ağ izni olmadığı için tek kalan çevresel hatadır; uygulama model cache'iyle
  yapılan gerçek Avengers CLIP/özet doğrulamaları başarılıdır.
- Komutlar `venv\\Scripts\\python.exe` ile çalıştırılmalıdır; sistem Python'ında
  `librosa` kurulu değildir.

## 12. 27 Ağustos 2026 — 300 saniyelik özet kullanıcı geri bildirimi

### 12.1. Geri bildirim sınıfları

300 saniyelik Avengers önem özetinde şu sorunlar raporlandı:

- Cümle ve diyalogların ortasında kesim.
- Birinci kişinin sorusu/konuşması gösterilip ters açıdaki kişinin cevabı gelmeden
  başka sahneye geçilmesi.
- Konuşmacının 2–3 saniyelik dramatik beklemesinin yanlışlıkla konuşma sonu sayılması.
- Diyalog içinden parçalar çıkarıldığı için anlam bütünlüğünün bozulması.
- Ant-Man'in küçülmesi/zaman olayı, Natasha'nın ölüm sonucu ve Captain America'nın
  çekici kullanması gibi hazırlık–sonuç zincirlerinin eksik kalması.
- Iron Man final sahnesinin başlangıcı seçilmesine rağmen “Ben de Iron Man'im”
  repliğinden önce kesilmesi.

### 12.2. Debug teşhisi

`video58avengers_importance_300s_segments.json` çıktısı 282,32 saniye ve 23
segmentten oluşuyordu. LLM 56 adayın beş batch'inden yalnızca ikisini başarıyla
işledi; üç batch başarısız olduğu için karar `partial=true` idi. Bu da anlatı
kapsamının yalnızca kısmi LLM görüşüyle verilmesine yol açtı.

Eski debug metadata 23 segmentin 22'sini `complete_utterance=true` göstermesine
rağmen kullanıcı algısı bunu doğrulamadı. Ölçümün yalnızca mevcut Whisper kelimelerini
bildiği, Whisper'ın kaçırdığı konuşmaları “sessizlik” sandığı tespit edildi.

Özellikle 8981–9075 saniyeleri arasında ana Whisper çıktısında hiçbir segment veya
kelime bulunmuyordu. Bu nedenle 9008,1–9028,1 aralığındaki final segmenti konuşmasız
sayılmış ve cümle koruma mekanizması devreye girememişti.

### 12.3. Uygulanan diyalog bütünlüğü değişiklikleri

- Noktalama bulunmayan konuşmada sessizlik sınırı 0,75 saniyeden 2,75 saniyeye
  çıkarıldı. Konuşmacının kısa dramatik duraklaması artık otomatik cümle sonu değil.
- `importance` cümle genişleme hakkı 8 saniyeye çıkarıldı.
- Soru işaretiyle biten utterance'tan sonra cevap en geç 5 saniye içinde başlıyorsa
  soru ve ilk cevap aynı `exchange` bloğuna alınıyor.
- Bu pencere içinde cevap yoksa, yalnızca sessiz ters-açı reaksiyonunu taşıyan kuyruk
  soru sonuna kırpılıyor (`trimmed_unanswered_question_reaction`).
- Önem segmenti maksimum süresi 20 saniyeden 30 saniyeye çıkarıldı. Uzun diyalog ve
  soru–cevap blokları sırf eski sert sınır nedeniyle parçalanmıyor.
- Dramatik bekleme ve soru–cevap davranışları için regresyon testleri eklendi.

Yeni 300 saniyelik local simülasyon yaklaşık 305,46 saniye ve 19 daha uzun segment
üretti. Soru–cevap genişletmesi gerçek veride `boundary_mode=exchange` olarak
çalıştı. Konuşmalı segmentlerden yalnızca biri eksik/bozuk kaynak ASR penceresi
nedeniyle `complete_utterance=false` kaldı.

### 12.4. Hedefli ikinci ASR deneyi

Finaldeki 8980–9080 saniyelik 100 saniye, mevcut WAV'dan ayrılıp Faster-Whisper
`small`, CUDA/float16, beam 5, `language=tr` ve `vad_filter=false` ile yeniden
okutuldu. Model yaklaşık 16 saniyede normal analizde bulunmayan şu repliği çıkardı:

```text
46,78–49,94 sn (mutlak 9026,78–9029,94): “Bende Iron Man.”
```

Mevcut özet segmenti 9028,1 saniyede bittiği için repliği gerçekten ortadan kesiyordu.
Bu deney, problemi yalnızca kesim mantığının değil ilk ASR recall'ının da oluşturduğunu
kanıtladı.

### 12.5. Sıradaki uygulama fazı

1. Importance seçimi bittikten sonra `has_speech=false` görünen fakat görsel olay
   skoru yüksek segmentleri belirle.
2. Yalnızca bu kısa pencerelerde Türkçe, VAD kapalı hedefli ikinci ASR çalıştır.
3. Bulunan kelimeleri mutlak video zamanına taşı; segment başı/sonunu tam utterance'a
   genişlet ve debug metadata'ya `targeted_asr_applied` alanı ekle.
4. Soru–cevap ve çoklu konuşma turunu tek `dialogue_exchange` nesnesi olarak süre
   optimizasyonuna sok; Knapsack tek cümleyi değil bütün alışverişi seçsin.
5. Ant-Man, Natasha ve çekiç gibi konuşmasız/görsel olaylar için çeşitlendirilmiş
   kısa listedeki keyframe'lere VLM caption üret. `setup`, `turning_point`, `payoff`
   ilişkilerini RAG belgesine ve Paralon payload'ına ekle.
6. Geçici güvenlik özelliği olarak kullanıcıya `mutlaka dahil et` metin alanı sun;
   mevcut CLIP embedding'leriyle bulunan görsel anchor'lar nihai bütçede korunmalı.

Bu sıra, bütün filmi medium Whisper veya VLM ile tekrar işlemek yerine yalnızca seçilen
kritik adayları ikinci kez analiz eder; kalite artarken ek süre kontrollü tutulur.

### 12.6. Hedefli ASR üretim entegrasyonu

Yukarıdaki ilk üç adım aynı fazda uygulandı:

- `src/summary/targeted_asr.py` eklendi.
- Importance seçimi sonrasında konuşmasız görünen ve
  `narrative_event_score >= 0.90` olan en fazla dört segment seçiliyor.
- Her aday için yalnızca 2 saniye öncesi ve 8 saniye sonrası WAV penceresi çıkarılıyor.
- Faster-Whisper `small`, beam 5, algılanan video dili ve `vad_filter=false` ile
  ikinci kez çalışıyor.
- Güvenli konuşma bulunursa segment sınırı utterance sonuna genişletiliyor;
  `targeted_asr_applied`, eski sınırlar, olay skoru ve yeni transkript debug JSON'a
  kaydediliyor.
- Hata olursa özet üretimi durmuyor; mevcut segment korunuyor.

Gerçek 300 saniyelik debug segmentleriyle entegrasyon testinde dört aday incelendi,
ikisinde kaçırılmış konuşma bulundu. Final segmenti:

```text
Önce: 9008,129–9028,129
Sonra: 9008,129–9029,949
Metin: “…ve… ben de… Iron Man.”
```

olarak düzeltildi. Temiz Python sürecinde CLIP ve Whisper model yüklemeleri dahil dört
pencerenin kontrolü 16,7 saniye sürdü. Web sunucusunda model cache'i kullanıldığında
tekrarların daha kısa olması beklenir. Hedefli ASR ve yeni speech-exchange davranışını
kapsayan hedef paket başarılıdır. İnternet gerektirmeyen ana pytest paketi 69/69,
sanal ortam gerektiren entegrasyon testleri ayrıca 4/4 başarılıdır.

Ant-Man küçülme/zaman olayı, Natasha'nın ölüm sonucu ve Captain America'nın çekici
gibi seçime hiç girmeyen görsel olaylar ASR ile çözülemez. Bunlar için kalan ana iş,
Samet'in raporundaki LLaVA/BLIP yaklaşımının bütün filme değil çeşitlendirilmiş kısa
listeye uygulanması ve görsel caption'ların RAG/Paralon kararına eklenmesidir.

## 13. 27 Ağustos 2026 — İkinci 300 saniye geri bildirimi ve sert kalite kapıları

### 13.1. Son çıktıda doğrulanan iki hata

Yeni kodun çalıştığı son debug çıktısı yaklaşık 325,18 saniyeydi. Paralon beş batch'in
üçünü tamamlamış, ikisini kaybetmiş ve sonuç yine `partial=true` olmuştu.

- 1926,18–1933,84 segmenti metadata'da `complete_utterance=false` olmasına rağmen
  nihai videoya kabul edilmişti. Metin “ise tek sorun...” ile başlayıp “...kontrol
  edip” ile bitiyordu. `complete_utterance` yalnız teşhis alanıydı; seçim kapısı değildi.
- İkinci Knapsack 2306/2307 yerine 2308 aftermath shot'ını seçmişti. Segment
  9039,827'de başladığından 9026,78–9029,94 arasındaki Iron Man repliği hedefli ASR'nin
  eski iki saniyelik geriye bakış penceresinin dışında kalmıştı.

### 13.2. Uygulanan sert kurallar

- Tam utterance maksimum 30 saniyeye sığıyorsa normal genişleme mesafesi aşılsa bile
  başlangıç ve bitiş zorla tam utterance sınırına taşınıyor.
- Buna rağmen `has_speech=true` ve `complete_utterance=false` kalan aday nihai bütçe
  optimizasyonuna hiç alınmıyor. Reddedilen sayı `rejected_incomplete_segments`
  metadata'sına yazılıyor.
- Hedefli ASR geriye bakışı 2 saniyeden 15 saniyeye çıkarıldı; aftermath shot seçilse
  bile hazırlık, replik ve belirleyici hareket yeniden taranıyor.
- Hedefli ASR adayı üst sınırı 4'ten 6'ya çıkarıldı.
- `trimmed_unanswered_question_reaction` alan soru segmentleri, konuşmalı görünseler
  dahi hedefli VAD-kapalı ASR kontrolüne giriyor. Böylece ilk geçişin kaçırdığı cevap
  bulunabiliyor.
- Kısmi Paralon sonucunda ağırlıklar local `%75`, LLM `%25` oluyor ve ilk local
  seçimdeki anchor'lara koruma bonusu veriliyor. Eksik LLM batch'leri artık sağlam
  yerel sonucu kolayca silemiyor.

### 13.3. Gerçek veri doğrulaması

Önceden ortadan kesilen kuantum/zaman konuşması yeni local simülasyonda
1914,22–1943,10 aralığında 28,88 saniyelik tam utterance olarak üretildi. Yaklaşık
310,42 saniyelik 18 segmentin hiçbirinde `complete_utterance=false` kalmadı.

Son debug çıktısındaki yalnız aftermath 2308 shot'ı kullanılarak yapılan yeni hedefli
ASR doğrulaması segmenti 9024,827 saniyesine geri genişletti ve Iron Man repliğinin
bulunduğu kaynak aralığı videoya geri aldı. Temiz süreçte kontrol 12,8 saniye sürdü.

Konuşma sınırı, aftermath geriye bakışı, hedefli ASR ve temporal seçim paketindeki
26 hedef test başarılıdır.

### 13.4. Hâlâ çözülemeyen semantik seçim problemi

Bu sert kapılar seçilmiş bir olayın/cümlenin parçalanmasını çözer; seçime hiç girmeyen
Ant-Man, Natasha sonucu ve Captain America/çekiç gibi görsel olayları keşfedemez.
CLIP prompt puanı nesne/olay kimliğini kesin açıklamaya yetmez ve mevcut Paralon
payload'ında görüntü/caption yoktur. Bundan sonraki çalışma eşik ayarlamak değil,
çeşitlendirilmiş keyframe kısa listesine VLM caption eklemek olmalıdır.

## 14. 27 Ağustos 2026 — Samet hattıyla diyalog karşılaştırması

Samet'in `cinesum samet dokuman 16.08.26.md` ve `samet 26.08.2026.md` belgeleri
baştan sona yeniden incelendi. Diyalog davranışındaki temel farklar şunlardır:

| Konu | Samet / SceneMind | Mevcut CineSum |
|---|---|---|
| Konuşma atomu | Whisper'ın kendi `start_sec/end_sec` repliği | Kelimelerden yeniden kurulan utterance |
| Sınır koruma | Cut repliğe değerse bütün repliği al | Noktalama, sessizlik, kelime ve maksimum süre birlikte karar verir |
| Tampon | Repliğin iki yanına 0,22 sn | Önceden tam kelime sınırı, tampon yoktu |
| Süre maliyeti | Speech-safe genişletilmiş süre Knapsack maliyetidir | Local tahmin + final alignment; LLM geçişinde safe duration |
| Export | FFmpeg `trim/atrim`, PTS sıfırlama, concat | Segment bazlı kesim + konuşmada da 0,15/0,25 sn audio fade |
| Görsel anlam | Her cut için LLaVA/BLIP caption | CLIP benzerliği var, doğal dil caption yok |
| Diyalog bağlamı | Kendi diyalogu iki kez + önceki/sonraki cut diyaloğu embed edilir | Kendi transkript + StoryScene/RAG bağlamı |
| Konuşmacı | MFCC tabanlı yaklaşık diarization | Konuşmacı kimliği yok |
| Özet sıkıştırması | UI minimum %5, varsayılan %22 | 300 sn / 181 dk yaklaşık %2,76 |

Son satır karşılaştırma açısından önemlidir: Samet'in arayüzü bu Avengers örneğindeki
kadar agresif bir 5 dakikalık sıkıştırmaya izin vermiyor; minimum yaklaşık 9 dakika,
varsayılan yaklaşık 40 dakikadır. Dokümandaki “konuşmayı bölmeme” doğrulaması yapay
birim testidir; gerçek uzun film kör kalite testi raporlanmamıştır. Dolayısıyla Samet
hattının kusursuz olduğu kanıtlanmış değildir, ancak sınır politikası daha basit ve
konuşma açısından daha muhafazakârdır.

Bu inceleme sonucunda CineSum'a şu davranışlar taşındı:

- Konuşma sınırlarına Samet ile aynı 0,22 saniyelik akustik ön/arka tampon eklendi.
- `has_speech=true` segmentlerde 0,15 saniye fade-in ve 0,25 saniye fade-out kapatıldı;
  ilk/son hecenin ses seviyesinin yapay biçimde bastırılması engellendi.
- Hedefli ASR ile bulunan repliklere de 0,22 saniye tampon uygulanıyor.
- Tam utterance sığıyorsa genişleme mesafesine bakılmadan atomik korunuyor; sığmayan
  eksik konuşma nihai seçime kabul edilmiyor.

Samet'ten henüz taşınmayan ve sonraki kalite fazında ele alınması gereken iki güçlü
özellik LLaVA/BLIP görsel caption ile konuşmacı/komşu konuşma bloklarıdır. Görsel
caption; Ant-Man, Natasha ve çekiç olaylarını seçime sokmak için diarization'dan daha
yüksek önceliklidir.

## 15. 27 Ağustos 2026 — Cache'li BLIP + LLaVA görsel anlam katmanı

Eski CineSum seçimi ve güvenli local fallback korunarak eksik görsel semantik katmanı
uygulandı. VLM bütün filmin 2840 keyframe'inde çalıştırılmıyor. Yalnız `rag_llm`
özetinde zaten üretilen dinamik anlatı aday havuzundan, StoryScene başına en fazla iki
kare olacak biçimde en fazla 40 temsilci seçiliyor.

### 15.1. Uygulanan mimari

- `src/core/vlm_config.py` ile VLM açma/kapama, model adları, kare sınırı, LLaVA alt
  sınırı ve 60 saniyelik yumuşak süre bütçesi merkezi hale getirildi.
- `src/features/visual_captioner.py` eklendi. Mevcut PySceneDetect keyframe'lerini
  yeniden çıkarmadan okuyor; her kareyi SHA-256 ile kimliklendiriyor.
- BLIP en fazla 40 kısa-listelenmiş kareye temel açıklama üretir.
- Daha güçlü fakat daha pahalı LLaVA-OneVision 0.5B; konuşması az, olay skoru ve yerel
  önemi yüksek en fazla 12 kareyi ayrıntılandırır.
- BLIP modeli bittikten sonra GPU'dan çıkarılıp CUDA cache temizlenir, ardından LLaVA
  yüklenir. İki model aynı anda RTX 4050 belleğinde tutulmaz.
- Sonuçlar `outputs/cache/vlm/<video>/captions.json` altında model kimliği ve görüntü
  hash'iyle kalıcı saklanır. Aynı kare/model ikinci özette tekrar çalışmaz.
- Görsel açıklama adayın RAG sorgu metnine ve Paralon payload'ındaki
  `visual_caption` alanına eklenir. Ham kare veya video Paralon'a gönderilmez.
- Paralon prompt sürümü `1.4-visual-semantics` oldu. Görsel kritik eylem, nesne, ölüm,
  zafer ve dönüm noktalarını değerlendirmesi; caption dışına taşarak kimlik
  uydurmaması açıkça istendi.
- Görsel caption Paralon cache anahtarının parçasıdır. Yeni caption üretildiğinde eski
  LLM kararı yanlışlıkla kullanılmaz.
- Model, GPU, dosya veya süre sorunu bütün özet üretimini kesmez. `partial` veya
  `fallback_no_captions` metadata'sıyla eski RAG+LLM/local seçime devam edilir.

### 15.2. Varsayılan kontrollü profil

| Ayar | Değer |
|---|---:|
| BLIP kısa liste | En fazla 40 kare |
| LLaVA zenginleştirme | En fazla 12 kare |
| StoryScene çeşitliliği | En fazla 2 kare / StoryScene |
| VLM yumuşak süre bütçesi | 60 sn |
| BLIP modeli | `Salesforce/blip-image-captioning-base` |
| LLaVA modeli | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` |

Seçilen LLaVA-OneVision 0.5B modeli Apache-2.0 lisanslı hafif sürümdür. Önce düşünülen
LLaVA-Interleave checkpoint'inin ticari kullanım kısıtı bulunduğu için varsayılan
olarak kullanılmadı.

### 15.3. Kurulum ve doğrulama

- Ayrı `requirements-vlm.txt` eklendi; mevcut GPU Torch kurulumu değiştirilmedi.
- Sanal ortama eksik `sentencepiece 0.2.2` kuruldu. Transformers 5.14.1, CUDA Torch
  2.11.0 ve Pillow 12.3.0 import kontrolünden geçti.
- RTX 4050 Laptop GPU, CUDA tarafından erişilebilir durumda doğrulandı.
- BLIP ve LLaVA model ağırlıkları henüz yerel cache'te değildir. İlk gerçek
  `rag_llm` çalışması ağırlıkları bir kez Hugging Face'ten indirecek; bu tek seferlik
  indirme 60 saniyelik normal inference beklentisine dahil değildir.
- Yeni VLM config, cache, görsel açıklama ve serializer testleri dahil hedef paket
  23/23; internet gerektirmeyen ana paket 78/78; sanal ortam analiz entegrasyonları
  4/4 başarılıdır.

Bu faz seçim sistemine görsel olayları anlayabileceği doğal dil sinyalini ekler. Yine
de tek bir keyframe olayın birkaç saniye önce/sonra gerçekleşen sonucunu kaçırabilir.
Avengers kör kalite testinden sonra gerekirse yalnız yüksek değerli 12 LLaVA adayı
için komşu keyframe çiftleri değerlendirilecektir; bütün film VLM taramasına
dönülmeyecektir.

## 16. 27 Ağustos 2026 — Olay çeşitliliği, LLM kurtarma ve çoklu-kare VLM

Avengers 300 saniyelik Importance çıktısı tekrar incelendi. Natasha'nın fedakârlık
öncesi konuşmasının seçilip düşüş sonucunun kesildiği, Kaptan Amerika–Mjolnir
bölgesinin yaklaşık 290 saniyelik hatalı Whisper zaman bloğunda metinsiz kaldığı ve
Iron Man finaline yakın `scene_id=2306` adayının ilk seçimden sonra bölgesel süre
optimizasyonunda elendiği doğrulandı. Son LLM çalışmasında 5 batch'in yalnız 2'si
başarılı olduğu için kararın büyük bölümü yerel skora geri dönmüştü.

### 16.1. Uygulanan regresyon düzeltmeleri

- Küçük harfle başlayan her Whisper parçasının otomatik olarak önceki cümlenin
  devamı sayılması kaldırıldı. Yalnız açık devam bağlaçları veya semantik olarak
  tamamlanmamış önceki ifade geriye bağlanır.
- Bölgesel Importance kotaları yalnız 90 saniye ve üzeri özetlerde uygulanır. Kısa
  özetlerin bölge başına çok küçük bütçe yüzünden boş dönmesi engellendi.
- Konuşma padding metadata'sı gerçek `0.45 sn` sabitiyle eşlendi.
- Özet algoritması sürümü `3.6.0` oldu.

### 16.2. LLM aday ve hata kurtarma profili

- Uzun metrajlarda küresel LLM güvenlik sınırı 64'ten 72 adaya çıkarıldı. 165–210
  dakikalık videolar 72 aday kullanır.
- Paralon isteği hâlâ en fazla 12 aday taşır ve en fazla 2 istek paralel çalışır.
  Bu nedenle 72 aday 6 batch/3 dalgadır; 56 adayın 5 batch/3 dalga düzenine yeni
  bir normal bekleme dalgası eklemez. Token kullanımı yaklaşık %29 artar.
- Yerel seçim kotası azaltılarak her zaman dilimindeki `narrative_event_score`
  zirveleri için ayrı `event_peak` aday kanalı eklendi. Amaç ölüm, keşif, güç
  değişimi ve sonuç hamlesi gibi kısa olayların uzun aksiyon/diyalog altında
  kaybolmasını azaltmaktır.
- Başarısız 12'li Paralon batch'i bir kez iki küçük gruba bölünerek yeniden denenir.
  Başarılı normal çalıştırmada ek istek oluşmaz. Metadata artık retry ve kurtarılan
  özgün batch sayılarını raporlar.

### 16.3. Çoklu-kare LLaVA ve caption güven kapısı

- BLIP merkez karede çalışmaya devam eder.
- Yalnız en kritik 12 LLaVA adayı için önceki, merkez ve sonraki sahnenin keyframe'i
  kronolojik olarak birlikte verilir. Prompt görünür değişimi `önce → eylem → sonuç`
  şeklinde açıklamayı ister.
- Çoklu-kare hash'i VLM cache kaydına eklendi. Eski tek-kare LLaVA caption'ları yeni
  bağlamla bir kez yenilenir, ardından tekrar cache'ten kullanılır.
- `movie trailer`, `coming to theaters` ve aşırı kelime tekrarı gibi düşük bilgi
  taşıyan BLIP metinleri LLaVA'ya güvenilir ön bilgi veya LLM'ye görsel kanıt olarak
  gönderilmez.

### 16.4. Doğrulama ve ortam notu

- İnternet gerektirmeyen test paketi `80/80` başarılıdır.
- Analiz entegrasyon testleri kod hatasından değil, aktif sistem Python'unda
  `librosa` bulunmadığı için test toplama aşamasında çalışmadı.
- Makinede RTX 4050 Laptop GPU (6 GB) görünmektedir; ancak `run_web_app.py` ile
  kullanılan `C:\Program Files\Python312\python.exe` ortamındaki PyTorch şu anda
  `2.8.0+cpu` ve `torch.cuda.is_available()` false döndürmektedir. Önceki bölümdeki
  CUDA doğrulaması farklı/önceki Python ortamına aittir. Gerçek VLM hızlandırması
  için uygulamanın CUDA Torch ve bütün proje bağımlılıklarının kurulu olduğu tek bir
  sanal ortamdan çalıştırılması gerekir.

## 17. 27 Ağustos 2026 — RTX 4050 ortamı ve ASR olay zinciri sonlandırması

- Proje kökünde izole `.venv` oluşturuldu. CUDA PyTorch `2.11.0+cu128`, torchvision,
  Faster-Whisper, librosa, Transformers ve web/analiz bağımlılıkları kuruldu.
- RTX 4050 Laptop GPU (6 GB), `torch.cuda.is_available() = true` ile doğrulandı.
- `python run_web_app.py` komutu, `.venv` mevcutsa kendisini otomatik olarak bu
  ortamda yeniden başlatır ve başlangıçta kullanılan GPU'yu yazar.
- Uzun ve seyrek Whisper bloklarını tespit eden cache'li `transcript_repair` analiz
  aşaması eklendi. Yalnız bozuk ses aralıkları 30 saniyelik parçalar halinde yeniden
  işlenir; bütün film tekrar transkribe edilmez.
- Avengers üzerindeki 4 bozuk blok (`105.06`, `83.83`, `49.12`, `289.57` saniye)
  RTX 4050 üzerinde `cuda/float16` ile onarıldı. Bloklar 35 düzgün zamanlı ASR
  parçasına dönüştü; işlem sonunda kalan anomali sayısı sıfırdır.
- Importance algoritması `3.7.0` oldu. Yüksek değerli olay adayları için en fazla
  30 saniyelik `hazırlık → eylem → sonuç` penceresi oluşturulur ve bu korumalı
  pencereler son bölgesel knapsack aşamasında elenmez.
- CUDA sanal ortamında tam paket sonucu: `88 test + 9 alt test` başarılı.

## 18. 17 Eylül 2026 — Süre aşımı, jenerik sızması ve kelime kırpılması düzeltmesi

Kullanıcı geri bildirimi: (1) hedef süre seçildiğinde özet bu süreden biraz uzun
üretiliyordu, (2) filmin açılış/kapanış jeneriği özete sızıyordu, (3) cümle
başı/sonu bazı kelimeler kesitte duyulmuyordu. Kod incelemesi üç sorunun da kök
nedenini doğruladı ve aşağıdaki düzeltmeler uygulandı.

### 18.1. Kesin süre tavanı (hard duration ceiling)

Önceden `SUMMARY_DURATION_TOLERANCE = 0.10` bütçe döngüsünde ve bölgesel Knapsack'ta
nihai çıktıya kadar +%10 aşımı doğrudan izin veriyordu; ayrıca bütçe seçiminden
sonraki komşu-segment birleştirme adımı ile importance kategorisindeki hedefli ASR
genişletmesi toplam süreyi bir daha kontrol etmeden büyütebiliyordu.

- `src/summary/speech_boundaries.py`'ye `enforce_duration_ceiling()` eklendi. Bu
  fonksiyon segmentleri kronolojik gezip kümülatif süreyi hedefin üzerine
  çıkarmayan tek yetkili merci olur: aşımın başladığı segmenti mevcut
  `find_safe_budget_end()` mantığıyla güvenli cümle/kelime/shot sınırına kırpar,
  sığmıyorsa tamamen düşürür, sonraki tüm segmentleri düşürür. Kelime/cümle
  ortasından asla kesim yapmaz.
- `src/summary/summary_exporter.py :: export_category_summary()` içinde bu fonksiyon
  hedefli ASR adımından SONRA, export'tan hemen önce çağrılır — böylece bütçe
  döngüsü, komşu birleştirme ve hedefli ASR'nin hiçbiri artık nihai süreyi
  denetimsiz büyütemez. Debug JSON'a `duration_ceiling` (uygulandı mı, kırpma
  öncesi/sonrası süre, düşürülen segment sayısı) eklendi.
- `SUMMARY_DURATION_TOLERANCE` kaldırılmadı; artık yalnızca iç aday havuzu arama
  genişliği, nihai çıktının üst sınırı değil.

### 18.2. İçerik bazlı jenerik (credits) tespiti ve sert dışlama

Eskiden jenerik için tek koruma, yalnızca 1 saatten uzun videolarda ve yalnızca
importance kategorisinde çalışan, sahne-index bazlı (gerçek video zamanına göre
değil) bir yumuşak ceza idi (`position_ratio >= 0.95` → `importance_score *= 0.15`).
Bu mekanizma açılış jeneriğini hiç kapsamıyordu ve yeterince yüksek puanlı bir
jenerik shot'ı yine de seçilebiliyordu.

- `src/scoring/scoring_engine.py`'deki mevcut kontrastif CLIP prompt + within-video
  percentile deseni jenerik için de uygulandı: "movie end credits", "opening title
  card", "blank/static text screen" prompt'ları eklendi.
- Yeni `credit_probability` sinyali, CLIP jenerik benzerliği + gerçek video-zamanı
  bazlı kenar konumu (`start_seconds / video_duration`, ilk ~%3 veya son ~%4) +
  düşük diyalog sinyalinden hesaplanıp her `scored_scene`'e eklendi.
- `src/summary/temporal_segment_builder.py :: build_temporal_segments()` başında,
  `credit_probability >= 0.65` olan shot'lar **tüm kategoriler için** (action,
  dialogue, importance, custom) aday havuzundan tamamen çıkarılıyor — skorlama,
  seçim ve LLM reranking hiçbirine girmiyor. Kaç shot'ın çıkarıldığı debug
  JSON'daki `excluded_credit_shots` alanında raporlanıyor. Eski yumuşak
  importance cezası ikincil güvenlik ağı olarak korundu.

### 18.3. Bütçe kırpmasında akustik pay kaybı

`SPEECH_PADDING_SEC = 0.45` payı normal aday segment oluşturulurken zaten
ekleniyordu, ancak bütçe kısıtı nedeniyle segment sonu `find_safe_budget_end()` ile
yeniden hesaplandığında bu pay geri eklenmiyordu — tam olarak kullanıcının
bahsettiği senaryo (hedef süreye yaklaşırken son segment(ler) kırpılıp payını
kaybediyordu). Bütçe döngüsündeki kırpma noktası artık aynı 0.45s payı, varsa
konuşma içeriyorsa, video sınırına kadar geri ekliyor; nihai süre tavanı zaten
`enforce_duration_ceiling()` tarafından garanti altında olduğu için bu pay
gerekirse orada sonradan kırpılıyor — ama gerçek kelimeler asla kırpılmıyor.

### 18.4. Doğrulama

- Özet algoritması sürümü `3.8.0` oldu.
- Yeni testler: `enforce_duration_ceiling` için 4 birim testi (hiç aşmama, güvenli
  cümle sınırından kırpma, güvenli sınır yoksa segmenti düşürme, sınırsız hedefte
  no-op), jenerik dışlama için tüm kategorilerde 1 entegrasyon testi, bütçe
  kırpmasında payın korunduğunu doğrulayan 1 entegrasyon testi.
- Tam otomatik test paketi (`.venv`, pytest): **94 test + 9 alt test başarılı**
  (önceki 88'den +6).

## 19. 1 Ekim 2026 — Mimari incelemesi, CUDA doğrulaması ve üç perde planı

### Kullanıcı hedefi ve yetki

Kullanıcı sinematografik recap için üç perde dağılımı, zorunlu dönüm noktaları, duygu/ses/ana karakter sinyalleri istedi; önce plan çıkarılıp onay beklenmesini belirtti. Sonraki talebi planın Markdown'a yazılması ve tüm çalışma bilgisinin başka bir ajana devredilebilir biçimde belgelenmesiydi. Bu aşamada algoritma kodu değiştirilmedi; uygulama onayı bekleniyor.

### Tamamlanan inceleme

- Git geçmişi ve mevcut çalışma ağacı incelendi. Son görülen commit `32728ba0`; çok sayıda kaynak, test ve cache değişikliği zaten commit edilmemiş durumdaydı ve korundu.
- Skorlama, aday oluşturma, ilk/son seçim, LLM reranking, model yapılandırmaları, analiz manifesti ve son debug kayıtları okundu.
- İlk seçimin 10, son dengelemenin 6 bölge kullandığı; konumun çoğunlukla sahne sırasına dayandığı; mevcut korumanın nihai bütçe kırpması boyunca mutlak olmadığı belirlendi.
- Mevcut NLP'nin Whisper, TF-IDF, hash metin vektörleri ve konuşma sınırı kuralları olduğu; ayrı duygu/protagonist/crescendo katmanlarının bulunmadığı kaydedildi.

### Bugün doğrulanan ortam

`.venv/Scripts/python.exe` üzerinden PyTorch `2.11.0+cu128`, CUDA `12.8`, RTX 4050 Laptop GPU, `cuda.is_available()=true` ve CTranslate2 CUDA cihaz sayısı 1 doğrulandı. GPU üzerinde tensor hesabı başarıyla tamamlandı. Güvenli public config LLM/VLM enabled=true ve uzak model `gemma3-4b-mac` gösterdi. Sırlar yazdırılmadı. Bugün uzak LLM çağrısı, model inference benchmark'ı veya tam test paketi çalıştırılmadı.

### Son çıktıların gösterdiği sorunlar

- 17 Eylül `video58avengers_importance_190s_segments.json`: LLM `provider_http_400` nedeniyle local fallback. HTTP 400 kök nedeni henüz araştırılmadı.
- Aynı kayıtta VLM CUDA'da 29.034 sn, cache ağırlıklı partial sonuç; LLaVA `OutOfMemoryError` ve sıfır yeni LLaVA açıklaması. CUDA'nın çalışması bu bellek sorununu çözmüş sayılmaz.
- Aynı özet 190 sn hedefe karşı 170.759 sn/14 segment; 31 jenerik shot dışlanmış, hedefli ASR 2 adaydan 1'ine uygulanmış.
- 27 Ağustos action 300 sn kaydında uzak LLM 72 aday/6 başarılı grup ve retry'larla 254.834 sn sürmüş. Bu nedenle eski genel 20–40 sn LLM beklentisi güncel güvenilir süre olarak sunulmamalı.
- Eski 22 dk 48 sn ilk analiz benchmark'ı keyframe reuse içerir; sonraki onarım, LLM ve tam export bu toplamda değildir. Manifestteki 0.0826 sn transcript kaydı tam Whisper maliyeti olarak yorumlanmamalı.

### Belgeler ve devam noktası

- [Üç perde planı](CineSum_Three_Act_Implementation_Plan.md): tamamlanan incelemeler `[x] Bitirdim`; uygulama, entegrasyon ve kabul testleri `[ ]` olarak kaydedildi.
- [Güncel mimari/devir](CineSum_Current_State_and_Handoff.md): proje fikri, modül haritası, LLM/NLP/CUDA koşulları, ölçümler, bilinen sorunlar ve sonraki ajanın çalışma sırası yazıldı.
- Mevcut ana mimari belgelerine tarihsel kapsam ve güncel kaynak bağlantıları eklendi.
- Her tamamlanan işte plan, bu günlük ve devir belgesi birlikte güncellenecek. Sonraki adım uygulama onayından sonra LLM/VLM sorunlarını teşhis ederek plandaki fazlara devam etmek.

Doğrulama: Bu tur yalnız Markdown değişiklikleri içerir; belge dosyaları, yerel bağlantılar ve diff kontrol edilir. Testlerin önceki başarılı sonucu yeni bir test çalışması gibi sunulmaz.

## 20. 1 Ekim 2026 — Üç perde uygulamasının ilk dilimi

Kullanıcı planı onaylayıp uygulamaya başlama talimatı verdi. Bu bölüm tamamlanan kodu ve açık sınırlamaları kaydeder; planın bütün maddeleri bitmiş değildir.

- `src/selection/three_act.py`: Importance için ortak %20/%50/%30 bütçe ve gerçek kaynak zamanına göre %0–25/%25–75/%75–100 perde sınıflandırması eklendi. İlk seçim ve hizalama sonrası son seçim bu politikayı kullanır.
- %15–30, %45–55, %80–95 aralığında dönüm noktası adayları aranır. Metin terimi ve CLIP olay skoru kanıtı yoksa rol `unverified` kalır. Bu heuristik olayın gerçekten inciting/midpoint/climax olduğunu doğrulamaz.
- Kanıtlanan adaylar LLM seçiminden sonra havuza eklenip `hard_anchor_roles` ile segment ve final süre tavanına taşınır. Zorunlu süre bütçeyi aşarsa açık hata verilir. Son dışa aktarma debug'ına perde süreleri, hedef oranları ve anchor kapsamı eklendi.
- LLM seçiminde zenginleşen görsel açıklama/gerekçeler dönüm noktası kanıt taramasına da taşınır; tarama bütün sahne havuzunu kapsar.
- Ses özellikleri için 3 saniyelik önceki pencereye göre `audio_onset_contrast` eklendi; ses feature cache sürümü değişti. Importance skorunda anlatısal CLIP ağırlığı %34'e, düz RMS %10'a; başlangıç kontrastı %10'a ayarlandı. Müzik/diyalog ayrımı henüz yok.
- Özet algoritması 3.9.0. Yeni `tests/test_three_act.py` gerçek zaman sınıflandırması, yetersiz kanıt, final anchor koruması ve yetersiz bütçeyi sınar. `.venv/Scripts/python.exe -m pytest -q tests`: **105 test + 9 alt test geçti**. Kökten `pytest -q` ağır `scripts/test_clip_prompts.py` testinde çıktı vermeden beklediği için sonlandırıldı; `tests/` paketi tamamlandı.
- **Açık işler:** Uzak LLM HTTP 400 ve LLaVA OOM teşhisi, LLM kanıt şeması, ayrı final çözüm sahnesi, duygu NLP'si, protagonist doğrulaması, gerçek crescendo, kısa özetlerde tam anchor garantisi, perde sınırını aşan segmentlerin süre bölüştürmesi ve gerçek film A/B doğrulaması. Bu başlıklar görev planında açık bırakıldı.

Ek sağlamlaştırma: Çoklu kare LLaVA CUDA OOM verirse tek merkez kare ile bir defa yeniden dener, başarılıysa cache kaydına `single_frame_after_oom` yazar. Paralon HTTP hatasında ham yanıt gövdesini saklamadan yalnız güvenli `error.code/type` alanını hata koduna ekler. Bu iki davranış için testler eklendi; `tests/` sonucu **105 test + 9 alt test başarılıdır**. Gerçek GPU inference ve canlı Paralon çağrısıyla düzelme henüz doğrulanmadı.

## 1 Ekim 2026 — çalışan yerel ürün teslimi

- Üç perde zaman bölüştürmesi final raporda gerçek saniyeleri kullanıyor. Zorunlu sahneler ilk havuz, knapsack, LLM havuzu, segment birleştirme/hizalama ve son süre tavanından korunuyor; kanıt yoksa rol zorunlu sayılmıyor ve `narrative_constraints_unverified` yazılıyor. Geç yüksek olay zincirine sonuç görüntüsü için sınırlı uzatma eklendi.
- Sınırlı İngilizce/Türkçe sözcüksel duygu ve karar ipuçları, RMS başlangıç kontrastı ve daha anlatı odaklı skor çalışıyor. Bunlar protagonist takibi veya anlamsal olay doğrulaması sağlamıyor.
- RTX 4050 CUDA üzerinde gerçek üç kare BLIP+LLaVA açıklaması 7,96 sn'de üretildi. Avengers analiz cache tazelemesi 12,53 sn sürdü. Avengers yerel 300 sn export 90,5 sn sürdü, 299,7235 sn MP4 ve 20 segment üretti; perde payları yaklaşık %19,7/%50,8/%29,5. İkinci `video55` yerel 180 sn export 33,92 sn sürdü, 174,6951 sn MP4 ve 9 segment üretti; paylar %20,7/%49,1/%30,3. Çıktılar `outputs/summaries/three_act_validation/` altında, ayrıntılı JSON'lar `debug/` altında.
- `.venv/Scripts/python.exe -m pytest -q tests`: **111 passed, 9 subtests passed**. Kök `pytest` ağır `scripts/` model testini de topladığı için sonlandırıldı. Web uygulaması `python run_web_app.py` ile yerel olarak açılır; `local` varsayılan moddur. Bu turda tarayıcı üzerinden ayrıca uçtan uca istek yapılmadı.
- Canlı Paralon testi için sandbox dışı çalışma isteği otomatik izin incelemesinde reddedildi; gerekçe transkript türevi içeriğin doğrulanmamış dış sağlayıcıya gönderilmesiydi. Uzak `rag_llm` akışının güncel çalışırlığı doğrulanmadı. Dönüm noktalarının semantik sertifikasyonu, görünür protagonist garantisi ve insan anlatı kalite onayı da açık görev olarak bırakıldı. Kullanıcı düşük kullanım hakkı nedeniyle çalışan yerel ürünü şimdi teslim edip kalanları ertelemeyi istedi.
