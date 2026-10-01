# CineSum — Güncel Mimari ve Ajan Devir Belgesi

Son doğrulama: 1 Ekim 2026. Bu belge mevcut durumun giriş noktasıdır; tarihsel tasarım belgelerindeki hedefler otomatik olarak uygulanmış sayılmaz. Kullanıcı uygulamaya onay verdi; ilk üç perde dilimi uygulandı.

**Güncel teslim özeti (1 Ekim, son kontrol):** Kullanıcı çalışan ürünü hemen teslim edip kalan araştırmayı ertelemeyi istedi. `python run_web_app.py` yerel arayüz girişidir; varsayılan `local` modu iki gerçek filmde MP4 üretti. `outputs/summaries/three_act_validation/recap_v5.mp4` 299,72 sn (Avengers, hedef 300); `video55_recap.mp4` 174,70 sn (hedef 180). Perde payları sırasıyla yaklaşık %19,7/%50,8/%29,5 ve %20,7/%49,1/%30,3. `tests/` sonucu 111 test, 9 alt test başarılı. Her iki gerçek-film debug raporunda `narrative_constraints_unverified`: dört dönüm noktası semantik olarak sertifikalanmadı. Protagonist görsel garantisi de yok. Bu üstteki özet, aşağıdaki tarihsel “henüz yapılmadı” kayıtlarının yerine geçen güncel durumdur.

## Proje fikri

CineSum uzun videoları bir kez çok modlu analiz eder; önbellekteki özelliklerden action, dialogue, importance veya custom sorgu odaklı farklı sürelerde özet videolar üretir. Kullanıcının yeni hedefi importance özetini filmin neden–sonuç ilişkisini, karakterlerini ve final çözümünü koruyan sinematografik recap'e dönüştürmektir. Üç perde ve dönüm noktası korumasının ilk dilimi uygulandı; tam anlamsal doğrulama ve uçtan uca garanti bekliyor.

## Okuma sırası ve kaynakların rolü

1. Bu belge: güncel durum, mimari, sorunlar, devam noktası.
2. [Üç perde planı](CineSum_Three_Act_Implementation_Plan.md): açık görevler ve kabul ölçütleri.
3. [Uygulama günlüğü](CineSum_Implementation_Progress.md): tarihli tamamlanan işler ve doğrulama geçmişi.
4. [Mimari ve sunum rehberi](CineSum_Final_Sistem_Mimarisi_ve_Sunum_Rehberi.md): ayrıntılı v3.7 mimari anlatımı; güncellik notunu dikkate al.
5. [Grand Master](CineSum_AI_Grand_Master_Document.md): geniş teknik/tasarım arka planı; mevcut kodla doğrula.

## Yetki ve çalışma durumu

- Kullanıcı önce plan, ardından onayla uygulama istedi. Son talep planı Markdown'a yazmak ve tüm çalışma bilgisini güncel belgelere kaydetmektir. Bu turda yalnız dokümantasyon değiştirildi.
- Üç perde uygulamasının ilk dilimi tamamlandı; kalan işler görev planında açıktır.
- Özet algoritması kodda `3.9.0`; analiz pipeline sürümü ayrı olarak `3.1.0`. Bunları tek sürüm sanma.
- Çalışma ağacında bu oturumdan önce çok sayıda değiştirilmiş kaynak/test/cache dosyası ve yeni dosya vardı. Bunları silme, geri alma veya kendine ait sayma. Son görülen commit `32728ba0`; yerel değişiklikler bu commit'ten ileride.
- 1 Ekim uygulamasından sonra `pytest -q tests` sonucu **105 test + 9 alt test başarılı**. Kök dizindeki `scripts/test_clip_prompts.py` ağır model testi içerdiğinden kök `pytest` çağrısı sonlandırıldı; `tests/` paketi tamamlandı.

## Çalışan mimari

Giriş: `run_web_app.py` → `app.py` (FastAPI/Uvicorn); arayüz `templates/` ve `static/`. Başlatıcı `.venv/Scripts/python.exe` varsa o ortamda yeniden başlatır. API varsayılan narrative_mode: `local`.

| Aşama | Uygulama | Cihaz / işlev |
|---|---|---|
| Kamera kesmeleri | `src/scene_detection/pyscenedetect_runner.py` | CPU, PySceneDetect adaptive |
| Temsil kareleri | `src/scene_detection/keyframe_extractor.py` | Shot'lar için keyframe |
| Görsel vektörler | `src/features/clip_extractor.py` | CLIP ViT-B/32; CUDA varsa GPU |
| Ses çıkarma / özellikler | `src/audio/audio_extractor.py` | FFmpeg + CPU RMS enerji |
| Konuşma metni | `src/audio/whisper_transcriber.py` | Faster-Whisper; balanced: small, beam 3, CUDA float16, kelime zamanları |
| Anomalileri onarma | `src/audio/transcript_repair.py` | Yalnız bozuk bloklarda Whisper |
| Hikâye blokları | `src/story/story_scene_builder.py` | Görsel benzerlik + TF-IDF metin + konuşma ve zaman devamlılığı |
| Analiz/cache yönetimi | `src/core/analysis_pipeline.py`, `feature_manifest.py` | Aşama/ayar hash'i, artifact ve süre kontrolü |
| Yerel puanlama | `src/scoring/scoring_engine.py` | CLIP, ses, konuşma, konum ve jenerik sinyalleri |
| Yerel seçim | `src/selection/narrative_selector.py` | Bölge kotaları + çeşitlilik/MMR + knapsack |
| Bağlam ve segmentler | `src/summary/temporal_segment_builder.py` | Olay zincirleri, konuşma hizalama, birleştirme ve son bölgesel seçim |
| Özet dışa aktarma | `src/summary/summary_exporter.py` | Hedefli ASR, süre tavanı, FFmpeg libx264 (CPU) |

Veri akışı: video → shot/keyframe + ses/transkript → CLIP/ses özellikleri → StoryScene → cache → seçim → güvenli konuşma sınırları → MP4.

## İsteğe bağlı RAG + LLM kolu

`narrative_mode=rag_llm` seçildiğinde `src/llm/narrative_reranker.py` devreye girer:

1. Yerel seçimi güvenli fallback olarak saklar.
2. `src/rag/scene_rag.py`, SQLite içindeki StoryScene metin/görsel vektörlerinden ilgili bağlamı getirir. Metin vektörleri sözcük/ikili sözcük hash'idir; ayrı bir büyük dil modeli değildir.
3. `narrative_candidates.py` süreye bağlı aday havuzu kurar; uzun Avengers için üst sınır 72.
4. `visual_captioner.py`: BLIP en fazla 40 kare, LLaVA-OneVision 0.5B en fazla 12 adayda çoklu kare açıklaması. CUDA seçilir; cache kullanılır; 60 sn yumuşak bütçe kesin toplam süre garantisi değildir.
5. `paralon_client.py`: uzak Paralon API üzerinden `gemma3-4b-mac`; 12 aday/istek, en fazla 2 paralel istek, hatalı grup için parçalayarak yeniden deneme.
6. Tam yanıtta yerel %45 + LLM %55; kısmi yanıtta yerel %75 + LLM %25. Seçim yine yerel optimizasyonla yapılır. Hata varsa local fallback.

LLM bütün videoyu izlemez; aday transkriptleri, görsel açıklamalar ve getirilen bağlamı değerlendirir. 60 sn timeout istek başınadır; bütün grupların toplamı birkaç dakika sürebilir.

Yapılandırma: `src/core/llm_config.py`, `vlm_config.py`, `pipeline_config.py`; ortam değişkeni adları `.env.example` üzerinden incelenebilir. `.env` içindeki sırları belgelere/loglara kopyalama. 1 Ekim'de güvenli public config üzerinden LLM ve VLM enabled=true doğrulandı; uzak API çağrısı yapılmadı.

## Mevcut NLP ve eksikler

Var: konuşma tanıma, TF-IDF benzerliği, hash metin vektörleri, kelime/noktalama/sessizlik tabanlı sınırlar, isteğe bağlı LLM anlatı değerlendirmesi.

Henüz yok: özel duygu değişimi modeli, doğrulanmış protagonist görünürlüğü/takibi, müzik/diyalog ayrımı ve tüm son işlemler boyunca zorunlu turning-point garantisi. 3.9.0'da tek üç perde politikası ve 3 saniyelik sessizlik sonrası RMS kontrastı eklendi. Kontrast dramatik önem veya crescendo kanıtı değildir.

## 1 Ekim 2026 uygulama dilimi: v3.9.0

- `src/selection/three_act.py`: gerçek video zamanına göre Act I %0–25, Act II %25–75, Act III %75–100; özet bütçeleri %20/%50/%30. İçerik kanıtı yetersiz turning point `unverified` raporlanır.
- İlk importance seçimi ve hizalama sonrası segment seçimi ortak perde politikasını kullanır. Dışa aktarma raporu hedef/gerçek perde sürelerini ve anchor kapsamını debug'a yazar.
- Kanıtlanan inciting/midpoint/climax sahneleri LLM sonrasında aday havuzuna eklenir; segmentte `hard_anchor_roles` ile taşınır. Final süre tavanı zorunlu segmentleri saklar, sığmazlarsa açık hata verir.
- Ses özelliklerine `audio_onset_contrast` eklendi; importance formülünde görsel hareketlilik ve düz RMS payı azaltılıp anlatısal CLIP ve ses kontrastı payı artırıldı. Audio features config hash'i yeni sürümle geçersizleşir; gerçek filmde yeniden özellik çıkarma gerekecektir.
- Sınırlamalar: turning point kanıtı şimdilik kelime + CLIP event heuristiği; LLM kanıt şeması, final çözüm ayrı kilidi, duygu modeli ve ana karakter doğrulaması henüz yapılmadı. Kısa özetlerin son seçimi eski basit bütçe yolundan geçer. Final debug'daki perde süreleri segment orta noktasına göre hesaplanır; perdeyi aşan segmentlerin saniyeleri henüz paylaştırılmaz. Gerçek Avengers A/B testi henüz yapılmadı.
- Çoklu kare LLaVA OOM olursa tek merkez kareyle bir defa yeniden denenir ve başarılı sonuç özel cache işaretiyle saklanır. Paralon 400 teşhisi için yalnız güvenli hata `code/type` alanı korunur; ham gövde yazılmaz. İki davranış gerçek GPU/uzak servis üzerinde henüz yeniden ölçülmedi.

## GPU doğrulaması ve bilinen hatalar

1 Ekim 2026 `.venv` kontrolü: PyTorch `2.11.0+cu128`, CUDA runtime `12.8`, `torch.cuda.is_available()=true`, RTX 4050 Laptop GPU; CTranslate2 CUDA cihaz sayısı 1. CUDA tensor hesabı başarılı. Bu test bütün modellerin inference veya bellek kapasitesini doğrulamaz.

Son 17 Eylül çıktısı: `outputs/summaries/debug/video58avengers_importance_190s_segments.json`:

- Algoritma 3.8.0, rag_llm isteği; LLM `fallback_local`, neden `provider_http_400`.
- VLM partial: 40 kısa liste, 32 açıklamalı sahne, 36 cache hit, 4 yeni BLIP, 0 yeni LLaVA; 29.034 sn, CUDA; `llava:OutOfMemoryError`.
- Hedefli ASR 2 adayı inceleyip 1 segmenti güncellemiş.
- 190 sn hedefe karşı 170.759 sn çıktı, 14 segment; 31 jenerik shot'ı dışlanmış.

27 Ağustos başarılı LLM örneği: `video58avengers_action_300s_segments.json`; 72 aday, 6/6 başarılı grup, 4 retry, kurtarılan 2 özgün grup; 254.834 sn. Aynı kayıtta LLaVA OOM var. API isteğinin başarılı olması anlatı değerlendirmesinin doğruluğunu garanti etmez; gerekçeler ayrıca gözle incelenmelidir.

## Süreler: ölçüm ve beklentiyi ayır

Yaklaşık 181.2 dk Avengers için önceki tam analiz kaydı: 25 Ağustos 2026. Bugün yeni uçtan uca benchmark yapılmadı.

| Aşama | Kayıtlı süre | Koşul |
|---|---|---|
| Sahne tespiti | 1046.872 sn | CPU |
| Keyframe | 0.668 sn | Eski kareler yeniden kullanıldı; soğuk çıkarma ölçümü değil |
| CLIP | 74.867 sn | RTX 4050, batch 32 |
| Ses çıkarma | 18.807 sn | FFmpeg |
| Tam Whisper | 213.382 sn | Eski tam çalışma, small/float16 |
| Ses özellikleri | 11–13 sn | CPU |
| StoryScene | 0.6–0.8 sn | Hazır özellikler |
| İlk analiz toplamı | 1368.094 sn | Keyframe reuse dahil; sonraki LLM/onarım/export hariç |
| Transkript onarımı | 65.213 sn | Sonraki çalışma, 4 bozuk blok |
| Yerel skor/seçim | Yaklaşık 1–1.6 sn | Önceki benchmark |
| Son VLM | 29.034 sn | Cache ağırlıklı, LLaVA hatalı |
| Başarılı uzak LLM örneği | 254.834 sn | Retry dahil |
| Export smoke | 4.121 sn | Yalnız 12 sn video; 300 sn çıktı ölçümü değil |

Kaynak: `outputs/manifests/video58avengers.json`, debug JSON'lar ve uygulama günlüğü. Manifestte transcript için sonradan yazılmış `0.0826 sn` kaydı tam film transkripsiyon hızı diye kullanılmamalı. Geçerli cache'de ağır aşamalar atlanır; okuma/doğrulama, olası hedefli ASR, LLM ve kodlama maliyeti yine vardır.

## Geliştirmeye devam

1. `git status --short` ile devralınan değişiklikleri kontrol et; mevcut işleri koru.
2. Plan onayının gelip gelmediğini son kullanıcı mesajından belirle.
3. Onaydan sonra önce LLM HTTP 400 ve LLaVA OOM sorunlarını teşhis et; ardından plandaki fazları uygula.
4. Doğrulama için `.venv/Scripts/python.exe -m pytest` kullan; sonuçları gerçekten çalıştırıldıktan sonra kaydet. Uygulama `python run_web_app.py` ile başlatılabilir; bu devir turunda sunucu başlatılmadı.
5. Her tamamlanan görev için plan kutusunu işaretle, günlüğe tarihli kanıt ekle, bu belgedeki değişen mimari/durum/süreleri güncelle. Yeni ölçümlerde cache, model, cihaz, video ve hata koşullarını belirt.

Bu belgeler kanıtı olmayan “garanti”, “sorunsuz”, “anında” gibi iddiaların yerine gözlenen davranışı kaydetmelidir.

## Güncel teknik devir — son uygulama

- `src/selection/three_act.py`: gerçek kaynak zamanından %0–25/%25–75/%75–100 sınıflandırması, %20/%50/%30 özet hedefi, perde sınırını aşan saniyelerin bölüştürülmesi, metin + CLIP ile temkinli dönüm noktası kanıtı. Roller `evidence_found`, `candidate_unverified`, `unverified` durumlarını taşır; yalnız kanıtlı aday hard anchor olur.
- `src/selection/narrative_selector.py` ve LLM aday havuzu zorunlu shot ID'lerini önceden dahil eder. `temporal_segment_builder.py` rollerin merge, hizalama ve kısa/uzun özet seçiminde korunmasını sağlar. Son sürede önce isteğe bağlı segmentler çıkarılır; sert zorunlular sığmazsa açık hata verilir. Geç yüksek olay için en fazla 50 sn sonuç bağlamı korunabilir.
- `src/features/emotion_features.py` İngilizce/Türkçe sınırlı sözcüksel duygu ve karar ipuçları üretir; eğitilmiş NLP modeli veya olayın kesin semantik kanıtı değildir. `audio_onset_contrast` önceki 3 sn RMS'ine göre ani girişi ölçer; müzik/diyalog ayrımı yapmaz. `scoring_engine.py` aksiyon ve düz ses düzeyinin ağırlığını azaltıp anlatısal CLIP/duygu/ses ipuçlarını artırır. Ana karakter yüz/görünürlük takibi henüz yoktur.
- 336 piksel LLaVA girdi ve 48 token sınırı ile gerçek RTX 4050'de BLIP + LLaVA 3 kare zenginleştirmesi 7,96 sn'de çalıştı. Tek kare OOM tekrar denemesi de test edildi. Paralon HTTP hata teşhisinde ham gövde loglanmaz. Canlı Paralon doğrulaması, transkript türevi içeriğin doğrulanmamış dış hedefe gönderilmesi gerekçesiyle otomatik izin incelemesinde reddedildi; `rag_llm` bu teslimatta doğrulanmadı.
- Avengers cache yeniden işleme 12,53 sn (ağır aşamalar cache hit); 300 sn MP4 export 90,5 sn, gerçek süre 299,7235 sn ve 20 segment. Son korumalı olay zinciri 61,97 sn; final tavanı iki isteğe bağlı segmenti çıkardı. İkinci film MP4 export 33,92 sn, gerçek süre 174,6951 sn ve 9 segment. Bu süreler donanım/cache/video koşullarına bağlıdır.
- Son test: `.venv/Scripts/python.exe -m pytest -q tests` → **111 passed, 9 subtests passed**. Kök `pytest` ağır `scripts/` model testini topladığı için bu paket komutu kullanılır. Açık kalanlar plan dosyasındaki işaretlenmemiş maddelerdir. Mevcut uncommitted değişiklikleri ve üretilmiş MP4'leri koru.
