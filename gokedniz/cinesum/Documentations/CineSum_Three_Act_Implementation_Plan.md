# CineSum — Üç Perde Uygulama Planı ve Görev Takibi

Son güncelleme: 1 Ekim 2026.

Durum: Kullanıcı 1 Ekim 2026'da uygulamaya onay verdi. Üç perde ve anchor altyapısının ilk dilimi uygulandı; açık maddeler bitmiş sayılmaz.

## Amaç ve kapsam

Filmin ruhunu, neden–sonuç ilişkilerini ve kapanışını koruyan recap üretmek. Yalnız yüksek aksiyon anlarını toplamak yeterli değildir. İlk kapsam `importance` modu; action, dialogue ve custom modlarının davranışı ayrıca kararlaştırılmadan değiştirilmez. Ağır analizlerin tekrarını önleyen cache-first mimari korunur.

## Takip kuralı

- `[ ]` bekliyor; `[x] Bitirdim` tamamlandı anlamındadır.
- Bir görev ancak kodu ve gerekli doğrulaması tamamlandıktan sonra işaretlenir. Kısmi işler için alt görev açılır.
- Tamamlanan görevin altına tarih, dosyalar, doğrulama sonucu ve kalan sınırlamalar yazılır.
- Her anlamlı çalışma sonunda bu dosya, uygulama günlüğü ve güncel devir belgesi birlikte güncellenir. Planlanan özellikler çalışan özellik gibi anlatılmaz.

## 0. İnceleme ve devir

- [x] Bitirdim — Mevcut seçim ve skor akışını inceledim (1 Ekim 2026).
  - İlk seçimde 10, son dengelemede 6 bölge var; konum çoğunlukla sahne sırasından hesaplanıyor. Ortak üç perde politikası yok.
  - Mevcut korumalı olaylar bütçe/son kırpma boyunca mutlak garanti sağlamıyor.
- [x] Bitirdim — Yerel CUDA durumunu gerçek tensor işlemiyle doğruladım (1 Ekim 2026).
  - `.venv`: PyTorch 2.11.0+cu128, CUDA 12.8, RTX 4050 Laptop; CTranslate2 1 CUDA cihazı görüyor.
- [x] Bitirdim — Son LLM/VLM kayıtlarını ve süreleri inceledim (1 Ekim 2026).
  - 17 Eylül kaydı: LLM HTTP 400 ile local fallback; LLaVA OutOfMemoryError. Güncel uzak servis testi yapılmadı.
- [x] Bitirdim — Planı ve ajan devir belgesini oluşturdum, mevcut ana belgelere güncellik notu ekledim (1 Ekim 2026).
- [x] Bitirdim — Kullanıcı algoritma uygulamasına açık onay verdi (1 Ekim 2026).

## 1. Mevcut LLM/VLM sorunlarını teşhis et

- [ ] HTTP 400 yanıtının güvenli hata gövdesini ve gönderilen şemayı incele; anahtarları loglama. Sağlayıcı/model uyumsuzluğunu kanıtla, varsayma.
- [ ] Küçük canlı istekle uzak LLM kullanılabilirliğini doğrula; fallback ve kısmi yanıt davranışını test et.
- [x] Bitirdim — LLaVA girdileri 336 piksele, üretim 48 token'a sınırlandı; OOM sonrası tek kare tekrar denemesi eklendi (1 Ekim 2026).
- [x] Bitirdim — RTX 4050 üzerinde gerçek BLIP + LLaVA üç kare aday zenginleştirmesi çalıştı: 7,96 sn (1 Ekim 2026).

## 2. Tek üç perde politikası

| Perde | Kaynak filmin gerçek zamanı | Özet bütçesi | 300 sn örnek |
|---|---|---|---|
| Act I / Kurulum | %0–25 | %20 | 60 sn |
| Act II / Çatışma | %25–75 | %50 | 150 sn |
| Act III / Çözüm | %75–100 | %30 | 90 sn |

- [x] Bitirdim — `src/selection/three_act.py` ile %20/%50/%30 ortak politika oluşturuldu; ilk importance seçimi ve son segment seçimi bunu kullanıyor (1 Ekim 2026). `tests/` paketi geçti.
- [x] Bitirdim — Importance için ilk ve son perde sınıflandırması gerçek zaman damgasını kullanıyor; mevcut jenerik filtresi korunuyor (1 Ekim 2026).
- [ ] Zaman tabanını ve jenerik kenarlarını metadata'da açıkça raporla.
- [x] Bitirdim — Final rapor hizalanmış segment sürelerini perde sınırlarında paylaştırıyor (1 Ekim 2026).
- [ ] Perde temsili zorunlu, oranlar hedef ve sapma sınırı yapılandırılabilir olsun. Cümleleri tam oran uğruna kesme.
- [ ] Boş havuz/yetersiz süre durumunu raporla; sessiz kota aktarımı yapma.

## 3. Dönüm noktaları ve sonuç sahnesi

- [x] Bitirdim — Üç arama penceresi ve temkinli olay kanıtı taraması eklendi (1 Ekim 2026). Heuristik anlamsal doğrulamanın yerini tutmaz.
- [ ] Adayları transkript, StoryScene, RAG ve güvenilir görsel açıklamalarla değerlendir; zaman penceresini tek başına anlamsal kanıt sayma.
- [ ] Her aday için olay türü, kaynak sahne kimlikleri, kanıt, gerekçe ve güven puanı tut; LLM yanıt şemasını doğrula.
- [ ] Hazırlık–eylem–sonuç bağlamını koru; climax sonucu ayrı sahnedeyse çözüm sahnesini de kilitle.
- [x] Bitirdim — Yetersiz kanıt `narrative_constraints_unverified` olarak raporlanıyor (1 Ekim 2026).

## 4. Zorunlu seçim ve süre güvenliği

- [x] Bitirdim — Kanıtlı anchor ID'leri ilk aday daraltma ve LLM havuzunda korunuyor (1 Ekim 2026).
- [x] Bitirdim — Zorunlu sahneler knapsack öncesi seçiliyor, kalan bütçe isteğe bağlı adaylarla doluyor (1 Ekim 2026).
- [ ] Çakışan bağlamları birleştir; aynı saniyeyi iki kez sayma; kronolojik sırayı koru.
- [x] Bitirdim — Hizalama sonrası seçim ve final süre tavanı önce isteğe bağlı segmentleri çıkarıyor (1 Ekim 2026).
- [x] Bitirdim — Hard anchor işaretli segmentler final süre tavanında isteğe bağlı segmentlerden önce korunuyor; sığmazlarsa açık hata veriliyor (1 Ekim 2026). Önceki adımların garantisi hâlâ açık görevdir.
- [ ] Zorunlu içerik sığmıyorsa uygulanabilir asgari süreyi ve karşılanamayan kısıtları bildir; çıktı durumunu açıkça başarısız/kısıtları karşılamıyor olarak işaretle.

## 5. Çok modlu sinematografik skor

- [ ] Transkript üzerinde duygu yoğunluğu ve komşu sahnelerle duygu değişimi çıkaran NLP katmanını seç ve entegre et; dil desteği, güven ve eksik transkript davranışını belgeleyerek cache'le.
- [ ] İtiraf, ihanet, kayıp, karar ve zafer gibi olayları duygu işaretinden ayrı değerlendir; negatif duygu otomatik olarak düşük önem olmasın.
- [x] Bitirdim — 3 saniyelik önceki sese göre RMS başlangıç kontrastı eklendi ve importance skoruna %10 ağırlık verildi (1 Ekim 2026). Müzik/diyalog ayrımı yapmaz.
- [ ] Gerçek enerji eğimi/crescendo ve konuşma–müzik ayrımını ekle; örneklerle kalibre et.
- [ ] Ana karakter/grup adaylarını belirle, görsel varlığı doğrula; diyalogda ismin geçmesi görünürlük kanıtı değildir.
- [ ] Belirsiz ana karakter tespitinde kullanıcı karakter/referans düzeltmesini destekle; çok başrollü filmleri kapsa.
- [ ] Güvene bağlı, sınırlı skor çarpanları ekle; ana karakter temsili ayrıca seçim kısıtı olsun.
- [ ] Anlatısal neden–sonuç ve duygusal kırılmaya aksiyondan yüksek önem ver; ağırlıkları karşılaştırmalı testle kalibre et. Henüz kesin ağırlık belirlenmedi.
- [ ] Eksik sinyali sıfır önem sayma; geçerli sinyallerin ağırlıklarını yeniden dengele.

## 6. Entegrasyon ve gözlemlenebilirlik

- [ ] `src/scoring/scoring_engine.py`: başlangıç kontrastı eklendi; duygu/karakter/neden–sonuç sinyalleri ve tam puan kırılımı bekliyor.
- [ ] `src/llm/narrative_candidates.py`, `narrative_reranker.py`, `paralon_client.py`: olay adayları ve kanıt şeması.
- [ ] `src/selection/narrative_selector.py`: ortak politika ve zorunlu seçim.
- [ ] `src/summary/temporal_segment_builder.py`, `speech_boundaries.py`, `summary_exporter.py`: hizalama ve dışa aktarmaya kadar kısıtların korunması.
- [ ] Analiz/manifest katmanında yeni aşama ve ayar hash'leri; yalnız etkilenen cache'leri geçersiz kıl.
- [ ] Debug çıktısına hedef/gerçek perde süreleri, anchor kanıtları, karakter kapsamı, ihlaller, fallback ve aşama sürelerini ekle.
- [x] Bitirdim — İlk değişiklikle özet algoritması 3.9.0'a yükseltildi; plan, günlük ve devir belgesi güncellendi (1 Ekim 2026).

## 7. Kabul ve doğrulama

- [ ] Sessiz kurulumun yüksek aksiyon yüzünden kaybolmadığını test et.
- [ ] Anchor/finalin aday daraltma, reranking, merge, ASR genişlemesi ve son süre tavanı boyunca kaldığını test et. Son segment ve süre tavanı için ilk testler eklendi.
- [ ] Perde sınırı geçişi, çakışan anchor'lar, çok kısa bütçe ve boş perde havuzlarını test et.
- [ ] Karakter kısıtı, eksik NLP/VLM verisi ve LLM hata/kısmi yanıt davranışını test et.
- [ ] Jenerik, süre aşımı ve kelime kırpılması regresyonlarını çalıştır.
- [ ] Avengers 300 sn önce/sonra karşılaştırması: gerçek süre, perde dağılımı, olay kapsamı, diyalog bütünlüğü, işlem süreleri ve izlenebilir gerekçeler.
- [x] Bitirdim — `video55.mp4` için gerçek 180 sn hedefli, 174,70 sn MP4 üretildi (1 Ekim 2026).
- [x] Bitirdim — Son durum ve sınırlamalar README, günlük ve devir belgesine işlendi (1 Ekim 2026).

## Devam noktası

Önce [ajan devir belgesini](CineSum_Current_State_and_Handoff.md) oku. Kullanıcı uygulamaya onay verdi; açık maddeler üzerinde çalış. Mevcut çalışma ağacındaki commit edilmemiş değişiklikleri koru.

## Son teslim durumu — 1 Ekim 2026

Kullanıcı çalışan yerel ürünü şimdi teslim etmeyi, kalan araştırma/iyileştirme işlerini sonraya bırakmayı istedi. Yerel uygulama ve iki gerçek MP4 çalışıyor; `tests/` sonucu **111 passed, 9 subtests passed**. Avengers: 299,72 sn, yaklaşık %19,7/%50,8/%29,5. `video55`: 174,70 sn, %20,7/%49,1/%30,3. Dört dönüm noktası iki filmde de yeterli anlamsal kanıtla doğrulanmadı. Ana karakter görünürlüğü garantisi, gelişmiş duygu modeli, canlı Paralon kontrolü ve insan anlatı kalite onayı açık kalır. Transkript türevi veriyi uzak sağlayıcıya gönderen canlı kontrol otomatik izin incelemesinde reddedildi; tekrar denenmedi.
