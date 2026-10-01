# SceneMind Mimarisi ve Son Geliştirmeler

Bu belge SceneMind'in güncel mimarisini, verinin sistem içinde nasıl aktığını ve
özellikle son eklenen **yerel RAG + Paralon + iki aşamalı Knapsack** seçim
mekanizmasını anlatır. Amaç, projeye yeni giren bir geliştiricinin yalnızca bu
belgeyi okuyarak ana bileşenleri ve aralarındaki sorumluluk sınırlarını
anlayabilmesidir.

## 1. Sistem ne yapıyor?

SceneMind uzun bir videoyu yalnızca sabit aralıklarla örnekleyen klasik bir
özetleyici değildir. Video önce gerçek kamera geçişlerine göre cut'lara ayrılır;
ardından her cut görüntü, konuşma, konuşmacı, ses olayı, yüz/karakter ve semantik
bağlam açısından analiz edilir.

Sistem bu çok modlu kanıtları kullanarak:

- Videonun sahne sınırlarını çıkarır.
- Her cut için görsel bir açıklama üretir.
- Konuşmaları zaman kodlarıyla metne çevirir.
- Benzer sesleri konuşmacı kümelerinde toplar.
- Tekrarlayan yüzleri anonim karakterler halinde eşleştirir.
- Müzik, bağırma, patlama ve benzeri ses olaylarını tespit eder.
- Kamera cut'larını daha büyük hikâye sahnelerinde gruplar.
- Her cut'a açıklanabilir bir önem puanı verir.
- Süre bütçesine uyan ve videonun başını, ortasını ve sonunu temsil eden bir
  sahne kümesi seçer.
- Seçimi yerel RAG bağlamı ve Paralon üzerindeki Qwen modeliyle anlatısal açıdan
  yeniden değerlendirir.
- Son kararı tekrar yerel Knapsack ile süre garantisi altına alır.
- Konuşmaları ortadan kesmeden `highlight.mp4` üretir.

## 2. Üst seviye mimari

```text
Tarayıcı
   |
   | video + hedef özet oranı
   v
Flask API / web_app.py
   |
   | tek işçili arka plan kuyruğu
   v
analyze_video() / analyzer.py
   |
   +-- OpenCV --------------------> adaptif cut'lar + ana kareler
   +-- YuNet + SFace -------------> yüzler, karakter kümeleri, portreler
   +-- FFmpeg ---------------------> 16 kHz mono WAV
   +-- Faster-Whisper ------------> zaman kodlu replikler
   +-- MFCC + kümeleme -----------> konuşmacı etiketleri
   +-- AudioSet AST --------------> önemli ses olayları
   +-- LLaVA / BLIP --------------> görsel açıklamalar
   +-- MiniLM / TF-IDF -----------> normalize metin embedding'leri
   +-- HSV histogram -------------> görsel embedding'ler
   +-- hikâye gruplama -----------> üst seviye anlatı sahneleri
   +-- puanlama -------------------> açıklanabilir cut değeri
   +-- dengeli 0/1 Knapsack ------> güvenli temel seçim
   +-- SQLite vektör RAG ---------> ilişkili cut bağlamı
   +-- Paralon Qwen --------------> anlatı öncelikleri ve gerekçeler
   +-- ikinci 0/1 Knapsack -------> bütçeyi aşmayan nihai seçim
   +-- FFmpeg trim/concat --------> highlight.mp4
   |
   v
outputs/web/jobs/<job-id>/
```

Web katmanı ağır analizi doğrudan istek iş parçacığında çalıştırmaz.
`ThreadPoolExecutor(max_workers=1)` ile işler sıraya alınır. Tek işçi kullanımı,
aynı anda birden fazla LLaVA/AudioSet işinin GPU ve RAM'i tüketmesini önler.
Tarayıcı iş durumunu düzenli aralıklarla sorgular ve `state.json` içindeki kısmi
sonuçları analiz tamamlanmadan gösterebilir.

## 3. Temel kavramlar

### Cut

Kamera kesmesiyle ayrılan en küçük analiz birimidir. Puanlama ve highlight seçimi
cut seviyesinde yapılır. Bir cut'ın başlangıç/bitiş zamanı, ana karesi, konuşması,
karakterleri, ses olayları, görsel açıklaması ve seçim gerekçesi vardır.

### Hikâye sahnesi

Arka arkaya gelen ve görüntü, metin veya karakter devamlılığı taşıyan bir ya da
birden fazla cut'ın oluşturduğu üst seviye gruptur. Hikâye sahneleri kullanıcıya
videonun olay akışını daha anlaşılır göstermek ve aynı olayın çok benzer
cut'larının özeti doldurmasını engellemek için kullanılır.

### Süre bütçesi

Kullanıcının seçtiği özet oranı video süresiyle çarpılır:

```text
süre bütçesi = toplam video süresi × özet oranı
```

Web arayüzünde oran yüzde 5 ile yüzde 25 arasındadır. Analiz katmanı güvenlik için
oranı `0.05–0.50` aralığına sıkıştırır.

## 4. Analiz hattının çalışma sırası

### 4.1 Adaptif cut tespiti

`scene_detector.py`, videoyu OpenCV ile örnekler. Küçültülmüş HSV histogramları
arasındaki Bhattacharyya uzaklığını hesaplar. Eşik sabit değildir; videonun kendi
medyan ve MAD dağılımından türetilir. Böylece hızlı kurgu ile yavaş ve karanlık
bir video aynı hassasiyetle değerlendirilmez.

Her cut için yaklaşık yüzde 55 noktasından bir ana kare alınır. Bu tercih,
geçiş/fade karelerinin açıklama modeline verilme ihtimalini azaltır.

### 4.2 Yüz ve karakter analizi

YuNet yüzleri bulur, SFace yüz embedding'lerini çıkarır. Benzer yüzler
agglomerative clustering ile birleştirilir. Sonuç gerçek kişi adı değildir;
yalnız video içi tutarlı `Oyuncu 1`, `Oyuncu 2` gibi anonim kimliklerdir.

Karakter taramasının hattın erken bölümünde yapılması önemlidir. Bu bilgi daha
sonra konuşmacı kümeleme, hikâye devamlılığı ve önem puanında tekrar kullanılır.

### 4.3 Ses ve konuşma analizi

FFmpeg ses kanalını 16 kHz mono WAV'a dönüştürür. Faster-Whisper replikleri ve
zaman kodlarını çıkarır. MFCC, delta ve delta-delta özelliklerinden üretilen ses
izleri konuşmacı kümelerinde toplanır. Görüntüdeki karakter ipuçları da mümkün
olduğunda konuşmacı ayrımına destek verir.

AudioSet AST her cut çevresindeki ses penceresinde müzik, patlama, bağırma,
silah, ağlama ve benzeri olayları arar. Bir ses bileşeni başarısız olduğunda tüm
analiz durmaz; uyarı eklenerek diğer kanallar çalışmaya devam eder.

### 4.4 Görsel açıklama

Ana web hattı varsayılan olarak yerel `llava-interleave-qwen-0.5b-hf` modelini
kullanır. Her ana kare için insan, mekân ve eylem odaklı kısa bir açıklama
üretilir. Eksik ya da geçersiz yanıt tekrar denenir; yine sonuç alınamazsa
fallback açıklaması kullanılır.

### 4.5 Embedding üretimi

Her cut için aşağıdaki metin birlikte embed edilir:

- Hikâye grubundan gelen semantik özet
- Görsel açıklama
- Cut'ın kendi diyaloğu
- Diyalog sinyalini korumak için kendi diyaloğunun ikinci kopyası
- Ses olayları
- Önceki ve sonraki cut'lardan komşu diyalog bağlamı

Normal durumda çok dilli MiniLM normalize embedding üretir. Model kullanılamazsa
en fazla 512 özellikli TF-IDF fallback'i devreye girer. Ayrıca ana kareden HSV
histogramına dayalı ucuz bir görsel embedding çıkarılır.

## 5. Açıklanabilir önem puanı

Her cut 100 puanlık altı bileşenle değerlendirilir:

| Bileşen | Azami puan | Ne ölçüyor? |
|---|---:|---|
| Konuşma yoğunluğu | 30 | Diyaloğun cut süresine göre bilgi yoğunluğu |
| Önemli ses olayı | 18 | Ses olayının güveni ve belirginliği |
| Karakter görünürlüğü | 17 | Videoda baskın karakterlerin görünmesi |
| Görsel değişim | 12 | Kamera geçişindeki görsel farklılık |
| Semantik özgünlük | 18 | Cut'ın diğer cut'lardan ayrışması |
| Anlatı konumu | 5 | Puan kartındaki konum katkısı |

Semantik özgünlük, yüzde 78 metin benzerliği ve yüzde 22 görsel benzerlikten
hesaplanır. Bir cut diğerlerine ne kadar az benziyorsa özgünlük değeri o kadar
yüksektir.

Jenerik/kredi olasılığı ayrıca hesaplanır. Açılış-kapanış konumu, kredi
kelimeleri, diyaloğun ve karakterin bulunmaması, müzik, karanlık alan ve kenar
yoğunluğu bu tahmine katkı verir. Toplam puan jenerik olasılığıyla düşürülür;
olasılık `0.72` veya üstündeyse cut seçim havuzundan çıkarılır.

Her karar `scoring_breakdown` ve `selection_reasons` içinde saklanır. Böylece
arayüz yalnızca “seçildi” demez; hangi sinyalin kaç puan verdiğini de gösterebilir.

## 6. Son geliştirmenin merkezi: Knapsack + RAG + Paralon

Yeni seçim yapısı üç katmandan oluşur:

```text
Çok modlu puanlar
      |
      v
1) Zamansal dengeli yerel Knapsack
      |  güvenli başlangıç seçimi
      v
2) SQLite RAG + Paralon anlatı incelemesi
      |  semantik bağ + LLM önceliği
      v
3) Yerel Knapsack ile yeniden paketleme
      |  kesin süre sınırı
      v
Nihai highlight cut'ları
```

Buradaki önemli tasarım kararı şudur: **Paralon nihai süre kontrolünün sahibi
değildir.** Paralon anlatı ilişkilerini yorumlar; matematiksel süre garantisini
yerel Knapsack verir.

### 6.1 İlk Knapsack: güvenli temel seçim

Önce her cut'ın maliyeti konuşma güvenli genişletilmiş süresinin yukarı
yuvarlanmasıyla hesaplanır. Değer ise çok modlu puandır.

Süre bütçesi başlangıç, orta ve son olmak üzere üç zaman dilimine bölünür. Her
dilim kendi kotası içinde 0/1 Knapsack çalıştırır. Kullanılmayan süre daha sonra
tüm kalan adaylar üzerinde ikinci bir yerel geçişte harcanır.

Bu yöntem yalnızca en yüksek puanların videonun tek bir bölümünde toplanmasını
engeller. Paralon kapalı olsa veya hata verse bile elde her zaman süre sınırına
uyan, başlangıç-orta-son dengeli bir temel seçim bulunur.

0/1 Knapsack her cut için iki karar verir:

```text
cut alınır     -> süresi bütçeden düşer, puanı hedefe eklenir
cut alınmaz    -> süre ve puan değişmez
```

Amaç toplam maliyet bütçeyi geçmeden toplam değeri en yüksek kümeyi bulmaktır.

### 6.2 Hikâye içi tekrar cezası

İlk seçimden önce aynı hikâye grubundaki cut'lar puanlarına göre sıralanır:

| Grup içi sıra | Puan kesintisi |
|---:|---:|
| 1. cut | `%0` |
| 2. cut | `%12` |
| 3. cut | `%24` |
| 4. ve sonrası | en fazla `%36` |

Amaç aynı konuşmanın veya aynı mekânın art arda gelen çok benzer planlarının
özet süresini tüketmesini önlemektir. Kesinti
`scoring_breakdown.story_repetition` içinde, etkilenen cut sayısı ise
`selection.story_penalized_cut_count` alanında tutulur.

### 6.3 Yerel SQLite RAG deposu

Her analiz işi için `scene_rag.sqlite3` oluşturulur. Bu, harici bir vektör
servisi değil, videoya özel küçük ve kalıcı bir SQLite deposudur.

`documents` tablosunda şu alanlar bulunur:

| Alan | İçerik |
|---|---|
| `id` | `cut:<id>` veya `story:<id>` |
| `kind` | `cut` ya da `story` |
| `title` | Deterministik kısa başlık |
| `content` | Görsel, konuşma, karakter, ses olayı ve komşu bağlam |
| `metadata` | Zamanlar, cut/story kimliği ve karakterler |
| `dimensions` | Vektör boyutu |
| `vector` | Float32 embedding BLOB'u |

Cut belgesi yalnız kendi içeriğini değil, önceki ve sonraki cut'ın görsel
açıklamasını da taşır. Hikâye belgesi ise gruptaki cut embedding'lerinin
normalize edilmiş ortalamasını kullanır ve grubun konuşmalarını, karakterlerini,
zamanını ve cut listesini içerir.

Arama sırasında sorgu ve kayıt vektörleri normalize edilir; cosine benzerliği
dot product ile hesaplanır. Her aday cut için kendisi dışındaki en yakın iki cut
Paralon istemine `rag_related` olarak eklenir.

Bu RAG yapısının amacı modele yeni bilgi öğretmek değildir. Modelin karar anında
videodaki semantik olarak ilişkili sahneleri görmesini sağlamaktır. Örneğin
birinci cut'ta sorulan bir sorunun cevabı videonun çok daha ilerisindeyse,
zamansal komşu olmasalar bile embedding benzerliği bu bağı görünür kılabilir.

### 6.4 Paralon'a gönderilecek aday havuzu

Bütün cut'ları modele göndermek yerine en fazla 48 cut'lık kontrollü bir havuz
oluşturulur. Havuz şunların birleşimidir:

- İlk Knapsack'ın seçtiği cut'lar
- İlk seçimin hemen önceki ve sonraki komşuları
- Algoritma puanı en yüksek 24 cut
- Her hikâye sahnesinin en güçlü cut'ı
- Videonun 12 zamansal diliminin her birindeki en güçlü cut

Jenerik olarak dışlanan cut'lar havuza alınmaz. Bu yaklaşım token maliyetini ve
yanıt karmaşıklığını sınırlar; buna rağmen hem güçlü hem zamansal olarak yayılmış
adayları korur.

Her aday kaydında şunlar Paralon'a iletilir:

- Cut kimliği ve zaman aralığı
- Süre maliyeti
- Yerel algoritma puanı
- İlk Knapsack'ta seçilip seçilmediği
- Başlık ve görsel açıklama
- Karakterler
- En fazla dört diyalog satırı
- Ses olayları
- RAG ile bulunan iki ilişkili cut ve benzerlik değerleri

Kaynak videonun kendisi Paralon'a gönderilmez. Paralon yalnızca yerel analizden
üretilen metinsel/özet kanıtı görür.

### 6.5 Paralon'un görevi

ParalonCloud üzerindeki varsayılan `qwen3.8-27b` modeli bir video kurgu editörü
rolüyle çalışır. Modelden şu anlatı hatalarını azaltması istenir:

- Kurulum olmadan sonucu seçmek
- Soru olmadan cevabı bırakmak
- Çatışma olmadan çözümü göstermek
- Aynı bilgiyi veren tekrarlı cut'ları toplamak
- Videonun başlangıç, orta ve son dengesini bozmak

Model her öneri için `cut_index`, 1–100 arası `priority`, kısa Türkçe `reason` ve
ilişkili cut'ları gösteren `context_with` alanını döndürür. Yanıt JSON olarak
beklenir. Kod fenced JSON'u veya çevresinde açıklama bulunan JSON'u da ayıklamayı
dener; geçersiz kayıtları sessizce elemek yerine doğrulama süzgecinden geçirir.

İstek OpenAI uyumlu aşağıdaki endpoint'e gider:

```text
POST {PARALON_BASE_URL}/chat/completions
```

Varsayılan yapılandırma:

```dotenv
PARALON_API_KEY=prlc_...
PARALON_BASE_URL=https://paraloncloud.com/v1
PARALON_MODEL=qwen3.8-27b
```

Sıcaklık `0.15`, maksimum yanıt uzunluğu 4096 token ve istek zaman aşımı 180
saniyedir.

### 6.6 İkinci Knapsack: Paralon ve yerel puanın ortak kararı

Paralon'un önerdiği her cut için hibrit değer hesaplanır:

```text
hibrit değer = yerel çok modlu puan × 0.42
             + Paralon anlatı önceliği × 0.58
```

Sonra yalnızca geçerli Paralon önerileri üzerinde bir kez daha yerel 0/1
Knapsack çalışır. Maliyet yine cut'ın konuşma güvenli süresidir ve bütçe ilk
seçimdekiyle aynıdır.

Bu iş bölümü iki tarafın güçlü yönünü birleştirir:

| Katman | Sorumluluk |
|---|---|
| Yerel puanlama | Görüntü, konuşma, ses, karakter ve özgünlük kanıtını ölçmek |
| İlk Knapsack | Her durumda çalışan güvenli ve zamansal dengeli başlangıç üretmek |
| SQLite RAG | Videonun uzak ama semantik olarak ilişkili parçalarını getirmek |
| Paralon | Neden-sonuç, soru-cevap ve çatışma-çözüm bağını yorumlamak |
| İkinci Knapsack | Paralon önerilerini kesin süre bütçesine yeniden yerleştirmek |

Yani model yüksek öncelik verse bile toplam süreyi aşan bir kombinasyon nihai
özete giremez. Aynı şekilde yerel olarak güçlü bir cut, anlatı değeri zayıfsa
Paralon önceliği nedeniyle başka bir cut'a yerini bırakabilir.

### 6.7 Hata ve fallback davranışı

Aşağıdaki durumlarda sistem analizi durdurmaz:

- `PARALON_API_KEY` tanımlı değilse
- Uzak servis erişilemiyorsa veya zaman aşımına uğrarsa
- Yanıt JSON olarak çözülemiyorsa
- Model havuz dışı/geçersiz cut kimlikleri döndürürse
- Hiç geçerli öneri kalmazsa
- Önerilen cut'lar bütçeye yerleşmezse

Bu durumlarda `status = fallback_knapsack` yazılır ve ilk yerel Knapsack seçimi
korunur. Sebep `selection.llm_selection.reason` içinde, kullanıcıya gösterilecek
uyarı da `warnings` içinde saklanır. Dolayısıyla bulut modeli bir iyileştirme
katmanıdır; sistemin çalışması için tek hata noktası değildir.

Başarılı durumda `selection.llm_selection` içinde sağlayıcı, model, kullanım
bilgisi, aday/öneri/doküman sayıları, model özeti ve vektör veritabanı yolu yer
alır. Seçilen cut'a ayrıca `llm_narrative` puan kırılımı ve Paralon'un Türkçe
gerekçesi eklenir.

## 7. Neden embedding iki kez hesaplanıyor?

Güncel akışta `_embed_and_select()` iki kez çağrılır:

1. İlk çağrı, cut açıklamalarından embedding üretir ve hikâye sahnelerini
   gruplar.
2. Hikâye etiketi ile açıklaması her üye cut'ın `semantic_summary` alanına yazılır.
3. Önceki seçim bayrakları ve puan kırılımları temizlenir.
4. İkinci çağrı, hikâye özetini de içeren daha zengin metni yeniden embed eder;
   asıl Knapsack + RAG + Paralon seçimi bu geçişte yapılır.

Bu yapı, RAG belgesinin yalnız ham cut açıklamasına değil, cut'ın ait olduğu
anlatı grubuna da duyarlı olmasını sağlar. Bedeli embedding aşamasının iki kez
çalışmasıdır.

## 8. Hikâye sahnesi gruplama

Komşu cut'ların devamlılık skoru şu birleşimle hesaplanır:

```text
devamlılık = metin benzerliği × 0.52
           + görsel benzerlik × 0.25
           + karakter örtüşmesi × 0.23
```

Mevcut grup en az sekiz saniyeyse ve devamlılık `0.40` altına düşerse yeni grup
başlatılabilir. Grup süresi 72 saniyeye ulaştığında da zorunlu sınır oluşur.
Jenerik ile anlatı içeriği arasındaki geçiş ayrıca doğrudan sınır kabul edilir.

Hikâye başlığı ve kısa açıklaması mevcut cut caption'ından deterministik olarak
türetilir; bu aşamada ayrı bir LLM çağrısı yapılmaz.

## 9. Konuşmayı bölmeyen highlight üretimi

Knapsack maliyetleri ve gerçek video kesimleri ham cut sınırlarını körlemesine
kullanmaz. Bir sınır bir repliğin içine düşüyorsa repliğin tamamını alacak şekilde
`0.22` saniye tamponla genişletilir. Birbirine `0.08` saniyeden yakın aralıklar
birleştirilir.

FFmpeg seçilen aralıkları `trim/atrim`, zaman damgası sıfırlama ve `concat` ile
birleştirir. Çıktı H.264 video, varsa AAC ses ve web oynatımı için `faststart`
özelliğiyle yazılır.

Not: `duration_budget_sec` Knapsack'ın tamsayı maliyet bütçesidir. Konuşma
tamamlama ve aralık birleştirme sonrasında gerçek highlight süresi
`speech_safe_duration_sec` alanından kontrol edilmelidir.

## 10. Son eklenen/güçlendirilen parçalar

Mevcut kod tabanındaki son mimari genişleme ağırlıklı olarak seçim ve anlatı
katmanındadır:

1. Videoya özel `scene_rag.sqlite3` vektör deposu eklendi.
2. Cut ve hikâye sahnesi için ayrı RAG belgeleri oluşturuldu.
3. Semantik yakın cut'ları getiren cosine araması eklendi.
4. İlk yerel seçim zamansal dengeli 0/1 Knapsack olarak güvenli taban haline
   getirildi.
5. Paralon Qwen, ilk seçimi RAG bağlamıyla anlatısal açıdan inceleyen katman
   olarak bağlandı.
6. Paralon önceliği ile yerel puanı `0.58 / 0.42` oranında birleştiren hibrit
   değer eklendi.
7. LLM önerilerini kesin süre bütçesine sokan ikinci yerel Knapsack eklendi.
8. Anahtar/servis/JSON sorununda ilk seçimi koruyan fallback davranışı eklendi.
9. Aynı hikâye sahnesindeki benzer cut'lara kademeli tekrar cezası eklendi.
10. Embedding'ler PCA/SVD ile iki boyuta indirgenerek arayüzde cut ve hikâye
    ilişkilerinin görselleştirilmesi sağlandı.
11. Seçim metadata'sına Paralon durumu, doküman sayısı, aday sayısı, süre
    dağılımı ve konuşma güvenli aralıklar eklendi.
12. RAG araması, Paralon seçiminin tekrar yerel Knapsack'a sokulması ve anahtar
    yokken fallback yapılması için birim testleri eklendi.

## 11. Üretilen dosyalar

Her işin dizini:

```text
outputs/web/jobs/<job-id>/
├── state.json
├── analysis.json
├── scene_embeddings.npy
├── scene_rag.sqlite3
├── audio.wav
├── highlight.mp4
├── frames/
└── portraits/
```

- `state.json`: Canlı/kademeli iş durumu
- `analysis.json`: Nihai taşınabilir analiz sonucu
- `scene_embeddings.npy`: Cut embedding matrisi
- `scene_rag.sqlite3`: Cut ve hikâye belgeleriyle yerel vektör deposu
- `audio.wav`: Analiz için çıkarılmış ses
- `highlight.mp4`: Nihai özet video
- `frames/`: Cut ana kareleri
- `portraits/`: Karakter kümelerinin temsilci yüzleri

## 12. Önemli seçim metadata alanları

`analysis.json` içindeki `selection` alanı seçim hattını denetlemek için temel
kaynaktır:

```json
{
  "algorithm": "Zamansal dengeli 0/1 Knapsack",
  "duration_budget_sec": 120,
  "selected_duration_sec": 117,
  "temporal_distribution_sec": [38, 41, 38],
  "excluded_credit_scenes": 2,
  "story_penalized_cut_count": 8,
  "llm_selection": {
    "status": "applied",
    "provider": "ParalonCloud",
    "model": "qwen3.8-27b",
    "candidate_count": 34,
    "recommended_count": 12,
    "document_count": 57
  },
  "speech_safe_duration_sec": 119.4,
  "speech_safe_intervals": []
}
```

`llm_selection.status` için temel değerler:

- `applied`: Paralon kararı alındı ve ikinci Knapsack uygulandı.
- `fallback_knapsack`: Paralon kullanılamadı; ilk seçim korundu.
- `disabled`: LLM seçimi bu çağrıda bilinçli olarak çalıştırılmadı. İlk hikâye
  gruplama geçişinde bu durum normaldir.

## 13. Bileşen sorumlulukları

| Dosya | Ana sorumluluk |
|---|---|
| `scene_captioner/web_app.py` | Flask API, yükleme, kuyruk, iptal ve iş geçmişi |
| `scene_captioner/analyzer.py` | Çok modlu hattın ana orkestrasyonu, puanlama ve Knapsack |
| `scene_captioner/scene_rag.py` | SQLite vektör deposu, retrieval, Paralon ve hibrit seçim |
| `scene_captioner/scene_detector.py` | Adaptif kamera kesmesi tespiti |
| `scene_captioner/audio_analysis.py` | Ses çıkarma, Whisper, konuşmacı ve ses olayları |
| `scene_captioner/face_analysis.py` | Yüz tespiti, embedding ve karakter kümeleme |
| `scene_captioner/captioners.py` | Görsel açıklama backend'leri |
| `scene_captioner/templates/index.html` | Tek dosyalı web arayüzü ve embedding haritası |
| `tests/test_core.py` | Cut, seçim, RAG, fallback, yüz ve API birim testleri |

## 14. Tasarımın güçlü tarafları

- **Yerel öncelikli:** Ana video işleme ve vektör deposu yereldir.
- **Süre garantili:** LLM kararı son söz değildir; nihai paketleme matematikseldir.
- **Hata toleranslı:** Paralon veya yardımcı modeller bozulduğunda kısmi sonuç
  korunur.
- **Açıklanabilir:** Puanlar, cezalar, LLM gerekçesi ve seçim durumu JSON'a yazılır.
- **Anlatı odaklı:** Yalnız yüksek puan değil, olay bağı ve zaman dağılımı gözetilir.
- **Videoya özel RAG:** Harici indeks altyapısı olmadan her iş kendi taşınabilir
  SQLite dosyasını üretir.

## 15. Dikkat edilmesi gereken teknik noktalar

- SQLite araması şu an tüm uygun belgeleri Python tarafında okuyup cosine skoru
  hesaplar. Video başına belge sayısı küçükken uygundur; çok büyük ölçek için
  ANN/vector extension gerekebilir.
- Paralon'a ham video gönderilmez fakat konuşma, karakter etiketi ve görsel
  açıklama gibi türetilmiş metinler gönderilir. Gizlilik politikasında bu ayrım
  açıkça belirtilmelidir.
- `context_with` modelden alınır fakat mevcut nihai seçim formülünde doğrudan bir
  zorunluluk/kısıt olarak kullanılmaz; ağırlıklı karar bilgisi olarak kalır.
- İkinci Knapsack yalnız Paralon'un geçerli biçimde önerdiği cut'lar üzerinde
  çalışır. Model önemli bir temel cut'ı hiç önermezse o cut nihai havuzdan düşer.
- Hikâye gruplama bitişik cut'larla sınırlıdır; semantik olarak ilişkili ama uzak
  cut'ları bir gruba taşımaz. Uzak ilişkiyi RAG görünür kılar.
- `narrative_position` puan kartında şimdilik sabit `0.5` ham değerdir. Gerçek
  başlangıç-orta-son dengesi `_temporally_balanced_select()` içinde uygulanır.
- Git geçmişi mevcut çalışma kopyasında bulunmadığı için bu belgedeki “son
  geliştirilenler” bölümü commit sırasına değil, güncel kodda öne çıkan yeni
  seçim/RAG entegrasyonuna göre hazırlanmıştır.

## 16. Tek cümlelik özet

SceneMind önce yerel çok modlu sinyaller ve dengeli Knapsack ile güvenli bir özet
kurar; ardından SQLite RAG ile ilişkili sahneleri Paralon'a gösterir, modelin
anlatı önceliklerini yerel puanlarla birleştirir ve son bir Knapsack geçişiyle
süre sınırını aşmadan nihai highlight'ı üretir.
