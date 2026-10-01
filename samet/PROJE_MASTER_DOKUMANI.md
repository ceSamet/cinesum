# SceneMind — Proje Master Dokümanı

> Bu belge, projeye ilk kez bakan birinin kodu tek tek okumadan “Bu proje ne yapıyor, neden böyle tasarlandı, nasıl kurulur, veri hangi aşamalardan geçer, çıktılar nerede tutulur ve nereden geliştirilmeye devam edilir?” sorularını yanıtlamak için hazırlanmıştır.

## 1. Projenin kısa tanımı

**SceneMind**, bir videoyu yerel makinede analiz eden ve videonun içeriğini sahne sahne açıklanabilir hale getiren Python tabanlı bir video anlama ve özetleme uygulamasıdır.

Sistem yalnızca belirli saniye aralıklarından ekran görüntüsü almaz. Videonun kendi görsel değişim dağılımından adaptif kesme eşiği hesaplar, kamera kesmelerini bulur ve her kesiti birden fazla sinyalle inceler:

- Görüntüde ne olduğu
- Konuşulan cümleler ve zaman kodları
- Yaklaşık konuşmacı grupları
- Müzik, patlama, silah, bağırma gibi ses olayları
- Tekrarlayan yüzler ve yaklaşık ekran süreleri
- Sahnenin görsel ve anlamsal özgünlüğü
- Videonun başlangıç, orta ve son bölümlerindeki anlatı konumu

Bu sinyaller puanlanır, süre bütçesi altında en yararlı sahneler seçilir ve konuşmalar ortadan kesilmeden bir `highlight.mp4` oluşturulur.

Projenin ana hedefi: **uzun bir videoyu aranabilir, incelenebilir ve neden o sahnelerin seçildiği anlaşılabilir bir yapıya dönüştürmek.**

## 2. Bugün elimizde ne var?

Mevcut uygulama bir PoC/prototipten daha ileri, çalışan yerel bir analiz aracı durumundadır. Şu yetenekler kodda bulunur:

1. MP4, MOV, AVI, MKV, WebM ve M4V yükleme
2. En fazla 8 GB dosya kabul eden yerel Flask arayüzü
3. Videoya özgü adaptif kamera kesmesi tespiti
4. Her kesitten ana kare çıkarma
5. LLaVA veya BLIP ile görsel sahne açıklaması
6. Faster-Whisper ile zaman kodlu transkripsiyon ve otomatik dil tespiti
7. MFCC tabanlı akustik özelliklerle konuşmacı kümeleme
8. AudioSet AST ile önemli ses olayı tespiti
9. YuNet + SFace ile yüz bulma, benzer yüzleri kümeleme ve portre çıkarma
10. Kamera kesmelerini daha üst seviye “hikâye sahneleri” halinde gruplama
11. Altı bileşenli, kullanıcıya gösterilebilen sahne puanı
12. Başlangıç/orta/son dengesini koruyan 0/1 Knapsack seçimi
13. Konuşma sınırlarına duyarlı otomatik özet video
14. Canlı ilerleme, iptal, geçmiş işleri yeniden açma, arama ve filtreleme
15. Analiz sonuçlarını JSON, NumPy embedding, kareler, portreler ve video olarak saklama

## 3. Kullanıcı deneyimi

Ana kullanım yolu web arayüzüdür:

1. Kullanıcı video dosyasını seçer veya sürükleyip bırakır.
2. Özet süresini videonun `%5`–`%25` aralığında belirler. Varsayılan `%22`'dir.
3. Analiz tek işlik arka plan kuyruğuna alınır.
4. Arayüz yaklaşık 1,3 saniyede bir iş durumunu sorgular.
5. Analiz ilerlerken bulunan sahneler, replikler ve açıklamalar kademeli olarak görünür.
6. Kullanıcı kamera kesmesini veya gruplanmış hikâye sahnesini videoda oynatabilir.
7. Sonuçlar konuşmalı veya özete seçilmiş sahneler olarak filtrelenebilir; açıklama, oyuncu, konuşmacı ve replik metninde arama yapılabilir.
8. Özet video oluştuysa arayüzden indirilebilir.
9. Son altı iş geçmiş listesinden yeniden açılabilir.

Arayüz bilerek sade tutulmuştur. Web akışında görsel model `llava`, dil otomatik algılama ve ses olayı analizi açık olarak sabitlenmiştir.

## 4. Sistem mimarisi

```text
Kullanıcı / Tarayıcı
        |
        | video + özet oranı
        v
Flask web uygulaması (127.0.0.1:7860)
        |
        | tek işçili ThreadPoolExecutor
        v
analyze_video ana orkestrasyonu
        |
        +-- OpenCV adaptif kesme tespiti ------> cut sınırları + ana kareler
        +-- FFmpeg ----------------------------> 16 kHz mono WAV
        +-- Faster-Whisper --------------------> replikler + dil
        +-- MFCC + kümeleme ------------------> Konuşmacı N
        +-- AudioSet AST ----------------------> ses olayları
        +-- LLaVA / BLIP ----------------------> görsel açıklamalar
        +-- YuNet + SFace ---------------------> Oyuncu N + ekran süresi
        +-- MiniLM / TF-IDF -------------------> sahne embedding'leri
        +-- puanlama + dengeli Knapsack -------> seçilen sahneler
        +-- FFmpeg trim + concat --------------> highlight.mp4
        |
        v
outputs/web/jobs/<job-id>/
```

Buradaki iki farklı “sahne” kavramına dikkat edilmelidir:

- **Cut/kesit:** Kamera kesmesiyle ayrılan temel analiz birimi.
- **Hikâye sahnesi:** Birbirini takip eden, mekân/karakter/anlam devamlılığı olan bir veya daha fazla cut grubu.

Arayüz ikisini de gösterir; `scene_count` cut sayısını, `story_scene_count` üst seviye sahne sayısını ifade eder.

## 5. Analiz hattı: adım adım

### 5.1 Adaptif kesme tespiti

Dosya: `scene_captioner/scene_detector.py`

- Video OpenCV ile okunur ve varsayılan olarak saniyede yaklaşık 4 kare analiz edilir.
- Kareler `160x90` boyutuna küçültülür.
- HSV renk uzayında `24x16` histogram imzası çıkarılır.
- Ardışık imzalar arasındaki Bhattacharyya uzaklığı hesaplanır.
- Eşik, sabit bir sayı yerine videonun medyan ve MAD dağılımından elde edilir:

```text
threshold = clamp(median + max(0.08, 5.5 * MAD), 0.20, 0.72)
```

- Minimum cut süresi `1.25 sn`, maksimum statik cut süresi `30 sn`'dir.
- Her cut için, geçiş/fade etkisini azaltmak amacıyla yaklaşık `%55` noktasından JPEG ana kare alınır.

Bu tasarım karanlık, yavaş bir drama ile hızlı aksiyon videosuna aynı sabit eşiği uygulamama amacı taşır.

### 5.2 Ses çıkarma ve konuşma dökümü

Dosya: `scene_captioner/audio_analysis.py`

- FFmpeg videodan `16 kHz`, mono, PCM WAV çıkarır.
- Yerel `faster-whisper-small` modeli VAD ile sessiz alanları ayırır.
- `beam_size=1`, `best_of=1` ve `condition_on_previous_text=False` hız odaklı ayarlardır.
- Web uygulaması dili otomatik algılatır.
- Elde edilen her replik `start_sec`, `end_sec`, `text` ve `speaker` alanlarına sahiptir.
- Ana web analizinde Whisper CPU `int8` çalışır. Yardımcı fonksiyon CUDA istendiğinde `float16` destekler ve GPU başarısız olursa CPU'ya döner.

### 5.3 Konuşmacı kümeleme

Bu bir kimlik tanıma sistemi değil, videonun içindeki sesleri birbirinden ayıran yaklaşık diarization katmanıdır.

- Her yeterli uzunluktaki replikten MFCC, delta ve delta-delta ses özellikleri çıkarılır.
- Özellikler standartlaştırılır ve cosine uzaklıklı agglomerative clustering uygulanır.
- Farklı küme sayıları silhouette skoru, denge katkısı ve karmaşıklık cezasıyla karşılaştırılır.
- Çok küçük aykırı kümeler yeni konuşmacı sayılmaz.
- Çok kısa replikler zaman olarak en yakın kullanılabilir repliğin konuşmacısını devralır.
- Sonuçlar gerçek isim yerine `Konuşmacı 1`, `Konuşmacı 2` olarak etiketlenir.

### 5.4 Ses olayları

- Yerel Audio Spectrogram Transformer modeli kullanılır.
- Her cut'ın merkezinden en fazla 7 saniyelik bir pencere incelenir.
- Çalışma süresini sınırlamak için en fazla 96 pencere analiz edilir.
- Patlama, silah sesi, çığlık, bağırma, ağlama, kahkaha, müzik, araç sesi gibi seçili AudioSet etiketleri Türkçeleştirilir.
- Skoru en az `0.04` olan en fazla üç olay cut'a eklenir.
- Yeterli boş GPU belleği varsa model GPU'da, yoksa CPU'da çalışır.

### 5.5 Görsel açıklama

Dosya: `scene_captioner/captioners.py`

Web hattı varsayılan olarak proje içine indirilmiş `llava-interleave-qwen-0.5b-hf` kullanır. Modelin 0.5B seçilmesinin nedeni 6 GB VRAM'li dizüstü GPU hedefidir.

Her ana kare için insanları, mekânı ve eylemi tek somut cümleyle anlatan İngilizce istem kullanılır. Geçersiz, çok kısa veya ret niteliğindeki yanıt ikinci istemle tekrar denenir. Yine başarısızsa genel bir fallback açıklaması yazılır.

Desteklenen caption backend'leri:

| Backend | Kullanım | Not |
|---|---|---|
| `mock` | Test ve hafif kurulum | Gerçek içerik açıklamaz |
| `blip` | Daha hafif yerel model | Web hattı kod seviyesinde destekler |
| `blip2` | Basit CLI | Daha ağır model |
| `llava` | Ana web backend'i | Varsayılan yerel model |
| `video-llava` | Basit CLI, kare modu | 7B model; donanım ihtiyacı yüksek |
| `gemini` | Basit CLI, uzak API | `GEMINI_API_KEY` ve ayrı bağımlılık ister |

### 5.6 Yüz ve karakter analizi

Dosya: `scene_captioner/face_analysis.py`

- YuNet yüzleri bulur; SFace hizalanmış yüz embedding'i çıkarır.
- Ana kareye ek olarak video uzunluğuna/cut sayısına göre cut içinden bir veya iki ek örnek alınabilir.
- Cosine uzaklıklı agglomerative clustering benzer yüzleri gruplar.
- Profil ve cepheden görünen aynı kişinin bölünmesini azaltan ikinci bir birleştirme geçişi vardır.
- Aynı karede birlikte görülen iki yüz aynı kişi olarak birleştirilmez.
- Kalite skoru; detector güveni, yüz alanı, keskinlik ve cepheden görünümü birleştirir. En kaliteli örnek portre olur.
- En fazla 24 tekrarlayan kişi arayüzde listelenir.
- Ekran süresi, cut süresi ve kaç örnekte yüz bulunduğu kullanılarak yaklaşık hesaplanır.

`Oyuncu 1` gerçek oyuncu adı değildir. Gerçek ad için ayrı bir kadro verisi, referans yüz galerisi ve eşleştirme katmanı gerekir.

### 5.7 Embedding ve puanlama

Her cut için görsel açıklama, kendi diyaloğu, komşu diyalog bağlamı ve ses olaylarından bir metin oluşturulur.

- Normal yol: çok dilli MiniLM ile normalize embedding
- Hafif kurulum/fallback: en fazla 512 özellikli TF-IDF
- Ayrıca HSV histogramından ucuz bir görsel embedding hesaplanır.
- Semantik özgünlük `%78` metin, `%22` görsel benzerlikten türetilir.

Sahne puanı 100 puan üzerindendir:

| Bileşen | Azami puan | Anlamı |
|---|---:|---|
| Konuşma yoğunluğu | 30 | Cut içindeki diyaloğun süreye oranı |
| Önemli ses olayı | 18 | Algılanan olayın güveni |
| Karakter görünürlüğü | 17 | Videoda baskın karakterlerin varlığı |
| Görsel değişim | 12 | Cut başlangıcındaki değişim |
| Semantik özgünlük | 18 | Diğer sahnelere benzememe |
| Anlatı konumu | 5 | Mevcut uygulamada sabit `0.5` ham değer |

Jenerik olasılığı; açıklamadaki kredi kelimeleri, videodaki konum, diyalog/karakter yokluğu, müzik ve karanlık/kenar yapısından tahmin edilir. Puan bu olasılıkla çarpımsal olarak azaltılır. Olasılık `0.72` veya üstündeyse cut özet adaylarından çıkarılır.

Her cut'ın `scoring_breakdown` ve `selection_reasons` alanları tutulur. Dolayısıyla sistem yalnızca karar vermez; kararın gerekçesini de arayüzde gösterir.

### 5.8 RAG destekli özet seçimi

Toplam süre bütçesi:

```text
video süresi x kullanıcının seçtiği oran
```

Bütçe başlangıç, orta ve son için üç kotaya bölünür. Her bölümde puanı maksimize eden cut kümesi 0/1 Knapsack ile ilk aday kümesi olarak seçilir; artan bütçe tüm kalan adaylarda yeniden kullanılır.

Her analiz işi ayrıca `scene_rag.sqlite3` adlı yerel bir vektör veritabanı üretir. Cut ve anlatı grubu belgelerinde deterministik başlık, zaman aralığı, görsel tanım, karakterler, ses olayları, konuşmalar, komşu cut özeti, metadata ve normalize embedding birlikte saklanır. Aday cut için en yakın semantik cutlar bu depodan cosine benzerliğiyle getirilir.

ParalonCloud üzerindeki `qwen3.8-27b`, ilk Knapsack sonucunu ve RAG ile getirilen bağları kurulum/sonuç, soru/cevap, çatışma/çözüm devamlılığı açısından inceler. Model öncelik ve kısa Türkçe gerekçe döndürür. Uzak modelin seçimi doğrudan uygulanmaz: öneriler süre maliyeti altında ikinci bir yerel 0/1 Knapsack geçişine sokulur; böylece model bağlamı yorumlarken bütçe matematiksel olarak korunur.

Paralon isteği OpenAI uyumlu `POST /v1/chat/completions` endpoint'ine gider. `PARALON_API_KEY`, `PARALON_BASE_URL` ve `PARALON_MODEL` ortam değişkenleri kullanılır. Anahtar yoksa, yanıt geçersizse veya servis kullanılamazsa ilk Knapsack seçimi korunur.

Bu nedenle algoritma yalnızca en yüksek puanlı art arda sahneleri toplamaz; videonun tamamını ve olayların anlaşılması için gereken sahne bağlarını temsil etmeye çalışır.

Aynı hikâye sahnesindeki benzer cut'ların özeti doldurmasını azaltmak için seçimden önce ek bir tekrar cezası uygulanır. Hikâye grubu içinde puana göre ilk cut ceza almaz; ikinci `%12`, üçüncü `%24`, sonraki cut'lar en fazla `%36` oranında puan kaybeder. Bu kesinti `scoring_breakdown.story_repetition` alanında görünür ve kaç cut'a uygulandığı `selection.story_penalized_cut_count` ile raporlanır.

### 5.9 Konuşmayı bölmeyen highlight

Seçilen cut'ın başlangıcı veya sonu bir repliğin içine denk gelirse sınır repliğin tamamını alacak biçimde `0.22 sn` tamponla genişletilir. Örtüşen/yakın aralıklar birleştirilir. FFmpeg bu aralıkları `trim/atrim`, zaman damgası sıfırlama ve `concat` ile birleştirir.

Video `H.264`, `CRF 23`, `veryfast`; ses varsa `AAC` ile yazılır ve web oynatımı için `faststart` uygulanır.

### 5.10 Hikâye sahnesi gruplama

Komşu cut'lar şu devamlılık sinyalleriyle gruplandırılır:

- `%52` metin embedding benzerliği
- `%25` görsel histogram benzerliği
- `%23` karakter örtüşmesi
- Mekân/konu anahtar kelimeleri
- Jenerik ile anlatı arasındaki geçiş
- Grubun mevcut süresi

Bilinen etiketler arasında Asansör, Araba, Restoran/kafe, Ofis, Mutfak, Hastane, Ev, Mağaza, Açık alan ve Jenerik vardır. Uygun etiket bulunamazsa karakterler, konuşma varlığı veya genel sahne numarası kullanılır.

Her hikâye grubu ayrıca `description` alanı alır. Bu kısa açıklama, grubun etiketi ile cut'lardaki görsel açıklama ve diyaloglardan derlenir; yeni bir model inference'ı değildir.

## 6. Klasör ve dosya rehberi

```text
videotest/
├── README.md                         Kısa tanıtım ve hızlı başlangıç
├── PROJE_MASTER_DOKUMANI.md          Bu kapsamlı teknik/operasyonel belge
├── requirements-light.txt            Model olmadan temel web/CLI bağımlılıkları
├── requirements-models.txt           Tüm yerel ML bağımlılıkları
├── requirements-gemini.txt           Uzak Gemini CLI backend'i
├── start_video_ui.sh / .bat          Linux/macOS ve Windows başlatıcıları
├── scripts/
│   └── download_models.py              Model indirme ve proje içine yerleştirme
├── scene_captioner/
│   ├── analyzer.py                    Ana çok modlu analiz orkestrasyonu
│   ├── audio_analysis.py              Ses, Whisper, konuşmacı ve AudioSet
│   ├── captioners.py                  Mock/BLIP/BLIP-2/LLaVA/Video-LLaVA/Gemini
│   ├── face_analysis.py               YuNet/SFace ve kimlik kümeleme
│   ├── scene_detector.py              Adaptif cut tespiti
│   ├── web_app.py                     Flask API, kuyruk ve kalıcı iş durumu
│   ├── templates/index.html           Tek dosyalı HTML/CSS/JS arayüz
│   ├── cli.py                         Basit aralıklı kare CLI'ı
│   ├── pipeline.py                    Basit CLI veri hattı
│   └── frame_extractor.py             Sabit aralıklı kare çıkarma
├── tests/test_core.py                 Çekirdek birim ve Flask smoke testleri
├── models/                            İndirilen modeller; Git'e girmez
└── outputs/                           Yüklemeler ve analiz sonuçları; Git'e girmez
```

`cli.py` + `pipeline.py` + `frame_extractor.py` eski/basit bir yol sunar: belirli saniye aralıklarıyla kare çıkarıp caption üretir. Asıl SceneMind deneyimi `web_app.py` + `analyzer.py` hattıdır.

## 7. Kurulum

### 7.1 Sistem gereksinimleri

- Python 3 (proje kesin bir minor sürüm kilitlemiyor)
- `ffmpeg` komutunun PATH içinde olması
- Yerel modeller için birkaç GB disk alanı
- CPU ile çalışabilir; CUDA uyumlu NVIDIA GPU görsel ve ses olayı modellerini hızlandırır
- Hedef donanım profili: 6 GB VRAM'li RTX 3060 Laptop veya benzeri

FFmpeg kontrolü:

```bash
ffmpeg -version
```

GPU kontrolü:

```bash
nvidia-smi
```

### 7.2 Tam yerel kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-models.txt
.venv/bin/python scripts/download_models.py
```

Yalnızca belirli model grupları indirilebilir:

```bash
.venv/bin/python scripts/download_models.py llava whisper audio-events embeddings faces
```

Geçerli adlar: `blip`, `llava`, `whisper`, `audio-events`, `embeddings`, `faces`.

Modeller ortak Hugging Face cache'ine bel bağlamak yerine proje içindeki `models/` dizinine indirilir. Ana analiz, gerekli yerel model yoksa internetten kendiliğinden indirmez; açık hata veya bileşen uyarısı üretir.

### 7.3 Hafif kurulum

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-light.txt
```

Bu profil kare çıkarma, mock caption ve temel test/geliştirme içindir. Tam web analizi için Whisper, LLaVA ve diğer model paketleri gerekir.

### 7.4 Gemini CLI kurulumu

```bash
.venv/bin/pip install -r requirements-gemini.txt
export GEMINI_API_KEY="..."
```

Gemini yalnızca basit CLI caption yolunda tanımlıdır; web arayüzü Gemini seçeneği sunmaz.

## 8. Çalıştırma

### 8.1 Web arayüzü

Linux/macOS:

```bash
./start_video_ui.sh
```

veya:

```bash
.venv/bin/python -m scene_captioner.web_app
```

Windows:

```bat
start_video_ui.bat
```

Adres: `http://127.0.0.1:7860`

Sunucu yalnızca loopback adresinde dinler; mevcut haliyle ağa açık bir servis değildir.

### 8.2 Basit CLI

Hızlı smoke test:

```bash
.venv/bin/python -m scene_captioner.cli video.mp4 \
  --backend mock \
  --interval 2 \
  --max-frames 5
```

LLaVA ile:

```bash
.venv/bin/python -m scene_captioner.cli video.mp4 \
  --backend llava \
  --model models/llava \
  --device cuda
```

CLI varsayılan çıktıları:

- `outputs/frames/`
- `outputs/captions.json`
- `outputs/captions.txt`

CLI parametreleri için:

```bash
.venv/bin/python -m scene_captioner.cli --help
```

## 9. Web API sözleşmesi

| Metot | Endpoint | Görevi |
|---|---|---|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/api/models` | Yerel modellerin hazır olup olmadığı |
| `POST` | `/api/process` | Video yükleme ve analiz işi oluşturma |
| `GET` | `/api/jobs/<job_id>` | İş durumu ve kademeli/tam analiz sonucu |
| `POST` | `/api/jobs/<job_id>/cancel` | Sıradaki veya çalışan işi iptal etme |
| `GET` | `/api/history` | Diskteki analiz geçmişi |
| `GET` | `/outputs/<job_id>/<path>` | Kare, portre ve highlight sunumu |
| `GET` | `/uploads/<job_id>/<filename>` | Kaynak videoyu sunma |
| `POST` | `/api/shutdown` | Yalnızca yerel istemciden sunucuyu kapatma |

`POST /api/process`, `multipart/form-data` bekler:

- `video`: zorunlu video dosyası
- `summary_ratio`: `5`–`25` arası yüzde

Başarılı yanıt HTTP `202` ve bir `job_id` döndürür.

İş durumları:

```text
queued -> running -> complete
                  -> cancelling -> cancelled
                  -> error
```

Not: Sırada bekleyen bir future anında iptal edilebilirse doğrudan `cancelled` olabilir.

## 10. Veri ve çıktı yapısı

Her web işi 32 karakterlik UUID hex kimliği kullanır.

```text
outputs/web/
├── uploads/<job-id>/<orijinal-guvenli-dosya-adi>
└── jobs/<job-id>/
    ├── state.json
    ├── analysis.json
    ├── errors.log                 Yalnızca beklenmeyen ana hata varsa
    ├── audio.wav
    ├── scene_embeddings.npy
    ├── highlight.mp4             Oluşturma başarılıysa
    ├── frames/
    │   └── scene_XXXX_....jpg
    └── portraits/
        └── actor_XX.jpg
```

`state.json`, web işinin canlı durumunu ve analiz sırasında oluşan snapshot'ı tutar. `analysis.json` nihai, taşınabilir analiz sonucudur.

### 10.1 Nihai analiz ana alanları

```json
{
  "duration_sec": 123.4,
  "processing_sec": 45.6,
  "scene_count": 42,
  "story_scene_count": 9,
  "selected_count": 8,
  "speaker_count": 3,
  "language": "tr",
  "cast": [],
  "selection": {},
  "story_scenes": [],
  "warnings": [],
  "highlight_path": ".../highlight.mp4",
  "scenes": []
}
```

### 10.2 Bir cut kaydının önemli alanları

| Alan | Anlamı |
|---|---|
| `index` | 1 tabanlı cut numarası |
| `start_sec`, `end_sec`, `duration_sec` | Zaman aralığı |
| `summary_start_sec`, `summary_end_sec` | Konuşma güvenli genişletilmiş aralık |
| `keyframe_sec`, `frame_path` | Ana kare zamanı ve dosyası |
| `description` | Görsel model açıklaması |
| `transcript` | Cut ile örtüşen replikler |
| `speakers` | Cut'taki benzersiz konuşmacı etiketleri |
| `actors` | Tekrarlayan yüz etiketleri |
| `faces` | Ana karedeki normalize yüz kutuları |
| `audio_events` | Etiket ve güven skorları |
| `visual_change_score` | Adaptif detector uzaklık skoru |
| `importance_score` | 0–1 aralığına çevrilmiş toplam puan |
| `selected` | Highlight için seçildi mi? |
| `selection_rank` | Seçim sırası |
| `selection_reasons` | İnsan tarafından okunabilir gerekçeler |
| `scoring_breakdown` | Her bileşenin ham değeri, ağırlığı ve puanı |
| `credit_probability` | Jenerik olasılığı |

Web API bu dosya yollarına ek olarak tarayıcının kullanacağı `frame_url`, `portrait_url`, `highlight_url` ve `source_url` alanlarını çalışma anında ekler; bunlar `analysis.json`'a yazılmaz.

## 11. Hata toleransı ve iptal davranışı

Analiz bileşenleri mümkün olduğunca birbirinden izole edilmiştir:

- Konuşma çözülemezse görsel ve yüz analizi devam eder.
- Konuşmacı kümeleme başarısızsa tek konuşmacı varsayılır.
- Ses olayı veya yüz analizi başarısızsa uyarı eklenir.
- MiniLM çalışmazsa TF-IDF fallback kullanılır.
- Highlight üretilemezse sahne analizi yine kullanılabilir kalır.
- Beklenmeyen ana hata kullanıcıya genel mesaj, diskte `errors.log` üretir.
- GPU/CUDA içeren teknik uyarılar web yanıtında sadeleştirilir.

İptal kontrolü belli aşama sınırlarında ve uzun görsel/ses döngülerinde yapılır. Bir alt model çağrısının tam ortasında anında kesilme garantisi yoktur; mevcut çağrı tamamlandıktan sonra iptal uygulanabilir. O ana kadar kaydedilmiş snapshot korunur.

## 12. Testler ve doğrulama

Testleri çalıştırmak için:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Mevcut test kapsamı:

- Yapay renk geçişli videoda adaptif cut tespiti
- Dashboard ve model durum endpoint'i smoke testi
- Knapsack'in bütçe altında en iyi değeri seçmesi
- Başlangıç/orta/son zamansal kotası
- Highlight sınırlarının replik içine düşmemesi
- Poz nedeniyle bölünen aynı yüzün birleşmesi, aynı sahnedeki ikinci kişinin birleşmemesi

Testler model kalitesi, gerçek uzun video performansı, tüm API hata durumları, FFmpeg highlight çıktısı ve tarayıcı etkileşimlerini henüz kapsamaz.

## 13. Performans karakteri

45 dakikalık bir bölümü 2–3 dakikada analiz etmek bir **hedef donanım profili**, genel garanti değildir. Süreyi en fazla etkileyenler:

- Cut sayısı ve her cut için yapılan LLaVA çağrısı
- CPU'daki Whisper transkripsiyonu
- En fazla 96 pencereye kadar AudioSet analizi
- Yüz analizi için cut başına ek kare örnekleri
- FFmpeg yeniden kodlama süresi
- Model ilk yükleme/soğuk başlatma maliyeti

Bellek yönetimi için büyük model aşamaları sonrası Python GC ve CUDA cache temizleme uygulanır. LLaVA için yaklaşık 2,7 GB, BLIP ve ses olayları için yaklaşık 1,2 GB boş GPU belleği aranır; yeterli değilse CPU tercih edilir.

Tek `ThreadPoolExecutor(max_workers=1)` kullanılması bilinçlidir: aynı anda birden fazla ağır model işinin GPU/RAM'i tüketmesini engeller. Sonraki işler sırada bekler.

## 14. Bilinen sınırlar ve teknik borç

1. **Kimlikler anonimdir.** Yüz ve ses grupları gerçek kişi adı vermez.
2. **Diarization yaklaşıktır.** Müzik, gürültü, kısa replik, benzer sesler ve dublaj hataya yol açabilir.
3. **Caption dili İngilizce olabilir.** Web görsel istemi İngilizcedir; Türkçe UI içinde model çıktısı İngilizce görünebilir.
4. **Mekân/hikâye etiketi sezgiseldir.** Anahtar kelime ve benzerlik tabanlıdır; gerçek olay grafiği değildir.
5. **Jenerik tespiti sezgiseldir.** OCR kullanılmadığı için yazı yoğun jenerikler kaçabilir veya karanlık sessiz sahneler yanlış puanlanabilir.
6. **Web servisi tek kullanıcı/yerel odaklıdır.** Kimlik doğrulama, yetkilendirme, CSRF koruması, kota ve çok kullanıcı izolasyonu yoktur.
7. **Kalıcı veritabanı yoktur.** Durum JSON dosyası ve proses içi sözlükte tutulur.
8. **Depolama temizliği yoktur.** Yüklemeler, WAV'lar, kareler ve videolar elle temizlenene kadar diskte kalır.
9. **Bağımlılıklar alt sınırla tanımlıdır.** Kilit dosyası olmadığı için gelecekteki paket sürümleri uyumsuzluk yaratabilir.
10. **Python ve FFmpeg sürüm matrisi belgelenmemiştir.** CI yapılandırması yoktur.
11. **`scenedetect` bağımlılığı mevcut kod yolunda kullanılmıyor.** Cut tespiti tamamen OpenCV ile yazılmıştır.
12. **Basit CLI ve web hattı farklıdır.** CLI sabit aralıklı kare çıkarır; tam analiz yapmaz. Bu ayrım yeni geliştiriciler için yanıltıcı olabilir.
13. **Model sağlık endpoint'i eksik olabilir.** Dizin varlığını kontrol eder; dosyaların eksiksiz veya yüklenebilir olduğunu doğrulamaz.
14. **Arayüz tek HTML dosyasıdır.** Prototip için kolay, büyüdükçe bakım için zordur.
15. **`narrative_position` ham değeri şu an sabittir.** Zamansal denge seçim algoritmasında gerçek uygulanır ama puan kartındaki 5 puan dinamik değildir.

## 15. Güvenlik ve gizlilik

Olumlu mevcut davranışlar:

- Ana modeller yerel çalışır ve web sunucusu `127.0.0.1`'e bağlanır.
- Dosya adı `secure_filename` ile temizlenir.
- Job ID UUID olarak doğrulanır.
- Shutdown endpoint'i yalnızca loopback istemcisini kabul eder.
- Teknik hata ayrıntısı normalde arayüze doğrudan verilmez.

Dikkat edilmesi gerekenler:

- `GeminiCaptioner` kullanılırsa seçilen kareler harici API'ye gönderilir; bu yerel web akışının parçası değildir.
- Proje ağa açılacaksa önüne kimlik doğrulama, TLS, ters proxy, dosya içeriği doğrulama, istek kotası ve yetkili medya sunumu eklenmelidir.
- `outputs/` hassas konuşma metinleri, portreler, kaynak video ve ses içerebilir. Yedekleme/silme politikası tanımlanmalıdır.

## 16. Geliştirme rehberi

### Yeni bir caption backend'i eklemek

1. `captioners.py` içinde `Captioner` sınıfından türet.
2. `caption(image_path, prompt)` metodunu uygula.
3. `create_captioner()` factory'sine backend adını ekle.
4. CLI'da kullanılacaksa `--backend` choices listesine ekle.
5. Web'de kullanılacaksa `analyzer._caption_scenes()` içindeki yerel model seçimini ve `/api/models` sağlık kontrolünü güncelle.
6. Model indirilecekse `scripts/download_models.py` kaydı ekle.

### Yeni bir sahne puanı eklemek

1. `SCORE_WEIGHTS` toplamını yine 100 olacak şekilde düzenle.
2. `SCORE_LABELS` içine insan tarafından okunabilir etiketi ekle.
3. `_embed_and_select()` içinde 0–1 normalize ham değeri hesapla.
4. `scoring_breakdown` çıktısını koru; arayüz yeni kalemi otomatik listeler.
5. Seçim testlerini ve toplam ağırlık testini güncelle.

### Yeni bir analiz aşaması eklemek

- Uzun işlerde `cancel_check` kontrolü koy.
- `progress` ile kullanıcıya aşama ve yüzde bildir.
- Model bittiğinde referansları serbest bırak ve gerekiyorsa `_release_models()` çağır.
- Bileşen ana sonucu engellemek zorunda değilse hatayı `warnings`'e indirge.
- Yeni alanı `SceneResult` veya ana payload'da açık ve JSON uyumlu tut.
- Kademeli snapshot sırasında alanın varsayılan değerle bulunmasını sağla.

### UI geliştirirken

- API'den gelen tüm serbest metinler `esc()` ile HTML kaçışlanmalıdır.
- Büyük sahne listesi ilk 24 kayıtla sınırlanır; “daha fazla” davranışı korunmalıdır.
- Analiz tamamlanmadan `analysis` kısmi olabilir; renderer eksik alanlara dayanıklı kalmalıdır.
- Browser cache, HTML ve JSON için bilerek kapatılmıştır.

## 17. Önerilen sonraki yol haritası

### Kısa vadede

1. Python/FFmpeg/CUDA için test edilmiş sürüm matrisi ve kilitli bağımlılık dosyası eklemek
2. Kurulum öncesi `ffmpeg`, disk, RAM, CUDA ve model bütünlüğünü kontrol eden doctor komutu yazmak
3. Gerçek küçük fixture video ile uçtan uca test eklemek
4. `analysis.json` için sürümlü JSON Schema tanımlamak
5. Eski işler için arayüzden silme ve otomatik saklama süresi eklemek
6. Caption çıktısı için seçilebilir Türkçe/otomatik dil davranışı eklemek

### Orta vadede

1. OCR ile jenerik, mekân tabelası ve ekrandaki metni analiz etmek
2. Daha güçlü diarization/voice embedding ile konuşmacı kararlılığını artırmak
3. Mevcut hikâye-içi tekrar cezasını embedding tabanlı genel çeşitlilik/MMR seçimiyle geliştirmek
4. Kullanıcının ağırlıkları ve seçilen sahneleri elle düzenleyebilmesi
5. Altyazı dosyası (`SRT/VTT`), CSV ve rapor dışa aktarımı
6. Frontend CSS/JS'yi modüllere ayırmak

### Ürünleştirme aşamasında

1. SQLite/PostgreSQL tabanlı kalıcı iş kuyruğu ve metadata
2. Ayrı işçi prosesleri, GPU iş planlama ve yeniden deneme politikası
3. Kimlik doğrulama, kullanıcı/organizasyon izolasyonu ve yetkili dosya sunumu
4. Gözlemlenebilirlik: yapısal log, aşama süreleri, GPU/RAM metrikleri
5. Veri saklama, silme, anonimleştirme ve açık rıza politikaları

## 18. Hızlı sorun giderme

### “Model eksik” görünüyor

```bash
.venv/bin/python scripts/download_models.py
```

`models/llava`, `models/whisper`, `models/audio-events`, `models/embeddings` ve `models/opencv` içeriğini kontrol et.

### Video yükleniyor ama analiz hemen duruyor

`outputs/web/jobs/<job-id>/errors.log` dosyasına bak. Ayrıca:

```bash
ffmpeg -version
.venv/bin/python -c "import cv2, flask, numpy; print('temel paketler hazır')"
```

### Highlight oluşmuyor

- FFmpeg'de `libx264` ve `aac` encoder'larının bulunduğunu kontrol et.
- `analysis.json` içindeki `selected_count`, `selection.speech_safe_intervals` ve `warnings` alanlarına bak.
- Kaynak videonun bozuk ses/video zaman damgaları olup olmadığını kontrol et.

### GPU kullanılmıyor

- `nvidia-smi` çalışmalı.
- Kurulu PyTorch CUDA destekli olmalı.
- Boş VRAM modelin minimum tercih eşiğinin altındaysa sistem bilerek CPU'ya döner.
- Whisper ana web hattında mevcut kod gereği CPU'da çalışır.

### İş iptal olmuyor gibi görünüyor

İptal kooperatiftir. Devam eden tek bir model inference veya FFmpeg alt prosesi bitene kadar durum `cancelling` kalabilir.

## 19. Projenin tasarım ilkeleri

Kodun mevcut kararlarından çıkan temel ilkeler şunlardır:

- **Yerel öncelikli:** Video ve ana modeller kullanıcının makinesinde kalır.
- **Açıklanabilir:** Her özet kararı puan bileşenleri ve metinsel gerekçelerle gösterilir.
- **Kısmi başarı değerlidir:** Bir model bozulsa bile diğer analiz sonuçları korunur.
- **Donanıma uyarlanır:** VRAM uygunsa GPU, değilse CPU kullanılır.
- **Anlatı bütünlüğü:** Başlangıç/orta/son dengesi ve tam replik sınırları korunur.
- **Kademeli geri bildirim:** Kullanıcı uzun analiz bitene kadar boş ekrana bakmaz.

## 20. Tek cümlelik devir notu

Projeyi devralan kişi için en kritik bilgi şudur: **Ana ürün yolu `web_app.py -> analyzer.py` akışıdır; `cli.py` ise aynı ad alanında duran ama tam SceneMind analizi yapmayan daha basit, sabit aralıklı kare-caption aracıdır.**
