# CineSum Audio Master Proje Raporu

> **Belge sürümü:** 1.0  
> **Tarih:** 15 Ağustos 2026  
> **Kapsam:** `cinesum-audio` çalışma alanının mevcut durumu  
> **Birincil giriş noktası:** `process_video.py`  
> **Hedef okuyucu:** Yazılım, veri/ML, test, ürün ve operasyon ekipleri  
> **Son doğrulama:** 6/6 otomatik test başarılı (`.venv`, 15 Ağustos 2026)

## Belgenin amacı

Bu belge, projeye yeni katılan bir ekip üyesinin:

- sistemi kurup çalıştırabilmesini,
- mimariyi ve veri akışını anlayabilmesini,
- çıktı dosyalarını doğru yorumlayabilmesini,
- teknik kararların gerekçesini görebilmesini,
- bilinen riskleri ve sınırlamaları öğrenebilmesini,
- güvenli biçimde geliştirme ve test yapabilmesini

sağlamak üzere hazırlanmış tek kaynak niteliğindeki ekip içi master rapordur.

## İçindekiler

1. [Yönetici özeti](#1-yönetici-özeti)
2. [Sistem bağlamı ve sınırlar](#2-sistem-bağlamı-ve-sınırlar)
3. [Ekip sorumluluk haritası](#3-ekip-sorumluluk-haritası)
4. [Hızlı başlangıç ve işletim kılavuzu](#4-hızlı-başlangıç-ve-işletim-kılavuzu)
5. [Komut satırı sözleşmesi](#5-komut-satırı-sözleşmesi)
6. [Çıktı ve veri sözleşmeleri](#6-çıktı-ve-veri-sözleşmeleri)
7. [Doğrulama ve kabul ölçütleri](#7-doğrulama-ve-kabul-ölçütleri)
8. [Örnek çalışmaların anlık görünümü](#8-örnek-çalışmaların-anlık-görünümü)
9. [Riskler ve azaltma planı](#9-riskler-ve-azaltma-planı)
10. [Önceliklendirilmiş yol haritası](#10-önceliklendirilmiş-yol-haritası)
11. [Devir teslim kontrol listesi](#11-devir-teslim-kontrol-listesi)
12. [Ayrıntılı teknik çalışma raporu](#12-ayrıntılı-teknik-çalışma-raporu)
13. [Kaynak kod sorumluluk matrisi](#13-kaynak-kod-sorumluluk-matrisi)
14. [Kritik fonksiyonlar](#14-kritik-fonksiyonlar)
15. [Teknik karar kayıtları](#15-teknik-karar-kayıtları)
16. [Sorun giderme](#16-sorun-giderme)
17. [Sürümleme ve güncelleme protokolü](#17-sürümleme-ve-güncelleme-protokolü)

## 1. Yönetici özeti

CineSum Audio, bir video dosyasındaki sesi tek komutla analiz eden Python tabanlı bir işlem hattıdır. Sistem konuşmayı metne dönüştürür, konuşmacı turlarını ayırır, kelimeleri konuşmacılara atar, eş zamanlı konuşmaları belirler ve isteğe bağlı olarak konuşma dışı ses olaylarını raporlar. Sonuçlar insan tarafından okunabilir metin ve makine tarafından işlenebilir JSON biçiminde saklanır.

| Alan | Durum | Ekip için anlamı |
|---|---|---|
| Ana işlem hattı | Çalışır | `process_video.py` tek ve önerilen giriş noktasıdır. |
| Konuşmacı düzeltmesi | Çalışır | Embedding tabanlı otomatik düzeltme ve video bazlı manuel override birlikte kullanılabilir. |
| Otomatik testler | 6/6 başarılı | Sınır ataması, override, birleştirme ve overlap davranışları korunmaktadır. |
| Örnek çıktılar | Mevcut | `output/test` ve `output/test3` inceleme için kullanılabilir. |
| İşlem cihazı | CPU | Uzun videolarda işlem süresi yüksek olabilir. |
| Önbellek | Kısmi | Yalnızca dış ses olayları önbelleğe alınır. |

### Mevcut durumun kısa yorumu

Ana akış çalışır durumdadır. Projedeki en önemli üretim riskleri CPU performansı, model indirme ve erişim bağımlılığı, otomatik diarization hataları ve transkripsiyon/diarization önbelleğinin henüz bulunmamasıdır. Mevcut testler eşleme ve düzeltme algoritmalarını doğrular; gerçek model kalitesini ölçen referans anotasyonlu uçtan uca değerlendirme henüz yoktur.

## 2. Sistem bağlamı ve sınırlar

Sistem video dosyasını girdi olarak alır ve yerel dosya sistemine video bazlı sonuç paketi yazar. FFmpeg medya çözümleme katmanıdır. Hugging Face üzerinden edinilen modeller transkripsiyon, diarization, embedding ve dış ses sınıflandırmasını gerçekleştirir.

```text
Video
  └─ FFmpeg → 16 kHz mono PCM ses
       ├─ Pyannote diarization → ham konuşmacı turları
       │    └─ Embedding düzeltmesi → nihai konuşmacı turları
       ├─ Faster-Whisper → segmentler ve kelime zamanları
       │    └─ Kelime/konuşmacı eşleme → konuşma blokları
       └─ AST dış ses analizi (isteğe bağlı) → ses olayları

Nihai birleştirme → result.json + result.txt + denetim çıktıları
```

### 2.1. Kapsam dışındaki noktalar

- Gerçek kişi adının otomatik belirlenmesi veya yüz tanıma
- Çıktıların web arayüzünde görsel olarak düzenlenmesi
- Dağıtık iş kuyruğu, çok kullanıcılı servis veya bulut dağıtımı
- Diarization ve transkripsiyon için güvenli yeniden kullanım önbelleği
- Model doğruluğunun referans veri kümesinde nicel kıyaslaması

> `SPEAKER_00`, `SPEAKER_01` gibi etiketler gerçek kişi kimliği değildir. Yalnızca işlendiği video içindeki konuşmacı kümelerini temsil eder.

## 3. Ekip sorumluluk haritası

| Rol | Ana sorumluluk | Kontrol noktaları |
|---|---|---|
| Teknik lider | Mimari kararlar ve sürüm kapsamı | Model/algoritma değişiklikleri, performans-risk dengesi |
| ML geliştiricisi | Diarization, embedding, ASR ve eşikler | Ham/düzeltilmiş sonuç karşılaştırması ve regresyon örnekleri |
| Uygulama geliştiricisi | CLI, dosya akışı ve hata yönetimi | Girdi doğrulama ve deterministik çıktı şeması |
| Test sorumlusu | Davranış ve uçtan uca doğrulama | Birim testleri ve örnek video kabul kriterleri |
| Ürün/içerik sorumlusu | Kullanım senaryosu ve çıktı kalitesi | Metin okunabilirliği ve konuşmacı geri bildirimi |
| Operasyon | Kurulum, model erişimi, FFmpeg ve GPU | Sürüm sabitleme, disk/CPU/GPU ve token erişimi |

## 4. Hızlı başlangıç ve işletim kılavuzu

1. Python 3.13 sanal ortamını etkinleştirin.
2. `requirements.txt` içindeki bağımlılıkları kurun.
3. FFmpeg'in sistem `PATH` değişkeninde, `FFMPEG_PATH` içinde veya proje tarafından kontrol edilen yerel konumlardan birinde erişilebilir olduğunu doğrulayın.
4. Hugging Face hesabında Pyannote model koşullarını kabul edin.
5. `hf auth login` ile oturum açın.
6. Videoyu `input/` klasörüne koyun veya komutta dosya yolunu belirtin.
7. İlk doğrulamayı dış ses analizini kapatarak çalıştırın.

```powershell
.\.venv\Scripts\python.exe process_video.py input\video.mp4 --skip-events
```

8. `output/<video_adı>/result.txt` ve `result.json` dosyalarını kontrol edin.
9. Otomatik konuşmacı düzeltmelerini incelemek için `diarization_raw.json`, `diarization.json` ve `speaker_correction_report.json` dosyalarını karşılaştırın.

### Önerilen çalışma modları

```powershell
# Doğruluk öncelikli, dış ses analizi kapalı
python process_video.py input\video.mp4 --skip-events

# Konuşma + konuşmacı + dış ses analizi
python process_video.py input\video.mp4

# Daha hızlı fakat daha düşük doğruluk potansiyeli
python process_video.py input\video.mp4 --fast --skip-events

# Konuşmacı sayısı kesin biliniyorsa
python process_video.py input\video.mp4 --speakers 5 --skip-events
```

> Doğruluk öncelikliyse `medium` Whisper modeli ve beam size 5 korunmalıdır. `--skip-events` yalnızca dış ses analizini kapatır. `--fast` ise model ve beam ayarını değiştirdiğinden doğruluk ödünüdür.

## 5. Komut satırı sözleşmesi

| Seçenek | Tür | Varsayılan | Etkisi |
|---|---|---|---|
| `video` | Zorunlu | - | İşlenecek video yolu |
| `--language` | İsteğe bağlı | `en` | Whisper konuşma dili |
| `--whisper-model` | İsteğe bağlı | `medium` | ASR modeli; `--fast` ile varsayılan `small` olur |
| `--device` | İsteğe bağlı | `auto` | `auto`, `cpu` veya `cuda` |
| `--fast` | Bayrak | Kapalı | Daha küçük ASR modeli ve daha seyrek olay taraması |
| `--skip-events` | Bayrak | Kapalı | AST dış ses analizini atlar |
| `--recompute-events` | Bayrak | Kapalı | Dış ses önbelleğini yok sayar |
| `--event-threshold` | İsteğe bağlı | `0.35` | Dış ses kabul eşiği |
| `--speakers` | İsteğe bağlı | Otomatik | Kesin konuşmacı sayısı |
| `--min-speakers` | İsteğe bağlı | Otomatik | Minimum konuşmacı sayısı |
| `--max-speakers` | İsteğe bağlı | Otomatik | Maksimum konuşmacı sayısı |
| `--speaker-threshold` | İsteğe bağlı | `0.40` | Pyannote clustering hassasiyeti |
| `--no-speaker-correction` | Bayrak | Kapalı | Embedding doğrulama katmanını kapatır |
| `--disable-intra-speaker-split` | Bayrak | Kapalı | Aynı etiketteki farklı sesleri ayırmayı kapatır |
| `--speaker-merge-threshold` | İsteğe bağlı | `0.88` | Benzer kümeleri birleştirme eşiği |
| `--reassign-similarity` | İsteğe bağlı | `0.72` | Yeniden atama asgari benzerliği |
| `--reassign-margin` | İsteğe bağlı | `0.15` | Yeniden atama fark şartı |
| `--speaker-overrides` | İsteğe bağlı | Video çıktı klasörü | Manuel düzeltme JSON yolu |

## 6. Çıktı ve veri sözleşmeleri

| Dosya | Kaynak | Kullanım amacı |
|---|---|---|
| `result.txt` | Nihai timeline | İnsan incelemesi ve hızlı kalite kontrol |
| `result.json` | Tüm aşamalar | Uygulama entegrasyonu ve arşiv |
| `transcript.json` | Faster-Whisper | Segment/kelime zamanları ve ASR denetimi |
| `diarization_raw.json` | Pyannote | Düzeltme öncesi referans |
| `diarization.json` | Düzeltme katmanı | Nihai turlar, overlap ve rapor özeti |
| `speaker_correction_report.json` | Embedding katmanı | Birleştirme, bölme ve yeniden atama denetimi |
| `speaker_overrides.json` | Kullanıcı | Video bazlı kesin düzeltmeler |
| `sound_events.json` | AST | Olaylar ve önbellek anahtarı |

### 6.1. `result.json` üst seviye alanları

| Alan | Anlam |
|---|---|
| `input_file` | Kaynak video yolu |
| `language` | Transkripsiyon dili |
| `speaker_count` | Düzeltme sonrası benzersiz konuşmacı sayısı |
| `segments` | Konuşma blokları: `start`, `end`, `speaker`, `text`, `type` |
| `sound_events` | Dış ses olayları; `--skip-events` durumunda boş olabilir |
| `overlaps` | Aynı zaman aralığındaki çoklu konuşmacılar |
| `speaker_correction` | Diarization düzeltme özeti |
| `timeline` | Konuşma, overlap ve olayların zamana göre birleşik görünümü |

### 6.2. Manuel konuşmacı düzeltme sözleşmesi

```json
{
  "overrides": [
    {
      "start": 26.7,
      "end": 28.7,
      "speaker": "SPEAKER_02",
      "note": "Dinlenerek doğrulanan bölüm"
    }
  ]
}
```

Manuel düzeltmeler video bazlıdır. Bir videodaki `SPEAKER_02` etiketi başka bir videoda aynı kişiyi temsil etmek zorunda değildir.

## 7. Doğrulama ve kabul ölçütleri

15 Ağustos 2026 tarihinde proje içindeki `.venv` kullanılarak altı davranış testi yeniden çalıştırılmış ve tamamı geçmiştir.

| Doğrulanan davranış | Sonuç |
|---|---|
| Kelime zamanı yoksa segment metnini koruma | Başarılı |
| En büyük örtüşmeye göre kelime atama | Başarılı |
| Manuel override ile segment bölme | Başarılı |
| Aynı konuşmacının ardışık Whisper segmentlerini birleştirme | Başarılı |
| Overlap için düzeltilmiş en yakın kimliği kullanma | Başarılı |
| Cümle başı zaman kaymasında tek kelimelik yanlış etiketi önleme | Başarılı |

Test komutu:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 7.1. Test kapsamının sınırı

Bu altı test hızlı ve model bağımsız davranış testleridir. Aşağıdakileri ölçmez:

- Whisper kelime hata oranı
- Pyannote diarization hata oranı
- Dış ses sınıflandırma doğruluğu
- Uzun video performansı ve bellek tüketimi
- Gerçek video üzerinde insan anotasyonuyla uçtan uca kalite

### 7.2. Uçtan uca kabul kontrol listesi

- [ ] Komut hatasız tamamlandı ve beklenen video çıktı klasörü oluştu.
- [ ] `result.txt` kronolojik, okunabilir ve boş değil.
- [ ] `speaker_count`, segmentlerdeki benzersiz konuşmacı sayısıyla uyumlu.
- [ ] Ham ve düzeltilmiş diarization çıktıları karşılaştırıldı.
- [ ] Otomatik düzeltmeler `speaker_correction_report.json` içinde açıklanıyor.
- [ ] Overlap etiketleri düzeltilmiş konuşmacı kimlikleriyle uyumlu.
- [ ] Manuel override kullanıldıysa kapsam yalnızca ilgili videoya ait.
- [ ] Kullanılan model, eşik ve çalışma seçenekleri teslim notuna kaydedildi.

## 8. Örnek çalışmaların anlık görünümü

| Örnek | Konuşmacı | Konuşma bloğu | Overlap | Timeline |
|---|---:|---:|---:|---:|
| `output/test` | 5 | 12 | 2 | 14 |
| `output/test3` | 4 | 36 | 12 | 48 |

Bu sayılar doğruluk metriği değildir. Yalnızca depodaki sonuç paketlerinin yapısal özetidir. Model kalitesi, video dinlenerek veya izlenerek referans anotasyonla karşılaştırılmalıdır.

## 9. Riskler ve azaltma planı

| Risk | Etki | Azaltma yaklaşımı |
|---|---|---|
| CPU üzerinde uzun çalışma | Teslim süresi ve kullanıcı deneyimi | GPU, `--skip-events`, güvenli ASR/diarization önbelleği |
| Model ağı/token bağımlılığı | İlk kurulumda başarısızlık | Kurulum ön kontrolü, model koşulları ve yerel cache doğrulaması |
| Kısa veya benzer seslerde diarization hatası | Yanlış konuşmacı | Embedding doğrulaması, override, bilinen kişi sayısı ve örnek QA |
| Overlap, müzik ve gürültü | Sınır/kimlik belirsizliği | Ham/nihai sonuç ayrımı, güven puanı ve manuel inceleme |
| JSON şeması değişikliği | Tüketici uygulamaların kırılması | Şema sürümü, sözleşme testi ve geriye uyumluluk |
| Testlerin model kalitesini ölçmemesi | Yanlış güven | Referans anotasyonlu uçtan uca değerlendirme seti |

## 10. Önceliklendirilmiş yol haritası

| Öncelik | İş | Tamamlanma ölçütü |
|---|---|---|
| P0 | Transkripsiyon ve diarization önbelleği | Video özeti, ayarlar ve model sürümüyle güvenli cache invalidation |
| P0 | Uçtan uca örnek video regresyonu | Referans anotasyon ve ölçülen kalite metrikleri |
| P1 | JSON şema sürümleme | `schema_version` ve sözleşme testleri |
| P1 | Konuşmacı isim eşleme | `SPEAKER_XX` etiketinden kullanıcı onaylı görünen ada dönüşüm |
| P1 | Güven puanları | Düşük güvenli ASR, atama ve olayların işaretlenmesi |
| P2 | Görsel düzeltme arayüzü | Timeline üzerinde dinle-düzelt-kaydet akışı |
| P2 | Dağıtım ve paketleme | Tekrarlanabilir kurulum ve ortam sağlık kontrolü |

## 11. Devir teslim kontrol listesi

- [ ] Yeni ekip üyesi `README.md` ve bu master raporu okudu.
- [ ] `process_video.py --help` çıktısı incelendi.
- [ ] En az bir örnek komut çalıştırıldı.
- [ ] Altı birim test yerel `.venv` ile çalıştırıldı.
- [ ] `output/test` paketinde ham ve nihai diarization karşılaştırıldı.
- [ ] `speaker_overrides.json` kapsamının video bazlı olduğu anlaşıldı.
- [ ] Model, Hugging Face token ve FFmpeg bağımlılıkları doğrulandı.
- [ ] Planlanan değişiklik için etkilenen fonksiyonlar ve çıktı şemaları belirlendi.
- [ ] Yeni davranış için test ve örnek çıktı güncelleme sorumlusu atandı.

## 12. Ayrıntılı teknik çalışma raporu

Aşağıdaki bölüm, 14 Ağustos 2026 tarihli teknik çalışma raporunun ayrıntılarını master belgenin parçası olarak korur.

**Rapor tarihi:** 14 Ağustos 2026  
**Proje dizini:** `cinesum-audio`  
**Ana çalışma dosyası:** `process_video.py`

## 1. Çalışmanın amacı

CineSum Audio projesinin amacı, bir video dosyasındaki ses kanalını analiz ederek:

1. konuşmaları yazıya dönüştürmek,
2. konuşan kişileri birbirinden ayırmak,
3. her kelimeyi veya cümleyi doğru konuşmacıyla eşleştirmek,
4. eş zamanlı konuşmaları belirlemek,
5. isteğe bağlı olarak müzik, kahkaha ve benzeri konuşma dışı sesleri bulmak,
6. sonuçları hem insan tarafından okunabilir metin hem de yazılımlar tarafından işlenebilir JSON biçiminde üretmektir.

Çalışma sırasında yalnızca sistemin çalıştırılması değil; proje yapısının sadeleştirilmesi, hata mesajlarının iyileştirilmesi, konuşmacı sınırlarının düzeltilmesi ve tekrar oluşabilecek hatalara karşı otomatik testlerin hazırlanması da hedeflenmiştir.

## 2. Kullanılan temel teknolojiler

| Bileşen | Kullanım amacı | Projedeki sürüm/ayar |
|---|---|---|
| Python | Ana uygulama dili | Python 3.13 |
| FFmpeg | Videodan sesi çıkarma ve dönüştürme | 16 kHz, mono, PCM |
| faster-whisper | Konuşmayı yazıya dönüştürme | 1.2.1 |
| Whisper `medium` | Varsayılan ve daha doğru transkripsiyon | CPU üzerinde `int8` |
| Whisper `small` | `--fast` modundaki hızlı transkripsiyon | CPU üzerinde `int8` |
| pyannote.audio | Konuşmacı diarization işlemi | 4.0.7 |
| `speaker-diarization-community-1` | Konuşmacı segmentlerini üretme | Hugging Face modeli |
| PyTorch | Model çalıştırma ve tensör işlemleri | 2.13.0 CPU |
| AST AudioSet modeli | Konuşma dışı ses olaylarını sınıflandırma | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| NumPy | Ses örneklerinin işlenmesi | Gereksinim dosyasından kurulur |

Mevcut ortamda CUDA destekli PyTorch bulunmadığı doğrulanmıştır. Bu nedenle modeller CPU üzerinde çalışmaktadır.

## 3. Güncel proje yapısı

Projenin önerilen tek giriş noktası `process_video.py` dosyasıdır.

```text
cinesum-audio/
├── input/                         Girdi videoları
├── output/                        Video bazlı sonuç klasörleri
├── tests/
│   └── test_process_video.py      Otomatik davranış testleri
├── process_video.py               Ana uçtan uca çalışma akışı
├── speaker_correction.py          Embedding tabanlı konuşmacı düzeltmeleri
├── transcribe.py                  Eski deneysel transkripsiyon betiği
├── diarize.py                     Eski deneysel diarization betiği
├── merge_results.py               Eski deneysel birleştirme betiği
├── requirements.txt               Python bağımlılıkları
├── README.md                      Kısa kullanım belgesi
└── PROJE_TEKNIK_RAPORU.md         Bu rapor
```

`transcribe.py`, `diarize.py` ve `merge_results.py` önceki üç aşamalı deneysel akıştan kalmıştır. Güncel kullanımda bunların ayrı ayrı çalıştırılması gerekmez.

## 4. Uçtan uca işlem akışı

### 4.1. Girdinin doğrulanması

Program komut satırından video yolunu alır. Dosyanın varlığı, eşik değerleri, konuşmacı sayısı seçenekleri ve işlem cihazı doğrulanır. Geçersiz bir değer varsa işlem model yüklenmeden önce durdurulur.

### 4.2. FFmpeg’in bulunması

FFmpeg aşağıdaki sırayla aranır:

1. `FFMPEG_PATH` ortam değişkeni,
2. proje içindeki `.venv/Scripts/ffmpeg.exe`,
3. bilinen indirme klasörleri,
4. sistem `PATH` değişkeni.

Bu düzenleme sayesinde program yalnızca belirli bir çalışma klasöründen başlatılmaya bağımlı değildir.

### 4.3. Sesin belleğe alınması

FFmpeg video sesini doğrudan belleğe şu biçimde aktarır:

- örnekleme hızı: 16.000 Hz,
- kanal: mono,
- veri biçimi: 16 bit PCM,
- uygulama içi gösterim: `float32` NumPy dizisi.

Sesin pyannote tarafından tekrar dosyadan açılması yerine bellekte verilmesi, TorchCodec bağımlılığına duyulan ihtiyacı azaltır.

### 4.4. Konuşmacı diarization

`pyannote/speaker-diarization-community-1` modeli konuşma turlarını başlangıç, bitiş ve konuşmacı etiketiyle üretir. İstenirse:

- kesin konuşmacı sayısı `--speakers`,
- minimum sayı `--min-speakers`,
- maksimum sayı `--max-speakers`,
- clustering hassasiyeti `--speaker-threshold`

ile belirtilebilir.

Ham sonuç `diarization_raw.json` dosyasında korunur. Bu, sonraki düzeltmelerin etkisini ham model sonucuyla karşılaştırmayı mümkün kılar.

### 4.5. Embedding tabanlı konuşmacı düzeltmesi

`speaker_correction.py`, yeterince uzun konuşma segmentlerinden ses kimliği embedding’leri çıkarır. Bu katmanda:

- konuşmacı prototipleri oluşturulur,
- yanlış kümeye atanmış segmentler yeniden değerlendirilebilir,
- birbirinin tekrarı olduğu düşünülen kümeler birleştirilebilir,
- tek etiket altında kalmış farklı sesler kontrollü şekilde ayrılabilir,
- çok kısa ve şüpheli A–B–A geçişleri incelenir,
- yapılan işlemler raporlanır.

Sonuçlar `speaker_correction_report.json` dosyasına yazılır. Böylece otomatik düzeltmeler görünür ve denetlenebilir kalır.

### 4.6. Konuşmanın yazıya dönüştürülmesi

faster-whisper kelime zaman damgaları açık şekilde çalışır. Varsayılan ayarlar:

- model: `medium`,
- dil: İngilizce (`en`),
- CPU hesaplama türü: `int8`,
- beam size: 5,
- VAD filtresi: açık,
- kelime zaman damgaları: açık.

`--fast` kullanıldığında varsayılan `medium` modeli `small` modeline çevrilir ve beam size 1 olur. Bu seçenek daha hızlıdır; ancak kelime zamanları ve konuşmacı sınırları daha kaba olabileceği için doğruluk düşebilir.

### 4.7. Kelimelerin konuşmacılarla eşleştirilmesi

İlk sürümde her kelime, diarization segmentleriyle örtüşme süresine göre bağımsız olarak atanıyordu. Bu yöntem genel olarak çalışsa da konuşmacı değişiminde bazı cümlelerin ilk kelimesi önceki konuşmacıda kalabiliyordu.

Bu problem için `assign_word_speakers` adlı segment düzeyinde bir atama mekanizması geliştirilmiştir. Yeni yaklaşım:

1. Bir Whisper segmentindeki tüm kelimeleri birlikte değerlendirir.
2. Her kelimenin konuşmacılarla örtüşme oranını hesaplar.
3. Ardışık kelimeler arasında gereksiz konuşmacı değişimine ceza uygular.
4. Küçük zaman damgası kaymalarının tek kelimelik sahte konuşmacı segmenti üretmesini engeller.
5. Açıkça belirtilen manuel düzeltmeleri en yüksek öncelikle uygular.

Bu yaklaşım, Viterbi benzeri dinamik programlama mantığı kullanır. `switch_penalty=0.35` değeri, akustik kanıt yeterli değilse konuşmacı etiketinin tek kelime için gidip gelmesini önler.

### 4.8. Ardışık aynı konuşmacıların birleştirilmesi

Whisper aynı kişinin konuşmasını birden fazla segment halinde döndürebilir. Kullanıcı isteğine göre, arada başka bir konuşmacı yoksa aynı konuşmacıya ait ardışık bölümler tek bir konuşma bloğunda birleştirilmektedir.

Örneğin iki ayrı Whisper segmenti:

```text
SPEAKER_04: A real man makes his own luck, Archie.
SPEAKER_04: Right, Dawson?
```

nihai sonuçta şu şekilde gösterilebilir:

```text
SPEAKER_04: A real man makes his own luck, Archie. Right, Dawson?
```

Konuşmacı değiştiğinde yeni bir blok başlatılır.

### 4.9. Eş zamanlı konuşmaların işlenmesi

Exclusive diarization, normal konuşma metnini iki kişiye birden atamamak için kullanılır. Non-exclusive diarization ise eş zamanlı konuşma bölgelerini bulmak için ayrıca değerlendirilir.

Bir hata, embedding tabanlı bölme sonrasında overlap kayıtlarında eski konuşmacı etiketlerinin kalmasına neden oluyordu. Örneğin otomatik düzeltmeyle `SPEAKER_00`, `SPEAKER_03` olarak ayrıldığı halde overlap hâlâ `SPEAKER_00` gösterebiliyordu.

`update_overlap_speakers` fonksiyonu iyileştirilerek:

- overlap ile doğrudan kesişen düzeltilmiş tur varsa o tur,
- exclusive diarization nedeniyle kesişen tur yoksa zamansal olarak en yakın düzeltilmiş tur

kullanılmaya başlanmıştır. Böylece örnek sonuçtaki yanlış:

```text
SPEAKER_00 + SPEAKER_02
```

etiketi doğru düzeltilmiş kimliklerle:

```text
SPEAKER_02 + SPEAKER_03
```

olarak güncellenmiştir.

### 4.10. Konuşma dışı ses olayları

Ortam sesi analizi açıkken AST AudioSet modeli ses pencerelerini tarar. Sistem kahkaha, ağlama, öksürük, müzik, siren, araç, kapı, patlama ve benzeri seçili olayları kullanıcı dostu etiketlere dönüştürür.

`--skip-events` yalnızca bu aşamayı kapatır. Transkripsiyon ve konuşmacı diarization çalışmaya devam eder. Bu nedenle doğruluk açısından en başarılı bulunan kullanım:

```powershell
python process_video.py input\test.mp4 --skip-events
```

olmuştur. Bu seçenek konuşmacı veya transkripsiyon modelini küçültmez; yalnızca gerekli olmayan ses olayı analizini atlar.

## 5. Video bazlı manuel konuşmacı düzeltmeleri

Tam otomatik diarization modelleri; kısa cümlelerde, benzer seslerde, arka plan müziğinde ve aynı anda konuşmada hata yapabilir. Bu nedenle modele özgü etiketleri kaynak koda sabitlemek yerine video bazlı düzeltme sistemi eklenmiştir.

Program varsayılan olarak şu dosyayı arar:

```text
output/<video_adı>/speaker_overrides.json
```

Örnek yapı:

```json
{
  "overrides": [
    {
      "start": 26.7,
      "end": 28.7,
      "speaker": "SPEAKER_02",
      "note": "All life is a game of luck."
    },
    {
      "start": 29.6,
      "end": 32.6,
      "speaker": "SPEAKER_04",
      "note": "A real man makes his own luck, Archie. Right, Dawson?"
    }
  ]
}
```

Bu düzeltmeler yalnızca ilgili videoya uygulanır. `SPEAKER_02` gibi etiketlerin başka videolarda başka kişileri temsil edebilmesi nedeniyle düzeltmeler genel kaynak kod kuralına dönüştürülmemiştir.

İstenirse farklı dosya şu seçenekle verilebilir:

```powershell
python process_video.py input\test.mp4 --speaker-overrides duzeltmeler.json
```

## 6. Çalışma sırasında bulunan ve giderilen hatalar

### 6.1. Proje yapısının belirsizliği

**Sorun:** Eski üç aşamalı betikler ile yeni uçtan uca betik aynı dizinde bulunuyor ve hangi dosyanın kullanılacağı anlaşılmıyordu.  
**Çözüm:** `process_video.py` önerilen tek giriş noktası olarak belirlendi; README ve gereksinim dosyası eklendi.

### 6.2. Yardım ekranının geç açılması

**Sorun:** `--help` çalıştırıldığında bile PyTorch, pyannote ve Whisper gibi ağır kütüphaneler yükleniyordu.  
**Çözüm:** Ağır import işlemleri ilgili fonksiyonların içine taşındı. Böylece yardım ve hafif kontroller model yüklemeden çalışır.

### 6.3. Kelime zamanı bulunmadığında boş sonuç

**Sorun:** Whisper segment metni üretip kelime zaman damgası üretmediğinde birleşik konuşmacı metni boş kalabiliyordu.  
**Çözüm:** Kelime zamanları yoksa segment başlangıç/bitiş zamanlarını kullanan geri dönüş yolu eklendi.

### 6.4. `np is not defined` hatası

**Sorun:** NumPy importu geç yüklemeye taşındıktan sonra ses olayı analizindeki `np.sqrt` ve `np.mean` çağrıları import olmadan çalışıyordu.  
**Belirti:** Program `[3/4] Searching for...` aşamasında `NameError` ile kapanıyordu.  
**Çözüm:** `detect_events` içine yerel `import numpy as np` eklendi ve regresyon kontrolü yapıldı.

### 6.5. Gereksiz TorchCodec ve standart sapma uyarıları

**Sorun:** Ses zaten FFmpeg ile belleğe alındığı halde TorchCodec hakkında uzun uyarılar görüntüleniyordu. Çok kısa ses parçalarında pyannote standart sapma uyarısı da terminali kirletiyordu.  
**Çözüm:** İlgili ve zararsız uyarılar modül bazında filtrelendi.

### 6.6. Hugging Face bağlantı hatalarının anlaşılmaz olması

**Sorun:** Model indirilemediğinde uzun ağ/proxy traceback çıktısı oluşuyordu.  
**Çözüm:** Model erişimi ve bağlantı problemleri kısa Türkçe hata mesajlarına dönüştürüldü. Kullanıcıya `hf auth login`, internet bağlantısı ve model koşullarını kontrol etmesi söylenir.

### 6.7. Çıktı yolunun çalışma klasörüne bağlı olması

**Sorun:** Program başka klasörden çağrıldığında göreli yollar yanlış sonuç üretebilirdi.  
**Çözüm:** Proje kökü `Path(__file__).resolve().parent` ile belirlenerek çıktı ve yerel FFmpeg yolları buna bağlandı.

### 6.8. Yanlış konuşmacı bölünmesi

**Sorun:** Ham model aynı kişiye ait yakın cümleleri aynı etikette bulsa bile embedding tabanlı kısa segment bölmesi bazı parçaları farklı kimliklere ayırabiliyordu. Ayrıca kullanıcının dinleyerek doğruladığı bazı kısa bölümler otomatik model tarafından yanlış kişiye atanıyordu.  
**Çözüm:** Video bazlı `speaker_overrides.json` sistemi eklendi ve test videosundaki doğrulanmış aralıklar kaydedildi.

### 6.9. Ardışık cümlelerin gereksiz bölünmesi

**Sorun:** Aynı kişi art arda konuştuğunda her Whisper segmenti ayrı satır olabiliyordu.  
**Çözüm:** Arada konuşmacı değişimi yoksa ardışık segmentler tek blokta birleştirildi.

### 6.10. İlk kelimenin önceki konuşmacıda kalması

**Sorun:** Özellikle `--fast` modunda daha kaba zaman damgaları nedeniyle yeni cümlenin ilk kelimesi eski konuşmacıyla daha fazla örtüşebiliyordu.  
**Çözüm:** Kelimeleri bağımsız seçmek yerine segment genelini birlikte optimize eden, konuşmacı geçiş cezası kullanan dinamik atama eklendi.

## 7. Üretilen çıktı dosyaları

`input/test.mp4` için sonuçlar `output/test/` klasöründedir.

| Dosya | İçerik |
|---|---|
| `result.txt` | İnsan tarafından okunabilir nihai zaman çizelgesi |
| `result.json` | Konuşmalar, olaylar, overlap ve raporlarla birleşik sonuç |
| `transcript.json` | Whisper segmentleri ve kelime zaman damgaları |
| `diarization_raw.json` | Pyannote’ın düzeltme öncesi konuşmacı segmentleri |
| `diarization.json` | Düzeltme sonrası konuşmacı segmentleri ve overlap’ler |
| `speaker_correction_report.json` | Embedding tabanlı düzeltme ayrıntıları |
| `speaker_overrides.json` | Kullanıcı tarafından doğrulanmış video bazlı düzeltmeler |
| `sound_events.json` | Ses olayı sonuçları ve önbellek anahtarı |

## 8. Otomatik testler

`tests/test_process_video.py` içinde altı davranış testi bulunmaktadır:

1. Kelime zamanları yoksa segment metninin korunması.
2. Kelimenin en büyük zaman örtüşmesine sahip konuşmacıya atanması.
3. Manuel düzeltmenin otomatik segmenti doğru noktada bölebilmesi.
4. Ardışık Whisper segmentlerinin konuşmacı aynıysa birleştirilmesi.
5. Overlap’in en yakın düzeltilmiş konuşmacı kimliğini kullanması.
6. Cümle başındaki küçük zaman kaymasının ilk kelimeyi önceki kişide bırakmaması.

Test komutu:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Son doğrulamada altı testin tamamı başarıyla geçmiştir. Ayrıca `py_compile` ile ana dosya ve test dosyaları sözdizimi kontrolünden geçirilmiştir.

## 9. Kullanım seçenekleri

### Önerilen doğru kullanım

```powershell
python process_video.py input\test.mp4 --skip-events
```

Bu kullanım `medium` Whisper modelini, beam size 5’i ve tam konuşmacı düzeltme akışını korur; yalnızca ortam sesi sınıflandırmasını atlar.

### Tam analiz

```powershell
python process_video.py input\test.mp4
```

Konuşma, konuşmacılar, overlap ve ortam sesleri birlikte analiz edilir.

### Hızlı analiz

```powershell
python process_video.py input\test.mp4 --fast --skip-events
```

Whisper `small` ve beam size 1 kullanılır. Daha hızlıdır ancak kelime ve konuşmacı sınırı doğruluğu varsayılan çalışmadan düşük olabilir.

### Bilinen konuşmacı sayısı

```powershell
python process_video.py input\test.mp4 --speakers 5 --skip-events
```

Gerçek konuşmacı sayısı güvenilir biçimde biliniyorsa otomatik küme sayısı tahminindeki belirsizliği azaltabilir.

## 10. Performans değerlendirmesi

Mevcut sistem CPU üzerinde çalışmaktadır. CPU modunda:

- ses olayı analizini kapatmak güvenli bir hız kazanımı sağlar,
- `int8` Whisper hesaplaması zaten kullanılmaktadır,
- `--fast` daha fazla hız sağlar fakat model ve beam ayarını değiştirdiği için aynı doğruluğu garanti etmez,
- CUDA destekli NVIDIA GPU ve uygun PyTorch kurulumu ilk çalışmayı önemli ölçüde hızlandırabilir,
- aynı videoyu tekrar işlerken diarization ve transkripsiyon önbelleği eklemek gelecekte en güvenli hızlandırma olacaktır.

Projede şu anda yalnızca ses olayı sonuçları için önbellek bulunmaktadır. Diarization ve transkripsiyon sonuçlarını ayar/video özetiyle doğrulayıp yeniden kullanacak bir önbellek henüz uygulanmamıştır.

## 11. Bilinen sınırlamalar

1. Konuşmacı etiketleri gerçek kişi adı değil, video içi küme kimliğidir.
2. Çok kısa ifadelerde güvenilir ses embedding’i üretilemeyebilir.
3. Aynı anda konuşma, müzik ve gürültü diarization doğruluğunu düşürebilir.
4. Benzer sesli kişilerin kümeleri birleşebilir veya tek kişinin sesi birden fazla kümeye ayrılabilir.
5. Whisper kelime zaman damgaları yaklaşık değerlerdir; akustik konuşmacı sınırıyla birebir uyuşmayabilir.
6. `--fast` doğruluk-hız değiş tokuşu içerir.
7. İlk model indirmesi için internet ve Hugging Face model erişimi gerekir.
8. CPU kullanımında ilk tam analiz uzun sürebilir.
9. Manuel override dosyaları zaman aralığına bağlıdır; video kesilir veya zaman çizelgesi değişirse güncellenmeleri gerekir.

## 12. Önerilen sonraki geliştirmeler

### 12.1. Diarization ve transkripsiyon önbelleği

Video boyutu, değiştirilme zamanı, model adı, dil, cihaz ve konuşmacı ayarlarından bir önbellek anahtarı oluşturulabilir. Girdi ve ayarlar değişmediyse pahalı model aşamaları atlanabilir. Bu, özellikle konuşmacı düzeltmelerini deneyerek geliştirme sırasında büyük zaman kazandırır.

### 12.2. Konuşmacılara gerçek isim verme

`SPEAKER_00` gibi etiketleri `Jack`, `Rose` veya kullanıcı tarafından girilen adlarla eşleştiren video bazlı bir isim dosyası eklenebilir.

### 12.3. Görsel düzeltme arayüzü

Zaman çizelgesinde segment seçip konuşmacı değiştirmeye yarayan küçük bir arayüz, `speaker_overrides.json` dosyasını elle yazma ihtiyacını ortadan kaldırabilir.

### 12.4. Güven puanları

Her konuşma segmentine diarization örtüşmesi, embedding benzerliği ve kelime atama kararlılığından üretilen bir güven puanı eklenebilir. Böylece yalnızca düşük güvenli bölümler insan tarafından kontrol edilir.

### 12.5. GPU kurulumu

Bilgisayarda desteklenen NVIDIA donanımı varsa CUDA uyumlu PyTorch ve CTranslate2 kurulumu araştırılabilir. Bu işlem sürücü ve donanım uyumluluğu doğrulandıktan sonra yapılmalıdır.

## 13. Sonuç

Çalışma sonunda proje, birbirinden kopuk deneysel betiklerden daha anlaşılır tek giriş noktalı bir video-ses analiz sistemine dönüştürülmüştür. Ana akış; ses çıkarma, konuşmacı diarization, embedding tabanlı düzeltme, Whisper transkripsiyonu, kelime-konuşmacı eşleştirmesi, overlap analizi, isteğe bağlı ses olayı tespiti ve çoklu çıktı üretimini tek komutta gerçekleştirmektedir.

En önemli iyileştirmeler; kelime zamanları bulunmadığında sonucun kaybolmaması, ilk kelime sınır kaymalarının segment genelinde düzeltilmesi, ardışık aynı konuşmacıların birleştirilmesi, overlap etiketlerinin son konuşmacı kimlikleriyle uyumlu tutulması ve video bazlı kullanıcı doğrulamalarının kaynak kodu kirletmeden uygulanabilmesidir.

Mevcut test videosunda en dengeli kullanım biçimi:

```powershell
python process_video.py input\test.mp4 --skip-events
```

olarak belirlenmiştir. Bu komut, daha doğru `medium` transkripsiyon modelini ve tam konuşmacı analizini korurken kullanılmayan ortam sesi sınıflandırmasının işlem süresini ortadan kaldırır.

## 13. Kaynak kod sorumluluk matrisi

| Dosya | Rol | Değişiklikte dikkat edilecek nokta |
|---|---|---|
| `process_video.py` | Ana uçtan uca orkestrasyon | CLI geriye uyumluluğu, çıktı şeması ve model yükleme maliyeti |
| `speaker_correction.py` | Embedding tabanlı yeniden atama, birleştirme ve bölme | Eşik kalibrasyonu ve kısa segment güvenilirliği |
| `tests/test_process_video.py` | Saf davranış regresyonları | Her hata düzeltmesine karşı test eklenmesi |
| `transcribe.py` | Eski deneysel ASR akışı | Ana üretim yolu değildir |
| `diarize.py` | Eski deneysel diarization akışı | Ana üretim yolu değildir |
| `merge_results.py` | Eski deneysel birleştirme | Ana algoritmayla karıştırılmamalıdır |
| `requirements.txt` | Bağımlılık sürümleri | Model/kütüphane uyumu ve Python sürümü |

## 14. Kritik fonksiyonlar

| Fonksiyon | Sorumluluk |
|---|---|
| `find_ffmpeg` | FFmpeg yürütülebilir dosyasını öncelik sırasıyla bulur |
| `load_audio` | Videoyu 16 kHz mono PCM olarak belleğe alır |
| `diarize` | Pyannote konuşmacı turlarını üretir |
| `correct_diarization` | Embedding tabanlı düzeltme zincirini yürütür |
| `transcribe` | Faster-Whisper segment ve kelimelerini üretir |
| `assign_word_speakers` | Dinamik programlama ile kelime-konuşmacı dizisini seçer |
| `build_speaker_transcript` | Atamaları ardışık konuşma bloklarına dönüştürür |
| `update_overlap_speakers` | Overlap kimliklerini düzeltilmiş turlarla eşler |
| `detect_events` | AST ile seçili dış ses olaylarını tarar |
| `save_outputs` | Nihai JSON ve metin paketini yazar |

### 14.1. `speaker_correction.py` iç işlem sırası

1. Yeterince uzun segmentlerden embedding çıkarılır.
2. Konuşmacı prototipleri oluşturulur.
3. Yüksek benzerlik ve yeterli fark koşulunu sağlayan segmentler yeniden atanır.
4. Aynı etiket altında iki farklı ses kümesi bulunursa konservatif bölme denenir.
5. Birbirine çok benzeyen ve overlap ile çelişmeyen konuşmacı kümeleri birleştirilir.
6. Güvenilir A-B-A kısa kesintileri düzeltilir.
7. Aynı konuşmacının bitişik turları birleştirilir.
8. Yapılan işlemler raporlanır.

## 15. Teknik karar kayıtları

| Karar | Gerekçe | Sonuç |
|---|---|---|
| Tek giriş noktası olarak `process_video.py` | Eski üç betikli akışın yarattığı belirsizliği kaldırmak | Kurulum ve kullanım sadeleşti |
| Sesin FFmpeg ile belleğe alınması | Codec/TorchCodec bağımlılığını azaltmak | Tek PCM kaynağı ve daha öngörülebilir giriş |
| Ham diarization çıktısının korunması | Otomatik düzeltmeleri denetlenebilir kılmak | Önce/sonra karşılaştırması mümkün |
| Video bazlı override | Konuşmacı etiketlerinin videolar arasında sabit olmaması | Düzeltme kapsamı güvenli biçimde yerel |
| Viterbi benzeri kelime ataması | Tek kelimelik sınır sıçramalarını azaltmak | Cümle içi konuşmacı sürekliliği |
| Dış ses analizinin isteğe bağlı olması | CPU maliyetini kontrol etmek | `--skip-events` ile güvenli hız kazanımı |

## 16. Sorun giderme

| Belirti | Olası neden | İlk kontrol |
|---|---|---|
| FFmpeg bulunamadı | `PATH` veya `FFMPEG_PATH` yanlış | `ffmpeg.exe` konumunu ve ortam değişkenini doğrulayın |
| Model indirilemiyor | Ağ, proxy, token veya model koşulları | `hf auth login` ve model erişim onayını kontrol edin |
| CUDA seçilemiyor | CPU PyTorch veya uyumsuz sürücü | `torch.cuda.is_available()` ve PyTorch/CUDA eşleşmesini doğrulayın |
| Yanlış konuşmacı | Kısa segment, benzer ses veya yanlış kişi sayısı | Ham/nihai diarization, `--speakers` ve override kullanımını inceleyin |
| İlk kelime yanlış kişide | Zaman damgası sınır kayması | Varsayılan modu ve ilgili regresyon testini kullanın |
| İşlem çok yavaş | CPU, `medium` model veya olay analizi | Önce `--skip-events`; gerekirse `--fast` ve doğruluk kontrolü |
| Olay sonucu eski | Önbellek kullanıldı | `--recompute-events` ile yeniden tarayın |

## 17. Sürümleme ve güncelleme protokolü

1. Davranış değişikliğini ve etkilediği çıktı alanlarını tanımlayın.
2. Mevcut örnek üzerinde önce/sonra sonuçlarını saklayın.
3. En az bir regresyon testi ekleyin veya mevcut testi güncelleyin.
4. CLI varsayılanı değişiyorsa `README.md`, master rapor ve teslim notunu birlikte güncelleyin.
5. Model veya eşik değişiyorsa kullanılan sürüm ve ayarları sonuç metadata alanlarına ekleyin.
6. Tüm testleri ve en az bir uçtan uca örneği çalıştırın.
7. Doğrulama tarihini, kullanılan ortamı ve bilinen sapmaları kaydedin.

---

**Belge bakım kuralı:** Mimari, CLI, çıktı şeması, model seçimi, eşikler veya çalışma adımları değiştiğinde bu master rapor da aynı değişiklik kapsamında güncellenmelidir.
