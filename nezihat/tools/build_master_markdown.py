from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PROJE_TEKNIK_RAPORU.md"
OUTPUT = ROOT / "CineSum_Audio_Master_Proje_Raporu.md"


MASTER_FRONT = r"""# CineSum Audio Master Proje Raporu

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

"""


MASTER_BACK = r"""

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
"""


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    # Eski belgenin H1 başlığını, master belgenin içinde yinelenmemesi için kaldır.
    source_lines = source.splitlines()
    if source_lines and source_lines[0].startswith("# "):
        source_lines = source_lines[1:]
    source = "\n".join(source_lines).strip()
    OUTPUT.write_text(MASTER_FRONT + source + MASTER_BACK, encoding="utf-8", newline="\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
