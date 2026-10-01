# Thanos — birleşik video özetleme

`thanos/`, Gökdeniz'in CineSum analiz, önbellek, StoryScene, RAG/Paralon ve web akışını; Samet'in esnek içerik seçimi, tam konuşma turu koruması, sesli FFmpeg çıktısı ve yüz eşleştirmesini; Nezihat'ın pyannote konuşmacı ayrımı, embedding düzeltmesi ve kelime düzeyinde konuşmacı atamasını birleştirir. Kaynak projelerdeki dosyalar değiştirilmez. Bileşenlerin bağlantıları ve veri akışı için [ayrıntılı mimari belgesine](MIMARI_VE_CALISMA_AKISI.md) bakın.

## Çalıştırma

Proje kökünden `./thanos/start.sh` çalıştırın. Arayüz varsayılan olarak `http://127.0.0.1:8001` adresindedir; `THANOS_PORT` ve `THANOS_HOST` değiştirilebilir. Bağımlılıklar için kökteki `.venv` kullanılır; eksikler `thanos/requirements-core.txt` ve isteğe bağlı `requirements-llm.txt`, `requirements-vlm.txt` dosyalarındadır. FFmpeg/FFprobe sistemde kurulu olmalıdır.

CLIP ağırlığı `thanos/models/clip/` altındadır; Faster-Whisper small, BLIP, LLaVA ve OpenCV yüz modelleri `samet/models/` altından yeniden kullanılır. `quality` profilinin Whisper medium modeli ayrıca indirme gerektirebilir; çevrimdışı kullanımda `fast`/`balanced` seçin. Uzak LLM için `PARALON_API_KEY` ve diğer ayarlar kök `.env` veya `thanos/.env` üzerinden alınır; `local` modda anahtar gerekmez. Dış sağlayıcı başarısızsa yerel seçim korunur.

Torch GPU'yu görse bile Faster-Whisper'ın CUDA kütüphaneleri (`libcublas.so.12`, `libcudnn.so.9`) eksikse transkripsiyon otomatik CPU/int8 kullanır. GPU çalışma sırasında hata verirse yine CPU'da yeniden denenir. İsterseniz `THANOS_WHISPER_DEVICE=cpu` ile CPU'yu doğrudan seçebilirsiniz. Kod güncellemesinden sonra çalışan sunucuyu yeniden başlatın.

### Nezihat konuşmacı ayrımı

Pyannote modeli ayrı, isteğe bağlı bir bağımlılıktır:

```bash
.venv/bin/pip install -r thanos/requirements-diarization.txt
hf auth login
```

Hugging Face hesabında `pyannote/speaker-diarization-community-1` modelinin kullanım koşulları kabul edilmiş olmalı. `hf auth login` yerine `HF_TOKEN` ortam değişkeni veya kök/`thanos/.env` dosyası kullanılabilir. `THANOS_DIARIZATION_BACKEND=auto` (varsayılan), pyannote ve model erişimi hazırsa Nezihat yolunu çalıştırır. `THANOS_DIARIZATION_BACKEND=samet` eski hafif MFCC yolunu seçer; `pyannote` tercihi hazır değilse nedenini çıktı dosyasına kaydedip MFCC'ye döner. Nezihat modeli varsayılan olarak CPU'da çalışır; uygun CUDA kurulumu varsa `THANOS_DIARIZATION_DEVICE=cuda` kullanılabilir.

Konuşmacı sonucu `outputs/features/audio/<video>_people.json` içindedir. `backend` etkin yöntemi, `turns` Whisper kelimeleriyle hizalanmış etiketleri, `diarization_turns` ham zaman dilimlerini, `overlaps` eşzamanlı konuşmayı, `correction` embedding düzeltme raporunu gösterir. Yüz algılama başarısız olsa da ses tarafı analiz edilir. Mevcut Whisper transkripti ve konuşma kesme koruması değişmez.

## Ana akış

```text
Video yükleme → SHA-256 ile aynı dosyayı bul/önbelleği kullan
              → PySceneDetect + kareler + CLIP + ses + Faster-Whisper
              → Samet yüz kimlikleri + Nezihat pyannote konuşmacı ayrımı
                (hazır değilse Samet MFCC; tüm kişi aşaması opsiyonel)
              → StoryScene + yerel önem/aksiyon/diyalog puanları
              → içerik odaklı seçim + yumuşak zaman/olay tekrar cezası
              → istenirse SQLite RAG + Paralon yeniden sıralaması
              → hedefli ASR → Samet tam konuşma turu koruması
              → FFmpeg trim/atrim + concat → sesli MP4
```

Önem özetinde sabit üç perde yüzdeleri zorlanmaz; zayıf bir baş/orta/son diliminden sırf kota dolsun diye sahne seçilmez. Zamanın 10 diliminden birine %34'ün, aynı olay grubuna %28'in üzerinde yığılma olursa tekrar eden adayların değeri yumuşak biçimde düşer. Bu cezalar yeterince güçlü içeriği tamamen yasaklamaz. Samet'in 0,22 saniyelik payla genişlettiği Whisper konuşma turları **segment bütçe optimizasyonundan önce** süre maliyeti olarak uygulanır ve en son tekrar kontrol edilir; bütçe aşılırsa cümle ortasından kesmek yerine hedef süreye en çok yaklaşan tam aralık alt kümesi seçilir. Kısa kalan aksiyon/diyalog/özel özetler için uygun yedek sahnelerle güvenli süre doldurma yapılır. Çok kısa hedefe hiçbir tam cümle sığmazsa açık hata döner.

Analiz dosyaları `thanos/outputs/`, yüklenen videolar `thanos/dataset/video/` altındadır. İçerik hash'i aynı videonun yeniden yüklenmesinde eski alias'ı ve özellikleri kullanır. `THANOS_ENABLE_PEOPLE=false` yüz/konuşmacı aşamasını kapatır; diğer özetleme özellikleri çalışır. Bu dalın çıktısı `outputs/features/audio/*_people.json` ve isteğe bağlı `outputs/portraits/` altındadır.

TVSum üzerinde %15 süre bütçeli, 20 anotatörlü F1 ölçümü için [değerlendirme rehberine](TVSUM_DEGERLENDIRME.md) bakın. Kaynak TVSum videoları anotasyonlardan ayrı gerekir; başka bir `video1.mp4` dosyası otomatik olarak TVSum sayılmaz.

## Sınırlar ve doğrulama

- `../.venv/bin/python -m pytest -q tests` ile test edin (bu dizinde).
- Yerel 8 saniyelik sesli video denemesinde tam analiz ve 5 saniyelik H.264/AAC özet üretildi; aynı videoda ikinci analiz ağır aşamalarda cache hit verdi.
- Gerçek uzun filmlerde sahne kalitesi ve cümlelerin semantik tamamlığı manuel dinleme gerektirir; Whisper yanlış zaman damgası veya eksik konuşma verebilir. Son koruma yalnızca **tespit edilen** konuşma turlarını garanti eder.
- Paralon'un canlı yanıtı, `quality` profilindeki medium model ve Nezihat pyannote modelinin gerçek video üzerindeki sonucu bu birleşimde doğrulanmadı. Whisper giriş noktası `src/audio/whisper_transcriber.py`; mevcut transkript JSON şeması (`segments`/`start`/`end`/`words`) korunur.
