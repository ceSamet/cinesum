# Thanos — birleşik video özetleme

`thanos/`, Gökdeniz'in CineSum analiz, önbellek, StoryScene, RAG/Paralon ve web akışını; Samet'in esnek içerik seçimi, tam konuşma turu koruması, sesli FFmpeg çıktısı ve yüz/konuşmacı eşleştirmesiyle birleştirir. Kaynak projelerdeki dosyalar değiştirilmez. Nezihat'ın gelecek Whisper/diarization çalışması bu sürüme henüz eklenmedi.

## Çalıştırma

Proje kökünden `./thanos/start.sh` çalıştırın. Arayüz varsayılan olarak `http://127.0.0.1:8001` adresindedir; `THANOS_PORT` ve `THANOS_HOST` değiştirilebilir. Bağımlılıklar için kökteki `.venv` kullanılır; eksikler `thanos/requirements-core.txt` ve isteğe bağlı `requirements-llm.txt`, `requirements-vlm.txt` dosyalarındadır. FFmpeg/FFprobe sistemde kurulu olmalıdır.

CLIP ağırlığı `thanos/models/clip/` altındadır; Faster-Whisper small, BLIP, LLaVA ve OpenCV yüz modelleri `samet/models/` altından yeniden kullanılır. `quality` profilinin Whisper medium modeli ayrıca indirme gerektirebilir; çevrimdışı kullanımda `fast`/`balanced` seçin. Uzak LLM için `PARALON_API_KEY` ve diğer ayarlar kök `.env` veya `thanos/.env` üzerinden alınır; `local` modda anahtar gerekmez. Dış sağlayıcı başarısızsa yerel seçim korunur.

Torch GPU'yu görse bile Faster-Whisper'ın CUDA kütüphaneleri (`libcublas.so.12`, `libcudnn.so.9`) eksikse transkripsiyon otomatik CPU/int8 kullanır. GPU çalışma sırasında hata verirse yine CPU'da yeniden denenir. İsterseniz `THANOS_WHISPER_DEVICE=cpu` ile CPU'yu doğrudan seçebilirsiniz. Kod güncellemesinden sonra çalışan sunucuyu yeniden başlatın.

## Ana akış

```text
Video yükleme → SHA-256 ile aynı dosyayı bul/önbelleği kullan
              → PySceneDetect + kareler + CLIP + ses + Faster-Whisper
              → Samet yüz/konuşmacı kimlikleri (opsiyonel)
              → StoryScene + yerel önem/aksiyon/diyalog puanları
              → içerik odaklı seçim + yumuşak zaman/olay tekrar cezası
              → istenirse SQLite RAG + Paralon yeniden sıralaması
              → hedefli ASR → Samet tam konuşma turu koruması
              → FFmpeg trim/atrim + concat → sesli MP4
```

Önem özetinde sabit üç perde yüzdeleri zorlanmaz; zayıf bir baş/orta/son diliminden sırf kota dolsun diye sahne seçilmez. Zamanın 10 diliminden birine %34'ün, aynı olay grubuna %28'in üzerinde yığılma olursa tekrar eden adayların değeri yumuşak biçimde düşer. Bu cezalar yeterince güçlü içeriği tamamen yasaklamaz. Samet'in 0,22 saniyelik payla genişlettiği Whisper konuşma turları **en son** aşamada tekrar kontrol edilir; bütçe aşılırsa cümle ortasından kesmek yerine en düşük yararlı tam aralık çıkarılır. Çok kısa hedefe hiçbir tam cümle sığmazsa açık hata döner.

Analiz dosyaları `thanos/outputs/`, yüklenen videolar `thanos/dataset/video/` altındadır. İçerik hash'i aynı videonun yeniden yüklenmesinde eski alias'ı ve özellikleri kullanır. `THANOS_ENABLE_PEOPLE=false` yüz/konuşmacı aşamasını kapatır; diğer özetleme özellikleri çalışır. Bu dalın çıktısı `outputs/features/audio/*_people.json` ve isteğe bağlı `outputs/portraits/` altındadır.

## Sınırlar ve doğrulama

- `../.venv/bin/python -m pytest -q tests` ile test edin (bu dizinde).
- Yerel 8 saniyelik sesli video denemesinde tam analiz ve 5 saniyelik H.264/AAC özet üretildi; aynı videoda ikinci analiz ağır aşamalarda cache hit verdi.
- Gerçek uzun filmlerde sahne kalitesi ve cümlelerin semantik tamamlığı manuel dinleme gerektirir; Whisper yanlış zaman damgası veya eksik konuşma verebilir. Son koruma yalnızca **tespit edilen** konuşma turlarını garanti eder.
- Paralon'un canlı yanıtı ve `quality` profilindeki medium model bu birleşimde doğrulanmadı. Nezihat'ın ilerideki Whisper modülü için giriş noktası `src/audio/whisper_transcriber.py`; mevcut transkript JSON şeması (`segments`/`start`/`end`/`words`) korunmalıdır.
