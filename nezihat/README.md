# CineSum Audio

Bir videodaki konuşmaları yazıya döker, konuşmacıları ayırır ve isteğe bağlı olarak kahkaha, müzik ve benzeri ses olaylarını bulur.

## Hızlı kullanım

PowerShell'de proje klasöründe:

```powershell
.\.venv\Scripts\Activate.ps1
hf auth login
python process_video.py input\test.mp4 --fast
```

Sonuçlar videonun adıyla `output` klasörüne yazılır. Örneğin `input\test.mp4` için ana sonuçlar:

- `output\test\result.txt`: okunabilir zaman çizelgesi
- `output\test\result.json`: tüm yapısal sonuç
- `output\test\transcript.json`: ham transkript
- `output\test\diarization.json`: konuşmacı bölümleri

Ortam seslerini taramadan daha hızlı denemek için:

```powershell
python process_video.py input\test.mp4 --fast --skip-events
```

Konuşmacı sayısı biliniyorsa doğruluğu artırmak için:

```powershell
python process_video.py input\test.mp4 --speakers 2
```

Tüm seçenekler:

```powershell
python process_video.py --help
```

## Proje düzeni

Güncel ve önerilen tek giriş noktası `process_video.py` dosyasıdır. `transcribe.py`, `diarize.py` ve `merge_results.py` eski, ayrı ayrı çalışan deneysel akıştır; normal kullanımda gerekmez. `speaker_correction.py`, ana akışın konuşmacı düzeltme yardımcısıdır.

## Gereksinimler

- Python 3.13
- FFmpeg (`.venv\Scripts\ffmpeg.exe`, sistem PATH'i veya `FFMPEG_PATH`)
- Hugging Face hesabı ve `pyannote/speaker-diarization-community-1` model erişimi
- İlk çalıştırmada modelleri indirmek için internet bağlantısı

Bir hata olursa önce `--fast --skip-events` ile temel konuşma akışını sınayın.
