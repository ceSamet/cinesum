# SceneMind — Yerel video özetleyici

Projenin mimarisi, analiz hattı, API'leri, veri şeması, sınırları ve geliştirme
rehberi için [Proje Master Dokümanı](PROJE_MASTER_DOKUMANI.md) dosyasına bakın.

SceneMind videoyu sabit saniye aralıklarına bölmez. Videonun kendi görsel değişim
dağılımından adaptif kesme eşiği çıkarır; her sahneyi görüntü, konuşma, konuşmacı,
ses olayı, yüz ve metin embedding'leriyle analiz eder.

## Üretilen bilgiler

- Adaptif sahne başlangıç/bitişleri ve ana kare
- LLaVA ile Türkçe görsel sahne açıklaması
- Faster-Whisper ile zaman kodlu konuşma dökümü
- Akustik voice-print kümeleme ile konuşmacı sayısı/etiketleri
- AudioSet AST ile patlama, bağırma, silah, ağlama, müzik gibi olaylar
- YuNet + SFace ile tekrarlayan oyuncular, sahne ve yaklaşık ekran süreleri
- Çok dilli MiniLM + yerel SQLite vektör RAG + Paralon Qwen 3.8 27B destekli Knapsack seçimi
- Her sahne için konuşma, ses olayı, karakter, görsel değişim, semantik özgünlük ve anlatı konumu puanları
- Seçilmiş sahnelerden otomatik `highlight.mp4`

## Model seti

Varsayılan LLaVA modeli `llava-interleave-qwen-0.5b-hf`'dir. 1.5-7B yerine bu
modelin seçilme nedeni RTX 3060 Laptop'ın 6 GB VRAM'inde makul hız vermesidir.
Model dosyaları repo dışındaki ortak cache yerine proje içindeki `models/`
dizinine kurulur.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-models.txt
.venv/bin/python scripts/download_models.py
```

Yalnız belirli modelleri indirmek için:

```bash
.venv/bin/python scripts/download_models.py llava whisper audio-events embeddings faces
```

Toplam model alanı birkaç GB'dir. `models/` Git tarafından izlenmez.

## Web arayüzü

Linux/macOS:

```bash
./start_video_ui.sh
```

veya doğrudan:

```bash
.venv/bin/python -m scene_captioner.web_app
```

Windows'ta `start_video_ui.bat` kullanılabilir. Tarayıcı adresi:

```text
http://127.0.0.1:7860
```

Analiz arka planda tek işlik kuyrukta çalışır. Sayfa durum endpoint'ini izler;
önceki işler soldaki geçmişten yeniden açılabilir. Her işin çıktıları
`outputs/web/jobs/<job-id>/` altında bulunur:

- `analysis.json`
- `scene_embeddings.npy`
- `scene_rag.sqlite3` (başlık, görsel tanım, karakter, konuşma ve embedding belgeleri)
- `frames/`
- `audio.wav`
- `highlight.mp4`

## Performans notu

45 dakikalık bir bölümü 2–3 dakikada bitirmek bir hedef profildir, CPU garantisi
değildir. RTX 3060 üzerinde CUDA çalışmalı; LLaVA/Whisper GPU'da, sahne tespiti ve
yüz kümeleme CPU'da çalışır. `nvidia-smi` hata veriyorsa önce sürücü/kernel
eşleşmesini düzeltip sistemi yeniden başlatın. İlk çalıştırma model ısınması
nedeniyle sonraki çalıştırmalardan yavaştır. AudioSet taramasını kapatmak ve BLIP
seçmek en hızlı profildir.

## Mimari

```text
Video
 ├─ adaptif görsel farklar ──> sahne sınırları + ana kareler
 ├─ LLaVA/BLIP ──────────────> sahne açıklaması
 ├─ Faster-Whisper ──────────> replikler ──> voice-print kümeleri
 ├─ AudioSet AST ────────────> ses olayları
 └─ YuNet + SFace ───────────> oyuncu kümeleri ve ekran süresi
                 ↓
        multilingual MiniLM embedding
                  ↓
       altı bileşenli önem puanı + yerel SQLite RAG
                  ↓
    ilk 0/1 Knapsack adayları
                  ↓
 Paralon Qwen 3.8 bağlam incelemesi
                  ↓
 süre maliyeti altında son 0/1 Knapsack
                 ↓
        zaman çizelgesi + highlight.mp4
```

Yüz kümeleri gerçek oyuncu adı üretmez; yalnız aynı görünen kişileri `Oyuncu 1`,
`Oyuncu 2` şeklinde gruplar. İsim eşleştirmek için ayrıca film kadrosu verisi ve
yüz galerisi gerekir.

## Paralon yapılandırması

`.env` dosyasına Paralon Console'dan üretilen `prlc_...` anahtarını ekleyin:

```dotenv
PARALON_API_KEY=prlc_...
PARALON_BASE_URL=https://paraloncloud.com/v1
PARALON_MODEL=qwen3.8-27b
```

Anahtar yoksa veya uzak servis cevap vermezse analiz durmaz; yerel, zamansal
dengeli Knapsack sonucu kullanılır ve fallback nedeni `analysis.json` içindeki
`selection.llm_selection` alanına yazılır. Yerel LLM sahne adı üretimi kullanılmaz.
