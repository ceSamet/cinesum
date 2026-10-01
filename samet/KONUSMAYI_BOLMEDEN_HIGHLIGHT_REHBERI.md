# Konuşmayı Bölmeden Highlight Üretme Rehberi

Bu rehber, SceneMind'in özet videoyu oluştururken bir konuşmayı cümlenin
ortasında kesmemek için kullandığı mimariyi açıklar. Anlatım doğrudan güncel kod
akışına dayanır ve kritik fonksiyonlardan sadeleştirilmiş örnekler içerir.

## 1. Problem nedir?

Kamera cut'ları ile konuşma sınırları her zaman aynı noktada değildir.

Örneğin kamera görüntüsü 15. saniyede değişebilir fakat konuşmacının cümlesi
17.4. saniyeye kadar devam edebilir:

```text
Kamera cut'ı:  10.0 ├──────────────┤ 15.0
Konuşma:            12.3 ├────────────────────┤ 17.4
```

Özet doğrudan `10.0–15.0` aralığını keserse cümle yarıda kalır. SceneMind bunun
yerine sınırı konuşmanın sonuna kadar genişletir:

```text
Güvenli özet:  10.0 ├───────────────────────────┤ 17.62
                                                     +0.22 sn tampon
```

Aynı işlem başlangıç sınırı için de uygulanır. Cut bir konuşmanın ortasında
başlıyorsa başlangıç, konuşmanın başından biraz önceye çekilir.

## 2. Kullanılan araçlar

| Araç/bileşen | Görevi |
|---|---|
| FFmpeg | Videodan WAV çıkarma ve seçilen aralıkları kesip birleştirme |
| Faster-Whisper | Konuşmaları metne çevirme ve başlangıç/bitiş zamanlarını üretme |
| VAD | Sessizlikleri ayırarak Whisper segment sınırlarını iyileştirme |
| `SpeechTurn` | Her konuşma parçasının zamanını ve metnini taşıyan veri modeli |
| `_speech_safe_bounds()` | Bir cut sınırı konuşmanın içindeyse dışarı genişletme |
| `_speech_safe_intervals()` | Seçilen aralıkları sıralama, birleştirme ve yeniden doğrulama |
| 0/1 Knapsack | Genişletilmiş gerçek süreleri seçim maliyeti olarak kullanma |
| `_make_highlight()` | Güvenli aralıkları FFmpeg `trim/atrim/concat` grafiğine dönüştürme |

Ana uygulama dosyaları:

```text
scene_captioner/audio_analysis.py  → ses çıkarma, Whisper ve SpeechTurn
scene_captioner/analyzer.py        → güvenli sınırlar, seçim maliyeti ve highlight
tests/test_core.py                 → konuşma sınırlarının bölünmediğini doğrulayan test
```

## 3. Uçtan uca veri akışı

```text
Kaynak video
    |
    v
FFmpeg ile 16 kHz mono WAV
    |
    v
Faster-Whisper + VAD
    |
    v
SpeechTurn(start_sec, end_sec, text)
    |
    +-------------------------------+
    |                               |
    v                               v
Her cut için güvenli sınır      Konuşmaların cut'lara atanması
summary_start/end              transcript alanı
    |
    v
Knapsack maliyeti güvenli süreden hesaplanır
    |
    v
Seçilmiş cut'lar
    |
    v
Aralıkları yeniden genişlet + yakın olanları birleştir
    |
    v
FFmpeg trim / atrim / concat
    |
    v
highlight.mp4
```

Burada aynı güvenlik kuralı iki farklı zamanda uygulanır:

1. **Seçimden önce:** Knapsack sahnenin genişletilmiş süresini maliyet olarak
   görsün diye.
2. **Video üretilmeden hemen önce:** Birleştirme sonrası oluşabilecek yeni
   sınırlar da konuşmanın içine düşmesin diye.

## 4. Sesin hazırlanması

Dosya: `scene_captioner/audio_analysis.py`

Kaynak videonun ses kanalı FFmpeg ile analiz edilebilir standart biçime
dönüştürülür:

```python
process = subprocess.run([
    "ffmpeg",
    "-hide_banner",
    "-loglevel", "error",
    "-y",
    "-i", str(video_path),
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    str(wav_path),
])
```

Önemli parametreler:

- `-vn`: Video görüntüsünü çıkarmaz, yalnız sesi işler.
- `-ac 1`: Sesi mono yapar.
- `-ar 16000`: Örnekleme hızını 16 kHz yapar.
- `pcm_s16le`: Whisper için kayıpsız PCM WAV üretir.

Bu aşama zaman eksenini değiştirmez. WAV üzerindeki `12.5` saniye kaynak
videodaki `12.5` saniyeye karşılık gelir.

## 5. Konuşma zamanlarının çıkarılması

Faster-Whisper her konuşma segmenti için başlangıç, bitiş ve metin döndürür.
SceneMind bunları `SpeechTurn` nesnelerine çevirir:

```python
@dataclass
class SpeechTurn:
    start_sec: float
    end_sec: float
    text: str
    speaker: str = "Konuşmacı 1"
    actor: str | None = None
```

Transkripsiyon ayarlarının ilgili bölümü:

```python
segments, info = model.transcribe(
    str(wav_path),
    language=language,
    beam_size=1,
    best_of=1,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 400},
    condition_on_previous_text=False,
)

turns = [
    SpeechTurn(
        round(float(item.start), 3),
        round(float(item.end), 3),
        item.text.strip(),
    )
    for item in segments
    if item.text.strip()
]
```

`vad_filter=True`, konuşma ile sessiz alanların ayrılmasına yardım eder.
`min_silence_duration_ms=400`, yaklaşık 400 milisaniyelik sessizliği segment
ayırmak için yeterli kabul eder.

Örnek Whisper çıktısı:

```text
SpeechTurn(12.300, 17.400, "Bu işi bugün bitirmemiz gerekiyor.")
SpeechTurn(18.100, 20.900, "Tamam, hemen başlayalım.")
```

Sistemin cümlenin dilbilgisel olarak bitip bitmediğini ayrıca anlamaya
çalışmadığına dikkat edilmelidir. Güvenli kesme kararı Whisper'ın ürettiği
segment zamanlarına dayanır.

## 6. Temel algoritma: güvenli sınır genişletme

Dosya: `scene_captioner/analyzer.py`

Ana fonksiyon `_speech_safe_bounds()`'dur:

```python
def _speech_safe_bounds(start, end, turns, duration, padding=0.22):
    start = max(0.0, float(start))
    end = min(float(duration), float(end))

    for _ in range(len(turns) + 1):
        previous = (start, end)

        for turn in turns:
            if turn.start_sec < start < turn.end_sec:
                start = max(0.0, turn.start_sec - padding)

            if turn.start_sec < end < turn.end_sec:
                end = min(float(duration), turn.end_sec + padding)

        if start == previous[0] and end == previous[1]:
            break

    return round(start, 3), round(end, 3)
```

Gerçek kod kayan nokta karşılaştırmasında `1e-6` toleransı kullanır. Yukarıdaki
örnek okunabilirlik için sadeleştirilmiştir.

### Başlangıç kontrolü

```python
if turn.start_sec < start < turn.end_sec:
    start = turn.start_sec - 0.22
```

Cut başlangıcı bir konuşmanın içindeyse başlangıç konuşmanın başından `0.22`
saniye önceye taşınır.

Örnek:

```text
Cut başlangıcı:       15.000
Konuşma:              13.400–16.200
Yeni başlangıç:       13.180
```

### Bitiş kontrolü

```python
if turn.start_sec < end < turn.end_sec:
    end = turn.end_sec + 0.22
```

Cut bitişi bir konuşmanın içindeyse bitiş konuşmanın sonundan `0.22` saniye
sonraya taşınır.

Örnek:

```text
Cut bitişi:           15.000
Konuşma:              12.300–17.400
Yeni bitiş:           17.620
```

### Video sınırlarının korunması

Genişletme kaynak videonun dışına çıkamaz:

```python
start = max(0.0, turn.start_sec - padding)
end = min(duration, turn.end_sec + padding)
```

Dolayısıyla negatif başlangıç veya video süresinden büyük bitiş üretilmez.

## 7. Neden algoritma döngülü çalışıyor?

Tek geçiş bazı zincirleme konuşma durumlarında yeterli olmayabilir.

Örnek:

```text
İlk cut:       10.0 ├────────┤ 15.0
Konuşma A:          14.0 ├────────┤ 17.0
Konuşma B:                     16.8 ├────────┤ 19.5
```

İlk kontrol bitişi Konuşma A nedeniyle `17.22` yapar. Bu yeni sınır artık
Konuşma B'nin içinde kalır. İkinci kontrol bitişi `19.72` yapar.

Bu nedenle fonksiyon sınırlar sabitlenene kadar tekrar eder:

```python
for _ in range(len(turns) + 1):
    previous = (start, end)
    # Bütün konuşmaları kontrol et.
    ...
    if sınırlar_değişmediyse:
        break
```

Üst sınırın `len(turns) + 1` olması sonsuz döngü ihtimalini engeller. Her
genişleme sınırı yalnız dışarı doğru taşıdığı için algoritma sonunda sabitlenir.

## 8. Neden 0.22 saniye tampon var?

Whisper zaman damgaları mükemmel örnek hassasiyetinde değildir. Ayrıca sesin
nefes, hece sonu veya doğal duraklama bölümü segment sınırına çok yakın olabilir.

`0.22` saniyelik tampon:

- Son hecenin sert kesilme ihtimalini azaltır.
- FFmpeg ve codec zaman tabanı yuvarlamalarına küçük güvenlik alanı verir.
- Cut geçişinin konuşmanın tam dibine oturmasını önler.
- Gereksiz yere uzun sessizlik eklemeyecek kadar küçüktür.

Bu değer sabit varsayılandır; fonksiyonun `padding` parametresi değiştirilerek
ayarlanabilir.

## 9. Güvenli sürelerin her cut'a yazılması

Transkripsiyon bittikten sonra bütün cut'lar için güvenli sınırlar önceden
hesaplanır:

```python
for result in results:
    result.summary_start_sec, result.summary_end_sec = _speech_safe_bounds(
        result.start_sec,
        result.end_sec,
        turns,
        duration,
    )
```

Bir sahne kaydında iki farklı zaman çifti bulunmasının nedeni budur:

```json
{
  "start_sec": 10.0,
  "end_sec": 15.0,
  "summary_start_sec": 10.0,
  "summary_end_sec": 17.62
}
```

- `start_sec/end_sec`: Gerçek kamera cut sınırları
- `summary_start_sec/summary_end_sec`: Highlight için konuşma güvenli sınırlar

Arayüz sahneyi gerçek cut sınırlarıyla gösterebilir; özet üretimi güvenli
sınırları kullanır.

## 10. Knapsack neden güvenli süreyi kullanıyor?

Sınır genişletmesi sahnenin gerçek highlight maliyetini artırabilir. Beş
saniyelik bir kamera cut'ı, konuşmayı tamamlamak için sekiz saniyelik klibe
dönüşebilir.

Bu nedenle seçim maliyeti ham cut süresinden hesaplanmaz:

```python
safe_start = (
    scene.summary_start_sec
    if scene.summary_start_sec is not None
    else scene.start_sec
)
safe_end = (
    scene.summary_end_sec
    if scene.summary_end_sec is not None
    else scene.end_sec
)
costs.append(max(1, math.ceil(safe_end - safe_start)))
```

Örnek:

```text
Ham cut süresi:       5.00 sn
Güvenli aralık:       10.00–17.62
Knapsack maliyeti:    ceil(7.62) = 8 sn
```

Böylece seçim algoritması konuşmayı tamamlamanın getireceği süreyi önceden
hesaba katar. Aksi halde teoride bütçeye sığan sahneler genişletildikten sonra
özet süresini ciddi biçimde aşabilirdi.

## 11. Seçilen aralıkların birleştirilmesi

Seçimden sonra `_speech_safe_intervals()` çalışır:

```python
def _speech_safe_intervals(selected, turns, duration):
    intervals = []

    for scene in selected:
        start = scene.summary_start_sec
        end = scene.summary_end_sec
        intervals.append(_speech_safe_bounds(start, end, turns, duration))

    intervals.sort()

    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 0.08:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [
        _speech_safe_bounds(start, end, turns, duration)
        for start, end in merged
    ]
```

### Kronolojik sıralama

Paralon veya Knapsack sahneleri puana göre seçse bile highlight zaman sırasını
bozmaz. Aralıklar `intervals.sort()` ile kronolojik hale getirilir.

### Örtüşen aralıklar

İki seçilmiş cut konuşma genişletmesi nedeniyle örtüşebilir:

```text
Aralık A:  10.00–17.62
Aralık B:        16.80–22.00
Birleşik:  10.00–22.00
```

Bunlar ayrı ayrı kesilse ortak bölüm videoda iki kez görünürdü. Kod örtüşen
aralıkları tek klibe dönüştürür.

### Çok yakın aralıklar

İki aralık arasında en fazla `0.08` saniye varsa yine birleştirilir:

```python
start <= previous_end + 0.08
```

Bu, neredeyse bitişik iki klip arasında oluşabilecek görüntü/ses tıklamasını ve
gereksiz concat geçişini azaltır.

### Birleştirme sonrası ikinci kontrol

Birleştirilmiş aralıklar `_speech_safe_bounds()` fonksiyonundan tekrar geçirilir.
Bu savunmacı kontrol, yeni oluşan dış sınırların hiçbir konuşmanın içinde
kalmadığını garanti etmeye çalışır.

## 12. FFmpeg ile gerçek highlight üretimi

Dosya: `scene_captioner/analyzer.py`, `_make_highlight()`

Güvenli aralıklar hazırlandıktan sonra her aralık için görüntü ve ses ayrı ayrı
kesilir:

```python
video_filter = (
    f"[0:v]trim=start={start}:end={end},"
    f"setpts=PTS-STARTPTS[v{index}]"
)

audio_filter = (
    f"[0:a]atrim=start={start}:end={end},"
    f"asetpts=PTS-STARTPTS[a{index}]"
)
```

Filtrelerin görevi:

- `trim`: Görüntüyü güvenli başlangıç ve bitişten keser.
- `atrim`: Sesi aynı zaman aralığından keser.
- `setpts=PTS-STARTPTS`: Her video parçasının zamanını sıfırdan başlatır.
- `asetpts=PTS-STARTPTS`: Ses parçasının zamanını sıfırdan başlatır.

Parçalar şu sırayla concat filtresine verilir:

```text
[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[v][a]
```

Son FFmpeg ayarları:

```python
command.extend([
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "23",
    "-c:a", "aac",
    "-movflags", "+faststart",
    str(output_path),
])
```

- `libx264`: Yaygın desteklenen H.264 çıktısı
- `veryfast`: Hız ve dosya boyutu arasında pratik denge
- `crf 23`: Varsayılan düzeyde görüntü kalitesi
- `aac`: Web uyumlu ses
- `faststart`: MP4 metadata'sını başa alarak tarayıcı oynatımını hızlandırır

Video ses kanalı içermiyorsa aynı akış yalnız görüntü filtreleriyle çalışır ve
concat filtresinde `a=0` kullanılır.

## 13. Baştan sona örnek

Kaynak veriler:

```text
Video süresi: 30 saniye

Cut 1:  5.0–10.0
Cut 2: 10.0–15.0  ← özete seçildi
Cut 3: 15.0–20.0  ← özete seçildi

Konuşma A:  9.2–12.5
Konuşma B: 14.4–17.8
```

### Cut 2 güvenli sınırı

```text
Başlangıç 10.0, Konuşma A'nın içinde → 9.2 - 0.22 = 8.98
Bitiş 15.0, Konuşma B'nin içinde    → 17.8 + 0.22 = 18.02

Cut 2 güvenli aralığı: 8.98–18.02
```

### Cut 3 güvenli sınırı

```text
Başlangıç 15.0, Konuşma B'nin içinde → 14.4 - 0.22 = 14.18
Bitiş 20.0, konuşmanın içinde değil   → 20.0

Cut 3 güvenli aralığı: 14.18–20.0
```

### Birleştirme

```text
Cut 2:  8.98 ├────────────────────┤ 18.02
Cut 3:              14.18 ├─────────────┤ 20.0

Sonuç:  8.98 ├──────────────────────────┤ 20.0
```

FFmpeg tek bir `8.98–20.0` klibi üretir. İki seçilmiş sahne olmasına rağmen
örtüşen bölüm tekrarlanmaz ve iki konuşma da tamamlanır.

## 14. Analiz sonucunda nasıl kontrol edilir?

Her cut'ın güvenli sınırları `analysis.json` içinde görülebilir:

```json
{
  "index": 12,
  "start_sec": 140.0,
  "end_sec": 145.0,
  "summary_start_sec": 138.74,
  "summary_end_sec": 147.31,
  "selected": true
}
```

Nihai kullanılan klipler `selection` metadata'sında saklanır:

```json
{
  "speech_safe_clip_count": 6,
  "speech_safe_duration_sec": 94.7,
  "speech_safe_intervals": [
    {"start_sec": 8.98, "end_sec": 20.0},
    {"start_sec": 54.2, "end_sec": 67.8}
  ]
}
```

Kontrol edilmesi gereken alanlar:

- `scenes[].start_sec/end_sec`: Kamera cut sınırları
- `scenes[].summary_start_sec/summary_end_sec`: Konuşma güvenli sınırlar
- `selection.duration_budget_sec`: Knapsack bütçesi
- `selection.selected_duration_sec`: Seçimde kullanılan tamsayı maliyet toplamı
- `selection.speech_safe_duration_sec`: Gerçek birleştirilmiş highlight süresi
- `selection.speech_safe_intervals`: FFmpeg'e verilen nihai aralıklar

## 15. Mevcut birim testi

`tests/test_core.py` içinde her nihai sınırın konuşmaların dışında kaldığını
doğrulayan test bulunur:

```python
scenes = [
    Scene(start=1.0, end=2.0),
    Scene(start=5.0, end=6.0),
]

turns = [
    SpeechTurn(0.8, 2.4, "ilk konuşma"),
    SpeechTurn(5.6, 6.3, "ikinci konuşma"),
]

intervals = _speech_safe_intervals(scenes, turns, duration=8.0)

for start, end in intervals:
    for turn in turns:
        assert not (turn.start_sec < start < turn.end_sec)
        assert not (turn.start_sec < end < turn.end_sec)
```

Testin garanti ettiği invariant şudur:

```text
Hiçbir nihai klip başlangıcı veya bitişi,
algılanmış bir SpeechTurn aralığının içinde olamaz.
```

Testleri çalıştırmak için:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 16. Tasarımın güçlü tarafları

- Konuşma güvenliği seçimden sonra eklenen kozmetik bir işlem değildir;
  Knapsack maliyetine önceden yansıtılır.
- Hem başlangıç hem bitiş sınırı korunur.
- Zincirleme/örtüşen konuşmalar iteratif olarak çözülür.
- Örtüşen seçilmiş sahneler videoda tekrar edilmez.
- Video süresinin dışına taşan zamanlar üretilmez.
- Ses ve görüntü aynı zaman aralığından kesilir.
- FFmpeg'den hemen önce aralıklar tekrar doğrulanır.
- Nihai aralıklar JSON metadata'sında denetlenebilir biçimde tutulur.

## 17. Bilinen sınırlar

Bu mekanizma güçlüdür ancak doğruluğu Whisper segmentlerine bağlıdır.

### Whisper yanlış sınır üretirse

Whisper konuşmanın sonunu erken işaretlerse algoritma gerçek cümlenin kalanını
bilemez. Kod metindeki noktalama işaretinden bağımsız olarak yalnız zaman
aralıklarına bakar.

### Çok uzun Whisper segmentleri

Müzik, arka plan gürültüsü veya kesintisiz konuşma Whisper'ın çok uzun bir
segment üretmesine neden olabilir. Tek bir seçilmiş cut bu segmentin tamamını
alarak beklenenden uzun highlight oluşturabilir.

### Konuşma algılanamazsa

Ses yoksa veya transkripsiyon başarısızsa `turns` boş kalır. Bu durumda güvenli
sınırlar kamera cut sınırlarıyla aynı olur; sistem algılamadığı konuşmayı
koruyamaz.

### Semantik cümle tamamlama yapılmıyor

Bir Whisper segmenti dilbilgisel olarak iki cümle veya yarım bir düşünce
içerebilir. Mevcut sistem “cümlenin anlamı tamamlandı mı?” sorusunu LLM ile
kontrol etmez. Korunan birim Whisper'ın akustik segmentidir.

### Bütçe ile gerçek süre farkı

Knapsack her cut maliyetini ayrı ayrı yukarı yuvarlar. Daha sonra örtüşen
aralıklar birleştirildiğinden gerçek highlight süresi seçilen maliyet toplamından
daha kısa olabilir. Tersi yönde küçük codec/zaman tabanı farkları da görülebilir.

## 18. Geliştirme önerileri

### Kelime düzeyi zaman damgası

Faster-Whisper `word_timestamps=True` ile çalıştırılarak kesme sınırları kelime
düzeyinde incelenebilir. Bu, uzun segmentlerde daha hassas doğal durak seçimi
sağlayabilir.

### Noktalama destekli doğal sınır

Bir cut sonu konuşmanın içindeyse yalnız segment sonuna gitmek yerine sonraki
nokta, soru işareti veya yeterli sessizlik aranabilir. Bunun için kelime zaman
damgası gerekir.

### Maksimum genişleme politikası

Bozuk bir Whisper segmentinin özeti onlarca saniye uzatmasını engellemek için
`max_expansion_sec` sınırı eklenebilir. Bu durumda sınır aşıldığında en yakın
güvenli sessizlik seçilmelidir.

### Ses enerjisiyle ikinci doğrulama

Whisper sınırının çevresinde RMS/VAD kontrolü yapılarak kesme noktasının gerçekten
sessiz alana denk gelip gelmediği doğrulanabilir.

### Daha kapsamlı testler

Eklenebilecek test durumları:

- Zincirleme örtüşen iki konuşma
- Video sıfırına taşan başlangıç
- Video sonuna taşan bitiş
- Birbiriyle örtüşen seçilmiş üç cut
- Ses kanalı olmayan video
- Çok uzun tek Whisper segmenti
- FFmpeg çıktısında gerçek A/V süre eşitliği

## 19. Kısa özet

SceneMind konuşmayı bölmemeyi şu dört kuralla sağlar:

```text
1. Whisper konuşmaların zaman aralıklarını çıkarır.
2. Cut sınırı konuşmanın içindeyse sınır konuşmanın dışına genişletilir.
3. Knapsack genişletilmiş süreyi maliyet olarak kullanır.
4. FFmpeg'den önce aralıklar birleştirilip yeniden kontrol edilir.
```

Bu nedenle özet videosu yalnızca seçilmiş kamera cut'larını mekanik biçimde
kesmez; algılanan konuşmaların tamamını koruyan yeni zaman aralıkları üretir.
