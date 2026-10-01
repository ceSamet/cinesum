# CineSum

Uzun videoları bir kez görüntü, ses ve konuşma üzerinden analiz edip önbellekteki özelliklerden farklı süre ve kategorilerde özet video üreten proje.

## Geliştirmeye buradan devam edin

- [Güncel mimari, durum ve ajan devir belgesi](Documentations/CineSum_Current_State_and_Handoff.md)
- [Üç perde uygulama planı ve tamamlanma listesi](Documentations/CineSum_Three_Act_Implementation_Plan.md)
- [Tarihli uygulama ve doğrulama günlüğü](Documentations/CineSum_Implementation_Progress.md)
- [Ayrıntılı mimari ve sunum rehberi](Documentations/CineSum_Final_Sistem_Mimarisi_ve_Sunum_Rehberi.md)
- [Geniş teknik başvuru](Documentations/CineSum_AI_Grand_Master_Document.md)

1 Ekim 2026: Özet algoritması v3.9.0. Yerel modda iki gerçek filmden çalışan MP4 üretildi; `tests/` için 111 test ve 9 alt test geçti. Üç perde süreleri, korumalı olay zinciri ve kanıt varsa zorunlu dönüm noktası koruması çalışıyor. Dönüm noktalarının anlamsal doğruluğu ve ana karakter görünürlüğü hâlâ tamamlanmadı; ayrıntılar devir belgesinde.

## Yerel çalıştırma

Mevcut bağımlılıkları kurulu proje ortamında:

```powershell
python run_web_app.py
```

Başlatıcı `.venv/Scripts/python.exe` varsa bu ortamı kullanır. Web adresi `http://localhost:8000`. Varsayılan özet modu `local`; uzak LLM için `rag_llm` ve geçerli sağlayıcı yapılandırması gerekir. Ayar isimleri `.env.example` dosyasındadır; gizli değerleri dokümantasyona eklemeyin.

Doğrulanmış örnek çıktılar: [Avengers 300 sn](outputs/summaries/three_act_validation/recap_v5.mp4) ve [ikinci film 180 sn hedef](outputs/summaries/three_act_validation/video55_recap.mp4). Dosyalar mevcut yerel çalışma alanındadır; ilk çıktının gerçek süresi 299,72 sn, ikincinin 174,70 sn'dir. `rag_llm` canlı sağlayıcı doğrulaması bu teslimatta yapılmadı.

## Dokümantasyon bakım kuralı

Her anlamlı geliştirmede görev planındaki tamamlanan kutuları işaretleyin, uygulama günlüğüne tarih/dosyalar/doğrulama/sınırlamaları ekleyin ve devir belgesindeki değişen mimariyi güncelleyin. Yapılmamış testleri veya planlanan özellikleri tamamlanmış göstermeyin. Mevcut commit edilmemiş çalışmaları koruyun.

