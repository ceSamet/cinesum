# Sahne Seçimi: Literatür Notu ve Tasarım Kararları

## Kısa sonuç

Film özeti yalnızca en yüksek puanlı sahnelerin peş peşe eklenmesiyle
oluşturulmamalıdır. İyi bir seçim aynı anda şu dört hedefi dengelemelidir:

1. Kullanıcının metin sorgusuna ve seçtiği sahne/kişi türlerine uygunluk.
2. İçeriğin kendi olay kümelerini ve karakter/konu hatlarını kapsama.
3. Neden-sonuç ve soru-cevap gibi gerekli bağlam bağlarını koruma.
4. Aynı olayın veya görsel olarak benzer sahnelerin gereksiz tekrarını azaltma.

## Literatürden bulgular

### Anlatı yapısı

Papalampidi ve arkadaşları, uzun anlatılarda yalnızca konum veya genel
önem puanı kullanmanın özeti başlara yığabildiğini; anlatısal olay yapısını
modellemenin daha eksiksiz ve çeşitli özetler verdiğini gösterir. Bununla
birlikte TRIPOD'un yaklaşık dönüm noktaları klasik, hedef odaklı filmler için
yararlı bir açıklama modelidir; dizi, epizodik komedi, sanat filmi, belgesel ve
doğrusal olmayan anlatılar için evrensel seçim kuralı değildir.

- [Screenplay Summarization Using Latent Narrative Structure, ACL 2020](https://aclanthology.org/2020.acl-main.174/)
- [Movie Plot Analysis via Turning Point Identification, EMNLP-IJCNLP 2019](https://aclanthology.org/D19-1180/)

Bu nedenle projede sabit perde kotası ve sabit yüzdelerde dönüm noktası bonusu
kullanılmaz. Bitişik cut'lar; anlamsal içerik, görüntü ve karakter devamlılığıyla
videonun kendi olay kümelerine ayrılır. On eşit zaman penceresi yalnızca seçimin
tek noktaya aşırı yığılmasını azaltan yumuşak bir sinyaldir. Boş veya zayıf bir
pencereden kota doldurmak için sahne seçilmez.

### Kullanıcı sorgusuna göre özet

Video özetinin kullanıcıya göre değişmesi literatürde
`query-focused video summarization` olarak ele alınır. Metin sorgusu, genel
olarak önemli sahneler yerine kullanıcının aradığı kavramlarla ilgili sahnelerin
seçilmesini sağlar. Etkileşimli yaklaşımlar, yalnızca serbest metnin her zaman
yeterli olmadığını ve düzenlenebilir hazır niyetlerin de faydalı olduğunu vurgular.

- [Query-Focused Video Summarization, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Sharghi_Query-Focused_Video_Summarization_CVPR_2017_paper.html)
- [IntentVizor, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_IntentVizor_Towards_Generic_Query_Guided_Interactive_Video_Summarization_CVPR_2022_paper.html)

Bu nedenle arayüz hem serbest prompt hem de aksiyon, konuşmalı,
konuşmasız, müzikli, ana karakter ve yan karakter gibi açık tercihler sunar.
`Sadece` veya `yalnız` ifadesi sert filtre; diğer istekler puan tercihi olarak
yorumlanır.

### Çeşitlilik ve tekrar cezası

Video özetleme çalışmaları, önem ve temsil gücünün yanında görsel/anlamsal
çeşitliliği ayrı bir hedef olarak kullanır. Aynı bağlamdan birden fazla sahne
bazen anlaşılırlık için gereklidir; ancak marjinal fayda giderek azalmalıdır.

- [Video Summarization by Learning From Unpaired Data, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Rochan_Video_Summarization_by_Learning_From_Unpaired_Data_CVPR_2019_paper.html)
- [Improving Sequential DPPs for Supervised Video Summarization, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Aidean_Sharghi_Improving_Sequential_Determinantal_ECCV_2018_paper.html)
- [Movie Script Summarization as Graph-based Scene Extraction, NAACL 2015](https://aclanthology.org/anthology-files/anthology-files/pdf/N/N15/N15-1113.pdf)

Bu projede ceza yalnızca aynı olay kümesindeki cut'lar gerçekten anlamsal ve
görsel olarak birbirine benziyorsa uygulanır. Diyalog/bağlam yapılarında ilk iki,
aksiyon zincirlerinde ilk üç temsilci cezasız kalabilir. Farklı bilgi, motif veya
bakış açısı taşıyan sahne sırf aynı kümede olduğu için cezalandırılmaz.

### İçerik profili

Sistem videoya bir tür etiketi veya senaryo şablonu dayatmaz. Gözlenebilir
sinyallerden `diyalog ağırlıklı`, `hareket ağırlıklı`, `atmosfer/görsel motif
ağırlıklı`, `çok karakterli/paralel hatlı` veya `karma` bir çalışma profili
çıkarır. Bu profil puan ağırlıklarını değiştirir; kesin tür hükmü sayılmaz.

## Uygulanan seçim formülü

Kullanıcı isteği yoksa mevcut 100 puanlı yerel önem modeli kullanılır.
Bir istek varsa temel sinyaller toplam etkinin `%65`'ini, kullanıcı niyetine
uygunluk `%35`'ini oluşturur. Paralon incelemesinden sonra ikinci seçimde:

```text
LLM tarafından değerlendirilen aday = yerel puan × 0.42 + LLM puanı × 0.58
LLM'in atladığı aday              = yerel puan korunur
```

Son seçim kullanıcının süre bütçesini kesin sınır olarak uygular. Global
Knapsack sonucu üzerinde yalnızca gerçekten oluşan olay ve zaman yoğunlaşmaları
azalan getiriyle yeniden değerlendirilir; herhangi bir bölüm için zorunlu kota
bulunmaz.
