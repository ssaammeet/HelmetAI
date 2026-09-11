# HelmetAI V7 — kablo beklerken bilgisayarda çalışma

Bu paket **Windows bilgisayarda** fotoğraf, kayıtlı video ve eski Pi loglarını
kontrol etmek içindir. Pi'nin açık olması, kamera bağlantısı, GPU, GPS/IMU veya
GPIO gerekmez. Gerçek Pi'de doğrulama yerine geçmez. Pi'ye kurulu V6'yı şimdilik
koruyun. Buradaki `deploy` dosyaları eski dağıtım testleri için arşivde vardır;
bu V7 masaüstü çalışması için **çalıştırılmayacaklar**.

## 1. Ayrı klasöre çıkar ve bir kez kur

`HelmetAI_Offline_Validation_v7.zip` içeriğini `C:\HelmetAI_V7` klasörüne çıkarın.
Doğru kökte `src`, `scripts`, `models`, `tests` olmalı; ZIP'in içine `cd` yapılmaz.
ONNX model arşive dahildir; eski modelinizi kopyalamanıza gerek yoktur.
`.venv` bilgisayara özeldir; diğer bilgisayardan kopyalanmaz.

PowerShell'i normal kullanıcı olarak açın. **Komutlar Pi terminaline değil Windows'a yazılır.**
Yalnız kendinizin güvendiği bu proje betikleri için süreç kapsamındaki izni kullanın:

```powershell
cd C:\HelmetAI_V7
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\setup_webcam.ps1
```

Python 3.10 veya üstü gerekir. Kurulum mevcut proje sanal ortamını yeniden
oluşturmaz; bağımlılık kurulum/import hatasında "başarılı" yazmaz. İlk bağımlılık
kurulumu internet gerektirebilir; fotoğraf/video/log analizi verileri yüklemez.
Önceden kurulu ortamla test çalıştırmak için internet gerekmez.

## 2. Fotoğrafta model hangi kutuyu çiziyor?

Aşağıdaki örnek dosya yollarını kendi gerçek dosyanızla değiştirin:

```powershell
.\scripts\run_image_check.ps1 -Image "C:\DemoVideolari\arac.jpg" -Direction front
```

Yeni `C:\HelmetAI_V7\results\image-...` klasöründe:

- `annotated.jpg`: kutular, numaralar ve okunaklı sınıf/güven puanı listesi.
- `detections.json`: gerçek model çıktıları, görüntü/model SHA-256 ve sınırlar.
- `report.md`: okunabilir açıklama ve çakışan sınıflar.

`Direction` yalnız sizin verdiğiniz etikettir, fiziksel kameranın yönünü kanıtlamaz.
EXIF otomatik yönlendirmesi yapılmaz. Yan duran fotoğrafı tanılama amacıyla açıkça
döndürmek için `-Rotate 90` kullanabilirsiniz; bu dönüş rapora işlenir. Kamera
montajını/kalibrasyonunu değiştirmez. Sonuç klasörleri otomatik benzersizdir;
`-OutputDir` verirseniz hedef yeni olmalı, üst klasör mevcut olmalıdır.

İki farklı sınıfın kutuları çok büyük oranda örtüşürse inceleme uyarısı verilir;
etiketlerden biri sessizce silinmez. Güven puanı doğruluk oranı veya kaza
olasılığı değildir. Aracın görünmemesi halinde eşik düşürmeden önce giriş
fotoğrafını, kadrajı, netliği ve çıkan kutuları kontrol edin.

## 3. Videolu demo ve sonunda döküm

```powershell
.\scripts\run_video_demo.ps1 -Video "C:\DemoVideolari\frontsafe.mp4" -Direction front -NoPreview
```

Varsayılan olarak **en fazla 300 kare** işlenir. Program `results` altında
işlenmiş MP4 ve ayrı `video-...` rapor klasörü oluşturur. `-NoPreview` önizleme
penceresini kapatır; çıktı MP4'ü tamamlanınca normal video oynatıcıda açın.
Önizleme istiyorsanız bu seçeneği kaldırın. ONNX bu sürümde CPU ile çalışır;
GPU'lu bilgisayarda otomatik CUDA kullanılmaz.

Belirli bir bölümü işlemek için:

```powershell
.\scripts\run_video_demo.ps1 -Video "C:\DemoVideolari\front_risk.mp4" -Direction front -StartSeconds 5 -MaxFrames 120 -NoPreview
```

Dosyanın tamamını özellikle istiyorsanız `-MaxFrames 0` kullanın. Uzun dosyalar
zaman/disk alanı tüketir; Ctrl+C ile durdurma çıktıyı kısmi bırakabilir. Arka
kamera bakışından gerçek bir video için `-Direction rear` kullanılır. Önden
çekilmiş videoyu sadece `rear` diye etiketlemek arka senaryoyu doğrulamaz.

Rapor dosyaları:

- `frames.jsonl`: her işlenen karenin kaynak zamanı, tespitleri, takip kimlikleri,
  risk motoru seviyesi/durumu ve gerekçeleri.
- `summary.json`: sınıf, takip, durum/seviye sayıları ve koşunun tamamlanma türü.
- `summary.md`: okunabilir özet. `max_frames` veya `user_stop` tam video EOF değildir.

Kalibrasyonu bilinmeyen videolarda metre/TTC sayıları gösterilmez. `NONE` güvenli
sahne demek değildir. Kare sayıları fiziksel araç/insan sayısı değildir; takip
kimlikleri bölünebilir veya değişebilir. Kaynak zamanları kare/FPS üzerinden
hesaplanır (sabit FPS varsayımı); değişken hızlı/time-lapse klipler hareket
doğrulaması için uygun değildir. Kaynak FPS belirsizse otomatik 20 FPS uydurulmaz.

Eski ön-yön görsel gösterim profili `-DemoVisualAlerts` ile isteğe bağlıdır;
raporda `illustrative_demo` olarak işaretlenir ve gerçek metre/TTC kanıtı vermez.
Bu seçenek varsayılan değildir. Risk motorunun kontrollü sentetik ön/arka
senaryoları ayrıca `scripts\run_demo_dashboard.ps1` ile incelenebilir; simülasyon
sonucu gerçek kamera veya yol testi değildir.

## 4. Eski Pi loglarından otomatik rapor

```powershell
.\scripts\run_log_report.ps1 -Runtime "C:\DemoVideolari\runtime.log" -Before "C:\DemoVideolari\before.txt" -After "C:\DemoVideolari\after.txt"
```

`results\logs-...\summary.md` ve `report.json` oluşur. Sadece terminalden
kopyaladığınız metin varsa `-Runtime` ile o `.txt` dosyasını verin; `Before/After`
isteğe bağlıdır. Eksik veriler **bilinmiyor** kalır, sıfır sayılmaz.

Aynı metindeki farklı koşular ayrılır. Kesik son satırın varlığı programın
çöktüğü anlamına gelmez. Bir tespit hem heartbeat hem hedef satırında yazılmış
olsa bile iki kez sayılmaz. Eski arka sonuçlarının önbellek sayaçları yeni
tespit olarak sayılmaz. `0x50000` güncel düşük gerilim değil, geçmiş düşük
gerilim ve kısıtlama bitleri içerir; önce/sonra farkı ayrı raporlanır.
Birden fazla koşuyla tek sağlık dosyası çifti verilirse otomatik eşleştirilmez.

## 5. Otomatik testler

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dağıtılan ZIP'in içindeki `docs/V7_TEST_OUTPUT.txt` o sürümün geliştirme
bilgisayarı sonuçlarını içerir. Sahte kamera/uyarı sürücüsüyle geçen testler
fiziksel Pi veya GPIO testi sayılmaz. Windows'a özel PowerShell testleri başka
işletim sisteminde atlanabilir; geliştirme bilgisayarındaki paket kabulünde
atlanan test olmamalıdır.

## Bilinen sınırlar ve kablo gelince

- Aynı COCO uyumlu YOLO11n ONNX modeli korunmuştur; fine-tuning veya yeni model yok.
- Algılayıcı hâlâ 640 kare girişe esneterek boyutlandırır (letterbox yok).
  Kenara taşan kutuların ortak algılayıcı geometrisi değiştirilmedi; fotoğraf
  raporu bu kutuları işaretler. Bu sınırlamalar ayrı doğruluk ölçümü gerektirir.
- Pi risk çekirdeği, eşikler ve performans izinleri bu masaüstü iterasyonunda
  gevşetilmedi. Arka/ön lenslerin fiziksel eşleşmesi hâlâ doğrulanmalıdır.
- Kamera kontrol JPEG'lerinde BGR kanal hatası kaynak kodda düzeltildi. Bu,
  canlı modele verilen girişin değiştiği veya eski yanlış sınıfların nedeninin
  kesin bulunduğu anlamına gelmez. Pi'de yeniden fotoğraf testi bekliyor.
- Kutulama örnekleri genel doğruluk yüzdesi, mesafe/TTC doğruluğu, Pi FPS artışı,
  tüm yönleri kapsama veya nihai yol kullanım onayı değildir.
- Gerçek videolar ve kişisel Pi logları GitHub/release ZIP'ine eklenmez;
  paylaşacağınız raporlarda yerel dosya yolları ve kamera görüntüleri olabilir.

Kablo geldikten sonra: sağlam bağlantı → hangi lens hangi kanal → net kamera
görüntüsü → karşılaştırmalı araç tespiti → gerçek kalibrasyon → güç/sıcaklık ve
gecikme ölçümü → kapalı alanda etiketlenmiş testler. V7'yi Pi'ye taşıma ayrı,
kontrollü bir güncelleme adımı olmalıdır.
