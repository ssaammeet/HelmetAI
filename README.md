# HelmetAI V7 — Bilgisayarda görüntü, video ve log doğrulama paketi

**Önce burayı okuyun:** Bu çalışma dalı, Pi bağlantısı onarılana kadar bilgisayarda
kanıt toplamayı kolaylaştırmak için V6 üzerine hazırlanmıştır. Yeni bir Pi saha
onayı, nihai ürün veya hızlandırılmış model değildir. Kurulu V6 Pi paketini
değiştirmeyin; bu paketteki `deploy/install_pi.sh` veya `deploy/update_pi.sh`
dosyalarını şimdi Pi'de çalıştırmayın. Aşağıdaki eski proje notları tarihsel
bağlam için korunmuştur; V7 kullanımı için **[V7 bilgisayar rehberini](docs/V7_DESKTOP_TR.md)** izleyin.

- `image-check`: tek fotoğrafta gerçek model kutuları, sınıflar, güven puanları ve sınıf çakışmaları.
- `video-run`: işlenmiş MP4, her kare için JSONL ve açıklanabilir özet raporu.
- `log-report`: Pi kayıtlarında gerçek tespit sayısı, tamamlanan/eksik koşu ayrımı ve güç kayıtları.
- Kamera kontrol fotoğraflarının kırmızı/mavi kanal değişimi düzeltildi; donanımda tekrar doğrulama bekliyor.
- Masaüstünde kalibrasyon/demo bayrakları gerçek JSON boolean olmalı; `"false"` kabul edilmez.
- Doğrulanmamış/degraded görüntülerde metre/TTC ve yeşil "güvenli" izlenimi verilmez.

Model ve risk eşikleri değiştirilmedi. Kaydedilmiş görüntüyle tespit görmek;
mesafe, TTC, kör nokta kapsaması veya trafikte güvenli kullanım doğrulaması değildir.

## Önceki proje açıklaması (V6 ve öncesi)

Bu sürüm, iki ayrı kamera ile çalışan bir motosiklet-kaskı prototipinin yazılım çekirdeğidir. Ön ve arka görüntüden COCO uyumlu YOLO ONNX yol-sahnesi algılama, hedef takibi, tek-kamera mesafe tahmini, bağıl yaklaşma hızı ve TTC tabanlı risk değerlendirmesi yapar.

> Güvenlik sınırı: Bu bir geliştirme prototipidir; sertifikalı bir ADAS ürünü değildir. Eşikler gerçek trafik kullanımı için doğrulanmamıştır. Yalnızca kapalı alanda, kontrollü testlerle kullanılmalıdır.

## V6 yazılım düzeltmesi — mevcut Pi kurulumu

Bu paket `HelmetAI_Pi_Software_Fix_v6.zip` adıyla dağıtılır. **Mevcut kurulumda**
[Türkçe güncelleme rehberini](docs/PI_SOFTWARE_UPDATE_V6_TR.md) kullanın;
işletim sistemini veya modeli yeniden kurmayın. Güncelleme kodu yedekler,
kamera eşlemesini/kalibrasyonu ve mevcut modeli korur, GPIO'yu kapatır,
servisi durdurulmuş bırakır. Bağımsız donma gözetimi bulunmadığından bu V6
teslimi kamera/model tanılama içindir; GPIO'yu veya trafik kullanımını açmaz.

- Arka kamera yakalama/çıkarım süresi artık son işlem içinde ikinci kez sayılmaz.
- `end_to_end_ms` tek döngüde kamera okuma çağrısından karar kontrol noktasına
  kadar geçen süredir; pozlama zamanından başlayan sensör gecikmesi değildir.
- `effective_fps`, kararlar arasındaki gerçek süreyi (önceki JSON çıktı beklemesi
  dahil) kullanır. `average_cycle_interval_ms` bunu işlem gecikmesinden ayırır.
- `directional_performance` iki yönün gerçekten tamamlanan çıkarım sayılarını,
  hızlarını ve son güncelleme yaşını ayrı gösterir. Sensörün ilan edilen FPS'i
  değildir. İlk çıkarımda hız bilinmediği için sıfırdır; ilk kısa pencerelerde
  ve güncellenmeyen yönde muhafazakâr hesaplanır.
- Her döngü `runtime_status` kaydı üretir. `detections` tüm kabul edilen
  algılamaların sınıfını/güvenini/kutusunu içerir; eski `scene_objects` yalnız
  mesafe modeli bulunmayan sınıflar içindir, boş olması "araç yok" demek değildir.
- `runtime_final.completed` ve `error` kesilmiş/başarısız bir denemeyi belirtir.
  Başarılı son ölçüm `performance_timestamp_s` anına aittir; kamerayı kapatma
  süresi sonradan bu ölçüme eklenmez. Hatalı koşuda yön sayaçları son kısmi
  işlemdeki başarıları da içerebilir; bu koşu benchmark geçemez.
- Ön kalibrasyon hatasının yanıltıcı `stereo_...` adı `range_calibration_or_frame_invalid`
  oldu. Bu değişiklik stereo kamera gerektirmez ve kalibrasyon kapısını gevşetmez.
- YOLO adayları vektörle süzülür; sınıflar, güven eşiği, kutu geometrisi ve
  sınıf bazlı NMS korunur. Sabit 640 ONNX modeli değiştirilmedi.

Güncellemeden sonra, normal Pi kullanıcısıyla (sudo olmadan) kısa kayıt:

```bash
bash /opt/helmetai-fcw/scripts/run_pi_diagnostic.sh
```

Bu komut 100 döngü çalıştırır; GPIO'yu bellekte zorunlu kapatır, ayar dosyasını
değiştirmez. Başlangıç/bitiş sıcaklık-güç bilgileri ve çalışma günlüğü ev
klasöründeki `helmetai-diagnostics/run.*` altına kaydedilir. Donanım koşulları
uygun değilse test kısa tutulmalı, gerekirse Ctrl+C ile durdurulmalıdır.
Bu komutun tamamlanması bir performans ya da güvenlik onayı değildir.

Kamera-check ile alınmış JPEG üzerinde 1/2/3/4 CPU iş parçacığını karşılaştırmak:

```bash
bash /opt/helmetai-fcw/scripts/profile_detector_pi.sh --image ~/helmetai-camera-check/front_camera_check.jpg --config /etc/helmetai-fcw/pi5_dual_camera.json --output ~/helmetai-thread-profile-v6.json
```

Profiler kamera/GPIO açmaz; yalnız sabit JPEG üzerinde algılayıcı süresini
ölçer. Var olan raporu ezmez ve en hızlı görünen ayarı otomatik uygulamaz.
Sıralama, CPU yükü ve sıcaklık sonucu etkileyebilir. Bu sayı tam sistem FPS'i
değildir. Hız artışı ve performans kapısının geçilmesi gerçek Pi'de henüz
doğrulanmadı; yazılım güncellemesi soğutma/güç ihtiyacını kaldırmaz.

## Çalışan fonksiyonlar

- **Ön risk:** Aynı sürüş koridorundaki öndeki araca yaklaşma için `ADVISORY`, `WARNING` veya `CRITICAL` üretir.
- **Arka risk:** Arka kamerada algılanan ve motosiklete yaklaşan araç için ayrı TTC ve uyarı seviyesi üretir.
- **Yol-sahnesi farkındalığı:** Araç, motosiklet, bisiklet ve yayaları algılar ve takip eder; trafik ışığı, dur levhası ve belirli statik nesneleri etiketler. TTC yalnızca sınıfa özgü mesafe belirsizliği kabul edilebilir seviyedeyse üretilir. Özellikle yaya/bisiklet kutusundan güvenilir metrik mesafe çıkarılamıyorsa hedef görünür kalır, fakat sahte bir TTC alarmı üretilmez.
- **Kısmi çevre risk haritası:** Ön/arka sol-merkez-sağ sektörlerdeki hedeflerin en yüksek riskini JSON çıktısında bildirir; iki yan tarafın fiziksel olarak gözlenmediğini açıkça belirtir.
- **Yazılımsal kör nokta yardımcısı:** Arka kamerada görünen yakın yan hedefleri sol/sağ olarak sınıflandırır. Bir `--turn-intent left|right` yazılım girdisi verilirse eşleşen tarafta uyarı kararı üretir. Sinyal, IMU veya kafa hareketi entegrasyonu bu girdiyi besleyecek ayrı donanım katmanıdır.
- **Edge performans telemetrisi:** Kamera alma, çıkarım ve son işlem gecikmesini; etkin FPS'i ve yapılandırılabilir gecikme bütçesini ölçer. Pi koşusunda `performance_final` çıktısı ile raporlanır.
- **Sürücü uyarı çıkışı:** İsteğe bağlı Raspberry Pi GPIO katmanı, `ADVISORY` için LED, `WARNING` için darbeli LED/buzzer/titreşim ve `CRITICAL` için sürekli LED/buzzer/titreşim uygular. Varsayılan olarak kapalıdır; ancak açık kablo ve BCM pin yapılandırmasıyla etkinleşir.
- **Fail-closed davranış:** Ön veya arka kamera kalibrasyonu onaylanmamışsa o yön için risk alarmı açılmaz.
- **Ayrı ön/arka kalibrasyon:** Her kameranın odak uzunluğu ve ölçüm belirsizliği bağımsız yapılandırılır.
- **Demo/test:** Deterministik ön ve arka yaklaşma simülasyonları ile otomatik regresyon testleri bulunur.

## Henüz kapsam dışında olanlar

- Sol ve sağ kör noktanın **tam** kapsaması ve gerçek 360 derece görüş (iki normal ön/arka kamera bunu fiziksel olarak göremez; yan kamera gerekir).
- Yandan çarpışma, şerit değiştirme veya kavşak çarpışması için gerçek trafikte doğrulanmış risk motoru.
- GPS/IMU ile motosikletin gerçek hızı ve yönelimi.
- Ses/titreşim/LED donanımının fiziksel kablolaması ve nihai kask üzerindeki elektriksel testi.
- Raspberry Pi üzerinde sıcaklık ve gerçek trafik doğrulaması.

## Bilgisayarda test

PowerShell'i proje kökünde açın:

```powershell
cd C:\HelmetAI_FCW_MVP
Set-ExecutionPolicy -Scope Process Bypass
$env:PYTHONPATH = "src"

.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m helmetai_fcw demo
.\.venv\Scripts\python.exe -m helmetai_fcw rear-demo
```

`demo` önden yaklaşmayı, `rear-demo` arkadan yaklaşmayı simüle eder. Her ikisi de tehlike arttıkça `ADVISORY -> WARNING -> CRITICAL` sıralamasını göstermelidir.

## Şirket sahibine görsel demo

Tek pencerede ön ve arka yaklaşma senaryolarını, hareket eden araçları, tahmini mesafeyi, TTC'yi ve risk seviyesini göstermek için:

```powershell
.\scripts\run_demo_dashboard.ps1
```

Alternatif olarak doğrudan risk-motoru adıyla aynı ekranı açabilirsiniz:

```powershell
.\scripts\run_risk_engine_demo.ps1
```

Penceredeki `Start` düğmesi, mevcut ön ve arka risk motorlarının kontrollü simülasyon çıktısını oynatır. Ekran mesafe, yaklaşma hızı, TTC, gerekli yavaşlama, koridor kontrolü ve karar gerekçesini gösterir. Bu ekran, canlı trafik veya fiziksel Raspberry Pi testi olduğunu iddia etmez; canlı kamera algılama demoyu ayrı olarak webcam komutuyla gösterin.

## Gerçek video ile gelişmiş demo

Kapalı alanda çekilmiş bir MP4 kaydını araç algılama ve risk hattından geçirmek için önce webcam ortamını bir kez kurun. Ardından ön veya arka yöndeki kaydı oynatın:

```powershell
.\scripts\run_video_demo.ps1 -Video C:\DemoVideolari\front_controlled.mp4 -Direction front -Output C:\DemoVideolari\front_annotated.mp4
.\scripts\run_video_demo.ps1 -Video C:\DemoVideolari\rear_controlled.mp4 -Direction rear -Output C:\DemoVideolari\rear_annotated.mp4
```

Canlı pencerede araç kutusu, araç sınıfı, yaklaşık mesafe, yön ve risk seviyesi görünür. `Output` parametresi, toplantı sırasında tekrar oynatılabilecek işlenmiş MP4 üretir. Risk alarmını göstermek için her kamera kendi kontrollü saha ölçümüyle kalibre edilmeli ve ilgili yapılandırmada `calibrated` değeri ancak doğrulama sonrasında `true` yapılmalıdır.

### Görsel uyarı demosu (yalnızca sunum)

İnternetten alınan ya da kamerası kalibre edilmemiş bir videoda, kutuların risk seviyesine göre renk değiştirmesini yalnızca sunum amacıyla göstermek için aşağıdaki anahtarı kullanın:

```powershell
.\scripts\run_video_demo.ps1 -Video C:\DemoVideolari\front_risk.mp4 -Direction front -DemoVisualAlerts -Output C:\DemoVideolari\front_risk_annotated.mp4
```

Bu mod üst bantta `DEMO MODE - VISUAL ALERTS ONLY / NOT ROAD-VALIDATED` yazar. Kullanılan odak ve mesafe parametreleri bu kaynak video için doğrulanmamıştır; çıkan metre, TTC ve renkler gerçek trafik kullanımına veya ürün performansına dair bir iddia değildir. Raspberry Pi/gerçek kullanım yapılandırması bundan etkilenmez.

## Bilgisayar kamerası ile kontrollü test

Önce bir kez ortamı kurun:

```powershell
.\scripts\setup_webcam.ps1
```

Ön kamera modunu açın:

```powershell
.\scripts\run_webcam.ps1
```

Arka kamera modunu, web kamerasını arka yönde konumlandırarak açın:

```powershell
.\scripts\run_webcam.ps1 config\webcam_rear.example.json
```

İlk çalıştırmada `calibrated` değeri `false` olduğu için araç algılama gösterilir fakat risk alarmı kapalı kalır. Bu beklenen güvenlik davranışıdır. Ön ve arka kamera için ayrı ölçüm ve doğrulama yapılmadan `true` yapılmamalıdır.

## Kalibrasyon

Kamera nihai montaj konumundayken, genişliği bilinen bir aracı bilinen bir mesafeye yerleştirin. Algılanan kutunun piksel genişliğini ölçün ve her kamera için ayrı komutu kullanın:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m helmetai_fcw calibrate-monocular --camera-role front --reference-distance 10 --object-width 1.80 --pixel-width 104
.\.venv\Scripts\python.exe -m helmetai_fcw calibrate-monocular --camera-role rear --reference-distance 10 --object-width 1.80 --pixel-width 104
```

Çıkan değeri sırasıyla `front_calibration.focal_length_px` ve `rear_calibration.focal_length_px` alanına yazın. En az üç farklı mesafede ölçüm hatasını kaydedip kabul edilebilir olduğundan emin olduktan sonra ilgili `calibrated` alanını `true` yapın.

## Raspberry Pi 5 kurulumu

Gerekenler: Raspberry Pi 5, aktif soğutma, iki **CSI/libcamera uyumlu** kamera, iki uygun kamera kablosu, kaliteli Pi uyumlu güç kaynağı ve 64-bit Raspberry Pi OS. İlk paket Bookworm hedefliydi; kullanıcı ilk çalıştırmayı Trixie üzerinde yaptı. V6'nın Pi üzerinde tam doğrulaması henüz yapılmadı. Sadece Trixie kullanıldığı için format atmayın. Bu Pi çalışma yolu USB webcam kullanmaz.

1. Güncel proje paketini Pi'ye kopyalayıp açın; sonra kurulum betiğini çalıştırın:

   ```bash
   sudo apt update
   sudo apt install -y unzip
   mkdir -p ~/helmetai-fcw-release
   unzip ~/HelmetAI_Pi_Software_Fix_v6.zip -d ~/helmetai-fcw-release
   cd ~/helmetai-fcw-release
   chmod +x deploy/install_pi.sh
   ./deploy/install_pi.sh
   ```

   Betik, servisi çalıştıran kullanıcıyı otomatik olarak `HELMETAI_USER`, ardından `SUDO_USER`, ardından mevcut kullanıcıdan seçer; `pi` adlı kullanıcıya bağlı değildir. Betiği doğrudan root olarak çalıştırmanız gerekirse örneğin `HELMETAI_USER=helmet ./deploy/install_pi.sh` kullanın.
   Betik `rpicam-apps`, `gpiod`, Picamera2, OpenCV ve gerekli Python paketlerini yükler. Kullanıcıyı kamera için `video`, mevcutsa GPIO için `gpio` grubuna ekler; ilk interaktif kamera/GPIO testi öncesinde yeniden oturum açın veya Pi'yi bir kez yeniden başlatın.

2. Kameraların görülüp görülmediğini, model yolunu ve yapılandırmayı kontrol edin:

   ```bash
   /opt/helmetai-fcw/scripts/preflight_pi.sh
   ```

   `rpicam-hello --list-cameras` çıktısında iki kamera görünmelidir. Görünmüyorsa servis başlatmayın.

3. Fiziksel ön/arka yön eşlemesini fotoğrafla doğrulayın:

   ```bash
   cd /opt/helmetai-fcw
   PYTHONPATH=src python3 -m helmetai_fcw pi-camera-check \
     --config /etc/helmetai-fcw/pi5_dual_camera.json \
     --output-dir ~/helmetai-camera-check
   ```

   `front_camera_check.jpg` ve `rear_camera_check.jpg` dosyalarını açın. Lensleri sırayla kapatarak `camera_index` eşlemesinin gerçekten ön/arka olduğunu doğrulayın. Arka kamera fiziksel olarak ters yönde baktığı için örnek profil `rear_calibration.lateral_sign: -1.0` kullanır; solda duran bir hedefin sürücünün solunda raporlandığını bu aşamada doğrulayın.

4. Yapılandırmayı düzenleyin:

   ```bash
   sudo nano /etc/helmetai-fcw/pi5_dual_camera.json
   ```

  `front_camera` ve `rear_camera` indekslerini doğrulanmış kamera-check görüntülerine göre güncelleyin. Aynı indeks iki rol için kullanılamaz. Her iki kamera için kalibrasyon değerlerini girin. Model yolu `/opt/helmetai-fcw/models/detector.onnx` olmalıdır.
   Bu paketle gelen sabit girişli ONNX model için `detector.input_size_px` değeri **640** kalmalıdır. `416` bu modelle uyumlu değildir.
   İlk bağlantıda `calibrated` alanlarını `false` bırakın: algılama test edilir fakat risk alarmı açılmaz.

5. Kapalı alanda sınırlı kare testi yapın:

   ```bash
   cd /opt/helmetai-fcw
   export PYTHONPATH=src
   python3 -m helmetai_fcw pi-run --config /etc/helmetai-fcw/pi5_dual_camera.json --frames 100
   ```

   Çıktıda her değerlendirme için `direction: front` veya `direction: rear`, risk seviyesi, TTC, sahne nesneleri, kör nokta kararı, kapsama bilgisi ve gecikme telemetrisi görülür. Koşu bittiğinde `performance_final` satırı; ortalama/p95 gecikme, etkin FPS ve gecikme bütçesi sonucunu verir.

   Kör nokta karar motorunu donanım sinyali olmadan kapalı alanda denemek için:

   ```bash
   python3 -m helmetai_fcw pi-run --config /etc/helmetai-fcw/pi5_dual_camera.json --frames 100 --turn-intent left
   ```

6. Her iki kamera kalibre edildikten sonra 300 karelik benchmark çalıştırın:

   ```bash
   /opt/helmetai-fcw/scripts/benchmark_pi.sh /etc/helmetai-fcw/pi5_dual_camera.json 300
   ```

   Komut yalnız 60 karelik ısınma tamamlandığında, p95 gecikme bütçesi geçildiğinde, genel FPS ve her etkin yönün gerçek karar hızı `minimum_risk_fps` değerini karşıladığında `BENCHMARK PASSED` ile biter. Başarısızsa servis başlatmayın; soğutma, güç kaynağı, kamera ayarı veya model performansı incelenmelidir.

7. GPIO uyarı donanımı bağlanacaksa aşağıdaki **GPIO uyarı çıkışı** bölümünü tamamlayın. Bağlanmayacaksa örnek yapılandırmadaki `alert_actuator.enabled: false` değerini değiştirmeyin.

8. V6 tanılama aşamasında otomatik servis kapalı kalır. İleride preflight, kamera yön eşlemesi, kalibrasyon, kontrollü test ve benchmark doğrulandıktan sonra yalnız kamera için servis başlatma komutları:

   ```bash
   sudo systemctl enable --now helmetai-fcw
   journalctl -u helmetai-fcw -f
   ```

   Fiziksel Pi, kamera, mesafe ve GPIO doğrulamasını kaydetmek için [Pi ve kontrollü saha kontrol listesini](docs/PI_FIELD_VALIDATION_CHECKLIST.md) sırayla uygulayın.

## GPIO uyarı çıkışı

**V6 güncelleme kapsamı dışındadır.** Aşağıdaki bilgiler mevcut sürücü kodunun
referansıdır; bu paketle çıkışları açmayın. Eşzamanlı kamera/model çağrısı
takıldığında çıkışı bağımsız olarak kapatacak watchdog henüz yoktur. Sağlık
kontrolü ancak çalışma döngüsü ilerlediğinde uygulanır; donmaya karşı garanti değildir.

Sistem, GPIO yapılandırması açık değilken fiziksel pinlere dokunmaz. Gerçek bağlantı yapılmadan önce `alert_actuator.enabled` değerini `true` yapmayın.

Motosiklet kaskındaki ses/titreşim/LED donanımı için Pi GPIO pinleri **3.3 V lojik** seviyesindedir. Vibration motoru veya yüksek akım çeken buzzer doğrudan GPIO pinine bağlanmaz; uygun MOSFET/transistör sürücüsü, ortak GND ve gerekiyorsa diyot içeren sürücü devresi gerekir.

Kablolama ve sürücü devresi doğrulandıktan sonra `/etc/helmetai-fcw/pi5_dual_camera.json` içindeki örneği kendi **BCM GPIO numaralarınızla** doldurun. BCM numarası fiziksel header sıra numarası değildir:

```json
"alert_actuator": {
  "enabled": true,
  "gpio_chip": null,
  "buzzer_bcm_pin": 18,
  "vibration_bcm_pin": 23,
  "led_bcm_pin": 24,
  "active_high": true,
  "warning_pulse_period_s": 0.5
}
```

Bu yalnız örnektir; `18`, `23` ve `24` değerleri varsayılan kablolama değildir. Her çıkış ayrı BCM pininde olmalıdır. `active_high`, sürücü modülünüzün hangi lojik seviyede etkin olduğuna göre ayarlanır. `gpio_chip: null`, Pi'nin fiziksel 40-pin header'ına ait `pinctrl*` denetleyicisini otomatik bulur. Bazı eski Pi 5 Bookworm sürümlerinde bu `gpiochip4`, güncel çekirdeklerde `gpiochip0` olabilir. Auto-detection belirsiz kalırsa `gpiodetect` çalıştırın ve yalnız doğruladığınız chip numarasını yazın.

Risk çıkış desenleri şöyledir:

- `NONE`: tüm çıkışlar kapalı.
- `ADVISORY`: LED açık; buzzer ve titreşim kapalı.
- `WARNING`: LED, buzzer ve titreşim darbeli.
- `CRITICAL`: LED, buzzer ve titreşim sürekli açık.

Sağlık kapısı en az `performance_warmup_frames` döngü, bütçe içinde p95 gecikme ve yeterli genel FPS ister. Buna ek olarak her yönün değerlendirmesi taze olmalı, gerçekten ölçülen işleme hızı `minimum_risk_fps` değerini karşılamalıdır. V6 Pi yolunda hız artık yalnız toplam FPS/katsayı tahmininden alınmaz. `alert_inputs.estimated_risk_fps` alan adı eski istemciler için korunur, ancak içindeki değer ölçülen yön hızıdır. Arka kamera daha seyrek işleniyorsa, ön kameranın hızlı olması arka yönü otomatik olarak yeterli yapmaz.

## Risk mantığı

Her yön için kamera hedefin yaklaşık mesafesini çıkarır. Takip filtresi zaman içindeki mesafe değişiminden bağıl yaklaşma hızını hesaplar. Mesafe küçülüyor, hedef ilgili koridorda bulunuyor ve takip yeterince kararlıysa sistem muhafazakâr TTC hesaplar:

```text
etkin_mesafe = mesafe - geometri_payı - belirsizlik_payı
TTC = etkin_mesafe / yaklaşma_hızı
gerekli_yavaşlama = yaklaşma_hızı^2 / (2 * etkin_mesafe)
```

TTC küçüldükçe veya gerekli yavaşlama arttıkça risk seviyesi yükselir. Sabit uzaklıktaki veya koridor dışındaki araçlar sırf belirsizlik yüzünden alarm üretmez. Bir TTC kararından önce sistem en az üç ardışık algılamada ve en az 0.35 saniye boyunca aynı yaklaşma eğilimini görür; böylece yalnız kare sayısına bağlı davranmaz. Göreli mesafe belirsizliği yüksekse o hedef için sistem `DEGRADED` duruma geçer ve alarm üretmez. Pi performansı 5 FPS'in altına veya gecikme bütçesinin üstüne düşerse çalışma döngüsündeki sağlık kapısı fiziksel uyarıya izin vermez; süreç donmasına karşı bağımsız watchdog yerine geçmez.

## Edge performans ve model optimizasyonu

`config/pi5_dual_camera.example.json`, paketle gelen sabit girişli YOLO ONNX model için 640 piksel giriş boyutu, üç CPU iş parçacığı, arka kamera için üçte bir çıkarım sıklığı, tam 60 örnekten oluşan ısınma penceresi, 0.45 saniyelik risk tazelik penceresi ve 2 saniyelik takip boşluğu toleransı içerir. Bu değerler ölçülmüş Pi performansı değildir; her Pi/kamera kombinasyonunda ölçülmelidir. GPIO uyarıları, yapılandırılmış ısınma süresi sonrasında p95 gecikme bütçesi altında ve en az `minimum_risk_fps` değerinde etkinleşir; arka yön için bu eşik, `rear_capture_every_n_frames` nedeniyle ayrıca yön bazında sağlanmalıdır. `performance_final` içindeki sağlık kapısı kapalıysa veya `alert_inputs` arka yönü düşük örnekleme hızı nedeniyle engelliyorsa servis gerçek kullanım için açılmamalıdır.

Pi üzerinde kalibre edilmiş kameralarla 300 karelik ölçüm için:

```bash
cd /opt/helmetai-fcw
chmod +x scripts/benchmark_pi.sh
./scripts/benchmark_pi.sh /etc/helmetai-fcw/pi5_dual_camera.json 300
```

Geliştirme bilgisayarında isteğe bağlı INT8 ONNX kopyası üretmek için:

```bash
python -m pip install -r requirements-optimization.txt
python scripts/quantize_onnx_int8.py --input models/detector.onnx --output models/detector.int8.onnx --validate-opencv --input-size 640
```

INT8 model ancak komut `opencv_validation=PASSED` çıktısını verdikten ve aynı Pi üzerinde algılama doğruluğu ile gecikmesi ölçüldükten sonra yapılandırmaya alınmalıdır. Dinamik INT8 dönüştürme, bazı OpenCV sürümlerinde desteklenmeyen ONNX düğümleri oluşturabilir; doğrulama başarısızsa varsayılan FP32 model kullanılmalıdır. Bu araç model eğitimi, pruning veya gerçek trafik doğrulaması yerine geçmez.
