# V6 — mevcut Raspberry Pi kurulumuna yazılım düzeltmesi

Bu yol, `/opt/helmetai-fcw` ve `/etc/helmetai-fcw/pi5_dual_camera.json`
kurulmuş Raspberry Pi 5, 64 bit Debian 13 Trixie sistemi içindir. Mevcut
Picamera2, OpenCV ve ONNX modeli kullanılır. İşletim sistemi yeniden kurulmaz;
`apt upgrade`, `pip install` veya eski `deploy/install_pi.sh` çalıştırılmaz.
Soğutma, kablolama ve diğer donanım değişiklikleri bu güncellemenin kapsamında
değildir. Pi üzerinde bu sürüme ait ölçüm yapılmış veya bir FPS değeri garanti
edilmiş değildir.

## Neler değişir, neler korunur?

- Çalışma zamanı gecikme hesabı ve yön bazlı ölçüm düzeltmeleri yüklenir.
  Ön/arka karar hızı, toplam FPS'i atlama katsayısına bölerek değil, gerçekten
  tamamlanmış yön güncellemelerinin zamanlarından ölçülür.
- `src`, `scripts`, `config`, `docs`, `deploy`, proje açıklaması ve bağımlılık
  liste dosyaları güncellenir. Bağımlılık listesi kopyalanır, paket kurulmaz.
- `/opt/helmetai-fcw/models` ve yapılandırılmış model dosyası değiştirilmez.
  Kalibrasyonlar, kamera indeksleri/çözünürlükleri, model yolu, eşikler ve
  mevcut diğer profil değerleri korunur. `config/` altındaki yeni örnek
  profiller etkin `/etc/` profilinin üstüne yazılmaz.
- Tek bilinçli etkin profil değişikliği `alert_actuator.enabled: false`
  yapılmasıdır. `calibrated: false` değerleri açılmaz. Kalibrasyon onaylıysa
  mevcut onay da değiştirilmez; bu sürüm kendiliğinden kalibrasyon yapmaz.
- Çalışan servis durdurulur ve otomatik açılışı kapatılır. Maskelenmiş servis
  maskeli bırakılır. Hiçbir koşulda otomatik başlatma yapılmaz.
- Eski dosyalar, eski JSON, etkin systemd tanımı ve ilk servis durumu benzersiz
  `/opt/helmetai-fcw-backups/v6-...` dizinine alınır. Güncelleyici servis
  tanımını değiştirmez; root/boş kullanıcı veya beklenmeyen çalışma yolu olan
  bir servisi düzeltmeye çalışmak yerine durur.

## 1. Düz ZIP'i yeni bir dizine çıkarın

Arşiv adı farklıysa ilk satırdaki dosya adını değiştirin. Düz ZIP'in kökünde
`src/`, `deploy/`, `scripts/` bulunmalıdır. Eski çıkarma dizininin üzerine
yazmayın ve güncelleyiciyi `/opt/helmetai-fcw` içinden çalıştırmayın.

```bash
RELEASE_ZIP="$HOME/HelmetAI_Pi_Software_Fix_v6.zip"
RELEASE_DIR="$(mktemp -d "$HOME/helmetai-v6-release-XXXXXX")"
unzip "$RELEASE_ZIP" -d "$RELEASE_DIR"
cd "$RELEASE_DIR"
test -f deploy/update_pi.sh
bash -n deploy/update_pi.sh
sudo bash deploy/update_pi.sh
```

`UPDATE FAILED` olursa devam etmeyin. Servis durdurulduktan sonra bir hata
oluştuysa servis kapalı kalır ve kurtarma dizini yazdırılır; eski kod otomatik
geri yüklenmez. Config/model veya systemd doğrulaması başarısızsa hata
mesajını inceleyin; modeli veya kamera kalibrasyonunu değiştirmeyin.

Başarı sonunda yazılan **Backup** yolunu kaydedin. Dizin root'a özeldir;
`sudo ls <tam-yol>` ile bakılır. Aşağıdaki sorgularda servis `inactive` ve
`disabled` (önceden maskeliyse `masked`) görünmelidir. `is-active` ve
`is-enabled` bu beklenen durumlarda sıfırdan farklı çıkış kodu verebilir.

```bash
systemctl is-active helmetai-fcw
systemctl is-enabled helmetai-fcw
```

## 2. Önce yalnız kamera ve yazılım kontrolü

Aşağıdakileri normal kamera erişimi olan kullanıcıyla çalıştırın, `sudo`
kullanmayın. Kalibrasyonu açmayın, GPIO'yu açmayın ve servisi başlatmayın.

```bash
/opt/helmetai-fcw/scripts/preflight_pi.sh /etc/helmetai-fcw/pi5_dual_camera.json
cd /opt/helmetai-fcw
PYTHONPATH=src python3 -m helmetai_fcw pi-camera-check \
  --config /etc/helmetai-fcw/pi5_dual_camera.json \
  --output-dir "$HOME/helmetai-v6-camera-check"
```

Ön/arka görüntüleri açıp lensleri sırayla kapatarak yön eşleşmesini doğrulayın.
Preflight hata verirse veya görüntüler beklenen kameralardan gelmiyorsa
ölçüme geçmeyin. Kalibre edilmemiş profilde algılama yapılması ve risk
uyarılarının kapalı kalması beklenen davranıştır.

Kısa masaüstü koşusunu kaydedin:

```bash
set -o pipefail
cd /opt/helmetai-fcw
PYTHONPATH=src python3 -m helmetai_fcw pi-run \
  --config /etc/helmetai-fcw/pi5_dual_camera.json --frames 20 --camera-only \
  | tee "$HOME/helmetai-v6-smoke.jsonl"
```

Bu 20 karelik koşu ısınma/performans kabul testi değildir. Son kayıtta
`completed: true`, `error: null` ve `processed_frames: 20` beklenir. Hata,
kameranın takılması veya rahatsız edici cihaz sıcaklığı varsa uzun test
yapmayın; `Ctrl+C` ile durdurun. V6 soğutma yeterliliğini doğrulamaz.

Sıcaklık/güç ve sürüm bilgileriyle birlikte günlüğü tek klasöre almak için:

```bash
bash /opt/helmetai-fcw/scripts/run_pi_diagnostic.sh /etc/helmetai-fcw/pi5_dual_camera.json 20
```

Bu komut GPIO'yu bellekte kapalı tutar; etkin dosyayı değiştirmez. Ekrana
yazılan `~/helmetai-diagnostics/run.*` dizininde `before.txt`, `runtime.log`,
`after.txt` bulunur. Paylaşmak için bu üç dosyayı gönderin. 20 döngü yalnız
ilk kontroldür; uygun koşullarda 100 döngülük kayıt daha sonra alınabilir.

## 3. Modeli ve ön/arka işleme maliyetini ayırın

Kamera kontrolünde kaydedilmiş gerçek bir görüntüyle profil çıkarıcıyı
çalıştırabilirsiniz. Çıktı rapordur; etkin ayarları kendiliğinden değiştirmez.

```bash
/opt/helmetai-fcw/scripts/profile_detector_pi.sh \
  --image "$HOME/helmetai-v6-camera-check/front_camera_check.jpg" \
  --config /etc/helmetai-fcw/pi5_dual_camera.json \
  --warmup 3 --runs 10 --threads 1,2,3,4 \
  --output "$HOME/helmetai-v6-detector-profile.json"
```

Bu deney yalnız detector maliyetini karşılaştırır; iki kamera ile toplam
karar hızı, sahne doğruluğu veya trafik güvenliği sonucu değildir. Paketle
gelen sabit girişli modelin giriş boyutunu sırf FPS artırmak için değiştirmeyin.
En hızlı görünen CPU iş parçacığı sayısını da kontrollü canlı ölçüm olmadan
kalıcı tercih yapmayın. Alternatif örnek profil varsa tüm JSON'u `/etc/`
üstüne kopyalamayın; böyle yapmak yerel kalibrasyon/kamera eşlemesini kaybeder.

Kısa koşu sorunsuzsa ve cihaz koşulları uygunsa sınırlandırılmış zamanlama
testi ayrı olarak çalıştırılır:

```bash
/opt/helmetai-fcw/scripts/benchmark_pi.sh \
  /etc/helmetai-fcw/pi5_dual_camera.json 300 \
  | tee "$HOME/helmetai-v6-benchmark.txt"
```

Benchmark istenen kare sayısının tamamlanmasını, ısınmayı, p95 bütçesini,
minimum etkin FPS'i ve **her etkin yönün ölçülen FPS/tazelik değerini**
denetler. Ön yön hızlıyken arka yön yavaşsa geçmez. Yön telemetrisi eksikse,
tek güncelleme varsa, değerler geçersizse veya son güncelleme bayatsa geçmez.
`BENCHMARK PASSED` yalnız zamanlama kapısı sonucudur; kalibrasyon, fiziksel
uyarı çıkışı veya yolda kullanım onayı değildir. Kalibrasyonu `false`
tutarak zamanlama ölçebilirsiniz; benchmark geçsin diye güvenlik eşiklerini
gevşetmeyin.

JSON son kayıtta şu alanları birlikte inceleyin:

| Alan | Anlam |
| --- | --- |
| `performance_final.p95_end_to_end_ms` | İşleme çevriminin p95 gecikmesi; fazların çift sayılmaması gerekir. |
| `performance_final.effective_fps` | Gerçek çevrim aralıklarından etkin döngü hızı. |
| `directional_performance.front/rear.processed_fps` | O yöndeki tamamlanmış işleme güncellemelerinin ölçülen hızı. |
| `directional_performance.front/rear.last_update_age_s` | Son güncellemenin ne kadar eski olduğu. |
| `directional_performance.front/rear.capture_ms`, `inference_ms`, `processing_ms` | Kamera, model ve yön işleme süreleri; ilgili alan tanımına göre son ölçüm/özet. |

Bu tablo faz sürelerinin birbirine veya toplam gecikmeye körlemesine
ekleneceği anlamına gelmez; kapsayıcı işleme süreleri alt fazları içerebilir.
Soğutma/donanım değerlendirmesi sonraki ayrı aşamadır. Bu güncelleme sonunda
servis kapalı, GPIO kapalı ve mevcut kalibrasyon durumları değişmemiş kalır.
`performance_timestamp_s`, son kararın ölçüm zamanını gösterir; kamera kapatma
işleminin ek süresi canlı yön tazeliği hesabına katılmaz.

Önemli açık sınır: fiziksel GPIO çıkışını bağımsız bir süreçten/donanımdan
izleyen watchdog bu kamera-odaklı sürümde eklenmemiştir. Ana süreç takılırsa
bağımsız güvenli kapatma sağlandığı iddia edilmez. GPIO bu nedenle kapalı
kalır; fiziksel uyarı donanımına geçiş ayrıca tasarım ve doğrulama gerektirir.

## Elle geri alma

Önce servis kapalı kalmalıdır:

```bash
sudo systemctl stop helmetai-fcw
sudo systemctl disable helmetai-fcw
```

Güncelleyicinin yazdığı **tam** yedek yolunu bulun. `service-state.txt` ilk
`active/enabled` durumunu gösterir; bu kayıt otomatik geri başlatma talimatı
değildir. `previous/` içindeki dosyalar gerçek eski kurulum varlıklarıdır.
`replaced-assets.txt` ve `new-assets.txt` kısmi hata durumunu incelemeye yardım
eder; bir kesinti tam taşımadan sonra olduysa esas kanıt `previous/` içeriğidir.

Tek bir varlığın geri alınmasına örnek (yer tutucuyu gerçek yedek adıyla
değiştirin; aynı taşıma yolunu ikinci kez kullanmayın):

```bash
BACKUP_DIR=/opt/helmetai-fcw-backups/v6-TAM-YEDEK-ADI
sudo ls -la "$BACKUP_DIR/previous"
sudo cat "$BACKUP_DIR/service-state.txt"
ROLLBACK_HOLD="$(sudo mktemp -d /opt/helmetai-fcw-backups/rollback-hold-XXXXXX)"
sudo mv /opt/helmetai-fcw/src "$ROLLBACK_HOLD/src-v6"
sudo cp -a "$BACKUP_DIR/previous/src" /opt/helmetai-fcw/src
```

`previous/` altında bulunan diğer güncellenmiş varlıklar için de aynı
`mv mevcut -> ROLLBACK_HOLD`, `cp -a eski -> /opt/helmetai-fcw/` işlemini
uygulayın: `scripts`, `config`, `docs`, `deploy`, `pyproject.toml`, `README.md`
ve iki `requirements-*.txt` dosyası. Yalnız V6 ile eklenmiş, önceki yedekte
bulunmayan varlıkları `new-assets.txt` ile karşılaştırarak tutma dizinine
taşıyın; geri kopyalanacak eski karşılıkları yoktur. Hedef zaten yoksa `mv`
adımını atlayın. **Modellere dokunmayın.** Bu işlem silme yapmaz; V6 dosyaları
tutma dizininden geri alınabilir.

Eski etkin profilin birebir yedeği `config-original.json` dosyasıdır:

```bash
sudo cp -a /etc/helmetai-fcw/pi5_dual_camera.json "$ROLLBACK_HOLD/config-v6.json"
sudo cp -a "$BACKUP_DIR/config-original.json" /etc/helmetai-fcw/pi5_dual_camera.json
sudo nano /etc/helmetai-fcw/pi5_dual_camera.json
```

Son adımda `alert_actuator.enabled` değerini **false** doğrulayın; eski yedek
GPIO açık bir profil içerebilir. Kalibrasyon alanlarını uyarı üretmek için
açmayın. Servis tanımı güncelleyici tarafından değiştirilmediği için genellikle
unit geri yükleme gerekmez; `service-resolved.txt` karşılaştırma içindir.
Geri alma sonrasında da servis açılmaz. Preflight ve kısa kontrollü kamera
testini yeniden yapın. Donanım/kalibrasyon doğrulaması ayrı tamamlanmadan
`systemctl enable --now` çalıştırmayın.

## Doğrulama sınırı

Geliştirme bilgisayarındaki otomatik testler; yapılandırma korunmasını,
telemetri ve benchmark kararlarını doğrular. Gerçek Pi kameraları, Trixie
systemd güncelleme akışı, model hızları ve fiziksel GPIO bu bilgisayarda
çalıştırılarak doğrulanmış sayılmaz. Kendi Pi loglarınız ve ölçümleriniz
olmadan yazılımsal düzeltmeyi saha kabulü olarak değerlendirmeyin.
