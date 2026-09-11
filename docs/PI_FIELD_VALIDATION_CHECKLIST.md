# HelmetAI Pi ve kontrollü saha doğrulama kontrol listesi

Bu belge bir "geçti" raporu değildir. Yazılım paketi, aşağıdaki ölçümler gerçek
Raspberry Pi, nihai kamera montajı ve gerçek uyarı devresi üzerinde kaydedilene
kadar yalnızca geliştirme prototipi olarak kalır.

## Servisi açmadan önce zorunlu kapılar

1. Raspberry Pi OS Bookworm, iki CSI kamera ve aktif soğutma ile kurulumu
   tamamlayın. Kurulumdan sonra grup üyelikleri için yeniden giriş yapın.
2. `/opt/helmetai-fcw/scripts/preflight_pi.sh` komutunu çalıştırın. İki kamera,
   model yükleme/smoke testi ve — GPIO etkinse — gerçek gpiochip erişimi başarılı
   olmadan devam etmeyin.
3. `pi-camera-check` ile görüntüdeki FRONT/REAR etiketlerini lensleri tek tek
   kapatarak doğrulayın. Arka görüntüde sol/sağ yön işaretinin sürücünün gerçek
   yönüyle eşleştiğini kaydedin.
4. Nihai kask montajında her kamera için en az üç bilinen mesafede mesafe hatası
   ölçün. Hata kabul edilmediyse ilgili `calibrated` alanı `false` kalmalıdır.
5. Kapalı alanda en az 100 karelik smoke test yapın. İstikrarsız kamera,
   boş kare, yanlış kamera indeksi veya `DEGRADED` nedeni varsa düzeltin.
6. Her iki kamera açıkken 300 karelik
   `benchmark_pi.sh /etc/helmetai-fcw/pi5_dual_camera.json 300` ölçümünü
   çalıştırın. Komut yalnızca tam ısınma, p95 gecikme, minimum FPS ve sağlık
   kapısı birlikte geçerse `BENCHMARK PASSED` döndürür.

## GPIO uyarı devresi kontrolü

1. Buzzer, LED ve titreşim motorunun hangi BCM pinine ve hangi sürücü
   devresine bağlı olduğunu şemada kaydedin. Motor veya yüksek akımlı buzzer
   Pi GPIO'suna doğrudan bağlanmaz.
2. Her kanal önce yük bağlı değilken, sonra uygun yük/sürücüyle düşük riskli
   masaüstü testinde doğrulanmalıdır. Ortak GND, lojik aktif seviyesi ve güç
   kaynağı ayrıca kontrol edilir.
3. Yapılandırmada `gpio_chip: null` bırakılırsa yazılım `pinctrl*` etiketli
   fiziksel header denetleyicisini otomatik bulur. Bu belirsiz olursa
   `gpiodetect` çıktısına göre yalnız doğrulanmış chip numarasını yazın.
4. `alert_actuator.enabled: true` yalnız kablolama tamamlandıktan sonra
   açılır. `ADVISORY`, `WARNING`, `CRITICAL` ve sağlık kapısı kapanma
   davranışını kontrollü senaryoda ayrı ayrı kaydedin.

## Kontrollü hareket testi

Ön ve arka yönde ayrı test serileri kaydedin: sabit hedef, sabit takip,
yavaş yaklaşma ve hızlı yaklaşma. Her seride gerçek başlangıç mesafesi,
yaklaşma hızı, ilk uyarı anı, TTC çıktısı ve sürücü uyarı çıktısı tutulur.
Trafiğe çıkmadan önce yanlış pozitif/negatif sonuçlar gözden geçirilmelidir.

Bu kontrol listesi trafik kullanım izni veya işlevsel güvenlik sertifikası
sağlamaz. Bu proje için üretim ADAS doğrulaması; daha geniş veri seti,
tekrarlanabilir saha testleri, hata analizi ve ilgili güvenlik/uyumluluk
sürecini gerektirir.
