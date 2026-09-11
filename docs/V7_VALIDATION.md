# V7 masaüstü doğrulama kaydı

Bu sürümün amacı Pi bağlantısı beklenirken incelenebilir kanıt ve tekrar
çalıştırılabilir araçlar üretmektir. Yeni fiziksel Pi/kamera/GPIO koşusu yapılmadı.
Pi'ye daha önce kurulmuş V6 ve eski teslim ZIP'leri değiştirilmedi.

## Yapılan yazılım kontrolleri

- Ön/arka risk çekirdeğinin mevcut regresyon testleri korundu.
- Fotoğraf raporunda gerçek ONNX çıkarımı, geçersiz görüntü, giriş sınırları,
  çıktı üzerine yazmayı reddetme ve çakışan sınıf kutuları test edildi.
- Video testlerinde kaynak zamanı/FPS, ön/arka takip kimliği, kalibrasyonsuz
  sayıları gizleme, kaynak boyutu uyuşmazlığı, EOF/kısmi durdurma ayrımı,
  dosya çakışması ve hata halinde kaynak kapatma kontrol edildi.
- Log okuyucu birden fazla koşuyu, yinelenen kayıtları, kesik JSON'u,
  eksik ölçümleri ve güncel/geçmiş güç durumunu ayrı ele alır.
- Windows PowerShell 5.1 testleri Türkçe ve Unicode/boşluk/köşeli parantezli
  dosya yollarını, Türkçe ondalık yerel ayarını ve alt program hatalarını kapsar.
  Testlerde sahte pip kullanılır; internetten kurulum yapılmaz.
- Kamera kontrol fotoğrafı için BGR renk sırası düzeltildi. Picamera2'nin
  RGB888 dizisi OpenCV açısından BGR'dir; tekrar RGB→BGR dönüştürmek kırmızı
  ve maviyi değiştiriyordu. Bu düzeltme canlı algılama ön işlemesini değiştirmez.
  Kaynak: [Picamera2 request.py FORMAT_TABLE](https://github.com/raspberrypi/picamera2/blob/main/picamera2/request.py).
- Web kamerası konfigürasyonunda `calibrated` ve `demonstration_only` artık
  gerçek JSON boolean gerektirir; `"false"` gibi metinler reddedilir.

Tam test adı ve sonucu `V7_TEST_OUTPUT.txt` içindedir. Paketleme hem kaynakta
hem ZIP'in ayrı klasöre açılan kopyasında testleri çalıştırır; atlanan test
varsa bu geliştirme bilgisayarında paket kabul edilmez. Gerçek bir Pi'de test
edildiği anlamına gelmez. Test ortamı Windows/Python 3.10, CPU OpenCV'dir;
tüm Python/OpenCV sürüm birleşimleri denenmiş değildir.

## Gerçek yerel görüntülerde yapılan örnek koşular

| Girdi | Gözlenen sonuç | Kanıtlamadığı şey |
| --- | --- | --- |
| Kullanıcının 1920×1080 kamyon ekran görüntüsü | Ortadaki kamyon üzerinde bir `truck` kutusu, model puanı yaklaşık 0.523 | Diğer araçların tamamının bulunması veya %52.3 doğruluk |
| Kullanıcının `frontsafe.mp4` dosyası | 24 FPS kaynak, 213 kare/8.875 saniye; kaynak sonuna kadar işlendi | Pi'nin bu FPS'te çıkarım yapması veya risk doğruluğu |
| Aynı video raporu | 423 kare-bazlı tespit; 55 yerel takip kimliği; kalibrasyonsuz sayısal mesafe/TTC yok | 423 fiziksel nesne, 55 farklı araç veya güvenli trafik |
| Önceden gönderilmiş Pi logları | Tam ve kesik koşular ayrıldı; çakışan `truck`/`parking_meter` kutuları işaretlendi | Fotoğraf olmadan hangi sınıfın doğru olduğu veya fiziksel lens yönü |

Bu örnekler elle etiketlenmiş bir doğruluk veri kümesi değildir. Yakın kutular
ve model etiketi tek başına mesafe/hız doğruluğunu göstermez. Yalnız yazılım
akışının somut çıktılarıdır. İşlenmiş video ve özel log raporları ana ZIP'ten
ayrı teslim edilir; kullanıcı videosu, oda görüntüsü ve kişisel loglar ZIP'e
eklenmez. Çıktı videonun ayrıca yeniden okunması bu teslim için yapılır;
genel video çalışma modülü her çıktı MP4'ü otomatik yeniden doğrulamaz.

## Değiştirilmeyenler ve açık işler

Model SHA-256:

```text
4eefaab9b1b2547ae59434abfb7d77e30481e4fcafdeb25579f7a484129ade05
```

Model dosyası V6 ile bayt düzeyinde aynıdır; fine-tuning yapılmadı. Ön/arka
risk eşikleri, mesafe hesaplayıcı, filtre, takip, Pi performans sağlık kapısı,
GPIO sürücüsü ve Pi konfigürasyonları değiştirilmedi. Pi kontrol fotoğrafının
renk düzeltmesi donanımda yeniden sınanmalıdır. Mevcut Pi runtime sürüm
işareti bu nedenle tarihsel V6 değerini korur; V7 masaüstü paket kimliği
kök `RELEASE_MANIFEST.json` dosyasındadır.

Algılayıcının kareye esnetilen girişi, kenara taşan kutular, düşük FPS,
kameraların fiziksel yönü ve ölçüm belirsizliği için doğruluk çalışması hâlâ
gereklidir. İnternet videosuna hayalî kalibrasyon yazılmadı. `NONE` veya
uyarının bastırılması sahnenin güvenli olduğuna dair karar değildir.

Kablo geldiğinde önce net ve doğru yönlü kamera görüntüleri, ardından gerçek
kalibrasyon, sürdürülen güç/sıcaklık/gecikme ölçümü ve etiketli kapalı alan
senaryoları gerekir. Bu sürüm nihai trafik ürünü olarak sunulmamalıdır.
