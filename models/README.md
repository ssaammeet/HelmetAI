# Algılama modeli

Bu final proje paketi `detector.onnx` dosyasını içerir. Raspberry Pi çalışma
zamanı, YOLOv8/YOLO11 biçiminde ONNX dışa aktarılan COCO yol-sahnesi modelini
bekler.

Modelin lisansı, eğitim verisi ve Raspberry Pi üzerindeki hız/doğruluk ölçümü
ayrıca doğrulanmalıdır. İlk risk testinde yalnız `car`, `motorcycle`, `bus`
ve `truck` sınıfları tercih edilir; diğer algılanan sınıflar için metrik risk
kararı kalibrasyon/belirsizlik politikasına bağlıdır.

Modeli koyduktan sonra `config/pi5_dual_camera.json` içindeki `model_path`
değerinin bu dosyayı gösterdiğini doğrulayın.
