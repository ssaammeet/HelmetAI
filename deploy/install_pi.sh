#!/usr/bin/env bash
# Run manually on Raspberry Pi OS Bookworm/Trixie. This installer deliberately does
# not enable the service: cameras must be mapped, calibrated and benchmarked
# before a rider-facing process is started.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_USER="${HELMETAI_USER:-${SUDO_USER:-$(id -un)}}"

if [[ "$TARGET_USER" == "root" ]]; then
  echo "Run this as the regular Pi user, or set HELMETAI_USER=<username> before running it." >&2
  exit 2
fi
if ! id "$TARGET_USER" >/dev/null 2>&1; then
  echo "HelmetAI service user does not exist: $TARGET_USER" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_DIR/models/detector.onnx" ]]; then
  echo "Detector model is missing: $PROJECT_DIR/models/detector.onnx" >&2
  exit 2
fi

sudo apt update
# rpicam-hello is used by the mandatory preflight check.  Raspberry Pi OS
# Bookworm packages it as rpicam-apps (or includes it already), but install it
# explicitly so a minimal image cannot appear to install successfully and then
# fail before any camera validation.
sudo apt install -y --no-install-recommends rpicam-apps gpiod python3-picamera2 python3-opencv python3-numpy
# GPIO alerts are disabled by default.  Do not make the camera-only install
# fail on a distribution image whose package index does not expose lgpio.
if apt-cache show python3-lgpio >/dev/null 2>&1; then
  sudo apt install -y --no-install-recommends python3-lgpio
else
  echo "WARNING: python3-lgpio is not available in this apt index. GPIO alerts remain disabled; preflight will block an enabled GPIO profile."
fi

sudo install -d -m 755 /opt/helmetai-fcw /opt/helmetai-fcw/models /opt/helmetai-fcw/scripts /etc/helmetai-fcw
sudo cp -r "${PROJECT_DIR}/src" /opt/helmetai-fcw/
sudo cp "${PROJECT_DIR}/pyproject.toml" /opt/helmetai-fcw/
sudo cp -r "${PROJECT_DIR}/models/." /opt/helmetai-fcw/models/
sudo install -m 755 "${PROJECT_DIR}/scripts/benchmark_pi.sh" /opt/helmetai-fcw/scripts/benchmark_pi.sh
sudo install -m 755 "${PROJECT_DIR}/scripts/preflight_pi.sh" /opt/helmetai-fcw/scripts/preflight_pi.sh
sudo install -m 755 "${PROJECT_DIR}/scripts/profile_detector_pi.sh" /opt/helmetai-fcw/scripts/profile_detector_pi.sh
sudo install -m 755 "${PROJECT_DIR}/scripts/run_pi_diagnostic.sh" /opt/helmetai-fcw/scripts/run_pi_diagnostic.sh

if [[ ! -f /etc/helmetai-fcw/pi5_dual_camera.json ]]; then
  sudo cp "${PROJECT_DIR}/config/pi5_dual_camera.example.json" /etc/helmetai-fcw/pi5_dual_camera.json
  echo "Config copied: /etc/helmetai-fcw/pi5_dual_camera.json"
  echo "Önce config içindeki calibration ve model_path değerlerini güncelleyin."
fi

if getent group video >/dev/null 2>&1; then
  sudo usermod -aG video "$TARGET_USER"
fi
if getent group gpio >/dev/null 2>&1; then
  sudo usermod -aG gpio "$TARGET_USER"
fi

sed "s|__HELMETAI_USER__|${TARGET_USER}|g" "${PROJECT_DIR}/deploy/helmetai-fcw.service" |
  sudo tee /etc/systemd/system/helmetai-fcw.service >/dev/null
sudo chmod 644 /etc/systemd/system/helmetai-fcw.service
sudo systemctl daemon-reload
echo "Kurulum tamamlandı. Service user: $TARGET_USER"
echo "Oturum gruplarının (video/gpio) yenilenmesi için önce çıkış yapıp tekrar giriş yapın veya Pi'yi yeniden başlatın."
echo "1) /opt/helmetai-fcw/scripts/preflight_pi.sh çalıştırın ve ön/arka kamera indekslerini doğrulayın."
echo "2) /etc/helmetai-fcw/pi5_dual_camera.json içindeki iki kamerayı nihai montajda kalibre edin."
echo "3) /opt/helmetai-fcw/scripts/benchmark_pi.sh ile gecikme testini geçin."
echo "4) Yalnız bunlardan sonra başlatın: sudo systemctl enable --now helmetai-fcw"
