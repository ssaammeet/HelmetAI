#!/usr/bin/env bash
# Software-only update for an existing /opt installation. No apt, pip,
# model replacement, calibration edits, GPIO activation or automatic restart.
set -euo pipefail
umask 077

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TARGET_DIR=/opt/helmetai-fcw
CONFIG_PATH=/etc/helmetai-fcw/pi5_dual_camera.json
BACKUP_ROOT=/opt/helmetai-fcw-backups
SERVICE=helmetai-fcw.service
BACKUP_DIR=
SERVICE_TOUCHED=0
ASSETS=(src scripts config docs deploy pyproject.toml README.md requirements-webcam.txt requirements-optimization.txt)

fail() { echo "UPDATE FAILED: $*" >&2; exit 2; }
on_exit() {
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    if [[ "$SERVICE_TOUCHED" == 1 ]]; then
      systemctl stop "$SERVICE" || true
      if [[ "$ORIGINAL_ENABLED" != masked && "$ORIGINAL_ENABLED" != masked-runtime ]]; then
        systemctl disable "$SERVICE" || true
      fi
      echo "Service was not restarted. Inspect the backup and rollback instructions." >&2
    fi
    [[ -z "$BACKUP_DIR" ]] || echo "Recovery directory: $BACKUP_DIR" >&2
  fi
}
trap on_exit EXIT

[[ "${1:-}" == "" ]] || fail "No arguments supported. Run sudo bash deploy/update_pi.sh from the extracted release."
[[ "$(uname -s)" == Linux ]] || fail "This updater is for Linux on the existing Pi only."
[[ "$EUID" == 0 ]] || fail "Run with sudo bash deploy/update_pi.sh."
for command in python3 systemctl realpath mktemp install cp mv flock chown; do
  command -v "$command" >/dev/null 2>&1 || fail "Required command is unavailable: $command"
done
[[ -d "$TARGET_DIR/src/helmetai_fcw" ]] || fail "Existing installation not found: $TARGET_DIR"
[[ -f "$CONFIG_PATH" ]] || fail "Existing profile not found: $CONFIG_PATH"
[[ "$(realpath -e "$TARGET_DIR")" == "$TARGET_DIR" ]] || fail "Installation path must not be a symlink."
[[ "$(realpath -e "$CONFIG_PATH")" == "$CONFIG_PATH" ]] || fail "Configuration path must not be a symlink."
case "$PROJECT_DIR/" in
  "$TARGET_DIR/"*|"$BACKUP_ROOT/"*) fail "Extract the release outside the installation and backup directories." ;;
esac
for asset in "${ASSETS[@]}"; do
  [[ -e "$PROJECT_DIR/$asset" && ! -L "$PROJECT_DIR/$asset" ]] || fail "Missing or symlinked release asset: $asset"
  [[ ! -L "$TARGET_DIR/$asset" ]] || fail "Installed asset is a symlink; review manually: $asset"
done
[[ -f "$PROJECT_DIR/deploy/prepare_camera_only_config.py" ]] || fail "Config staging helper is missing."
[[ -f "$PROJECT_DIR/src/helmetai_fcw/pi_runtime.py" ]] || fail "Invalid flat release layout."

# Reject links and syntax errors before staging or stopping the service.
python3 - "$PROJECT_DIR" "${ASSETS[@]}" <<'PY'
import ast
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in sys.argv[2:]:
    asset = root / name
    paths = [asset, *asset.rglob("*")] if asset.is_dir() else [asset]
    for path in paths:
        if path.is_symlink():
            raise SystemExit(f"Release symlink is not allowed: {path}")
        if path.is_file() and path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
PY
for script in "$PROJECT_DIR"/deploy/*.sh "$PROJECT_DIR"/scripts/*.sh; do
  bash -n "$script"
done

[[ "$(systemctl show "$SERVICE" --property=LoadState --value)" == loaded ]] || fail "Existing systemd unit is not loaded."
SERVICE_USER="$(systemctl show "$SERVICE" --property=User --value)"
[[ -n "$SERVICE_USER" && "$SERVICE_USER" != root && "$SERVICE_USER" != __HELMETAI_USER__ ]] || fail "Service must have an explicit non-root User."
SERVICE_UID="$(id -u "$SERVICE_USER")" || fail "Service user does not exist: $SERVICE_USER"
[[ "$SERVICE_UID" != 0 ]] || fail "Service user resolves to root."
[[ "$(systemctl show "$SERVICE" --property=WorkingDirectory --value)" == "$TARGET_DIR" ]] || fail "Unexpected service working directory."
SERVICE_EXEC="$(systemctl show "$SERVICE" --property=ExecStart --value)"
[[ "$SERVICE_EXEC" == *"-m helmetai_fcw pi-run --config $CONFIG_PATH"* ]] || fail "Service does not use the expected Pi runtime/config path."

if [[ -e "$BACKUP_ROOT" ]]; then
  [[ -d "$BACKUP_ROOT" && "$(realpath -e "$BACKUP_ROOT")" == "$BACKUP_ROOT" ]] || fail "Backup root must be a real directory."
else
  install -d -m 700 "$BACKUP_ROOT"
fi
exec 9>"$BACKUP_ROOT/update.lock"
flock -n 9 || fail "Another software update is already running."
BACKUP_DIR="$(mktemp -d "$BACKUP_ROOT/v6-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
install -d -m 700 "$BACKUP_DIR/staged" "$BACKUP_DIR/previous"
ORIGINAL_ENABLED="$(systemctl is-enabled "$SERVICE" 2>/dev/null || true)"
ORIGINAL_ACTIVE="$(systemctl is-active "$SERVICE" 2>/dev/null || true)"
printf 'service=%s\nuser=%s\nenabled=%s\nactive=%s\nrelease=%s\n' \
  "$SERVICE" "$SERVICE_USER" "$ORIGINAL_ENABLED" "$ORIGINAL_ACTIVE" "$PROJECT_DIR" >"$BACKUP_DIR/service-state.txt"
systemctl cat "$SERVICE" >"$BACKUP_DIR/service-resolved.txt"
cp -a "$CONFIG_PATH" "$BACKUP_DIR/config-original.json"
PYTHONPATH="$PROJECT_DIR/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 "$PROJECT_DIR/deploy/prepare_camera_only_config.py" "$CONFIG_PATH" "$BACKUP_DIR/staged/config-camera-only.json"
for asset in "${ASSETS[@]}"; do
  cp -a "$PROJECT_DIR/$asset" "$BACKUP_DIR/staged/$asset"
done
chown -R root:root "$BACKUP_DIR/staged"
chmod -R u=rwX,go=rX "$BACKUP_DIR/staged/src" "$BACKUP_DIR/staged/scripts" "$BACKUP_DIR/staged/config" "$BACKUP_DIR/staged/docs" "$BACKUP_DIR/staged/deploy"
chmod 644 "$BACKUP_DIR/staged/"*.toml "$BACKUP_DIR/staged/"*.md "$BACKUP_DIR/staged/"*.txt
chmod 755 "$BACKUP_DIR/staged/"{scripts,deploy}/*.sh
# Delete only generated bytecode inside the newly-created stage, never inside
# the installed tree or source release. Source files remain unchanged.
python3 - "$BACKUP_DIR/staged" <<'PY'
import sys
from pathlib import Path

stage = Path(sys.argv[1]).resolve()
for path in stage.rglob("*.pyc"):
    if path.is_file() and path.resolve().is_relative_to(stage):
        path.unlink()
PY

echo "Validated existing profile. Backup/recovery directory: $BACKUP_DIR"
SERVICE_TOUCHED=1
systemctl stop "$SERVICE"
if [[ "$ORIGINAL_ENABLED" != masked && "$ORIGINAL_ENABLED" != masked-runtime ]]; then
  systemctl disable "$SERVICE"
fi
[[ "$(systemctl show "$SERVICE" --property=ActiveState --value)" == inactive ]] || fail "Service is not stopped."

# Each old asset is moved intact into the unique backup. Never touch models,
# other installation data, the unit file, calibration or camera mapping.
for asset in "${ASSETS[@]}"; do
  if [[ -e "$TARGET_DIR/$asset" ]]; then
    mv -- "$TARGET_DIR/$asset" "$BACKUP_DIR/previous/$asset"
    printf '%s\n' "$asset" >>"$BACKUP_DIR/replaced-assets.txt"
  else
    printf '%s\n' "$asset" >>"$BACKUP_DIR/new-assets.txt"
  fi
  mv -- "$BACKUP_DIR/staged/$asset" "$TARGET_DIR/$asset"
done
CONFIG_TEMP="$(mktemp /etc/helmetai-fcw/.v6-camera-only-XXXXXX)"
cp --preserve=mode,ownership "$CONFIG_PATH" "$CONFIG_TEMP"
cp "$BACKUP_DIR/staged/config-camera-only.json" "$CONFIG_TEMP"
mv -- "$CONFIG_TEMP" "$CONFIG_PATH"
PYTHONPATH="$TARGET_DIR/src" PYTHONDONTWRITEBYTECODE=1 \
  python3 - "$CONFIG_PATH" <<'PY'
import sys
from helmetai_fcw.pi_runtime import PiRuntimeConfig

config = PiRuntimeConfig.from_json(sys.argv[1])
if config.alert_actuator.enabled:
    raise SystemExit("Post-update check failed: GPIO alerts must stay disabled.")
PY
printf 'complete\n' >"$BACKUP_DIR/update-status.txt"
echo "Software update complete. Existing models and profile values preserved; GPIO disabled."
echo "Service remains stopped and disabled (an existing mask remains in place)."
echo "Backup: $BACKUP_DIR"
echo "Next: read $TARGET_DIR/docs/PI_SOFTWARE_UPDATE_V6_TR.md and run preflight as $SERVICE_USER, not root."
