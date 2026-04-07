#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="llama-server"
SERVICE_FILE="${SCRIPT_DIR}/llama-server.service"
SYSTEMD_DIR="${HOME}/.config/systemd/user"
SYSTEMD_SERVICE="${SYSTEMD_DIR}/${SERVICE_NAME}.service"

install_service() {
  if [ ! -f "${SERVICE_FILE}" ]; then
    echo "[gpu-setup] ERROR: Service file not found at ${SERVICE_FILE}" >&2
    exit 1
  fi

  mkdir -p "${SYSTEMD_DIR}"

  # Replace %h with actual home directory for compatibility
  sed "s|%h|${HOME}|g" "${SERVICE_FILE}" > "${SYSTEMD_SERVICE}"

  systemctl --user daemon-reload
}

start_daemon() {
  install_service
  echo "[gpu-setup] Starting llama-server as a systemd user service..."
  systemctl --user start "${SERVICE_NAME}.service"
  systemctl --user enable "${SERVICE_NAME}.service"
  echo "[gpu-setup] Service started. Check status with: ./start.sh status"
  echo "[gpu-setup] View logs with: ./start.sh logs"
}

stop_daemon() {
  systemctl --user stop "${SERVICE_NAME}.service" 2>/dev/null
  systemctl --user disable "${SERVICE_NAME}.service" 2>/dev/null
  echo "[gpu-setup] Service stopped."
}

status_daemon() {
  systemctl --user status "${SERVICE_NAME}.service"
}

logs_daemon() {
  journalctl --user -u "${SERVICE_NAME}.service" -f
}

cmd="${1:-start}"
case "${cmd}" in
  start)   start_daemon ;;
  stop)    stop_daemon ;;
  restart) stop_daemon && start_daemon ;;
  status)  status_daemon ;;
  logs)    logs_daemon ;;
  *)
    echo "Usage: $(basename "$0") {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
