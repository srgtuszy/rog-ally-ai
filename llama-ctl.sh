#!/usr/bin/env bash
# Helper script to manage the llama.cpp ROCm server
# Usage: ./llama-ctl.sh {start|stop|restart|status|logs}

set -euo pipefail

SERVICE="llama-server.service"

case "${1:-status}" in
    start)
        systemctl --user start "$SERVICE"
        echo "✓ llama-server started"
        sleep 2
        systemctl --user status "$SERVICE" --no-pager
        ;;
    stop)
        systemctl --user stop "$SERVICE"
        echo "✓ llama-server stopped"
        ;;
    restart)
        systemctl --user restart "$SERVICE"
        echo "✓ llama-server restarted"
        sleep 2
        systemctl --user status "$SERVICE" --no-pager
        ;;
    status)
        systemctl --user status "$SERVICE" --no-pager
        ;;
    logs)
        journalctl --user -u "$SERVICE" -n 50 --no-pager
        ;;
    enable)
        systemctl --user enable "$SERVICE"
        echo "✓ llama-server enabled (auto-start at login)"
        ;;
    disable)
        systemctl --user disable "$SERVICE"
        echo "✓ llama-server disabled (manual start only)"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
        exit 1
        ;;
esac
