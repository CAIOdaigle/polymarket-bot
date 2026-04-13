#!/bin/sh
set -e

# Start dashboard in background
python -m crypto_sniper.dashboard.app &
DASHBOARD_PID=$!

cleanup() {
    kill $DASHBOARD_PID 2>/dev/null
    wait $DASHBOARD_PID 2>/dev/null
}
trap cleanup INT TERM EXIT

# Run sniper in foreground
exec python -m crypto_sniper.main
