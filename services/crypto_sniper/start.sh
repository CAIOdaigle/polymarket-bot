#!/bin/sh
set -e

# Only the "primary" asset container runs the dashboard. Others (e.g. ETH)
# write their stats to the shared volume and the primary dashboard picks them up.
if [ "${RUN_DASHBOARD:-1}" = "1" ]; then
    python -m crypto_sniper.dashboard.app &
    DASHBOARD_PID=$!
    cleanup() {
        kill $DASHBOARD_PID 2>/dev/null
        wait $DASHBOARD_PID 2>/dev/null
    }
    trap cleanup INT TERM EXIT
fi

exec python -m crypto_sniper.main
