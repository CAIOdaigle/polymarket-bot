#!/bin/sh
set -e

# Start dashboard in background
python -m src.dashboard.app &
DASHBOARD_PID=$!

# Forward signals to both processes for clean shutdown
cleanup() {
    kill $DASHBOARD_PID 2>/dev/null
    wait $DASHBOARD_PID 2>/dev/null
}
trap cleanup INT TERM EXIT

# Run bot in foreground (exec replaces shell so signals reach Python directly)
exec python -m src.main
