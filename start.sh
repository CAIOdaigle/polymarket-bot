#!/bin/sh
# Start dashboard in background, then run the bot in foreground
python -m src.dashboard.app &
DASHBOARD_PID=$!

# If bot exits, kill dashboard too
trap "kill $DASHBOARD_PID 2>/dev/null" EXIT

python -m src.main
