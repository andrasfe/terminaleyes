#!/usr/bin/env bash
# jiggler.sh — Move the BT HID mouse every 3 minutes to prevent sleep.
# Usage: bash scripts/jiggler.sh [PI_HOST]
# Kill with Ctrl+C.

PI="${1:-10.0.0.2}"
URL="http://$PI:8080/bt/mouse/move"
INTERVAL=180

echo "Jiggling mouse on $PI every ${INTERVAL}s. Ctrl+C to stop."
while true; do
    curl -s -X POST -H 'Content-Type: application/json' -d '{"x":50,"y":0}' "$URL" > /dev/null
    echo "$(date '+%H:%M:%S') jiggled"
    sleep $INTERVAL
done
