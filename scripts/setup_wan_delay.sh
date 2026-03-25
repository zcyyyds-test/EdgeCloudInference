#!/bin/bash
# Apply WAN delay to the edge Docker container.
# Usage: bash scripts/setup_wan_delay.sh [delay_ms] [jitter_ms]
#
# Requires: edge-sim container running with NET_ADMIN capability

DELAY=${1:-50}
JITTER=${2:-10}

echo "Setting WAN delay: ${DELAY}ms ± ${JITTER}ms on edge-sim"

# Clear existing rules
docker exec edge-sim tc qdisc del dev eth0 root 2>/dev/null || true

# Apply netem delay
docker exec edge-sim tc qdisc add dev eth0 root netem \
    delay ${DELAY}ms ${JITTER}ms distribution normal

echo "Done. Verify with: docker exec edge-sim tc qdisc show dev eth0"
