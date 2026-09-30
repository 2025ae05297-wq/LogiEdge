# FreightBridge Constraint Analysis

## Latency

The operational SLA is detection and alerting within 90 seconds of a fault signature. A cloud-only path adds sensor uplink, rural cellular scheduling, Internet backhaul, cloud inference, and downlink acknowledgement. Even a nominal 0.5–2 second round trip becomes non-deterministic at the seven documented 35–90 minute outage locations. Local filtering, feature extraction, and inference execute continuously, so the safety decision does not wait for a network response.

## Bandwidth

Using four bytes per scalar, temperature produces `1 x 86,400 x 4 = 345,600 B/day`. Three-axis vibration at 500 Hz produces `3 x 500 x 86,400 x 4 = 518,400,000 B/day`. The raw total is approximately 518.75 MB/day/truck, or ₹51.88/truck/day at ₹0.10/MB and ₹4,409.80/day for 85 trucks. The edge sends only compact event records; even ten 512-byte alerts/day is 0.00512 MB and ₹0.000512/truck/day.

## Connectivity

During a cloud outage, a cloud-only system has no reliable current state, cannot acknowledge a critical event, and risks losing the chain-of-custody timeline. LogiEdge continues sampling, maintains a local MQTT broker, evaluates windows, and appends immutable timestamped alerts. A store-and-forward uplink drains the queue after coverage returns, preserving event order and a connectivity-gap marker.

## Privacy

Raw vibration and temperature streams stay on the truck. The backend receives a signed, minimal alert record rather than a continuous cargo telemetry feed. Device identity keys, encrypted local storage, TLS uplink, and access-controlled retention provide evidence for pharmaceutical clients that unauthorised third parties cannot inspect the cargo condition stream.
