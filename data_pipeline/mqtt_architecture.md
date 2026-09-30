# MQTT Architecture

```text
logibridge/trucks/{truck_id}/
├── sensors/temperature       (QoS 1, retained false)
├── sensors/vibration_rms     (QoS 1, retained false)
├── sensors/door_event        (QoS 1, retained false)
├── inference                 (QoS 1, retained false)
├── alerts                    (QoS 2, retained false)
└── sync/status               (QoS 1, retained false)
```

The broker is bound to localhost. Temperature and vibration use QoS 1 because an occasional duplicate is safer than silent loss and the consumer de-duplicates by timestamp. Door events and alerts use QoS 2 because a missed state transition or escalation is unacceptable. Retained messages are disabled for raw readings so a new subscriber does not receive stale sensor data. The local alert log is durable and is synchronised to the operations backend over a TLS cellular uplink after reconnect.
