"""Deterministic cold-chain sensor simulator with optional MQTT publishing."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np


MODES = ("none", "temp_drift", "vibration", "combined")


def generate_streams(duration_seconds: int, anomaly: str = "none", seed: int = 7) -> dict[str, np.ndarray]:
	if anomaly not in MODES:
		raise ValueError(f"anomaly must be one of {MODES}")
	rng = np.random.default_rng(seed)
	temperature = rng.normal(4.0, 0.3, duration_seconds)
	vibration = rng.normal(0.45, 0.05, max(1, round(duration_seconds * 0.5)))
	if anomaly in ("temp_drift", "combined"):
		temperature += np.arange(duration_seconds, dtype=float) * 0.08
	if anomaly in ("vibration", "combined"):
		vibration = rng.normal(1.2, 0.15, vibration.size)
	door = np.array([], dtype=object)
	if anomaly == "combined":
		door = np.array(["OPEN", "CLOSE"], dtype=object)
	return {"temperature": temperature, "vibration_rms": vibration, "door_event": door}


def mqtt_publish(streams: dict[str, np.ndarray], truck_id: str, host: str = "localhost", port: int = 1883) -> None:
	"""Publish one generated batch when paho-mqtt is installed; otherwise fail clearly."""
	try:
		import paho.mqtt.client as mqtt
	except ImportError as exc:
		raise SystemExit("Install paho-mqtt to publish to Mosquitto") from exc
	client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
	client.connect(host, port, 60)
	timestamp = datetime.now(timezone.utc).isoformat()
	for value in streams["temperature"]:
		client.publish(f"logibridge/trucks/{truck_id}/sensors/temperature", json.dumps({"ts": timestamp, "value": float(value)}), qos=1)
	for value in streams["vibration_rms"]:
		client.publish(f"logibridge/trucks/{truck_id}/sensors/vibration_rms", json.dumps({"ts": timestamp, "value": float(value)}), qos=1)
	for value in streams["door_event"]:
		client.publish(f"logibridge/trucks/{truck_id}/sensors/door_event", json.dumps({"ts": timestamp, "event": str(value)}), qos=1)
	client.disconnect()


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--anomaly", choices=MODES, default="none")
	parser.add_argument("--duration", type=int, default=60)
	parser.add_argument("--truck-id", default="FB-001")
	parser.add_argument("--publish", action="store_true")
	args = parser.parse_args()
	streams = generate_streams(args.duration, args.anomaly)
	if args.publish:
		mqtt_publish(streams, args.truck_id)
	else:
		print(json.dumps({"anomaly": args.anomaly, "temperature_samples": len(streams["temperature"]), "vibration_samples": len(streams["vibration_rms"]), "door_events": streams["door_event"].tolist()}))


if __name__ == "__main__":
	main()
