"""Offline-first inference service; MQTT is an optional transport adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.preprocessing import normalise
from training.train_model import forward


def load_model(path: str | Path) -> dict[str, np.ndarray]:
	with np.load(path) as model:
		return {key: model[key] for key in model.files}


def infer(features: np.ndarray, model_path: str | Path, stats_path: str | Path) -> tuple[int, float, np.ndarray]:
	model = load_model(model_path)
	probabilities, _ = forward(normalise(np.asarray(features), stats_path), model)
	index = int(np.argmax(probabilities[0]))
	return index, float(probabilities[0, index]), probabilities[0]


def publish_result(result: dict, truck_id: str, host: str = "localhost", port: int = 1883) -> None:
	try:
		import paho.mqtt.client as mqtt
	except ImportError as exc:
		raise SystemExit("Install paho-mqtt to publish inference results") from exc
	client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
	client.connect(host, port, 60)
	client.publish(f"logibridge/trucks/{truck_id}/inference", json.dumps(result), qos=1)
	client.disconnect()


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--features", nargs=6, type=float, required=True)
	parser.add_argument("--truck-id", default="FB-001")
	parser.add_argument("--publish", action="store_true")
	args = parser.parse_args()
	model_path = os.environ.get("MODEL_PATH", str(ROOT / "inference" / "model.tflite"))
	label, confidence, scores = infer(np.array([args.features], dtype=np.float32), model_path, ROOT / "data_pipeline" / "training_stats.npy")
	result = {"truck_id": args.truck_id, "class": label, "confidence": confidence, "scores": scores.tolist()}
	(ROOT / "inference" / "local_alert_log.jsonl").open("a").write(json.dumps(result) + "\n")
	if args.publish:
		publish_result(result, args.truck_id)
	print(json.dumps(result))


if __name__ == "__main__":
	main()
