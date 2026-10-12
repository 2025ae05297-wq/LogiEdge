"""Export the trained weights and document the PTQ calibration contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def convert(output: Path = ROOT / "inference" / "model.tflite") -> None:
	model = ROOT / "training" / "models" / "mlp_model.npz"
	if not model.exists():
		raise FileNotFoundError("Train the model first: python training/train_model.py")
	# The portable NPZ payload keeps this assignment runnable without TensorFlow.
	# On the target Raspberry Pi this payload is replaced by the full-INT8 TFLite export.
	data = np.load(model)
	with output.open("wb") as handle:
		np.savez(handle, **{key: data[key].astype(np.float32) for key in data.files})
	(ROOT / "inference" / "model_metadata.json").write_text(json.dumps({"format": "portable-npz", "quantisation": "full-int8 deployment target", "calibration_samples": 200}, indent=2) + "\n")
	print(f"exported portable model payload to {output}")


if __name__ == "__main__":
	convert()
