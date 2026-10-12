"""Generate the reproducible labelled dataset used by the training script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.preprocessing import fit_stats, window_features
from data_pipeline.simulator import generate_streams


def build_dataset(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
	specifications = ((0, "none", 20 * 60), (1, "temp_drift", 15 * 60), (2, "combined", 15 * 60))
	feature_sets = []
	labels = []
	for label, mode, duration in specifications:
		streams = generate_streams(duration, mode, seed + label)
		features = window_features(streams["temperature"], streams["vibration_rms"])
		feature_sets.append(features)
		labels.append(np.full(features.shape[0], label, dtype=np.int64))
	return np.vstack(feature_sets), np.concatenate(labels)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", default=str(ROOT / "training" / "dataset.npz"))
	args = parser.parse_args()
	features, labels = build_dataset()
	clean = features[labels == 0]
	stats_path = ROOT / "data_pipeline" / "training_stats.npy"
	fit_stats(clean, stats_path)
	np.savez(args.output, features=features, labels=labels)
	print(f"saved {len(labels)} windows to {args.output}; clean stats to {stats_path}")


if __name__ == "__main__":
	main()
