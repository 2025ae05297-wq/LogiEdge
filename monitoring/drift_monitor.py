"""PSI monitor for model confidence scores."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BINS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])


def distribution(scores: np.ndarray) -> np.ndarray:
	counts, _ = np.histogram(np.clip(scores, 0, 1), bins=BINS)
	return (counts + 1e-6) / (np.sum(counts) + 4e-6)


def psi(reference: np.ndarray, current: np.ndarray) -> float:
	reference = np.maximum(reference, 1e-6)
	current = np.maximum(current, 1e-6)
	return float(np.sum((current - reference) * np.log(current / reference)))


def save_reference(scores: np.ndarray, path: str | Path = ROOT / "monitoring" / "reference_dist.json") -> None:
	Path(path).write_text(json.dumps({"bins": BINS.tolist(), "distribution": distribution(scores).tolist()}, indent=2) + "\n")


def monitor(scores: list[float], reference_path: str | Path = ROOT / "monitoring" / "reference_dist.json") -> float:
	reference = np.asarray(json.loads(Path(reference_path).read_text())["distribution"])
	value = psi(reference, distribution(np.asarray(scores)))
	print(f"PSI={value:.3f}")
	if value > 0.25:
		print(f"[LOGIBRIDGE DRIFT ALERT] PSI={value:.3f}")
	return value


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--scores", nargs="+", type=float, required=True)
	args = parser.parse_args()
	monitor(args.scores)


if __name__ == "__main__":
	main()
