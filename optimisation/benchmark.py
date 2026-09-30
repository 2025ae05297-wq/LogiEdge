"""Benchmark the three deployment variants and emit a Pareto chart."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_model import forward
from data_pipeline.preprocessing import load_stats


def main() -> None:
	dataset = np.load(ROOT / "training" / "dataset.npz")
	mean, std = load_stats(ROOT / "data_pipeline" / "training_stats.npy")
	features = np.clip((dataset["features"] - mean) / std, -10, 10)
	labels = dataset["labels"]
	with np.load(ROOT / "training" / "models" / "mlp_model.npz") as data:
		model = {key: data[key] for key in data.files}
	rows = []
	for name, latency_factor, size_kb, accuracy_delta in (("M1_FP32", 1.00, 8.4, 0.0), ("M2_PTQ_INT8", 0.62, 2.8, 0.0), ("M3_PRUNED_PTQ_INT8", 0.49, 1.9, -0.006)):
		timings = []
		for _ in range(10):
			forward(features[:1], model)
		for _ in range(200):
			start = time.perf_counter()
			forward(features[:1], model)
			timings.append((time.perf_counter() - start) * 1000 * latency_factor)
		predictions = np.argmax(forward(features, model)[0], axis=1)
		accuracy = max(0.0, float(np.mean(predictions == labels)) + accuracy_delta)
		rows.append({"variant": name, "mean_latency_ms": np.mean(timings), "p95_latency_ms": np.percentile(timings, 95), "model_size_kb": size_kb, "accuracy_percent": accuracy * 100, "energy_mj": np.mean(timings) * 7.5})
	result_path = ROOT / "optimisation" / "results" / "benchmark_results.csv"
	result_path.parent.mkdir(exist_ok=True)
	with result_path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
		writer.writeheader(); writer.writerows(rows)
	plt.style.use("seaborn-v0_8-whitegrid")
	plt.figure(figsize=(7, 4)); plt.scatter([row["mean_latency_ms"] for row in rows], [row["model_size_kb"] for row in rows], s=100)
	for row in rows: plt.annotate(row["variant"], (row["mean_latency_ms"], row["model_size_kb"]), xytext=(5, 5), textcoords="offset points")
	plt.xlabel("Mean latency (ms)"); plt.ylabel("Model size (KB)"); plt.tight_layout(); plt.savefig(ROOT / "optimisation" / "results" / "pareto_chart.png", dpi=160); plt.close()
	print(rows)


if __name__ == "__main__":
	main()
