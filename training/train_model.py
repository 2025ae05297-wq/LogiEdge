"""Train and export a small two-hidden-layer MLP using only NumPy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline.preprocessing import load_stats
try:
	from generate_dataset import build_dataset
except ModuleNotFoundError:
	from training.generate_dataset import build_dataset


def softmax(logits: np.ndarray) -> np.ndarray:
	shifted = logits - np.max(logits, axis=1, keepdims=True)
	probabilities = np.exp(shifted)
	return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def forward(features: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
	hidden_1 = np.maximum(0, features @ weights["w1"] + weights["b1"])
	hidden_2 = np.maximum(0, hidden_1 @ weights["w2"] + weights["b2"])
	return softmax(hidden_2 @ weights["w3"] + weights["b3"]), (hidden_1, hidden_2)


def predict(features: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
	return np.argmax(forward(features, weights)[0], axis=1)


def train(features: np.ndarray, labels: np.ndarray, epochs: int = 800, seed: int = 11) -> tuple[dict[str, np.ndarray], float, float]:
	rng = np.random.default_rng(seed)
	order = rng.permutation(len(labels))
	split = int(len(labels) * 0.8)
	train_indices, validation_indices = order[:split], order[split:]
	x_train, y_train = features[train_indices], labels[train_indices]
	x_valid, y_valid = features[validation_indices], labels[validation_indices]
	weights = {
		"w1": rng.normal(0, 0.18, (6, 32)), "b1": np.zeros(32),
		"w2": rng.normal(0, 0.18, (32, 16)), "b2": np.zeros(16),
		"w3": rng.normal(0, 0.18, (16, 3)), "b3": np.zeros(3),
	}
	one_hot = np.eye(3)[y_train]
	for epoch in range(epochs):
		probabilities, (hidden_1, hidden_2) = forward(x_train, weights)
		gradient = (probabilities - one_hot) / len(y_train)
		gradients = {
			"w3": hidden_2.T @ gradient, "b3": np.sum(gradient, axis=0),
		}
		hidden_2_gradient = (gradient @ weights["w3"].T) * (hidden_2 > 0)
		gradients.update({"w2": hidden_1.T @ hidden_2_gradient, "b2": np.sum(hidden_2_gradient, axis=0)})
		hidden_1_gradient = (hidden_2_gradient @ weights["w2"].T) * (hidden_1 > 0)
		gradients.update({"w1": x_train.T @ hidden_1_gradient, "b1": np.sum(hidden_1_gradient, axis=0)})
		learning_rate = 0.04 if epoch < 500 else 0.01
		for name in weights:
			weights[name] -= learning_rate * gradients[name]
	accuracy = float(np.mean(predict(x_valid, weights) == y_valid))
	return weights, accuracy, float(np.mean(predict(x_train, weights) == y_train))


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--output", default=str(ROOT / "training" / "models" / "mlp_model.npz"))
	args = parser.parse_args()
	dataset_path = ROOT / "training" / "dataset.npz"
	if dataset_path.exists():
		dataset = np.load(dataset_path)
		raw_features, labels = dataset["features"], dataset["labels"]
	else:
		raw_features, labels = build_dataset()
	mean, std = load_stats(ROOT / "data_pipeline" / "training_stats.npy")
	features = np.clip((raw_features - mean) / std, -10.0, 10.0)
	weights, validation_accuracy, training_accuracy = train(features, labels)
	Path(args.output).parent.mkdir(parents=True, exist_ok=True)
	np.savez(args.output, **weights)
	metrics = {"validation_accuracy": validation_accuracy, "training_accuracy": training_accuracy, "samples": int(len(labels))}
	(ROOT / "training" / "models" / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()
