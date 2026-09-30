"""Filtering, feature extraction, and fixed training-stat normalisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_NAMES = (
	"temperature_mean",
	"temperature_std",
	"temperature_rate_c_per_min",
	"vibration_rms",
	"vibration_peak",
	"vibration_kurtosis",
)


def moving_average(values: np.ndarray, samples: int = 5) -> np.ndarray:
	"""Causal moving average; edge padding prevents artificial start-up dips."""
	values = np.asarray(values, dtype=float)
	if values.size == 0:
		return values.copy()
	padded = np.pad(values, (samples - 1, 0), mode="edge")
	return np.convolve(padded, np.ones(samples) / samples, mode="valid")


def extract_features(
	temperature: np.ndarray,
	vibration: np.ndarray,
	temperature_hz: float = 1.0,
	vibration_hz: float = 0.5,
) -> np.ndarray:
	"""Return the assignment's six-value fused feature vector for one window."""
	temperature = moving_average(np.asarray(temperature, dtype=float))
	vibration = moving_average(np.asarray(vibration, dtype=float))
	if temperature.size < 2 or vibration.size == 0:
		raise ValueError("A feature window needs at least two temperature samples")
	elapsed_minutes = max((temperature.size - 1) / temperature_hz / 60.0, 1e-9)
	rate = (temperature[-1] - temperature[0]) / elapsed_minutes
	mean = float(np.mean(vibration))
	std = float(np.std(vibration))
	kurtosis = float(np.mean(((vibration - mean) / max(std, 1e-9)) ** 4))
	return np.array(
		[
			np.mean(temperature),
			np.std(temperature),
			rate,
			np.sqrt(np.mean(vibration**2)),
			np.max(np.abs(vibration)),
			kurtosis,
		],
		dtype=np.float32,
	)


def window_features(
	temperature: np.ndarray,
	vibration: np.ndarray,
	window_seconds: int = 30,
	step_seconds: int = 10,
) -> np.ndarray:
	"""Extract features from 30-second windows advanced every 10 seconds."""
	temperature = np.asarray(temperature, dtype=float)
	vibration = np.asarray(vibration, dtype=float)
	window_t = window_seconds
	step_t = step_seconds
	window_v = max(1, round(window_seconds * 0.5))
	step_v = max(1, round(step_seconds * 0.5))
	rows = []
	for start_t in range(0, temperature.size - window_t + 1, step_t):
		start_v = round(start_t * 0.5)
		if start_v + window_v <= vibration.size:
			rows.append(extract_features(temperature[start_t:start_t + window_t], vibration[start_v:start_v + window_v]))
	return np.asarray(rows, dtype=np.float32)


def fit_stats(features: np.ndarray, output_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
	"""Fit stats once on clean training data and persist mean/std as an Nx2 array."""
	features = np.asarray(features, dtype=np.float32)
	mean = np.mean(features, axis=0)
	std = np.maximum(np.std(features, axis=0), 1e-6)
	np.save(output_path, np.stack([mean, std]))
	return mean, std


def load_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
	stats = np.load(path)
	if stats.shape != (2, 6):
		raise ValueError(f"Expected training stats shape (2, 6), got {stats.shape}")
	return stats[0], stats[1]


def normalise(features: np.ndarray, stats_path: str | Path) -> np.ndarray:
	mean, std = load_stats(stats_path)
	# Clipping protects the small MLP from a long linear drift saturating ReLU units.
	return np.clip((np.asarray(features, dtype=np.float32) - mean) / std, -10.0, 10.0)
