"""Create the structured-pruning/PTQ variant metadata used by benchmarking."""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
	metadata = {"variant": "M3", "method": "35% structured filter pruning followed by full INT8 PTQ", "schedule": "PolynomialDecay", "target_sparsity": 0.35}
	output = ROOT / "training" / "models" / "m3_metadata.json"
	output.write_text(json.dumps(metadata, indent=2) + "\n")
	print(f"wrote {output}")


if __name__ == "__main__":
	main()
