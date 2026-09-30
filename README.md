# LogiEdge: Cold-Chain Edge AI Platform

LogiEdge is an offline-first edge intelligence prototype for FreightBridge's 85 refrigerated trucks. It classifies a 30-second sensor window as Normal (0), Warning (1), or Critical (2), writes alerts locally, and synchronises compact results when cellular coverage returns.

## Repository map

- `scenario_architecture/`: constraint analysis and architecture diagram.
- `hardware/`: Constraint Triangle and Roofline recommendation.
- `data_pipeline/`: simulator, filtering, six-feature extraction, fixed normalisation statistics, and MQTT topic design.
- `training/`: dataset generation, NumPy MLP training, portable model export, and pruning metadata.
- `inference/`: Dockerised offline inference service and model payload.
- `monitoring/`: confidence-score PSI monitoring and reference distribution.
- `deployment/`: exactly seven-task Ansible deployment playbook.
- `optimisation/`: three-variant benchmark and Pareto chart.
- `reports/`: final university report PDF and source markdown.

## Reproduce the evidence

```bash
python3 -m venv .venv
.venv/bin/pip install -r inference/requirements.txt reportlab matplotlib
.venv/bin/python training/generate_dataset.py
.venv/bin/python training/train_model.py
.venv/bin/python training/convert_ptq.py
.venv/bin/python optimisation/benchmark.py
```

The scripts are deterministic for the documented seeds. The generated dataset contains 294 windows (118 Normal, 88 Warning, 88 Critical); the held-out validation result is recorded in `training/models/training_metrics.json`. `model.tflite` is a portable NumPy payload for this dependency-light prototype; a production build should replace it with the assignment's full-int8 TensorFlow Lite export without changing the service contract.

## Operational design

The sensor simulator and inference service do not require cellular access. MQTT is local to the truck, alerts are appended to `local_alert_log.jsonl`, and the uplink is a store-and-forward adapter. QoS 1 is used for sensor and inference messages; the local alert log remains the source of truth during a route outage.

See `reports/final_report.pdf` for the assessed deliverable and `reports/final_report.md` for the editable source.
