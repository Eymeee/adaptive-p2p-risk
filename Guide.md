# Guide

This guide explains how to use the `adaptive-p2p-risk` repository from a clean checkout through data preparation, modeling, deployment, and monitoring.

The short version: put the raw freMTPL2 CSVs in `data/raw/`, install dependencies with uv, then run the phase modules in order. Tests use synthetic data and do not require the real CSVs.

## Prerequisites

- Python 3.11 or newer. The current local version file uses Python 3.12.
- `uv` for dependency management.
- Docker and Docker Compose for the containerized API/demo/monitoring stack.
- The raw freMTPL2 files:
  - `data/raw/freMTPL2freq.csv`
  - `data/raw/freMTPL2sev.csv`

Install the environment:

```bash
uv sync --all-groups
```

Run tests:

```bash
uv run pytest
```

## Important Repository Rules

- Do not commit `data/raw/`, `data/processed/`, `artifacts/`, or `mlruns/`.
- Run phases in order unless you know exactly which upstream artifacts are already current.
- Each phase writes an audit report. Read these reports when validating results.
- Tests must remain synthetic-data based; they should pass even if `data/raw/` is empty.

## Full Pipeline

Use this sequence after placing the raw CSVs in `data/raw/`:

```bash
uv run python -m src.data.cleaning
uv run python -m src.data.pools
uv run python -m src.data.features
uv run python -m src.data.streaming
uv run python -m src.data.versioning
uv run python -m src.models.risk
uv run python -m src.continual.drift
uv run python -m src.mlops.pipeline --dvc-status-check
uv run python -m src.monitoring.traceability
```

`src.data.ingestion` can also be run by itself to validate raw files, but `src.data.cleaning` already calls ingestion internally and writes both the ingestion and cleaning reports.

## Phase 0 - Repo and Environment Setup

Role: establish the Python project, dependency lockfile, gitignore rules, and test harness.

Useful commands:

```bash
uv sync --all-groups
uv run pytest
```

Key files:

- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `.gitignore`
- `tests/test_environment.py`

## Phase 1 - Data Ingestion

Role: load raw frequency and severity CSVs, validate schemas, reject missing columns, duplicate policy IDs, missing files, and empty inputs.

Command:

```bash
uv run python -m src.data.ingestion
```

Optional paths:

```bash
uv run python -m src.data.ingestion \
  --frequency-path data/raw/freMTPL2freq.csv \
  --severity-path data/raw/freMTPL2sev.csv
```

Default inputs:

- `data/raw/freMTPL2freq.csv`
- `data/raw/freMTPL2sev.csv`

Output: the ingestion module prints an ingestion report to stdout. When run through Phase 2, the report is written to `data/processed/ingestion_report.json`.

## Phase 2 - Data Cleaning

Role: resolve documented freMTPL2 quality issues and create cleaned CSVs.

What it does:

- drops severity rows whose `IDpol` does not exist in frequency;
- reconciles `ClaimNb` against observed severity rows while preserving the declared value;
- caps exposure artifacts above the configured maximum;
- casts categorical fields cleanly;
- writes processed files and reports.

Command:

```bash
uv run python -m src.data.cleaning
```

Optional output directory:

```bash
uv run python -m src.data.cleaning --output-dir data/processed
```

Outputs:

- `data/processed/freMTPL2freq_cleaned.csv`
- `data/processed/freMTPL2sev_cleaned.csv`
- `data/processed/ingestion_report.json`
- `data/processed/cleaning_report.json`

## Phase 3 - Pool Construction

Role: create deterministic synthetic collaborative-insurance pools because freMTPL2 has no native pool structure.

What it does:

- builds clustering features from shared risk attributes;
- scales numeric features;
- one-hot encodes categorical features;
- scans `k=10` to `k=50` with a fixed subsampled silhouette score;
- selects `k` programmatically;
- assigns exactly one `pool_id` per contract.

Command:

```bash
uv run python -m src.data.pools
```

Useful options:

```bash
uv run python -m src.data.pools \
  --k-min 10 \
  --k-max 50 \
  --random-seed 42 \
  --silhouette-sample-size 10000 \
  --silhouette-sample-seed 42
```

Outputs:

- `data/processed/freMTPL2freq_pooled.csv`
- `data/processed/pool_construction_report.json`

Important note: pool construction is a modeling decision. The selected methodology is documented, but final supervisor sign-off is still marked in `AGENTS.md`.

## Phase 4 - Feature Engineering

Role: build contract-level and pool-level feature/target tables for modeling.

What it does:

- creates contract features such as `Density_log1p` and age/bonus/density bands;
- creates contract targets including claim count, claim probability target, frequency, and severity aggregates;
- creates pool features such as pool size, exposure totals, heterogeneity metrics, and member-profile aggregates;
- creates pool targets while guarding against zero-exposure division;
- documents leakage exclusions.

Command:

```bash
uv run python -m src.data.features
```

Outputs:

- `data/processed/contract_features.csv`
- `data/processed/contract_targets.csv`
- `data/processed/pool_features.csv`
- `data/processed/pool_targets.csv`
- `data/processed/feature_engineering_report.json`

Important note: `pool_claim_rate` is treated as a target, not as a model input, to avoid leakage.

## Phase 5 - Temporal Stream Simulation

Role: split contracts into temporal batches and inject known drift signals for later detector evaluation.

What it does:

- creates deterministic stream batches;
- injects abrupt data drift starting at batch 12;
- injects abrupt concept drift starting at batch 15;
- logs the overlap window, batches 15-19;
- writes machine-readable drift ground truth.

Command:

```bash
uv run python -m src.data.streaming
```

Useful options:

```bash
uv run python -m src.data.streaming \
  --n-batches 20 \
  --random-seed 42 \
  --severity-seed 42 \
  --data-drift-start-batch 12 \
  --concept-drift-start-batch 15
```

Outputs:

- `data/processed/stream/batch_000.csv` through `batch_019.csv`
- `data/processed/stream/drift_ground_truth.json`
- `data/processed/stream/stream_simulation_report.json`

## Phase 6 - Dataset Versioning

Role: create a traceable dataset snapshot report from DVC metadata and processed artifacts.

Command:

```bash
uv run python -m src.data.versioning
```

Useful options:

```bash
uv run python -m src.data.versioning --allow-missing
uv run python -m src.data.versioning --processed-dir data/processed
```

Outputs:

- `data/processed/dataset_version_report.json`
- `data/processed.dvc`

Important note: this project versions `data/processed` as one DVC snapshot for simplicity. That means a change in one processed artifact bumps the snapshot for the whole processed dataset.

## Phase 7 - Risk Modeling

Role: train the contract-level risk model, compute pool scores, and generate anomaly scores.

What it does:

- trains a calibrated logistic regression classifier for `target_has_claim`;
- reports AUC, normalized Gini, Brier score, calibration error, and probability-rate delta;
- computes pool risk scores for all constructed pools;
- fits an IsolationForest anomaly detector with numeric transformed inputs;
- writes the model artifact and scoring outputs.

Command:

```bash
uv run python -m src.models.risk
```

Useful options:

```bash
uv run python -m src.models.risk \
  --test-size 0.2 \
  --output-dir artifacts/phase7
```

For exploratory runs where you still want artifacts even if acceptance fails:

```bash
uv run python -m src.models.risk --no-enforce-acceptance
```

Outputs:

- `artifacts/phase7/risk_model.pkl`
- `artifacts/phase7/risk_modeling_report.json`
- `artifacts/phase7/contract_test_predictions.csv`
- `artifacts/phase7/contract_predictions.csv`
- `artifacts/phase7/pool_risk_scores.csv`
- `artifacts/phase7/contract_anomaly_scores.csv`

## Phase 8 - Continual Learning and Drift Detection

Role: replay the synthetic stream, detect drift, and emit retraining events.

What it does:

- uses PSI for feature/data drift;
- uses residual shift checks for concept drift;
- reuses Phase 7 fitted preprocessing instead of refitting online;
- maintains an online SGD candidate strategy;
- uses a sliding window plus reference replay to reduce forgetting;
- logs retraining events when thresholds fire.

Command:

```bash
uv run python -m src.continual.drift
```

Useful options:

```bash
uv run python -m src.continual.drift \
  --stream-dir data/processed/stream \
  --provisional-latency-batches 2
```

For exploratory runs:

```bash
uv run python -m src.continual.drift --no-enforce-acceptance
```

Outputs:

- `artifacts/phase8/continual_learning_report.json`
- `artifacts/phase8/drift_metrics.csv`
- `artifacts/phase8/retraining_events.csv`

Important note: the 2-batch latency threshold is provisional, chosen on Codex's technical recommendation, and pending supervisor confirmation.

## Phase 9 - MLOps Pipeline

Role: track the model lifecycle with MLflow, validate retrained candidates, and attempt local registry promotion only after passing the validation gate.

What it does:

- reads Phase 7, Phase 8, and Phase 6 reports;
- retrains a candidate when Phase 8 emits retraining events;
- evaluates the candidate on comparable held-out data;
- enforces AUC and calibration validation gates;
- logs MLflow tags including the DVC dataset hash;
- registers the model locally only if validation passes.

Command:

```bash
uv run python -m src.mlops.pipeline --dvc-status-check
```

Useful options:

```bash
uv run python -m src.mlops.pipeline \
  --mlflow-tracking-uri sqlite:///mlruns/mlflow.db \
  --force-retraining-check
```

Outputs:

- `artifacts/phase9/mlops_pipeline_report.json`
- `artifacts/phase9/candidate_validation_report.json`
- `artifacts/phase9/retrained_candidate/risk_model.pkl`
- `artifacts/phase9/retrained_candidate/candidate_contract_test_predictions.csv`
- `artifacts/phase9/retrained_candidate/candidate_pool_risk_scores.csv`
- `mlruns/`

Important note: Kafka simulation is optional and intentionally deferred.

## Phase 10 - Deployment

Role: serve the validated risk model through FastAPI, package it with Docker, and expose a Streamlit demo.

Run the API locally:

```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Run the demo locally:

```bash
API_BASE_URL=http://localhost:8000 uv run streamlit run demo/app.py
```

Run the full Compose stack:

```bash
docker compose up --build
```

API endpoints:

- `GET /health`
- `GET /model/version`
- `GET /metrics`
- `POST /score/contract`
- `POST /score/pool`

Model resolution order:

1. `MODEL_ARTIFACT_PATH` environment variable;
2. validated Phase 9 candidate;
3. accepted Phase 7 reference model.

Outputs:

- `artifacts/phase10/deployment_report.json`

Important notes:

- `/score/pool` supports member-list requests, not pool-id-only lookup.
- API logs operational metadata only, not raw request/response payloads.
- Kubernetes is optional and intentionally deferred.

## Phase 11 - Monitoring and Traceability

Role: expose runtime metrics, configure local dashboards and alerts, and close the traceability chain from risk decision back to model and dataset.

Run traceability report generation:

```bash
uv run python -m src.monitoring.traceability
```

Monitoring stack:

```bash
docker compose up --build
```

Services:

- FastAPI: `http://localhost:8000`
- Streamlit: `http://localhost:8501`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` with `admin` / `admin`

Outputs:

- `artifacts/phase11/risk_decisions.jsonl` when trace logging is enabled and scoring requests are made;
- `artifacts/phase11/monitoring_report.json`;
- `artifacts/phase11/retraining_cycle_log.csv`.

Trace environment variables:

```bash
TRACE_HASH_SALT='replace-with-a-long-secret' uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

For local demo only:

```bash
TRACE_DEV_MODE=true uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

If neither `TRACE_HASH_SALT` nor `TRACE_DEV_MODE=true` is set, decision trace logging is disabled. The code does not use a fixed fallback salt.

## Docker Commands

Build the image:

```bash
docker build -t adaptive-p2p-risk-api .
```

Start all services:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up --build -d
```

Stop services:

```bash
docker compose down
```

View logs:

```bash
docker compose logs -f api
docker compose logs -f demo
```

## Common Checks

Run all tests:

```bash
uv run pytest
```

Run a focused test file:

```bash
uv run pytest tests/models/test_risk.py
uv run pytest tests/api/test_app.py
```

Check CLI help:

```bash
uv run python -m src.data.pools --help
uv run python -m src.models.risk --help
uv run python -m src.mlops.pipeline --help
```

Check generated reports quickly:

```bash
python -m json.tool data/processed/dataset_version_report.json
python -m json.tool artifacts/phase7/risk_modeling_report.json
python -m json.tool artifacts/phase11/monitoring_report.json
```

## Expected Artifact Map

| Location | Meaning |
|---|---|
| `data/processed/*.csv` | cleaned, pooled, and feature-engineered data |
| `data/processed/*.json` | data-stage audit reports |
| `data/processed/stream/` | temporal batch simulation and drift ground truth |
| `artifacts/phase7/` | trained reference model and model reports |
| `artifacts/phase8/` | drift metrics and retraining events |
| `artifacts/phase9/` | MLflow validation and retraining candidate artifacts |
| `artifacts/phase10/` | deployment report |
| `artifacts/phase11/` | monitoring report, retraining cycle log, decision traces |
| `mlruns/` | local MLflow tracking database and run artifacts |

## Troubleshooting

If raw data is missing, Phase 1 or Phase 2 will fail. Place the two freMTPL2 CSVs under `data/raw/`.

If Docker returns a permission error for `/var/run/docker.sock`, your user does not currently have Docker daemon access. Use your system's Docker group setup, or run the Docker command with `sudo` for local testing.

If `docker build` says it requires one argument, include the build context:

```bash
docker build -t adaptive-p2p-risk-api .
```

If the API starts as `degraded`, check:

- `artifacts/phase7/risk_model.pkl`
- `artifacts/phase7/risk_modeling_report.json`
- `artifacts/phase9/candidate_validation_report.json`
- `artifacts/phase10/deployment_report.json`

If decision traces are not written, set either `TRACE_HASH_SALT` for reproducible tracing or `TRACE_DEV_MODE=true` for a local non-reproducible demo salt.
