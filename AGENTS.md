# AGENTS.md — adaptive-p2p-risk

Guidance for any coding agent (Codex, etc.) working in this repository.
Read this file in full before planning or writing any code.

## Progress Checklist

Check items off as each phase is actually implemented and tested in the
repo — not when merely planned. Keep this section up to date; it's the
fastest way to answer "where are we?" without re-reading the whole roadmap.

- [x] **Phase 0 — Repo & Environment Setup**
- [x] **Phase 1 — Data Ingestion (FR-DM-01)**
- [x] **Phase 2 — Data Cleaning (FR-DM-02)**
- [x] **Phase 3 — Pool Construction (FR-DM-03)** ⚠️ sign-off required first
  - [x] Implementation + reproducibility tests
- [x] **Phase 4 — Feature Engineering (FR-DM-04)**
- [ ] **Phase 5 — Temporal Stream Simulation (FR-DM-05)**
- [ ] **Phase 6 — Dataset Versioning (FR-DM-06)**
- [ ] **Phase 7 — Risk Modeling**
  - [ ] FR-RM-01 contract-level claim probability
  - [ ] FR-RM-02 pool-level aggregated score
  - [ ] FR-RM-03 anomaly / fraud-like detection
  - [ ] FR-RM-05 performance/interpretability trade-off documented
  - [ ] FR-RM-06 probability calibration
  - [ ] FR-RM-07 Gini / AUC / calibration error tracked
- [ ] **Phase 8 — Continual Learning & Drift Detection** ⚠️ threshold sign-off required
  - [ ] FR-CL-01 drift detector implemented
  - [ ] FR-CL-02 data drift vs. concept drift distinguished
  - [ ] FR-CL-03 incremental/continual learning strategy
  - [ ] FR-CL-04 sliding-window retraining
  - [ ] FR-CL-05 catastrophic forgetting mitigation
  - [ ] FR-CL-06 automated drift-triggered retraining thresholds
- [ ] **Phase 9 — MLOps Pipeline**
  - [ ] FR-ML-01 MLflow tracking
  - [ ] FR-ML-02 model registry
  - [ ] FR-ML-03 DVC artifact versioning
  - [ ] FR-ML-04 retraining orchestration
  - [ ] FR-ML-05 candidate validation gate
  - [ ] FR-ML-06 Kafka streaming simulation (Could — optional)
- [ ] **Phase 10 — Deployment**
  - [ ] FR-DP-01 REST API endpoints
  - [ ] FR-DP-02 Dockerized
  - [ ] FR-DP-03 Kubernetes (Could — optional)
  - [ ] FR-DP-04 CI/CD
  - [ ] FR-DP-05 Streamlit demo
- [ ] **Phase 11 — Monitoring & Traceability**
  - [ ] FR-MT-01 Grafana dashboards
  - [ ] FR-MT-02 Prometheus metrics
  - [ ] FR-MT-03 alerting
  - [ ] FR-MT-04 full traceability
  - [ ] FR-MT-05 retraining cycle logging

## 1. Project Overview

This repo implements **Adaptive Risk Management System: MLOps and Continual
Learning for Collaborative Insurance** — a 2-month PFA internship project at
Smart Automation Technologies (Tangier), supervised by Dr. Amina Jbilou, with
Pr. A. El Afia (ENSIAS) as academic supervisor.

The system continuously scores insurance risk at two levels — **individual
contract** and **collective pool** — for a collaborative (peer-to-peer)
insurance setting, and keeps itself accurate over time via drift detection
and automated retraining (continual learning).

Full requirements live in the project's CdCT (Cahier des Charges
Techniques) — this file is a working translation of that document into an
implementation roadmap. When in doubt about *what* to build, the CdCT is the
source of truth; this file governs *how* and *in what order*.

## 2. Working Dataset

**freMTPL2** (frequency + severity), French motor third-party liability
insurance. Two files:

- `freMTPL2freq.csv` — 678,013 rows, one per policy. Columns: `IDpol`,
  `ClaimNb`, `Exposure`, `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`,
  `VehBrand`, `VehGas`, `Area`, `Density`, `Region`.
- `freMTPL2sev.csv` — 26,639 rows, one per claim. Columns: `IDpol`,
  `ClaimAmount`. One-to-many relationship with freq via `IDpol`.

Source: https://huggingface.co/datasets/mabilton/fremtpl2 (mirror of the
CASdatasets freMTPL2 data). **Never commit these CSVs to git** — they belong
in `data/raw/`, which is gitignored. The person running this repo places
them there manually.

**Known data-quality issues to handle during cleaning** (see Section 5,
Step 2):
- 195 claims in freMTPL2sev reference a policy not present in freMTPL2freq
  ("unmatched claims").
- 9,117 policies have a declared `ClaimNb` that does not match the number of
  claims actually observed for them in freMTPL2sev ("ClaimNb mismatch").
- `Exposure` ranges up to 2.01; values above 1 are documented recording
  artifacts.

**Collaborative insurance "pools" do not exist in this dataset.** They must
be constructed (clustering on shared risk attributes). This is a modeling
decision with downstream effects on every pool-level requirement — see
Section 6 (Decisions Requiring Sign-Off) before implementing.

## 3. Repository Structure

Target structure — create directories as each phase needs them, don't
scaffold everything upfront:

```
adaptive-p2p-risk/
├── data/
│   ├── raw/              # gitignored: freMTPL2freq.csv, freMTPL2sev.csv
│   └── processed/        # gitignored: pipeline outputs
├── src/
│   ├── data/              # FR-DM-*: ingestion, cleaning, pools, features, streaming
│   ├── models/             # FR-RM-*: contract & pool risk models
│   ├── continual/          # FR-CL-*: drift detection, incremental learning
│   ├── mlops/               # FR-ML-*: MLflow/DVC integration, retraining orchestration
│   ├── api/                  # FR-DP-*: FastAPI app
│   └── monitoring/          # FR-MT-*: Prometheus/Grafana hooks, alerting, logging
├── tests/                  # mirrors src/ structure
├── notebooks/               # exploration only — nothing here is a dependency of src/
├── demo/                    # FR-DP-05: Streamlit app
├── .github/workflows/        # FR-DP-04: CI/CD
├── pyproject.toml
├── uv.lock
├── AGENTS.md
└── README.md
```

## 4. Conventions

- **Python 3.11+, type hints everywhere**, `from __future__ import annotations`
  at the top of modules that need it.
- Every pipeline stage returns a **report object** (dataclass) documenting
  what it did (counts, thresholds applied, anomalies found) — never mutate
  data silently. This is how we satisfy NFR-06 (traceability) at the code
  level, not just at the MLOps-tool level.
- **No model-specific encoding in the cleaning stage.** Cleaning = missing
  values, outlier capping, known-issue resolution, dtype correctness.
  One-hot / target encoding belongs to the modeling stage (FR-RM-*), decided
  per model family.
- **Every module ships with tests that run on synthetic mock data**, not the
  real CSVs. Tests must pass with an empty `data/raw/` — real-data runs are
  a separate, manual verification step, never a CI dependency.
- Docstrings explain *why*, not just *what*, especially for any cleaning or
  thresholding decision — a future reader (including the supervisor) should
  be able to audit a decision without reading the CdCT side-by-side.
- Commit messages reference the FR/NFR ID(s) they implement, e.g.
  `feat(data): implement FR-DM-01 ingestion with schema validation`.

## 5. Implementation Roadmap

Work through phases **in order**. Do not start a phase until the previous
one has passing tests. Each phase = one planning round with the supervising
Claude session before implementation starts (per the person's workflow —
plan first, get it reviewed, then code).

### Phase 0 — Repo & Environment Setup
- Initialize repo structure (Section 3), `.gitignore` (must exclude
  `data/raw/*`, `data/processed/*`, `__pycache__`, `.pytest_cache`,
  `mlruns/`), uv project metadata (`pyproject.toml`, `uv.lock`), and the
  initial test harness.
- **Definition of done:** `uv run pytest` runs clean through the project
  virtual environment. At this repo's current Phase 0 level, a small smoke
  test is enough to prove the harness works; real pipeline tests begin in
  Phase 1.

### Phase 1 — Data Ingestion — FR-DM-01 (Must)
- Load both CSVs, validate schema (expected columns, non-empty, no
  duplicate `IDpol` in freq), raise clear errors otherwise.
- Return an ingestion report (row counts, file paths) for logging.
- **Definition of done:** tests cover valid load, missing file, missing
  column, duplicate `IDpol`.

### Phase 2 — Data Cleaning — FR-DM-02 (Must)
- Resolve the two documented data-quality issues (Section 2) explicitly and
  auditably — drop unmatched claims, reconcile `ClaimNb` against observed
  severity counts, preserve the original declared value for traceability.
- Cap `Exposure` and `ClaimNb` at documented/conventional thresholds.
- Cast categorical columns to proper dtype; leave ML encoding for later.
- **Definition of done:** tests on synthetic mock data covering every
  documented quirk (unmatched claim, `ClaimNb` mismatch, `Exposure` > 1);
  cleaning report is fully populated and asserted in tests.

### Phase 3 — Pool Construction — FR-DM-03 (Must) ⚠️ needs sign-off first
- See Section 6 before planning this phase.
- Cluster contracts into simulated pools on shared risk attributes.
- Pool assignment must be reproducible from a fixed random seed (this is
  the CdCT acceptance criterion for this block).
- **Definition of done:** deterministic pool assignment given a seed;
  tests assert reproducibility and that every contract is assigned to
  exactly one pool.

### Phase 4 — Feature Engineering — FR-DM-04 (Must)
- Contract-level features (policyholder profile, claim history).
- Pool-level aggregated features (average claim rate, pool size, risk
  heterogeneity / within-pool variance).
- **Definition of done:** feature matrix shapes match expected
  contract/pool counts; no leakage of `ClaimNb`/`ClaimAmount` into features
  used as model inputs for the target they predict.

### Phase 5 — Temporal Stream Simulation — FR-DM-05 (Must)
- Split cleaned data into successive batches simulating a live feed.
- Inject controlled drift (data drift and/or concept drift) between
  batches, with **known, logged injection points** — this ground truth is
  required later to evaluate drift detectors (FR-CL-01).
- **Definition of done:** batch boundaries and injected drift points are
  retrievable programmatically (not just visually inspectable), for use as
  ground truth in Phase 8.

### Phase 6 — Dataset Versioning — FR-DM-06 (Should)
- Wire up DVC (or equivalent) for the datasets produced by Phases 1-5.
- **Definition of done:** a dataset snapshot can be pinned and referenced
  from a model training run (needed for NFR-06 traceability later).

### Phase 7 — Risk Modeling — FR-RM-01, 02, 03, 05, 06, 07 (Must/Should)
*(FR-RM-04 "benchmark multiple model families" was removed from scope —
do not implement a multi-family benchmarking harness unless asked.)*
- Contract-level claim probability model (FR-RM-01).
- Pool-level aggregated risk score (FR-RM-02).
- Anomaly/fraud-like detection on contracts or claims (FR-RM-03, Should).
- Performance/interpretability trade-off documented, not just optimized for
  raw metric (FR-RM-05).
- Probability calibration (FR-RM-06, Should).
- Gini, AUC, calibration error tracked for every candidate (FR-RM-07).
- **Definition of done:** contract-level model beats a naive baseline (mean
  claim frequency) on normalized Gini and AUC; pool-level score computed for
  100% of constructed pools — these are the literal CdCT acceptance
  criteria for this block.

### Phase 8 — Continual Learning & Drift Detection — FR-CL-01 to 06 (Must/Should)
- Implement at least one drift detector (ADWIN, DDM, or PSI) (FR-CL-01,
  Must).
- Explicitly distinguish data drift vs. concept drift in detection logic
  (FR-CL-02, Must).
- Incremental/continual learning strategy for model updates (FR-CL-03,
  Must).
- Sliding-window retraining as an alternative to full retraining (FR-CL-04,
  Should).
- Catastrophic forgetting mitigation (FR-CL-05, Should).
- Drift thresholds that trigger retraining automatically (FR-CL-06, Must).
- **Definition of done:** injected drift from Phase 5 is detected within a
  bounded number of batches — exact threshold is `[TBD with supervisor]`
  per the CdCT; do not hardcode an arbitrary number without flagging it.

### Phase 9 — MLOps Pipeline — FR-ML-01 to 06 (Must/Should/Could)
- MLflow tracking + model registry (FR-ML-01, 02, Must).
- DVC artifact versioning (FR-ML-03, Should).
- Automated retraining orchestration, scheduled or drift-triggered
  (FR-ML-04, Must).
- Candidate validation gate before promotion — **no model reaches the
  registry without passing this** (FR-ML-05, Must, also NFR-05).
- Kafka streaming simulation (FR-ML-06, **Could** — lowest priority, skip
  first if time is short).

### Phase 10 — Deployment — FR-DP-01 to 05 (Must/Should/Could)
- FastAPI with `/score/contract`, `/score/pool`, `/model/version`,
  `/health` (FR-DP-01, Must — see CdCT Section 9.1 for the indicative
  contract).
- Dockerize (FR-DP-02, Must).
- Kubernetes orchestration (FR-DP-03, **Could** — lowest priority).
- CI/CD (FR-DP-04, Should).
- Streamlit demo (FR-DP-05, Must).
- **Definition of done:** app starts from a clean Docker build; all
  documented endpoints return valid responses.

### Phase 11 — Monitoring & Traceability — FR-MT-01 to 05 (Should/Must)
- Grafana dashboards (FR-MT-01, Should), Prometheus metrics (FR-MT-02,
  Should).
- Alerting on degradation/drift (FR-MT-03, Must).
- Full traceability of model version → dataset version → risk decision
  (FR-MT-04, Must, also NFR-06).
- Retraining cycle logging (FR-MT-05, Must).

## 6. Decisions Requiring Sign-Off Before Implementation

Do **not** implement these unilaterally — plan them, flag the open
question explicitly, and get confirmation before writing code:

1. **Pool construction methodology** (Phase 3): which features and which
   clustering algorithm define a "pool" is a modeling decision with
   project-wide downstream effects, not a default to pick silently.
2. **Drift detection thresholds** (Phase 8): the CdCT leaves the exact
   "detected within N batches" threshold as `[TBD with supervisor]`.
3. **API latency targets** (NFR-01) and other numeric NFR thresholds not
   already fixed in the CdCT.

## 7. Out of Scope (do not implement)

Per the CdCT: legal/financial structuring of collaborative insurance pools,
surplus redistribution mechanisms between pool members, and the
business/commercial mechanisms of collaborative insurance itself. If a task
seems to require any of these, stop and flag it rather than building around
it.

## 8. Reference Documents

The person has the following deliverables available for context if a plan
needs more detail than this file provides: Project Specifications, List of
Functionalities, Dataset Documentation, Related Work (collaborative
insurance literature), CdCT (Technical Specifications), MLOps Architecture
diagram. Ask the person to paste the relevant section rather than assuming
content not stated here.
