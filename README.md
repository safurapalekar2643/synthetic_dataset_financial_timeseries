# Adaptive Break-Type-Aware Ensemble Learning for Structural Break Detection in Financial Time Series

**MSc Computational Data Science — Thesis Repository (Proposal Stage)**
Khalifa University · Data Generation Modules

---

## Overview

This repository contains the synthetic data generation pipeline for a two-stage framework that detects structural breaks in financial time series. The core idea is that different structural breaks — mean shifts, volatility changes, distributional changes, etc. — require different statistical detectors. Rather than applying all detectors equally, the framework first *classifies* what type of break is likely present, then routes detection effort accordingly.

**Stage 1** learns to infer the dominant break type from time-series meta-features, producing a soft probability vector p̂ over break types.

**Stage 2** converts p̂ into per-detector weights via an affinity matrix (w = Aᵀp̂), then adjusts each detector's penalty schedule so that well-matched detectors become more aggressive and poorly-matched detectors are effectively gated off.

This repo covers **the data generation layer only** — the two corpora that support training and benchmarking of both stages. The full detection pipeline is in active development and is not yet included.

---

## Repository Structure

​```
synthetic_dataset_financial_timeseries/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── data_generation/
    ├── __init__.py
    │
    ├── pure_breaks/                        # Single-break corpus (Stage 2 benchmarking)
    │   ├── __init__.py
    │   └── generate_corpus.py
    │
    ├── mixed_breaks/                       # Multi-break corpus (Stage 1 training)
    │   ├── __init__.py
    │   └── stage1_corpus.py
    │
    └── validators/                         # Automated label-purity validation
        ├── __init__.py
        ├── smoke_test_validator.py         # GARCH-aware (two-pass: off + on)
        └── smoke_test_validator_nogarch.py # GARCH background disabled
​```

## Corpora

### Single-Break Corpus — `generate_corpus.py`

Generates windows each containing **exactly one injected structural break**. Break type is one of five active types; break location is sampled uniformly over admissible positions subject to a minimum segment constraint.

**Purpose:** calibrating the affinity matrix (how well each detector detects each break type) and benchmarking per-detector specialisation in Chapter 4 of the proposal.

**Output per instance:**
- Time series of returns (`t`, `returns`, `regime` columns in CSV)
- Break type and true location as hard labels
- Effect size (normalised, break-type-specific)
- JSON config for full reproducibility

**Diversity grid** (Cartesian product swept):

| Knob | Values |
|---|---|
| Window length T | 500, 1000, 2000 |
| Break location τ_frac | 0.25, 0.50, 0.75 |
| Location jitter | 0.0, ±5% |
| Magnitude | small, medium, large |
| Baseline σ | 0.005, 0.01, 0.02 |
| Innovation distribution | Gaussian, Student-t |
| AR(1) background φ | 0.0, 0.3 |
| GARCH(1,1) background | off, on |
| Transition sharpness | abrupt, smooth |

---

### Multi-Break Corpus — `stage1_corpus.py`

Generates windows with **a variable number of breaks** per window, drawn from Poisson(λ=2). Break types and locations are sampled independently. Zero-break windows are retained as the no-break class.

**Purpose:** training and evaluating the Stage 1 break-type classifier.

**Labelling:** each window carries a **soft probability vector** over six classes (five break types + no-break), with entries proportional to the empirical frequencies of break types in that window. This allows Stage 1 to learn a distribution over break types rather than a hard single-class label, and connects the classifier output directly to Stage 2 weighting without a conversion step.

**Data splits** (deterministic, stratified by dominant break type and break-count bucket):

| Split | Fraction | Purpose |
|---|---|---|
| train | 60% | Stage 1 classifier training |
| val | 10% | Hyperparameter selection |
| test | 10% | Final held-out evaluation |
| robustness_val | 20% | Stress-test on complex multi-break windows |

---

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Quick Start

### Single-break corpus (small grid for validation)

```python
from data_generation.generate_corpus import generate_corpus, BREAK_TYPES

small_grid = {
    "T":                [500, 1000],
    "tau_frac":         [0.25, 0.5, 0.75],
    "tau_jitter":       [0.0, 0.05],
    "magnitude":        ["small", "medium", "large"],
    "baseline_sigma":   [0.01],
    "sigma_jitter":     [0.0],
    "innovation":       ["gaussian", "student_t"],
    "ar_background":    [0.0, 0.3],
    "garch_background": [False, True],
    "smooth_transition":[False, True],
}

manifest = generate_corpus(
    break_types    = BREAK_TYPES,
    grid           = small_grid,
    n_replicates   = 2,
    output_dir     = "./data/synthetic",
    log_to_mlflow  = False,
    verbose        = True,
)
```

Swap in `DIVERSITY_GRID` from `generate_corpus.py` for the full corpus run.

---

### Multi-break corpus (small grid for validation)

```python
from data_generation.stage1_corpus import generate_stage1_corpus, get_train, get_val, get_test

small_grid = {
    "T":                [500, 1000],
    "baseline_sigma":   [0.01],
    "innovation":       ["gaussian", "student_t"],
    "ar_background":    [0.0, 0.3],
    "garch_background": [False],
    "smooth_transition":[False],
    "magnitude":        ["medium", "random"],
}

manifest = generate_stage1_corpus(
    grid          = small_grid,
    n_replicates  = 5,
    output_dir    = "./data/stage1",
    verbose       = True,
)

print(f"Train : {len(get_train(manifest))}")
print(f"Val   : {len(get_val(manifest))}")
print(f"Test  : {len(get_test(manifest))}")
```

---

### Smoke test (label purity validation)

Run this before any corpus generation to verify the generators are producing the correct type of break:

```python
from data_generation.smoke_test_validator import validate_smoke_test

validate_smoke_test(interactive=True)
```

The validator runs two passes (GARCH off, GARCH on) and applies **hard** and **soft** checks per break type:
- **Hard failures** block corpus generation (e.g., wrong statistic changed — label contamination).
- **Soft warnings** are logged and must be documented; generation proceeds.

---

## Structural Break Types

| Type | What changes at τ | Primary detector family |
|---|---|---|
| `mean_shift` | Level / intercept | PELT with L2 cost on returns |
| `volatility_shift` | Unconditional variance | PELT with L2 cost on squared returns |
| `dependence_shift` | AR(1) coefficient φ | Dynp with autoregressive cost |
| `distributional_shift` | Student-t degrees of freedom | t-LLR ratio test |
| `trend_shift` | Linear slope | Dynp with clinear cost |
| `no_break` | Nothing | — (false alarm rate calibration) |

> **Note:** The detector suite listed above is **preliminary** — five placeholder detectors chosen to demonstrate the break-type specialisation pattern at proposal stage. Final detector selection will follow a literature-driven search in the first month of the thesis and may differ substantially.

---

## Known Limitations (Proposal Stage)

**FAR = 1.00 for Dynp detectors.** The `Dynp` algorithm requires a fixed `n_bkps` argument, which forces it to return exactly one breakpoint regardless of whether a true break is present. This means false alarm rate (FAR) is 1.0 for `dependence_shift` and `trend_shift` detectors on no-break windows. A post-hoc cost-ratio threshold stage is the planned resolution in the main thesis phase.

**Synthetic-only evaluation.** All evaluation at proposal stage uses Tier A synthetic data with exact ground-truth labels. Tier B semi-synthetic evaluation (controlled breaks injected into real return windows) is planned for the main thesis phase.

**Detector suite is preliminary.** See the note above. The five current detectors are illustrative examples for the proposal-stage benchmark only.

---

## Reproducibility

Every generated instance is fully reproducible from its config alone. Seeds are derived deterministically from the combination of break type, grid knobs, and replicate index — no global random state is modified. Data splits use a hash-based deterministic assignment, not a random shuffle, so splits are stable across Python versions.

---

## MLflow Tracking (Optional)

`generate_corpus.py` supports optional per-instance MLflow logging. Set `log_to_mlflow=True` in `generate_corpus()` and point `MLFLOW_TRACKING_URI` to your tracking server. When disabled (`log_to_mlflow=False`), corpus generation runs without any MLflow dependency.

---

## Citation / Reference

This codebase accompanies the MSc thesis proposal:

> *Adaptive Break-Type-Aware Ensemble Learning for Structural Break Detection in Financial Time Series*
> Khalifa University, MSc Computational Data Science, 2025.

Key references for the methodology:
- Li et al. (2024, JRSSB) — closest competitor framework
- Katser et al. (2021) — canonical CPD ensemble baseline
- Martins et al. (2025) — meta-learning for change-point detection
- Romano et al. (2021) — DeCAFS detector
- Londschien et al. (2023) — changeforest detector
- Inclán & Tiao (1994) — ICSS variance break detector
