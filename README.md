# Adaptive Structural Break Detection in Financial Time Series

**MSc Thesis · Khalifa University · Computational Data Science · 2026**  
Supervisors: Dr. Jorge P. Zubelli (main) · Dr. Ibrahim Elfadel (co-adviser)  
External committee member: Dr. Emanuele Olivetti (ADIA)

---

## Overview

Financial time series exhibit distinct types of structural breaks — volatility regime shifts, mean shifts, changes in autocorrelation structure, tail-shape changes — each of which requires a different detection strategy. Applying a single detector regardless of break type produces either missed breaks or excess false alarms. This project develops a **two-stage adaptive ensemble** that accounts for break-type heterogeneity.

**Stage 1 — Break-Type Classifier**  
Extracts a compact set of statistical meta-features from a window of returns (volatility dynamics, autocorrelation structure, distributional shape, within-window stability) and outputs a soft probability vector p̂ over six classes: mean shift, volatility shift, dependence shift, distributional shift, trend shift, and no-break.

**Stage 2 — Adaptive Detector Ensemble**  
Translates the Stage 1 posterior into per-detector weights via:

```
w = Aᵀ p̂
```

where **A** is an empirically calibrated affinity matrix. Detectors matched to the inferred break type are activated; mismatched detectors are suppressed, reducing false alarms without sacrificing detection power.

This repository contains the **synthetic data generation pipeline** that supports training and benchmarking of both stages. The full detection pipeline is in active development.

---

## Key Results (Proposal Stage — 300 Synthetic Windows)

| Break Type | Specialist Detection Rate | Mismatch Detection Rate | Localisation Error Ratio |
|---|---|---|---|
| Mean shift | 1.00 | ~0.00 | 1× (baseline) |
| Volatility shift | 0.93 | ~0.00 | — |
| Dependence shift | 0.68 | ~0.00 | 2–7× higher for mismatched |
| Distributional shift | 0.81 | ~0.00 | — |
| Trend shift | 0.74 | ~0.00 | — |

Clear diagonal dominance in the affinity matrix confirms that exploitable specialisation exists and that the adaptive weighting scheme is well-motivated.

---

## Repository Structure

```
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
        └── smoke_test_validator_nogarch.py
```

---

## Structural Break Taxonomy

| Type | What changes at τ | Primary detector |
|---|---|---|
| `mean_shift` | Level / intercept | PELT with L2 cost on returns |
| `volatility_shift` | Unconditional variance | PELT with L2 cost on squared returns |
| `dependence_shift` | AR(1) coefficient φ | Dynp with autoregressive cost |
| `distributional_shift` | Student-t degrees of freedom | t-LLR ratio test |
| `trend_shift` | Linear slope | Dynp with clinear cost |
| `no_break` | Nothing | — (false alarm rate calibration) |

---

## Installation

```bash
git clone https://github.com/safurapalekar2643/synthetic_dataset_financial_timeseries.git
cd synthetic_dataset_financial_timeseries
pip install -r requirements.txt   # Python 3.9+
```

---

## Quick Start

**Step 1 — Validate generators (run this first)**

```python
from data_generation.validators.smoke_test_validator import validate_smoke_test
validate_smoke_test(interactive=True)
```

Expected: PASS for all six break types across both GARCH passes. Hard failures block corpus generation.

**Step 2 — Generate single-break corpus**

```python
from data_generation.pure_breaks.generate_corpus import generate_corpus, BREAK_TYPES

manifest = generate_corpus(
    break_types    = BREAK_TYPES,
    grid           = small_grid,     # or DIVERSITY_GRID for full corpus
    n_replicates   = 2,
    output_dir     = "./data/synthetic",
    log_to_mlflow  = False,
    verbose        = True,
)
```

**Step 3 — Generate multi-break corpus (Stage 1 training)**

```python
from data_generation.mixed_breaks.stage1_corpus import generate_stage1_corpus

manifest = generate_stage1_corpus(
    grid          = small_grid,      # or STAGE1_DIVERSITY_GRID for full corpus
    n_replicates  = 5,
    output_dir    = "./data/stage1",
    verbose       = True,
)
```

---

## Corpora

### Single-Break Corpus

Generates windows each containing exactly one injected structural break. Used to calibrate the affinity matrix **A** by measuring each detector's performance on each break type.

Diversity grid sweeps: window length T ∈ {500, 1000, 2000}, break location τ_frac ∈ {0.25, 0.50, 0.75}, magnitude ∈ {small, medium, large}, innovation ∈ {Gaussian, Student-t}, AR(1) background φ ∈ {0.0, 0.3}, GARCH(1,1) ∈ {off, on}, transition ∈ {abrupt, smooth}.

### Multi-Break Corpus

Generates windows with a variable number of breaks per window (Poisson(λ=2)). Each window carries a **soft label vector** — empirical frequencies of break types within that window — connecting Stage 1 output directly to Stage 2 weighting without a conversion step.

Data splits: 60% train / 10% val / 10% test / 20% robustness val (deterministic hash-based, stable across Python versions).

---

## Reproducibility

Seeds are derived deterministically from break type, grid knobs, and replicate index — no global random state is modified. All configs are saved as JSON alongside generated data.

---

## Theoretical Framework

This work draws on:
- **Mixture of Experts (MoE)** — gating network analogy for Stage 1
- **Dynamic Ensemble Selection (DES)** — per-instance detector weighting at inference time
- **Affinity matrix calibration** — empirical measurement of detector-break-type specialisation

Key references: Li et al. (2024, JRSSB) · Katser et al. (2021) · Martins et al. (2025) · Adams & MacKay (2007)

---

## Applications

The framework targets:
- **VaR recalibration** — updating risk models when volatility regime changes
- **Volatility targeting** — detecting shifts in realised volatility regime
- **Portfolio construction** — regime-aware rebalancing triggered by structural changes
- **Strategy backtesting** — isolating in-regime vs. cross-regime performance

---

## Status

This repository is at **proposal stage**. The detector suite is preliminary; final selection follows a literature-driven search in the main thesis phase. The Dynp FAR=1.00 limitation (forced `n_bkps`) is a known constraint; a post-hoc cost-ratio threshold stage is the planned resolution.
