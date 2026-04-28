"""
=============================================================================
SYNTHETIC BREAK-TYPE SIMULATION GENERATORS — TIER A  (MERGED)
=============================================================================

Merged from:
  - synthetic_generators_claude.py   (base)
  - synthetic_break_generators_cursor.py (adopted components)

Adopted from Cursor
  - trend_shift generator          (ported to BreakConfig + MAGNITUDE_TABLE)
  - sigma_jitter baseline noise    (optional randomisation without losing reproducibility)
  - (meta-features are computed in `meta_features.py`)

New in this merge (both gaps resolved)
  - generate_no_break()            (required for false alarm rate estimation)
  - GARCH(1,1) background          (realistic volatility clustering condition)

Design decisions applied
  1. Single file — everything in one .py for immediate use
  2. Breakpoint location — tau_frac sets centre; tau_jitter adds optional uniform noise
  3. Both critical gaps — no_break + GARCH implemented (not stubs)
  4. Magnitude — current units kept; computed effect_size added to every instance output
  5. Meta-features — full Stage-1 set: rolling stats, Ljung-Box, kurtosis,
                     ARCH proxy, tail index, bimodality indicator

Architecture (logical sections)
  §1   BreakConfig dataclass          — all knobs
  §2   MAGNITUDE_TABLE                — scale-invariant break sizes per type
  §3   DIVERSITY_GRID                 — Cartesian sweep for corpus generation
  §4   sample_innovations()           — Gaussian / Student-t, variance-normalised
  §5   transition_weights()           — abrupt step or sigmoid blend
  §6   garch_volatility()             — GARCH(1,1) background sigma path  [NEW]
  §7   compute_effect_size()          — normalised effect size per break type [NEW]
  §8   Generator 1 — mean_shift
  §9   Generator 2 — volatility_shift
  §10  Generator 3 — dependence_shift
  §11  Generator 4 — distributional_shift
  §12  Generator 5 — trend_shift      [FROM CURSOR]
  §13  Generator 6 — no_break         [NEW]
  §14  generate_instance()            — dispatcher + effect_size annotation
  §15  (moved) meta-feature extraction — see `meta_features.py`
  §16  log_instance_to_mlflow()       — per-instance MLflow logging
  §17  generate_corpus()              — diversity-grid batch generator
  §18  demo_all_generators()          — smoke test
=============================================================================
"""

import numpy as np
import pandas as pd
import mlflow
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Literal, Optional
from scipy import stats
from scipy.signal import periodogram
from smoke_test_validator import validate_smoke_test


# =============================================================================
# §1 — CONFIGURATION DATACLASS
# =============================================================================
# Every diversity dimension from the design matrix maps to one field here.
# Storing it as a dataclass lets us:
#   (a) log it directly to MLflow as params
#   (b) serialise it to JSON alongside each saved window
#   (c) reconstruct any run exactly from its config alone

@dataclass
class BreakConfig:
    """
    Knob panel for one window instance.
    Maps directly to the diversity grid in Structure_1.md.
    """

    # --- Identity ---
    break_type: Literal[
        "mean_shift", "volatility_shift", "dependence_shift",
        "distributional_shift", "trend_shift", "no_break"
    ] = "mean_shift"
    seed: int = 42

    # --- Window shape ---
    T: int = 1000
    tau_frac: float = 0.5           # centre of break as fraction of T
                                    # [0.25=early, 0.5=mid, 0.75=late]
    tau_jitter: float = 0.0         # DECISION 2: uniform noise on tau_frac
                                    # actual tau drawn from Uniform(tau_frac±tau_jitter)
                                    # 0.0 = pure Claude fixed-location behaviour
                                    # 0.1 = ±10% of T variation around centre

    # --- Baseline noise regime ---
    baseline_sigma: float = 0.01    # calm=0.005, normal=0.01, turbulent=0.02
    baseline_mu: float = 0.0        # kept ~0 for non-mean breaks
    sigma_jitter: float = 0.0       # ADOPTED FROM CURSOR: adds N(0, sigma_jitter²)
                                    # noise to baseline_sigma at generation time
                                    # 0.0 = pure Claude fixed-baseline behaviour

    # --- Innovation distribution ---
    innovation: Literal["gaussian", "student_t"] = "gaussian"
    t_df: float = 5.0               # degrees of freedom when innovation="student_t"

    # --- Autocorrelation background ---
    ar_background: float = 0.0      # AR(1) phi before break; 0=iid, 0.3=mild

    # --- GARCH background [NEW] ---
    garch_background: bool = False  # if True, replace constant sigma with GARCH(1,1) path
    garch_omega: float = 1e-6       # GARCH unconditional variance component
    garch_alpha: float = 0.10       # ARCH coefficient
    garch_beta: float = 0.85        # GARCH coefficient
                                    # stationarity requires alpha + beta < 1

    # --- Break magnitude ---
    magnitude: Literal["small", "medium", "large"] = "medium"

    # --- Break sharpness ---
    smooth_transition: bool = False
    transition_width: int = 20      # sigmoid width in time steps


# =============================================================================
# §2 — MAGNITUDE LOOKUP TABLE
# =============================================================================
# Each break type has its own "what does small/medium/large mean" table.
# Units are always relative to baseline_sigma to stay scale-invariant.
# effect_size is computed separately in §7 for cross-type comparability.

MAGNITUDE_TABLE = {
    "mean_shift": {
        # δ as multiples of baseline_sigma  →  effect size = δ/σ
        "small":  0.5,
        "medium": 1.5,
        "large":  3.0,
    },
    "volatility_shift": {
        # Ratio σ₂/σ₁  →  effect size = log(σ₂/σ₁)
        "small":  1.5,
        "medium": 2.5,
        "large":  4.0,
    },
    "dependence_shift": {
        # Δφ  →  effect size = Δφ / (1 − φ₁²)
        "small":  0.2,
        "medium": 0.4,
        "large":  0.6,
    },
    "distributional_shift": {
        # (df_pre, df_post) — kept in natural df units per DECISION 4
        # effect_size = approx KL divergence t(df_pre) || t(df_post)
        "small":  {"df_pre": 30, "df_post": 10},
        "medium": {"df_pre": 30, "df_post":  5},
        "large":  {"df_pre": 30, "df_post":  3},
    },
    "trend_shift": {
        # Δslope expressed as SNR target: actual slope = SNR * sigma / sqrt(T)
        # is computed at generation time using cfg.baseline_sigma and cfg.T.
        # Stored here as SNR targets so magnitudes are scale-invariant.
        # At SNR=2, cumulative post-break drift over T/2 steps ≈ sigma * sqrt(T)/2
        # — clearly detectable but not exploding.
        # effect_size = SNR = |Δslope| * sqrt(T) / σ
        "small":  1.0,
        "medium": 2.0,
        "large":  4.0,
    },
    "no_break": {
        # No break — magnitude unused; effect_size = 0 by definition
        "small":  0.0,
        "medium": 0.0,
        "large":  0.0,
    },
}


# =============================================================================
# §3 — DIVERSITY GRID
# =============================================================================
# Cartesian product of these knobs defines the design matrix.
# generate_corpus() sweeps all combinations × n_replicates.

DIVERSITY_GRID = {
    "T":                  [500, 1000, 2000],
    "tau_frac":           [0.25, 0.5, 0.75],       # early / mid / late
    "tau_jitter":         [0.0, 0.05],              # fixed / ±5% jitter
    "magnitude":          ["small", "medium", "large"],
    "baseline_sigma":     [0.005, 0.01, 0.02],      # calm / normal / turbulent
    "sigma_jitter":       [0.0, 0.002],             # fixed / jittered baseline
    "innovation":         ["gaussian", "student_t"],
    "ar_background":      [0.0, 0.3],               # iid / mild autocorr
    "garch_background":   [False, True],            # no clustering / GARCH
    "smooth_transition":  [False, True],            # abrupt / gradual
}

BREAK_TYPES = [
    "mean_shift",
    "volatility_shift",
    "dependence_shift",
    "distributional_shift",
    "trend_shift",
    "no_break",
]


# =============================================================================
# §4 — INNOVATION SAMPLER
# =============================================================================
# Shared by all generators.
# t innovations are rescaled so Var = sigma² regardless of df,
# preserving comparability of sigma across configs.

def sample_innovations(T: int, sigma: float, innovation: str,
                       t_df: float, rng: np.random.Generator) -> np.ndarray:
    """
    Draw T innovations with std ≈ sigma.

    Gaussian : N(0, sigma²)
    Student-t: t(df) rescaled so Var = sigma² for any df > 2.
               Scale factor = sqrt(df/(df-2)).
    """
    if innovation == "gaussian":
        return rng.normal(0, sigma, size=T)
    elif innovation == "student_t":
        raw = rng.standard_t(df=t_df, size=T)
        scale = np.sqrt(t_df / (t_df - 2)) if t_df > 2 else 1.0 #makes it unit variance
        return (raw / scale) * sigma #rescale heavy t distributions to have unit variance then resize to have target sigma
    else:
        raise ValueError(f"Unknown innovation type: {innovation}")


# =============================================================================
# §5 — SMOOTH TRANSITION WEIGHT
# =============================================================================
# w[t] = 0 → fully pre-break regime
# w[t] = 1 → fully post-break regime
# Abrupt: hard step at tau.
# Smooth: sigmoid centred at tau; steepness k = 10/width.

def transition_weights(T: int, tau: int, smooth: bool,
                       width: int) -> np.ndarray:
    """
    Returns weight vector w[0:T] in [0, 1].
    """
    if not smooth:
        w = np.zeros(T)
        w[tau:] = 1.0
        return w
    else:
        t_idx = np.arange(T)
        k = 10.0 / max(width, 1) #steepness of the sigmoid, defines how sharply the weight changes around tau
        #as width increases, k decerases, smoothes the transition
        w = 1.0 / (1.0 + np.exp(-k * (t_idx - tau)))
        return w


# =============================================================================
# §6 — GARCH(1,1) VOLATILITY PATH  [NEW]
# =============================================================================
# Produces a time-varying sigma path h_t that follows GARCH(1,1) dynamics:
#   h_t = omega + alpha * eps_{t-1}² + beta * h_{t-1}
#
# Used as a background condition: when garch_background=True, generators
# replace their constant sigma_vec with sigma_vec * sqrt(h_t / h_bar),
# where h_bar = unconditional variance = omega / (1 - alpha - beta).
# This multiplicative rescaling preserves the intended break magnitude
# while adding realistic volatility clustering on top.
#
# Stationarity requires alpha + beta < 1 (enforced with a clip).

def garch_volatility_path(T: int, omega: float, alpha: float, beta: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a GARCH(1,1) conditional variance path h[0:T].

    Returns
    -------
    h : np.ndarray, shape (T,)
        Conditional variance at each time step.
        Multiply series innovations by sqrt(h / h_bar) to apply clustering.
    """
    # Clip to ensure stationarity
    alpha = np.clip(alpha, 0.0, 0.49)
    beta  = np.clip(beta,  0.0, 0.49)
    if alpha + beta >= 1.0:
        beta = 0.99 - alpha

    h_bar = omega / max(1.0 - alpha - beta, 1e-8)   # unconditional variance
    h = np.zeros(T)
    h[0] = h_bar

    # Burn-in innovations for h initialisation to ensure the series is stationary before the break
    eps_init = rng.normal(0, np.sqrt(h_bar))

    eps_prev = eps_init
    for t in range(1, T):
        h[t] = omega + alpha * eps_prev**2 + beta * h[t - 1]
        eps_prev = rng.normal(0, np.sqrt(h[t]))

    return h, h_bar #returns the volatility path and the unconditional variance


def apply_garch_scaling(sigma_vec: np.ndarray, cfg: BreakConfig,
                        rng: np.random.Generator) -> np.ndarray:
    """
    If cfg.garch_background is True, multiply sigma_vec by a GARCH(1,1)
    scaling factor sqrt(h_t / h_bar), preserving break magnitude in expectation.
    Returns sigma_vec unchanged if garch_background is False.
    """
    if not cfg.garch_background:
        return sigma_vec
    h, h_bar = garch_volatility_path(
        len(sigma_vec), cfg.garch_omega, cfg.garch_alpha, cfg.garch_beta, rng
    )
    garch_scale = np.sqrt(h / max(h_bar, 1e-12))
    return sigma_vec * garch_scale


# =============================================================================
# §7 — EFFECT SIZE COMPUTATION  [DECISION 4]
# =============================================================================
# Computes a normalised, interpretable effect size for every break type.
# Stored as a field in every instance dict for downstream use in benchmarking.
# Units differ by type but are each meaningful within-type:
#
#   mean_shift         : Cohen's d  = δ / σ
#   volatility_shift   : log ratio  = log(σ₂/σ₁)
#   dependence_shift   : normalised = Δφ / (1 − φ₁²)
#   distributional_shift: approx KL = KL[t(df_post) || t(df_pre)] (numerical)
#   trend_shift        : SNR        = |Δslope| × √T / σ
#   no_break           : 0.0        (by definition)

def _kl_t_approx(df1: float, df2: float) -> float:
    """
    Numerical approximation of KL(t(df1) || t(df2)) using a fine grid.
    Both distributions have zero mean and are rescaled to unit variance.
    """
    x = np.linspace(-20, 20, 4000)
    s1 = np.sqrt(df1 / (df1 - 2)) if df1 > 2 else 1.0
    s2 = np.sqrt(df2 / (df2 - 2)) if df2 > 2 else 1.0
    p = stats.t.pdf(x, df=df1, scale=1/s1) + 1e-300
    q = stats.t.pdf(x, df=df2, scale=1/s2) + 1e-300
    dx = x[1] - x[0]
    return float(np.sum(p * np.log(p / q) * dx))


def compute_effect_size(cfg: BreakConfig) -> float:
    """
    Return a normalised effect size for the break in cfg.
    """
    bt  = cfg.break_type
    mag = cfg.magnitude
    s   = cfg.baseline_sigma

    if bt == "mean_shift":
        delta = MAGNITUDE_TABLE["mean_shift"][mag] * s
        return delta / s   # Cohen's d

    elif bt == "volatility_shift":
        ratio = MAGNITUDE_TABLE["volatility_shift"][mag]
        return float(np.log(ratio))   # log ratio

    elif bt == "dependence_shift":
        phi1 = cfg.ar_background
        dphi = MAGNITUDE_TABLE["dependence_shift"][mag]
        denom = max(1 - phi1**2, 1e-4)
        return dphi / denom

    elif bt == "distributional_shift":
        params   = MAGNITUDE_TABLE["distributional_shift"][mag]
        df_pre   = params["df_pre"]
        df_post  = params["df_post"]
        return _kl_t_approx(df_post, df_pre)   # KL from post to pre

    elif bt == "trend_shift":
        # MAGNITUDE_TABLE already stores the SNR target directly
        return float(MAGNITUDE_TABLE["trend_shift"][mag])

    elif bt == "no_break":
        return 0.0

    return float("nan")


# =============================================================================
# §8 — GENERATOR 1: MEAN SHIFT
# =============================================================================
# r_t = mu_1 + eps_t  for t < tau
# r_t = mu_2 + eps_t  for t >= tau
#
# Variance and AR(phi) are held constant — break is purely in level.
# AR background applied to demeaned residuals so phi does not interact
# with the mean shift.
#
# DECISION 2: tau is drawn from Uniform(tau_frac ± tau_jitter) when
# tau_jitter > 0, otherwise fixed at tau_frac.

def _resolve_tau(cfg: BreakConfig, rng: np.random.Generator) -> int:
    """
    Resolve the actual breakpoint index from tau_frac and tau_jitter.
    Clips to [5%, 95%] of T to avoid degenerate windows. avod breaks at start/end
    """
    if cfg.tau_jitter > 0:
        frac = rng.uniform(
            max(cfg.tau_frac - cfg.tau_jitter, 0.05),
            min(cfg.tau_frac + cfg.tau_jitter, 0.95)
        )
    else:
        frac = cfg.tau_frac
    return int(cfg.T * frac)


def _resolve_sigma(cfg: BreakConfig, rng: np.random.Generator) -> float:
    """
    Resolve effective baseline_sigma with optional jitter.
    ADOPTED FROM CURSOR: sigma_jitter adds N(0, sigma_jitter²) noise.
    Clips to a minimum of 1e-5 to avoid degenerate series.
    """
    if cfg.sigma_jitter > 0:
        noise = rng.normal(0, cfg.sigma_jitter)
        return max(cfg.baseline_sigma + noise, 1e-5)
    return cfg.baseline_sigma


def generate_mean_shift(cfg: BreakConfig) -> tuple[np.ndarray, int]:
    #initialise the random number generator and the breakpoint and sigma
    rng = np.random.default_rng(cfg.seed)
    tau = _resolve_tau(cfg, rng)
    sigma = _resolve_sigma(cfg, rng)

    #compute the magnitude of the break
    delta = MAGNITUDE_TABLE["mean_shift"][cfg.magnitude] * sigma
    mu1 = cfg.baseline_mu
    mu2 = cfg.baseline_mu + delta

    #sample the innovations
    eps = sample_innovations(cfg.T, sigma, cfg.innovation, cfg.t_df, rng)

    #compute the transition weights
    w = transition_weights(cfg.T, tau, cfg.smooth_transition, cfg.transition_width)
    mean_vec = mu1 * (1 - w) + mu2 * w

    # GARCH: build a sigma path and rescale eps
    sigma_vec = np.full(cfg.T, sigma) #create a sigma path of the same length as the series and set it to the initial sigma
    sigma_vec = apply_garch_scaling(sigma_vec, cfg, rng) #apply the garch scaling to the sigma path
    eps = eps * (sigma_vec / max(sigma, 1e-10))  # rescale pre-drawn eps

    series = np.zeros(cfg.T)
    series[0] = mean_vec[0] + eps[0]
    phi = cfg.ar_background #apply the ar background to the series
    for t in range(1, cfg.T):
        series[t] = mean_vec[t] + phi * (series[t - 1] - mean_vec[t - 1]) + eps[t]

    return series, tau #returns the series and the breakpoint


# =============================================================================
# §9 — GENERATOR 2: VOLATILITY SHIFT
# =============================================================================
# r_t = sigma_1 * z_t  for t < tau
# r_t = sigma_2 * z_t  for t >= tau
#
# z_t is a standardised (unit-variance) innovation, possibly AR(1).
# AR is applied on the standardised series so autocorr stays constant
# even as volatility changes — preserving break-type label purity.

def generate_volatility_shift(cfg: BreakConfig) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(cfg.seed)
    tau = _resolve_tau(cfg, rng)
    sigma = _resolve_sigma(cfg, rng)

    #compute the magnitude of the break
    ratio  = MAGNITUDE_TABLE["volatility_shift"][cfg.magnitude]
    sigma1 = sigma
    sigma2 = sigma * ratio


    w = transition_weights(cfg.T, tau, cfg.smooth_transition, cfg.transition_width)
    sigma_vec = sigma1 * (1 - w) + sigma2 * w

    # GARCH applied on top of the break-driven sigma path
    #sigma_vec encodes volatility change over time
    sigma_vec = apply_garch_scaling(sigma_vec, cfg, rng)

    # Standardised unit-variance innovations
    #z encodes gaussian vs t and ar dependance 
    if cfg.innovation == "gaussian":
        z = rng.standard_normal(cfg.T) #z 
    else:
        raw = rng.standard_t(cfg.t_df, size=cfg.T)
        scale = np.sqrt(cfg.t_df / (cfg.t_df - 2)) if cfg.t_df > 2 else 1.0
        z = raw / scale

    # AR(1) on standardised innovations, then rescale by sigma_vec
    z_ar = np.zeros(cfg.T)
    phi = cfg.ar_background
    z_ar[0] = z[0]
    for t in range(1, cfg.T):
        z_ar[t] = phi * z_ar[t - 1] + z[t]

    series = cfg.baseline_mu + sigma_vec * z_ar
    return series, tau


# =============================================================================
# §10 — GENERATOR 3: DEPENDENCE SHIFT
# =============================================================================
# r_t = phi_1 * r_{t-1} + eps_t  for t < tau
# r_t = phi_2 * r_{t-1} + eps_t  for t >= tau
#
# Variance stabilisation: sigma_eps = sigma * sqrt(1 - phi²)
# This keeps marginal Var(r_t) ≈ sigma² on both sides of the break,
# so the AR change does NOT co-produce a variance break.
# This is the critical fix absent from Cursor's implementation.

def generate_dependence_shift(cfg: BreakConfig) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(cfg.seed)
    tau = _resolve_tau(cfg, rng)
    sigma = _resolve_sigma(cfg, rng)

    delta_phi = MAGNITUDE_TABLE["dependence_shift"][cfg.magnitude]
    phi1 = cfg.ar_background
    phi2 = np.clip(phi1 + delta_phi, -0.95, 0.95)

    # Variance-stabilised innovation sigmas — key for label purity
    sigma_eps1 = sigma * np.sqrt(max(1 - phi1**2, 0.01))
    sigma_eps2 = sigma * np.sqrt(max(1 - phi2**2, 0.01))

    w = transition_weights(cfg.T, tau, cfg.smooth_transition, cfg.transition_width)
    phi_vec   = phi1 * (1 - w) + phi2 * w
    sigma_vec = sigma_eps1 * (1 - w) + sigma_eps2 * w

    # GARCH applied on top of the variance-stabilised sigma path
    sigma_vec = apply_garch_scaling(sigma_vec, cfg, rng)

    series = np.zeros(cfg.T)
    series[0] = sample_innovations(1, sigma, cfg.innovation, cfg.t_df, rng)[0]
    for t in range(1, cfg.T):
        eps_t   = sample_innovations(1, sigma_vec[t], cfg.innovation, cfg.t_df, rng)[0]
        series[t] = phi_vec[t] * series[t - 1] + eps_t

    return series, tau


# =============================================================================
# §11 — GENERATOR 4: DISTRIBUTIONAL SHIFT
# =============================================================================
# Pre-break:  r_t ~ t(df_pre)  * scale_pre
# Post-break: r_t ~ t(df_post) * scale_post
#
# scale chosen so Var(r) = sigma² on both sides — break is purely in SHAPE
# (kurtosis / tail weight), not in mean or variance.
#
# FIX from comparison: smooth transition now uses transition_weights()
# uniformly, removing the separate blending block that could mismatch.

def generate_distributional_shift(cfg: BreakConfig) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(cfg.seed)
    tau = _resolve_tau(cfg, rng)
    sigma = _resolve_sigma(cfg, rng)

    params  = MAGNITUDE_TABLE["distributional_shift"][cfg.magnitude]
    df_pre  = params["df_pre"]
    df_post = params["df_post"]

    # Variance-normalised scales
    scale_pre  = sigma / np.sqrt(df_pre  / (df_pre  - 2)) if df_pre  > 2 else sigma
    scale_post = sigma / np.sqrt(df_post / (df_post - 2)) if df_post > 2 else sigma

    # Build full-length series from both distributions, then blend via w(t)
    raw_pre  = rng.standard_t(df=df_pre,  size=cfg.T)
    raw_post = rng.standard_t(df=df_post, size=cfg.T)

    series_pre  = cfg.baseline_mu + raw_pre  * scale_pre
    series_post = cfg.baseline_mu + raw_post * scale_post

    # Unified blending path — FIX: uses transition_weights() consistently
    w = transition_weights(cfg.T, tau, cfg.smooth_transition, cfg.transition_width)
    series = series_pre * (1 - w) + series_post * w

    # GARCH: rescale magnitudes by clustering path (preserves shape break identity)
    sigma_vec = np.full(cfg.T, sigma)
    sigma_vec = apply_garch_scaling(sigma_vec, cfg, rng)
    garch_scale = sigma_vec / max(sigma, 1e-10)
    series = series * garch_scale

    return series, tau


# =============================================================================
# §12 — GENERATOR 5: TREND SHIFT  [ADOPTED FROM CURSOR]
# =============================================================================
# Pre-break:  r_t = intercept + slope1 * t + eps_t
# Post-break: r_t = (intercept + slope1 * tau) + slope2 * (t - tau) + eps_t
#
# The post-break intercept continues from where the pre-break trend ended,
# ensuring series continuity at tau (Cursor's approach — correct).
# Ported to use BreakConfig, MAGNITUDE_TABLE, transition_weights, and
# GARCH background — fully consistent with the rest of the framework.

def generate_trend_shift(cfg: BreakConfig) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(cfg.seed)
    tau = _resolve_tau(cfg, rng)
    sigma = _resolve_sigma(cfg, rng)

    # MAGNITUDE_TABLE stores SNR target; convert to per-step slope
    # slope = SNR * sigma / sqrt(T)  so cumulative drift ≈ SNR * sigma * sqrt(T)/2
    snr_target = MAGNITUDE_TABLE["trend_shift"][cfg.magnitude]
    d_slope    = snr_target * sigma / max(np.sqrt(cfg.T), 1.0)

    # Direction randomised via a separate seed stream
    rng_sign = np.random.default_rng(cfg.seed + 1)
    sign     = rng_sign.choice([-1, 1])
    slope1   = 0.0                       # flat pre-break baseline #generator meant to detect pure trend, not change in existing trend
    slope2   = slope1 + sign * d_slope   # post-break drift

    intercept = cfg.baseline_mu

    # Build trend vectors for each regime
    t_idx = np.arange(cfg.T)
    trend_pre  = intercept + slope1 * t_idx
    post_intercept = intercept + slope1 * tau   # continuity at tau
    trend_post = post_intercept + slope2 * (t_idx - tau)

    # Blend via transition weights
    w = transition_weights(cfg.T, tau, cfg.smooth_transition, cfg.transition_width)
    trend_vec = trend_pre * (1 - w) + trend_post * w

    eps = sample_innovations(cfg.T, sigma, cfg.innovation, cfg.t_df, rng)

    # GARCH on residuals
    sigma_vec = np.full(cfg.T, sigma)
    sigma_vec = apply_garch_scaling(sigma_vec, cfg, rng)
    eps = eps * (sigma_vec / max(sigma, 1e-10))

    series = trend_vec + eps
    return series, tau


# =============================================================================
# §13 — GENERATOR 6: NO BREAK  [NEW]
# =============================================================================
# Generates a stationary series with NO structural break.
# Required for false alarm rate (FAR) estimation:
#   FAR = P(detector fires | no break truly present)
#
# The series still respects all background conditions (AR, GARCH, tails)
# so FAR is measured under realistic market texture, not toy white noise.
# tau is returned as None — downstream code must handle this.
#
# In the manifest and MLflow, tau = -1 is used as a sentinel value
# (None is not CSV/JSON-serialisable in all contexts).

def generate_no_break(cfg: BreakConfig) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(cfg.seed)
    sigma = _resolve_sigma(cfg, rng)

    # Standardised innovations (unit variance)
    if cfg.innovation == "gaussian":
        z = rng.standard_normal(cfg.T)
    else:
        raw = rng.standard_t(cfg.t_df, size=cfg.T)
        scale = np.sqrt(cfg.t_df / (cfg.t_df - 2)) if cfg.t_df > 2 else 1.0
        z = raw / scale

    # AR(1) background on standardised residuals
    z_ar = np.zeros(cfg.T)
    phi = cfg.ar_background
    z_ar[0] = z[0]
    for t in range(1, cfg.T):
        z_ar[t] = phi * z_ar[t - 1] + z[t]

    # GARCH scaling — no break in the sigma path
    sigma_vec = np.full(cfg.T, sigma)
    sigma_vec = apply_garch_scaling(sigma_vec, cfg, rng)

    series = cfg.baseline_mu + sigma_vec * z_ar

    # tau = -1 is the sentinel for "no breakpoint"
    return series, -1


# =============================================================================
# §14 — GENERATE INSTANCE (dispatcher + effect_size annotation)
# =============================================================================
# Single entry point. Dispatches to the correct generator, then annotates
# the output dict with the computed effect_size (DECISION 4).

_GENERATORS = {
    "mean_shift":           generate_mean_shift,
    "volatility_shift":     generate_volatility_shift,
    "dependence_shift":     generate_dependence_shift,
    "distributional_shift": generate_distributional_shift,
    "trend_shift":          generate_trend_shift,
    "no_break":             generate_no_break,
}


def generate_instance(cfg: BreakConfig) -> dict:
    """
    Generate one labeled window instance.

    Returns
    -------
    dict with keys:
        series       : np.ndarray [T]
        tau          : int  (-1 for no_break)
        break_type   : str
        effect_size  : float  (normalised, break-type-specific)
        config       : dict   (full BreakConfig as dict)
        instance_id  : str    (unique human-readable key)
    """
    if cfg.break_type not in _GENERATORS:
        raise ValueError(f"Unknown break type: {cfg.break_type}")

    series, tau = _GENERATORS[cfg.break_type](cfg)

    effect_size = compute_effect_size(cfg)

    instance_id = (
        f"{cfg.break_type}"
        f"__T{cfg.T}"
        f"__tau{cfg.tau_frac}"
        f"__jit{cfg.tau_jitter}"
        f"__mag{cfg.magnitude}"
        f"__s{cfg.baseline_sigma}"
        f"__sj{cfg.sigma_jitter}"
        f"__{cfg.innovation}"
        f"__ar{cfg.ar_background}"
        f"__garch{int(cfg.garch_background)}"
        f"__smooth{int(cfg.smooth_transition)}"
        f"__seed{cfg.seed}"
    )

    return {
        "series":      series,
        "tau":         tau,
        "break_type":  cfg.break_type,
        "effect_size": effect_size,
        "config":      asdict(cfg),
        "instance_id": instance_id,
    }


"""
Meta-feature extraction has been moved out of this module.

Keep `synthetic_generators_merged.py` focused on generation only.
See `meta_features.py` for Stage-1 feature extraction utilities.
"""


# =============================================================================
# §16 — MLFLOW LOGGING
# =============================================================================
# Every generated instance is logged as one MLflow run inside a named experiment.
#
# What gets logged:
#   PARAMS   → all BreakConfig knobs (searchable/filterable in MLflow UI)
#   TAGS     → break_type, magnitude, innovation, noise_regime, tau_location
#   METRICS  → pre/post stats + effect_size (sanity check)
#   ARTIFACTS→ raw series CSV + config JSON

def log_instance_to_mlflow(instance: dict,
                            experiment_name: str = "synthetic_breaks",
                            output_dir: str = r"C:\Users\safur\OneDrive\Desktop\KU assignments\thesis\codes") -> str:
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=instance["instance_id"]) as run:
        cfg    = instance["config"]
        series = instance["series"]
        tau    = instance["tau"]

        # --- Params ---
        mlflow.log_params({k: v for k, v in cfg.items()})

        # --- Tags ---
        mlflow.set_tags({
            "break_type":  cfg["break_type"],
            "magnitude":   cfg["magnitude"],
            "innovation":  cfg["innovation"],
            "garch":       str(cfg["garch_background"]),
            "noise_regime": (
                "turbulent" if cfg["baseline_sigma"] > 0.015 else
                "calm"      if cfg["baseline_sigma"] < 0.008 else "normal"
            ),
            "tau_location": (
                "early" if cfg["tau_frac"] < 0.35 else
                "late"  if cfg["tau_frac"] > 0.65 else "mid"
            ),
            "instance_id": instance["instance_id"],
        })

        # --- Metrics ---
        safe_tau = tau if tau >= 0 else len(series) // 2
        pre  = series[:safe_tau]
        post = series[safe_tau:]

        mlflow.log_metrics({
            "series_mean":  float(np.mean(series)),
            "series_std":   float(np.std(series)),
            "series_skew":  float(stats.skew(series)),
            "series_kurt":  float(stats.kurtosis(series)),
            "pre_mean":     float(np.mean(pre)),
            "post_mean":    float(np.mean(post)),
            "pre_std":      float(np.std(pre)),
            "post_std":     float(np.std(post)),
            "mean_delta":   float(np.mean(post) - np.mean(pre)),
            "std_ratio":    float(np.std(post) / (np.std(pre) + 1e-10)),
            "effect_size":  float(instance["effect_size"]),
        })

        # --- Artifacts ---
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        csv_path = os.path.join(r"C:\Users\safur\OneDrive\Desktop\KU assignments\thesis\codes", f"{instance['instance_id'][:120]}.csv")
        df = pd.DataFrame({
            "t":       np.arange(len(series)),
            "returns": series,
            "regime":  [0 if t < safe_tau else 1 for t in range(len(series))], #0 for pre-break, 1 for post-break
        })
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        config_path = csv_path.replace(".csv", "_config.json")
        with open(config_path, "w") as f:
            json.dump(instance["config"], f, indent=2)
        mlflow.log_artifact(config_path)

        return run.info.run_id


# =============================================================================
# §17 — GENERATE CORPUS  (diversity-grid batch generator)
# =============================================================================
# Sweeps the full Cartesian product of DIVERSITY_GRID × BREAK_TYPES × n_replicates.
# For each instance:
#   1. Generates the series via generate_instance()
#   2. Logs to MLflow (optional)
#   3. Appends one row to the manifest
#
# The manifest CSV is the index linking every instance_id to its
# tau, break_type, effect_size, all knobs, and MLflow run_id.

def generate_corpus(
    break_types:     list   = BREAK_TYPES,
    grid:            dict   = DIVERSITY_GRID,
    n_replicates:    int    = 3,
    experiment_name: str    = "synthetic_breaks",
    output_dir:      str    = r"C:\Users\safur\OneDrive\Desktop\KU assignments\thesis\codes",
    log_to_mlflow:   bool   = True,
    verbose:         bool   = True,
) -> pd.DataFrame:
    """
    Generate the full diversity-grid corpus.

    Parameters
    ----------
    break_types     : list of break type strings to include
    grid            : dict of knob → list of values (Cartesian product swept)
    n_replicates    : random seeds per grid combination
    experiment_name : MLflow experiment name
    output_dir      : directory for CSV artifacts
    log_to_mlflow   : whether to log each instance as an MLflow run
    verbose         : print progress

    Returns
    -------
    manifest_df  : pd.DataFrame — one row per instance (knobs + tau + effect_size + run_id)
    """
    import itertools

    grid_keys   = list(grid.keys())
    grid_values = list(grid.values())
    combos      = list(itertools.product(*grid_values))

    manifest_rows  = []
    total = len(break_types) * len(combos) * n_replicates

    if verbose:
        print(f"Generating corpus: {total} instances total")
        print(f"  Break types  : {break_types}")
        print(f"  Grid combos  : {len(combos)}")
        print(f"  Replicates   : {n_replicates}")
        print(f"  GARCH        : {grid.get('garch_background', [False])}")
        print(f"  Output dir   : {output_dir}")
        print(f"  MLflow exp   : {experiment_name}")
        print("-" * 60)

    count = 0
    for break_type in break_types:
        for combo in combos:
            combo_dict = dict(zip(grid_keys, combo))
            for rep in range(n_replicates):
                seed = hash((break_type, str(combo_dict), rep)) % (2**31)

                cfg = BreakConfig(
                    break_type        = break_type,
                    seed              = seed,
                    T                 = combo_dict.get("T", 1000),
                    tau_frac          = combo_dict.get("tau_frac", 0.5),
                    tau_jitter        = combo_dict.get("tau_jitter", 0.0),
                    magnitude         = combo_dict.get("magnitude", "medium"),
                    baseline_sigma    = combo_dict.get("baseline_sigma", 0.01),
                    sigma_jitter      = combo_dict.get("sigma_jitter", 0.0),
                    innovation        = combo_dict.get("innovation", "gaussian"),
                    ar_background     = combo_dict.get("ar_background", 0.0),
                    garch_background  = combo_dict.get("garch_background", False),
                    smooth_transition = combo_dict.get("smooth_transition", False),
                    transition_width  = 20,
                )

                instance = generate_instance(cfg)

                run_id = None
                if log_to_mlflow:
                    try:
                        run_id = log_instance_to_mlflow(
                            instance,
                            experiment_name=experiment_name,
                            output_dir=output_dir,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"  [MLflow warning] {e}")

                # Manifest row
                row = {
                    "instance_id":   instance["instance_id"][:120],
                    "break_type":    break_type,
                    "tau":           instance["tau"],
                    "effect_size":   instance["effect_size"],
                    "mlflow_run_id": run_id,
                    **combo_dict,
                    "seed":          seed,
                    "replicate":     rep,
                }
                manifest_rows.append(row)

                count += 1
                if verbose and count % 100 == 0:
                    print(f"  [{count}/{total}] {instance['instance_id'][:70]}...")

    manifest_df = pd.DataFrame(manifest_rows)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    manifest_path = os.path.join(output_dir, "corpus_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    if verbose:
        print("-" * 60)
        print(f"Done. {count} instances generated.")
        print(f"  Manifest      : {manifest_path}")
        print("\nBreak type counts:")
        print(manifest_df["break_type"].value_counts().to_string())

    return manifest_df


# =============================================================================
# §18 — SMOKE TEST
# =============================================================================
# Generates one instance per break type, prints pre/post stats to verify
# each generator is producing the intended kind of break.
# (Meta-feature extraction lives in a separate module.)

def demo_all_generators(garch: bool = False) -> None:
    """
    Quick smoke test: one instance per break type, printed pre/post stats.

    Parameters
    ----------
    garch : bool
        If True, run the smoke test with GARCH background enabled so you
        can visually verify the clustering path does not distort break types.
    """
    print("=" * 65)
    print(f"SMOKE TEST — one instance per break type  (GARCH={garch})")
    print("=" * 65)

    base_cfg = dict(
        seed=0, T=1000, tau_frac=0.5, tau_jitter=0.0,
        magnitude="medium", baseline_sigma=0.01, sigma_jitter=0.0,
        innovation="gaussian", ar_background=0.0,
        garch_background=garch, smooth_transition=False,
    )

    for bt in BREAK_TYPES:
        cfg = BreakConfig(break_type=bt, **base_cfg)
        inst = generate_instance(cfg)
        s    = inst["series"]
        tau  = inst["tau"]

        safe_tau = tau if tau >= 0 else cfg.T // 2
        pre  = s[:safe_tau]
        post = s[safe_tau:]

        pre_mean  = np.mean(pre);   post_mean  = np.mean(post)
        pre_std   = np.std(pre);    post_std   = np.std(post)
        pre_kurt  = stats.kurtosis(pre); post_kurt = stats.kurtosis(post)

        print(f"\n[{bt}]  tau={tau}  effect_size={inst['effect_size']:.4f}")
        print(f"  pre  → mean={pre_mean:+.5f}  std={pre_std:.5f}  kurt={pre_kurt:.2f}")
        print(f"  post → mean={post_mean:+.5f}  std={post_std:.5f}  kurt={post_kurt:.2f}")
        print(f"  Δmean={post_mean-pre_mean:+.5f}  "
              f"std_ratio={post_std/(pre_std+1e-10):.3f}  "
              f"Δkurt={post_kurt-pre_kurt:.2f}")

    print("\n" + "=" * 65)
    print("Smoke test complete. All generators functional.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # --- Step 1: Smoke test (without and with GARCH) ---
    # new — single call, both passes handled internally
    validate_smoke_test(interactive=True)

    # --- Step 2: Small corpus for rapid validation ---
    # Subset of the full grid — swap in DIVERSITY_GRID for the full run.
    small_grid = {
        "T":                [500, 1000],
        "tau_frac":         [0.25, 0.5, 0.75],
        "tau_jitter":       [0.0, 0.05],
        "magnitude":        ["small", "medium", "large"],
        "baseline_sigma":   [0.01],
        "sigma_jitter":     [0.0, 0.002],
        "innovation":       ["gaussian", "student_t"],
        "ar_background":    [0.0, 0.3],
        "garch_background": [False, True],
        "smooth_transition":[False, True],
    }

    manifest = generate_corpus(
        break_types      = BREAK_TYPES,
        grid             = small_grid,
        n_replicates     = 2,
        experiment_name  = "synthetic_breaks_merged_v1",
        output_dir       = "./data/synthetic",
        log_to_mlflow    = False,   # set True when MLflow tracking server is ready
        verbose          = True,
    )

    print(f"\nManifest shape  : {manifest.shape}")
    print(manifest[["break_type", "tau", "effect_size", "garch_background"]].head(12))
