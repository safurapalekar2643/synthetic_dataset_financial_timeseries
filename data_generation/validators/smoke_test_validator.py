"""
=============================================================================
SMOKE TEST VALIDATOR — smoke_test_validator.py
=============================================================================

Standalone module for automatically validating synthetic generator output.
Import into synthetic_generators_merged.py and call from __main__.

Usage
-----
    from smoke_test_validator import validate_smoke_test

    validate_smoke_test(interactive=True)

    # That single call runs two passes internally:
    #   Pass 1 — GARCH=False: stats + validation
    #   Pass 2 — GARCH=True:  stats + validation (wider bounds)
    # If either pass has a hard failure, execution stops immediately.
    # Interactive confirmation fires once, after both passes pass.

Two tiers of checks
-------------------
  HARD (raises SystemExit(1))
      Label contamination — the wrong statistic changed.
      Corpus generation is blocked until the issue is fixed.

  SOFT (prints warning, execution continues)
      Known limitations that are acceptable but must be documented
      in the thesis. Corpus generation proceeds.

GARCH bounds
------------
  GARCH=False uses tight bounds — only finite-sample noise moves stats.
  GARCH=True  uses wider bounds — clustering legitimately widens std_ratio
              even in clean windows. Each check function receives a garch
              flag and adjusts its thresholds accordingly.

Checks per break type
---------------------
  mean_shift           Δmean large, std_ratio ≈ 1, Δkurt small
  volatility_shift     std_ratio ≈ expected ratio, Δmean ≈ 0, Δkurt small
  dependence_shift     std_ratio ≈ 1 (variance stabilisation), Δmean ≈ 0
  distributional_shift abs(Δkurt) large, std_ratio ≈ 1, Δmean ≈ 0
  trend_shift          pre_slope ≈ 0, post_slope ≠ 0, detrended resid_ratio ≈ 1
  no_break             tau == -1, effect_size == 0, all deltas small

Dependencies
------------
  Imports from synthetic_generators_merged.py:
      BREAK_TYPES, BreakConfig, generate_instance
  Standard: numpy, scipy.stats
=============================================================================
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class SmokeCheckResult:
    """Holds the outcome of one break type's label purity check."""
    break_type: str
    passed:     bool
    hard_fails: list   # list[str] — block corpus generation
    soft_warns: list   # list[str] — document in thesis, do not block


# =============================================================================
# HELPERS
# =============================================================================

def _wrap(text: str, width: int = 70) -> list:
    """Word-wrap a string to fit within terminal width."""
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _estimate_slope(segment: np.ndarray) -> float:
    """
    OLS slope of segment against time index.
    slope = cov(t, y) / var(t).
    Used by _check_trend_shift — raw std_ratio is not a valid diagnostic
    for a trending series because trend accumulation inflates post-break std.
    """
    t = np.arange(len(segment), dtype=float)
    t_mean = t.mean()
    return float(
        np.sum((t - t_mean) * (segment - segment.mean())) /
        np.sum((t - t_mean) ** 2)
    )


# =============================================================================
# PER-BREAK-TYPE CHECK FUNCTIONS
# =============================================================================
# Each function receives a garch: bool flag and adjusts thresholds accordingly.
#
# GARCH=False bounds — tight, only finite-sample noise moves stats.
# GARCH=True  bounds — wider, clustering legitimately moves std_ratio
#                      even in clean windows. Hard-fail ceiling prevents
#                      GARCH from creating a co-occurring volatility break.

def _check_mean_shift(
    bt:             str,
    delta_mean:     float,
    std_ratio:      float,
    delta_kurt:     float,
    baseline_sigma: float,
    magnitude:      str,
    garch:          bool,
) -> SmokeCheckResult:
    """
    Mean shift: only the level should change.
    Δmean must match MAGNITUDE_TABLE. std_ratio and Δkurt must stay near 1/0.
    """
    hard, soft = [], []

    expected_delta = {"small": 0.5, "medium": 1.5, "large": 3.0}[magnitude] * baseline_sigma

    # Δmean — break must appear
    if abs(delta_mean) < 0.5 * expected_delta:
        hard.append(
            f"Δmean={delta_mean:.5f} is less than 50% of expected {expected_delta:.5f}. "
            f"Mean vector is not being applied. "
            f"Check mean_vec construction in generate_mean_shift()."
        )

    # std_ratio — variance must not change.
    # GARCH widens the band but caps at 1.50 — above that, GARCH is creating
    # a co-occurring volatility break that contaminates the label.
    hard_lo, hard_hi = (0.70, 1.50) if garch else (0.85, 1.15)
    soft_lo, soft_hi = (0.80, 1.30) if garch else (0.90, 1.10)

    if not (hard_lo <= std_ratio <= hard_hi):
        hard.append(
            f"std_ratio={std_ratio:.3f} is outside [{hard_lo}, {hard_hi}]. "
            f"Variance is leaking into a pure level break. "
            f"Check that GARCH rescaling of eps does not interact with mean_vec "
            f"and that AR is applied on demeaned residuals."
        )
    elif not (soft_lo <= std_ratio <= soft_hi):
        if garch:
            soft.append(
                f"std_ratio={std_ratio:.3f} outside ideal [{soft_lo}, {soft_hi}] under GARCH. "
                f"GARCH clustering creates half-to-half variance asymmetry — "
                f"expected background effect, not label contamination."
            )
        else:
            soft.append(
                f"std_ratio={std_ratio:.3f} slightly outside ideal [0.90, 1.10]. "
                f"Acceptable at T=1000 — monitor in full corpus."
            )

    # Δkurt — tail shape must not change
    if abs(delta_kurt) > 1.0:
        soft.append(
            f"Δkurt={delta_kurt:.2f} elevated. Distribution shape should not "
            f"change in a level break. Likely finite-sample noise — verify at larger T."
        )
    elif abs(delta_kurt) > 0.5:
        soft.append(f"Δkurt={delta_kurt:.2f} slightly elevated — monitor in corpus.")

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_volatility_shift(
    bt:             str,
    delta_mean:     float,
    std_ratio:      float,
    delta_kurt:     float,
    magnitude:      str,
    pre_std:        float,
    baseline_sigma: float,
    garch:          bool,
) -> SmokeCheckResult:
    """
    Volatility shift: only the scale should change.
    std_ratio must match MAGNITUDE_TABLE ratio. Δmean and Δkurt near zero.
    """
    hard, soft = [], []

    expected_ratio = {"small": 1.5, "medium": 2.5, "large": 4.0}[magnitude]

    # std_ratio — break must appear at approximately the right magnitude.
    # GARCH adds clustering on top of the intentional break so tolerance widens.
    hard_tol = 0.40 if garch else 0.30
    soft_tol = 0.30 if garch else 0.20

    if not ((1 - hard_tol) * expected_ratio <= std_ratio <= (1 + hard_tol) * expected_ratio):
        hard.append(
            f"std_ratio={std_ratio:.3f} is far from expected ≈{expected_ratio} "
            f"({magnitude} magnitude). Volatility break is not being applied. "
            f"Check sigma_vec construction in generate_volatility_shift()."
        )
    elif not ((1 - soft_tol) * expected_ratio <= std_ratio <= (1 + soft_tol) * expected_ratio):
        soft.append(
            f"std_ratio={std_ratio:.3f} slightly off expected ≈{expected_ratio}. "
            f"{'GARCH clustering inflating post-break variance — acceptable.' if garch else 'Monitor in full corpus.'}"
        )

    # Δmean — must not change
    if abs(delta_mean) > 0.05:
        hard.append(
            f"Δmean={delta_mean:.5f} too large for a volatility break. "
            f"Mean is being contaminated. Check baseline_mu and AR demeaning."
        )

    # Δkurt — must not change. GARCH adds unconditional kurtosis so tolerance widens.
    kurt_hard = 1.5 if garch else 1.0
    kurt_soft = 0.8 if garch else 0.5
    if abs(delta_kurt) > kurt_hard:
        soft.append(
            f"Δkurt={delta_kurt:.2f} substantially elevated. Distribution shape "
            f"should not change when only scale changes. "
            f"Check sigma_vec rescaling is on standardised innovations."
        )
    elif abs(delta_kurt) > kurt_soft:
        soft.append(f"Δkurt={delta_kurt:.2f} slightly elevated — monitor in corpus.")

    # pre_std — should be close to baseline_sigma. GARCH widens this band.
    pre_tol = 0.40 if garch else 0.20
    if not ((1 - pre_tol) * baseline_sigma <= pre_std <= (1 + pre_tol) * baseline_sigma):
        hard.append(
            f"pre_std={pre_std:.5f} not close to baseline_sigma={baseline_sigma:.5f}. "
            f"_resolve_sigma() or AR background is distorting the pre-break level."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_dependence_shift(
    bt:         str,
    delta_mean: float,
    std_ratio:  float,
    delta_kurt: float,
    garch:      bool,
) -> SmokeCheckResult:
    """
    Dependence shift: only the AR coefficient should change.
    std_ratio is THE critical check — variance stabilisation must hold.
    GARCH widens the band slightly since clustering adds some variance noise.
    """
    hard, soft = [], []

    # std_ratio — variance stabilisation check
    hard_lo, hard_hi = (0.75, 1.25) if garch else (0.85, 1.15)
    soft_lo, soft_hi = (0.85, 1.15) if garch else (0.90, 1.10)

    if not (hard_lo <= std_ratio <= hard_hi):
        hard.append(
            f"std_ratio={std_ratio:.3f} outside [{hard_lo}, {hard_hi}]. "
            f"Variance stabilisation FAILED. AR change is co-producing a variance "
            f"change — label contaminated with volatility shift. "
            f"Fix: sigma_eps = sigma * sqrt(1 - phi^2) on BOTH sides of break."
        )
    elif not (soft_lo <= std_ratio <= soft_hi):
        soft.append(
            f"std_ratio={std_ratio:.3f} slightly outside [{soft_lo}, {soft_hi}]. "
            f"Variance stabilisation working but imperfect — document in thesis."
        )

    # Δmean — must not change
    if abs(delta_mean) > 0.02:
        hard.append(
            f"Δmean={delta_mean:.5f} too large for a dependence break. "
            f"Check baseline_mu stability and AR recursion."
        )

    # Δkurt — stricter than other break types because nothing about the
    # innovation distribution changes in a dependence shift.
    # GARCH adds kurtosis through clustering so tolerance widens slightly.
    kurt_hard = 1.0 if garch else 0.75
    kurt_soft = 0.6 if garch else 0.4
    if abs(delta_kurt) > kurt_hard:
        soft.append(
            f"Δkurt={delta_kurt:.2f} elevated for a dependence shift. "
            f"Innovation distribution should be identical on both sides. "
            f"If this persists at larger T, check sample_innovations() parameters."
        )
    elif abs(delta_kurt) > kurt_soft:
        soft.append(f"Δkurt={delta_kurt:.2f} slightly elevated — monitor in corpus.")

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_distributional_shift(
    bt:         str,
    delta_mean: float,
    std_ratio:  float,
    delta_kurt: float,
    pre_kurt:   float,
    garch:      bool,
) -> SmokeCheckResult:
    """
    Distributional shift: only tail shape should change.
    abs(Δkurt) must be large. std_ratio and Δmean near zero.
    Both directions (light→heavy and heavy→light) are valid.
    """
    hard, soft = [], []

    # abs(Δkurt) — break must appear as kurtosis change in either direction.
    # GARCH adds its own kurtosis so the required minimum is slightly lower.
    min_kurt = 0.4 if garch else 0.5
    if abs(delta_kurt) < min_kurt:
        hard.append(
            f"abs(Δkurt)={abs(delta_kurt):.2f} is too small. "
            f"Tail shape is not changing in either direction. "
            f"Check df_pre/df_post in MAGNITUDE_TABLE and variance normalisation."
        )
    elif abs(delta_kurt) < 1.0:
        soft.append(
            f"abs(Δkurt)={abs(delta_kurt):.2f} present but weak. "
            f"Stage 1 may struggle to detect this magnitude reliably."
        )

    # std_ratio — variance normalisation check. GARCH widens the band.
    hard_lo, hard_hi = (0.75, 1.25) if garch else (0.85, 1.15)
    if not (hard_lo <= std_ratio <= hard_hi):
        hard.append(
            f"std_ratio={std_ratio:.3f} outside [{hard_lo}, {hard_hi}]. "
            f"Variance normalisation failed — both t-distributions should be "
            f"rescaled to the same sigma^2. Check scale_pre and scale_post."
        )

    # Δmean — must stay near zero
    if abs(delta_mean) > 0.02:
        hard.append(
            f"Δmean={delta_mean:.5f} too large. Mean should be stable. "
            f"Check baseline_mu symmetry and centring of t-draws."
        )

    # pre_kurt sanity check — should be near zero for t(df=30)
    if pre_kurt > 2.0:
        soft.append(
            f"pre_kurt={pre_kurt:.2f} higher than expected for t(df=30). "
            f"Finite-sample noise likely — check df_pre."
        )

    # Effect size scale note — always flagged
    soft.append(
        f"effect_size is on a KL-divergence scale, not comparable to "
        f"Cohen's d or SNR used by other break types. Document in thesis."
    )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_trend_shift(
    bt:             str,
    pre:            np.ndarray,
    post:           np.ndarray,
    delta_kurt:     float,
    baseline_sigma: float,
    magnitude:      str,
    garch:          bool,
) -> SmokeCheckResult:
    """
    Trend shift: post-break segment has a linear drift.
    Diagnosed via OLS slope — NOT std_ratio (trend accumulation inflates it).
    pre_slope must be near zero. post_slope must match SNR target.
    Detrended residual std must stay near baseline_sigma on both sides.
    """
    hard, soft = [], []

    pre_slope   = _estimate_slope(pre)
    post_slope  = _estimate_slope(post)
    delta_slope = post_slope - pre_slope

    # Detrend each half to recover true innovation std
    t_pre  = np.arange(len(pre),  dtype=float)
    t_post = np.arange(len(post), dtype=float)
    pre_resid  = pre  - (pre_slope  * t_pre  + pre.mean()  - pre_slope  * t_pre.mean())
    post_resid = post - (post_slope * t_post + post.mean() - post_slope * t_post.mean())
    resid_ratio = float(np.std(post_resid) / (np.std(pre_resid) + 1e-10))

    # Expected slope from MAGNITUDE_TABLE SNR target: slope = SNR * sigma / sqrt(T)
    snr = {"small": 1.0, "medium": 2.0, "large": 4.0}[magnitude]
    T   = len(pre) + len(post)
    expected_slope = snr * baseline_sigma / max(np.sqrt(T), 1.0)

    # pre-break slope must be near zero (slope1=0 by generator design)
    if abs(pre_slope) > 0.5 * expected_slope:
        hard.append(
            f"pre_slope={pre_slope:.6f} too large — pre-break should be flat. "
            f"Check trend_pre uses slope1=0 in generate_trend_shift()."
        )

    # post-break slope must be present
    if abs(post_slope) < 0.3 * expected_slope:
        hard.append(
            f"post_slope={post_slope:.6f} too small (expected ≈{expected_slope:.6f}). "
            f"Trend not being applied. Check slope2 and trend_vec in generate_trend_shift()."
        )

    # Δslope must be meaningful
    if abs(delta_slope) < 0.3 * expected_slope:
        hard.append(
            f"Δslope={delta_slope:.6f} near zero — no slope change detected."
        )

    # Detrended residual std_ratio — noise level must not change across the break.
    # GARCH widens the band since clustering adds variance noise to residuals.
    hard_lo, hard_hi = (0.65, 1.35) if garch else (0.70, 1.30)
    soft_lo, soft_hi = (0.80, 1.20) if garch else (0.85, 1.15)

    if not (hard_lo <= resid_ratio <= hard_hi):
        hard.append(
            f"Detrended resid_ratio={resid_ratio:.3f} outside [{hard_lo}, {hard_hi}]. "
            f"Innovation noise level is changing across the trend break — "
            f"co-occurring volatility shift. Check GARCH scaling in generate_trend_shift()."
        )
    elif not (soft_lo <= resid_ratio <= soft_hi):
        soft.append(
            f"Detrended resid_ratio={resid_ratio:.3f} slightly off 1.0 — "
            f"{'GARCH clustering effect — acceptable.' if garch else 'monitor in corpus.'}"
        )

    # Document the std_ratio limitation — always flagged
    soft.append(
        f"Raw std_ratio not checked for trend_shift — trend accumulation always "
        f"inflates it. Detrend before computing variance-based meta-features."
    )

    if delta_kurt > 2.0:
        soft.append(
            f"Δkurt={delta_kurt:.2f} unexpectedly positive. "
            f"Trend shifts typically produce negative or near-zero Δkurt."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_no_break(
    bt:          str,
    delta_mean:  float,
    std_ratio:   float,
    delta_kurt:  float,
    tau:         int,
    effect_size: float,
    garch:       bool,
) -> SmokeCheckResult:
    """
    No break: stationary series with no parameter change.
    tau must be -1. effect_size must be 0. All deltas near zero.
    GARCH adds clustering so std_ratio will naturally deviate more from 1.0 —
    this sets the realistic FAR baseline for Stage 2 benchmarking.
    """
    hard, soft = [], []

    if tau != -1:
        hard.append(
            f"tau={tau} but expected -1 for no_break. "
            f"Sentinel value overwritten in generate_no_break() or generate_instance()."
        )

    if effect_size != 0.0:
        hard.append(
            f"effect_size={effect_size} but expected exactly 0.0 for no_break."
        )

    if abs(delta_mean) > 0.02:
        hard.append(
            f"Δmean={delta_mean:.5f} too large for a stationary series. "
            f"Spurious drift detected — check ar_background and baseline_mu."
        )

    # std_ratio — GARCH widens the natural deviation from 1.0 substantially.
    # The GARCH hard ceiling is 1.50: beyond that, a stationary GARCH series
    # would trigger any reasonable variance-based detector, making FAR
    # estimation meaningless.
    hard_lo, hard_hi = (0.70, 1.50) if garch else (0.80, 1.20)
    soft_lo, soft_hi = (0.80, 1.30) if garch else (0.90, 1.10)

    if not (hard_lo <= std_ratio <= hard_hi):
        hard.append(
            f"std_ratio={std_ratio:.3f} outside [{hard_lo}, {hard_hi}] for no_break. "
            f"Implausibly large variance asymmetry in a stationary series. "
            f"Check no parameters change across the window."
        )
    elif not (soft_lo <= std_ratio <= soft_hi):
        soft.append(
            f"std_ratio={std_ratio:.3f} deviates from 1.0. "
            f"{'GARCH clustering creates half-to-half variance variation — expected. ' if garch else 'Finite-sample variance — expected at T=500 per half. '}"
            f"FAR BASELINE: detectors seeing std_ratio ≈ {std_ratio:.3f} should "
            f"not fire. Use as lower reference in Stage 2 threshold calibration."
        )

    if abs(delta_kurt) > 1.5:
        soft.append(
            f"Δkurt={delta_kurt:.2f} elevated in no_break window. "
            f"Kurtosis is noisy at small samples — acceptable."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


# =============================================================================
# SINGLE PASS — stats + validation for one GARCH setting
# =============================================================================

def _run_one_pass(
    garch:            bool,
    baseline_sigma:   float,
    magnitude:        str,
    BREAK_TYPES:      list,
    BreakConfig,
    generate_instance,
) -> list:
    """
    Generate one instance per break type, print stats, run checks.
    Returns list of SmokeCheckResult — one per break type.
    """
    print("\n" + "=" * 65)
    print(f"PASS — GARCH={garch}")
    print("=" * 65)

    base_cfg = dict(
        seed=0, T=1000, tau_frac=0.5, tau_jitter=0.0,
        magnitude=magnitude, baseline_sigma=baseline_sigma,
        sigma_jitter=0.0, innovation="gaussian", ar_background=0.0,
        garch_background=garch, smooth_transition=False,
    )

    results = []

    for bt in BREAK_TYPES:
        cfg  = BreakConfig(break_type=bt, **base_cfg)
        inst = generate_instance(cfg)
        s    = inst["series"]
        tau  = inst["tau"]

        safe_tau = tau if tau >= 0 else cfg.T // 2
        pre      = s[:safe_tau]
        post     = s[safe_tau:]

        pre_std    = float(np.std(pre))
        delta_mean = float(np.mean(post) - np.mean(pre))
        std_ratio  = float(np.std(post) / (np.std(pre) + 1e-10))
        delta_kurt = float(stats.kurtosis(post) - stats.kurtosis(pre))
        pre_kurt   = float(stats.kurtosis(pre))
        effect_size = inst["effect_size"]

        # --- Raw stats ---
        print(f"\n[{bt}]  tau={tau}  effect_size={effect_size:.4f}")
        print(f"  pre  → mean={np.mean(pre):+.5f}  std={pre_std:.5f}  kurt={stats.kurtosis(pre):.2f}")
        print(f"  post → mean={np.mean(post):+.5f}  std={np.std(post):.5f}  kurt={stats.kurtosis(post):.2f}")
        print(f"  Δmean={delta_mean:+.5f}  std_ratio={std_ratio:.3f}  Δkurt={delta_kurt:.2f}")
        if bt == "trend_shift":
            pre_sl  = _estimate_slope(pre)
            post_sl = _estimate_slope(post)
            print(f"  pre_slope={pre_sl:+.6f}  post_slope={post_sl:+.6f}  Δslope={post_sl - pre_sl:+.6f}")

        # --- Validation ---
        if bt == "mean_shift":
            r = _check_mean_shift(
                bt, delta_mean, std_ratio, delta_kurt,
                baseline_sigma, magnitude, garch
            )
        elif bt == "volatility_shift":
            r = _check_volatility_shift(
                bt, delta_mean, std_ratio, delta_kurt,
                magnitude, pre_std, baseline_sigma, garch
            )
        elif bt == "dependence_shift":
            r = _check_dependence_shift(
                bt, delta_mean, std_ratio, delta_kurt, garch
            )
        elif bt == "distributional_shift":
            r = _check_distributional_shift(
                bt, delta_mean, std_ratio, delta_kurt, pre_kurt, garch
            )
        elif bt == "trend_shift":
            r = _check_trend_shift(
                bt, pre, post, delta_kurt,
                baseline_sigma, magnitude, garch
            )
        elif bt == "no_break":
            r = _check_no_break(
                bt, delta_mean, std_ratio, delta_kurt, tau, effect_size, garch
            )
        else:
            r = SmokeCheckResult(bt, True, [], [f"No validator defined for {bt}."])

        results.append(r)

    # --- Validation report for this pass ---
    print("\n" + "-" * 65)
    print(f"VALIDATION REPORT — GARCH={garch}")
    print("-" * 65)

    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        n_warn = len(r.soft_warns)
        warn_label = f"  ({n_warn} warning{'s' if n_warn != 1 else ''})" if n_warn else ""
        print(f"\n  [{status}]  {r.break_type}{warn_label}")

        for msg in r.hard_fails:
            for line in _wrap(f"  ✗ HARD FAIL: {msg}", width=68):
                print(f"     {line}")

        for msg in r.soft_warns:
            for line in _wrap(f"  ⚠ WARNING:  {msg}", width=68):
                print(f"     {line}")

    return results


# =============================================================================
# MAIN VALIDATOR — runs both passes then gates corpus generation
# =============================================================================

def validate_smoke_test(
    baseline_sigma: float = 0.01,
    magnitude:      str   = "medium",
    interactive:    bool  = True,
) -> None:
    """
    Run smoke test for GARCH=False then GARCH=True in a single call.

    For each pass:
      - Generates one instance per break type
      - Prints raw pre/post stats
      - Validates label purity with GARCH-appropriate bounds
      - Prints pass/fail report for that pass

    After both passes:
      - If any hard failure exists in either pass → SystemExit(1)
      - If all checks pass → optional interactive confirmation

    Parameters
    ----------
    baseline_sigma : noise level used in checks (default 0.01)
    magnitude      : break size used in checks (default "medium")
    interactive    : if True, ask for confirmation before continuing

    Raises
    ------
    SystemExit(1)  any hard failure detected in either pass
    SystemExit(0)  user declines to continue
    """

    # Late import to avoid circular dependency when imported into generators file
    from synthetic_generators_merged import (
        BREAK_TYPES, BreakConfig, generate_instance
    )

    # --- Pass 1: GARCH=False ---
    results_no_garch = _run_one_pass(
        garch=False,
        baseline_sigma=baseline_sigma,
        magnitude=magnitude,
        BREAK_TYPES=BREAK_TYPES,
        BreakConfig=BreakConfig,
        generate_instance=generate_instance,
    )

    # --- Pass 2: GARCH=True ---
    results_garch = _run_one_pass(
        garch=True,
        baseline_sigma=baseline_sigma,
        magnitude=magnitude,
        BREAK_TYPES=BREAK_TYPES,
        BreakConfig=BreakConfig,
        generate_instance=generate_instance,
    )

    # --- Combined summary ---
    all_results   = results_no_garch + results_garch
    any_hard_fail = any(not r.passed for r in all_results)
    any_soft_warn = any(r.soft_warns for r in all_results)

    print("\n" + "=" * 65)
    print("COMBINED SUMMARY")
    print("=" * 65)

    print("\n  GARCH=False")
    for r in results_no_garch:
        status = "✓" if r.passed else "✗"
        print(f"    [{status}] {r.break_type}")

    print("\n  GARCH=True")
    for r in results_garch:
        status = "✓" if r.passed else "✗"
        print(f"    [{status}] {r.break_type}")

    print()

    if any_soft_warn and not any_hard_fail:
        print("⚠  Soft warnings detected — document in thesis.")
        print("   Corpus generation can proceed.\n")

    if any_hard_fail:
        failed_no_g = [r.break_type for r in results_no_garch if not r.passed]
        failed_g    = [r.break_type for r in results_garch    if not r.passed]
        print("=" * 65)
        if failed_no_g:
            print(f"✗  HARD FAILURES (GARCH=False): {', '.join(failed_no_g)}")
        if failed_g:
            print(f"✗  HARD FAILURES (GARCH=True):  {', '.join(failed_g)}")
        print("   Fix the issues above before running the full corpus.")
        print("=" * 65)
        raise SystemExit(1)

    print("✓  All checks passed for both GARCH=False and GARCH=True.")
    print("=" * 65)

    # --- Interactive confirmation ---
    if interactive:
        print("\nReview any warnings above, then confirm:")
        print("  [y / Enter]  Continue to corpus generation")
        print("  [n]          Stop here")
        choice = input("  Your choice: ").strip().lower()
        if choice in ("n", "no"):
            print("\nStopping. Call generate_corpus() manually when ready.")
            raise SystemExit(0)
        print()
