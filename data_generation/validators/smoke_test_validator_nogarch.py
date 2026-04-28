"""
=============================================================================
SMOKE TEST VALIDATOR — smoke_test_validator.py
=============================================================================

Standalone module for automatically validating synthetic generator output.
Import into synthetic_generators_merged.py and call from __main__.

Usage
-----
    from smoke_test_validator import validate_smoke_test

    # verbose=True prints raw pre/post stats before the pass/fail verdict,
    # replacing demo_all_generators() entirely.
    validate_smoke_test(garch=False, interactive=True, verbose=True)
    validate_smoke_test(garch=True,  interactive=True, verbose=True)

Two tiers of checks
-------------------
  HARD (raises SystemExit(1))
      Label contamination — the wrong statistic changed.
      Corpus generation is blocked until the issue is fixed.

  SOFT (prints warning, execution continues)
      Known limitations that are acceptable but must be documented
      in the thesis. Corpus generation proceeds.

Checks per break type
---------------------
  mean_shift          Δmean large, std_ratio ≈ 1, Δkurt small
  volatility_shift    std_ratio ≈ expected ratio, Δmean ≈ 0, Δkurt small
  dependence_shift    std_ratio ≈ 1 (variance stabilisation), Δmean ≈ 0
  distributional_shift Δkurt large, std_ratio ≈ 1, Δmean ≈ 0
  trend_shift         pre_slope ≈ 0, post_slope ≠ 0, detrended resid_ratio ≈ 1
  no_break            tau == -1, effect_size == 0, all deltas small

Dependencies
------------
  Imports from synthetic_generators_merged.py:
      BREAK_TYPES, MAGNITUDE_TABLE, BreakConfig, generate_instance
  Standard: numpy, scipy.stats
=============================================================================
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# These are imported from the main generator file at call time.
# Kept as a late import inside validate_smoke_test() to avoid circular issues.
# ---------------------------------------------------------------------------


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
# PER-BREAK-TYPE CHECK FUNCTIONS
# =============================================================================

def _check_mean_shift(
    bt:             str,
    delta_mean:     float,
    std_ratio:      float,
    delta_kurt:     float,
    baseline_sigma: float,
    magnitude:      str,
) -> SmokeCheckResult:
    """
    Mean shift: only the level should change.
    Δmean must match MAGNITUDE_TABLE. std_ratio and Δkurt must stay near zero.
    """
    hard, soft = [], []

    expected_delta = {"small": 0.5, "medium": 1.5, "large": 3.0}[magnitude] * baseline_sigma

    # Δmean — the break must actually appear
    if abs(delta_mean) < 0.5 * expected_delta:
        hard.append(
            f"Δmean={delta_mean:.5f} is less than 50% of the expected "
            f"{expected_delta:.5f} ({magnitude} magnitude × baseline_sigma). "
            f"The mean vector is not being applied. "
            f"Check mean_vec construction in generate_mean_shift()."
        )

    # std_ratio — variance must not change
    if not (0.85 <= std_ratio <= 1.15):
        hard.append(
            f"std_ratio={std_ratio:.3f} is outside [0.85, 1.15]. "
            f"Variance is leaking into what should be a pure level break. "
            f"Check that GARCH rescaling of eps does not interact with mean_vec, "
            f"and that AR is applied on demeaned residuals."
        )
    elif not (0.90 <= std_ratio <= 1.10):
        soft.append(
            f"std_ratio={std_ratio:.3f} is slightly outside the ideal [0.90, 1.10]. "
            f"Acceptable at T=1000 — monitor in full corpus."
        )

    # Δkurt — tail shape must not change
    if abs(delta_kurt) > 1.0:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is substantially elevated for a mean shift. "
            f"Distribution shape should not change when only the level changes. "
            f"Verify with larger T to rule out sampling noise."
        )
    elif abs(delta_kurt) > 0.5:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is slightly elevated. "
            f"Likely finite-sample noise at T=500 per half — monitor in full corpus."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_volatility_shift(
    bt:             str,
    delta_mean:     float,
    std_ratio:      float,
    delta_kurt:     float,
    magnitude:      str,
    pre_std:        float,
    baseline_sigma: float,
) -> SmokeCheckResult:
    """
    Volatility shift: only the scale should change.
    std_ratio must match MAGNITUDE_TABLE ratio. Δmean and Δkurt must stay near zero.
    pre_std must be close to baseline_sigma.
    """
    hard, soft = [], []

    expected_ratio = {"small": 1.5, "medium": 2.5, "large": 4.0}[magnitude]

    # std_ratio — the break must appear at approximately the right magnitude
    if not (0.70 * expected_ratio <= std_ratio <= 1.30 * expected_ratio): #+-30% tolerance
        hard.append(
            f"std_ratio={std_ratio:.3f} is far from the expected ≈{expected_ratio} "
            f"({magnitude} magnitude). The volatility break is not being applied. "
            f"Check sigma_vec construction and that AR is applied on standardised "
            f"innovations before rescaling in generate_volatility_shift()."
        )
    elif not (0.80 * expected_ratio <= std_ratio <= 1.20 * expected_ratio): #+-20% tolerance
        soft.append(
            f"std_ratio={std_ratio:.3f} is slightly off the expected ≈{expected_ratio}. "
            f"GARCH clustering may be inflating post-break variance — acceptable."
        )

    # Δmean — mean must not change
    if abs(delta_mean) > 0.05: # expected is 0, with 5% tolerance due to finite-sample noise and potential GARCH interaction
        hard.append(
            f"Δmean={delta_mean:.5f} is too large for a volatility break. "
            f"The mean is being contaminated by the volatility change. "
            f"Check that baseline_mu is stable and AR residuals are correctly demeaned."
        )

    # Δkurt — tail shape must not change
    if abs(delta_kurt) > 1.0:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is substantially elevated for a volatility shift. "
            f"Distribution shape should not change when only scale changes. "
            f"Check that sigma_vec rescaling in generate_volatility_shift() is "
            f"applied to standardised unit-variance innovations, not raw series. "
            f"Verify with student_t innovations and larger T to rule out sampling noise."
        )
    elif abs(delta_kurt) > 0.5:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is slightly elevated. "
            f"Likely finite-sample noise at T=500 per half — monitor in full corpus."
        )

    # pre_std — should be close to baseline_sigma
    if not (0.8 * baseline_sigma <= pre_std <= 1.2 * baseline_sigma):
        hard.append(
            f"pre_std={pre_std:.5f} is not close to baseline_sigma={baseline_sigma:.5f}. "
            f"Expected [0.8*baseline_sigma, 1.2*baseline_sigma] = [{0.8*baseline_sigma:.5f}, {1.2*baseline_sigma:.5f}]. "
            f"_resolve_sigma() or AR background is distorting the pre-break level. "
            f"Check sigma resolution and AR application."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_dependence_shift(
    bt:         str,
    delta_mean: float,
    std_ratio:  float,
    delta_kurt: float,
) -> SmokeCheckResult:
    """
    Dependence shift: only the AR coefficient should change.
    std_ratio is THE critical check — variance stabilisation must hold.
    Δmean and Δkurt must stay near zero. The ACF change is invisible to
    these summary stats, so checking what does NOT change is the only test
    available from pre/post means and stds.
    """
    hard, soft = [], []

    # std_ratio — the most important check in the entire smoke test
    if not (0.85 <= std_ratio <= 1.15):
        hard.append(
            f"std_ratio={std_ratio:.3f} is outside [0.85, 1.15]. "
            f"Variance stabilisation has FAILED. The AR coefficient change "
            f"is co-producing a marginal variance change, which contaminates "
            f"the label — this window would be indistinguishable from a "
            f"volatility shift. "
            f"Fix: verify sigma_eps = sigma * sqrt(1 - phi^2) is applied on "
            f"BOTH sides of the break in generate_dependence_shift(). "
            f"This was the exact bug in the Cursor implementation."
        )
    elif not (0.90 <= std_ratio <= 1.10):
        soft.append(
            f"std_ratio={std_ratio:.3f} is slightly outside [0.90, 1.10]. "
            f"Variance stabilisation is working but imperfect. "
            f"Document as a known limitation — residual variance leakage "
            f"is small enough to proceed."
        )

    # Δmean — must not change
    if abs(delta_mean) > 0.02:
        hard.append(
            f"Δmean={delta_mean:.5f} is too large for a dependence break. "
            f"Mean is being contaminated. Check that baseline_mu is consistent "
            f"and that the AR recursion does not drift."
        )

    # Δkurt — must stay near zero, stricter than volatility shift
    # because nothing about the innovation distribution changes in a dependence shift
    if abs(delta_kurt) > 0.75:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is elevated for a dependence shift. "
            f"Innovation distribution is identical on both sides of this break — "
            f"any kurtosis difference is finite-sample noise. "
            f"If this persists at larger T, check that sample_innovations() is "
            f"called with the same parameters pre and post break."
        )
    elif abs(delta_kurt) > 0.4:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is slightly elevated — monitor in full corpus."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_distributional_shift(
    bt:         str,
    delta_mean: float,
    std_ratio:  float,
    delta_kurt: float,
    pre_kurt:   float,
) -> SmokeCheckResult:
    """
    Distributional shift: only the tail shape should change.
    Δkurt must be large and positive. std_ratio and Δmean must stay near zero.
    """
    hard, soft = [], []

    # Δkurt — the break must appear as a kurtosis change
    if delta_kurt < 0.5:
        hard.append(
            f"Δkurt={delta_kurt:.2f} is too small. "
            f"The tail shape is not changing across the break. "
            f"Check df_pre and df_post in MAGNITUDE_TABLE and confirm that "
            f"variance normalisation (dividing by sqrt(df/(df-2))) is applied "
            f"correctly in generate_distributional_shift()."
        )
    elif delta_kurt < 1.0:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is present but weak. "
            f"The distributional break is subtle at this magnitude. "
            f"Consider whether Stage 1 will be able to reliably detect it."
        )

    # std_ratio — variance normalisation check
    if not (0.85 <= std_ratio <= 1.15):
        hard.append(
            f"std_ratio={std_ratio:.3f} is outside [0.85, 1.15]. "
            f"Variance normalisation has failed — both t-distributions "
            f"should be rescaled to the same sigma^2. "
            f"Check scale_pre and scale_post calculation."
        )

    # Δmean — must stay near zero
    if abs(delta_mean) > 0.02:
        hard.append(
            f"Δmean={delta_mean:.5f} is too large for a distributional break. "
            f"The mean should be stable. Check that baseline_mu is applied "
            f"symmetrically and t-draws are centred."
        )

    # pre_kurt — sanity check that pre-break is approximately Gaussian
    if pre_kurt > 2.0:
        soft.append(
            f"pre_kurt={pre_kurt:.2f} is higher than expected for t(df=30). "
            f"At large df the t-distribution should approximate Gaussian "
            f"(kurtosis ≈ 0). Finite-sample noise is likely, but check df_pre."
        )

    # Note on effect_size scale — always a soft warn
    soft.append(
        f"effect_size for distributional_shift is on a KL-divergence scale "
        f"and is not directly comparable to other break types (which use "
        f"Cohen's d, log-ratio, or SNR). Document this in the thesis when "
        f"discussing cross-type effect size comparisons."
    )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _estimate_slope(segment: np.ndarray) -> float:
    """
    Fit OLS linear regression on a segment and return the slope.
    slope = cov(t, y) / var(t).

    This is the correct diagnostic for a trending series. Raw std is
    inflated by trend accumulation and is meaningless as a purity check —
    a flat noise series and a strongly trending series can have the same
    innovation std but wildly different raw segment std.
    """
    t = np.arange(len(segment), dtype=float)
    t_mean = t.mean()
    slope = float(
        np.sum((t - t_mean) * (segment - segment.mean())) /
        np.sum((t - t_mean) ** 2)
    )
    return slope


def _check_trend_shift(
    bt:             str,
    pre:            np.ndarray,
    post:           np.ndarray,
    delta_kurt:     float,
    baseline_sigma: float,
    magnitude:      str,
) -> SmokeCheckResult:
    """
    Trend shift: post-break segment has a linear drift.
    The correct diagnostic is slope change, not std_ratio.

    std_ratio is INTENTIONALLY NOT CHECKED here — it will always be large
    because trend accumulation inflates post-break segment std. Checking
    std_ratio would flag every valid trend_shift as broken.

    Instead:
      - pre-break slope must be near zero (flat baseline)
      - post-break slope must be nonzero and match the SNR-calibrated magnitude
      - Δslope = post_slope - pre_slope must be meaningfully large
      - innovation std (computed on detrended residuals) must stay near
        baseline_sigma on both sides — the noise level must not change
    """
    hard, soft = [], []

    pre_slope   = _estimate_slope(pre)
    post_slope  = _estimate_slope(post)
    delta_slope = post_slope - pre_slope

    # Detrend each half to recover the true innovation std
    t_pre  = np.arange(len(pre),  dtype=float)
    t_post = np.arange(len(post), dtype=float)
    pre_resid  = pre  - (pre_slope  * t_pre  + pre.mean()  - pre_slope  * t_pre.mean())
    post_resid = post - (post_slope * t_post + post.mean() - post_slope * t_post.mean())
    pre_resid_std  = float(np.std(pre_resid))
    post_resid_std = float(np.std(post_resid))
    resid_ratio    = post_resid_std / (pre_resid_std + 1e-10)

    # Expected slope magnitude from SNR target in MAGNITUDE_TABLE:
    # slope = SNR * sigma / sqrt(T)
    # For medium magnitude SNR=2, T=1000, sigma=0.01 → slope ≈ 0.000632
    snr = {"small": 1.0, "medium": 2.0, "large": 4.0}[magnitude]
    T   = len(pre) + len(post)
    expected_slope = snr * baseline_sigma / max(np.sqrt(T), 1.0)

    # --- Hard checks ---

    # pre-break slope must be near zero (generator sets slope1 = 0)
    if abs(pre_slope) > 0.5 * expected_slope:
        hard.append(
            f"pre_slope={pre_slope:.6f} is too large — pre-break should be flat "
            f"(slope1=0 in generate_trend_shift()). "
            f"Check that trend_pre is built from slope1=0."
        )

    # post-break slope must be present at approximately the right magnitude
    if abs(post_slope) < 0.3 * expected_slope:
        hard.append(
            f"post_slope={post_slope:.6f} is too small relative to expected "
            f"≈{expected_slope:.6f} (SNR={snr}, magnitude={magnitude}). "
            f"The trend is not being applied. Check slope2 computation "
            f"and that trend_vec is added to the series in generate_trend_shift()."
        )

    # Δslope must be meaningfully nonzero
    if abs(delta_slope) < 0.3 * expected_slope:
        hard.append(
            f"Δslope={delta_slope:.6f} is near zero. "
            f"No slope change detected across the break. "
            f"Expected |Δslope| ≈ {expected_slope:.6f}."
        )

    # Detrended residual std must stay near baseline_sigma on both sides
    if not (0.7 <= resid_ratio <= 1.3):
        hard.append(
            f"Detrended residual std_ratio={resid_ratio:.3f} is outside [0.7, 1.3]. "
            f"The innovation noise level is changing across the trend break — "
            f"this contaminates the label with a co-occurring volatility shift. "
            f"Check that GARCH scaling and sigma_vec are applied uniformly "
            f"across the full series in generate_trend_shift()."
        )
    elif not (0.85 <= resid_ratio <= 1.15):
        soft.append(
            f"Detrended residual std_ratio={resid_ratio:.3f} is slightly off 1.0. "
            f"Innovation noise is mostly stable — acceptable. Monitor in corpus."
        )

    # --- Soft checks ---

    # Document the std_ratio limitation for meta-feature computation
    soft.append(
        f"Raw std_ratio is not checked for trend_shift — it will always be "
        f"large due to trend accumulation in the post-break segment. "
        f"IMPORTANT for meta_features.py: detrend each half before computing "
        f"rolling_std, realized_vol, or any variance-based feature. "
        f"Otherwise Stage 1 will classify trend_shift by high variance rather "
        f"than by slope structure. Document this in the thesis."
    )

    # Δkurt — trend produces platykurtic distributions (negative kurt expected)
    if delta_kurt > 2.0:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is unexpectedly positive. "
            f"Trend shifts typically produce negative or near-zero Δkurt. "
            f"Verify the trend is linear and innovations are Gaussian."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


def _check_no_break(
    bt:          str,
    delta_mean:  float,
    std_ratio:   float,
    delta_kurt:  float,
    tau:         int,
    effect_size: float,
) -> SmokeCheckResult:
    """
    No break: stationary series. tau must be -1, effect_size must be 0.
    All deltas should be near zero — any deviation is finite-sample noise
    and sets the baseline false alarm rate for Stage 2 benchmarking.
    """
    hard, soft = [], []

    # tau must be the sentinel value
    if tau != -1:
        hard.append(
            f"tau={tau} but expected -1 for no_break. "
            f"The sentinel value is being overwritten somewhere in "
            f"generate_no_break() or generate_instance()."
        )

    # effect_size must be exactly zero by definition
    if effect_size != 0.0:
        hard.append(
            f"effect_size={effect_size} but expected exactly 0.0 for no_break. "
            f"Check compute_effect_size() for the no_break case."
        )

    # Δmean must be near zero — no drift in a stationary series
    if abs(delta_mean) > 0.02:
        hard.append(
            f"Δmean={delta_mean:.5f} is too large for a stationary no-break series. "
            f"Something is introducing a spurious drift. "
            f"Check that ar_background is not causing the series to wander "
            f"and that baseline_mu is stable."
        )

    # std_ratio — finite-sample asymmetry is expected, hard fail only on extremes
    if not (0.80 <= std_ratio <= 1.20):
        hard.append(
            f"std_ratio={std_ratio:.3f} is outside [0.80, 1.20] for a stationary series. "
            f"This is implausibly large variance asymmetry in a no-break window. "
            f"Check that no parameters are changing across the window."
        )
    elif not (0.90 <= std_ratio <= 1.10):
        soft.append(
            f"std_ratio={std_ratio:.3f} deviates slightly from 1.0. "
            f"This is finite-sample variance at T=500 per half — expected and acceptable. "
            f"FAR BASELINE NOTE: detectors should not fire on std_ratio this small "
            f"({std_ratio:.3f}). Use this as the lower reference threshold when "
            f"calibrating detection thresholds in Stage 2 benchmarking."
        )

    # Δkurt — should be near zero
    if abs(delta_kurt) > 1.5:
        soft.append(
            f"Δkurt={delta_kurt:.2f} is elevated in the no_break window. "
            f"Kurtosis is noisy at small samples — acceptable."
        )

    return SmokeCheckResult(bt, len(hard) == 0, hard, soft)


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

def validate_smoke_test(
    garch:          bool  = False,
    baseline_sigma: float = 0.01,
    magnitude:      str   = "medium",
    interactive:    bool  = True,
    verbose:        bool  = True,
) -> None:
    """
    Run the smoke test and validate label purity. Replaces demo_all_generators().

    Generates one instance per break type, optionally prints raw pre/post
    stats (verbose=True), applies all label purity checks, prints a
    structured pass/fail report, and either proceeds or raises SystemExit.

    Parameters
    ----------
    garch          : run with GARCH background enabled
    baseline_sigma : baseline noise level (default 0.01)
    magnitude      : break size used for checks (default "medium")
    interactive    : if True, pause after all checks pass and ask for
                     user confirmation before returning
    verbose        : if True, print raw pre/post stats for every break type
                     before the pass/fail verdict — replaces demo_all_generators()

    Raises
    ------
    SystemExit(1)  if any hard failure is detected — corpus generation blocked
    SystemExit(0)  if user declines to continue in interactive mode
    """

    # Late import to avoid circular dependency
    from synthetic_generators_merged import (
        BREAK_TYPES, BreakConfig, generate_instance
    )

    print("\n" + "=" * 65)
    print(f"SMOKE TEST  (GARCH={garch})")
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

        safe_tau   = tau if tau >= 0 else cfg.T // 2
        pre        = s[:safe_tau]
        post       = s[safe_tau:]

        pre_std    = float(np.std(pre))
        delta_mean  = float(np.mean(post)  - np.mean(pre))
        std_ratio   = float(np.std(post)   / (np.std(pre) + 1e-10))
        delta_kurt  = float(stats.kurtosis(post) - stats.kurtosis(pre))
        pre_kurt    = float(stats.kurtosis(pre))
        effect_size = inst["effect_size"]

        # --- Optional raw stats printout (replaces demo_all_generators) ---
        if verbose:
            print(f"\n[{bt}]  tau={tau}  effect_size={inst['effect_size']:.4f}")
            print(f"  pre  → mean={np.mean(pre):+.5f}  std={np.std(pre):.5f}  kurt={stats.kurtosis(pre):.2f}")
            print(f"  post → mean={np.mean(post):+.5f}  std={np.std(post):.5f}  kurt={stats.kurtosis(post):.2f}")
            print(f"  Δmean={delta_mean:+.5f}  std_ratio={std_ratio:.3f}  Δkurt={delta_kurt:.2f}")
            if bt == "trend_shift":
                pre_sl  = _estimate_slope(pre)
                post_sl = _estimate_slope(post)
                print(f"  pre_slope={pre_sl:+.6f}  post_slope={post_sl:+.6f}  Δslope={post_sl - pre_sl:+.6f}")

        if bt == "mean_shift":
            r = _check_mean_shift(
                bt, delta_mean, std_ratio, delta_kurt, baseline_sigma, magnitude
            )
        elif bt == "volatility_shift":
            r = _check_volatility_shift(
                bt, delta_mean, std_ratio, delta_kurt, magnitude, pre_std, baseline_sigma
            )
        elif bt == "dependence_shift":
            r = _check_dependence_shift(bt, delta_mean, std_ratio, delta_kurt)
        elif bt == "distributional_shift":
            r = _check_distributional_shift(
                bt, delta_mean, std_ratio, delta_kurt, pre_kurt
            )
        elif bt == "trend_shift":
            r = _check_trend_shift(
                bt, pre, post, delta_kurt, baseline_sigma, magnitude
            )
        elif bt == "no_break":
            r = _check_no_break(
                bt, delta_mean, std_ratio, delta_kurt, tau, effect_size
            )
        else:
            r = SmokeCheckResult(bt, True, [], [f"No validator defined for {bt}."])

        results.append(r)

    # -------------------------------------------------------------------------
    # Print structured report
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 65)
        print("VALIDATION REPORT")
        print("=" * 65)
    any_hard_fail = any(not r.passed for r in results)
    any_soft_warn = any(r.soft_warns for r in results)

    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        n_warn = len(r.soft_warns)
        warn_label = f"  ({n_warn} warning{'s' if n_warn != 1 else ''})" if n_warn else ""
        print(f"\n  [{status}]  {r.break_type}{warn_label}")

        for msg in r.hard_fails:
            # Wrap long messages at 70 chars for readability
            lines = _wrap(f"✗ HARD FAIL: {msg}", width=70)
            for i, line in enumerate(lines):
                prefix = "           " if i > 0 else "           "
                print(f"{prefix}{line}")

        for msg in r.soft_warns:
            lines = _wrap(f"⚠ WARNING:  {msg}", width=70)
            for i, line in enumerate(lines):
                prefix = "           " if i > 0 else "           "
                print(f"{prefix}{line}")

    print()

    # -------------------------------------------------------------------------
    # Summary and gate
    # -------------------------------------------------------------------------
    if any_soft_warn and not any_hard_fail:
        print("⚠  Soft warnings detected (see above).")
        print("   These are known limitations — document in thesis.")
        print("   Corpus generation can proceed.\n")

    if any_hard_fail:
        failed = [r.break_type for r in results if not r.passed]
        print("=" * 65)
        print(f"✗  HARD FAILURES in: {', '.join(failed)}")
        print("   Label contamination or broken generator detected.")
        print("   Fix the issues above before running the full corpus.")
        print("=" * 65)
        raise SystemExit(1)

    print("✓  All label purity checks passed.")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Interactive confirmation
    # -------------------------------------------------------------------------
    if interactive:
        print("\nReview any warnings above, then confirm:")
        print("  [y / Enter]  Continue to corpus generation")
        print("  [n]          Stop here")
        choice = input("  Your choice: ").strip().lower()
        if choice in ("n", "no"):
            print("\nStopping. Call generate_corpus() manually when ready.")
            raise SystemExit(0)
        print()


# =============================================================================
# HELPERS
# =============================================================================

def _wrap(text: str, width: int = 70) -> list:
    """Simple word-wrap for terminal output."""
    words  = text.split()
    lines  = []
    current = ""
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
