"""
=============================================================================
STAGE 1 CORPUS — stage1_corpus.py
=============================================================================

Generates the training corpus for Stage 1 of the two-stage framework.

Stage 1 is a break-type mixture estimator. Given a window of returns,
it estimates a probability distribution over structural break types,
reflecting the relative prevalence of different break mechanisms in
that window. Stage 2 uses this distribution directly as detector weights.

This file is self-contained. It wraps generate_corpus.py's existing
single-break generators without modifying them. generate_corpus.py
remains the clean single-break corpus for Stage 2 benchmarking.

=============================================================================
DESIGN DECISIONS (record for thesis)
=============================================================================

Decision 1 — Soft labels, not hard labels
------------------------------------------
Each window carries a TARGET PROBABILITY VECTOR over 6 break types:
  [mean_shift, volatility_shift, dependence_shift,
   distributional_shift, trend_shift, no_break]

For a window with n_breaks > 0:
  p_k = count(breaks of type k) / n_breaks

For a window with n_breaks = 0:
  p = [0, 0, 0, 0, 0, 1]   (all mass on no_break)

Rationale: a window with two volatility breaks and one mean break is not
well-described by a single class label. It is genuinely mixed. The
probability vector represents that naturally, keeps all windows in
training, and produces an output that connects directly to Stage 2
weighting without a conversion step.

Decision 2 — Count-based proportions, not effect-size-weighted
---------------------------------------------------------------
Training targets use plain counts. Effect-size adjustment is applied
at inference time as a separate post-processing step in Stage 2:
  w̃_k = (p_k × ẽ_k) / Σ(p_j × ẽ_j)
where ẽ_k is the normalised effect size for break type k.

Rationale: keeping training targets simple and interpretable. The
effect-size weighting is a heuristic on top of a clean estimator,
not a learned quantity. Basically we dont want it to learn effect size
only prob vector based on meta features

Decision 3 — No ambiguity exclusion
------------------------------------
All windows are train-eligible (samples are suited to be included in training) 
regardless of how mixed their break
sequence is. Ambiguity is now part of the target, not noise.

Decision 4 — Split stratification
-----------------------------------
Stratified by (dominant_break_type, n_breaks_bucket) where:
n_breaks_bucket is a categorical grouping used to stratify the dataset splits 
based on the number of breaks in a window. The buckets are defined as follows:
  n_breaks_bucket: "0", "1", "2-3", "4+"

the bucket is a part of split startification strategy to ensure that 
the distribution of break counts is balanced across 
the train, val, test, and robustness_val splits. This helps to prevent any one split from
being dominated by windows with a particular number of breaks, which could bias the training and evaluation of the model.
  ratios: 60% train / 10% val / 10% test / 20% robustness_val

Prevents overrepresenattion of common scenariis (single breaks) and underrepresentation of complex scenarios (multi-break windows) 
improving models generalisation accross diverse break complexities

robustness_val(idation) receives a 20% hash-based sample from every stratum.
The high-n-breaks bucket (4+) provides a natural stress-test (evaluation on challenging, mixed breaks) stratum
within robustness_val — no explicit routing rule needed.

Decision 5 — dominant_break_type retained as a convenience column
------------------------------------------------------------------
Even though Stage 1 trains on soft labels, dominant_break_type (the
argmax of the target vector) is stored in the manifest for filtering,
visualisation, and comparison with the old hard-label approach.

=============================================================================
BREAK TYPE ORDER (fixed — never change after corpus is generated)
=============================================================================
Index  Type
  0    mean_shift
  1    volatility_shift
  2    dependence_shift
  3    distributional_shift
  4    trend_shift
  5    no_break

=============================================================================
Public API
=============================================================================
  Stage1WindowConfig              configuration dataclass
  generate_stage1_window(cfg)     one window
  generate_stage1_corpus(...)     full corpus + split assignment
  get_train(manifest)             convenience accessor
  get_val(manifest)               convenience accessor
  get_test(manifest)              convenience accessor
  get_robustness_val(manifest)    convenience accessor
  summarise_corpus(manifest)      readable breakdown
  smoke_test(n_samples)           statistical validation

Loading a window after corpus generation
-----------------------------------------
  row    = manifest_df.iloc[i]
  series = np.load(f"{output_dir}/series/{row.instance_id}.npy")
  target = json.loads(row.target_vector)   # list of 6 floats, sums to 1.0

=============================================================================
"""

import hashlib
import itertools
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from generate_corpus import (
    BreakConfig,
    BREAK_TYPES,
    generate_instance,
    apply_garch_scaling,
)


# =============================================================================
# §1 — CONSTANTS
# =============================================================================

# Fixed index mapping for the target probability vector.
# This order must never change after corpus generation begins —
# it is the contract between the corpus and the Stage 1 model.
BREAK_TYPE_ORDER = [
    "mean_shift",           # index 0
    "volatility_shift",     # index 1
    "dependence_shift",     # index 2
    "distributional_shift", # index 3
    "trend_shift",          # index 4
    "no_break",             # index 5
]
N_BREAK_TYPES = len(BREAK_TYPE_ORDER)  # 6

# Break types that can appear as individual breaks within a window.
# no_break is excluded: it is the label for n_breaks=0 windows.
ACTIVE_BREAK_TYPES = BREAK_TYPE_ORDER[:5]

POISSON_LAMBDA = 2.0   # expected number of breaks per window
MAX_BREAKS     = 5     # hard ceiling after Poisson draw
MIN_SEG_FRAC   = 0.10  # minimum SEGMENT length as fraction of T, (any inter break segment must be at least this long)
MIN_SEG_ABS    = 50    # absolute minimum regardless of T, lower bound (any inter break segment must be at least this long)

SPLIT_RATIOS = {
    "train":          0.60,
    "val":            0.10,
    "test":           0.10,
    "robustness_val": 0.20,
}


# =============================================================================
# §2 — WINDOW CONFIG
# =============================================================================

@dataclass
class Stage1WindowConfig:
    """
    Configuration for one Stage 1 training window.

    Window-level knobs shared across all segments:
      seed, T, baseline_sigma, sigma_jitter, innovation, t_df,
      ar_background, garch_background, smooth_transition —
      same meaning as in BreakConfig.

    Break-level knobs drawn randomly at generation time:
      magnitude: "small" / "medium" / "large" / "random"
                 "random" draws independently per break from all three.
      poisson_lambda: Poisson rate for break count draw.
      max_breaks: hard ceiling on break count after Poisson draw.
      min_seg_frac: minimum segment length as fraction of T.
    """
    seed:              int   = 42
    T:                 int   = 1000
    baseline_sigma:    float = 0.01
    sigma_jitter:      float = 0.0
    innovation:        str   = "student_t"
    t_df:              float = 5.0
    ar_background:     float = 0.0
    garch_background:  bool  = False
    garch_omega:       float = 1e-6
    garch_alpha:       float = 0.10
    garch_beta:        float = 0.85
    smooth_transition: bool  = False
    transition_width:  int   = 20
    magnitude:         str   = "medium"
    poisson_lambda:    float = POISSON_LAMBDA
    max_breaks:        int   = MAX_BREAKS
    min_seg_frac:      float = MIN_SEG_FRAC


# =============================================================================
# §3 — TARGET VECTOR CONSTRUCTION
# =============================================================================

def build_target_vector(break_types_seq: list) -> list:
    """
    Construct the 6-element target probability vector from a sequence of
    break types in a window.

    Parameters
    ----------
    break_types_seq : list of break type strings for each break in the window.
                      Empty list means n_breaks = 0 (stationary window).

    Returns
    -------
    target : list of 6 floats summing to 1.0, in BREAK_TYPE_ORDER.

    Examples
    --------
    []                                    → [0, 0, 0, 0, 0, 1]  (no_break)
    ["volatility_shift"]                  → [0, 1, 0, 0, 0, 0]
    ["volatility_shift", "mean_shift"]    → [0.5, 0.5, 0, 0, 0, 0]
    ["vol", "vol", "mean"]  (3 breaks)    → [0.333, 0.667, 0, 0, 0, 0]
    """
    if not break_types_seq:
        # Zero-break window: all mass on no_break (index 5)
        vec = [0.0] * N_BREAK_TYPES
        vec[-1] = 1.0
        return vec

    counts = {bt: 0 for bt in BREAK_TYPE_ORDER}
    for bt in break_types_seq:
        if bt in counts:
            counts[bt] += 1
    #output: proportion of each break type in the window, in the fixed order of BREAK_TYPE_ORDER

    n = len(break_types_seq)
    return [counts[bt] / n for bt in BREAK_TYPE_ORDER]


def dominant_type_from_vector(target_vector: list) -> str:
    """
    Return the break type with the highest probability in the target vector.
    Ties broken by first occurrence in BREAK_TYPE_ORDER.
    """
    max_val = max(target_vector)
    return BREAK_TYPE_ORDER[target_vector.index(max_val)]


# =============================================================================
# §4 — INTERNAL HELPERS
# =============================================================================

def _resolve_sigma(cfg: Stage1WindowConfig, rng: np.random.Generator) -> float:
    if cfg.sigma_jitter > 0:
        return max(cfg.baseline_sigma + rng.normal(0, cfg.sigma_jitter), 1e-5)
    return cfg.baseline_sigma


def _allocate_segment_lengths(
    T: int, n_segments: int, min_seg: int, rng: np.random.Generator
) -> tuple:
    """
    Allocate T steps across n_segments segments.

    Each gets at least min_seg points. Remaining budget split via
    Dirichlet(α=1) — uniform over the simplex, all partitions equally likely.

    randomizing segment length is randomizing tau

    ####but note that i need to work more on dirichlet choice, work extension, refer to obsedian stage1corpus####

    Returns (lengths: list[int], feasible: bool).
    feasible=False if T was too short; equal partitioning was used instead.
    """
    budget = T - n_segments * min_seg
    #min_seg = max(MIN_SEG_ABS, int(cfg.min_seg_frac * cfg.T))
    #in allocate-segment_length, each of the n_segments gets atleast min_seg points
    #and budget is the extra length to split among them
    
    if budget < 0:
        base   = T // n_segments
        extra  = T - base * n_segments
        return [base + (1 if i < extra else 0) for i in range(n_segments)], False

    elif budget == 0:
        return [min_seg] * n_segments, True

    else:
        raw    = rng.exponential(1.0, size=n_segments)
        fracs  = raw / raw.sum() #fraction of extra budget assigned to each segment
        extras = np.floor(fracs * budget).astype(int)

        remainder = budget - int(extras.sum())
        if remainder > 0:
            top_idx = np.argsort(fracs * budget - extras)[::-1][:remainder]
            extras[top_idx] += 1

        lengths = [min_seg + int(e) for e in extras]
        diff    = T - sum(lengths)
        if diff != 0:
            lengths[-1] += diff

        return lengths, True


def _stationary_segment(
    T: int, sigma: float, mu: float,
    cfg: Stage1WindowConfig, rng: np.random.Generator,
) -> np.ndarray:
    """
    Stationary AR(1)+GARCH segment of length T at level mu.
    Used for n_breaks=0 windows and the segment0 (pre-break baseline) of multi-break windows.
    A length-T path that is mean mu, has AR(1) dependance
    on the standardised noise, and optionally time-varying volatility via GARCH scaling of sigma
    """

    if cfg.innovation == "student_t":
        raw   = rng.standard_t(cfg.t_df, size=T)
        scale = np.sqrt(cfg.t_df / (cfg.t_df - 2)) if cfg.t_df > 2 else 1.0
        z     = raw / scale
    else:
        z = rng.standard_normal(T)
    
    #Apply AR(1) to z --> z_ar
    z_ar    = np.zeros(T)
    phi     = cfg.ar_background
    z_ar[0] = z[0]
    for t in range(1, T):
        z_ar[t] = phi * z_ar[t - 1] + z[t]

    sigma_vec = np.full(T, sigma)
    if cfg.garch_background:
        proxy = BreakConfig(
            break_type       = "no_break",
            seed             = int(rng.integers(0, 2**31)),
            T                = T,
            baseline_sigma   = sigma,
            garch_background = True,
            garch_omega      = cfg.garch_omega,
            garch_alpha      = cfg.garch_alpha,
            garch_beta       = cfg.garch_beta,
        )
        sigma_vec = apply_garch_scaling(sigma_vec, proxy, rng)

    return mu + sigma_vec * z_ar


# =============================================================================
# §5 — CORE WINDOW GENERATOR
# =============================================================================

def generate_stage1_window(cfg: Stage1WindowConfig) -> dict:
    """
    Generate one Stage 1 training window.

    Steps
    -----
    1.  Draw n_breaks ~ Poisson(λ), clipped to [0, max_breaks].
        Reduce if T is too short to accommodate all segments.
    2.  n_breaks == 0 → stationary window, target = [0,0,0,0,0,1].
    3.  Draw break types and magnitudes independently per break.
    4.  Allocate T across (n_breaks+1) segments via Dirichlet draw.
    5.  Generate each segment:
          Segment 0 — stationary at level mu=0.
          Segment k (k≥1) — via generate_instance() with tau_frac=0.05
          so the whole segment is effectively post-break (the break
          sits at 5% of the segment, the rest is the new regime).
    6.  Continuity adjustment for non-mean-shift / non-trend-shift breaks:
          re-pin segment[0] to the last value of the previous segment,
          preventing spurious level jumps at boundaries where only
          variance, dependence, or tail shape is supposed to change.
    7.  Stitch segments into series of length T.
    8.  Build target vector from break_types_seq via build_target_vector().

    Returns
    -------
    dict with keys:
      series              np.ndarray [T]
      target_vector       list[float]  — 6 floats summing to 1.0
      dominant_break_type str          — argmax of target_vector (convenience)
      n_breaks            int
      tau_list            list[int]    — all breakpoint indices
      tau                 int          — first breakpoint, or -1 if n=0
      break_sequence      list[dict]   — [{tau, break_type, magnitude, effect_size}]
      effect_size         float        — mean effect size across breaks
      config              dict
      instance_id         str
    """
    rng     = np.random.default_rng(cfg.seed)
    sigma   = _resolve_sigma(cfg, rng)
    min_seg = max(MIN_SEG_ABS, int(cfg.min_seg_frac * cfg.T))

    # ------------------------------------------------------------------
    # Step 1: Draw break count
    # ------------------------------------------------------------------
    n_breaks = int(np.clip(rng.poisson(cfg.poisson_lambda), 0, cfg.max_breaks))

    while n_breaks > 0 and (n_breaks + 1) * min_seg > cfg.T: #decrease n_breaks until segments fit
        n_breaks -= 1

    # ------------------------------------------------------------------
    # Step 2: Zero-break window
    # ------------------------------------------------------------------
    if n_breaks == 0:
        series        = _stationary_segment(cfg.T, sigma, 0.0, cfg, rng)
        target_vector = build_target_vector([])   # [0,0,0,0,0,1]

        iid = (
            f"s1w__no_break__T{cfg.T}__n0"
            f"__s{cfg.baseline_sigma}__{cfg.innovation}"
            f"__ar{cfg.ar_background}__g{int(cfg.garch_background)}"
            f"__seed{cfg.seed}"
        )

        return {
            "series":              series,
            "target_vector":       target_vector,
            "dominant_break_type": "no_break",
            "n_breaks":            0,
            "tau_list":            [],
            "tau":                 -1,
            "break_sequence":      [],
            "effect_size":         0.0,
            "config":              asdict(cfg),
            "instance_id":         iid,
            "_seg_lengths":        [cfg.T],
            "_feasible":           True,
        }

    # ------------------------------------------------------------------
    # Step 3: Draw break types and magnitudes
    # ------------------------------------------------------------------
    break_types_seq = [
        str(rng.choice(ACTIVE_BREAK_TYPES)) for _ in range(n_breaks)
    ]

    if cfg.magnitude == "random":
        magnitudes = [
            str(rng.choice(["small", "medium", "large"]))
            for _ in range(n_breaks)
        ]
    else:
        magnitudes = [cfg.magnitude] * n_breaks

    # ------------------------------------------------------------------
    # Step 4: Allocate segment lengths
    # ------------------------------------------------------------------
    seg_lengths, feasible = _allocate_segment_lengths(
        cfg.T, n_breaks + 1, min_seg, rng
    )

    tau_list = []
    pos = 0
    for seg_len in seg_lengths[:-1]:
        pos += seg_len
        tau_list.append(pos)

    # ------------------------------------------------------------------
    # Steps 5–7: Generate and stitch segments
    # ------------------------------------------------------------------
    stitched       = np.zeros(cfg.T)
    break_sequence = []
    effect_sizes   = []
    current_mu     = 0.0
    write_pos      = 0

    for seg_idx in range(n_breaks + 1):
        seg_T = seg_lengths[seg_idx]

        if seg_idx == 0:
            seg = _stationary_segment(seg_T, sigma, current_mu, cfg, rng)

        else:
            bt  = break_types_seq[seg_idx - 1]
            mag = magnitudes[seg_idx - 1]

            # tau_frac=0.05: break sits near the segment start so the
            # segment body is almost entirely the post-break regime.
            seg_cfg = BreakConfig(
                break_type        = bt,
                seed              = int(rng.integers(0, 2**31)),
                T                 = seg_T,
                tau_frac          = 0.05,
                tau_jitter        = 0.0 , #check into changing jitter
                magnitude         = mag,
                baseline_sigma    = sigma,
                sigma_jitter      = 0.0,
                baseline_mu       = current_mu,
                innovation        = cfg.innovation,
                t_df              = cfg.t_df,
                ar_background     = cfg.ar_background,
                garch_background  = cfg.garch_background,
                garch_omega       = cfg.garch_omega,
                garch_alpha       = cfg.garch_alpha,
                garch_beta        = cfg.garch_beta,
                smooth_transition = cfg.smooth_transition,
                transition_width  = cfg.transition_width,
            )

            seg_inst = generate_instance(seg_cfg)
            seg      = seg_inst["series"]
            es       = float(seg_inst["effect_size"])
            effect_sizes.append(es)

            # Step 6: Continuity adjustment.
            # Re-pin first value to current_mu for break types that do
            # not intentionally change the level. This prevents spurious
            # mean-shift signals at volatility/dependence/distributional
            # boundaries.
            if bt not in ("mean_shift", "trend_shift"):
                seg = seg - seg[0] + current_mu

            break_sequence.append({
                "tau":        tau_list[seg_idx - 1],
                "break_type": bt,
                "magnitude":  mag,
                "effect_size": es,
            })

        stitched[write_pos: write_pos + seg_T] = seg #slicing from write_pos to write_pos + seg_T and assigning the segment to the stitched array
        current_mu = float(stitched[write_pos + seg_T - 1])
        write_pos += seg_T

    # ------------------------------------------------------------------
    # Step 8: Target vector and metadata
    # ------------------------------------------------------------------
    target_vector       = build_target_vector(break_types_seq)
    dominant_break_type = dominant_type_from_vector(target_vector)
    mean_es             = float(np.mean(effect_sizes)) if effect_sizes else 0.0

    bt_abbrev = "-".join(b["break_type"][:4] for b in break_sequence)
    iid = (
        f"s1w__{dominant_break_type}__T{cfg.T}__n{n_breaks}"
        f"__seq{bt_abbrev}__mag{cfg.magnitude}"
        f"__s{cfg.baseline_sigma}__{cfg.innovation}"
        f"__ar{cfg.ar_background}__g{int(cfg.garch_background)}"
        f"__seed{cfg.seed}"
    )[:180]

    return {
        "series":              stitched,
        "target_vector":       target_vector,
        "dominant_break_type": dominant_break_type,
        "n_breaks":            n_breaks,
        "tau_list":            tau_list,
        "tau":                 tau_list[0] if tau_list else -1,
        "break_sequence":      break_sequence,
        "effect_size":         mean_es,
        "config":              asdict(cfg),
        "instance_id":         iid,
        "_seg_lengths":        seg_lengths,
        "_feasible":           feasible,
    }


# =============================================================================
# §6 — SPLIT ASSIGNMENT
# =============================================================================

def _hash_to_unit(instance_id: str, salt: str = "") -> float:
    """Deterministic float in [0,1) from MD5 hash of instance_id + salt."""
    h = hashlib.md5((instance_id + salt).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _n_breaks_bucket(n: int) -> str:
    if n == 0:   return "0"
    if n == 1:   return "1"
    if n <= 3:   return "2-3"
    return "4+"


def assign_splits(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """
    Assign a deterministic `split` column to every row.

    Stratified by (dominant_break_type, n_breaks_bucket).
    All windows are train-eligible — no ambiguity exclusion.
    Split assigned by thresholding a hash of instance_id.

    Thresholds (cumulative SPLIT_RATIOS):
      [0.00, 0.60) → train
      [0.60, 0.70) → val
      [0.70, 0.80) → test
      [0.80, 1.00) → robustness_val

    The 4+ bucket appears disproportionately in robustness_val because
    it is a small stratum — its 20% robustness_val slice is still 20%
    of that stratum, giving the most complex windows a natural stress-test
    presence in the evaluation set.
    """
    df   = df.copy()
    df["_nb_bucket"] = df["n_breaks"].apply(_n_breaks_bucket)
    df["split"]      = "unassigned"

    cum = 0.0
    thresholds = []
    for name, ratio in SPLIT_RATIOS.items():
        thresholds.append((name, cum, cum + ratio))
        cum += ratio

    salt = str(seed)

    for (_, _), idx in df.groupby(
        ["dominant_break_type", "_nb_bucket"]
    ).groups.items():
        for row_idx in idx:
            h = _hash_to_unit(str(df.at[row_idx, "instance_id"]), salt=salt)
            for name, lo, hi in thresholds:
                if lo <= h < hi:
                    df.at[row_idx, "split"] = name
                    break

    df.loc[df["split"] == "unassigned", "split"] = "train"
    return df.drop(columns=["_nb_bucket"])


# =============================================================================
# §7 — DIVERSITY GRID AND CORPUS GENERATOR
# =============================================================================

STAGE1_DIVERSITY_GRID = {
    "T":                [500, 1000, 2000],
    "baseline_sigma":   [0.005, 0.01, 0.02],
    "innovation":       ["gaussian", "student_t"],
    "ar_background":    [0.0, 0.3],
    "garch_background": [False, True],
    "smooth_transition":[False, True],
    "magnitude":        ["small", "medium", "large", "random"],
}


def generate_stage1_corpus(
    grid:            dict  = None,
    n_replicates:    int   = 10,
    poisson_lambda:  float = POISSON_LAMBDA,
    max_breaks:      int   = MAX_BREAKS,
    split_seed:      int   = 0,
    output_dir:      str   = "./data/stage1",
    save_series:     bool  = True,
    verbose:         bool  = True,
) -> pd.DataFrame:
    """
    Generate the full Stage 1 training corpus.

    For each (grid combination × replicate):
      - draws n_breaks ~ Poisson(poisson_lambda)
      - generates a multi-break window via generate_stage1_window()
      - optionally saves the raw series as a .npy file
      - records one manifest row

    After all windows are generated, assign_splits() assigns
    deterministic stratified splits in one pass.

    Parameters
    ----------
    grid            : diversity grid (default: STAGE1_DIVERSITY_GRID)
    n_replicates    : windows per grid cell; each replicate uses a
                      different seed, so n_breaks and break types vary
    poisson_lambda  : Poisson rate for break count draw
    max_breaks      : hard ceiling on break count
    split_seed      : integer salt for deterministic split assignment
    output_dir      : root directory for manifest and series files
    save_series     : if True, saves each series to output_dir/series/
    verbose         : print progress

    Returns
    -------
    manifest_df : pd.DataFrame written to
                  {output_dir}/stage1_corpus_manifest.csv

    Manifest columns
    ----------------
    instance_id          unique key
    target_vector        JSON array of 6 floats — Stage 1 training target
    target_vector_dict   JSON dict {break_type: probability} — human-readable
    dominant_break_type  argmax of target_vector — convenience column
    n_breaks             number of breaks drawn
    tau_list             JSON list of breakpoint indices
    break_sequence       JSON list of {tau, break_type, magnitude, effect_size}
    effect_size          mean effect size across all breaks
    split                train / val / test / robustness_val
    [diversity knobs]    T, baseline_sigma, innovation, ar_background, etc.
    """
    if grid is None:
        grid = STAGE1_DIVERSITY_GRID

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    series_dir = os.path.join(output_dir, "series")
    if save_series:
        Path(series_dir).mkdir(parents=True, exist_ok=True)

    grid_keys = list(grid.keys())
    combos    = list(itertools.product(*grid.values()))
    total     = len(combos) * n_replicates

    if verbose:
        print("=" * 65)
        print("STAGE 1 CORPUS GENERATION")
        print("=" * 65)
        print(f"  Grid combos        : {len(combos)}")
        print(f"  Replicates         : {n_replicates}")
        print(f"  Total windows      : {total:,}")
        print(f"  Poisson λ          : {poisson_lambda}")
        print(f"  Max breaks         : {max_breaks}")
        print(f"  Target             : 6-class soft label (BREAK_TYPE_ORDER)")
        print(f"  Output dir         : {output_dir}")
        print("-" * 65)

    rows  = []
    count = 0

    for combo in combos:
        combo_dict = dict(zip(grid_keys, combo))

        for rep in range(n_replicates):
            seed = hash((str(combo_dict), rep, "s1_v2")) % (2**31) #stage1version2

            cfg = Stage1WindowConfig(
                seed              = seed,
                T                 = combo_dict["T"],
                baseline_sigma    = combo_dict["baseline_sigma"],
                innovation        = combo_dict["innovation"],
                ar_background     = combo_dict["ar_background"],
                garch_background  = combo_dict["garch_background"],
                smooth_transition = combo_dict["smooth_transition"],
                magnitude         = combo_dict["magnitude"],
                poisson_lambda    = poisson_lambda,
                max_breaks        = max_breaks,
            )

            inst = generate_stage1_window(cfg)

            if save_series:
                np.save(
                    os.path.join(series_dir, f"{inst['instance_id']}.npy"),
                    inst["series"].astype(np.float32),
                )

            # Human-readable dict alongside the array
            tv      = inst["target_vector"] #tv --> target vector
            tv_dict = {bt: round(tv[i], 6) for i, bt in enumerate(BREAK_TYPE_ORDER)}

            rows.append({
                "instance_id":          inst["instance_id"],
                "target_vector":        json.dumps([round(x, 6) for x in tv]),
                "target_vector_dict":   json.dumps(tv_dict),
                "dominant_break_type":  inst["dominant_break_type"],
                "n_breaks":             inst["n_breaks"],
                "tau":                  inst["tau"],
                "tau_list":             json.dumps(inst["tau_list"]),
                "break_sequence":       json.dumps(inst["break_sequence"]),
                "effect_size":          round(inst["effect_size"], 6),
                "T":                    combo_dict["T"],
                "baseline_sigma":       combo_dict["baseline_sigma"],
                "innovation":           combo_dict["innovation"],
                "ar_background":        combo_dict["ar_background"],
                "garch_background":     combo_dict["garch_background"],
                "smooth_transition":    combo_dict["smooth_transition"],
                "magnitude":            combo_dict["magnitude"],
                "seed":                 seed,
                "replicate":            rep,
                "seg_lengths":          json.dumps(inst["_seg_lengths"]),
                "feasible":             inst["_feasible"],
            })

            #progress logging while the corpus loop runs
            count += 1 #number of windows completed so far 1,2, ..., total
            if verbose and count % 500 == 0: #print progress every 500 windows
                print(
                    f"  [{count:>6}/{total}]  "
                    f"n={inst['n_breaks']}  "
                    f"dominant={inst['dominant_break_type']:<22}  "
                    f"target={[round(x,2) for x in tv]}"
                )

    """
    Series is saved as individual .npy files in the series directory
    each .npy file contains one 1D array of length T representing the time series
    if you want to load the series into memory, you can use np.load(os.path.join(series_dir, f"{inst['instance_id']}.npy"))
    if you want a time index, you can use np.arange(T)

    the manifest dataframe contains the metadata for each window
    the manifest dataframe is saved as a csv file in the output directory
    the manifest dataframe is used to load the series files into memory
    the manifest dataframe is used to assign splits to the windows
    the manifest dataframe is used to load the series files into memory   
    """

    manifest_df = pd.DataFrame(rows) #dataframe for metadata of all windows
    manifest_df = assign_splits(manifest_df, seed=split_seed)

    manifest_path = os.path.join(output_dir, "stage1_corpus_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    if verbose:
        print("-" * 65)
        print(f"Done. {count:,} windows generated.")
        print(f"Manifest: {manifest_path}\n")
        summarise_corpus(manifest_df)

    return manifest_df


# =============================================================================
# §8 — CONVENIENCE ACCESSORS
# =============================================================================

def get_train(manifest: pd.DataFrame) -> pd.DataFrame:
    """Training rows. All windows eligible — no ambiguity exclusion."""
    return manifest[manifest["split"] == "train"].reset_index(drop=True)


def get_val(manifest: pd.DataFrame) -> pd.DataFrame:
    """Validation rows for hyperparameter tuning."""
    return manifest[manifest["split"] == "val"].reset_index(drop=True)


def get_test(manifest: pd.DataFrame) -> pd.DataFrame:
    """Held-out test rows for final evaluation."""
    return manifest[manifest["split"] == "test"].reset_index(drop=True)


def get_robustness_val(manifest: pd.DataFrame) -> pd.DataFrame:
    """
    Robustness evaluation rows (20% of each stratum).
    High-n-breaks windows (4+) are naturally concentrated here.
    Compare Stage 1 performance here vs get_test() to quantify
    degradation under complex multi-break conditions.
    """
    return manifest[manifest["split"] == "robustness_val"].reset_index(drop=True)


# =============================================================================
# §9 — SUMMARY PRINTER
# =============================================================================

def summarise_corpus(manifest_df: pd.DataFrame) -> None:
    """Print a readable breakdown of the assembled corpus."""
    total = len(manifest_df)
    print(f"Total windows : {total:,}")

    print("\nBreak count distribution:")
    from scipy.stats import poisson as _p

    #break count distribution
    for n, cnt in manifest_df["n_breaks"].value_counts().sort_index().items(): #(n_breaks, count(each break))
        
        #empirical, observed probablity of each break count, with a split for the top bucket
        pct = 100 * cnt / total

        #theoretical expected probablity under poisson design, with a split for the top bucket
        exp = 100 * (
            _p.pmf(n, POISSON_LAMBDA) if n < MAX_BREAKS
            else 1 - _p.cdf(MAX_BREAKS - 1, POISSON_LAMBDA)
        )
        print(f"  n={n}: {cnt:>7,}  ({pct:.1f}%)  expected ≈ {exp:.1f}%")

    #dominant break type distribution
    print("\nDominant break type distribution:")
    for bt, cnt in manifest_df["dominant_break_type"].value_counts().items():
        print(f"  {bt:<25} {cnt:>7,}  ({100*cnt/total:.1f}%)")

    #split distribution
    print("\nSplit distribution:")
    for sp, cnt in manifest_df["split"].value_counts().sort_index().items():
        print(f"  {sp:<20} {cnt:>7,}  ({100*cnt/total:.1f}%)")

    #mean target vector across all windows
    print("\nMean target vector across all windows:")
    all_targets = manifest_df["target_vector"].apply(json.loads)
    mean_vec    = np.array(all_targets.tolist()).mean(axis=0)
    for i, bt in enumerate(BREAK_TYPE_ORDER):
        print(f"  {bt:<25} {mean_vec[i]:.4f}")
    print(f"  (should be ~uniform across active types + ~{100*np.exp(-POISSON_LAMBDA):.0f}% on no_break)")

    #break type × split distribution
    #helps detect imbalance in splits
    print("\nBreak type × split:")
    ct = pd.crosstab(manifest_df["dominant_break_type"], manifest_df["split"])
    print(ct.to_string())


# =============================================================================
# §10 — SMOKE TEST
# =============================================================================

def smoke_test(n_samples: int = 300, verbose: bool = True) -> None:
    """
    Validate statistical properties of generate_stage1_window().

    Checks
    ------
    Vector sum       : target_vector sums to 1.0 (±1e-9) for every window.
    Zero-break rule  : n_breaks=0 windows have target_vector[5] == 1.0.
    Single-break rule: n_breaks=1 windows have exactly one entry == 1.0.
    Length check     : len(series) == cfg.T for every window.
    Finite check     : no NaN or Inf in any series.
    Tau list check   : len(tau_list) == n_breaks.
    Seg length sum   : sum(seg_lengths) == T.
    Continuity check : |series[tau] - series[tau-1]| < 10*sigma for
                       non-mean-shift / non-trend-shift breaks.
    Distribution     : observed n_breaks distribution matches Poisson(2).
    """
    print("=" * 65)
    print(f"STAGE 1 WINDOW SMOKE TEST  (n={n_samples})")
    print("=" * 65)

    rng_m    = np.random.default_rng(0)
    seeds    = rng_m.integers(0, 2**31, size=n_samples)
    failures = []
    n_counts = {}

    for seed in seeds:
        cfg  = Stage1WindowConfig(seed=int(seed), T=1000, baseline_sigma=0.01)
        inst = generate_stage1_window(cfg)
        nb   = inst["n_breaks"]
        tv   = inst["target_vector"]

        # Vector sum
        vec_sum = sum(tv)
        if abs(vec_sum - 1.0) > 1e-9:
            failures.append(f"seed={seed}: target_vector sums to {vec_sum:.8f}")

        # Zero-break rule
        if nb == 0 and tv[5] != 1.0:
            failures.append(f"seed={seed}: n=0 but no_break entry = {tv[5]}")

        # Single-break rule
        if nb == 1:
            ones = [x for x in tv if x == 1.0]
            if len(ones) != 1:
                failures.append(f"seed={seed}: n=1 but target not a unit vector: {tv}")

        # Series length
        if len(inst["series"]) != cfg.T:
            failures.append(f"seed={seed}: length {len(inst['series'])} != {cfg.T}")

        # Finite
        if not np.isfinite(inst["series"]).all():
            failures.append(f"seed={seed}: NaN or Inf in series")

        # Tau list length
        if len(inst["tau_list"]) != nb:
            failures.append(f"seed={seed}: len(tau_list)={len(inst['tau_list'])} != {nb}")

        # Segment length sum
        if sum(inst["_seg_lengths"]) != cfg.T:
            failures.append(
                f"seed={seed}: sum(seg_lengths)={sum(inst['_seg_lengths'])} != T={cfg.T}"
            )

        # Continuity
        for bk in inst["break_sequence"]:
            if bk["break_type"] not in ("mean_shift", "trend_shift"):
                tau = bk["tau"]
                if 0 < tau < cfg.T:
                    jump = abs(inst["series"][tau] - inst["series"][tau - 1])
                    if jump > 10 * cfg.baseline_sigma:
                        failures.append(
                            f"seed={seed}: jump={jump:.5f} at tau={tau} "
                            f"for {bk['break_type']}"
                        )

        n_counts[nb] = n_counts.get(nb, 0) + 1

    # Distribution check
    from scipy.stats import poisson as _p
    print(f"\nBreak count distribution (n={n_samples}):")
    for k in sorted(n_counts):
        obs = n_counts[k]
        pct = 100 * obs / n_samples
        exp = 100 * (
            _p.pmf(k, POISSON_LAMBDA) if k < MAX_BREAKS
            else 1 - _p.cdf(MAX_BREAKS - 1, POISSON_LAMBDA)
        )
        flag = "  ← check" if abs(pct - exp) > 6 else ""
        print(f"  n={k}: {obs:4d} ({pct:.1f}%)  expected ≈ {exp:.1f}%{flag}")

    if failures:
        print(f"\n[FAILURES] {len(failures)} issues:")
        for msg in failures[:10]:
            print(f"  {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")
    else:
        print(f"\n[PASS] All {n_samples} windows passed all checks.")

    print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Step 1: Validate the generator
    smoke_test(n_samples=300)

    # Step 2: Small corpus for rapid validation
    # For full run: grid=STAGE1_DIVERSITY_GRID, n_replicates=10
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
        grid           = small_grid,
        n_replicates   = 5,
        poisson_lambda = 2.0,
        max_breaks     = 5,
        split_seed     = 0,
        output_dir     = "./data/stage1",
        save_series    = True,
        verbose        = True,
    )

    print(f"\nAccessors:")
    print(f"  get_train()          {len(get_train(manifest)):>6,} rows")
    print(f"  get_val()            {len(get_val(manifest)):>6,} rows")
    print(f"  get_test()           {len(get_test(manifest)):>6,} rows")
    print(f"  get_robustness_val() {len(get_robustness_val(manifest)):>6,} rows")

    # Show sample target vectors
    print(f"\nSample target vectors:")
    sample = manifest[["dominant_break_type", "n_breaks",
                        "target_vector"]].head(8)
    for _, row in sample.iterrows():
        tv = json.loads(row["target_vector"])
        print(f"  n={row['n_breaks']}  {row['dominant_break_type']:<22}  "
              f"{[round(x, 2) for x in tv]}")
