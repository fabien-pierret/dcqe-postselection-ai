#!/usr/bin/env python3
"""Experiment 7: Numerical Proofs of the Two Critical PhD Jury Traps
(Detection Loophole in Bell-CHSH/LLM Tests & Quantum Grover Data-Dredging Paradox).

This self-contained, deterministic script implements and numerically verifies
two foundational traps raised in Chapter 5 (Jury Questions 1 and 5):

Part A — Detection Loophole & Selective Refusal Bias (Pearle 1970; Garg & Mermin 1987;
Eberhard 1993; Larsson 1998):
  - Constructs an explicit, 100% classical Local Hidden-Variable (LHV) model with
    ternary outcomes {-1, +1, empty} where `empty` models photon loss or LLM refusal /
    rejection-sampling filtering.
  - Strictly preserves no-signaling, zero single-side marginals, setting-independent
    single-detection efficiency P(A != empty | x) = P(B != empty | y) = eta, and
    quantum-consistent coincidence rate P(coinc | x, y) = eta^2 across all eta in [0.50, 1.00].
  - Sweeps eta in [0.50, 1.00] both analytically and by Monte Carlo (N = 200,000
    pairs per efficiency point) to prove:
      (1) Post-selecting on joint detection (collider bias) drives a purely classical
          LHV source above |S| = 2 for all eta < 1, matching Tsirelson's quantum bound
          2*sqrt(2) ~= 2.8284 at the exact symmetric threshold eta_sym* = 2/(1+sqrt(2))
          ~= 82.8427%, and saturating the algebraic PR-box ceiling S = 4.0 for eta <= 2/3.
      (2) Closing the loophole via deterministic outcome assignment (empty -> +1, mandatory
          in any loophole-free Bell or LLM benchmark) collapses the exact same classical
          source immediately back to |S| <= 2.0000 for all eta in [0.50, 1.00].
      (3) Compares against an entangled quantum Bell state |Phi+> under independent
          detector efficiency eta, whose loophole-free CHSH S_QM_filled(eta) =
          2*sqrt(2)*eta^2 + 2*(1-eta)^2 crosses S = 2 at the exact same threshold eta_sym*.

Part B — Quantum Grover Amplitude Amplification + Data-Dredging Paradox
(Grover 1996; Boyer et al. 1998; Brassard et al. 2002; Bailey & Lopez de Prado 2014):
  - Part B.1: Exact unitary state-vector simulation of Grover's algorithm on q = 4..12
    qubits (M = 2^q in {16, ..., 4096} candidate strategies evaluated on pure i.i.d.
    white noise, T_IS = T_OOS = 252). Verifies unitarity (||psi|| = 1 to < 1e-14),
    exact match to analytical sin^2((2r+1)*theta) (< 1e-14), and near-unit success
    probability at R_Grover = floor((pi/4)*sqrt(M)) oracle queries.
  - Part B.2: Fixed oracle-query budget sweep C in {10, 30, 100, 300, 1000} (R = 80
    Monte Carlo trials per budget) comparing Classical Search (M_cl = C) against
    Quantum Grover Search (M_quant = floor((4*C/pi)^2)). Proves that:
      (1) At equal oracle budget C, Grover amplifies the apparent In-Sample Sharpe
          mirage by approximately sqrt(2) (+41%) relative to classical search;
      (2) Real Out-of-Sample Sharpe collapses identically to ~0.00 for both methods;
      (3) Deflating the Sharpe ratio (Bailey & Lopez de Prado 2014) using the quantum
          query count N = C fails catastrophically (false-positive rate exploding as C grows),
          whereas deflating by the true effective hypothesis space N = M_eff eliminates
          the mirage completely.

Hardware target: Apple Silicon (macOS ARM64), pure CPU NumPy/SciPy/Matplotlib.
Deterministic global seed: 42.
"""

from __future__ import annotations

import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless reproducibility
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Global reproducibility, paths, and publication aesthetic configuration
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

RESULTS_JSON_PATH = RESULTS_DIR / "exp7_jury_traps.json"
FIGURE_PATH = FIGURES_DIR / "fig7_detection_loophole_and_grover_dredging.png"

# Cohesive colorblind-safe publication palette (aligned with verify_dcqe_ai.py)
PALETTE = {
    "navy": "#1B365D",
    "coral": "#C83E3B",
    "teal": "#1B9E77",
    "amber": "#D97706",
    "purple": "#6B4C9A",
    "slate": "#5A6772",
    "light_gray": "#CBD5E1",
    "bg_tint": "#F8FAFC",
}


def configure_matplotlib() -> None:
    """Configure publication-grade 300 DPI Matplotlib styling."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "legend.frameon": True,
            "legend.framealpha": 0.93,
            "legend.edgecolor": "#CBD5E1",
            "grid.color": "#94A3B8",
            "grid.linestyle": "--",
            "grid.alpha": 0.28,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


# ===========================================================================
# PART A: Detection Loophole & Selective Refusal Bias (Bell-CHSH / LLM)
# ===========================================================================
# Local deterministic strategy tables (outcomes in {-1, +1, 0}, where 0 = empty / undetected).
# Columns correspond to (A0, A1, B0, B1) before multiplying by a global sign flip sigma in {-1, +1}.
#
# 1) Always-detecting CHSH-optimal vertices (4 strategies, S = 2, P_single = 1, P_coinc = 1):
STRAT_ALL_DETECT = np.array(
    [
        [+1, +1, +1, +1],  # wrong on (1,1)
        [+1, -1, +1, +1],  # wrong on (1,0)
        [+1, +1, +1, -1],  # wrong on (0,1)
        [-1, +1, +1, -1],  # wrong on (0,0)
    ],
    dtype=np.int8,
)

# 2) Single-omission strategies (4 strategies, S_post = 4, P_single = 3/4, P_coinc = 1/2):
STRAT_ONE_MISS = np.array(
    [
        [+1,  0, +1, +1],  # A1 = empty; coincides on (0,0):+1, (0,1):+1
        [ 0, +1, +1, -1],  # A0 = empty; coincides on (1,0):+1, (1,1):-1
        [+1, +1, +1,  0],  # B1 = empty; coincides on (0,0):+1, (1,0):+1
        [+1, -1,  0, +1],  # B0 = empty; coincides on (0,1):+1, (1,1):-1
    ],
    dtype=np.int8,
)

# 3) Double-omission strategies (4 strategies, S_post = 4, P_single = 1/2, P_coinc = 1/4):
STRAT_TWO_MISS = np.array(
    [
        [+1,  0, +1,  0],  # coincides only on (0,0):+1
        [+1,  0,  0, +1],  # coincides only on (0,1):+1
        [ 0, +1, +1,  0],  # coincides only on (1,0):+1
        [ 0, +1,  0, -1],  # coincides only on (1,1):-1
    ],
    dtype=np.int8,
)

# 4) Null-detection strategy (neither side detects on any setting):
STRAT_ZERO_DETECT = np.array([[0, 0, 0, 0]], dtype=np.int8)

# Unified 13-row lookup table of base strategies
ALL_BASE_STRATEGIES = np.vstack(
    [STRAT_ALL_DETECT, STRAT_ONE_MISS, STRAT_TWO_MISS, STRAT_ZERO_DETECT]
)


def lhv_mixture_weights(eta: float) -> tuple[float, float, float, float]:
    """Return exact nonnegative family weights (p_all, p_1miss, p_2miss, p_0det).

    For every eta in [0.5, 1.0], the resulting classical LHV source satisfies:
      - P(A_x != empty) = P(B_y != empty) = eta          for all x, y in {0, 1},
      - P(A_x != empty and B_y != empty) = eta^2         for all (x, y) in {0, 1}^2,
      - E[A_x | A_x != empty] = E[B_y | B_y != empty] = 0 (zero marginal bias),
      - Strict locality & no-signaling.
    """
    if eta < 0.5 - 1e-12 or eta > 1.0 + 1e-12:
        raise ValueError(f"eta must lie in [0.5, 1.0], got {eta}")
    eta = float(np.clip(eta, 0.5, 1.0))

    if eta >= 2.0 / 3.0:
        p_all = eta * (3.0 * eta - 2.0)
        p_1 = 4.0 * eta * (1.0 - eta)
        p_2 = 0.0
        p_0 = (1.0 - eta) ** 2
    else:
        p_all = 0.0
        p_1 = 4.0 * eta * (2.0 * eta - 1.0)
        p_2 = 4.0 * eta * (2.0 - 3.0 * eta)
        p_0 = (1.0 - 2.0 * eta) ** 2

    # Clip tiny floating-point negatives at regime boundaries and renormalize
    weights = np.clip(np.array([p_all, p_1, p_2, p_0], dtype=float), 0.0, 1.0)
    weights /= np.sum(weights)
    return float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3])


def analytic_lhv_chsh_post(eta: float) -> float:
    """Exact post-selected CHSH value of the symmetric LHV model at efficiency eta."""
    if eta >= 2.0 / 3.0:
        return float(4.0 / eta - 2.0)
    return 4.0


def analytic_lhv_chsh_filled(eta: float) -> float:
    """Exact loophole-free CHSH value under deterministic assignment (empty -> +1)."""
    _, _, p_2, _ = lhv_mixture_weights(eta)
    # Every family yields S_filled = 2.0 except STRAT_TWO_MISS which averages to 1.5
    return float(2.0 - 0.5 * p_2)


def analytic_qm_chsh_filled(eta: float) -> float:
    """Exact loophole-free CHSH for |Phi+> with independent efficiency eta and empty -> +1."""
    return float(2.0 * math.sqrt(2.0) * (eta ** 2) + 2.0 * ((1.0 - eta) ** 2))


def simulate_lhv_point(
    eta: float, n_pairs: int, rng: np.random.Generator
) -> dict[str, float]:
    """Monte Carlo simulation of N pairs from the local hidden-variable source.

    Each pair draws a hidden variable lambda = (strategy_id, global_sign) and
    independent uniform random settings (x, y) in {0, 1}^2. Outcomes are generated
    strictly locally: A = A(x, lambda), B = B(y, lambda).
    """
    p_all, p_1, p_2, p_0 = lhv_mixture_weights(eta)
    probs_13 = np.array(
        [p_all / 4.0] * 4 + [p_1 / 4.0] * 4 + [p_2 / 4.0] * 4 + [p_0],
        dtype=float,
    )
    probs_13 /= np.sum(probs_13)

    # Sample hidden variables and independent measurement settings
    strat_idx = rng.choice(13, size=n_pairs, p=probs_13)
    sigma = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_pairs)
    x = rng.integers(0, 2, size=n_pairs, dtype=np.int8)
    y = rng.integers(0, 2, size=n_pairs, dtype=np.int8)

    # Strictly local lookup: Alice sees ONLY x and lambda; Bob sees ONLY y and lambda
    base_rows = ALL_BASE_STRATEGIES[strat_idx]  # shape (n_pairs, 4)
    a_raw = base_rows[np.arange(n_pairs), x] * sigma
    b_raw = base_rows[np.arange(n_pairs), 2 + y] * sigma

    # Deterministic loophole-free filling: empty (0) -> +1
    a_filled = np.where(a_raw == 0, 1, a_raw).astype(np.int8)
    b_filled = np.where(b_raw == 0, 1, b_raw).astype(np.int8)

    # Compute correlators and efficiencies for all 4 setting pairs
    e_post = np.zeros(4, dtype=float)
    e_filled = np.zeros(4, dtype=float)
    coinc_rates = np.zeros(4, dtype=float)
    single_a_rates = np.zeros(2, dtype=float)
    single_b_rates = np.zeros(2, dtype=float)
    marginal_a = np.zeros(2, dtype=float)
    marginal_b = np.zeros(2, dtype=float)

    for xv in (0, 1):
        mask_x = x == xv
        det_a = a_raw[mask_x] != 0
        single_a_rates[xv] = float(np.mean(det_a))
        marginal_a[xv] = float(np.mean(a_raw[mask_x][det_a])) if np.any(det_a) else 0.0

    for yv in (0, 1):
        mask_y = y == yv
        det_b = b_raw[mask_y] != 0
        single_b_rates[yv] = float(np.mean(det_b))
        marginal_b[yv] = float(np.mean(b_raw[mask_y][det_b])) if np.any(det_b) else 0.0

    for idx_pair, (xv, yv) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        mask = (x == xv) & (y == yv)
        a_sub = a_raw[mask]
        b_sub = b_raw[mask]
        coinc = (a_sub != 0) & (b_sub != 0)
        coinc_rates[idx_pair] = float(np.mean(coinc))
        e_post[idx_pair] = float(np.mean((a_sub[coinc] * b_sub[coinc]).astype(float)))
        e_filled[idx_pair] = float(
            np.mean((a_filled[mask] * b_filled[mask]).astype(float))
        )

    s_post = float(e_post[0] + e_post[1] + e_post[2] - e_post[3])
    s_filled = float(e_filled[0] + e_filled[1] + e_filled[2] - e_filled[3])

    return {
        "eta": float(eta),
        "p_all": p_all,
        "p_1miss": p_1,
        "p_2miss": p_2,
        "p_0det": p_0,
        "s_post_mc": s_post,
        "s_post_analytic": analytic_lhv_chsh_post(eta),
        "s_filled_mc": s_filled,
        "s_filled_analytic": analytic_lhv_chsh_filled(eta),
        "s_qm_filled_analytic": analytic_qm_chsh_filled(eta),
        "abs_err_post": abs(s_post - analytic_lhv_chsh_post(eta)),
        "abs_err_filled": abs(s_filled - analytic_lhv_chsh_filled(eta)),
        "e_post": [float(v) for v in e_post],
        "e_filled": [float(v) for v in e_filled],
        "single_efficiency_mc_mean": float(
            0.5 * (np.mean(single_a_rates) + np.mean(single_b_rates))
        ),
        "single_efficiency_mc_max_err": float(
            max(
                np.max(np.abs(single_a_rates - eta)),
                np.max(np.abs(single_b_rates - eta)),
            )
        ),
        "coinc_rate_mc_mean": float(np.mean(coinc_rates)),
        "coinc_rate_mc_max_err": float(np.max(np.abs(coinc_rates - eta ** 2))),
        "max_abs_marginal_bias": float(
            max(np.max(np.abs(marginal_a)), np.max(np.abs(marginal_b)))
        ),
    }


def run_part_a_detection_loophole(rng: np.random.Generator) -> dict[str, Any]:
    """Run Part A: complete analytic and Monte Carlo audit of the Detection Loophole."""
    eta_sym_star = 2.0 / (1.0 + math.sqrt(2.0))  # ~0.8284271247461903
    eta_eberhard = 2.0 / 3.0                     # ~0.6666666666666666
    tsirelson = 2.0 * math.sqrt(2.0)

    # Grid including exact critical thresholds 0.50, 2/3, eta_sym*, and 1.00
    eta_grid = np.array(
        [
            0.50,
            0.55,
            0.60,
            0.64,
            eta_eberhard,
            0.70,
            0.74,
            0.78,
            eta_sym_star,
            0.86,
            0.90,
            0.94,
            0.97,
            1.00,
        ],
        dtype=float,
    )
    # 200,000 pairs per (x, y) setting pair => 800,000 independent pairs per eta point
    # (ensuring 1-sigma SE(S) <= 0.004 and max MC deviation < 0.01 across all 14 points).
    n_pairs_per_setting = 200_000
    n_pairs_per_point = 4 * n_pairs_per_setting

    sweep_records = [
        simulate_lhv_point(float(eta), n_pairs_per_point, rng) for eta in eta_grid
    ]

    # Locate exact threshold records
    rec_half = sweep_records[0]
    rec_eberhard = sweep_records[4]
    rec_sym_star = sweep_records[8]
    rec_unit = sweep_records[-1]

    max_err_post = max(r["abs_err_post"] for r in sweep_records)
    max_err_filled = max(r["abs_err_filled"] for r in sweep_records)
    max_filled_mc = max(abs(r["s_filled_mc"]) for r in sweep_records)
    max_single_err = max(r["single_efficiency_mc_max_err"] for r in sweep_records)
    max_coinc_err = max(r["coinc_rate_mc_max_err"] for r in sweep_records)

    # Strict mathematical & numerical verification assertions
    assert max_err_post < 0.01, f"MC post-selected error too large: {max_err_post}"
    assert max_err_filled < 0.01, f"MC loophole-free error too large: {max_err_filled}"
    assert abs(rec_sym_star["s_post_analytic"] - tsirelson) < 1e-14
    assert abs(rec_sym_star["s_post_mc"] - tsirelson) < 0.01
    assert abs(rec_sym_star["s_qm_filled_analytic"] - 2.0) < 1e-14
    assert abs(rec_eberhard["s_post_analytic"] - 4.0) < 1e-14
    assert abs(rec_eberhard["s_post_mc"] - 4.0) < 1e-12  # Exact deterministic 4.0 on all coincidences!
    assert abs(rec_unit["s_post_mc"] - 2.0) < 0.01
    assert max_filled_mc <= 2.01, f"Loophole-free CHSH exceeded 2.0: {max_filled_mc}"
    # Analytic loophole-free check: strictly <= 2.0 everywhere
    assert max(r["s_filled_analytic"] for r in sweep_records) <= 2.0 + 1e-15

    # Also verify by exact enumeration of all 13*2 = 26 local deterministic configurations
    # that EVERY individual local deterministic realization satisfies |S_filled| <= 2.
    indiv_filled_s = []
    for row in ALL_BASE_STRATEGIES:
        for sig in (-1, 1):
            r_sig = row * sig
            rf = np.where(r_sig == 0, 1, r_sig)
            s_val = int(rf[0] * rf[2] + rf[0] * rf[3] + rf[1] * rf[2] - rf[1] * rf[3])
            indiv_filled_s.append(s_val)
    assert max(abs(v) for v in indiv_filled_s) == 2

    return {
        "n_pairs_per_eta": n_pairs_per_point,
        "eta_sym_star": eta_sym_star,
        "eta_eberhard_saturation": eta_eberhard,
        "tsirelson_bound": tsirelson,
        "max_mc_err_post_selected": float(max_err_post),
        "max_mc_err_loophole_free": float(max_err_filled),
        "max_mc_loophole_free_abs_s": float(max_filled_mc),
        "max_single_efficiency_err": float(max_single_err),
        "max_coinc_rate_err": float(max_coinc_err),
        "exact_individual_realization_max_abs_s_filled": 2,
        "highlights": {
            "eta_0.50": rec_half,
            "eta_2_over_3": rec_eberhard,
            "eta_sym_star": rec_sym_star,
            "eta_1.00": rec_unit,
        },
        "sweep": sweep_records,
    }


# ===========================================================================
# PART B: Quantum Grover Amplitude Amplification + Data-Dredging Paradox
# ===========================================================================
def expected_max_normal_blp(m: int) -> float:
    """Bailey & Lopez de Prado (2014) extreme-value approximation for E[max_{1..M} Z_i]."""
    if m <= 1:
        return 0.0
    euler_gamma = 0.5772156649015329
    return float(
        (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / m)
        + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (m * math.e))
    )


def deflated_sharpe_ratio(
    sr_annualized: float,
    n_trials_assumed: int,
    t_obs: int = 252,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_std_across_trials: float = 1.0,
) -> float:
    """Compute Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR).

    Converts annualized Sharpe ratios to per-period ratios (dividing by sqrt(t_obs))
    to match Eq. (2) of Bailey & Lopez de Prado (2014) exactly:
      DSR = Phi( (SR_1 - SR_{0,1}) * sqrt(T - 1) / sqrt(1 - g3*SR_1 + ((g4-1)/4)*SR_1^2) ).
    """
    sr_0_ann = sr_std_across_trials * expected_max_normal_blp(n_trials_assumed)
    sr_1 = sr_annualized / math.sqrt(t_obs)
    sr_0_1 = sr_0_ann / math.sqrt(t_obs)
    denom_sq = 1.0 - skewness * sr_1 + 0.25 * (kurtosis - 1.0) * (sr_1 ** 2)
    denom = math.sqrt(max(denom_sq, 1e-12))
    z_val = (sr_1 - sr_0_1) * math.sqrt(t_obs - 1.0) / denom
    return float(stats.norm.cdf(z_val))


def simulate_exact_unitary_grover(
    q: int, target_index: int, n_iterations: int
) -> tuple[np.ndarray, float, float]:
    """Exact unitary state-vector simulation of Grover's algorithm on q qubits (M = 2^q).

    Starts from |s> = H^{otimes q}|0>^{otimes q} and applies G = (2|s><s| - I) O_{m*}
    exactly for n_iterations steps. Returns:
      - prob_trace: array of length (n_iterations + 1) with P_succ(r) = |<m*|G^r|s>|^2,
      - max_unitarity_err: max_r | ||G^r|s>||_2 - 1 |,
      - max_analytic_err: max_r | P_succ(r) - sin^2((2r + 1)*arcsin(1/sqrt(M))) |.
    """
    m = 1 << q
    psi = np.full(m, 1.0 / math.sqrt(m), dtype=np.complex128)
    theta = math.asin(1.0 / math.sqrt(m))

    prob_trace = np.zeros(n_iterations + 1, dtype=float)
    prob_trace[0] = float(abs(psi[target_index]) ** 2)

    max_norm_err = abs(float(np.linalg.norm(psi)) - 1.0)
    max_analytic_err = abs(prob_trace[0] - (math.sin(theta) ** 2))

    for r in range(1, n_iterations + 1):
        # 1. Phase oracle O_{m*} = I - 2 |m*><m*|
        psi[target_index] = -psi[target_index]
        # 2. Diffusion operator D = 2|s><s| - I (exact unitary reflection about |s>)
        mean_amp = np.mean(psi)
        psi = 2.0 * mean_amp - psi

        norm_err = abs(float(np.linalg.norm(psi)) - 1.0)
        if norm_err > max_norm_err:
            max_norm_err = norm_err

        p_r = float(abs(psi[target_index]) ** 2)
        prob_trace[r] = p_r
        p_exact = math.sin((2 * r + 1) * theta) ** 2
        err_r = abs(p_r - p_exact)
        if err_r > max_analytic_err:
            max_analytic_err = err_r

    return prob_trace, float(max_norm_err), float(max_analytic_err)


def run_part_b_grover_dredging(rng: np.random.Generator) -> dict[str, Any]:
    """Run Part B: Exact Unitary Grover Search + Fixed-Budget Data-Dredging Paradox."""
    t_is = 252
    t_oos = 252
    df_t = t_is - 1
    daily_sigma = 0.01
    ann_factor = math.sqrt(float(t_is))

    # -----------------------------------------------------------------------
    # Part B.1: Exact Unitary Grover Simulation on q = 4..12 qubits (M = 16..4096)
    # -----------------------------------------------------------------------
    qubit_list = list(range(4, 13))
    grover_unitary_records = []
    oscillation_traces: dict[int, list[float]] = {}

    for q in qubit_list:
        m = 1 << q
        # Actual M x 252 white-noise return matrices (frozen past IS & locked future OOS)
        r_is = rng.normal(0.0, daily_sigma, size=(m, t_is))
        r_oos = rng.normal(0.0, daily_sigma, size=(m, t_oos))

        sr_is_all = ann_factor * (np.mean(r_is, axis=1) / np.std(r_is, axis=1, ddof=1))
        sr_oos_all = ann_factor * (
            np.mean(r_oos, axis=1) / np.std(r_oos, axis=1, ddof=1)
        )
        m_star = int(np.argmax(sr_is_all))
        r_opt = int(math.floor((math.pi / 4.0) * math.sqrt(m)))

        # Simulate up to 2*r_opt + 2 iterations to capture full oscillation peak & over-rotation
        n_steps = max(2 * r_opt + 2, 6)
        prob_trace, max_norm_err, max_analytic_err = simulate_exact_unitary_grover(
            q=q, target_index=m_star, n_iterations=n_steps
        )
        p_succ_opt = float(prob_trace[r_opt])
        p_classical_same_queries = float(min(1.0, r_opt / m))

        assert max_norm_err < 1e-13, f"Unitarity error too large at q={q}: {max_norm_err}"
        assert max_analytic_err < 1e-12, f"Analytic mismatch at q={q}: {max_analytic_err}"
        assert p_succ_opt > 0.96, f"Expected Grover P_succ > 0.96 at q={q}, got {p_succ_opt}"

        # Save selected normalized traces for Figure 7 Panel (b)
        if q in (4, 6, 8, 10, 12):
            oscillation_traces[q] = [float(v) for v in prob_trace]

        grover_unitary_records.append(
            {
                "q": q,
                "M": m,
                "R_grover": r_opt,
                "speedup_ratio_M_over_R": float(m / r_opt),
                "p_succ_grover_at_R": p_succ_opt,
                "p_succ_classical_at_R": p_classical_same_queries,
                "max_unitarity_err": max_norm_err,
                "max_analytic_err": max_analytic_err,
                "champion_sr_is": float(sr_is_all[m_star]),
                "champion_sr_oos": float(sr_oos_all[m_star]),
            }
        )

    # -----------------------------------------------------------------------
    # Part B.2: Exact Equivalence Check — Full White-Noise Returns vs Exact t_251
    # -----------------------------------------------------------------------
    # Under i.i.d. N(0, sigma^2) returns with T = 252 and annualization sqrt(252),
    # SR = sqrt(252) * mean(r) / std(r, ddof=1) follows EXACT Student-t(df=251).
    # Verify empirically via two-sample Kolmogorov-Smirnov test at M = 500 (200 trials).
    ks_n_trials = 200
    ks_m = 500
    max_sr_direct = np.zeros(ks_n_trials)
    for k in range(ks_n_trials):
        rmat = rng.normal(0.0, daily_sigma, size=(ks_m, t_is))
        srs = ann_factor * (np.mean(rmat, axis=1) / np.std(rmat, axis=1, ddof=1))
        max_sr_direct[k] = float(np.max(srs))

    exp_draws = rng.exponential(scale=1.0, size=ks_n_trials)
    sf_exact = -np.expm1(-exp_draws / ks_m)
    max_sr_exact_t = stats.t.isf(sf_exact, df=df_t)
    ks_stat, ks_pval = stats.ks_2samp(max_sr_direct, max_sr_exact_t)
    assert ks_pval > 0.01, f"Unexpected KS discrepancy between direct returns and t_251: p={ks_pval}"

    # -----------------------------------------------------------------------
    # Part B.3: Fixed Oracle-Query Budget Sweep C in {10, 30, 100, 300, 1000}
    # -----------------------------------------------------------------------
    # Classical budget C probes M_cl = C candidates.
    # Quantum Grover budget C probes M_quant = floor((4 * C / pi)^2) candidates.
    # We perform R = 80 independent Monte Carlo trials per budget C.
    # For C <= 100 (M_quant <= 16,211) we simulate FULL T=252 return matrices directly;
    # for C in {300, 1000} (M_quant = 145,902 and 1,621,138) we draw the exact
    # Student-t(251) maximum order statistic for IS and exact independent t(251)
    # (equivalent to 252-day white-noise returns) for OOS, keeping total runtime < 5 s
    # with ZERO statistical approximation error.
    c_grid = [10, 30, 100, 300, 1000]
    n_trials = 80
    dsr_threshold = 0.95

    budget_records = []

    for c_budget in c_grid:
        m_cl = int(c_budget)
        m_q = int(math.floor(((4.0 / math.pi) * c_budget) ** 2))

        sr_is_cl = np.zeros(n_trials)
        sr_oos_cl = np.zeros(n_trials)
        sr_is_q = np.zeros(n_trials)
        sr_oos_q = np.zeros(n_trials)

        dsr_cl_proper = np.zeros(n_trials)
        dsr_q_naive = np.zeros(n_trials)   # WRONG: deflating by oracle queries N = C
        dsr_q_proper = np.zeros(n_trials)  # RIGHT: deflating by effective space N = M_quant

        use_full_return_matrix = m_q <= 20_000

        for tr in range(n_trials):
            if use_full_return_matrix:
                # Direct full 252-day white-noise matrix for both classical and Grover pools
                r_is_q = rng.normal(0.0, daily_sigma, size=(m_q, t_is))
                r_oos_q = rng.normal(0.0, daily_sigma, size=(m_q, t_oos))
                srs_is = ann_factor * (
                    np.mean(r_is_q, axis=1) / np.std(r_is_q, axis=1, ddof=1)
                )
                srs_oos = ann_factor * (
                    np.mean(r_oos_q, axis=1) / np.std(r_oos_q, axis=1, ddof=1)
                )
                idx_cl = int(np.argmax(srs_is[:m_cl]))
                idx_q = int(np.argmax(srs_is))

                val_is_cl = float(srs_is[idx_cl])
                val_oos_cl = float(srs_oos[idx_cl])
                val_is_q = float(srs_is[idx_q])
                val_oos_q = float(srs_oos[idx_q])

                skew_cl = float(stats.skew(r_is_q[idx_cl]))
                kurt_cl = float(stats.kurtosis(r_is_q[idx_cl], fisher=False))
                skew_q = float(stats.skew(r_is_q[idx_q]))
                kurt_q = float(stats.kurtosis(r_is_q[idx_q], fisher=False))
            else:
                # Exact Student-t(df=251) maximum order statistic for large M_quant,
                # plus explicit 252-day return draws for classical pool and OOS verification
                r_is_cl = rng.normal(0.0, daily_sigma, size=(m_cl, t_is))
                r_oos_cl = rng.normal(0.0, daily_sigma, size=(m_cl, t_oos))
                srs_is_c = ann_factor * (
                    np.mean(r_is_cl, axis=1) / np.std(r_is_cl, axis=1, ddof=1)
                )
                srs_oos_c = ann_factor * (
                    np.mean(r_oos_cl, axis=1) / np.std(r_oos_cl, axis=1, ddof=1)
                )
                idx_cl = int(np.argmax(srs_is_c))
                val_is_cl = float(srs_is_c[idx_cl])
                val_oos_cl = float(srs_oos_c[idx_cl])
                skew_cl = float(stats.skew(r_is_cl[idx_cl]))
                kurt_cl = float(stats.kurtosis(r_is_cl[idx_cl], fisher=False))

                # Exact max of M_quant i.i.d. Student-t(251) variables
                e_draw = float(rng.exponential(scale=1.0))
                sf_val = float(-math.expm1(-e_draw / m_q))
                val_is_q = float(stats.t.isf(sf_val, df=df_t))
                # Independent future 252-day white-noise trajectory for the Grover champion
                r_oos_champ = rng.normal(0.0, daily_sigma, size=t_oos)
                val_oos_q = float(
                    ann_factor * np.mean(r_oos_champ) / np.std(r_oos_champ, ddof=1)
                )
                # Under Gaussian white noise, theoretical skewness=0, kurtosis=3
                skew_q = 0.0
                kurt_q = 3.0

            sr_is_cl[tr] = val_is_cl
            sr_oos_cl[tr] = val_oos_cl
            sr_is_q[tr] = val_is_q
            sr_oos_q[tr] = val_oos_q

            dsr_cl_proper[tr] = deflated_sharpe_ratio(
                val_is_cl, n_trials_assumed=m_cl, t_obs=t_is,
                skewness=skew_cl, kurtosis=kurt_cl,
            )
            dsr_q_naive[tr] = deflated_sharpe_ratio(
                val_is_q, n_trials_assumed=c_budget, t_obs=t_is,
                skewness=skew_q, kurtosis=kurt_q,
            )
            dsr_q_proper[tr] = deflated_sharpe_ratio(
                val_is_q, n_trials_assumed=m_q, t_obs=t_is,
                skewness=skew_q, kurtosis=kurt_q,
            )

        mean_is_cl = float(np.mean(sr_is_cl))
        std_is_cl = float(np.std(sr_is_cl, ddof=1))
        mean_oos_cl = float(np.mean(sr_oos_cl))
        std_oos_cl = float(np.std(sr_oos_cl, ddof=1))

        mean_is_q = float(np.mean(sr_is_q))
        std_is_q = float(np.std(sr_is_q, ddof=1))
        mean_oos_q = float(np.mean(sr_oos_q))
        std_oos_q = float(np.std(sr_oos_q, ddof=1))

        amp_ratio_emp = float(mean_is_q / mean_is_cl)
        th_blp_cl = expected_max_normal_blp(m_cl)
        th_blp_q = expected_max_normal_blp(m_q)
        amp_ratio_blp = float(th_blp_q / th_blp_cl)

        # Asymptotic sqrt(2 * ln M) ratio for comparison
        amp_ratio_asymp = float(
            math.sqrt(2.0 * math.log(m_q)) / math.sqrt(2.0 * math.log(m_cl))
        )

        budget_records.append(
            {
                "C_budget": c_budget,
                "M_classical": m_cl,
                "M_grover": m_q,
                "simulation_mode": (
                    "full_252d_matrix" if use_full_return_matrix else "exact_t251_order_stat"
                ),
                "sr_is_cl_mean": mean_is_cl,
                "sr_is_cl_std": std_is_cl,
                "sr_is_cl_theory_blp": th_blp_cl,
                "sr_oos_cl_mean": mean_oos_cl,
                "sr_oos_cl_std": std_oos_cl,
                "sr_is_grover_mean": mean_is_q,
                "sr_is_grover_std": std_is_q,
                "sr_is_grover_theory_blp": th_blp_q,
                "sr_oos_grover_mean": mean_oos_q,
                "sr_oos_grover_std": std_oos_q,
                "amplification_ratio_emp": amp_ratio_emp,
                "amplification_ratio_blp": amp_ratio_blp,
                "amplification_ratio_sqrt2ln": amp_ratio_asymp,
                "dsr_cl_proper_mean": float(np.mean(dsr_cl_proper)),
                "dsr_cl_proper_fpr_95": float(np.mean(dsr_cl_proper > dsr_threshold)),
                "dsr_grover_naive_mean": float(np.mean(dsr_q_naive)),
                "dsr_grover_naive_fpr_95": float(np.mean(dsr_q_naive > dsr_threshold)),
                "dsr_grover_proper_mean": float(np.mean(dsr_q_proper)),
                "dsr_grover_proper_fpr_95": float(np.mean(dsr_q_proper > dsr_threshold)),
            }
        )

    # Strict quantitative assertions for Part B
    rec_c1000 = budget_records[-1]
    assert rec_c1000["amplification_ratio_emp"] > 1.35, "Expected >= 1.35x Grover mirage amplification"
    assert abs(rec_c1000["sr_oos_grover_mean"]) < 0.35, "Expected OOS collapse near 0"
    assert abs(rec_c1000["sr_oos_cl_mean"]) < 0.35, "Expected classical OOS collapse near 0"
    assert rec_c1000["dsr_grover_naive_mean"] > 0.94, "Expected mean naive DSR > 0.94 at C=1000"
    assert rec_c1000["dsr_grover_naive_fpr_95"] > 0.50, "Expected naive DSR FPR > 50% at C=1000"
    assert rec_c1000["dsr_grover_proper_mean"] < 0.65, "Expected mean proper DSR near 0.50"
    assert rec_c1000["dsr_grover_proper_fpr_95"] < 0.05, "Expected effective-space DSR FPR < 5%"

    return {
        "T_IS": t_is,
        "T_OOS": t_oos,
        "n_trials_per_budget": n_trials,
        "dsr_confidence_threshold": dsr_threshold,
        "ks_validation_direct_vs_exact_t251": {
            "M": ks_m,
            "n_trials": ks_n_trials,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pval),
        },
        "grover_unitary_q4_to_q12": grover_unitary_records,
        "oscillation_traces": oscillation_traces,
        "fixed_budget_sweep": budget_records,
    }


# ===========================================================================
# PUBLICATION-GRADE 4-PANEL FIGURE GENERATION (300 DPI)
# ===========================================================================
def plot_figure_7(part_a: dict[str, Any], part_b: dict[str, Any]) -> None:
    """Generate clean 4-panel publication figure fig7_detection_loophole_and_grover_dredging.png."""
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14.4, 9.6), constrained_layout=True)
    fig.suptitle(
        "Experiment 7 — PhD Jury Traps: Detection Loophole (Q1) & Quantum Grover Data-Dredging Paradox (Q5)",
        fontsize=12.6,
        fontweight="bold",
        y=1.015,
    )

    # -----------------------------------------------------------------------
    # Panel (a): Detection Loophole — Post-Selected vs Loophole-Free CHSH
    # -----------------------------------------------------------------------
    ax = axes[0, 0]
    eta_dense = np.linspace(0.50, 1.00, 401)
    s_lhv_post_curve = np.array([analytic_lhv_chsh_post(e) for e in eta_dense])
    s_lhv_fill_curve = np.array([analytic_lhv_chsh_filled(e) for e in eta_dense])
    s_qm_fill_curve = np.array([analytic_qm_chsh_filled(e) for e in eta_dense])

    sweep = part_a["sweep"]
    etas_mc = np.array([r["eta"] for r in sweep])
    s_post_mc = np.array([r["s_post_mc"] for r in sweep])
    s_fill_mc = np.array([r["s_filled_mc"] for r in sweep])

    eta_star = part_a["eta_sym_star"]
    eta_eb = part_a["eta_eberhard_saturation"]
    tsirelson = part_a["tsirelson_bound"]

    # Reference bands
    ax.axhspan(2.0, tsirelson, color=PALETTE["amber"], alpha=0.07, zorder=0)
    ax.axhspan(tsirelson, 4.05, color=PALETTE["coral"], alpha=0.05, zorder=0)
    ax.axhline(2.0, color=PALETTE["slate"], ls="--", lw=1.1, zorder=1)
    ax.axhline(tsirelson, color=PALETTE["purple"], ls=":", lw=1.2, zorder=1)
    ax.axhline(4.0, color=PALETTE["coral"], ls=":", lw=0.9, alpha=0.65, zorder=1)
    ax.axvline(eta_star, color=PALETTE["purple"], ls="--", lw=1.0, alpha=0.75, zorder=1)
    ax.axvline(eta_eb, color=PALETTE["slate"], ls=":", lw=1.0, alpha=0.75, zorder=1)

    # Analytic curves + MC points
    ax.plot(
        eta_dense, s_lhv_post_curve, color=PALETTE["coral"], lw=2.1,
        label=r"Classical LHV — Post-selected ($4/\eta - 2$)", zorder=3
    )
    ax.scatter(
        etas_mc, s_post_mc, color=PALETTE["coral"], edgecolor="white",
        s=34, linewidth=0.6, zorder=4, label=r"MC LHV ($2\times 10^5$ pairs/setting)"
    )
    ax.plot(
        eta_dense, s_lhv_fill_curve, color=PALETTE["teal"], lw=2.1,
        label=r"Same LHV — Loophole-free ($\emptyset \mapsto +1$)", zorder=3
    )
    ax.scatter(
        etas_mc, s_fill_mc, color=PALETTE["teal"], marker="s", edgecolor="white",
        s=28, linewidth=0.5, zorder=4
    )
    ax.plot(
        eta_dense, s_qm_fill_curve, color=PALETTE["navy"], lw=1.9, ls="-.",
        label=r"Quantum $|\Phi^+\rangle$ — Loophole-free ($\emptyset \mapsto +1$)", zorder=3
    )

    # Crisp non-overlapping annotations
    ax.annotate(
        r"$\eta_{\mathrm{sym}}^* = \frac{2}{1+\sqrt{2}} \approx 82.84\%$"
        + "\n"
        + r"LHV hits $2\sqrt{2}$; $|\Phi^+\rangle$ crosses $2$",
        xy=(eta_star, tsirelson),
        xytext=(0.73, 3.48),
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["purple"], alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=PALETTE["purple"], lw=1.0),
    )
    ax.annotate(
        r"$\eta = 2/3$" + "\n" + r"($S_{\mathrm{post}}=4.0$)",
        xy=(eta_eb, 4.0),
        xytext=(0.515, 3.55),
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["slate"], alpha=0.92),
        arrowprops=dict(arrowstyle="->", color=PALETTE["slate"], lw=0.9),
    )
    ax.text(0.506, 2.04, r"Bell-CHSH bound ($S=2$)", fontsize=7.8, color=PALETTE["slate"])
    ax.text(0.506, tsirelson + 0.05, r"Tsirelson ($2\sqrt{2} \approx 2.828$)", fontsize=7.8, color=PALETTE["purple"])

    ax.set_xlim(0.495, 1.01)
    ax.set_ylim(1.05, 4.25)
    ax.set_xlabel(r"Detector / Acceptance Efficiency $\eta$")
    ax.set_ylabel(r"CHSH Parameter $S$")
    ax.set_title(r"(a) Part A: Detection Loophole & Loophole Closure ($\emptyset \mapsto +1$)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=7.6)

    # -----------------------------------------------------------------------
    # Panel (b): Exact Unitary Grover Amplification (q = 4..12 qubits)
    # -----------------------------------------------------------------------
    ax = axes[0, 1]
    traces = part_b["oscillation_traces"]
    q_colors = {
        4: PALETTE["amber"],
        6: PALETTE["teal"],
        8: PALETTE["purple"],
        10: PALETTE["navy"],
        12: PALETTE["coral"],
    }
    rec_by_q = {r["q"]: r for r in part_b["grover_unitary_q4_to_q12"]}

    for q_val in (4, 6, 8, 10, 12):
        p_arr = np.array(traces[q_val])
        r_opt = rec_by_q[q_val]["R_grover"]
        m_val = rec_by_q[q_val]["M"]
        r_norm = np.arange(len(p_arr)) / float(r_opt)
        lw = 2.2 if q_val == 12 else 1.5
        alpha = 0.95 if q_val in (4, 12) else 0.78
        ax.plot(
            r_norm,
            p_arr,
            color=q_colors[q_val],
            lw=lw,
            alpha=alpha,
            label=f"$q={q_val}$ ($M={m_val}$, $R^*={r_opt}$, $P^*={rec_by_q[q_val]['p_succ_grover_at_R']:.4f}$)",
        )

    # Classical scan success at same query count for q = 12 (r / M)
    r_opt_12 = rec_by_q[12]["R_grover"]
    r_norm_dense = np.linspace(0.0, 2.05, 200)
    p_classical_q12 = np.minimum(1.0, (r_norm_dense * r_opt_12) / 4096.0)
    ax.plot(
        r_norm_dense,
        p_classical_q12,
        color=PALETTE["slate"],
        ls="--",
        lw=1.7,
        label=r"Classical scan at $q=12$ ($r/M$, only $1.22\%$ at $R^*$)",
    )

    ax.axvline(1.0, color=PALETTE["navy"], ls=":", lw=1.1, alpha=0.8)
    ax.axhline(1.0, color=PALETTE["slate"], ls=":", lw=0.8, alpha=0.5)

    # Compact summary callout box on exactness (placed in open upper-right space)
    max_u_err = max(r["max_unitarity_err"] for r in part_b["grover_unitary_q4_to_q12"])
    max_a_err = max(r["max_analytic_err"] for r in part_b["grover_unitary_q4_to_q12"])
    ax.text(
        1.32,
        0.74,
        "Exact State-Vector Audit ($q=4\\dots 12$):\n"
        f"• Max unitarity err: {max_u_err:.1e}\n"
        f"• Max $\\sin^2((2r+1)\\theta)$ err: {max_a_err:.1e}\n"
        "• Speedup at $q=12$: $4096 / 50 = 81.9\\times$",
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.35", fc=PALETTE["bg_tint"], ec=PALETTE["navy"], alpha=0.95),
    )

    ax.set_xlim(0.0, 2.08)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"Normalized Oracle Iterations $r / R_{\mathrm{Grover}}$, where $R_{\mathrm{Grover}} = \lfloor \frac{\pi}{4}\sqrt{M} \rfloor$")
    ax.set_ylabel(r"Target Success Probability $P_{\mathrm{succ}}(r)$")
    ax.set_title(r"(b) Part B: Exact Unitary Grover Amplitude Amplification ($q=4\dots 12$)")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7.4)

    # -----------------------------------------------------------------------
    # Panel (c): Fixed Oracle Budget C — Grover sqrt(2) Mirage vs OOS Collapse
    # -----------------------------------------------------------------------
    ax = axes[1, 0]
    bs = part_b["fixed_budget_sweep"]
    c_vals = np.array([r["C_budget"] for r in bs], dtype=float)
    is_cl = np.array([r["sr_is_cl_mean"] for r in bs])
    is_cl_err = np.array([r["sr_is_cl_std"] for r in bs])
    oos_cl = np.array([r["sr_oos_cl_mean"] for r in bs])
    oos_cl_err = np.array([r["sr_oos_cl_std"] for r in bs])

    is_q = np.array([r["sr_is_grover_mean"] for r in bs])
    is_q_err = np.array([r["sr_is_grover_std"] for r in bs])
    oos_q = np.array([r["sr_oos_grover_mean"] for r in bs])
    oos_q_err = np.array([r["sr_oos_grover_std"] for r in bs])

    th_cl = np.array([r["sr_is_cl_theory_blp"] for r in bs])
    th_q = np.array([r["sr_is_grover_theory_blp"] for r in bs])

    ax.axhline(0.0, color=PALETTE["slate"], ls="-", lw=1.0, alpha=0.7)
    ax.errorbar(
        c_vals, is_q, yerr=is_q_err, fmt="o-", color=PALETTE["coral"], lw=2.1,
        capsize=3.5, markersize=5.5,
        label=r"Grover IS $\overline{\mathrm{SR}}$ ($M_{\mathrm{quant}} = \lfloor(4C/\pi)^2\rfloor$)"
    )
    ax.plot(c_vals, th_q, color=PALETTE["coral"], ls="--", lw=1.2, alpha=0.8, label="BLP theory ($M_{\\mathrm{quant}}$)")

    ax.errorbar(
        c_vals, is_cl, yerr=is_cl_err, fmt="s-", color=PALETTE["navy"], lw=2.1,
        capsize=3.5, markersize=5.2,
        label=r"Classical IS $\overline{\mathrm{SR}}$ ($M_{\mathrm{cl}} = C$)"
    )
    ax.plot(c_vals, th_cl, color=PALETTE["navy"], ls="--", lw=1.2, alpha=0.8, label="BLP theory ($M_{\\mathrm{cl}}$)")

    # Slight x-offset on OOS error bars so both curves remain cleanly legible
    ax.errorbar(
        c_vals * 0.94, oos_q, yerr=oos_q_err, fmt="D-", color=PALETTE["amber"], lw=1.6,
        capsize=3.0, markersize=4.8, label=r"Grover OOS $\overline{\mathrm{SR}}$ (future collapse)"
    )
    ax.errorbar(
        c_vals * 1.06, oos_cl, yerr=oos_cl_err, fmt="^-", color=PALETTE["teal"], lw=1.6,
        capsize=3.0, markersize=4.8, label=r"Classical OOS $\overline{\mathrm{SR}}$ (future collapse)"
    )

    # Highlight the ~sqrt(2) (+41%) amplification gap at C = 1000
    ratio_1000 = bs[-1]["amplification_ratio_emp"]
    ax.annotate(
        "",
        xy=(1000.0, is_q[-1]),
        xytext=(1000.0, is_cl[-1]),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["purple"], lw=1.6),
    )
    ax.text(
        500.0,
        0.5 * (is_q[-1] + is_cl[-1]) - 0.12,
        f"$\\times {ratio_1000:.2f}$ (+{(ratio_1000 - 1.0)*100:.0f}%)\n(asympt. $\\sqrt{{2}} \\approx 1.414$)",
        fontsize=8.0,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["purple"], alpha=0.94),
    )

    ax.set_xscale("log")
    ax.set_xlim(7.5, 1450.0)
    ax.set_ylim(-1.55, 6.25)
    ax.set_xticks(c_vals)
    ax.set_xticklabels([str(int(c)) for c in c_vals])
    ax.set_xlabel(r"Fixed Oracle-Query Budget $C$ (White Noise, $T_{\mathrm{IS}}=T_{\mathrm{OOS}}=252$, $R=80$ trials)")
    ax.set_ylabel(r"Annualized Sharpe Ratio ($\mathrm{SR}$)")
    ax.set_title(r"(c) Part B: Fixed-Budget Grover Mirage ($\approx \sqrt{2}$) vs. OOS Collapse")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=7.3, ncol=2)

    # -----------------------------------------------------------------------
    # Panel (d): Deflated Sharpe Ratio (DSR) Trap — Naive C vs Effective M_eff
    # -----------------------------------------------------------------------
    ax = axes[1, 1]
    x_pos = np.arange(len(c_vals))
    w = 0.25

    fpr_naive = np.array([r["dsr_grover_naive_fpr_95"] * 100.0 for r in bs])
    fpr_proper = np.array([r["dsr_grover_proper_fpr_95"] * 100.0 for r in bs])
    fpr_cl = np.array([r["dsr_cl_proper_fpr_95"] * 100.0 for r in bs])

    dsr_naive_m = np.array([r["dsr_grover_naive_mean"] for r in bs])
    dsr_proper_m = np.array([r["dsr_grover_proper_mean"] for r in bs])

    bars_naive = ax.bar(
        x_pos - w, fpr_naive, width=w, color=PALETTE["coral"], alpha=0.90,
        edgecolor="#7F1D1D", linewidth=0.7,
        label=r"Grover + Naive $\mathrm{DSR}(N=C)$ [FAILS]"
    )
    bars_proper = ax.bar(
        x_pos, fpr_proper, width=w, color=PALETTE["teal"], alpha=0.90,
        edgecolor="#115E59", linewidth=0.7,
        label=r"Grover + Effective $\mathrm{DSR}(N=M_{\mathrm{eff}})$ [FIXED]"
    )
    bars_cl = ax.bar(
        x_pos + w, fpr_cl, width=w, color=PALETTE["navy"], alpha=0.82,
        edgecolor="#0F172A", linewidth=0.7,
        label=r"Classical + $\mathrm{DSR}(N=C)$ [Reference]"
    )

    ax.axhline(5.0, color=PALETTE["slate"], ls="--", lw=1.1, label=r"Nominal $\alpha = 5\%$ level")

    for rect, val in zip(bars_naive, fpr_naive):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val + 1.8,
            f"{val:.0f}%",
            ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=PALETTE["coral"],
        )
    for rect, val in zip(bars_proper, fpr_proper):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val + 1.8,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=7.4, color=PALETTE["teal"],
        )
    for rect, val in zip(bars_cl, fpr_cl):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            val + 1.8,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=7.4, color=PALETTE["navy"],
        )

    ax.text(
        0.03,
        0.54,
        "Mean DSR at $C = 1000$ ($M_{\\mathrm{eff}} = 1.62 \\times 10^6$):\n"
        f"• Naive $\\mathrm{{DSR}}(C)$: {dsr_naive_m[-1]:.3f}  (FPR = {fpr_naive[-1]:.1f}%)\n"
        f"• Proper $\\mathrm{{DSR}}(M_{{\\mathrm{{eff}}}})$: {dsr_proper_m[-1]:.3f}  (FPR = {fpr_proper[-1]:.1f}%)\n"
        "Rule: Deflate by $M_{\\mathrm{eff}}$, never by quantum queries $C$!",
        transform=ax.transAxes,
        fontsize=7.9,
        bbox=dict(boxstyle="round,pad=0.35", fc=PALETTE["bg_tint"], ec=PALETTE["coral"], alpha=0.95),
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"C={int(c)}\n(M={r['M_grover']:,})" for c, r in zip(c_vals, bs)], fontsize=7.8)
    ax.set_ylim(0.0, 108.0)
    ax.set_ylabel(r"False Positive Rate at $\mathrm{DSR} > 0.95$ (%)")
    ax.set_title(r"(d) Part B: Deflated Sharpe Ratio Trap — Naive $C$ vs. Effective $M_{\mathrm{eff}}$")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7.6)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Execute Experiment 7 end-to-end, save JSON results and 300 DPI figure."""
    t_start = time.perf_counter()
    rng = np.random.default_rng(GLOBAL_SEED)

    print("=" * 78)
    print("EXPERIMENT 7 — PhD JURY TRAPS: DETECTION LOOPHOLE & GROVER DATA DREDGING")
    print(f"Platform: {platform.platform()} | Python: {platform.python_version()} | Seed: {GLOBAL_SEED}")
    print("=" * 78)

    part_a = run_part_a_detection_loophole(rng)
    part_b = run_part_b_grover_dredging(rng)
    plot_figure_7(part_a, part_b)

    elapsed = time.perf_counter() - t_start

    # Compact JSON summary (exclude huge oscillation_traces arrays? Keep them small & clean!)
    payload = {
        "metadata": {
            "experiment": "exp7_jury_traps_loophole_grover",
            "seed": GLOBAL_SEED,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "runtime_seconds": round(elapsed, 3),
            "figure_path": str(FIGURE_PATH),
        },
        "part_a_detection_loophole": part_a,
        "part_b_grover_data_dredging": {
            k: v for k, v in part_b.items() if k != "oscillation_traces"
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    # Print concise, exact audit summary to stdout
    hl = part_a["highlights"]
    print("\n--- PART A: DETECTION LOOPHOLE (200,000 pairs/setting = 800,000 pairs/eta) ---")
    print(
        f"  eta = 1.0000: S_post_MC = {hl['eta_1.00']['s_post_mc']:.4f} (exact {hl['eta_1.00']['s_post_analytic']:.4f}) | "
        f"S_filled_MC = {hl['eta_1.00']['s_filled_mc']:.4f}"
    )
    print(
        f"  eta*= {hl['eta_sym_star']['eta']:.4f}: S_post_MC = {hl['eta_sym_star']['s_post_mc']:.4f} "
        f"(Tsirelson {part_a['tsirelson_bound']:.4f}) | S_filled_MC = {hl['eta_sym_star']['s_filled_mc']:.4f} | "
        f"QM_filled = {hl['eta_sym_star']['s_qm_filled_analytic']:.4f}"
    )
    print(
        f"  eta = 0.6667: S_post_MC = {hl['eta_2_over_3']['s_post_mc']:.4f} (exact 4.0000) | "
        f"S_filled_MC = {hl['eta_2_over_3']['s_filled_mc']:.4f}"
    )
    print(
        f"  eta = 0.5000: S_post_MC = {hl['eta_0.50']['s_post_mc']:.4f} (exact 4.0000) | "
        f"S_filled_MC = {hl['eta_0.50']['s_filled_mc']:.4f}"
    )
    print(
        f"  Max MC error vs analytic: post={part_a['max_mc_err_post_selected']:.5f}, "
        f"filled={part_a['max_mc_err_loophole_free']:.5f} | "
        f"max |S_filled_MC|={part_a['max_mc_loophole_free_abs_s']:.4f}"
    )

    print("\n--- PART B.1: EXACT UNITARY GROVER AMPLIFICATION (q = 4..12) ---")
    for rec in part_b["grover_unitary_q4_to_q12"]:
        print(
            f"  q={rec['q']:2d} (M={rec['M']:4d}): R*={rec['R_grover']:2d} | "
            f"P_succ={rec['p_succ_grover_at_R']:.6f} vs classical {rec['p_succ_classical_at_R']:.4f} | "
            f"unitarity_err={rec['max_unitarity_err']:.2e} | analytic_err={rec['max_analytic_err']:.2e}"
        )

    print("\n--- PART B.2/B.3: FIXED ORACLE BUDGET C (R = 80 trials) ---")
    ks = part_b["ks_validation_direct_vs_exact_t251"]
    print(
        f"  KS check (direct 252d returns vs exact t_251, M={ks['M']}, R={ks['n_trials']}): "
        f"stat={ks['ks_statistic']:.4f}, p={ks['ks_pvalue']:.4f}"
    )
    for rec in part_b["fixed_budget_sweep"]:
        print(
            f"  C={rec['C_budget']:4d} (M_q={rec['M_grover']:9,d}): "
            f"IS_cl={rec['sr_is_cl_mean']:.3f} vs IS_gr={rec['sr_is_grover_mean']:.3f} "
            f"(x{rec['amplification_ratio_emp']:.3f}) | "
            f"OOS_cl={rec['sr_oos_cl_mean']:+.3f}, OOS_gr={rec['sr_oos_grover_mean']:+.3f} | "
            f"FPR95 naive={rec['dsr_grover_naive_fpr_95']*100:5.1f}% vs proper={rec['dsr_grover_proper_fpr_95']*100:4.1f}%"
        )

    print(f"\nSaved JSON  : {RESULTS_JSON_PATH}")
    print(f"Saved Figure: {FIGURE_PATH}")
    print(f"Completed cleanly in {elapsed:.2f} s (all assertions passed).")


if __name__ == "__main__":
    main()
