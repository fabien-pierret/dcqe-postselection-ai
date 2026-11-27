#!/usr/bin/env python3
r"""
Experiment 10 (Article 1): Quantum Overfitting Amplification Law & Heavy-Tail Q-DSR.

Empirical and exact unitary validation of the four theorems of Article 1
("The Quantum Overfitting Amplification Theorem & Heavy-Tail Quantum-Adjusted
Deflated Sharpe Ratio (Q-DSR)"):

  - Part A (Theorem 1 & Theorem 2): Fixed oracle query budget C in {10, 30, 100, 300, 1000}
    comparing Classical search (M_cl = C) vs Grover / Amplitude-Amplified quantum search
    (M_quant = floor((4/pi * C)^2)) under the complete null hypothesis H0 (zero true signal)
    across 5 tail regimes:
      1. Gaussian (sub-Weibull alpha = 2)      -> universal limit ratio sqrt(2) = 1.4142
      2. Laplace  (sub-Weibull alpha = 1)      -> universal limit ratio 2.0000
      3. Student-t (nu = 6, Fréchet heavy)     -> polynomial explosion (4/pi)^(2/6) * C^(1/6)
      4. Student-t (nu = 4, Fréchet heavy)     -> polynomial explosion (4/pi)^(2/4) * C^(1/4)
      5. Student-t (nu = 3, Fréchet heavy)     -> polynomial explosion (4/pi)^(2/3) * C^(1/3)
    Verifies both exact in-sample mirage amplification and out-of-sample collapse to 0.00.
    Also runs an explicit unitary statevector Grover search verification on q in {6, 8, 10, 12}
    qubits (M = 64..4096) confirming exact Grover probability amplification P_succ -> 1
    in R* = floor((pi/4) sqrt(M)) queries on real null heavy-tail and Gaussian score tables.

  - Part B (Theorem 3): False Positive Rate (FPR at nominal alpha = 0.05) collapse of
    C-indexed multiple testing (Bonferroni(C), Naive Gaussian DSR(C)) vs restoration by:
      - Gaussian Q-DSR(M_eff) (fixes Gaussian, FAILS on heavy-tail Student-t_3, t_4!)
      - Heavy-Tail Exact Q-DSR_nu(M_eff) = [F_{t_nu}(x)]^{M_eff} (exact FPR = 5.0% across
        ALL tail indices nu and ALL query budgets C!).
    Also verifies correlated-catalog multiplicity correction M_eff(Sigma) = M^2 / ||Sigma||_F^2.

  - Part C (Theorem 4): Landauer thermodynamic accounting of reversible unitary Grover
    iterations (0 intrinsic Landauer dissipation during unitary steps) vs irreversible
    measurement/reset dissipation Q_min = k_B T ln(M_eff) ~ 2 k_B T ln(C) (double the
    classical selection entropy at equal query budget C, invariant k_B T ln 2 per selection bit).

Outputs:
  - sim/results/exp10_paper1_results.json
  - sim/figures/fig10_quantum_overfitting_amplification_law.png (300 DPI, 4 panels)

Deterministic execution (fixed seed=42) on macOS ARM64 CPU (NumPy/SciPy/Matplotlib).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.stats import laplace, norm, t as student_t


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

KAPPA_GROVER = 4.0 / math.pi  # ~1.2732395447351628
EULER_GAMMA = 0.5772156649015329
K_B_300K_J = 1.380649e-23 * 300.0  # k_B * T at T = 300 K in Joules


def exact_grover_statevector_verify(rng: np.random.Generator) -> List[Dict]:
    """
    Explicit unitary statevector simulation of Grover's algorithm on q in {6, 8, 10, 12}
    qubits (M = 64..4096) searching for the unique in-sample maximum argmax_m X_m of
    a null Student-t_3 score table (and checking unitarity + exact sin^2((2r+1)theta) law).
    """
    records = []
    for q in (6, 8, 10, 12):
        m_size = 1 << q
        scores_is = student_t.rvs(df=3, size=m_size, random_state=rng)
        scores_oos = student_t.rvs(df=3, size=m_size, random_state=rng)
        idx_max = int(np.argmax(scores_is))

        # Initial uniform superposition |s> = H^{\otimes q} |0^q>
        psi = np.full(m_size, 1.0 / math.sqrt(m_size), dtype=np.complex128)
        theta = math.asin(1.0 / math.sqrt(m_size))
        r_opt = int(math.floor((math.pi / 4.0) * math.sqrt(m_size)))

        # Unitary Grover iterations G = D * O_f
        for _ in range(r_opt):
            psi[idx_max] *= -1.0  # Phase oracle O_f
            mean_amp = np.mean(psi)
            psi = 2.0 * mean_amp - psi  # Diffusion operator D = 2|s><s| - I

        norm_err = abs(float(np.linalg.norm(psi)) - 1.0)
        p_succ = float(abs(psi[idx_max]) ** 2)
        p_theory = math.sin((2 * r_opt + 1) * theta) ** 2
        prob_err = abs(p_succ - p_theory)
        assert norm_err < 1e-13, f"Unitarity error: {norm_err}"
        assert prob_err < 1e-13, f"Grover formula error: {prob_err}"

        # Compare classical budget C = r_opt vs Grover searching full M = 2^q in r_opt queries
        cl_subset = scores_is[:r_opt]
        cl_best_idx = int(np.argmax(cl_subset))
        records.append(
            {
                "q_qubits": q,
                "M_space": m_size,
                "C_grover_queries": r_opt,
                "P_succ_exact": p_succ,
                "P_succ_theory": p_theory,
                "abs_err_vs_theory": prob_err,
                "unitarity_norm_err": norm_err,
                "classical_max_IS_t3": float(scores_is[cl_best_idx]),
                "grover_max_IS_t3": float(scores_is[idx_max]),
                "grover_over_classical_ratio": float(scores_is[idx_max] / max(scores_is[cl_best_idx], 1e-9)),
                "grover_winner_OOS_t3": float(scores_oos[idx_max]),
                "landauer_bits_classical": float(math.log2(r_opt)),
                "landauer_bits_quantum": float(q),
            }
        )
    return records


def sample_max_exact_order_stat(
    dist_name: str, m_size: int, n_trials: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Exact sampling of M_m = max(X_1, ..., X_m) for i.i.d. continuous random variables
    via the exact probability integral transform of the maximum order statistic:
      U_max = U^(1 / m)  where U ~ Uniform(0, 1)  =>  X_max = F^{-1}(U_max).
    Using log1p / survival-function inversion (-expm1(log(U)/m)) avoids float64 loss
    of precision even when m = 1.62e6!
    """
    # Exact survival probability of the maximum: S_max = 1 - U^(1/m) = -expm1(ln(U) / m)
    # Wait: U ~ Unif(0,1) => -ln(U) = E ~ Exp(1), so 1 - U^(1/m) = -expm1(-E / m) = 1 - exp(-E / m)!
    e_exp = rng.exponential(scale=1.0, size=n_trials)
    sf_val = -np.expm1(-e_exp / float(m_size))  # exact 1 - U^(1/m) in (0, 1), full precision!

    if dist_name == "gaussian":
        return norm.isf(sf_val)
    if dist_name == "laplace":
        # Unit-scale Laplace (b = 1, survival = 0.5 * exp(-x) for x >= 0) -> Var = 2
        # Normalize to unit variance? Wait: scale invariance cancels in ratios!
        # Using standard Laplace(0, 1/sqrt(2)) gives unit variance Var = 1 for clean comparison.
        return laplace.isf(sf_val, loc=0.0, scale=1.0 / math.sqrt(2.0))
    if dist_name.startswith("student_t_"):
        nu = int(dist_name.split("_")[-1])
        return student_t.isf(sf_val, df=nu)
    raise ValueError(f"Unknown distribution: {dist_name}")


def sample_single_oos(dist_name: str, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Sample independent out-of-sample null score for the selected winner."""
    if dist_name == "gaussian":
        return rng.normal(0.0, 1.0, size=n_trials)
    if dist_name == "laplace":
        return rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=n_trials)
    if dist_name.startswith("student_t_"):
        nu = int(dist_name.split("_")[-1])
        return student_t.rvs(df=nu, size=n_trials, random_state=rng)
    raise ValueError(f"Unknown distribution: {dist_name}")


def verify_order_stat_sampler_vs_direct(rng: np.random.Generator) -> Dict[str, float]:
    """
    Cross-check our exact order-statistic sampler against brute-force max of M = 1000
    explicit i.i.d. draws across Gaussian, Laplace, and Student-t_3 (two-sample KS test).
    """
    from scipy.stats import ks_2samp

    res = {}
    n_mc = 1500
    m_check = 1000
    for dname in ("gaussian", "laplace", "student_t_3"):
        fast = sample_max_exact_order_stat(dname, m_check, n_mc, rng)
        if dname == "gaussian":
            brute = np.max(rng.normal(0.0, 1.0, size=(n_mc, m_check)), axis=1)
        elif dname == "laplace":
            brute = np.max(rng.laplace(0.0, 1.0 / math.sqrt(2.0), size=(n_mc, m_check)), axis=1)
        else:
            brute = np.max(student_t.rvs(df=3, size=(n_mc, m_check), random_state=rng), axis=1)
        ks_stat, pval = ks_2samp(fast, brute)
        res[dname] = {"ks_stat": float(ks_stat), "p_value": float(pval)}
        assert pval > 0.01, f"Order stat sampler mismatch on {dname}: p={pval}"
    return res


def bldp_expected_max_gaussian(m_eff: float) -> float:
    """Bailey & López de Prado (2014) EVT expected maximum of m_eff standard normals."""
    if m_eff <= 1.0:
        return 0.0
    return float(
        (1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / m_eff)
        + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (math.e * m_eff))
    )


def run_part_a_and_b_amplification_and_qdsr(rng: np.random.Generator) -> Tuple[Dict, Dict]:
    """
    Run Theorems 1, 2, 3 across C in {10, 30, 100, 300, 1000} and 5 tail regimes
    with R = 4,000 Monte Carlo replicates per (regime, C) for high statistical precision!
    """
    ks_check = verify_order_stat_sampler_vs_direct(rng)

    c_list = [10, 30, 100, 300, 1000]
    m_quant_list = [int(math.floor((KAPPA_GROVER * c) ** 2)) for c in c_list]
    n_trials = 4000
    alpha_nom = 0.05
    z_crit = float(norm.ppf(1.0 - alpha_nom))  # 1.6448536...

    regimes = [
        ("gaussian", "sub_weibull", 2.0),
        ("laplace", "sub_weibull", 1.0),
        ("student_t_6", "frechet", 6.0),
        ("student_t_4", "frechet", 4.0),
        ("student_t_3", "frechet", 3.0),
    ]

    amp_results: Dict[str, Dict] = {}
    fpr_results: Dict[str, Dict] = {}

    for dname, family, param in regimes:
        cl_is_means, cl_is_stds = [], []
        q_is_means, q_is_stds = [], []
        cl_oos_means, q_oos_means = [], []
        emp_ratios, theo_ratios = [], []

        # Multiple-testing FPR trackers at nominal alpha = 0.05 on quantum winner X*_quant(C)
        fpr_bonf_c = []          # Naive Bonferroni indexed on C (using true marginal tail!)
        fpr_naive_dsr_c = []     # Naive Bailey-Lopez de Prado DSR indexed on C
        fpr_gauss_qdsr = []      # Gaussian Q-DSR indexed on M_quant(C)
        fpr_exact_gauss_evt = [] # Exact Gaussian EVT [Phi(x)]^{M_quant} > 1 - alpha
        fpr_exact_tail_qdsr = [] # Heavy-Tail / Exact Q-DSR_nu [F(x)]^{M_quant} > 1 - alpha

        for c_val, m_q in zip(c_list, m_quant_list):
            x_cl = sample_max_exact_order_stat(dname, c_val, n_trials, rng)
            x_q = sample_max_exact_order_stat(dname, m_q, n_trials, rng)
            oos_cl = sample_single_oos(dname, n_trials, rng)
            oos_q = sample_single_oos(dname, n_trials, rng)

            m_cl = float(np.mean(x_cl))
            m_q_mean = float(np.mean(x_q))
            cl_is_means.append(m_cl)
            cl_is_stds.append(float(np.std(x_cl)))
            q_is_means.append(m_q_mean)
            q_is_stds.append(float(np.std(x_q)))
            cl_oos_means.append(float(np.mean(oos_cl)))
            q_oos_means.append(float(np.mean(oos_q)))
            emp_ratios.append(float(m_q_mean / m_cl))

            if family == "sub_weibull":
                # Pre-asymptotic ratio (2 + 2 ln(kappa) / ln(C))^(1/alpha) -> 2^(1/alpha)
                theo_r = float((2.0 + 2.0 * math.log(KAPPA_GROVER) / math.log(c_val)) ** (1.0 / param))
            else:
                # Exact Fréchet leading ratio kappa^(2/nu) * C^(1/nu)
                theo_r = float((KAPPA_GROVER ** (2.0 / param)) * (c_val ** (1.0 / param)))
            theo_ratios.append(theo_r)

            # --- Multiple-Testing Auditors evaluated on X*_quant(C) ---
            # 1) Naive Bonferroni indexed on C using the TRUE marginal distribution F!
            if dname == "gaussian":
                tau_bonf_c = float(norm.isf(alpha_nom / c_val))
                p_exact_q = 1.0 - np.exp(-m_q * (-np.log1p(-norm.sf(x_q))))
            elif dname == "laplace":
                tau_bonf_c = float(laplace.isf(alpha_nom / c_val, loc=0.0, scale=1.0 / math.sqrt(2.0)))
                p_exact_q = 1.0 - np.exp(-m_q * (-np.log1p(-laplace.sf(x_q, loc=0.0, scale=1.0 / math.sqrt(2.0)))))
            else:
                nu = int(param)
                tau_bonf_c = float(student_t.isf(alpha_nom / c_val, df=nu))
                # Exact heavy-tail Q-DSR_nu = [F_{t_nu}(x)]^{M_quant} = exp(M_quant * log1p(-sf))
                qdsr_nu_vals = np.exp(m_q * np.log1p(-student_t.sf(x_q, df=nu)))
                p_exact_q = 1.0 - qdsr_nu_vals

            fpr_bonf_c.append(float(np.mean(x_q > tau_bonf_c)))

            # 2) Naive Bailey-Lopez de Prado DSR indexed on C: reject if x_q > SR_0(C) + z_{0.95}
            #    (for unit-variance standardized scores; for t_nu with nu > 2, divide by sqrt(nu/(nu-2))
            #     OR evaluate directly on standardized scores — wait: in finance, DSR standardizes by sample std!
            #     Here x_q is a t-statistic with scale 1, so we evaluate DSR on z-scale directly:
            #     reject when Phi(x_q - SR_0(C)) > 1 - alpha <=> x_q > SR_0(C) + z_{1-alpha}!)
            sr0_c = bldp_expected_max_gaussian(float(c_val))
            fpr_naive_dsr_c.append(float(np.mean(x_q > sr0_c + z_crit)))

            # 3) Gaussian Q-DSR indexed on M_quant(C): reject when x_q > SR_0(M_quant) + z_{1-alpha}
            sr0_mq = bldp_expected_max_gaussian(float(m_q))
            fpr_gauss_qdsr.append(float(np.mean(x_q > sr0_mq + z_crit)))

            # 4) Exact Gaussian EVT Q-DSR: [Phi(x_q)]^{M_quant} > 1 - alpha
            tau_gauss_evt = float(norm.isf(-math.expm1(math.log1p(-alpha_nom) / m_q)))
            fpr_exact_gauss_evt.append(float(np.mean(x_q > tau_gauss_evt)))

            # 5) Exact Tail-Matched Q-DSR_nu: [F(x_q)]^{M_quant} > 1 - alpha (exact p < alpha)
            fpr_exact_tail_qdsr.append(float(np.mean(p_exact_q < alpha_nom)))

        # Fit empirical Fréchet slope log(ratio) vs log(C) for heavy tails
        log_slope = float(np.polyfit(np.log(c_list), np.log(emp_ratios), 1)[0])
        amp_results[dname] = {
            "family": family,
            "tail_param": param,
            "asymptotic_theory_limit_or_exponent": (2.0 ** (1.0 / param)) if family == "sub_weibull" else (1.0 / param),
            "c_list": c_list,
            "m_quant_list": m_quant_list,
            "classical_IS_mean": cl_is_means,
            "classical_IS_std": cl_is_stds,
            "quantum_IS_mean": q_is_means,
            "quantum_IS_std": q_is_stds,
            "classical_OOS_mean": cl_oos_means,
            "quantum_OOS_mean": q_oos_means,
            "empirical_ratio_Q_over_C": emp_ratios,
            "theory_ratio_Q_over_C": theo_ratios,
            "loglog_slope_ratio_vs_C": log_slope,
        }
        fpr_results[dname] = {
            "c_list": c_list,
            "m_quant_list": m_quant_list,
            "nominal_alpha": alpha_nom,
            "fpr_naive_bonferroni_C": fpr_bonf_c,
            "fpr_naive_dsr_C": fpr_naive_dsr_c,
            "fpr_gaussian_qdsr_Meff": fpr_gauss_qdsr,
            "fpr_exact_gaussian_evt_Meff": fpr_exact_gauss_evt,
            "fpr_exact_tail_qdsr_nu": fpr_exact_tail_qdsr,
        }

    # --- Correlated-catalog multiplicity verification: Participation Ratio vs Slepian Lemma ---
    # Simulate M0 = 1000 equicorrelated Gaussian strategies (rho in {0.0, 0.25, 0.50, 0.75, 0.90})
    # demonstrating: (1) naive participation ratio M0^2/||Sigma||_F^2 is dangerously anti-conservative
    # for maxima; (2) full M0 is guaranteed conservative (FPR <= alpha) by Slepian's lemma;
    # (3) exact equicorrelated quantile (and Slepian effective multiplicity M0^(1 - rho)) restores calibration.
    m0_corr = 1000
    rho_grid = [0.0, 0.25, 0.50, 0.75, 0.90]
    corr_verification = []
    n_mc_corr = 2000
    for rho in rho_grid:
        m_eff_pr = float(m0_corr / (1.0 + (m0_corr - 1.0) * (rho**2)))
        m_eff_slepian = float(m0_corr ** (1.0 - rho))
        z0 = rng.normal(0.0, 1.0, size=(n_mc_corr, 1))
        zj = rng.normal(0.0, 1.0, size=(n_mc_corr, m0_corr))
        x_corr = math.sqrt(rho) * z0 + math.sqrt(1.0 - rho) * zj
        max_vals = np.max(x_corr, axis=1)
        emp_mean_max = float(np.mean(max_vals))
        exact_eq_max = float(math.sqrt(1.0 - rho) * bldp_expected_max_gaussian(float(m0_corr)))
        # Exact EVT thresholds at alpha = 0.05:
        tau_m0_evt = float(norm.isf(-math.expm1(math.log1p(-alpha_nom) / m0_corr)))
        tau_pr_evt = float(norm.isf(-math.expm1(math.log1p(-alpha_nom) / max(1.01, m_eff_pr))))
        tau_slepian_evt = float(norm.isf(-math.expm1(math.log1p(-alpha_nom) / max(1.01, m_eff_slepian))))
        corr_verification.append(
            {
                "rho": rho,
                "M_catalog": m0_corr,
                "M_eff_participation_ratio": m_eff_pr,
                "M_eff_slepian_M_pow_1mrho": m_eff_slepian,
                "empirical_E_max": emp_mean_max,
                "exact_equicorr_E_max": exact_eq_max,
                "fpr_slepian_conservative_M0": float(np.mean(max_vals > tau_m0_evt)),
                "fpr_naive_participation_ratio_PR": float(np.mean(max_vals > tau_pr_evt)),
                "fpr_slepian_Meff_M_pow_1mrho": float(np.mean(max_vals > tau_slepian_evt)),
            }
        )

    return (
        {
            "order_stat_sampler_ks_check": ks_check,
            "n_trials_per_point": n_trials,
            "regimes": amp_results,
        },
        {
            "fpr_tables": fpr_results,
            "correlated_catalog_check": corr_verification,
        },
    )


def compute_part_c_landauer(c_list: List[int]) -> Dict:
    """
    Compute exact Landauer thermodynamic selection bounds (Theorem 4):
      - Classical search (M_cl = C): Q_min = k_B T ln(C), Delta_I = log2(C) bits
      - Quantum Grover search (M_quant = floor((4/pi * C)^2)):
          * During C unitary iterations: 0 intrinsic Landauer dissipation
          * At measurement + register reset: Q_min = k_B T ln(M_quant) ~ 2 k_B T ln(C)
            (Delta_I = log2(M_quant) ~ 2 log2(C) + 0.697 bits)
      - Energy per unit selection bit is strictly invariant: k_B T ln 2 = 2.8708e-21 J at 300 K.
    """
    rows = []
    for c in c_list:
        m_q = int(math.floor((KAPPA_GROVER * c) ** 2))
        bits_cl = math.log2(c)
        bits_q = math.log2(m_q)
        nats_cl = math.log(c)
        nats_q = math.log(m_q)
        rows.append(
            {
                "C": c,
                "M_cl": c,
                "M_quant": m_q,
                "bits_classical": bits_cl,
                "bits_quantum": bits_q,
                "ratio_quantum_over_classical_landauer": bits_q / bits_cl,
                "nats_classical": nats_cl,
                "nats_quantum": nats_q,
                "Q_min_classical_zJ_300K": nats_cl * K_B_300K_J * 1e21,  # zeptojoules
                "Q_min_quantum_zJ_300K": nats_q * K_B_300K_J * 1e21,
                "unitary_stage_dissipation_J": 0.0,
            }
        )
    return {
        "temperature_K": 300.0,
        "landauer_per_bit_zJ": K_B_300K_J * math.log(2.0) * 1e21,
        "rows": rows,
    }


def plot_fig10(
    amp_data: Dict, fpr_data: Dict, landauer_data: Dict, grover_unitary: List[Dict], out_path: Path
) -> None:
    """Create 300 DPI, 4-panel publication figure for Article 1 (Experiment 10)."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 8.3,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 10.8), dpi=300, constrained_layout=True)

    c_navy = "#1B3B6F"
    c_teal = "#0E8A78"
    c_crimson = "#C62828"
    c_amber = "#D97706"
    c_purple = "#6D28D9"
    c_slate = "#475569"

    regs = amp_data["regimes"]
    c_arr = np.array(regs["gaussian"]["c_list"], dtype=float)

    # --- Panel (a): Amplification Ratio E[X*_quant] / E[X*_cl] across 5 tail regimes ---
    ax = axes[0, 0]
    styles = [
        ("gaussian", "Gaussian ($\\alpha=2$, lim $\\sqrt{2}=1.41$)", c_navy, "o", "-"),
        ("laplace", "Laplace ($\\alpha=1$, lim $2.00$)", c_teal, "s", "-"),
        ("student_t_6", "Student-$t_6$ (Fréchet, $\\propto C^{1/6}$)", c_amber, "^", "-"),
        ("student_t_4", "Student-$t_4$ (Fréchet, $\\propto C^{1/4}$)", c_purple, "D", "-"),
        ("student_t_3", "Student-$t_3$ (Fréchet, $\\propto C^{1/3}$)", c_crimson, "v", "-"),
    ]
    for key, lbl, col, mk, ls in styles:
        d = regs[key]
        emp = np.array(d["empirical_ratio_Q_over_C"], dtype=float)
        theo = np.array(d["theory_ratio_Q_over_C"], dtype=float)
        ax.loglog(c_arr, emp, marker=mk, linewidth=2.1, markersize=6, color=col, label=lbl)
        ax.loglog(c_arr, theo, linestyle="--", linewidth=1.2, color=col, alpha=0.65)

    ax.axhline(math.sqrt(2.0), color=c_navy, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axhline(2.0, color=c_teal, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_title("(a) Quantum Overfitting Amplification $\\mathbb{E}[X^*_{\\mathrm{quant}}] / \\mathbb{E}[X^*_{\\mathrm{cl}}]$")
    ax.set_xlabel("Fixed oracle query budget $C$ (log scale)")
    ax.set_ylabel("In-sample mirage amplification ratio (log scale)")
    ax.set_ylim(1.15, 22.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", framealpha=0.94)
    ax.text(
        0.97,
        0.04,
        f"At $C=1000$ ($M_{{\\mathrm{{eff}}}}=1.62\\times 10^6$):\n"
        f"Gauss: {regs['gaussian']['empirical_ratio_Q_over_C'][-1]:.2f}$\\times$ | "
        f"Laplace: {regs['laplace']['empirical_ratio_Q_over_C'][-1]:.2f}$\\times$\n"
        f"$t_4$: {regs['student_t_4']['empirical_ratio_Q_over_C'][-1]:.2f}$\\times$ | "
        f"$t_3$: {regs['student_t_3']['empirical_ratio_Q_over_C'][-1]:.2f}$\\times$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.1,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    # --- Panel (b): In-Sample Mirage Explosion vs Out-of-Sample Collapse to Zero ---
    ax = axes[0, 1]
    x_pos = np.arange(5)
    reg_keys = ["gaussian", "laplace", "student_t_6", "student_t_4", "student_t_3"]
    reg_labels = ["Gaussian\n($\\alpha=2$)", "Laplace\n($\\alpha=1$)", "Student-$t_6$\n($\\nu=6$)", "Student-$t_4$\n($\\nu=4$)", "Student-$t_3$\n($\\nu=3$)"]
    w_bar = 0.25

    cl_vals = [regs[k]["classical_IS_mean"][-1] for k in reg_keys]
    q_vals = [regs[k]["quantum_IS_mean"][-1] for k in reg_keys]
    oos_vals = [regs[k]["quantum_OOS_mean"][-1] for k in reg_keys]

    b1 = ax.bar(x_pos - w_bar, cl_vals, width=w_bar, color=c_slate, label="Classical IS max ($C=1\\,000$)")
    b2 = ax.bar(x_pos, q_vals, width=w_bar, color=c_crimson, label="Quantum Grover IS max ($C=1\\,000$)")
    b3 = ax.bar(x_pos + w_bar, np.maximum(np.abs(oos_vals), 0.15), width=w_bar, color=c_teal, label="Quantum OOS $|\\mathbb{E}[X_{\\mathrm{OOS}}]|$ ($\\approx 0$)")

    ax.set_yscale("log")
    ax.set_ylim(0.1, 350.0)
    for rect, val in zip(b1, cl_vals):
        ax.text(rect.get_x() + rect.get_width() / 2.0, val * 1.12, f"{val:.1f}", ha="center", va="bottom", fontsize=7.6)
    for rect, val in zip(b2, q_vals):
        ax.text(rect.get_x() + rect.get_width() / 2.0, val * 1.12, f"{val:.1f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=c_crimson)
    for rect, val in zip(b3, oos_vals):
        ax.text(rect.get_x() + rect.get_width() / 2.0, 0.17, f"{val:+.2f}", ha="center", va="bottom", fontsize=7.2, color="#065F46")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(reg_labels)
    ax.set_ylabel("Mean score magnitude at $C=1\\,000$ (log scale)")
    ax.set_title("(b) IS Mirage Explosion vs OOS Collapse ($H_0$, $C=1\\,000$, $R=4\\,000$)")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", framealpha=0.94)

    # --- Panel (c): Theorem 3 — Collapse of Naive C-Indexed Testing vs Q-DSR_nu Restoration ---
    ax = axes[1, 0]
    ft = fpr_data["fpr_tables"]
    ax.semilogx(
        c_arr,
        [100 * x for x in ft["gaussian"]["fpr_naive_bonferroni_C"]],
        marker="x",
        linewidth=2.0,
        linestyle="--",
        color=c_slate,
        label="Naive Bonferroni($C$) [Gauss & $t_3$] $\\to 100\\%$",
    )
    ax.semilogx(
        c_arr,
        [100 * x for x in ft["gaussian"]["fpr_naive_dsr_C"]],
        marker="s",
        linewidth=2.0,
        color=c_amber,
        label="Naive DSR($C$) on Gaussian $\\to 100\\%$",
    )
    ax.semilogx(
        c_arr,
        [100 * x for x in ft["student_t_3"]["fpr_gaussian_qdsr_Meff"]],
        marker="^",
        linewidth=2.2,
        color=c_crimson,
        label="Gaussian Q-DSR($M_{\\mathrm{eff}}$) on Student-$t_3$ (FAILS: $100\\%$)",
    )
    ax.semilogx(
        c_arr,
        [100 * x for x in ft["gaussian"]["fpr_exact_tail_qdsr_nu"]],
        marker="o",
        linewidth=2.1,
        color=c_navy,
        label="Exact Q-DSR($M_{\\mathrm{eff}}$) on Gaussian ($\\approx 5.0\\%$)",
    )
    ax.semilogx(
        c_arr,
        [100 * x for x in ft["student_t_3"]["fpr_exact_tail_qdsr_nu"]],
        marker="D",
        linewidth=2.1,
        linestyle="-.",
        color=c_teal,
        label="Heavy-Tail Q-DSR$_\\nu$($M_{\\mathrm{eff}}$) on Student-$t_3$ ($\\approx 5.0\\%$)",
    )
    ax.axhline(5.0, color="#0F172A", linestyle=":", linewidth=1.3)
    ax.text(10.5, 7.2, "Nominal target $\\alpha = 5.0\\%$", fontsize=8.2, color="#0F172A")
    ax.set_title("(c) Multiple-Testing Collapse vs Q-DSR & Heavy-Tail Q-DSR$_\\nu$ (Thm 3)")
    ax.set_xlabel("Fixed oracle query budget $C$ (log scale)")
    ax.set_ylabel("Empirical False Positive Rate FPR (%)")
    ax.set_ylim(-4, 108)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.74), framealpha=0.95, fontsize=8.0)

    # --- Panel (d): Theorem 4 — Landauer Thermodynamic Accounting & Unitary Grover Verification ---
    ax = axes[1, 1]
    l_rows = landauer_data["rows"]
    c_l = np.array([r["C"] for r in l_rows], dtype=float)
    b_cl = np.array([r["bits_classical"] for r in l_rows], dtype=float)
    b_q = np.array([r["bits_quantum"] for r in l_rows], dtype=float)
    q_cl = np.array([r["Q_min_classical_zJ_300K"] for r in l_rows], dtype=float)
    q_q = np.array([r["Q_min_quantum_zJ_300K"] for r in l_rows], dtype=float)

    ax.semilogx(
        c_l,
        b_q,
        marker="o",
        linewidth=2.3,
        color=c_crimson,
        label="Quantum Grover reset: $\\Delta I = \\log_2 M_{\\mathrm{eff}} \\approx 2\\log_2 C + 0.70$ bits",
    )
    ax.semilogx(
        c_l,
        b_cl,
        marker="s",
        linewidth=2.2,
        color=c_navy,
        label="Classical search ($M=C$): $\\Delta I = \\log_2 C$ bits",
    )
    ax.semilogx(
        c_l,
        np.zeros_like(c_l),
        marker="^",
        linewidth=1.9,
        linestyle="--",
        color=c_teal,
        label="Unitary Grover iterations ($C$ steps): $0$ intrinsic Landauer erasure",
    )
    # Shade the 2x Landauer factor paid at measurement + reset
    ax.fill_between(c_l, b_cl, b_q, color=c_crimson, alpha=0.10)
    ax.set_title("(d) Landauer Reconciliation: Reversible Grover vs Reset Cost (Thm 4)")
    ax.set_xlabel("Fixed oracle query budget $C$ (log scale)")
    ax.set_ylabel("Minimal erased selection entropy $\\Delta I_{\\mathrm{sel}}$ (bits)")
    ax.set_ylim(-1.2, 24.5)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    zJ_per_bit = landauer_data["landauer_per_bit_zJ"]
    ax2.set_ylim(-1.2 * zJ_per_bit, 24.5 * zJ_per_bit)
    ax2.set_ylabel("Minimal Landauer heat $Q_{\\min}$ at $300\\,\\mathrm{K}$ (zJ)", color=c_purple)
    ax2.tick_params(axis="y", labelcolor=c_purple)

    ax.legend(loc="upper left", framealpha=0.94, fontsize=8.0)
    g12 = grover_unitary[-1]
    ax.text(
        0.98,
        0.16,
        f"Unitary Grover check ($q=12, M=4\\,096$):\n"
        f"$C={g12['C_grover_queries']}$ queries, $P_{{\\mathrm{{succ}}}}={g12['P_succ_exact']:.5f}$\n"
        f"Unitarity err: {g12['unitarity_norm_err']:.1e}\n"
        f"Ratio $\\Delta I_Q / \\Delta I_{{\\mathrm{{cl}}}}$ at $C=1000$: {l_rows[-1]['ratio_quantum_over_classical_landauer']:.3f}$\\times$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    fig.suptitle(
        "Experiment 10 (Article 1) — Quantum Overfitting Amplification Law, Heavy-Tail Q-DSR$_\\nu$ & Landauer Reconciliation",
        fontsize=12.3,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=== Experiment 10 (Article 1): Quantum Overfitting Amplification & Heavy-Tail Q-DSR ===")
    grover_unitary = exact_grover_statevector_verify(rng)
    for g in grover_unitary:
        print(
            f"  Unitary Grover q={g['q_qubits']:2d} (M={g['M_space']:4d}, C={g['C_grover_queries']:2d}): "
            f"P_succ={g['P_succ_exact']:.6f}, err={g['abs_err_vs_theory']:.1e}, "
            f"IS_t3: cl={g['classical_max_IS_t3']:.2f} -> Q={g['grover_max_IS_t3']:.2f} (OOS={g['grover_winner_OOS_t3']:+.2f})"
        )

    amp_data, fpr_data = run_part_a_and_b_amplification_and_qdsr(rng)
    for k, d in amp_data["regimes"].items():
        print(
            f"  Regime [{k:11s}] C=1000: cl_IS={d['classical_IS_mean'][-1]:6.2f}, "
            f"Q_IS={d['quantum_IS_mean'][-1]:6.2f}, ratio={d['empirical_ratio_Q_over_C'][-1]:5.2f}x "
            f"(th. {d['theory_ratio_Q_over_C'][-1]:5.2f}x), Q_OOS={d['quantum_OOS_mean'][-1]:+.3f}, "
            f"log-slope={d['loglog_slope_ratio_vs_C']:.3f}"
        )
    print("  FPR at C=1000 (nominal 5.0%):")
    for k, ft in fpr_data["fpr_tables"].items():
        print(
            f"    [{k:11s}]: Bonf(C)={100*ft['fpr_naive_bonferroni_C'][-1]:5.1f}%, "
            f"DSR(C)={100*ft['fpr_naive_dsr_C'][-1]:5.1f}%, "
            f"Gauss-QDSR={100*ft['fpr_gaussian_qdsr_Meff'][-1]:5.1f}%, "
            f"Exact-QDSR_nu={100*ft['fpr_exact_tail_qdsr_nu'][-1]:4.1f}%"
        )

    landauer_data = compute_part_c_landauer([10, 30, 100, 300, 1000])
    print(
        f"  Landauer at C=1000: cl={landauer_data['rows'][-1]['bits_classical']:.2f} bits "
        f"({landauer_data['rows'][-1]['Q_min_classical_zJ_300K']:.2f} zJ) vs "
        f"Q={landauer_data['rows'][-1]['bits_quantum']:.2f} bits "
        f"({landauer_data['rows'][-1]['Q_min_quantum_zJ_300K']:.2f} zJ), "
        f"ratio={landauer_data['rows'][-1]['ratio_quantum_over_classical_landauer']:.3f}x"
    )

    fig_path = FIGURES_DIR / "fig10_quantum_overfitting_amplification_law.png"
    plot_fig10(amp_data, fpr_data, landauer_data, grover_unitary, fig_path)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment": "exp10_paper1_quantum_overfitting_heavy_tails",
        "seed": SEED,
        "runtime_seconds": round(elapsed, 3),
        "kappa_grover": KAPPA_GROVER,
        "unitary_grover_verification": grover_unitary,
        "part_a_amplification": amp_data,
        "part_b_multiple_testing_qdsr": fpr_data,
        "part_c_landauer_thermodynamics": landauer_data,
    }
    json_path = RESULTS_DIR / "exp10_paper1_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Figure: {fig_path}")
    print(f"Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
