#!/usr/bin/env python3
"""
Experiment 4 — Empirical Benchmark of the Post-Selection Illusion Score (SIPS)
==============================================================================
Doctoral Thesis Companion Suite: Delayed-Choice Quantum Eraser, Post-Selection & AI
Tests Hypothesis H_2.3 (Chapter 2, Section 2.3) and addresses PhD Jury Vigilance
Point #2 & Defense Question #3 (Chapter 5):
  - Evaluates K = 240 balanced scenarios across 4 structural regimes (60 each):
      1. True Invariant Causal Mechanisms (ground-truth label y = 1)
      2. Shortcut Learning (mixed causal + dominant spurious shortcut, y = 0)
      3. Berkson Collider Selection Bias (stochastic logistic gate + IPSW audit, y = 0)
      4. Pure Data Dredging / Post-Selection Overfitting (M in {100, 1000, 5000}, y = 0)
  - All 4 regimes are calibrated to occupy an overlapping, deceptively high
    i.i.d. 5-fold CV accuracy band (~0.74-0.88), reflecting the realistic audit
    dilemma where standard i.i.d. CV behaves near chance (AUROC ~ 0.50).
  - Compares standard i.i.d. 5-fold Cross-Validation vs. SIPS composite score,
    SIPS + Chapter 2 Definition 2.2 veto rule, Table 2.3 4-step veto rule,
    and single-pillar ablations (I_mul, I_OOD, I_inv, I_col).
  - Computes exact AUROC, AUPRC, Causal FPR at 90% TPR, and exact correlated
    DeLong test (DeLong, DeLong & Clarke-Pearson, 1988).
  - Performs formal sensitivity analyses over:
      (a) simplex weight vectors w = (w_mul, w_OOD, w_inv, w_col),
      (b) decision/veto thresholds (tau_SIPS, eta_veto),
      (c) imperfect / misspecified propensity scores pi_S in I_col.

Hardware target: Apple Silicon (macOS ARM64, CPU NumPy/SciPy/scikit-learn).
Outputs:
  - sim/results/exp4_sips_results.json
  - sim/figures/fig4_sips_roc_benchmark.png (300 DPI, 4 panels)

Usage:
  python3 sim/exp4_sips_benchmark.py
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Reproducibility & Paths
# ---------------------------------------------------------------------------
SEED = 42
N_PER_REGIME = 60
TOTAL_SCENARIOS = 4 * N_PER_REGIME  # K = 240
EPS = 1e-6

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_JSON = RESULTS_DIR / "exp4_sips_results.json"
FIGURE_PNG = FIGURES_DIR / "fig4_sips_roc_benchmark.png"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class ScenarioRecord:
    """Stores observable audit statistics and ground-truth outcome for one scenario."""

    scenario_id: int
    regime: str
    is_causal: int  # 1 = true causal invariant mechanism, 0 = post-selection illusion
    n_train: int
    n_candidates_M: int
    acc_is: float
    acc_cv_iid: float
    acc_ood_audit: float
    acc_ood_true: float
    delta_is: float
    tau_null_95: float
    theta_raw: float
    theta_ipsw: float
    i_mul: float
    i_ood: float
    i_inv: float
    i_col: float
    sips_uniform: float
    veto_2pillar: int  # Ch. 2 Def. 2.2: max(I_mul, I_OOD) > 0.75
    veto_4step: int  # Ch. 2 Table 2.3: I_mul>0.75 or I_OOD>0.75 or I_inv>0.50 or I_col>0.50
    pred_def22_causal: int  # 1 iff sips <= 0.50 and veto_2pillar == 0
    pred_table23_causal: int  # 1 iff sips <= 0.50 and veto_4step == 0
    sips_eff_2pillar: float  # continuous veto-penalized score (Def. 2.2)
    sips_eff_4step: float  # continuous 4-pillar veto-penalized score (Table 2.3)


# ---------------------------------------------------------------------------
# Exact Correlated DeLong Test (DeLong, DeLong & Clarke-Pearson, 1988)
# ---------------------------------------------------------------------------
def _mann_whitney_placements(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute exact Mann-Whitney AUROC and placement vectors V10, V01."""
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    m = len(pos_scores)
    n = len(neg_scores)
    if m == 0 or n == 0:
        raise ValueError("Both positive and negative classes must be non-empty.")

    diff = pos_scores[:, None] - neg_scores[None, :]
    psi = (diff > 0.0).astype(np.float64) + 0.5 * (diff == 0.0).astype(np.float64)

    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    auc = float(v10.mean())
    return auc, v10, v01


def delong_single_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, alpha: float = 0.05
) -> dict[str, float]:
    """Exact DeLong variance and (1 - alpha) confidence interval for a single AUROC."""
    auc, v10, v01 = _mann_whitney_placements(y_true, y_score)
    m, n = len(v10), len(v01)
    s10 = float(np.var(v10, ddof=1))
    s01 = float(np.var(v01, ddof=1))
    var_auc = s10 / m + s01 / n
    se = math.sqrt(max(var_auc, 0.0))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "auc": auc,
        "se": se,
        "ci_low": float(np.clip(auc - z_crit * se, 0.0, 1.0)),
        "ci_high": float(np.clip(auc + z_crit * se, 0.0, 1.0)),
    }


def delong_paired_test(
    y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray
) -> dict[str, float]:
    """Exact paired DeLong test comparing AUROC(score_a) vs AUROC(score_b)."""
    auc_a, v10_a, v01_a = _mann_whitney_placements(y_true, score_a)
    auc_b, v10_b, v01_b = _mann_whitney_placements(y_true, score_b)

    sk_a = float(roc_auc_score(y_true, score_a))
    sk_b = float(roc_auc_score(y_true, score_b))
    assert abs(auc_a - sk_a) < 1e-12, f"DeLong mismatch A: {auc_a} vs {sk_a}"
    assert abs(auc_b - sk_b) < 1e-12, f"DeLong mismatch B: {auc_b} vs {sk_b}"

    m, n = len(v10_a), len(v01_a)
    v10 = np.vstack([v10_a, v10_b])
    v01 = np.vstack([v01_a, v01_b])

    s10 = np.cov(v10, ddof=1)
    s01 = np.cov(v01, ddof=1)
    cov = s10 / m + s01 / n

    diff = auc_a - auc_b
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se_diff = math.sqrt(max(var_diff, 1e-30))
    z_stat = diff / se_diff
    p_value = float(2.0 * stats.norm.sf(abs(z_stat)))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "diff": float(diff),
        "se_diff": se_diff,
        "z_stat": float(z_stat),
        "p_value": p_value,
        "ci95_diff_low": float(diff - 1.96 * se_diff),
        "ci95_diff_high": float(diff + 1.96 * se_diff),
    }


def fpr_at_target_tpr(
    y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.90
) -> float:
    """Compute exact minimum FPR achieving TPR >= target_tpr along the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= target_tpr - 1e-12)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def evaluate_score_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """Compute AUROC (with DeLong 95% CI), AUPRC, and Causal FPR at 90% TPR."""
    dl = delong_single_auc_ci(y_true, y_score)
    auprc = float(average_precision_score(y_true, y_score))
    fpr90 = fpr_at_target_tpr(y_true, y_score, target_tpr=0.90)
    return {
        "auroc": dl["auc"],
        "auroc_se": dl["se"],
        "auroc_ci95_low": dl["ci_low"],
        "auroc_ci95_high": dl["ci_high"],
        "auprc": auprc,
        "fpr_at_90_tpr": fpr90,
    }


def summarize_binary_rule(
    y_true: np.ndarray, pred: np.ndarray, records: list[ScenarioRecord]
) -> dict[str, Any]:
    """Compute exact confusion matrix, TPR, Causal FPR, Precision, and F1."""
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "tpr": float(tp / max(tp + fn, 1)),
        "causal_fpr": float(fp / max(fp + tn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "false_positives_by_regime": {
            reg: int(sum(int(pred[i]) for i, r in enumerate(records) if r.regime == reg))
            for reg in ["Shortcut", "Collider", "Data_Dredging"]
        },
    }


# ---------------------------------------------------------------------------
# Fast Model Fitting, CV, Null Multiplicity Calibration & IPSW Audit
# ---------------------------------------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, c_reg: float = 0.25) -> LogisticRegression:
    """Fit L2-regularized logistic regression deterministically."""
    clf = LogisticRegression(C=c_reg, solver="lbfgs", max_iter=300)
    clf.fit(X, y)
    return clf


def stratified_cv_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    c_reg: float = 0.25,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Compute stratified K-fold out-of-fold predicted probabilities and accuracy."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in skf.split(X, y):
        clf = fit_logistic(X[tr_idx], y[tr_idx], c_reg=c_reg)
        oof_prob[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    oof_acc = float(((oof_prob >= 0.5).astype(int) == y).mean())
    return oof_prob, oof_acc


def select_top_k_features(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Select top-k features by absolute Pearson correlation with y (vectorized)."""
    y_centered = y.astype(np.float64) - y.mean()
    y_norm = np.linalg.norm(y_centered) + 1e-12
    x_centered = X - X.mean(axis=0, keepdims=True)
    x_norms = np.linalg.norm(x_centered, axis=0) + 1e-12
    corrs = np.abs((y_centered @ x_centered) / (y_norm * x_norms))
    if k >= X.shape[1]:
        return np.arange(X.shape[1])
    top_idx = np.argpartition(corrs, -k)[-k:]
    top_idx = top_idx[np.argsort(corrs[top_idx])[::-1]]
    return top_idx


def calibrate_null_threshold_95(
    n_train: int,
    n_candidates_m: int,
    k_selected: int,
    c_reg: float,
    rng: np.random.Generator,
    n_perm: int = 160,
) -> float:
    """
    Calibrate the 95th-percentile null in-sample accuracy gain tau_null^(0.95)
    under H_0 (pure noise) for the EXACT selection pipeline.
    """
    gains = np.empty(n_perm, dtype=np.float64)
    k_eff = min(k_selected, n_candidates_m)

    for b in range(n_perm):
        y_null = np.zeros(n_train, dtype=int)
        y_null[: n_train // 2] = 1
        rng.shuffle(y_null)
        y_c = y_null.astype(np.float64) - y_null.mean()
        y_unit = y_c / (np.linalg.norm(y_c) + 1e-12)

        if n_candidates_m <= 60:
            X_null = rng.standard_normal((n_train, n_candidates_m))
            if k_eff < n_candidates_m:
                sel = select_top_k_features(X_null, y_null, k_eff)
                X_sel = X_null[:, sel]
            else:
                X_sel = X_null
        else:
            # Exact order-statistic equivalence for M independent isotropic Gaussian
            # features: r_j^2 ~ Beta(1/2, (n-2)/2) independently for j = 1..M.
            df = n_train - 2
            r_sq = rng.beta(0.5, df / 2.0, size=n_candidates_m)
            top_r_sq = np.partition(r_sq, -k_eff)[-k_eff:]
            signs = rng.choice([-1.0, 1.0], size=k_eff)
            top_r = signs * np.sqrt(top_r_sq)

            Z_orth = rng.standard_normal((n_train, k_eff))
            Z_orth -= Z_orth.mean(axis=0, keepdims=True)
            Z_orth -= np.outer(y_unit, y_unit @ Z_orth)
            Z_orth /= np.linalg.norm(Z_orth, axis=0, keepdims=True) + 1e-12
            X_sel = math.sqrt(n_train) * (
                np.outer(y_unit, top_r) + Z_orth * np.sqrt(1.0 - top_r_sq)[None, :]
            )

        clf = fit_logistic(X_sel, y_null, c_reg=c_reg)
        acc = float((clf.predict(X_sel) == y_null).mean())
        gains[b] = acc - 0.5

    return float(np.quantile(gains, 0.95))


def compute_i_inv(
    clf: LogisticRegression,
    X_eval: np.ndarray,
    non_causal_cols: list[int],
    rng: np.random.Generator,
    n_transforms: int = 8,
) -> float:
    """
    Compute Invariance & Counterfactual Ablation Fragility Index I_inv (Eq. 2.39).
    Applies label-preserving perturbations g in T_nc (independent column-wise
    permutations and counterfactual zeroing/sign-flips of non-causal features).
    """
    if len(non_causal_cols) == 0:
        return 0.0

    p_orig = clf.predict_proba(X_eval)[:, 1]
    n = len(p_orig)
    perm_ref = rng.permutation(n)
    denom = float(np.mean(np.abs(p_orig - p_orig[perm_ref]))) + EPS

    max_tv = 0.0
    for t_idx in range(n_transforms):
        X_pert = X_eval.copy()
        if t_idx < n_transforms // 2:
            for col in non_causal_cols:
                X_pert[:, col] = X_pert[rng.permutation(n), col]
        else:
            for col in non_causal_cols:
                X_pert[:, col] = 0.0 if (t_idx % 2 == 0) else -X_pert[:, col]
        p_pert = clf.predict_proba(X_pert)[:, 1]
        tv = float(np.mean(np.abs(p_orig - p_pert)))
        if tv > max_tv:
            max_tv = tv

    return float(np.clip(max_tv / denom, 0.0, 1.0))


def compute_i_col_ipsw(
    X_sel: np.ndarray,
    y_sel: np.ndarray,
    oof_prob_sel: np.ndarray,
    X_aux: np.ndarray,
    y_aux: np.ndarray,
    s_aux: np.ndarray,
    pi_min: float = 0.005,
    propensity_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Compute Collider / Selection Bias Sensitivity Index I_col (Eq. 2.40) via
    empirical Inverse Probability of Selection Weighting (IPSW / Hajek estimator).
    """
    correct = ((oof_prob_sel >= 0.5).astype(int) == y_sel).astype(np.float64)
    theta_raw = float(np.mean(correct) - 0.5)

    s_rate = float(np.mean(s_aux))
    if s_rate > 0.995 or s_rate < 0.005:
        return 0.0, theta_raw, theta_raw

    Z_aux = np.column_stack([X_aux, y_aux.astype(np.float64)])
    Z_sel = np.column_stack([X_sel, y_sel.astype(np.float64)])
    prop_clf = LogisticRegression(C=5.0, solver="lbfgs", max_iter=300)
    prop_clf.fit(Z_aux, s_aux)

    logits_sel = prop_clf.decision_function(Z_sel)
    if propensity_noise_std > 0.0 and rng is not None:
        logits_sel = logits_sel + rng.normal(0.0, propensity_noise_std, size=len(logits_sel))
    pi_hat = np.clip(expit(logits_sel), pi_min, 1.0)

    weights = 1.0 / pi_hat
    theta_ipsw = float(np.sum(weights * (correct - 0.5)) / np.sum(weights))

    i_col = float(np.clip(abs(theta_raw - theta_ipsw) / (abs(theta_raw) + EPS), 0.0, 1.0))
    return i_col, theta_raw, theta_ipsw


def _apply_benign_covariate_sampling(
    X_pool: np.ndarray, y_pool: np.ndarray, n_needed: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Benign non-collider covariate sampling S <- X_0 (identical across non-collider regimes)."""
    p_sel = expit(0.35 * X_pool[:, 0] + 0.6)
    keep = rng.uniform(0.0, 1.0, size=len(y_pool)) < p_sel
    idx = np.where(keep)[0][:n_needed]
    return X_pool[idx], y_pool[idx]


def _finalize_record(
    scenario_id: int,
    regime: str,
    is_causal: int,
    n_train: int,
    n_candidates_m: int,
    acc_is: float,
    acc_cv: float,
    acc_ood_audit: float,
    acc_ood_true: float,
    delta_is: float,
    tau_null: float,
    theta_raw: float,
    theta_ipsw: float,
    i_mul: float,
    i_ood: float,
    i_inv: float,
    i_col: float,
) -> ScenarioRecord:
    """Assemble SIPS composite, 2-pillar veto (Def. 2.2), and 4-step veto (Table 2.3)."""
    sips = 0.25 * (i_mul + i_ood + i_inv + i_col)
    veto_2 = int(max(i_mul, i_ood) > 0.75)
    veto_4 = int((i_mul > 0.75) or (i_ood > 0.75) or (i_inv > 0.50) or (i_col > 0.50))
    pred_def22 = int((sips <= 0.50) and (veto_2 == 0))
    pred_tab23 = int((sips <= 0.50) and (veto_4 == 0))
    sips_eff_2 = float(max(sips, i_mul, i_ood) if veto_2 else sips)
    max_4_scaled = max(i_mul, i_ood, min(1.0, i_inv * 1.15), min(1.0, i_col * 1.15))
    sips_eff_4 = float(max(sips, max_4_scaled) if veto_4 else sips)
    return ScenarioRecord(
        scenario_id=scenario_id,
        regime=regime,
        is_causal=is_causal,
        n_train=n_train,
        n_candidates_M=n_candidates_m,
        acc_is=acc_is,
        acc_cv_iid=acc_cv,
        acc_ood_audit=acc_ood_audit,
        acc_ood_true=acc_ood_true,
        delta_is=delta_is,
        tau_null_95=tau_null,
        theta_raw=theta_raw,
        theta_ipsw=theta_ipsw,
        i_mul=i_mul,
        i_ood=i_ood,
        i_inv=i_inv,
        i_col=i_col,
        sips_uniform=sips,
        veto_2pillar=veto_2,
        veto_4step=veto_4,
        pred_def22_causal=pred_def22,
        pred_table23_causal=pred_tab23,
        sips_eff_2pillar=sips_eff_2,
        sips_eff_4step=sips_eff_4,
    )


# ---------------------------------------------------------------------------
# Regime Generators (4 Regimes x 60 Scenarios = 240 Scenarios)
# ---------------------------------------------------------------------------
def run_regime_1_causal(
    scenario_id: int,
    rng: np.random.Generator,
    null_cache: dict[tuple[int, int, int, float], float],
) -> tuple[ScenarioRecord, dict[str, Any]]:
    """
    Regime 1: True Invariant Causal Mechanism (y_true = 1).
    Invariant structural law P(Y | X_c) with realistic difficulty spectrum
    (i.i.d. CV ~ 0.74-0.88), benign covariate shift + extra observation noise
    in the small audit OOD sample (producing realistic borderline I_OOD values),
    and benign non-collider covariate sampling S <- X_0.
    """
    n_train = int(rng.choice([260, 320, 380]))
    n_ood_audit = 70
    n_ood_true = 1000
    d_c, d_nc = 3, 5
    d_tot = d_c + d_nc
    c_reg = 0.25

    signal_scale = float(rng.uniform(1.70, 3.05))
    beta_c = rng.normal(0.0, 1.0, size=d_c)
    beta_c = signal_scale * beta_c / (np.linalg.norm(beta_c) + 1e-12)
    # Small harmless correlation of one non-causal feature in training (realistic weak confounding)
    weak_nc_corr = float(rng.uniform(0.0, 0.28))

    def sample_env(
        n: int, shift_c: float, shift_nc: float, extra_noise: float, nc_corr: float
    ) -> tuple[np.ndarray, np.ndarray]:
        X_c = rng.normal(shift_c, 1.0, size=(n, d_c))
        logits = X_c @ beta_c + rng.normal(0.0, extra_noise, size=n)
        y = (rng.uniform(0.0, 1.0, size=n) < expit(logits)).astype(int)
        X_nc = rng.normal(shift_nc, 1.0, size=(n, d_nc))
        X_nc[:, 0] += nc_corr * (2.0 * y - 1.0)
        return np.hstack([X_c, X_nc]), y

    n_pool = int(n_train * 2.5)
    X_pool, y_pool = sample_env(n_pool, 0.0, 0.0, 0.25, weak_nc_corr)
    X_tr, y_tr = _apply_benign_covariate_sampling(X_pool, y_pool, n_train, rng)

    X_aux, y_aux = sample_env(1500, 0.0, 0.0, 0.25, weak_nc_corr)
    s_aux = (rng.uniform(0.0, 1.0, size=1500) < expit(0.35 * X_aux[:, 0] + 0.6)).astype(int)

    shift_val = float(rng.uniform(-0.8, 0.8))
    audit_noise = float(rng.uniform(0.45, 1.40))
    X_audit, y_audit = sample_env(n_ood_audit, shift_val, 1.2, audit_noise, 0.0)
    X_true, y_true_env = sample_env(n_ood_true, -shift_val, -1.2, 0.35, 0.0)

    clf = fit_logistic(X_tr, y_tr, c_reg=c_reg)
    acc_is = float((clf.predict(X_tr) == y_tr).mean())
    oof_prob, acc_cv = stratified_cv_oof_predictions(X_tr, y_tr, c_reg=c_reg, seed=scenario_id)
    acc_ood_audit = float((clf.predict(X_audit) == y_audit).mean())
    acc_ood_true = float((clf.predict(X_true) == y_true_env).mean())

    delta_is = max(acc_is - 0.5, EPS)
    cache_key = (n_train, d_tot, d_tot, c_reg)
    if cache_key not in null_cache:
        null_cache[cache_key] = calibrate_null_threshold_95(
            n_train, d_tot, d_tot, c_reg, rng
        )
    tau_null = null_cache[cache_key]

    i_mul = float(np.clip(tau_null / delta_is, 0.0, 1.0))
    i_ood = float(np.clip(1.0 - (acc_ood_audit - 0.5) / delta_is, 0.0, 1.0))
    i_inv = compute_i_inv(clf, X_tr, non_causal_cols=list(range(d_c, d_tot)), rng=rng)
    i_col, theta_raw, theta_ipsw = compute_i_col_ipsw(
        X_tr, y_tr, oof_prob, X_aux, y_aux, s_aux
    )

    rec = _finalize_record(
        scenario_id, "Causal", 1, n_train, d_tot,
        acc_is, acc_cv, acc_ood_audit, acc_ood_true, delta_is, tau_null,
        theta_raw, theta_ipsw, i_mul, i_ood, i_inv, i_col,
    )
    return rec, {
        "X_tr": X_tr, "y_tr": y_tr, "oof_prob": oof_prob,
        "X_aux": X_aux, "y_aux": y_aux, "s_aux": s_aux,
    }


def run_regime_2_shortcut(
    scenario_id: int,
    rng: np.random.Generator,
    null_cache: dict[tuple[int, int, int, float], float],
) -> tuple[ScenarioRecord, dict[str, Any]]:
    """
    Regime 2: Shortcut Learning (y_true = 0).
    Weak-to-moderate causal feature overshadowed by a stronger spurious non-causal
    shortcut (i.i.d. CV ~ 0.74-0.88 matching Causal). Audit OOD sometimes only
    partially weakens the shortcut (challenging I_OOD alone), whereas true OOD
    deployment reverses the shortcut and I_inv catches it directly.
    """
    n_train = int(rng.choice([260, 320, 380]))
    n_ood_audit = 70
    n_ood_true = 1000
    d_c, d_nc = 3, 5
    d_tot = d_c + d_nc
    c_reg = 0.25

    # Weak-to-moderate causal component so shortcut isn't caricaturally 1.000
    c_scale = float(rng.uniform(0.25, 0.75))
    beta_c = rng.normal(0.0, 1.0, size=d_c)
    beta_c = c_scale * beta_c / (np.linalg.norm(beta_c) + 1e-12)

    sp_tr = float(rng.uniform(0.58, 1.02))
    if rng.uniform() < 0.50:
        sp_audit = float(rng.uniform(0.25, 0.72)) * sp_tr
    else:
        sp_audit = float(rng.uniform(-0.90, 0.15)) * sp_tr
    sp_true = float(rng.uniform(-1.15, -0.35)) * sp_tr

    def sample_shortcut_env(n: int, sp_s: float) -> tuple[np.ndarray, np.ndarray]:
        X_c = rng.normal(0.0, 1.0, size=(n, d_c))
        y = (rng.uniform(0.0, 1.0, size=n) < expit(X_c @ beta_c)).astype(int)
        y_signed = 2.0 * y - 1.0
        X_nc = rng.normal(0.0, 1.0, size=(n, d_nc))
        X_nc[:, 0] = sp_s * y_signed + rng.normal(0.0, 1.05, size=n)
        X_nc[:, 1] = 0.65 * sp_s * y_signed + rng.normal(0.0, 1.10, size=n)
        return np.hstack([X_c, X_nc]), y

    n_pool = int(n_train * 2.5)
    X_pool, y_pool = sample_shortcut_env(n_pool, sp_tr)
    X_tr, y_tr = _apply_benign_covariate_sampling(X_pool, y_pool, n_train, rng)

    X_audit, y_audit = sample_shortcut_env(n_ood_audit, sp_audit)
    X_true, y_true_env = sample_shortcut_env(n_ood_true, sp_true)

    # Identical benign covariate sampling (no collider)
    X_aux, y_aux = sample_shortcut_env(1500, sp_tr)
    s_aux = (rng.uniform(0.0, 1.0, size=1500) < expit(0.35 * X_aux[:, 0] + 0.6)).astype(int)

    clf = fit_logistic(X_tr, y_tr, c_reg=c_reg)
    acc_is = float((clf.predict(X_tr) == y_tr).mean())
    oof_prob, acc_cv = stratified_cv_oof_predictions(X_tr, y_tr, c_reg=c_reg, seed=scenario_id)
    acc_ood_audit = float((clf.predict(X_audit) == y_audit).mean())
    acc_ood_true = float((clf.predict(X_true) == y_true_env).mean())

    delta_is = max(acc_is - 0.5, EPS)
    cache_key = (n_train, d_tot, d_tot, c_reg)
    if cache_key not in null_cache:
        null_cache[cache_key] = calibrate_null_threshold_95(
            n_train, d_tot, d_tot, c_reg, rng
        )
    tau_null = null_cache[cache_key]

    i_mul = float(np.clip(tau_null / delta_is, 0.0, 1.0))
    i_ood = float(np.clip(1.0 - (acc_ood_audit - 0.5) / delta_is, 0.0, 1.0))
    i_inv = compute_i_inv(clf, X_tr, non_causal_cols=list(range(d_c, d_tot)), rng=rng)
    i_col, theta_raw, theta_ipsw = compute_i_col_ipsw(
        X_tr, y_tr, oof_prob, X_aux, y_aux, s_aux
    )

    rec = _finalize_record(
        scenario_id, "Shortcut", 0, n_train, d_tot,
        acc_is, acc_cv, acc_ood_audit, acc_ood_true, delta_is, tau_null,
        theta_raw, theta_ipsw, i_mul, i_ood, i_inv, i_col,
    )
    return rec, {
        "X_tr": X_tr, "y_tr": y_tr, "oof_prob": oof_prob,
        "X_aux": X_aux, "y_aux": y_aux, "s_aux": s_aux,
    }


def run_regime_3_collider(
    scenario_id: int,
    rng: np.random.Generator,
    null_cache: dict[tuple[int, int, int, float], float],
) -> tuple[ScenarioRecord, dict[str, Any]]:
    """
    Regime 3: Berkson Collider Selection Bias (y_true = 0).
    In the general population, X _|_ Y strictly (true deployment accuracy = 0.50).
    Training data passes a stochastic logistic collider gate:
        P(S = 1 | X, Y) = sigmoid(alpha_0 + w^T X_c + gamma * (2Y - 1))
    spanning moderate-to-strong selection severity (CV ~ 0.72-0.86).
    """
    n_train = int(rng.choice([260, 320, 380]))
    n_ood_audit = 70
    n_ood_true = 1000
    d_c, d_nc = 3, 5
    d_tot = d_c + d_nc
    c_reg = 0.25

    w_col = rng.normal(0.0, 1.0, size=d_c)
    w_col = w_col / (np.linalg.norm(w_col) + 1e-12) * float(rng.uniform(1.65, 2.65))
    gamma_y = float(rng.uniform(1.60, 2.55))
    alpha_0 = float(rng.uniform(-2.1, -1.1))

    def draw_logistic_selected(
        n_needed: int, w_scale: float = 1.0, g_scale: float = 1.0, a0: float = alpha_0
    ) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        collected = 0
        while collected < n_needed:
            batch = max(5000, n_needed * 12)
            X_b = rng.normal(0.0, 1.0, size=(batch, d_tot))
            y_b = rng.integers(0, 2, size=batch)
            p_sel = expit(
                a0 + (w_scale * (X_b[:, :d_c] @ w_col)) + (g_scale * gamma_y) * (2.0 * y_b - 1.0)
            )
            keep = rng.uniform(0.0, 1.0, size=batch) < p_sel
            if np.any(keep):
                xs.append(X_b[keep])
                ys.append(y_b[keep])
                collected += int(np.sum(keep))
        return np.vstack(xs)[:n_needed], np.concatenate(ys)[:n_needed]

    X_tr, y_tr = draw_logistic_selected(n_train, w_scale=1.0, g_scale=1.0, a0=alpha_0)

    n_aux = 2000
    X_aux = rng.normal(0.0, 1.0, size=(n_aux, d_tot))
    y_aux = rng.integers(0, 2, size=n_aux)
    p_aux = expit(alpha_0 + X_aux[:, :d_c] @ w_col + gamma_y * (2.0 * y_aux - 1.0))
    s_aux = (rng.uniform(0.0, 1.0, size=n_aux) < p_aux).astype(int)

    # Audit OOD: 55% milder partial selection hospital (hard for I_OOD alone), 45% unselected
    if rng.uniform() < 0.55:
        X_audit, y_audit = draw_logistic_selected(
            n_ood_audit, w_scale=0.65, g_scale=0.65, a0=alpha_0 * 0.5
        )
    else:
        X_audit = rng.normal(0.0, 1.0, size=(n_ood_audit, d_tot))
        y_audit = rng.integers(0, 2, size=n_ood_audit)

    X_true = rng.normal(0.0, 1.0, size=(n_ood_true, d_tot))
    y_true_env = rng.integers(0, 2, size=n_ood_true)

    clf = fit_logistic(X_tr, y_tr, c_reg=c_reg)
    acc_is = float((clf.predict(X_tr) == y_tr).mean())
    oof_prob, acc_cv = stratified_cv_oof_predictions(X_tr, y_tr, c_reg=c_reg, seed=scenario_id)
    acc_ood_audit = float((clf.predict(X_audit) == y_audit).mean())
    acc_ood_true = float((clf.predict(X_true) == y_true_env).mean())

    delta_is = max(acc_is - 0.5, EPS)
    cache_key = (n_train, d_tot, d_tot, c_reg)
    if cache_key not in null_cache:
        null_cache[cache_key] = calibrate_null_threshold_95(
            n_train, d_tot, d_tot, c_reg, rng
        )
    tau_null = null_cache[cache_key]

    i_mul = float(np.clip(tau_null / delta_is, 0.0, 1.0))
    i_ood = float(np.clip(1.0 - (acc_ood_audit - 0.5) / delta_is, 0.0, 1.0))
    i_inv = compute_i_inv(clf, X_tr, non_causal_cols=list(range(d_c, d_tot)), rng=rng)
    i_col, theta_raw, theta_ipsw = compute_i_col_ipsw(
        X_tr, y_tr, oof_prob, X_aux, y_aux, s_aux
    )

    rec = _finalize_record(
        scenario_id, "Collider", 0, n_train, d_tot,
        acc_is, acc_cv, acc_ood_audit, acc_ood_true, delta_is, tau_null,
        theta_raw, theta_ipsw, i_mul, i_ood, i_inv, i_col,
    )
    return rec, {
        "X_tr": X_tr, "y_tr": y_tr, "oof_prob": oof_prob,
        "X_aux": X_aux, "y_aux": y_aux, "s_aux": s_aux,
    }


def run_regime_4_data_dredging(
    scenario_id: int,
    m_candidates: int,
    rng: np.random.Generator,
    null_cache: dict[tuple[int, int, int, float], float],
) -> tuple[ScenarioRecord, dict[str, Any]]:
    """
    Regime 4: Pure Data Dredging / Post-Selection Overfitting (y_true = 0).
    Pure white noise X ~ N(0, I) independent of Y ~ Bernoulli(0.5).
    Screens M in {100, 1000, 5000} candidate features on frozen training data
    (n_train = 88), selects top k = 5 features, and evaluates naive i.i.d. CV
    (double dipping -> CV ~ 0.72-0.86 overlapping Regime 1).
    """
    n_train = 88
    n_ood_audit = 70
    n_ood_true = 1000
    k_sel = 5
    d_nuisance = 2
    c_reg = 1.0

    X_full = rng.standard_normal((n_train, m_candidates))
    y_tr = np.zeros(n_train, dtype=int)
    y_tr[: n_train // 2] = 1
    rng.shuffle(y_tr)

    top_idx = select_top_k_features(X_full, y_tr, k_sel)
    X_sel_core = X_full[:, top_idx]
    X_nuis_tr = rng.standard_normal((n_train, d_nuisance))
    X_tr = np.hstack([X_sel_core, X_nuis_tr])

    X_audit = rng.standard_normal((n_ood_audit, k_sel + d_nuisance))
    y_audit = rng.integers(0, 2, size=n_ood_audit)
    X_true = rng.standard_normal((n_ood_true, k_sel + d_nuisance))
    y_true_env = rng.integers(0, 2, size=n_ood_true)

    # Identical benign covariate sampling on auxiliary audit sample (no collider)
    X_aux = rng.standard_normal((1500, k_sel + d_nuisance))
    y_aux = rng.integers(0, 2, size=1500)
    s_aux = (rng.uniform(0.0, 1.0, size=1500) < expit(0.35 * X_aux[:, 0] + 0.6)).astype(int)

    clf = fit_logistic(X_tr, y_tr, c_reg=c_reg)
    acc_is = float((clf.predict(X_tr) == y_tr).mean())
    oof_prob, acc_cv = stratified_cv_oof_predictions(X_tr, y_tr, c_reg=c_reg, seed=scenario_id)
    acc_ood_audit = float((clf.predict(X_audit) == y_audit).mean())
    acc_ood_true = float((clf.predict(X_true) == y_true_env).mean())

    delta_is = max(acc_is - 0.5, EPS)
    cache_key = (n_train, m_candidates, k_sel, c_reg)
    if cache_key not in null_cache:
        null_cache[cache_key] = calibrate_null_threshold_95(
            n_train, m_candidates, k_sel, c_reg, rng
        )
    tau_null = null_cache[cache_key]

    i_mul = float(np.clip(tau_null / delta_is, 0.0, 1.0))
    i_ood = float(np.clip(1.0 - (acc_ood_audit - 0.5) / delta_is, 0.0, 1.0))
    i_inv = compute_i_inv(
        clf, X_tr, non_causal_cols=list(range(k_sel, k_sel + d_nuisance)), rng=rng
    )
    i_col, theta_raw, theta_ipsw = compute_i_col_ipsw(
        X_tr, y_tr, oof_prob, X_aux, y_aux, s_aux
    )

    rec = _finalize_record(
        scenario_id, "Data_Dredging", 0, n_train, m_candidates,
        acc_is, acc_cv, acc_ood_audit, acc_ood_true, delta_is, tau_null,
        theta_raw, theta_ipsw, i_mul, i_ood, i_inv, i_col,
    )
    return rec, {
        "X_tr": X_tr, "y_tr": y_tr, "oof_prob": oof_prob,
        "X_aux": X_aux, "y_aux": y_aux, "s_aux": s_aux,
    }


# ---------------------------------------------------------------------------
# Sensitivity Analyses (Weights, Veto Thresholds, Imperfect Propensity IPSW)
# ---------------------------------------------------------------------------
def run_sensitivity_analyses(
    records: list[ScenarioRecord],
    aux_bundles: list[dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Address PhD Jury Vigilance Point #2 & Question #3:
      1. Dirichlet simplex weight sensitivity (250 random convex weights + profiles).
      2. 2D grid over (tau_SIPS, eta_veto) decision & veto thresholds (2-pillar & 4-pillar).
      3. Imperfect/misspecified propensity noise in I_col (sigma_pi in [0, 2.0])
         and structural non-recoverability (I_col omitted).
    """
    y_true = np.array([r.is_causal for r in records], dtype=int)
    cv_score = np.array([r.acc_cv_iid for r in records], dtype=np.float64)
    mat_i = np.array([[r.i_mul, r.i_ood, r.i_inv, r.i_col] for r in records], dtype=np.float64)

    n_dir = 250
    weights_dir = rng.dirichlet(np.ones(4), size=n_dir)
    aucs_full = np.empty(n_dir, dtype=np.float64)
    pvals_full = np.empty(n_dir, dtype=np.float64)
    for idx in range(n_dir):
        w = weights_dir[idx]
        s_w = 1.0 - (mat_i @ w)
        dl = delong_paired_test(y_true, s_w, cv_score)
        aucs_full[idx] = dl["auc_a"]
        pvals_full[idx] = dl["p_value"]

    structured_profiles = {
        "Uniform (0.25, 0.25, 0.25, 0.25)": [0.25, 0.25, 0.25, 0.25],
        "OOD-heavy (0.15, 0.55, 0.15, 0.15)": [0.15, 0.55, 0.15, 0.15],
        "Mul-heavy (0.55, 0.15, 0.15, 0.15)": [0.55, 0.15, 0.15, 0.15],
        "Inv-heavy (0.15, 0.15, 0.55, 0.15)": [0.15, 0.15, 0.55, 0.15],
        "Col-heavy (0.15, 0.15, 0.15, 0.55)": [0.15, 0.15, 0.15, 0.55],
        "No-I_col / Non-recoverable (1/3, 1/3, 1/3, 0)": [1 / 3, 1 / 3, 1 / 3, 0.0],
        "No-I_OOD / Single-env (1/3, 0, 1/3, 1/3)": [1 / 3, 0.0, 1 / 3, 1 / 3],
    }
    profile_results: dict[str, dict[str, float]] = {}
    for name, w_list in structured_profiles.items():
        w_arr = np.array(w_list, dtype=np.float64)
        s_w = 1.0 - (mat_i @ w_arr)
        m_w = evaluate_score_metrics(y_true, s_w)
        dl_w = delong_paired_test(y_true, s_w, cv_score)
        profile_results[name] = {
            **m_w,
            "delong_z_vs_cv": dl_w["z_stat"],
            "delong_p_vs_cv": dl_w["p_value"],
        }

    tau_grid = np.round(np.linspace(0.20, 0.65, 10), 2)
    eta_grid = np.round(np.linspace(0.50, 0.90, 9), 2)
    f1_grid_2p = np.zeros((len(eta_grid), len(tau_grid)), dtype=np.float64)
    fpr_grid_2p = np.zeros((len(eta_grid), len(tau_grid)), dtype=np.float64)
    tpr_grid_2p = np.zeros((len(eta_grid), len(tau_grid)), dtype=np.float64)
    f1_grid_4p = np.zeros((len(eta_grid), len(tau_grid)), dtype=np.float64)
    fpr_grid_4p = np.zeros((len(eta_grid), len(tau_grid)), dtype=np.float64)
    tpr_grid_4p = np.zeros((len(eta_grid), len(tau_grid)), dtype=np.float64)

    sips_u = mat_i.mean(axis=1)
    max_2p = np.maximum(mat_i[:, 0], mat_i[:, 1])
    max_4p = np.max(mat_i, axis=1)

    for i_e, eta in enumerate(eta_grid):
        for i_t, tau in enumerate(tau_grid):
            for max_v, f1_m, tpr_m, fpr_m in [
                (max_2p, f1_grid_2p, tpr_grid_2p, fpr_grid_2p),
                (max_4p, f1_grid_4p, tpr_grid_4p, fpr_grid_4p),
            ]:
                pred = ((sips_u <= tau) & (max_v <= eta)).astype(int)
                tp = int(np.sum((pred == 1) & (y_true == 1)))
                fp = int(np.sum((pred == 1) & (y_true == 0)))
                fn = int(np.sum((pred == 0) & (y_true == 1)))
                tn = int(np.sum((pred == 0) & (y_true == 0)))
                tpr_m[i_e, i_t] = tp / max(tp + fn, 1)
                fpr_m[i_e, i_t] = fp / max(fp + tn, 1)
                f1_m[i_e, i_t] = (2 * tp) / max(2 * tp + fp + fn, 1)

    # Propensity misspecification sensitivity for I_col (Jury Question #3)
    noise_levels = [0.0, 0.25, 0.50, 1.00, 1.50, 2.00]
    propensity_sensitivity: list[dict[str, float]] = []
    for sigma_pi in noise_levels:
        i_col_noisy = np.empty(len(records), dtype=np.float64)
        for idx, bundle in enumerate(aux_bundles):
            ic, _, _ = compute_i_col_ipsw(
                bundle["X_tr"],
                bundle["y_tr"],
                bundle["oof_prob"],
                bundle["X_aux"],
                bundle["y_aux"],
                bundle["s_aux"],
                propensity_noise_std=sigma_pi,
                rng=rng,
            )
            i_col_noisy[idx] = ic
        sips_noisy = 0.25 * (mat_i[:, 0] + mat_i[:, 1] + mat_i[:, 2] + i_col_noisy)
        m_icol = evaluate_score_metrics(y_true, 1.0 - i_col_noisy)
        m_sips = evaluate_score_metrics(y_true, 1.0 - sips_noisy)
        # Also evaluate collider-only separation (Causal vs Collider)
        mask_cc = np.array([r.regime in ("Causal", "Collider") for r in records])
        auc_col_cc = float(roc_auc_score(y_true[mask_cc], 1.0 - i_col_noisy[mask_cc]))
        propensity_sensitivity.append(
            {
                "propensity_logit_noise_std": sigma_pi,
                "i_col_causal_vs_collider_auroc": auc_col_cc,
                "i_col_global_auroc": m_icol["auroc"],
                "sips_composite_auroc": m_sips["auroc"],
                "sips_fpr_at_90_tpr": m_sips["fpr_at_90_tpr"],
            }
        )

    return {
        "dirichlet_250": {
            "auroc_min": float(aucs_full.min()),
            "auroc_q05": float(np.quantile(aucs_full, 0.05)),
            "auroc_median": float(np.median(aucs_full)),
            "auroc_q95": float(np.quantile(aucs_full, 0.95)),
            "auroc_max": float(aucs_full.max()),
            "fraction_p_lt_1e3": float(np.mean(pvals_full < 1e-3)),
            "max_delong_p": float(pvals_full.max()),
        },
        "structured_profiles": profile_results,
        "threshold_grid": {
            "tau_sips_values": tau_grid.tolist(),
            "eta_veto_values": eta_grid.tolist(),
            "f1_matrix_2pillar": np.round(f1_grid_2p, 4).tolist(),
            "tpr_matrix_2pillar": np.round(tpr_grid_2p, 4).tolist(),
            "fpr_matrix_2pillar": np.round(fpr_grid_2p, 4).tolist(),
            "f1_matrix_4pillar": np.round(f1_grid_4p, 4).tolist(),
            "tpr_matrix_4pillar": np.round(tpr_grid_4p, 4).tolist(),
            "fpr_matrix_4pillar": np.round(fpr_grid_4p, 4).tolist(),
        },
        "propensity_misspecification": propensity_sensitivity,
    }


# ---------------------------------------------------------------------------
# Publication-Grade 4-Panel Visualization (300 DPI)
# ---------------------------------------------------------------------------
def build_figure(
    records: list[ScenarioRecord],
    metrics_map: dict[str, dict[str, float]],
    delong_sips_vs_cv: dict[str, float],
    delong_sips_vs_ood: dict[str, float],
    rule_def22: dict[str, Any],
    rule_tab23: dict[str, Any],
    sensitivity: dict[str, Any],
) -> None:
    """Generate clean, non-overlapping 4-panel figure at 300 DPI."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 12.6), dpi=300)
    fig.subplots_adjust(
        left=0.065, right=0.965, top=0.91, bottom=0.115, wspace=0.25, hspace=0.33
    )

    y_true = np.array([r.is_causal for r in records], dtype=int)
    scores_dict = {
        "SIPS Composite (4 pillars)": np.array([1.0 - r.sips_uniform for r in records]),
        "SIPS + 4-Step Veto (Tab. 2.3)": np.array([1.0 - r.sips_eff_4step for r in records]),
        "I_OOD only": np.array([1.0 - r.i_ood for r in records]),
        "I_inv only (Ablation)": np.array([1.0 - r.i_inv for r in records]),
        "I_col only (IPSW)": np.array([1.0 - r.i_col for r in records]),
        "I_mul only (Multiplicity)": np.array([1.0 - r.i_mul for r in records]),
        "i.i.d. 5-Fold CV": np.array([r.acc_cv_iid for r in records]),
    }

    # --- Panel (a): ROC Curves & Exact DeLong Test ---
    ax = axes[0, 0]
    styles = {
        "SIPS Composite (4 pillars)": ("#0B4F6C", "-", 2.8),
        "SIPS + 4-Step Veto (Tab. 2.3)": ("#14A76C", "--", 2.2),
        "I_OOD only": ("#E67E22", "-.", 1.7),
        "I_inv only (Ablation)": ("#2980B9", ":", 1.6),
        "I_col only (IPSW)": ("#16A085", "-.", 1.6),
        "I_mul only (Multiplicity)": ("#8E44AD", ":", 1.6),
        "i.i.d. 5-Fold CV": ("#C0392B", "-", 2.4),
    }
    key_map = {
        "SIPS Composite (4 pillars)": "sips_uniform",
        "SIPS + 4-Step Veto (Tab. 2.3)": "sips_veto_4step",
        "I_OOD only": "i_ood_only",
        "I_inv only (Ablation)": "i_inv_only",
        "I_col only (IPSW)": "i_col_only",
        "I_mul only (Multiplicity)": "i_mul_only",
        "i.i.d. 5-Fold CV": "cv_iid",
    }

    for label, sc in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, sc)
        col, ls, lw = styles[label]
        auc_v = metrics_map[key_map[label]]["auroc"]
        ax.plot(fpr, tpr, color=col, linestyle=ls, linewidth=lw, label=f"{label} (AUC={auc_v:.3f})")

    ax.scatter(
        [rule_tab23["causal_fpr"]],
        [rule_tab23["tpr"]],
        s=135,
        color="#14A76C",
        edgecolor="black",
        zorder=10,
        marker="*",
        label=f"Tab. 2.3 Veto (TPR={rule_tab23['tpr']:.2f}, FPR={rule_tab23['causal_fpr']:.3f})",
    )
    ax.scatter(
        [rule_def22["causal_fpr"]],
        [rule_def22["tpr"]],
        s=62,
        color="#F39C12",
        edgecolor="black",
        zorder=9,
        marker="o",
        label=f"Def. 2.2 Veto (TPR={rule_def22['tpr']:.2f}, FPR={rule_def22['causal_fpr']:.3f})",
    )

    ax.plot([0, 1], [0, 1], color="#7F8C8D", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("False Positive Rate (Causal FPR on Illusions)")
    ax.set_ylabel("True Positive Rate (Causal Sensitivity)")
    ax.set_title("(a) ROC Benchmark & Exact DeLong Test (K = 240 Scenarios)")
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=7.4)

    p_cv_str = (
        f"p = {delong_sips_vs_cv['p_value']:.2e}"
        if delong_sips_vs_cv["p_value"] >= 1e-300
        else "p < 1e-300"
    )
    box_txt = (
        f"Exact DeLong Tests:\n"
        f"SIPS vs i.i.d. CV: ΔAUC = +{delong_sips_vs_cv['diff']:.3f}\n"
        f"  z = {delong_sips_vs_cv['z_stat']:.2f}, {p_cv_str}\n"
        f"SIPS vs I_OOD only: ΔAUC = {delong_sips_vs_ood['diff']:+.3f}\n"
        f"  z = {delong_sips_vs_ood['z_stat']:+.2f}, p = {delong_sips_vs_ood['p_value']:.2e}\n"
        f"FPR@90%TPR: {metrics_map['sips_uniform']['fpr_at_90_tpr']:.3f} vs {metrics_map['cv_iid']['fpr_at_90_tpr']:.3f} (CV)"
    )
    ax.text(
        0.47,
        0.455,
        box_txt,
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#F8F9FA", edgecolor="#0B4F6C", lw=1.1, alpha=0.96),
    )

    # --- Panel (b): Orthogonal SIPS Pillar Signatures Across 4 Regimes ---
    ax = axes[0, 1]
    regimes_order = ["Causal", "Shortcut", "Collider", "Data_Dredging"]
    regime_labels = [
        "True Causal\n(n=60)",
        "Shortcut\n(n=60)",
        "Berkson Collider\n(n=60)",
        "Data Dredging\n(n=60)",
    ]
    pillar_names = [
        ("i_mul", r"$I_{\mathrm{mul}}$ (Multiplicity)", "#8E44AD"),
        ("i_ood", r"$I_{\mathrm{OOD}}$ (OOD Collapse)", "#E67E22"),
        ("i_inv", r"$I_{\mathrm{inv}}$ (Ablation)", "#2980B9"),
        ("i_col", r"$I_{\mathrm{col}}$ (IPSW)", "#16A085"),
        ("sips_uniform", r"$\mathrm{SIPS}$ Composite", "#0B4F6C"),
    ]
    x_pos = np.arange(len(regimes_order))
    width = 0.155
    for idx_p, (attr, plabel, col) in enumerate(pillar_names):
        means, stds = [], []
        for reg in regimes_order:
            vals = [getattr(r, attr) for r in records if r.regime == reg]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        offset = (idx_p - 2.0) * width
        ax.bar(
            x_pos + offset,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=2.2,
            color=col,
            alpha=0.88 if attr != "sips_uniform" else 0.98,
            edgecolor="black" if attr == "sips_uniform" else "none",
            linewidth=0.9 if attr == "sips_uniform" else 0.0,
            error_kw={"elinewidth": 0.9, "alpha": 0.75},
            label=plabel,
        )

    ax.axhline(0.50, color="#C0392B", linestyle="--", linewidth=1.3, label=r"Threshold $\tau_{\mathrm{SIPS}}=0.50$")
    ax.axhline(0.75, color="#7B241C", linestyle=":", linewidth=1.3, label=r"Veto $\eta_{\mathrm{veto}}=0.75$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_labels)
    ax.set_ylim(0.0, 1.28)
    ax.set_ylabel("Normalized Illusion Index (Mean ± 1 SD)")
    ax.set_title("(b) Orthogonal Pillar Signatures by Structural Regime")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.92, fontsize=7.4)

    # --- Panel (c): i.i.d. CV vs True OOD Accuracy Colored by SIPS ---
    ax = axes[1, 0]
    markers = {"Causal": "o", "Shortcut": "s", "Collider": "^", "Data_Dredging": "D"}
    nice_reg = {
        "Causal": "True Causal",
        "Shortcut": "Shortcut Learning",
        "Collider": "Berkson Collider",
        "Data_Dredging": "Data Dredging",
    }
    sc_plot = None
    for reg in regimes_order:
        sub = [r for r in records if r.regime == reg]
        xs = [r.acc_cv_iid for r in sub]
        ys = [r.acc_ood_true for r in sub]
        cs = [r.sips_uniform for r in sub]
        sc_plot = ax.scatter(
            xs,
            ys,
            c=cs,
            cmap="RdYlGn_r",
            vmin=0.10,
            vmax=0.65,
            marker=markers[reg],
            s=46,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.45,
            label=nice_reg[reg],
        )

    ax.plot([0.55, 0.95], [0.55, 0.95], color="#566573", linestyle="--", linewidth=1.1, label="Ideal calibration (OOD = CV)")
    ax.axhline(0.50, color="#7F8C8D", linestyle=":", linewidth=1.1, label="Chance level (0.50)")
    ax.set_xlim(0.60, 0.94)
    ax.set_ylim(0.14, 0.94)
    ax.set_xlabel("Standard i.i.d. 5-Fold CV Accuracy (Training Sample)")
    ax.set_ylabel("Ground-Truth OOD Deployment Accuracy")
    ax.set_title("(c) Why i.i.d. CV Fails vs. SIPS Separation (All K = 240)")
    ax.grid(True, linestyle=":", alpha=0.45)
    cbar = fig.colorbar(sc_plot, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("SIPS Score (0 = Causal, 1 = Illusion)", fontsize=8.8)
    ax.legend(loc="lower left", frameon=True, framealpha=0.93, fontsize=7.8)

    # --- Panel (d): Sensitivity Analysis (4-Pillar Veto Threshold Grid + Simplex & IPSW) ---
    ax = axes[1, 1]
    tgrid = sensitivity["threshold_grid"]
    taus = np.array(tgrid["tau_sips_values"])
    etas = np.array(tgrid["eta_veto_values"])
    f1_mat = np.array(tgrid["f1_matrix_4pillar"])

    im = ax.imshow(
        f1_mat,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.50,
        vmax=1.00,
        extent=[taus[0] - 0.025, taus[-1] + 0.025, etas[0] - 0.025, etas[-1] + 0.025],
    )
    for i_e, eta in enumerate(etas):
        for i_t, tau in enumerate(taus):
            val = f1_mat[i_e, i_t]
            txt_col = "black" if val >= 0.84 else "white"
            ax.text(tau, eta, f"{val:.2f}", ha="center", va="center", fontsize=6.8, color=txt_col)

    ax.scatter(
        [0.50],
        [0.75],
        s=200,
        marker="*",
        color="#FF3B30",
        edgecolor="white",
        linewidth=1.2,
        zorder=10,
    )

    cbar2 = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar2.set_label("Causal Detection F1-Score (4-Pillar Veto)", fontsize=8.8)
    ax.set_xlabel(r"Composite Threshold $\tau_{\mathrm{SIPS}}$  (red $\bigstar$ = a priori default $\tau=0.50, \eta=0.75$)")
    ax.set_ylabel(r"Uniform 4-Pillar Veto Threshold $\eta_{\mathrm{veto}}$")
    ax.set_title("(d) Threshold, Weight & Propensity Sensitivity (Jury Q3)")
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t:.2f}" for t in taus], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(etas)
    ax.set_yticklabels([f"{e:.2f}" for e in etas], fontsize=8)

    dir_s = sensitivity["dirichlet_250"]
    prop_s = sensitivity["propensity_misspecification"]
    p0 = prop_s[0]["sips_composite_auroc"]
    p2 = prop_s[-1]["sips_composite_auroc"]
    c0 = prop_s[0]["i_col_causal_vs_collider_auroc"]
    c2 = prop_s[-1]["i_col_causal_vs_collider_auroc"]
    sens_txt = (
        f"Weight Simplex (250 Dirichlet draws): AUROC median = {dir_s['auroc_median']:.3f} "
        f"[Q05–Q95: {dir_s['auroc_q05']:.3f}–{dir_s['auroc_q95']:.3f}], DeLong p < 1e-3 in {100.0 * dir_s['fraction_p_lt_1e3']:.1f}% of simplex   |   "
        f"Misspecified IPSW (σ_π: 0→2.0): I_col(Causal/Collider) AUC {c0:.2f}→{c2:.2f}, SIPS AUC {p0:.3f}→{p2:.3f}"
    )
    fig.text(
        0.515,
        0.022,
        sens_txt,
        ha="center",
        va="center",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#F4F6F7", edgecolor="#2C3E50", lw=1.0),
    )

    fig.suptitle(
        "Experiment 4 — Empirical Benchmark of the Post-Selection Illusion Score (SIPS) vs. i.i.d. Cross-Validation ($H_{2.3}$)",
        fontsize=13.4,
        fontweight="bold",
        y=0.962,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    null_cache: dict[tuple[int, int, int, float], float] = {}

    records: list[ScenarioRecord] = []
    aux_bundles: list[dict[str, Any]] = []
    sid = 0

    for _ in range(N_PER_REGIME):
        rec, bundle = run_regime_1_causal(sid, rng, null_cache)
        records.append(rec)
        aux_bundles.append(bundle)
        sid += 1

    for _ in range(N_PER_REGIME):
        rec, bundle = run_regime_2_shortcut(sid, rng, null_cache)
        records.append(rec)
        aux_bundles.append(bundle)
        sid += 1

    for _ in range(N_PER_REGIME):
        rec, bundle = run_regime_3_collider(sid, rng, null_cache)
        records.append(rec)
        aux_bundles.append(bundle)
        sid += 1

    for m_val in [100, 1000, 5000]:
        for _ in range(N_PER_REGIME // 3):
            rec, bundle = run_regime_4_data_dredging(sid, m_val, rng, null_cache)
            records.append(rec)
            aux_bundles.append(bundle)
            sid += 1

    assert len(records) == TOTAL_SCENARIOS

    y_true = np.array([r.is_causal for r in records], dtype=int)
    score_map = {
        "cv_iid": np.array([r.acc_cv_iid for r in records], dtype=np.float64),
        "sips_uniform": np.array([1.0 - r.sips_uniform for r in records], dtype=np.float64),
        "sips_veto_2pillar": np.array([1.0 - r.sips_eff_2pillar for r in records], dtype=np.float64),
        "sips_veto_4step": np.array([1.0 - r.sips_eff_4step for r in records], dtype=np.float64),
        "i_mul_only": np.array([1.0 - r.i_mul for r in records], dtype=np.float64),
        "i_ood_only": np.array([1.0 - r.i_ood for r in records], dtype=np.float64),
        "i_inv_only": np.array([1.0 - r.i_inv for r in records], dtype=np.float64),
        "i_col_only": np.array([1.0 - r.i_col for r in records], dtype=np.float64),
    }

    metrics_map = {
        name: evaluate_score_metrics(y_true, sc) for name, sc in score_map.items()
    }

    # Regime-specific diagnostic AUROC (Causal vs each individual illusion regime)
    regime_pairwise_auroc: dict[str, dict[str, float]] = {}
    for ill_reg in ["Shortcut", "Collider", "Data_Dredging"]:
        mask = np.array([r.regime in ("Causal", ill_reg) for r in records])
        regime_pairwise_auroc[f"Causal_vs_{ill_reg}"] = {
            k: float(roc_auc_score(y_true[mask], sc[mask])) for k, sc in score_map.items()
        }

    delong_comparisons = {
        "sips_uniform_vs_cv_iid": delong_paired_test(
            y_true, score_map["sips_uniform"], score_map["cv_iid"]
        ),
        "sips_veto_4step_vs_cv_iid": delong_paired_test(
            y_true, score_map["sips_veto_4step"], score_map["cv_iid"]
        ),
        "sips_uniform_vs_i_ood_only": delong_paired_test(
            y_true, score_map["sips_uniform"], score_map["i_ood_only"]
        ),
        "sips_uniform_vs_i_mul_only": delong_paired_test(
            y_true, score_map["sips_uniform"], score_map["i_mul_only"]
        ),
        "sips_uniform_vs_i_inv_only": delong_paired_test(
            y_true, score_map["sips_uniform"], score_map["i_inv_only"]
        ),
        "sips_uniform_vs_i_col_only": delong_paired_test(
            y_true, score_map["sips_uniform"], score_map["i_col_only"]
        ),
    }

    pred_def22 = np.array([r.pred_def22_causal for r in records], dtype=int)
    pred_tab23 = np.array([r.pred_table23_causal for r in records], dtype=int)
    rule_def22 = {
        "rule": "Definition 2.2 (SIPS <= 0.50 & max(I_mul, I_OOD) <= 0.75)",
        **summarize_binary_rule(y_true, pred_def22, records),
        "veto_rate_by_regime": {
            reg: float(np.mean([r.veto_2pillar for r in records if r.regime == reg]))
            for reg in ["Causal", "Shortcut", "Collider", "Data_Dredging"]
        },
    }
    rule_tab23 = {
        "rule": "Table 2.3 4-Step Protocol (SIPS <= 0.50 & I_mul<=0.75 & I_OOD<=0.75 & I_inv<=0.50 & I_col<=0.50)",
        **summarize_binary_rule(y_true, pred_tab23, records),
        "veto_rate_by_regime": {
            reg: float(np.mean([r.veto_4step for r in records if r.regime == reg]))
            for reg in ["Causal", "Shortcut", "Collider", "Data_Dredging"]
        },
    }

    regime_summary: dict[str, dict[str, float]] = {}
    for reg in ["Causal", "Shortcut", "Collider", "Data_Dredging"]:
        sub = [r for r in records if r.regime == reg]
        regime_summary[reg] = {
            "count": len(sub),
            "acc_is_mean": float(np.mean([r.acc_is for r in sub])),
            "acc_cv_iid_mean": float(np.mean([r.acc_cv_iid for r in sub])),
            "acc_cv_iid_std": float(np.std([r.acc_cv_iid for r in sub])),
            "acc_ood_audit_mean": float(np.mean([r.acc_ood_audit for r in sub])),
            "acc_ood_true_mean": float(np.mean([r.acc_ood_true for r in sub])),
            "acc_ood_true_std": float(np.std([r.acc_ood_true for r in sub])),
            "theta_raw_mean": float(np.mean([r.theta_raw for r in sub])),
            "theta_ipsw_mean": float(np.mean([r.theta_ipsw for r in sub])),
            "i_mul_mean": float(np.mean([r.i_mul for r in sub])),
            "i_mul_std": float(np.std([r.i_mul for r in sub])),
            "i_ood_mean": float(np.mean([r.i_ood for r in sub])),
            "i_ood_std": float(np.std([r.i_ood for r in sub])),
            "i_inv_mean": float(np.mean([r.i_inv for r in sub])),
            "i_inv_std": float(np.std([r.i_inv for r in sub])),
            "i_col_mean": float(np.mean([r.i_col for r in sub])),
            "i_col_std": float(np.std([r.i_col for r in sub])),
            "sips_mean": float(np.mean([r.sips_uniform for r in sub])),
            "sips_std": float(np.std([r.sips_uniform for r in sub])),
            "sips_eff_4step_mean": float(np.mean([r.sips_eff_4step for r in sub])),
        }

    dredging_by_m: dict[str, dict[str, float]] = {}
    for m_val in [100, 1000, 5000]:
        sub = [r for r in records if r.regime == "Data_Dredging" and r.n_candidates_M == m_val]
        dredging_by_m[str(m_val)] = {
            "count": len(sub),
            "acc_is_mean": float(np.mean([r.acc_is for r in sub])),
            "acc_cv_iid_mean": float(np.mean([r.acc_cv_iid for r in sub])),
            "acc_ood_true_mean": float(np.mean([r.acc_ood_true for r in sub])),
            "tau_null_95": float(np.mean([r.tau_null_95 for r in sub])),
            "i_mul_mean": float(np.mean([r.i_mul for r in sub])),
            "sips_mean": float(np.mean([r.sips_uniform for r in sub])),
            "fp_def22": int(sum(r.pred_def22_causal for r in sub)),
            "fp_tab23": int(sum(r.pred_table23_causal for r in sub)),
        }

    sensitivity = run_sensitivity_analyses(records, aux_bundles, rng)

    dl_main = delong_comparisons["sips_uniform_vs_cv_iid"]
    dl_ood = delong_comparisons["sips_uniform_vs_i_ood_only"]
    fpr_reduction_rel = 1.0 - (
        metrics_map["sips_uniform"]["fpr_at_90_tpr"]
        / max(metrics_map["cv_iid"]["fpr_at_90_tpr"], EPS)
    )
    h23_verdict = {
        "criterion_auroc_sips_gt_cv": bool(dl_main["diff"] > 0.0),
        "criterion_delong_p_lt_1e3": bool(dl_main["p_value"] < 1e-3),
        "criterion_auroc_sips_gt_i_ood": bool(dl_ood["diff"] > 0.0),
        "criterion_fpr_reduction_ge_50pct": bool(fpr_reduction_rel >= 0.50),
        "criterion_fpr_at_90_tpr_lt_10pct": bool(
            metrics_map["sips_uniform"]["fpr_at_90_tpr"] < 0.10
        ),
        "criterion_table23_rule_fpr_lt_10pct": bool(rule_tab23["causal_fpr"] < 0.10),
        "criterion_def22_rule_fpr_lt_10pct": bool(rule_def22["causal_fpr"] < 0.10),
        "fpr_reduction_relative_at_90_tpr": float(fpr_reduction_rel),
        "confirmed": bool(
            dl_main["diff"] > 0.0
            and dl_main["p_value"] < 1e-3
            and fpr_reduction_rel >= 0.50
            and rule_tab23["causal_fpr"] < 0.10
        ),
    }

    build_figure(
        records, metrics_map, dl_main, dl_ood, rule_def22, rule_tab23, sensitivity
    )

    elapsed = time.perf_counter() - t0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "experiment": "Experiment 4 — SIPS Empirical Benchmark (H_2.3)",
            "seed": SEED,
            "total_scenarios": TOTAL_SCENARIOS,
            "scenarios_per_regime": N_PER_REGIME,
            "runtime_seconds": round(elapsed, 3),
            "figure_path": str(FIGURE_PNG),
        },
        "metrics": metrics_map,
        "regime_pairwise_auroc": regime_pairwise_auroc,
        "delong_comparisons": delong_comparisons,
        "operating_rules": {
            "definition_2_2_two_pillar_veto": rule_def22,
            "table_2_3_four_step_veto": rule_tab23,
        },
        "regime_summary": regime_summary,
        "dredging_by_M": dredging_by_m,
        "sensitivity": sensitivity,
        "h23_verdict": h23_verdict,
        "scenarios": [asdict(r) for r in records],
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 80)
    print(f"EXPERIMENT 4 — SIPS EMPIRICAL BENCHMARK (K = {TOTAL_SCENARIOS}, seed = {SEED})")
    print(f"Runtime: {elapsed:.2f} s | JSON: {RESULTS_JSON.name} | Fig: {FIGURE_PNG.name}")
    print("=" * 80)
    print(f"{'Method':<24} {'AUROC':>8} {'95% CI':>17} {'AUPRC':>8} {'FPR@90%TPR':>12}")
    print("-" * 80)
    for key in [
        "sips_uniform",
        "sips_veto_4step",
        "sips_veto_2pillar",
        "i_ood_only",
        "i_inv_only",
        "i_col_only",
        "i_mul_only",
        "cv_iid",
    ]:
        m = metrics_map[key]
        ci_s = f"[{m['auroc_ci95_low']:.3f}, {m['auroc_ci95_high']:.3f}]"
        print(
            f"{key:<24} {m['auroc']:8.4f} {ci_s:>17} {m['auprc']:8.4f} {m['fpr_at_90_tpr']:12.4f}"
        )
    print("-" * 80)
    print("PAIRED DELONG TESTS:")
    for cname, dl in delong_comparisons.items():
        print(
            f"  {cname:<30}: ΔAUC = {dl['diff']:+.4f} (SE={dl['se_diff']:.4f}), "
            f"z = {dl['z_stat']:+.3f}, p = {dl['p_value']:.3e}"
        )
    print("-" * 80)
    print(
        f"DEF 2.2 (2-pillar veto): TPR={rule_def22['tpr']:.4f}, Causal FPR={rule_def22['causal_fpr']:.4f}, "
        f"Prec={rule_def22['precision']:.4f}, F1={rule_def22['f1']:.4f}, FP={rule_def22['false_positives_by_regime']}"
    )
    print(
        f"TAB 2.3 (4-step veto)  : TPR={rule_tab23['tpr']:.4f}, Causal FPR={rule_tab23['causal_fpr']:.4f}, "
        f"Prec={rule_tab23['precision']:.4f}, F1={rule_tab23['f1']:.4f}, FP={rule_tab23['false_positives_by_regime']}"
    )
    print("-" * 80)
    print("REGIME SUMMARY (Means):")
    print(
        f"{'Regime':<15} {'CV_iid':>7} {'OOD_true':>9} {'I_mul':>7} {'I_OOD':>7} {'I_inv':>7} {'I_col':>7} {'SIPS':>7}"
    )
    for reg, rs in regime_summary.items():
        print(
            f"{reg:<15} {rs['acc_cv_iid_mean']:7.3f} {rs['acc_ood_true_mean']:9.3f} "
            f"{rs['i_mul_mean']:7.3f} {rs['i_ood_mean']:7.3f} {rs['i_inv_mean']:7.3f} "
            f"{rs['i_col_mean']:7.3f} {rs['sips_mean']:7.3f}"
        )
    print("-" * 80)
    print("PAIRWISE REGIME AUROC (Causal vs each Illusion):")
    for pair_k, vals in regime_pairwise_auroc.items():
        print(
            f"  {pair_k:<24}: SIPS={vals['sips_uniform']:.4f}, I_OOD={vals['i_ood_only']:.4f}, "
            f"I_mul={vals['i_mul_only']:.4f}, I_inv={vals['i_inv_only']:.4f}, "
            f"I_col={vals['i_col_only']:.4f}, CV={vals['cv_iid']:.4f}"
        )
    print("-" * 80)
    print("PROPENSITY MISSPECIFICATION (I_col noise):")
    for row in sensitivity["propensity_misspecification"]:
        print(
            f"  sigma_pi={row['propensity_logit_noise_std']:.2f}: "
            f"I_col(C/Col)={row['i_col_causal_vs_collider_auroc']:.4f}, "
            f"I_col_glob={row['i_col_global_auroc']:.4f}, SIPS_AUC={row['sips_composite_auroc']:.4f}, "
            f"FPR@90%={row['sips_fpr_at_90_tpr']:.4f}"
        )
    ds = sensitivity["dirichlet_250"]
    print(
        f"SENSITIVITY (250 Dirichlet weights): AUROC median={ds['auroc_median']:.4f}, "
        f"Q05-Q95=[{ds['auroc_q05']:.4f}, {ds['auroc_q95']:.4f}], "
        f"range=[{ds['auroc_min']:.4f}, {ds['auroc_max']:.4f}], "
        f"p<1e-3 in {100.0 * ds['fraction_p_lt_1e3']:.1f}% of simplex"
    )
    print(f"H_2.3 VERDICT: {h23_verdict}")
    print("=" * 80)


if __name__ == "__main__":
    main()
