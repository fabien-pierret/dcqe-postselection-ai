#!/usr/bin/env python3
r"""
Experiment 6: Barren Plateaus Scaling (2^-n vs 4^-n) and Geometric Difference g_CQ.

Validates Hypothesis H4 and Jury Correction C1 from the doctoral thesis dossier:
  - Part A: Exact vectorized statevector simulation of an alternating brickwork Hardware-Efficient
    Ansatz (HEA: Ry, Rz + alternating even/odd CNOT layers with periodic wrap) for
    n in {2, 4, 6, 8, 10, 12} qubits. Computes exact cost-function gradients via the
    parameter-shift rule (verified against finite differences) for:
      (1) Local cost O_loc = Z_1 at shallow depth L = 1 (no barren plateau, exact Var = 1/4 = Theta(1))
          vs deep depth L = 2n (exact 2-design scaling Var ~ 2^-n => N_shots ~ 2^n);
      (2) Global projector cost O_glob = |0><0|^{\otimes n} at shallow depth L = 1 (exact cost-
          induced barren plateau Var = (1/8)(3/8)^(n-1), Cerezo et al. 2021) and deep depth
          L = 2n (exact 2-design scaling Var ~ 4^-n = 2^(-2n) => N_shots ~ 4^n = 2^(2n)).
  - Part B: Exact Geometric Difference g_CQ = sqrt(||K_Q^{1/2} (K_C + lambda I)^{-1} K_Q^{1/2}||_inf)
    (Huang et al., Nature Communications 12, 2631, 2021) comparing shallow trainable vs deep
    scrambling quantum kernels against classical RBF and Polynomial kernels (scikit-learn) on:
      (i) Standard classical datasets (make_moons & Gaussian blobs), showing classical parity/
          superiority and kernel concentration collapse at high depth;
      (ii) Maximally separated quantum-structured target built from the dominant eigenvector of
          K_Q^{1/2} (K_C + lambda I)^{-1} K_Q^{1/2}, showing genuine QSVM test generalization
          advantage when g_CQ is large and the quantum kernel is not exponentially concentrated.

Outputs:
  - sim/results/exp6_barren_plateaus.json
  - sim/figures/fig6_barren_plateaus_gcq.png (300 DPI, 4 panels)

Deterministic execution (fixed seed=42) on macOS ARM64 CPU (NumPy/SciPy/scikit-learn/Matplotlib).
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
from scipy.linalg import eigh
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"


def apply_batched_ry(states: np.ndarray, angles: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply Ry(angles[b]) on `qubit` across a batch of B n-qubit statevectors."""
    b_size = states.shape[0]
    psi = states.reshape((b_size, 1 << qubit, 2, 1 << (n - 1 - qubit)))
    c = np.cos(0.5 * angles)[:, None, None]
    s = np.sin(0.5 * angles)[:, None, None]
    p0 = psi[:, :, 0, :].copy()
    p1 = psi[:, :, 1, :].copy()
    psi[:, :, 0, :] = c * p0 - s * p1
    psi[:, :, 1, :] = s * p0 + c * p1
    return psi.reshape(b_size, 1 << n)


def apply_batched_rz(states: np.ndarray, angles: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply Rz(angles[b]) on `qubit` across a batch of B n-qubit statevectors."""
    b_size = states.shape[0]
    psi = states.reshape((b_size, 1 << qubit, 2, 1 << (n - 1 - qubit)))
    em = np.exp(-0.5j * angles)[:, None, None]
    ep = np.exp(+0.5j * angles)[:, None, None]
    psi[:, :, 0, :] *= em
    psi[:, :, 1, :] *= ep
    return psi.reshape(b_size, 1 << n)


def apply_batched_cnot(states: np.ndarray, ctrl: int, targ: int, n: int) -> np.ndarray:
    """Apply CNOT(ctrl -> targ) across a batch of B n-qubit statevectors."""
    b_size = states.shape[0]
    # Reshape to (B, 2, 2, ..., 2) with n qubit axes
    psi = states.reshape((b_size,) + (2,) * n)
    idx0 = [slice(None)] * (n + 1)
    idx1 = [slice(None)] * (n + 1)
    idx0[ctrl + 1] = 1
    idx1[ctrl + 1] = 1
    idx0[targ + 1] = 0
    idx1[targ + 1] = 1
    tmp = psi[tuple(idx0)].copy()
    psi[tuple(idx0)] = psi[tuple(idx1)]
    psi[tuple(idx1)] = tmp
    return psi.reshape(b_size, 1 << n)


def eval_hea_costs(
    params_y: np.ndarray, params_z: np.ndarray, n: int, depth: int
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Evaluate both local cost C_loc = <Z_1> (qubit index 1) and global projector cost
    C_glob = |<0^n|psi>|^2 for a batch of B circuits of an alternating brickwork
    Hardware-Efficient Ansatz (Cerezo et al., Nature Communications 2021):
      Initial state |0>^{\otimes n}, followed by `depth` layers of:
        Ry(theta_{l,j}) then Rz(phi_{l,j}) on every qubit j,
        then alternating brickwork CNOT pairs:
          - even layers (l % 2 == 0): CNOT(0->1), CNOT(2->3), ..., CNOT(n-2->n-1)
          - odd layers  (l % 2 == 1): CNOT(1->2), CNOT(3->4), ..., CNOT(n-1->0)
            (for n == 2, odd layers apply CNOT(1->0)).
    At L = 1, differentiating qubit 0 w.r.t. O_loc = Z_1 probes an entangled 2-qubit
    light cone with exact constant variance Var = 1/4, while O_glob = |0^n><0^n|
    exhibits an exact cost-induced barren plateau Var = (1/8)(3/8)^{n-1}.
    """
    b_size = params_y.shape[0]
    states = np.zeros((b_size, 1 << n), dtype=np.complex128)
    states[:, 0] = 1.0

    for layer in range(depth):
        for q in range(n):
            states = apply_batched_ry(states, params_y[:, layer, q], q, n)
            states = apply_batched_rz(states, params_z[:, layer, q], q, n)
        if layer % 2 == 0:
            for q in range(0, n - 1, 2):
                states = apply_batched_cnot(states, q, q + 1, n)
        else:
            if n == 2:
                states = apply_batched_cnot(states, 1, 0, n)
            else:
                for q in range(1, n - 1, 2):
                    states = apply_batched_cnot(states, q, q + 1, n)
                states = apply_batched_cnot(states, n - 1, 0, n)

    # Local observable O_loc = Z_1 (qubit index 1 in MSB ordering)
    psi_q1 = states.reshape(b_size, 2, 2, 1 << (n - 2))
    p_zero = np.sum(np.abs(psi_q1[:, :, 0, :]) ** 2, axis=(1, 2))
    p_one = np.sum(np.abs(psi_q1[:, :, 1, :]) ** 2, axis=(1, 2))
    c_loc = p_zero - p_one

    # Global projector O_glob = |0^n><0^n|
    c_glob = np.abs(states[:, 0]) ** 2
    return c_loc, c_glob


def parameter_shift_gradients(
    params_y: np.ndarray, params_z: np.ndarray, n: int, depth: int, target_layer: int, target_qubit: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact gradient via parameter-shift rule on Ry(params_y[:, target_layer, target_qubit])."""
    py_plus = params_y.copy()
    py_minus = params_y.copy()
    py_plus[:, target_layer, target_qubit] += 0.5 * math.pi
    py_minus[:, target_layer, target_qubit] -= 0.5 * math.pi
    c_loc_p, c_glob_p = eval_hea_costs(py_plus, params_z, n, depth)
    c_loc_m, c_glob_m = eval_hea_costs(py_minus, params_z, n, depth)
    return 0.5 * (c_loc_p - c_loc_m), 0.5 * (c_glob_p - c_glob_m)


def verify_parameter_shift_vs_fd(rng: np.random.Generator) -> float:
    """Check parameter-shift rule against central finite differences on n=4, depth=4."""
    n, depth, b_size = 4, 4, 16
    py = rng.uniform(0.0, 2.0 * math.pi, size=(b_size, depth, n))
    pz = rng.uniform(0.0, 2.0 * math.pi, size=(b_size, depth, n))
    g_loc_ps, g_glob_ps = parameter_shift_gradients(py, pz, n, depth, target_layer=2, target_qubit=0)
    eps = 1e-6
    py_p = py.copy()
    py_m = py.copy()
    py_p[:, 2, 0] += eps
    py_m[:, 2, 0] -= eps
    c_loc_p, c_glob_p = eval_hea_costs(py_p, pz, n, depth)
    c_loc_m, c_glob_m = eval_hea_costs(py_m, pz, n, depth)
    g_loc_fd = (c_loc_p - c_loc_m) / (2.0 * eps)
    g_glob_fd = (c_glob_p - c_glob_m) / (2.0 * eps)
    err = float(max(np.max(np.abs(g_loc_ps - g_loc_fd)), np.max(np.abs(g_glob_ps - g_glob_fd))))
    assert err < 1e-8, f"Parameter-shift mismatch: {err}"
    return err


def run_part_a_barren_plateaus(rng: np.random.Generator) -> Dict:
    """Execute Part A: Exact Barren Plateaus across n = 2, 4, 6, 8, 10, 12."""
    ps_err = verify_parameter_shift_vs_fd(rng)

    n_list = [2, 4, 6, 8, 10, 12]
    # Batch sizes: large sample count for low variance estimation error while staying fast (< 15s)
    batch_sizes = {2: 1200, 4: 1000, 6: 800, 8: 600, 10: 400, 12: 240}

    var_loc_shallow: List[float] = []
    var_glob_shallow: List[float] = []
    var_loc_deep: List[float] = []
    var_glob_deep: List[float] = []
    haar_loc_exact: List[float] = []
    haar_glob_exact: List[float] = []

    for n in n_list:
        b_size = batch_sizes[n]
        d = float(1 << n)
        # Exact Haar 2-design variance formulas for Pauli generator (Tr(G)=0, G^2=I):
        # Var_Haar = d / (2 (d+1)(d^2 - 1)) * (Tr(O^2) - Tr(O)^2 / d)
        haar_loc_exact.append(d * d / (2.0 * (d + 1.0) * (d * d - 1.0)))
        haar_glob_exact.append(1.0 / (2.0 * (d + 1.0) ** 2))

        # 1) Shallow depth L = 1 (target layer 0, qubit 0)
        py_s = rng.uniform(0.0, 2.0 * math.pi, size=(b_size, 1, n))
        pz_s = rng.uniform(0.0, 2.0 * math.pi, size=(b_size, 1, n))
        gl_s, gg_s = parameter_shift_gradients(py_s, pz_s, n, depth=1, target_layer=0, target_qubit=0)
        var_loc_shallow.append(float(np.var(gl_s, ddof=1)))
        var_glob_shallow.append(float(np.var(gg_s, ddof=1)))

        # 2) Deep depth L = 3*n (target middle layer -> both U_A and U_B are deep alternating 2-designs)
        depth_deep = max(8, 3 * n)
        py_d = rng.uniform(0.0, 2.0 * math.pi, size=(b_size, depth_deep, n))
        pz_d = rng.uniform(0.0, 2.0 * math.pi, size=(b_size, depth_deep, n))
        gl_d, gg_d = parameter_shift_gradients(py_d, pz_d, n, depth=depth_deep, target_layer=depth_deep // 2, target_qubit=0)
        var_loc_deep.append(float(np.var(gl_d, ddof=1)))
        var_glob_deep.append(float(np.var(gg_d, ddof=1)))

    # Log2-linear regressions vs n (fit on n >= 4 where asymptotic 2-design regime holds cleanly,
    # and also report full fit on all n in {2..12})
    n_arr = np.array(n_list, dtype=float)
    mask_asymp = n_arr >= 4

    def fit_log2(vals: List[float], mask: np.ndarray) -> Tuple[float, float]:
        slope, intercept = np.polyfit(n_arr[mask], np.log2(np.array(vals)[mask]), 1)
        return float(slope), float(intercept)

    slope_loc_shallow, _ = fit_log2(var_loc_shallow, np.ones_like(mask_asymp, dtype=bool))
    slope_glob_shallow, _ = fit_log2(var_glob_shallow, mask_asymp)
    slope_loc_deep_all, _ = fit_log2(var_loc_deep, np.ones_like(mask_asymp, dtype=bool))
    slope_loc_deep_asymp, _ = fit_log2(var_loc_deep, mask_asymp)
    slope_glob_deep_all, _ = fit_log2(var_glob_deep, np.ones_like(mask_asymp, dtype=bool))
    slope_glob_deep_asymp, _ = fit_log2(var_glob_deep, mask_asymp)

    # Required shots for SNR = 1: N_shots = 1 / Var
    shots_loc_deep = [float(1.0 / v) for v in var_loc_deep]
    shots_glob_deep = [float(1.0 / v) for v in var_glob_deep]
    shots_loc_shallow = [float(1.0 / v) for v in var_loc_shallow]
    shots_glob_shallow = [float(1.0 / v) for v in var_glob_shallow]

    return {
        "parameter_shift_vs_fd_max_error": ps_err,
        "n_list": n_list,
        "depth_deep_list": [max(8, 3 * n) for n in n_list],
        "batch_sizes": batch_sizes,
        "var_local_shallow_L1": var_loc_shallow,
        "var_global_shallow_L1": var_glob_shallow,
        "var_local_deep_L3n": var_loc_deep,
        "var_global_deep_L3n": var_glob_deep,
        "haar_2design_exact_local": haar_loc_exact,
        "haar_2design_exact_global": haar_glob_exact,
        "shots_local_shallow_L1": shots_loc_shallow,
        "shots_global_shallow_L1": shots_glob_shallow,
        "shots_local_deep_L3n": shots_loc_deep,
        "shots_global_deep_L3n": shots_glob_deep,
        "log2_slopes_vs_n": {
            "local_shallow_L1": slope_loc_shallow,
            "global_shallow_L1_n_ge_4": slope_glob_shallow,
            "local_deep_L3n_all": slope_loc_deep_all,
            "local_deep_L3n_n_ge_4": slope_loc_deep_asymp,
            "global_deep_L3n_all": slope_glob_deep_all,
            "global_deep_L3n_n_ge_4": slope_glob_deep_asymp,
        },
    }


def quantum_feature_states(
    X: np.ndarray, n_qubits: int, depth: int, bandwidth: float, rng_weights: np.ndarray
) -> np.ndarray:
    """
    Compute quantum feature-map states |phi(x)> for all samples x in X (shape N x d_in).
    Uses a data-reuploading entangled feature map of `depth` layers:
      - Data encoding via Ry(bandwidth * (W_l @ x)) and Rz(bandwidth * (V_l @ x))
      - Entangling ring of CNOTs + fixed ZZ-like phase interactions
    Low depth (depth=1..2, moderate bandwidth) -> trainable, rich kernel spectrum.
    High depth (depth=10, large bandwidth) -> scrambling, exponential kernel concentration!
    """
    n_samples = X.shape[0]
    states = np.zeros((n_samples, 1 << n_qubits), dtype=np.complex128)
    states[:, 0] = 1.0

    for layer in range(depth):
        ang_y = bandwidth * (X @ rng_weights[layer, 0])  # (N, n_qubits)
        ang_z = bandwidth * (X @ rng_weights[layer, 1])  # (N, n_qubits)
        for q in range(n_qubits):
            states = apply_batched_ry(states, ang_y[:, q], q, n_qubits)
            states = apply_batched_rz(states, ang_z[:, q], q, n_qubits)
        # Entangling CNOT ring
        for q in range(n_qubits - 1):
            states = apply_batched_cnot(states, q, q + 1, n_qubits)
        if n_qubits > 2:
            states = apply_batched_cnot(states, n_qubits - 1, 0, n_qubits)
        # Second-order ZZ data-product phase rotation for nonlinear feature expressivity
        for q in range(n_qubits - 1):
            prod_ang = (bandwidth * 0.65) * (ang_y[:, q] * ang_y[:, q + 1])
            states = apply_batched_rz(states, prod_ang, q + 1, n_qubits)

    return states


def fidelity_kernel_matrix(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """Exact fidelity quantum kernel K_Q(x, x') = |<phi(x)|phi(x')>|^2."""
    overlaps = states_a.conj() @ states_b.T
    return np.abs(overlaps) ** 2


def normalize_kernel(K: np.ndarray) -> np.ndarray:
    """Normalize PSD kernel so diagonal entries equal 1 (Tr(K) = N)."""
    diag = np.sqrt(np.maximum(np.diag(K), 1e-12))
    return K / np.outer(diag, diag)


def compute_geometric_difference(
    K_Q: np.ndarray, K_C: np.ndarray, lam: float = 1e-2
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute exact regularized geometric difference (Huang et al., Nat. Comm. 12, 2631, 2021):
      g_CQ = sqrt( || K_Q^{1/2} (K_C + lam I)^{-1} K_Q^{1/2} ||_inf )
    Returns:
      g_cq: scalar geometric difference
      v_max: dominant unit eigenvector of M_CQ = K_Q^{1/2} (K_C + lam I)^{-1} K_Q^{1/2}
      y_max_cont: maximally separating quantum-structured target K_Q^{1/2} v_max
    """
    # Symmetrize for numerical precision
    Kq = 0.5 * (K_Q + K_Q.T)
    Kc = 0.5 * (K_C + K_C.T)
    n = Kq.shape[0]

    evals_q, evecs_q = eigh(Kq)
    evals_q_pos = np.maximum(evals_q, 0.0)
    Kq_sqrt = (evecs_q * np.sqrt(evals_q_pos)) @ evecs_q.T

    evals_c, evecs_c = eigh(Kc)
    evals_c_pos = np.maximum(evals_c, 0.0)
    Kc_inv_reg = (evecs_c * (1.0 / (evals_c_pos + lam))) @ evecs_c.T

    M_cq = Kq_sqrt @ Kc_inv_reg @ Kq_sqrt
    M_cq = 0.5 * (M_cq + M_cq.T)
    evals_m, evecs_m = eigh(M_cq)
    lam_max = float(max(evals_m[-1], 0.0))
    v_max = evecs_m[:, -1]
    g_cq = math.sqrt(lam_max)
    y_max_cont = Kq_sqrt @ v_max
    return g_cq, v_max, y_max_cont


def evaluate_precomputed_svm(
    K_full: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> Dict[str, float]:
    """Train SVC with precomputed kernel (tuned over C grid on train only) and evaluate on test."""
    K_tr = K_full[np.ix_(train_idx, train_idx)]
    K_te = K_full[np.ix_(test_idx, train_idx)]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # Simple internal 3-fold CV on train_idx to choose C cleanly without touching test_idx
    best_c = 1.0
    best_cv = -1.0
    n_tr = len(train_idx)
    folds = np.array_split(np.arange(n_tr), 3)
    for c_val in (0.1, 1.0, 10.0, 100.0):
        fold_accs = []
        for fi in range(3):
            va_loc = folds[fi]
            tr_loc = np.concatenate([folds[fj] for fj in range(3) if fj != fi])
            clf = SVC(kernel="precomputed", C=c_val)
            clf.fit(K_tr[np.ix_(tr_loc, tr_loc)], y_tr[tr_loc])
            pred = clf.predict(K_tr[np.ix_(va_loc, tr_loc)])
            fold_accs.append(float(np.mean(pred == y_tr[va_loc])))
        cv_mean = float(np.mean(fold_accs))
        if cv_mean > best_cv:
            best_cv = cv_mean
            best_c = c_val

    clf_final = SVC(kernel="precomputed", C=best_c)
    clf_final.fit(K_tr, y_tr)
    train_acc = float(np.mean(clf_final.predict(K_tr) == y_tr))
    test_acc = float(np.mean(clf_final.predict(K_te) == y_te))
    return {"train_acc": train_acc, "test_acc": test_acc, "best_C": best_c}


def effective_rank(K: np.ndarray) -> float:
    """Compute effective spectral rank exp(H(lambda / sum(lambda))) of a PSD Gram matrix."""
    evals = np.maximum(eigh(0.5 * (K + K.T), eigvals_only=True), 0.0)
    p = evals / np.sum(evals)
    p = p[p > 1e-12]
    return float(math.exp(-float(np.sum(p * np.log(p)))))


def run_part_b_geometric_difference(rng: np.random.Generator) -> Dict:
    """
    Execute Part B: Geometric Difference g_CQ and Classical SVM vs QSVM across:
      1. Standard classical datasets (make_moons, Gaussian blobs);
      2. Maximally separated quantum-structured target (dominant eigenvector of M_CQ);
      3. Feature-map depth sweep (d = 1..10) showing kernel concentration collapse!
    """
    n_samples = 160
    n_qubits = 6
    lam_reg = 0.05  # Regularization for well-conditioned g_CQ

    # Fixed random projection weights for quantum feature maps (up to depth 10)
    w_proj = rng.normal(0.0, 1.0, size=(10, 2, 2, n_qubits))

    # --- Dataset 1: Standard classical make_moons ---
    X_moons, y_moons = make_moons(n_samples=n_samples, noise=0.18, random_state=SEED)
    X_moons = StandardScaler().fit_transform(X_moons)

    # --- Dataset 2: Standard classical Gaussian blobs (nonlinearly overlapping slightly) ---
    X_blobs, y_blobs = make_blobs(
        n_samples=n_samples, centers=[(-1.1, -0.8), (1.1, 0.8)], cluster_std=1.05, random_state=SEED
    )
    X_blobs = StandardScaler().fit_transform(X_blobs)

    idx_all = np.arange(n_samples)
    train_idx, test_idx = train_test_split(idx_all, test_size=60, random_state=SEED, stratify=y_moons)
    train_idx_b, test_idx_b = train_test_split(idx_all, test_size=60, random_state=SEED, stratify=y_blobs)

    def build_classical_kernels(X: np.ndarray) -> Dict[str, np.ndarray]:
        k_rbf = normalize_kernel(rbf_kernel(X, gamma=0.5))
        k_poly = normalize_kernel(polynomial_kernel(X, degree=3, gamma=0.5, coef0=1.0))
        return {"rbf": k_rbf, "poly3": k_poly}

    # Build quantum kernels on make_moons:
    # (a) Shallow trainable feature map (depth=1, bandwidth=0.45)
    # (b) Structured entangled feature map (depth=2, bandwidth=1.15)
    # (c) Deep scrambling feature map (depth=10, bandwidth=2.50 -> exponential concentration!)
    st_moons_shallow = quantum_feature_states(X_moons, n_qubits, depth=1, bandwidth=0.45, rng_weights=w_proj)
    st_moons_struct = quantum_feature_states(X_moons, n_qubits, depth=2, bandwidth=1.15, rng_weights=w_proj)
    st_moons_deep = quantum_feature_states(X_moons, n_qubits, depth=10, bandwidth=2.50, rng_weights=w_proj)

    Kq_moons_shallow = fidelity_kernel_matrix(st_moons_shallow, st_moons_shallow)
    Kq_moons_struct = fidelity_kernel_matrix(st_moons_struct, st_moons_struct)
    Kq_moons_deep = fidelity_kernel_matrix(st_moons_deep, st_moons_deep)

    Kc_moons = build_classical_kernels(X_moons)
    g_shallow_rbf, _, _ = compute_geometric_difference(Kq_moons_shallow, Kc_moons["rbf"], lam=lam_reg)
    g_shallow_poly, _, _ = compute_geometric_difference(Kq_moons_shallow, Kc_moons["poly3"], lam=lam_reg)
    g_struct_rbf, _, y_struct_cont = compute_geometric_difference(Kq_moons_struct, Kc_moons["rbf"], lam=lam_reg)
    g_struct_poly, _, _ = compute_geometric_difference(Kq_moons_struct, Kc_moons["poly3"], lam=lam_reg)
    g_deep_rbf, _, _ = compute_geometric_difference(Kq_moons_deep, Kc_moons["rbf"], lam=lam_reg)
    g_deep_poly, _, _ = compute_geometric_difference(Kq_moons_deep, Kc_moons["poly3"], lam=lam_reg)

    # Also evaluate Gaussian blobs
    st_blobs_shallow = quantum_feature_states(X_blobs, n_qubits, depth=1, bandwidth=0.45, rng_weights=w_proj)
    st_blobs_deep = quantum_feature_states(X_blobs, n_qubits, depth=10, bandwidth=2.50, rng_weights=w_proj)
    Kq_blobs_shallow = fidelity_kernel_matrix(st_blobs_shallow, st_blobs_shallow)
    Kq_blobs_deep = fidelity_kernel_matrix(st_blobs_deep, st_blobs_deep)
    Kc_blobs = build_classical_kernels(X_blobs)
    g_blobs_shallow_rbf, _, _ = compute_geometric_difference(Kq_blobs_shallow, Kc_blobs["rbf"], lam=lam_reg)
    g_blobs_deep_rbf, _, _ = compute_geometric_difference(Kq_blobs_deep, Kc_blobs["rbf"], lam=lam_reg)

    # Construct the maximally separating quantum-structured target y_q_hard on X_moons
    # from the dominant eigenvector of K_Q_struct^{1/2} (K_RBF + lam I)^{-1} K_Q_struct^{1/2}
    # (Huang et al., Nature Communications 12, 2631, 2021)
    y_q_hard = (y_struct_cont >= float(np.median(y_struct_cont))).astype(int)
    train_idx_q, test_idx_q = train_test_split(idx_all, test_size=60, random_state=SEED, stratify=y_q_hard)

    # Compute exact model complexities s_Q and s_C on the continuous quantum target
    y_normed = y_struct_cont / np.linalg.norm(y_struct_cont)
    evals_q, evecs_q = eigh(0.5 * (Kq_moons_struct + Kq_moons_struct.T))
    Kq_inv = (evecs_q * (1.0 / (np.maximum(evals_q, 0.0) + lam_reg))) @ evecs_q.T
    evals_c, evecs_c = eigh(0.5 * (Kc_moons["rbf"] + Kc_moons["rbf"].T))
    Kc_inv = (evecs_c * (1.0 / (np.maximum(evals_c, 0.0) + lam_reg))) @ evecs_c.T
    s_Q_hard = float(y_normed @ Kq_inv @ y_normed)
    s_C_hard = float(y_normed @ Kc_inv @ y_normed)

    # Evaluate SVMs on all 3 tasks (make_moons, blobs, quantum-structured target)
    tasks_eval = {
        "classical_make_moons": {
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "sqrt_N_train": math.sqrt(len(train_idx)),
            "g_CQ_shallow_vs_rbf": g_shallow_rbf,
            "g_CQ_shallow_vs_poly": g_shallow_poly,
            "g_CQ_struct_vs_rbf": g_struct_rbf,
            "g_CQ_deep_vs_rbf": g_deep_rbf,
            "svm_rbf": evaluate_precomputed_svm(Kc_moons["rbf"], y_moons, train_idx, test_idx),
            "svm_poly3": evaluate_precomputed_svm(Kc_moons["poly3"], y_moons, train_idx, test_idx),
            "qsvm_shallow": evaluate_precomputed_svm(Kq_moons_shallow, y_moons, train_idx, test_idx),
            "qsvm_struct": evaluate_precomputed_svm(Kq_moons_struct, y_moons, train_idx, test_idx),
            "qsvm_deep_scrambling": evaluate_precomputed_svm(Kq_moons_deep, y_moons, train_idx, test_idx),
        },
        "classical_gaussian_blobs": {
            "n_train": len(train_idx_b),
            "n_test": len(test_idx_b),
            "g_CQ_shallow_vs_rbf": g_blobs_shallow_rbf,
            "g_CQ_deep_vs_rbf": g_blobs_deep_rbf,
            "svm_rbf": evaluate_precomputed_svm(Kc_blobs["rbf"], y_blobs, train_idx_b, test_idx_b),
            "svm_poly3": evaluate_precomputed_svm(Kc_blobs["poly3"], y_blobs, train_idx_b, test_idx_b),
            "qsvm_shallow": evaluate_precomputed_svm(Kq_blobs_shallow, y_blobs, train_idx_b, test_idx_b),
            "qsvm_deep_scrambling": evaluate_precomputed_svm(Kq_blobs_deep, y_blobs, train_idx_b, test_idx_b),
        },
        "quantum_structured_max_gCQ": {
            "n_train": len(train_idx_q),
            "n_test": len(test_idx_q),
            "sqrt_N_train": math.sqrt(len(train_idx_q)),
            "g_CQ_struct_vs_rbf": g_struct_rbf,
            "g_CQ_struct_vs_poly": g_struct_poly,
            "complexity_s_Q": s_Q_hard,
            "complexity_s_C_rbf": s_C_hard,
            "complexity_ratio_sC_over_sQ": float(s_C_hard / max(s_Q_hard, 1e-12)),
            "svm_rbf": evaluate_precomputed_svm(Kc_moons["rbf"], y_q_hard, train_idx_q, test_idx_q),
            "svm_poly3": evaluate_precomputed_svm(Kc_moons["poly3"], y_q_hard, train_idx_q, test_idx_q),
            "qsvm_shallow": evaluate_precomputed_svm(Kq_moons_shallow, y_q_hard, train_idx_q, test_idx_q),
            "qsvm_struct": evaluate_precomputed_svm(Kq_moons_struct, y_q_hard, train_idx_q, test_idx_q),
            "qsvm_deep_scrambling": evaluate_precomputed_svm(Kq_moons_deep, y_q_hard, train_idx_q, test_idx_q),
        },
    }

    # --- Depth / Bandwidth sweep showing kernel concentration & Goldilocks zone ---
    sweep_configs = [
        (1, 0.35),
        (1, 0.70),
        (2, 1.00),
        (2, 1.15),
        (3, 1.35),
        (4, 1.60),
        (6, 1.95),
        (8, 2.25),
        (10, 2.50),
    ]
    depth_sweep = []
    for dp, bw in sweep_configs:
        st = quantum_feature_states(X_moons, n_qubits, depth=dp, bandwidth=bw, rng_weights=w_proj)
        Kq = fidelity_kernel_matrix(st, st)
        off_diag = Kq[~np.eye(n_samples, dtype=bool)]
        g_val, _, y_c = compute_geometric_difference(Kq, Kc_moons["rbf"], lam=lam_reg)
        y_bin = (y_c >= float(np.median(y_c))).astype(int)
        tr_i, te_i = train_test_split(idx_all, test_size=60, random_state=SEED, stratify=y_bin)
        q_res_hard = evaluate_precomputed_svm(Kq, y_bin, tr_i, te_i)
        c_res_hard = evaluate_precomputed_svm(Kc_moons["rbf"], y_bin, tr_i, te_i)
        q_res_moons = evaluate_precomputed_svm(Kq, y_moons, train_idx, test_idx)
        depth_sweep.append(
            {
                "depth": dp,
                "bandwidth": bw,
                "g_CQ_vs_rbf": g_val,
                "mean_offdiag_kernel": float(np.mean(off_diag)),
                "var_offdiag_kernel": float(np.var(off_diag)),
                "effective_rank": effective_rank(Kq),
                "qsvm_acc_moons": q_res_moons["test_acc"],
                "qsvm_acc_hard_target": q_res_hard["test_acc"],
                "rbf_acc_hard_target": c_res_hard["test_acc"],
            }
        )

    return {
        "n_samples": n_samples,
        "n_qubits": n_qubits,
        "regularization_lambda": lam_reg,
        "tasks": tasks_eval,
        "depth_concentration_sweep": depth_sweep,
    }


def plot_fig6(part_a: Dict, part_b: Dict, out_path: Path) -> None:
    """Create 300 DPI, 4-panel publication figure for Experiment 6."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.6), dpi=300, constrained_layout=True)

    c_navy = "#1B3B6F"
    c_teal = "#0E8A78"
    c_crimson = "#C62828"
    c_amber = "#D97706"
    c_purple = "#6D28D9"
    c_slate = "#475569"

    n_arr = np.array(part_a["n_list"], dtype=float)
    slopes = part_a["log2_slopes_vs_n"]

    # --- Panel (a): Barren Plateaus — Gradient Variance vs n ---
    ax = axes[0, 0]
    ax.semilogy(
        n_arr,
        part_a["var_local_shallow_L1"],
        marker="o",
        linewidth=2.1,
        color=c_teal,
        label=f"Local $Z_1$, $L=1$ (slope {slopes['local_shallow_L1']:+.2f}, trainable)",
    )
    ax.semilogy(
        n_arr,
        part_a["var_global_shallow_L1"],
        marker="D",
        linewidth=1.9,
        linestyle="-.",
        color=c_amber,
        label=f"Global $|0\\rangle\\langle 0|^{{\\otimes n}}$, $L=1$ (slope {slopes['global_shallow_L1_n_ge_4']:.2f})",
    )
    ax.semilogy(
        n_arr,
        part_a["var_local_deep_L3n"],
        marker="s",
        linewidth=2.1,
        color=c_navy,
        label=f"Local $Z_1$, deep $L=3n$ (slope {slopes['local_deep_L3n_n_ge_4']:.2f}, th. $-1.00$)",
    )
    ax.semilogy(
        n_arr,
        part_a["haar_2design_exact_local"],
        linestyle="--",
        linewidth=1.2,
        color=c_navy,
        alpha=0.65,
    )
    ax.semilogy(
        n_arr,
        part_a["var_global_deep_L3n"],
        marker="^",
        linewidth=2.1,
        color=c_crimson,
        label=f"Global $|0\\rangle\\langle 0|^{{\\otimes n}}$, deep $L=3n$ (slope {slopes['global_deep_L3n_n_ge_4']:.2f}, th. $-2.00$)",
    )
    ax.semilogy(
        n_arr,
        part_a["haar_2design_exact_global"],
        linestyle="--",
        linewidth=1.2,
        color=c_crimson,
        alpha=0.65,
    )
    ax.set_title("(a) Barren Plateaus: Exact Gradient Variance $\\mathrm{Var}_\\theta[\\partial_k C]$")
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel("Gradient variance $\\mathrm{Var}_\\theta[\\partial_k C]$ (log scale)")
    ax.set_xticks(part_a["n_list"])
    ax.set_ylim(1e-8, 3.0)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="lower left", framealpha=0.94)
    ax.text(
        0.98,
        0.97,
        "Dashed: exact Haar 2-design\nLocal: $\\sim 2^{-n}/2$ | Global: $\\sim 4^{-n}/2$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    # --- Panel (b): Required Measurement Shots N_shots = 1 / Var (Correction C1!) ---
    ax = axes[0, 1]
    ax.semilogy(
        n_arr,
        part_a["shots_local_shallow_L1"],
        marker="o",
        linewidth=2.1,
        color=c_teal,
        label="Local $Z_1$, $L=1$: $\\mathcal{O}(1)$ (classically simulable cone)",
    )
    ax.semilogy(
        n_arr,
        part_a["shots_local_deep_L3n"],
        marker="s",
        linewidth=2.1,
        color=c_navy,
        label="Local $Z_1$, deep $L=3n$: $\\Omega(2^n)$ (Correction C1)",
    )
    ax.semilogy(
        n_arr,
        part_a["shots_global_deep_L3n"],
        marker="^",
        linewidth=2.1,
        color=c_crimson,
        label="Global $|0\\rangle\\langle 0|^{{\\otimes n}}$, deep $L=3n$: $\\Omega(4^n) = \\Omega(2^{2n})$",
    )
    ax.semilogy(n_arr, 2.0 * (2.0**n_arr), linestyle="--", linewidth=1.2, color=c_navy, alpha=0.6)
    ax.semilogy(n_arr, 2.0 * (4.0**n_arr), linestyle="--", linewidth=1.2, color=c_crimson, alpha=0.6)
    ax.set_title("(b) Measurement Cost $N_{\\mathrm{shots}} = \\Omega(1/\\mathrm{Var})$ (Correction C1)")
    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel("Shots required for $\\mathrm{SNR}=1$ (log scale)")
    ax.set_xticks(part_a["n_list"])
    ax.set_ylim(1.5, 2e8)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", framealpha=0.94)
    ax.annotate(
        f"$n=12$: $2^{{12}}$ scale ($\\approx {part_a['shots_local_deep_L3n'][-1]:,.0f}$)\n"
        f"vs $4^{{12}}$ scale ($\\approx {part_a['shots_global_deep_L3n'][-1]:.1e}$)",
        xy=(12, part_a["shots_local_deep_L3n"][-1]),
        xytext=(5.2, 6.0e2),
        arrowprops=dict(arrowstyle="->", color=c_slate, lw=1.2),
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    # --- Panel (c): Classical SVM vs QSVM across Classical & Quantum-Structured Tasks ---
    ax = axes[1, 0]
    tasks = part_b["tasks"]
    task_labels = ["Classical\nMake Moons", "Classical\nGaussian Blobs", "Quantum Target\n(Max $g_{CQ}$ Eigenvector)"]
    x_pos = np.arange(len(task_labels))
    w_bar = 0.19

    rbf_accs = [
        tasks["classical_make_moons"]["svm_rbf"]["test_acc"],
        tasks["classical_gaussian_blobs"]["svm_rbf"]["test_acc"],
        tasks["quantum_structured_max_gCQ"]["svm_rbf"]["test_acc"],
    ]
    poly_accs = [
        tasks["classical_make_moons"]["svm_poly3"]["test_acc"],
        tasks["classical_gaussian_blobs"]["svm_poly3"]["test_acc"],
        tasks["quantum_structured_max_gCQ"]["svm_poly3"]["test_acc"],
    ]
    q_good_accs = [
        tasks["classical_make_moons"]["qsvm_shallow"]["test_acc"],
        tasks["classical_gaussian_blobs"]["qsvm_shallow"]["test_acc"],
        tasks["quantum_structured_max_gCQ"]["qsvm_struct"]["test_acc"],
    ]
    q_deep_accs = [
        tasks["classical_make_moons"]["qsvm_deep_scrambling"]["test_acc"],
        tasks["classical_gaussian_blobs"]["qsvm_deep_scrambling"]["test_acc"],
        tasks["quantum_structured_max_gCQ"]["qsvm_deep_scrambling"]["test_acc"],
    ]

    b1 = ax.bar(x_pos - 1.5 * w_bar, rbf_accs, width=w_bar, color=c_slate, label="Classical RBF SVM")
    b2 = ax.bar(x_pos - 0.5 * w_bar, poly_accs, width=w_bar, color=c_amber, label="Classical Poly-3 SVM")
    b3 = ax.bar(x_pos + 0.5 * w_bar, q_good_accs, width=w_bar, color=c_teal, label="QSVM (Shallow / Structured)")
    b4 = ax.bar(x_pos + 1.5 * w_bar, q_deep_accs, width=w_bar, color=c_crimson, label="QSVM (Deep Scrambling, $L=10$)")

    for bars in (b1, b2, b3, b4):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + 0.015,
                f"{100 * h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.6,
            )

    ax.axhline(0.50, color="#64748B", linestyle=":", linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_labels)
    ax.set_ylim(0.0, 1.30)
    ax.set_ylabel("Held-out test accuracy ($N_{\\mathrm{test}}=60$)")
    ax.set_title("(c) Classical vs Quantum SVM Test Accuracy ($H_4$ & Huang et al. 2021)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper center", framealpha=0.94, ncol=2)

    # --- Panel (d): Depth / Bandwidth Sweep — g_CQ, Kernel Concentration & Generalization ---
    ax = axes[1, 1]
    sw = part_b["depth_concentration_sweep"]
    idx_s = np.arange(len(sw))
    x_lbls = [f"L{r['depth']}\nb={r['bandwidth']:.2f}" for r in sw]
    g_vals = [r["g_CQ_vs_rbf"] for r in sw]
    q_hard = [r["qsvm_acc_hard_target"] for r in sw]
    r_hard = [r["rbf_acc_hard_target"] for r in sw]
    q_moon = [r["qsvm_acc_moons"] for r in sw]
    off_k = [r["mean_offdiag_kernel"] for r in sw]

    ax.plot(idx_s, q_hard, marker="o", linewidth=2.1, color=c_teal, label="QSVM acc. on max-$g_{CQ}$ target")
    ax.plot(idx_s, r_hard, marker="s", linewidth=1.9, color=c_slate, linestyle="--", label="Classical RBF acc. on max-$g_{CQ}$ target")
    ax.plot(idx_s, q_moon, marker="^", linewidth=1.8, color=c_amber, linestyle=":", label="QSVM acc. on classical make_moons")
    ax.axhline(0.50, color="#94A3B8", linestyle=":", linewidth=1.1)
    ax.set_ylabel("Held-out test accuracy")
    ax.set_ylim(0.22, 1.08)
    ax.set_xticks(idx_s)
    ax.set_xticklabels(x_lbls, fontsize=8.0)
    ax.set_xlabel("Quantum feature-map regime (depth $L$, encoding bandwidth $b$)")
    ax.set_title("(d) Goldilocks Dilemma: $g_{CQ}$ vs Exponential Kernel Concentration")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(idx_s, g_vals, marker="D", linewidth=1.9, color=c_purple, alpha=0.85, label="Geometric diff. $g_{CQ}$ (right axis)")
    ax2.set_ylabel("Geometric difference $g_{CQ}$", color=c_purple)
    ax2.tick_params(axis="y", labelcolor=c_purple)
    ax2.set_ylim(-1.8, max(g_vals) * 1.22)

    # Combine legends cleanly in the lower empty region
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.94, fontsize=7.9)
    ax.text(
        0.98,
        0.04,
        f"Off-diag $\\bar{{K}}_Q$: {off_k[0]:.2f} (L1) $\\to$ {off_k[-1]:.3f} (L10)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    fig.suptitle(
        "Experiment 6 — Barren Plateaus ($2^{-n}$ vs $4^{-n}$) & Geometric Difference $g_{CQ}$ in Quantum Kernels (Hypothesis $H_4$ & Correction C1)",
        fontsize=12.4,
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

    print("=== Experiment 6: Barren Plateaus & Geometric Difference g_CQ (H4) ===")
    part_a = run_part_a_barren_plateaus(rng)
    t1 = time.perf_counter()
    print(f"[Part A completed in {t1 - t0:.2f}s] Parameter-shift vs FD max err = {part_a['parameter_shift_vs_fd_max_error']:.2e}")
    for i, n in enumerate(part_a["n_list"]):
        print(
            f"  n={n:2d}: Var(loc,L1)={part_a['var_local_shallow_L1'][i]:.3e}, "
            f"Var(glob,L1)={part_a['var_global_shallow_L1'][i]:.3e}, "
            f"Var(loc,L3n)={part_a['var_local_deep_L3n'][i]:.3e} (Haar {part_a['haar_2design_exact_local'][i]:.3e}), "
            f"Var(glob,L3n)={part_a['var_global_deep_L3n'][i]:.3e} (Haar {part_a['haar_2design_exact_global'][i]:.3e})"
        )
    print(f"  Log2 slopes (n>=4): {part_a['log2_slopes_vs_n']}")

    part_b = run_part_b_geometric_difference(rng)
    t2 = time.perf_counter()
    print(f"[Part B completed in {t2 - t1:.2f}s]")
    for tname, td in part_b["tasks"].items():
        print(f"  Task [{tname}]:")
        for k, v in td.items():
            if isinstance(v, dict) and "test_acc" in v:
                print(f"    {k:22s}: train={v['train_acc']:.3f}, test={v['test_acc']:.3f} (C={v['best_C']})")
            elif isinstance(v, float):
                print(f"    {k:22s}: {v:.4f}")

    fig_path = FIGURES_DIR / "fig6_barren_plateaus_gcq.png"
    plot_fig6(part_a, part_b, fig_path)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment": "exp6_barren_plateaus_gcq",
        "seed": SEED,
        "runtime_seconds": round(elapsed, 3),
        "part_a_barren_plateaus": part_a,
        "part_b_geometric_difference": part_b,
    }
    json_path = RESULTS_DIR / "exp6_barren_plateaus.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Figure: {fig_path}")
    print(f"Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
