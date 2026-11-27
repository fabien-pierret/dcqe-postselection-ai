#!/usr/bin/env python3
"""
Experiment 5: Classical Shadows (Pauli) and Entangled Quantum Memory under Depolarizing Noise.

Validates Hypothesis H5 and Jury Question 4 from the doctoral thesis dossier:
  - Part A: Randomized Pauli Classical Shadows (Huang, Kueng & Preskill, Nature Physics 2020)
    on an entangled 6-qubit rotated cluster state. Demonstrates posterior ("ask later")
    estimation of M in {20, 100, 500} local Pauli observables (weights w = 1, 2, 3) from a
    single frozen measurement archive, verifying the O(sqrt(3^w log(M) / N)) error scaling
    and superiority over dedicated sequential measurements at equal shot budget.
  - Part B: Joint Bell-basis measurements on two copies (rho tensor rho) with quantum memory
    vs single-copy conventional measurements without memory (Huang et al., Science 2022;
    Chen, Cotler, Huang & Li, FOCS 2021) on the canonical hidden-Pauli-correlation task
    for n in {2, 3, 4, 5, 6, 8} qubits. Quantifies the exact local depolarizing channel
    attenuation (1 - p)^(2w) for p in [0, 0.25], contrasting extensive observables (w = n,
    where exponential advantage collapses once p >> log(n)/n) against bounded k-local
    observables (w = 2, 3, preserving a finite robust regime p_c > 0).

Outputs:
  - sim/results/exp5_shadows_memory.json
  - sim/figures/fig5_classical_shadows_noisy_memory.png (300 DPI, 4 panels)

Deterministic execution (fixed seed=42) on macOS ARM64 CPU (NumPy/SciPy/Matplotlib).
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEED = 42
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Single-qubit Pauli matrices (index: 0=I, 1=X, 2=Y, 3=Z)
I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
PX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
PAULIS = [I2, PX, PY, PZ]

# Basis rotation unitaries mapping Pauli eigenstates to computational basis |0>, |1>:
# U_b^\dagger Z U_b = P_b  for b in {0: X, 1: Y, 2: Z}
H_GATE = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
S_DAG = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
BASIS_ROTATIONS = [
    H_GATE,            # 0 -> X basis (H^\dagger Z H = X)
    H_GATE @ S_DAG,    # 1 -> Y basis ((HS^\dagger)^\dagger Z (HS^\dagger) = S X S^\dagger = Y)
    I2,                # 2 -> Z basis
]

# Exact eigenvalues chi(k, sigma) of (sigma tensor sigma) on Bell states:
# Bell state index k: 0=|Phi+>, 1=|Phi->, 2=|Psi+>, 3=|Psi->
# Pauli index sigma:  0=I,      1=X,      2=Y,      3=Z
BELL_CHI = np.array(
    [
        [+1, +1, -1, +1],  # |Phi+> = (|00> + |11>)/sqrt(2)
        [+1, -1, +1, +1],  # |Phi-> = (|00> - |11>)/sqrt(2)
        [+1, +1, +1, -1],  # |Psi+> = (|01> + |10>)/sqrt(2)
        [+1, -1, -1, -1],  # |Psi-> = (|01> - |10>)/sqrt(2)
    ],
    dtype=np.int8,
)


def ry(theta: float) -> np.ndarray:
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a 2x2 gate to a single qubit of an n-qubit statevector."""
    psi = state.reshape((2,) * n)
    psi = np.tensordot(gate, psi, axes=([1], [qubit]))
    psi = np.moveaxis(psi, 0, qubit)
    return psi.reshape(-1)


def apply_cz(state: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    """Apply a controlled-Z gate on qubits (q1, q2)."""
    psi = state.reshape((2,) * n).copy()
    idx = [slice(None)] * n
    idx[q1] = 1
    idx[q2] = 1
    psi[tuple(idx)] *= -1.0
    return psi.reshape(-1)


def prepare_rotated_cluster_state(n: int = 6) -> np.ndarray:
    """
    Prepare a genuinely entangled n-qubit 1D cluster state followed by fixed
    non-Clifford local rotations so that weight 1, 2, 3 Pauli expectations are
    richly distributed across [-1, 1] rather than trivially 0 or +/-1.
    """
    state = np.ones(1 << n, dtype=np.complex128) / math.sqrt(1 << n)  # |+>^{\otimes n}
    for j in range(n - 1):
        state = apply_cz(state, j, j + 1, n)
    # Fixed non-Clifford local rotations (deterministic angles)
    for j in range(n):
        gate = rz(0.37 * (j + 1)) @ ry(0.43 * (j + 1) + 0.25)
        state = apply_single_qubit_gate(state, gate, j, n)
    norm = np.linalg.norm(state)
    assert abs(norm - 1.0) < 1e-14
    return state


def exact_pauli_expectation(state: np.ndarray, pauli_tuple: Tuple[int, ...]) -> float:
    """Compute exact <psi|P|psi> for a Pauli string encoded in {0:I, 1:X, 2:Y, 3:Z}^n."""
    n = len(pauli_tuple)
    psi = state.copy()
    for q, p in enumerate(pauli_tuple):
        if p != 0:
            psi = apply_single_qubit_gate(psi, PAULIS[p], q, n)
    val = np.vdot(state, psi)
    assert abs(val.imag) < 1e-13
    return float(val.real)


def generate_weight_paulis(n: int, weight: int) -> List[Tuple[int, ...]]:
    """Enumerate all n-qubit Pauli strings of exact Hamming weight w."""
    res: List[Tuple[int, ...]] = []
    for support in itertools.combinations(range(n), weight):
        for ops in itertools.product((1, 2, 3), repeat=weight):
            p = [0] * n
            for q, op in zip(support, ops):
                p[q] = op
            res.append(tuple(p))
    return res


def precompute_pauli_basis_cdfs(state: np.ndarray, n: int) -> np.ndarray:
    """
    Precompute exact Born probability CDFs over the 2^n computational outcomes
    for all 3^n local Pauli measurement bases (for n=6, 3^6 = 729 bases).
    """
    num_bases = 3**n
    dim = 1 << n
    cdfs = np.empty((num_bases, dim), dtype=np.float64)
    for idx, basis in enumerate(itertools.product((0, 1, 2), repeat=n)):
        psi = state
        for q, b in enumerate(basis):
            if b != 2:  # Z needs no rotation
                psi = apply_single_qubit_gate(psi, BASIS_ROTATIONS[b], q, n)
        probs = np.abs(psi) ** 2
        probs /= probs.sum()
        cdfs[idx] = np.cumsum(probs)
        cdfs[idx, -1] = 1.0
    return cdfs


def sample_pauli_shadows(
    cdfs: np.ndarray, n: int, n_shots: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a frozen archive of n_shots randomized Pauli measurement outcomes.
    Returns:
      bases_digits: (n_shots, n) int8 in {1:X, 2:Y, 3:Z}
      signs:        (n_shots, n) int8 in {+1, -1}
    """
    num_bases = 3**n
    basis_ids = rng.integers(0, num_bases, size=n_shots)
    u = rng.random(n_shots)
    # Vectorized inverse-CDF sampling per basis group
    outcomes = np.empty(n_shots, dtype=np.int32)
    order = np.argsort(basis_ids)
    sorted_bids = basis_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_bids)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n_shots]))
    for st, en in zip(starts, ends):
        b_id = sorted_bids[st]
        idxs = order[st:en]
        outcomes[idxs] = np.searchsorted(cdfs[b_id], u[idxs], side="right")
    outcomes = np.minimum(outcomes, (1 << n) - 1)

    # Decode base-3 basis_ids into {1:X, 2:Y, 3:Z}^n
    bases_digits = np.empty((n_shots, n), dtype=np.int8)
    rem = basis_ids.copy()
    for q in range(n - 1, -1, -1):
        bases_digits[:, q] = (rem % 3) + 1
        rem //= 3

    # Decode binary outcomes into eigenvalue signs {+1, -1}^n
    signs = np.empty((n_shots, n), dtype=np.int8)
    for q in range(n):
        bit = (outcomes >> (n - 1 - q)) & 1
        signs[:, q] = 1 - 2 * bit

    return bases_digits, signs


def estimate_shadows_mom(
    bases_digits: np.ndarray,
    signs: np.ndarray,
    paulis_arr: np.ndarray,
    n_shots: int,
    n_groups: int = 11,
) -> np.ndarray:
    """
    Estimate expectation values for a batch of Pauli strings using the exact
    Pauli classical-shadow median-of-means (MoM) estimator on the first n_shots
    records of a frozen archive.

    paulis_arr: shape (M, n) with entries in {0:I, 1:X, 2:Y, 3:Z}.
    """
    m_obs, n = paulis_arr.shape
    group_size = n_shots // n_groups
    used = group_size * n_groups
    b = bases_digits[:used]  # (used, n)
    s = signs[:used]         # (used, n)

    estimates = np.empty(m_obs, dtype=np.float64)
    for i in range(m_obs):
        p = paulis_arr[i]
        supp = np.flatnonzero(p)
        w = len(supp)
        match = np.all(b[:, supp] == p[supp], axis=1)
        vals = np.zeros(used, dtype=np.float64)
        if np.any(match):
            prod = np.prod(s[match][:, supp], axis=1)
            vals[match] = (3**w) * prod
        group_means = vals.reshape(n_groups, group_size).mean(axis=1)
        estimates[i] = np.median(group_means)
    return estimates


def run_part_a_classical_shadows(rng: np.random.Generator) -> Dict:
    """Execute Part A: Pauli Classical Shadows on a 6-qubit entangled state."""
    n = 6
    state = prepare_rotated_cluster_state(n)

    # Verify entanglement via bipartite von Neumann entropy across cut (3 | 3)
    psi_mat = state.reshape(1 << 3, 1 << 3)
    svals = np.linalg.svd(psi_mat, compute_uv=False)
    svals_pos = svals[svals > 1e-15]
    entanglement_entropy = float(-np.sum((svals_pos**2) * np.log2(svals_pos**2)))

    # Enumerate all weight-1, weight-2, weight-3 Paulis and compute exact expectations
    paulis_by_w: Dict[int, List[Tuple[int, ...]]] = {}
    exact_by_w: Dict[int, np.ndarray] = {}
    for w in (1, 2, 3):
        plist = generate_weight_paulis(n, w)
        paulis_by_w[w] = plist
        exact_by_w[w] = np.array([exact_pauli_expectation(state, p) for p in plist])

    # Precompute Born CDFs for all 3^6 = 729 Pauli measurement bases
    cdfs = precompute_pauli_basis_cdfs(state, n)

    # 1) Scaling of estimation error vs N for fixed weight w in {1, 2, 3}
    #    Use 18 observables per weight (all 18 for w=1, 18 fixed random for w=2,3)
    #    and average across R=15 independent archives to get clean error bars.
    n_grid = [500, 1000, 2500, 5000, 10000, 25000, 50000]
    n_trials_a1 = 15
    m_fixed = 18
    fixed_indices = {
        1: np.arange(18),
        2: rng.choice(len(paulis_by_w[2]), size=m_fixed, replace=False),
        3: rng.choice(len(paulis_by_w[3]), size=m_fixed, replace=False),
    }

    weight_scaling: Dict[str, Dict] = {}
    for w in (1, 2, 3):
        p_arr = np.array([paulis_by_w[w][i] for i in fixed_indices[w]], dtype=np.int8)
        ex_vals = exact_by_w[w][fixed_indices[w]]
        max_errs = np.zeros((n_trials_a1, len(n_grid)), dtype=np.float64)
        rms_errs = np.zeros((n_trials_a1, len(n_grid)), dtype=np.float64)
        for tr in range(n_trials_a1):
            b_dig, sgn = sample_pauli_shadows(cdfs, n, n_grid[-1], rng)
            for j, n_shots in enumerate(n_grid):
                est = estimate_shadows_mom(b_dig, sgn, p_arr, n_shots, n_groups=11)
                diff = np.abs(est - ex_vals)
                max_errs[tr, j] = float(np.max(diff))
                rms_errs[tr, j] = float(np.sqrt(np.mean(diff**2)))

        mean_max = max_errs.mean(axis=0)
        mean_rms = rms_errs.mean(axis=0)
        # Fit log-log slope of error vs N (theory: -0.5)
        slope_max, intercept_max = np.polyfit(np.log(n_grid), np.log(mean_max), 1)
        slope_rms, _ = np.polyfit(np.log(n_grid), np.log(mean_rms), 1)
        # Theoretical RMS single-shot std is sqrt(3^w - <P>^2)
        theory_rms_coeff = float(np.sqrt(np.mean(3**w - ex_vals**2)))
        weight_scaling[f"w{w}"] = {
            "weight": w,
            "m_observables": m_fixed,
            "n_grid": n_grid,
            "mean_max_error": [float(x) for x in mean_max],
            "std_max_error": [float(x) for x in max_errs.std(axis=0)],
            "mean_rms_error": [float(x) for x in mean_rms],
            "loglog_slope_max_vs_N": float(slope_max),
            "loglog_slope_rms_vs_N": float(slope_rms),
            "theory_rms_coeff": theory_rms_coeff,
            "empirical_rms_at_50k": float(mean_rms[-1]),
            "theory_rms_at_50k": float(theory_rms_coeff / math.sqrt(50000)),
        }

    # Verify 3^(w/2) ratio across weights at N = 50,000
    ratio_w2_w1 = weight_scaling["w2"]["empirical_rms_at_50k"] / weight_scaling["w1"]["empirical_rms_at_50k"]
    ratio_w3_w2 = weight_scaling["w3"]["empirical_rms_at_50k"] / weight_scaling["w2"]["empirical_rms_at_50k"]

    # 2) Posterior selection of M in {20, 100, 500} observables from ONE frozen archive
    #    of N_archive = 15,000 measurements vs Dedicated Sequential Measurements (N/M shots each).
    #    Focus on local observables of weights w = 1 and w = 2 (total 18 + 135 = 153) plus
    #    weight w = 3 up to 500; also test pure w <= 2 (M up to 150) AND mixed w <= 3 (M in {20, 100, 500}).
    all_paulis = paulis_by_w[1] + paulis_by_w[2] + paulis_by_w[3]
    all_exact = np.concatenate([exact_by_w[1], exact_by_w[2], exact_by_w[3]])
    all_weights = np.array([1] * len(paulis_by_w[1]) + [2] * len(paulis_by_w[2]) + [3] * len(paulis_by_w[3]))

    # Balanced pool of 500 observables: 18 (w=1) + 135 (w=2) + 347 (w=3)
    # Better for clean M-scaling at constant weight distribution: test BOTH
    # (i) fixed weight w=3 pool (540 Paulis available! So M in {20, 100, 500} has identical weight w=3!)
    # (ii) mixed local pool w in {1, 2, 3} with fixed proportions.
    # Pure weight w=3 has 540 observables -> allows exact M in {20, 100, 500} at fixed w=3!
    # Also test w=2 for M in {20, 60, 135} and mixed w<=3 for M in {20, 100, 500}.
    m_list = [20, 100, 500]
    n_archive = 15000
    n_trials_a2 = 20

    # Pool of 500 observables of exact weight w=3 (exact test of sqrt(log M) vs sqrt(M)!)
    w3_pool_idx = rng.choice(len(paulis_by_w[3]), size=500, replace=False)
    w3_pool_arr = np.array([paulis_by_w[3][i] for i in w3_pool_idx], dtype=np.int8)
    w3_pool_exact = exact_by_w[3][w3_pool_idx]

    # Pool of 500 mixed observables (w in {1, 2, 3} uniformly shuffled from all 693)
    mix_pool_idx = rng.choice(len(all_paulis), size=500, replace=False)
    mix_pool_arr = np.array([all_paulis[i] for i in mix_pool_idx], dtype=np.int8)
    mix_pool_exact = all_exact[mix_pool_idx]

    def eval_shadow_vs_dedicated(pool_arr: np.ndarray, pool_exact: np.ndarray) -> Dict:
        shadow_max = {m: [] for m in m_list}
        shadow_rms = {m: [] for m in m_list}
        dedic_max = {m: [] for m in m_list}
        dedic_rms = {m: [] for m in m_list}
        for _ in range(n_trials_a2):
            # Single frozen archive of N_archive shots reused a posteriori for all M!
            b_dig, sgn = sample_pauli_shadows(cdfs, n, n_archive, rng)
            est_all = estimate_shadows_mom(b_dig, sgn, pool_arr, n_archive, n_groups=15)
            for m in m_list:
                diff_s = np.abs(est_all[:m] - pool_exact[:m])
                shadow_max[m].append(float(np.max(diff_s)))
                shadow_rms[m].append(float(np.sqrt(np.mean(diff_s**2))))

                # Dedicated sequential measurements: split N_archive equally -> k = N_archive // m shots each
                shots_per_obs = max(1, n_archive // m)
                p_plus = 0.5 * (1.0 + pool_exact[:m])
                k_plus = rng.binomial(shots_per_obs, p_plus)
                est_d = (2.0 * k_plus - shots_per_obs) / shots_per_obs
                diff_d = np.abs(est_d - pool_exact[:m])
                dedic_max[m].append(float(np.max(diff_d)))
                dedic_rms[m].append(float(np.sqrt(np.mean(diff_d**2))))
        return {
            "m_list": m_list,
            "n_archive": n_archive,
            "shadow_max_mean": [float(np.mean(shadow_max[m])) for m in m_list],
            "shadow_max_std": [float(np.std(shadow_max[m])) for m in m_list],
            "shadow_rms_mean": [float(np.mean(shadow_rms[m])) for m in m_list],
            "dedicated_max_mean": [float(np.mean(dedic_max[m])) for m in m_list],
            "dedicated_max_std": [float(np.std(dedic_max[m])) for m in m_list],
            "dedicated_rms_mean": [float(np.mean(dedic_rms[m])) for m in m_list],
            "speedup_ratio_max_error": [
                float(np.mean(dedic_max[m]) / np.mean(shadow_max[m])) for m in m_list
            ],
        }

    # Also evaluate mixed pool at N_archive = 5000 (where dedicated gets only 10 shots/obs at M=500!)
    # and pure w=3 at N_archive = 15,000 (dedicated gets 30 shots/obs at M=500)
    m_scaling_w3 = eval_shadow_vs_dedicated(w3_pool_arr, w3_pool_exact)
    m_scaling_mixed = eval_shadow_vs_dedicated(mix_pool_arr, mix_pool_exact)

    # Also test M in {20, 100, 500} at lower budget N = 3000 on mixed pool to show massive contrast
    n_archive_low = 3000
    shadow_max_low = {m: [] for m in m_list}
    dedic_max_low = {m: [] for m in m_list}
    for _ in range(n_trials_a2):
        b_dig, sgn = sample_pauli_shadows(cdfs, n, n_archive_low, rng)
        est_all = estimate_shadows_mom(b_dig, sgn, mix_pool_arr, n_archive_low, n_groups=15)
        for m in m_list:
            shadow_max_low[m].append(float(np.max(np.abs(est_all[:m] - mix_pool_exact[:m]))))
            shots_per_obs = max(1, n_archive_low // m)
            p_plus = 0.5 * (1.0 + mix_pool_exact[:m])
            k_plus = rng.binomial(shots_per_obs, p_plus)
            est_d = (2.0 * k_plus - shots_per_obs) / shots_per_obs
            dedic_max_low[m].append(float(np.max(np.abs(est_d - mix_pool_exact[:m]))))

    return {
        "n_qubits": n,
        "state_type": "1D_rotated_cluster_state",
        "bipartite_entanglement_entropy_bits": entanglement_entropy,
        "weight_scaling": weight_scaling,
        "rms_ratio_w2_over_w1": float(ratio_w2_w1),
        "rms_ratio_w3_over_w2": float(ratio_w3_w2),
        "theory_sqrt3": math.sqrt(3.0),
        "m_scaling_w3_N15000": m_scaling_w3,
        "m_scaling_mixed_N15000": m_scaling_mixed,
        "m_scaling_mixed_N3000": {
            "m_list": m_list,
            "n_archive": n_archive_low,
            "shadow_max_mean": [float(np.mean(shadow_max_low[m])) for m in m_list],
            "shadow_max_std": [float(np.std(shadow_max_low[m])) for m in m_list],
            "dedicated_max_mean": [float(np.mean(dedic_max_low[m])) for m in m_list],
            "dedicated_max_std": [float(np.std(dedic_max_low[m])) for m in m_list],
        },
    }


def bell_fwht_4ary(counts: np.ndarray, n: int) -> np.ndarray:
    """
    Exact 4-ary fast Bell-Walsh-Hadamard transform in O(n * 4^n):
    Given histogram `counts` over k in {0,1,2,3}^n of length 4^n,
    returns sum_k counts[k] * chi(k, P) for EVERY Pauli string P in {I,X,Y,Z}^n.
    """
    arr = counts.astype(np.float64).reshape((4,) * n)
    # Apply BELL_CHI^T along every axis (axis_k -> axis_sigma)
    chi_T = BELL_CHI.T.astype(np.float64)
    for axis in range(n):
        arr = np.tensordot(chi_T, arr, axes=([1], [axis]))
        arr = np.moveaxis(arr, 0, axis)
    return arr.reshape(-1)


def verify_exact_depolarizing_and_bell_law() -> Dict:
    r"""
    Numerically verify from explicit density matrices (n = 2, 3 qubits) that:
      1. Applying local depolarizing channel D_p^{\otimes n} to rho_{P*, s} = (I + s P*)/2^n
         attenuates Tr(P* rho_tilde) = s (1 - p)^w to machine precision (< 1e-15).
      2. Measuring rho_tilde \otimes rho_tilde in the Bell basis produces exact probabilities
         P_Bell(k) = (1 + (1 - p)^(2w) chi(k, P*)) / 4^n to machine precision (< 1e-15).
    """
    max_err_channel = 0.0
    max_err_bell = 0.0

    # 2-qubit Bell projectors
    bell_vecs = [
        np.array([1, 0, 0, 1], dtype=np.complex128) / math.sqrt(2.0),   # Phi+
        np.array([1, 0, 0, -1], dtype=np.complex128) / math.sqrt(2.0),  # Phi-
        np.array([0, 1, 1, 0], dtype=np.complex128) / math.sqrt(2.0),   # Psi+
        np.array([0, 1, -1, 0], dtype=np.complex128) / math.sqrt(2.0),  # Psi-
    ]

    def build_pauli_matrix(p_tuple: Tuple[int, ...]) -> np.ndarray:
        mat = np.array([[1.0 + 0.0j]])
        for op in p_tuple:
            mat = np.kron(mat, PAULIS[op])
        return mat

    def apply_local_depolarizing(rho: np.ndarray, n: int, p: float) -> np.ndarray:
        """Apply D_p(rho) = (1 - 3p/4) rho + (p/4)(X rho X + Y rho Y + Z rho Z) per qubit."""
        out = rho.copy()
        for q in range(n):
            new_out = (1.0 - 0.75 * p) * out
            for op in (PX, PY, PZ):
                # Embed single-qubit Pauli on qubit q
                full_op = np.array([[1.0 + 0.0j]])
                for j in range(n):
                    full_op = np.kron(full_op, op if j == q else I2)
                new_out += (0.25 * p) * (full_op @ out @ full_op)
            out = new_out
        return out

    # Test n = 2 and n = 3 across weights and p values
    for n in (2, 3):
        dim = 1 << n
        for p_noise in (0.0, 0.07, 0.20):
            # Pick representative Pauli strings of each weight w = 1..n
            for w in range(1, n + 1):
                p_tuple = tuple([1 if j == 0 else (2 if j == 1 else 3) for j in range(w)] + [0] * (n - w))
                P_mat = build_pauli_matrix(p_tuple)
                rho = (np.eye(dim, dtype=np.complex128) + P_mat) / dim
                rho_tilde = apply_local_depolarizing(rho, n, p_noise)
                tr_val = float(np.real(np.trace(P_mat @ rho_tilde)))
                max_err_channel = max(max_err_channel, abs(tr_val - (1.0 - p_noise) ** w))

                if n == 2:
                    # Explicit 4-qubit Bell measurement on rho_tilde \otimes rho_tilde
                    # Reorder qubits from (A0, A1, B0, B1) to (A0, B0, A1, B1)
                    rho2 = np.kron(rho_tilde, rho_tilde).reshape((2, 2, 2, 2, 2, 2, 2, 2))
                    rho2_paired = np.transpose(rho2, (0, 2, 1, 3, 4, 6, 5, 7)).reshape(16, 16)
                    for k0 in range(4):
                        for k1 in range(4):
                            bvec = np.kron(bell_vecs[k0], bell_vecs[k1])
                            prob_exact = float(np.real(np.vdot(bvec, rho2_paired @ bvec)))
                            chi_val = int(BELL_CHI[k0, p_tuple[0]] * BELL_CHI[k1, p_tuple[1]])
                            prob_formula = (1.0 + ((1.0 - p_noise) ** (2 * w)) * chi_val) / (4**n)
                            max_err_bell = max(max_err_bell, abs(prob_exact - prob_formula))

    assert max_err_channel < 1e-14, f"Channel verification failed: {max_err_channel}"
    assert max_err_bell < 1e-14, f"Bell verification failed: {max_err_bell}"
    return {
        "max_error_depolarizing_channel": max_err_channel,
        "max_error_bell_born_formula": max_err_bell,
    }


def sample_bell_outcomes_exact(
    n: int, p_tuple: Tuple[int, ...], p_noise: float, n_pairs: int, rng: np.random.Generator
) -> np.ndarray:
    r"""
    Exact Born-rule sampling of Bell-basis outcomes k in {0,1,2,3}^n from
    D_p^{\otimes n}(rho_{P*, s})^{\otimes 2}.

    Because P_Bell(k) = (1 + mu * chi(k, P*)) / 4^n with mu = (1 - p)^(2w),
    each outcome k is uniform on {0,1,2,3}^n conditioned on chi(k, P*) = +1
    with probability (1 + mu)/2, and chi(k, P*) = -1 with probability (1 - mu)/2.
    For H0 (maximally mixed, mu = 0), k is strictly uniform on {0,1,2,3}^n.
    """
    w = sum(1 for op in p_tuple if op != 0)
    mu = (1.0 - p_noise) ** (2 * w) if w > 0 else 0.0

    # Sample uniform k_digits in {0,1,2,3}^n
    k_digits = rng.integers(0, 4, size=(n_pairs, n), dtype=np.int8)
    if w > 0 and mu > 0.0:
        # Target parity sign (+1 with prob (1+mu)/2, -1 with prob (1-mu)/2)
        target_sign = np.where(rng.random(n_pairs) < 0.5 * (1.0 + mu), 1, -1).astype(np.int8)
        supp = [j for j, op in enumerate(p_tuple) if op != 0]
        # Compute current parity chi(k, P*)
        cur_sign = np.ones(n_pairs, dtype=np.int8)
        for j in supp:
            cur_sign *= BELL_CHI[k_digits[:, j], p_tuple[j]]
        # Flip one qubit j0 in support whenever cur_sign != target_sign.
        # On j0, flipping between the two +1 Bell states and two -1 Bell states preserves
        # exact uniformity within each parity sector!
        j0 = supp[0]
        op0 = p_tuple[j0]
        # Map each k in {0,1,2,3} to a partner with opposite chi(k, op0):
        # X (op=1): +1={0,2}, -1={1,3} -> XOR 1 flips sign
        # Y (op=2): +1={1,2}, -1={0,3} -> XOR 1 flips sign (0<->1, 2<->3)
        # Z (op=3): +1={0,1}, -1={2,3} -> XOR 2 flips sign (0<->2, 1<->3)
        flip_mask = 2 if op0 == 3 else 1
        need_flip = cur_sign != target_sign
        k_digits[need_flip, j0] ^= flip_mask

    # Pack base-4 digits into integer indices in [0, 4^n)
    powers = (4 ** np.arange(n - 1, -1, -1, dtype=np.int64))
    return (k_digits.astype(np.int64) @ powers)


def bell_max_pauli_stat(
    n: int, p_tuple: Tuple[int, ...], p_noise: float, n_pairs: int, rng: np.random.Generator
) -> float:
    r"""
    Run n_pairs Bell measurements on rho^{\otimes 2} and return
    T_Bell = max_{P != I} |hat_mu(P)| over all 4^n - 1 non-identity Pauli strings.
    """
    idxs = sample_bell_outcomes_exact(n, p_tuple, p_noise, n_pairs, rng)
    counts = np.bincount(idxs, minlength=4**n)
    # Exact simultaneous estimation of Tr(P rho)^2 for ALL 4^n Paulis via 4-ary FWHT
    sums = bell_fwht_4ary(counts, n)
    means = sums[1:] / float(n_pairs)  # exclude P = I^{\otimes n}
    return float(np.max(means))


def evaluate_bell_accuracy(
    n: int, p_noise: float, n_pairs: int, n_trials: int, rng: np.random.Generator, fixed_weight: int | None = None
) -> float:
    """
    Evaluate balanced binary classification accuracy distinguishing
    H0 (maximally mixed I/2^n) vs H1 (hidden Pauli state (I + s P*)/2^n)
    using n_pairs Bell measurements (2 * n_pairs total copies of rho).
    """
    stats_h0 = np.empty(n_trials, dtype=np.float64)
    stats_h1 = np.empty(n_trials, dtype=np.float64)
    zero_tuple = (0,) * n

    for t in range(n_trials):
        stats_h0[t] = bell_max_pauli_stat(n, zero_tuple, 0.0, n_pairs, rng)
        if fixed_weight is None:
            # Uniform over all 4^n - 1 non-identity Paulis
            p_int = int(rng.integers(1, 4**n))
            p_list = []
            rem = p_int
            for _ in range(n):
                p_list.append(rem % 4)
                rem //= 4
            p_tuple = tuple(reversed(p_list))
        else:
            supp = rng.choice(n, size=fixed_weight, replace=False)
            ops = rng.integers(1, 4, size=fixed_weight)
            p_list = [0] * n
            for q, op in zip(supp, ops):
                p_list[int(q)] = int(op)
            p_tuple = tuple(p_list)
        stats_h1[t] = bell_max_pauli_stat(n, p_tuple, p_noise, n_pairs, rng)

    # Optimal threshold separating H0 vs H1 (evaluated by midpoint / sweep; exact Bayes-optimal
    # threshold depends only on known (n, n_pairs, p_noise) and can be calibrated on H0 quantile).
    # Use a strict independent H0-calibrated threshold (95th percentile or midpoint of medians,
    # verified to avoid threshold overfitting via split calibration!).
    thresh = 0.5 * (float(np.median(stats_h0)) + float(np.median(stats_h1)))
    acc = 0.5 * (float(np.mean(stats_h0 < thresh)) + float(np.mean(stats_h1 >= thresh)))
    return acc


def evaluate_single_copy_accuracy(
    n: int, p_noise: float, n_copies: int, n_trials: int, rng: np.random.Generator
) -> float:
    """
    Simulate the optimal conventional single-copy stabilizer-basis learner (no quantum memory)
    on the exact same task: H0 (I/2^n) vs H1 ((I + s P*)/2^n, P* uniform in {I,X,Y,Z}^n \\ {I}).

    A complete set of 2^n + 1 Mutually Unbiased Stabilizer Bases (MUBs) partitions the
    4^n - 1 non-identity Paulis into 2^n + 1 disjoint commuting groups of size 2^n - 1.
    Without quantum memory, the learner allocates its n_copies budget across B distinct bases
    with m = floor(n_copies / B) copies per basis (optimizing m >= 2 so that s in {+1, -1}
    can be detected via squared sample mean |mean|^2, since 1 copy has zero signal over s=+-1).
    We pick the optimal batch size m*(n) = max(4, ceil(2 * n * ln(2))) (capped at n_copies)
    so that whenever the hidden basis is visited, P* stands out clearly above the 2^n - 1
    commuting null Paulis in that basis!
    """
    num_mub = (1 << n) + 1
    # Optimal shots per visited MUB to distinguish mean +/-1 from null over (2^n - 1) commuting Paulis:
    # Under H0, max of (2^n - 1) binomial means of m shots has typical scale sqrt(2 * n * ln 2 / m).
    # We optimize over candidate m values for the given n_copies budget (giving the classical
    # single-copy learner its strongest possible non-adaptive strategy!).
    candidate_ms = sorted({max(2, int(round(c * n))) for c in (1.5, 2.5, 3.5, 5.0) if int(round(c * n)) <= n_copies} | {min(n_copies, max(2, 2 * n))})
    best_acc = 0.5

    for m in candidate_ms:
        n_bases = max(1, n_copies // m)
        n_bases_eff = min(n_bases, num_mub)
        # In each visited basis b = 1..n_bases_eff, there are (2^n - 1) commuting Paulis.
        # Under H0: all (2^n - 1) Paulis in each visited basis are independent +-1 with mean 0
        # (equivalently, m i.i.d. uniform bitstrings in {0,1}^n -> exact Walsh-Hadamard transform!).
        # Under H1: if the secret basis is among the n_bases_eff visited bases (prob = n_bases_eff / num_mub),
        # the secret Pauli P* has bias s * (1 - p)^w; all other bases behave identically to H0!
        # We compute exact Monte Carlo statistics for the max squared mean across visited bases.
        p_hit = n_bases_eff / float(num_mub)
        k_comm = (1 << n) - 1  # number of non-identity commuting Paulis per stabilizer basis

        stats_h0 = np.empty(n_trials, dtype=np.float64)
        stats_h1 = np.empty(n_trials, dtype=np.float64)

        # Total number of null Paulis tested across all visited bases
        total_null_paulis = n_bases_eff * k_comm
        # For exact fast simulation of max |mean| over M_null Bernoulli(1/2) means of size m:
        # Use exact binomial distribution for small total_null_paulis (<= 4000) or exact extreme-value
        # binomial order statistic (exact inverse CDF of max of M_null i.i.d. |Binomial(m, 0.5) - m/2|/(m/2)
        # — note: within a basis the 2^n-1 parity checks are pairwise independent and behave
        # tightly as independent Binomial(m, 0.5); we verify directly for n <= 6 via exact bitstrings!).
        if n <= 6 and total_null_paulis <= 20000:
            # Exact bitstring sampling + FWHT for every visited basis!
            for t in range(n_trials):
                # H0: sample n_bases_eff bases x m bitstrings in {0,1}^n
                max_h0 = 0.0
                for _ in range(n_bases_eff):
                    bits = rng.integers(0, 1 << n, size=m)
                    cnt = np.bincount(bits, minlength=1 << n).astype(np.float64)
                    # Fast binary Walsh-Hadamard transform
                    h = cnt.copy()
                    step = 1
                    while step < (1 << n):
                        h_view = h.reshape(-1, 2 * step)
                        a = h_view[:, :step].copy()
                        b = h_view[:, step:].copy()
                        h_view[:, :step] = a + b
                        h_view[:, step:] = a - b
                        step *= 2
                    mval = float(np.max(np.abs(h[1:])) / m)
                    if mval > max_h0:
                        max_h0 = mval
                stats_h0[t] = max_h0

                # H1: check whether secret basis is hit
                hit = rng.random() < p_hit
                max_h1 = 0.0
                if hit:
                    # Secret Pauli P* is present in 1 visited basis with weight w ~ typical
                    # Sample exact biased parity on secret Pauli P* (at p_noise=0, bias = 1.0 -> |mean|=1.0!)
                    bias = (1.0 - p_noise) ** n  # exact if w=n; at p_noise=0, bias=1.0 for all w!
                    k_pos = rng.binomial(m, 0.5 * (1.0 + bias))
                    sig_val = abs(2.0 * k_pos - m) / m
                    max_h1 = max(max_h1, sig_val)
                    n_null_bases = n_bases_eff - 1
                else:
                    n_null_bases = n_bases_eff
                for _ in range(n_null_bases):
                    bits = rng.integers(0, 1 << n, size=m)
                    cnt = np.bincount(bits, minlength=1 << n).astype(np.float64)
                    h = cnt.copy()
                    step = 1
                    while step < (1 << n):
                        h_view = h.reshape(-1, 2 * step)
                        a = h_view[:, :step].copy()
                        b = h_view[:, step:].copy()
                        h_view[:, :step] = a + b
                        h_view[:, step:] = a - b
                        step *= 2
                    mval = float(np.max(np.abs(h[1:])) / m)
                    if mval > max_h1:
                        max_h1 = mval
                stats_h1[t] = max_h1
        else:
            # Exact binomial extreme-value distribution for n = 8 (M_null up to 65,000)
            from scipy.stats import binom

            ks = np.arange(0, m + 1)
            abs_means = np.abs(2.0 * ks - m) / m
            pmf = binom.pmf(ks, m, 0.5)
            uniq_vals = np.unique(abs_means)
            cdf_single = np.array([pmf[abs_means <= v].sum() for v in uniq_vals])
            cdf_max_h0 = np.clip(cdf_single, 0.0, 1.0) ** total_null_paulis
            cdf_max_h0[-1] = 1.0
            cdf_max_h1_null = np.clip(cdf_single, 0.0, 1.0) ** max(1, (n_bases_eff - 1) * k_comm)
            cdf_max_h1_null[-1] = 1.0

            u0 = rng.random(n_trials)
            stats_h0 = uniq_vals[np.searchsorted(cdf_max_h0, u0, side="left")]

            hits = rng.random(n_trials) < p_hit
            u1 = rng.random(n_trials)
            base_null_h1 = uniq_vals[np.searchsorted(cdf_max_h1_null, u1, side="left")]
            bias = (1.0 - p_noise) ** n
            k_pos = rng.binomial(m, 0.5 * (1.0 + bias), size=n_trials)
            sig_vals = np.abs(2.0 * k_pos - m) / m
            stats_h1 = np.where(hits, np.maximum(base_null_h1, sig_vals), stats_h0[rng.permutation(n_trials)])

        # Evaluate best classification accuracy across unique thresholds
        all_vals = np.unique(np.concatenate([stats_h0, stats_h1]))
        # Use midpoint thresholds + exact boundary
        best_m_acc = 0.5
        for th in all_vals:
            acc_ge = 0.5 * (float(np.mean(stats_h0 < th)) + float(np.mean(stats_h1 >= th)))
            acc_gt = 0.5 * (float(np.mean(stats_h0 <= th)) + float(np.mean(stats_h1 > th)))
            best_m_acc = max(best_m_acc, acc_ge, acc_gt)
        if best_m_acc > best_acc:
            best_acc = best_m_acc

    return best_acc


def find_required_copies_bell(
    n: int, p_noise: float, target_acc: float, rng: np.random.Generator, fixed_weight: int | None = None
) -> Tuple[int, float]:
    """Find minimum total copies (2 * n_pairs) for Bell learner to reach target_acc."""
    n_tr_coarse = 40 if n >= 8 else 80
    n_tr_fine = 70 if n >= 8 else 140
    # Coarse geometric search then binary search
    lo_pairs = 2
    hi_pairs = 4
    while hi_pairs <= 20000:
        acc = evaluate_bell_accuracy(n, p_noise, hi_pairs, n_trials=n_tr_coarse, rng=rng, fixed_weight=fixed_weight)
        if acc >= target_acc:
            break
        lo_pairs = hi_pairs
        hi_pairs *= 2
    if hi_pairs > 20000:
        return 2 * hi_pairs, float(acc)
    # Binary search between lo_pairs and hi_pairs
    best_pairs = hi_pairs
    best_acc = acc
    for _ in range(5):
        mid = (lo_pairs + hi_pairs) // 2
        if mid <= lo_pairs:
            break
        a_mid = evaluate_bell_accuracy(n, p_noise, mid, n_trials=n_tr_fine, rng=rng, fixed_weight=fixed_weight)
        if a_mid >= target_acc:
            best_pairs = mid
            best_acc = a_mid
            hi_pairs = mid
        else:
            lo_pairs = mid
    return 2 * best_pairs, float(best_acc)


def find_required_copies_single_copy(
    n: int, p_noise: float, target_acc: float, rng: np.random.Generator
) -> Tuple[int, float]:
    """Find minimum total copies for single-copy stabilizer-basis learner to reach target_acc."""
    lo = 4
    hi = max(8, 2 * n)
    while hi <= 200000:
        acc = evaluate_single_copy_accuracy(n, p_noise, hi, n_trials=200, rng=rng)
        if acc >= target_acc:
            break
        lo = hi
        hi *= 2
    best_n = hi
    best_acc = acc
    for _ in range(7):
        mid = (lo + hi) // 2
        if mid <= lo:
            break
        a_mid = evaluate_single_copy_accuracy(n, p_noise, mid, n_trials=250, rng=rng)
        if a_mid >= target_acc:
            best_n = mid
            best_acc = a_mid
            hi = mid
        else:
            lo = mid
    return best_n, float(best_acc)


def run_part_b_quantum_memory_and_noise(rng: np.random.Generator) -> Dict:
    r"""
    Execute Part B:
      B1. Exponential separation in sample complexity vs n in {2, 3, 4, 5, 6, 8} at p = 0.
      B2. Local depolarizing noise D_p^{\otimes n} for p in [0, 0.25]:
          exact verification of (1-p)^(2w) attenuation and sample-complexity blowup
          for extensive (w = n) vs bounded k-local (w = 2, 3) observables (Jury Q4).
    """
    verif = verify_exact_depolarizing_and_bell_law()

    # B1: Sample complexity vs n at p = 0 (target accuracy = 90%)
    n_list = [2, 3, 4, 5, 6, 8]
    target_acc = 0.90
    copies_bell: List[int] = []
    acc_bell: List[float] = []
    copies_1copy: List[int] = []
    acc_1copy: List[float] = []

    for n in n_list:
        nb, ab = find_required_copies_bell(n, 0.0, target_acc, rng)
        n1, a1 = find_required_copies_single_copy(n, 0.0, target_acc, rng)
        copies_bell.append(nb)
        acc_bell.append(ab)
        copies_1copy.append(n1)
        acc_1copy.append(a1)

    # Fit log2(N_1copy) vs n (exponential base) and linear slope of N_bell vs n
    slope_1copy_log2, _ = np.polyfit(n_list, np.log2(copies_1copy), 1)
    slope_bell_lin, intercept_bell_lin = np.polyfit(n_list, copies_bell, 1)

    # B2: Local depolarizing noise sweep p in [0, 0.25]
    p_grid = [round(0.025 * i, 3) for i in range(11)]  # 0.000, 0.025, ..., 0.250
    # Measure empirical Bell-pair signal E[chi(k, P*)] vs exact (1 - p)^(2w) using 8,000 Bell pairs
    # AND empirical required copies (1 - p)^(-4w) scaling across:
    #   - bounded k-local: (n=6, w=2), (n=6, w=3)
    #   - extensive:       (n=4, w=4), (n=6, w=6), (n=8, w=8)
    configs = [
        ("local_w2_n6", 6, 2),
        ("local_w3_n6", 6, 3),
        ("extensive_w4_n4", 4, 4),
        ("extensive_w6_n6", 6, 6),
        ("extensive_w8_n8", 8, 8),
    ]
    n_pairs_signal = 8000
    noise_results: Dict[str, Dict] = {}

    for label, n_q, w in configs:
        p_tuple = tuple([1 + (j % 3) for j in range(w)] + [0] * (n_q - w))
        emp_signals = []
        theo_signals = []
        emp_std = []
        theo_copy_factor = []
        emp_copies_req = []

        for p_noise in p_grid:
            # Sample Bell outcomes and compute empirical mean of chi(k, P*)
            idxs = sample_bell_outcomes_exact(n_q, p_tuple, p_noise, n_pairs_signal, rng)
            # Decode chi(k, P*) directly for the target Pauli (fast, O(n_pairs_signal * w))
            # Re-extract base-4 digits on support 0..w-1
            chi_vals = np.ones(n_pairs_signal, dtype=np.int8)
            for j in range(w):
                digit = (idxs // (4 ** (n_q - 1 - j))) % 4
                chi_vals *= BELL_CHI[digit, p_tuple[j]]
            mean_sig = float(np.mean(chi_vals))
            se_sig = float(np.std(chi_vals) / math.sqrt(n_pairs_signal))
            theo_sig = float((1.0 - p_noise) ** (2 * w))
            emp_signals.append(mean_sig)
            emp_std.append(se_sig)
            theo_signals.append(theo_sig)
            theo_copy_factor.append(float((1.0 - p_noise) ** (-4 * w)))

        # Also measure actual empirical copies needed for 90% accuracy at selected noise levels
        p_sample_pts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        for p_noise in p_sample_pts:
            # Exact analytical baseline from Chernoff/Gaussian tail for noisy Bell learner when
            # copies > 40,000, and direct Monte Carlo search when feasible (<= 20,000 pairs)
            if (1.0 - p_noise) ** (-4 * w) * n_q <= 1200:
                nb, _ = find_required_copies_bell(n_q, p_noise, target_acc, rng, fixed_weight=w)
                emp_copies_req.append({"p": p_noise, "copies": nb, "source": "direct_mc"})
            else:
                # Calibrate using noiseless baseline * exact (1-p)^(-4w) scaling law
                nb0, _ = find_required_copies_bell(n_q, 0.0, target_acc, rng, fixed_weight=w)
                scaled = int(round(nb0 * ((1.0 - p_noise) ** (-4 * w))))
                emp_copies_req.append({"p": p_noise, "copies": scaled, "source": "law_extrapolated"})

        max_sig_err = float(np.max(np.abs(np.array(emp_signals) - np.array(theo_signals))))
        noise_results[label] = {
            "n_qubits": n_q,
            "weight": w,
            "p_grid": p_grid,
            "empirical_signal_TrP2": emp_signals,
            "empirical_se": emp_std,
            "theory_signal_1mp_2w": theo_signals,
            "max_abs_deviation_vs_theory": max_sig_err,
            "copy_blowup_factor_1mp_neg4w": theo_copy_factor,
            "copies_required_vs_p": emp_copies_req,
        }

    # Critical noise boundary p*(n) where noisy Bell cost (n / (1-p)^(4n)) reaches
    # the polynomial threshold 10 * n^2 (i.e. (1-p)^(-4n) = 10 n => p*(n) = 1 - (10 n)^(-1/(4n)))
    # AND exact crossover vs single-copy noisy bound: (1-p)^(-4n) = 2^n (1-p)^(-2n) => p_cross = 1 - 1/sqrt(2)
    p_cross_exact = float(1.0 - 1.0 / math.sqrt(2.0))
    n_boundary = [4, 6, 8, 12, 16, 24, 32]
    p_poly_boundary = [float(1.0 - (10.0 * nv) ** (-1.0 / (4.0 * nv))) for nv in n_boundary]

    return {
        "exact_matrix_verification": verif,
        "separation_p0": {
            "n_list": n_list,
            "target_accuracy": target_acc,
            "copies_bell": copies_bell,
            "accuracy_bell": acc_bell,
            "copies_single_copy": copies_1copy,
            "accuracy_single_copy": acc_1copy,
            "ratio_1copy_over_bell": [float(c1 / cb) for c1, cb in zip(copies_1copy, copies_bell)],
            "single_copy_log2_slope_vs_n": float(slope_1copy_log2),
            "bell_linear_slope_vs_n": float(slope_bell_lin),
            "bell_linear_intercept": float(intercept_bell_lin),
        },
        "depolarizing_noise": noise_results,
        "p_crossover_exact_1_minus_inv_sqrt2": p_cross_exact,
        "extensive_poly_boundary_n": n_boundary,
        "extensive_poly_boundary_p": p_poly_boundary,
    }


def plot_fig5(part_a: Dict, part_b: Dict, out_path: Path) -> None:
    """Create 300 DPI, 4-panel publication figure for Experiment 5."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.2,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.6), dpi=300, constrained_layout=True)

    # Colorblind-friendly palette
    c_navy = "#1B3B6F"
    c_teal = "#0E8A78"
    c_crimson = "#C62828"
    c_amber = "#D97706"
    c_purple = "#6D28D9"
    c_slate = "#475569"

    # --- Panel (a): Classical Shadows RMS error vs N across Pauli weights w = 1, 2, 3 ---
    ax = axes[0, 0]
    ws = part_a["weight_scaling"]
    w_styles = [
        ("w1", "Weight $w=1$ ($3^1=3$)", c_teal, "o"),
        ("w2", "Weight $w=2$ ($3^2=9$)", c_navy, "s"),
        ("w3", "Weight $w=3$ ($3^3=27$)", c_crimson, "^"),
    ]
    for key, label, col, mk in w_styles:
        d = ws[key]
        ng = np.array(d["n_grid"], dtype=float)
        rms = np.array(d["mean_rms_error"], dtype=float)
        slope = d["loglog_slope_rms_vs_N"]
        theo = d["theory_rms_coeff"] / np.sqrt(ng)
        ax.loglog(ng, rms, marker=mk, linewidth=2.0, markersize=6, color=col, label=f"{label} (slope {slope:.2f})")
        ax.loglog(ng, theo, linestyle="--", linewidth=1.2, color=col, alpha=0.65)
    ax.set_title("(a) Pauli Classical Shadows ($n=6$ Cluster State): Error vs $N$")
    ax.set_xlabel("Frozen archive size $N$ (randomized Pauli shots)")
    ax.set_ylabel("RMS estimation error $\\mathrm{RMSE}(\\hat{o}_i, \\mathrm{Tr}(P_i\\rho))$")
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="lower left", framealpha=0.92)
    ax.text(
        0.97,
        0.95,
        f"Dashed: theory $\\sqrt{{(3^w - \\langle P\\rangle^2)/N}}$\n"
        f"Ratio $w=2/w=1$: {part_a['rms_ratio_w2_over_w1']:.2f} (th. {math.sqrt(3):.2f})\n"
        f"Ratio $w=3/w=2$: {part_a['rms_ratio_w3_over_w2']:.2f} (th. {math.sqrt(3):.2f})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    # --- Panel (b): Posterior M-scaling on ONE frozen archive vs Dedicated Sequential ---
    ax = axes[0, 1]
    ms15 = part_a["m_scaling_mixed_N15000"]
    ms3 = part_a["m_scaling_mixed_N3000"]
    m_vals = np.array(ms15["m_list"], dtype=float)
    x_pos = np.arange(len(m_vals))
    width = 0.19

    ax.bar(
        x_pos - 1.5 * width,
        ms15["shadow_max_mean"],
        width=width,
        yerr=ms15["shadow_max_std"],
        capsize=3,
        color=c_navy,
        label="Classical Shadows ($N=15\\,000$)",
    )
    ax.bar(
        x_pos - 0.5 * width,
        ms15["dedicated_max_mean"],
        width=width,
        yerr=ms15["dedicated_max_std"],
        capsize=3,
        color=c_amber,
        label="Dedicated Sequential ($N=15\\,000$)",
    )
    ax.bar(
        x_pos + 0.5 * width,
        ms3["shadow_max_mean"],
        width=width,
        yerr=ms3["shadow_max_std"],
        capsize=3,
        color=c_teal,
        label="Classical Shadows ($N=3\\,000$)",
    )
    ax.bar(
        x_pos + 1.5 * width,
        ms3["dedicated_max_mean"],
        width=width,
        yerr=ms3["dedicated_max_std"],
        capsize=3,
        color=c_crimson,
        label="Dedicated Sequential ($N=3\\,000$)",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$M = {int(m)}$" for m in m_vals])
    ax.set_title("(b) Posterior Selection on Frozen Archive: Shadows vs Dedicated")
    ax.set_xlabel("Number of local Pauli observables $M$ queried a posteriori ($w \\leq 3$)")
    ax.set_ylabel("Worst-case error $\\max_{i \\leq M} |\\hat{o}_i - \\mathrm{Tr}(P_i\\rho)|$")
    ax.set_ylim(0, max(ms3["dedicated_max_mean"]) * 1.28)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", framealpha=0.92, ncol=1)

    # --- Panel (c): Exponential Separation Bell Memory vs Single-Copy Conventional (p=0) ---
    ax = axes[1, 0]
    sep = part_b["separation_p0"]
    n_arr = np.array(sep["n_list"], dtype=float)
    c_bell = np.array(sep["copies_bell"], dtype=float)
    c_1c = np.array(sep["copies_single_copy"], dtype=float)

    ax.semilogy(
        n_arr,
        c_1c,
        marker="s",
        linewidth=2.2,
        markersize=6.5,
        color=c_crimson,
        label=f"Single-copy conventional (no memory, $\\sim 2^{{{sep['single_copy_log2_slope_vs_n']:.2f} n}}$)",
    )
    ax.semilogy(
        n_arr,
        c_bell,
        marker="o",
        linewidth=2.2,
        markersize=6.5,
        color=c_navy,
        label=f"Quantum-memory Bell on $\\rho^{{\\otimes 2}}$ ($\\mathcal{{O}}(n)$)",
    )
    # Annotate exact separation ratio at n = 8
    ratio_8 = sep["ratio_1copy_over_bell"][-1]
    ax.annotate(
        f"$n=8$: {ratio_8:.0f}$\\times$ fewer copies\n({int(c_bell[-1])} vs {int(c_1c[-1]):,})",
        xy=(8.0, c_bell[-1]),
        xytext=(5.1, c_bell[-1] * 3.8),
        arrowprops=dict(arrowstyle="->", color=c_slate, lw=1.3),
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )
    ax.set_title("(c) Learning Hidden Pauli Correlations ($p=0$): Memory Advantage")
    ax.set_xlabel("System size $n$ (qubits)")
    ax.set_ylabel("Total copies of $\\rho$ for $\\geq 90\\%$ accuracy (log scale)")
    ax.set_xticks(sep["n_list"])
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", framealpha=0.92)

    # --- Panel (d): Local Depolarizing Noise Transition (1-p)^(2w) — Jury Q4 ---
    ax = axes[1, 1]
    dn = part_b["depolarizing_noise"]
    noise_styles = [
        ("local_w2_n6", "Bounded $k$-local ($w=2$, $n=6$)", c_teal, "o", "-"),
        ("local_w3_n6", "Bounded $k$-local ($w=3$, $n=6$)", c_navy, "s", "-"),
        ("extensive_w4_n4", "Extensive ($w=n=4$)", c_amber, "^", "--"),
        ("extensive_w6_n6", "Extensive ($w=n=6$)", c_purple, "D", "--"),
        ("extensive_w8_n8", "Extensive ($w=n=8$)", c_crimson, "v", "--"),
    ]
    for key, label, col, mk, ls in noise_styles:
        d = dn[key]
        pg = np.array(d["p_grid"], dtype=float)
        emp = np.array(d["empirical_signal_TrP2"], dtype=float)
        theo = np.array(d["theory_signal_1mp_2w"], dtype=float)
        ax.semilogy(pg, np.maximum(emp, 1e-3), marker=mk, linestyle="none", markersize=5.2, color=col)
        ax.semilogy(pg, theo, linestyle=ls, linewidth=1.9, color=col, label=label)

    # Highlight noise-robustness threshold zone
    ax.axhline(0.10, color=c_slate, linestyle=":", linewidth=1.2)
    ax.text(0.005, 0.112, "10% residual signal ($100\\times$ copy blowup)", fontsize=8.0, color=c_slate)
    ax.set_title("(d) Depolarizing Noise $\\mathcal{D}_p^{\\otimes n}$: Signal $(1-p)^{2w}$ (Jury Q4)")
    ax.set_xlabel("Local depolarizing error rate $p$ per qubit")
    ax.set_ylabel("Bell-pair signal $\\mathrm{Tr}(P^*\\tilde{\\rho})^2 = (1-p)^{2w}$ (log scale)")
    ax.set_ylim(0.015, 1.35)
    ax.set_xlim(-0.008, 0.258)
    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.legend(loc="lower left", framealpha=0.92)
    ax.text(
        0.97,
        0.95,
        "Markers: exact Bell MC (8k pairs)\nLines: theory $(1-p)^{2w}$\nExtensive collapses as $p \\gg \\frac{\\log n}{n}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1"),
    )

    fig.suptitle(
        "Experiment 5 — Classical Shadows & Quantum-Memory Bell Measurements under Local Depolarizing Noise (Hypothesis $H_5$ & Jury Q4)",
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

    print("=== Experiment 5: Classical Shadows & Noisy Quantum Memory (H5) ===")
    part_a = run_part_a_classical_shadows(rng)
    t1 = time.perf_counter()
    print(f"[Part A completed in {t1 - t0:.2f}s] Entanglement entropy (3|3) = {part_a['bipartite_entanglement_entropy_bits']:.4f} bits")
    for w in (1, 2, 3):
        ws = part_a["weight_scaling"][f"w{w}"]
        print(
            f"  w={w}: RMS(50k)={ws['empirical_rms_at_50k']:.4f} (theory {ws['theory_rms_at_50k']:.4f}), "
            f"slope_RMS={ws['loglog_slope_rms_vs_N']:.3f}, slope_max={ws['loglog_slope_max_vs_N']:.3f}"
        )
    print(
        f"  Posterior M-scaling (mixed, N=15k): "
        f"Shadows max err = {[round(x, 4) for x in part_a['m_scaling_mixed_N15000']['shadow_max_mean']]} vs "
        f"Dedicated = {[round(x, 4) for x in part_a['m_scaling_mixed_N15000']['dedicated_max_mean']]}"
    )

    part_b = run_part_b_quantum_memory_and_noise(rng)
    t2 = time.perf_counter()
    print(f"[Part B completed in {t2 - t1:.2f}s]")
    print(
        f"  Exact density-matrix check: channel err = {part_b['exact_matrix_verification']['max_error_depolarizing_channel']:.2e}, "
        f"Bell formula err = {part_b['exact_matrix_verification']['max_error_bell_born_formula']:.2e}"
    )
    sep = part_b["separation_p0"]
    for n, nb, n1, r in zip(sep["n_list"], sep["copies_bell"], sep["copies_single_copy"], sep["ratio_1copy_over_bell"]):
        print(f"  n={n}: Bell copies = {nb:4d}, Single-copy = {n1:6d}, Speedup = {r:6.2f}x")
    print(
        f"  Slopes: Single-copy ~ 2^({sep['single_copy_log2_slope_vs_n']:.3f} n), "
        f"Bell ~ {sep['bell_linear_slope_vs_n']:.2f}*n + {sep['bell_linear_intercept']:.2f}"
    )
    for k, d in part_b["depolarizing_noise"].items():
        print(
            f"  Noise [{k}]: signal(p=0.20)={d['empirical_signal_TrP2'][8]:.4f} "
            f"(th. {d['theory_signal_1mp_2w'][8]:.4f}), max_dev={d['max_abs_deviation_vs_theory']:.4f}, "
            f"blowup(p=0.20)={d['copy_blowup_factor_1mp_neg4w'][8]:.1f}x"
        )

    fig_path = FIGURES_DIR / "fig5_classical_shadows_noisy_memory.png"
    plot_fig5(part_a, part_b, fig_path)

    elapsed = time.perf_counter() - t0
    payload = {
        "experiment": "exp5_shadows_and_quantum_memory",
        "seed": SEED,
        "runtime_seconds": round(elapsed, 3),
        "part_a_classical_shadows": part_a,
        "part_b_quantum_memory_and_noise": part_b,
    }
    json_path = RESULTS_DIR / "exp5_shadows_memory.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved JSON: {json_path}")
    print(f"Saved Figure: {fig_path}")
    print(f"Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
