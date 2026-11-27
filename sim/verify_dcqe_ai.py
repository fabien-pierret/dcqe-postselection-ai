#!/usr/bin/env python3
"""Numerical verification suite for the PhD thesis on the Delayed-Choice
Quantum Eraser (DCQE), Post-Selection, and Artificial Intelligence.

This self-contained, deterministic script executes three rigorous numerical
experiments and generates three publication-grade figures:

1. Experiment 1 (``fig1_dcqe_no_signaling.png``):
   Exact bipartite Hilbert-space simulation of Kim et al. (2000) Delayed-Choice
   Quantum Eraser via density matrices, 4-outcome idler POVMs, steering
   assemblages, and Monte Carlo photon-pair sampling. Numerically proves
   exact fringe complementarity (V = 1 for R01 and R02) and exact marginal
   no-signaling invariance (< 1e-14 absolute error across all positions and
   across 100 random delayed-choice idler CPTP/POVM channels).

2. Experiment 2 (``fig2_latent_postselection_vs_chsh.png``):
   Part A: High-dimensional frozen latent-space model with Elhage-style
   geometric superposition (M = 128 quasi-orthogonal concepts in R^64) and
   scaled dot-product softmax attention. Shows high-entropy uniform marginal
   output under isotropic queries versus sharp complementary sub-structures
   under targeted prompts q_A, q_B without modifying frozen weights W*.
   Part B: Explicit Bell-CHSH falsifiable boundary comparing classical local
   hidden-variable (LHV) post-selection models (|S| <= 2, verified across all
   16 Bell polytope vertices, 5,000 random stochastic LHV models, and classical
   softmax-attention LHV models) against entangled quantum measurements on the
   Bell state |Phi+> reaching Tsirelson's bound S = 2*sqrt(2) ~= 2.8284, plus
   the Werner-state separability/steering/non-locality hierarchy.

3. Experiment 3 (``fig3_postselection_illusion_overfitting.png``):
   Part A: Pure i.i.d. white-noise data dredging across M in {1, 10, 100,
   1000, 10000} candidate strategies with strict In-Sample / Out-of-Sample
   temporal split (T_IS = T_OOS = 252). Demonstrates logarithmic mirage growth
   of apparent In-Sample Sharpe (> 3.8, naive p < 1e-4) matched against the
   Bailey-Lopez de Prado extreme-value law, alongside complete collapse of
   Out-of-Sample Sharpe to ~0.
   Part B: Berkson collider bias on N = 50,000 independent pairs (X _|_ Y)
   post-selected on Z = X + Y > tau, matched against the closed-form inverse
   Mills ratio law.

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
# Global reproducibility and aesthetic configuration
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"

# Cohesive colorblind-safe publication palette
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
    """Configure publication-grade Matplotlib styling."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.6,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CBD5E1",
            "grid.color": "#94A3B8",
            "grid.linestyle": "--",
            "grid.alpha": 0.28,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def partial_trace_idler(rho_si: np.ndarray) -> np.ndarray:
    """Compute Tr_I(rho_SI) for a 2-qubit (4x4) signal-idler density matrix."""
    reshaped = rho_si.reshape(2, 2, 2, 2)
    return np.einsum("aibj,ij->ab", reshaped, np.eye(2, dtype=complex))


def steering_element(rho_si: np.ndarray, povm_idler: np.ndarray) -> np.ndarray:
    """Compute unnormalized steered signal state Tr_I[(I_S (x) E_I) rho_SI]."""
    op = np.kron(np.eye(2, dtype=complex), povm_idler)
    return partial_trace_idler(op @ rho_si)


def random_haar_unitary_2x2(rng: np.random.Generator) -> np.ndarray:
    """Generate a Haar-random 2x2 unitary matrix via QR decomposition."""
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    diag = np.diag(r)
    phases = diag / np.abs(diag)
    return q * phases


def shannon_entropy_bits(probs: np.ndarray) -> float:
    """Compute Shannon entropy H(p) = -sum p log2(p) in bits."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def kl_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Kullback-Leibler divergence D_KL(p || q) in bits."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


# ===========================================================================
# EXPERIMENT 1: Delayed-Choice Quantum Eraser (Kim et al., 2000)
# ===========================================================================
def run_experiment_1(rng: np.random.Generator) -> dict[str, Any]:
    """Simulate Kim et al. (2000) DCQE and prove exact no-signaling."""
    # Physical geometry inspired by Kim et al. (2000, PRL 84, 1)
    wavelength_m = 702.2e-9      # Degenerate SPDC photons from 351.1 nm pump
    slit_sep_m = 0.70e-3         # Separation d between SPDC regions A and B
    slit_width_m = 0.20e-3       # Aperture width a (d/a = 3.5 > 1 => <psi_A|psi_B> = 0)
    screen_dist_m = 1.20         # Effective distance L to scanning detector D0

    # Spatial grid on D0 (mm)
    n_grid = 2401
    x_mm = np.linspace(-6.0, 6.0, n_grid)
    x_m = x_mm * 1e-3

    # Exact far-field single-slit diffraction envelope normalized in mm^-1:
    # integral_{-inf}^{+inf} sinc^2(a * x / (lambda * L)) dx = (lambda * L) / a
    alpha_per_mm = (slit_width_m / (wavelength_m * screen_dist_m)) * 1e-3
    envelope_amp = np.sqrt(alpha_per_mm) * np.sinc(alpha_per_mm * x_mm)
    envelope_intensity = envelope_amp ** 2  # |A(x)|^2 in mm^-1

    # Relative optical phase Delta_phi(x) = 2 * pi * d * x / (lambda * L)
    delta_phi = (2.0 * np.pi * slit_sep_m / (wavelength_m * screen_dist_m)) * x_m
    fringe_period_mm = (wavelength_m * screen_dist_m / slit_sep_m) * 1e3
    central_lobe_half_mm = (wavelength_m * screen_dist_m / slit_width_m) * 1e3

    # Bipartite maximally entangled path state |Psi>_SI = (|A>_S|A>_I + |B>_S|B>_I)/sqrt(2)
    ket_a = np.array([1.0, 0.0], dtype=complex)
    ket_b = np.array([0.0, 1.0], dtype=complex)
    ket_plus = (ket_a + ket_b) / np.sqrt(2.0)
    ket_minus = (ket_a - ket_b) / np.sqrt(2.0)

    psi_si = (np.kron(ket_a, ket_a) + np.kron(ket_b, ket_b)) / np.sqrt(2.0)
    rho_si = np.outer(psi_si, psi_si.conj())
    rho_s_reduced = partial_trace_idler(rho_si)

    # Exact 4-outcome idler POVM {E1, E2, E3, E4} corresponding to Kim et al. BS setup
    e1 = 0.5 * np.outer(ket_plus, ket_plus.conj())   # Eraser detector D1 (+)
    e2 = 0.5 * np.outer(ket_minus, ket_minus.conj())  # Eraser detector D2 (-)
    e3 = 0.5 * np.outer(ket_a, ket_a.conj())         # Which-path detector D3 (A)
    e4 = 0.5 * np.outer(ket_b, ket_b.conj())         # Which-path detector D4 (B)

    povm_completeness_err = float(np.max(np.abs((e1 + e2 + e3 + e4) - np.eye(2))))
    assert povm_completeness_err < 1e-15, "Idler POVM failed completeness!"

    # Unnormalized steering assemblage sigma_j = Tr_I[(I_S (x) E_j) rho_SI]
    sigma_1 = steering_element(rho_si, e1)
    sigma_2 = steering_element(rho_si, e2)
    sigma_3 = steering_element(rho_si, e3)
    sigma_4 = steering_element(rho_si, e4)

    steering_sum_err = float(
        np.max(np.abs((sigma_1 + sigma_2 + sigma_3 + sigma_4) - rho_s_reduced))
    )
    assert steering_sum_err < 1e-15, "Steering assemblage failed no-signaling sum!"

    # Signal spatial detection operator density M_S(x) on H_S
    # psi_A(x) = A(x) exp(-i Delta_phi/2), psi_B(x) = A(x) exp(+i Delta_phi/2)
    psi_a_x = envelope_amp * np.exp(-0.5j * delta_phi)
    psi_b_x = envelope_amp * np.exp(+0.5j * delta_phi)
    # Vectorized M_S(x) of shape (n_grid, 2, 2): M_S(x)_{kl} = psi_k*(x) psi_l(x)
    psi_vec = np.stack([psi_a_x, psi_b_x], axis=-1)  # (n_grid, 2)
    m_s_x = np.einsum("ni,nj->nij", psi_vec.conj(), psi_vec)

    # Joint detection probability densities R_0j(x) = Tr_S[M_S(x) sigma_j]
    r01 = np.real(np.einsum("nij,ji->n", m_s_x, sigma_1))
    r02 = np.real(np.einsum("nij,ji->n", m_s_x, sigma_2))
    r03 = np.real(np.einsum("nij,ji->n", m_s_x, sigma_3))
    r04 = np.real(np.einsum("nij,ji->n", m_s_x, sigma_4))
    r01_plus_r02 = r01 + r02
    r0_sum = r01 + r02 + r03 + r04
    r0_reduced = np.real(np.einsum("nij,ji->n", m_s_x, rho_s_reduced))

    # Closed-form analytic formulas for cross-verification
    r01_analytic = 0.5 * envelope_intensity * (np.cos(0.5 * delta_phi) ** 2)
    r02_analytic = 0.5 * envelope_intensity * (np.sin(0.5 * delta_phi) ** 2)
    r03_analytic = 0.25 * envelope_intensity
    r04_analytic = 0.25 * envelope_intensity

    err_r01_analytic = float(np.max(np.abs(r01 - r01_analytic)))
    err_r02_analytic = float(np.max(np.abs(r02 - r02_analytic)))
    err_r01_r02_cancel = float(np.max(np.abs(r01_plus_r02 - 0.5 * envelope_intensity)))
    err_r0_no_signaling = float(np.max(np.abs(r0_sum - r0_reduced)))
    err_r0_vs_envelope = float(np.max(np.abs(r0_sum - envelope_intensity)))

    assert err_r01_analytic < 1e-14
    assert err_r02_analytic < 1e-14
    assert err_r01_r02_cancel < 1e-14
    assert err_r0_no_signaling < 1e-14
    assert err_r0_vs_envelope < 1e-14

    # Envelope-normalized interference visibility inside central lobe (|x| <= 3.0 mm)
    central_mask = np.abs(x_mm) <= 3.0

    def compute_visibility(curve: np.ndarray, expected_weight: float) -> float:
        normalized = curve[central_mask] / (expected_weight * envelope_intensity[central_mask])
        c_max = float(np.max(normalized))
        c_min = float(np.min(normalized))
        return (c_max - c_min) / (c_max + c_min)

    vis_r01 = compute_visibility(r01, 0.25)
    vis_r02 = compute_visibility(r02, 0.25)
    vis_r03 = compute_visibility(r03, 0.25)
    vis_r04 = compute_visibility(r04, 0.25)
    vis_r01_r02 = compute_visibility(r01_plus_r02, 0.50)
    vis_r0_total = compute_visibility(r0_sum, 1.00)

    assert abs(vis_r01 - 1.0) < 1e-5
    assert abs(vis_r02 - 1.0) < 1e-5
    assert vis_r03 < 1e-14
    assert vis_r04 < 1e-14
    assert vis_r01_r02 < 1e-14
    assert vis_r0_total < 1e-14

    # General delayed-choice invariance test: 100 random idler unitary/CPTP/POVM measurements
    n_random_choices = 100
    max_cptp_marginal_err = 0.0
    for _ in range(n_random_choices):
        # Random 2-branch Kraus channel followed by random projective measurement
        u1 = random_haar_unitary_2x2(rng)
        u2 = random_haar_unitary_2x2(rng)
        p_mix = rng.uniform(0.0, 1.0)
        k1 = np.sqrt(p_mix) * u1
        k2 = np.sqrt(1.0 - p_mix) * u2
        # Apply local channel I_S (x) E_I to rho_SI
        ik1 = np.kron(np.eye(2, dtype=complex), k1)
        ik2 = np.kron(np.eye(2, dtype=complex), k2)
        rho_after = ik1 @ rho_si @ ik1.conj().T + ik2 @ rho_si @ ik2.conj().T

        # Random projective POVM in rotated basis
        u_meas = random_haar_unitary_2x2(rng)
        proj0 = np.outer(u_meas[:, 0], u_meas[:, 0].conj())
        proj1 = np.outer(u_meas[:, 1], u_meas[:, 1].conj())
        sig_a = steering_element(rho_after, proj0)
        sig_b = steering_element(rho_after, proj1)
        r_marg_test = np.real(np.einsum("nij,ji->n", m_s_x, sig_a + sig_b))
        err = float(np.max(np.abs(r_marg_test - r0_reduced)))
        if err > max_cptp_marginal_err:
            max_cptp_marginal_err = err

    assert max_cptp_marginal_err < 1e-14, "CPTP no-signaling invariance failed!"

    # Monte Carlo photon-pair click simulation (N = 200,000 entangled pairs)
    # Include explicit tail mass outside [-6, +6] mm so in-window histogram density
    # is 100% unbiased against the exact infinite-domain probability density R_0j(x).
    n_photons = 200_000
    dx_mm = x_mm[1] - x_mm[0]
    joint_probs = np.stack([r01, r02, r03, r04], axis=0) * dx_mm  # (4, n_grid)
    in_window_mass = float(np.sum(joint_probs))
    p_tail = max(0.0, 1.0 - in_window_mass)
    flat_probs = np.append(joint_probs.ravel(), p_tail)
    flat_probs /= np.sum(flat_probs)
    sampled_flat = rng.choice(flat_probs.size, size=n_photons, p=flat_probs)
    in_win_mask = sampled_flat < (4 * n_grid)
    sampled_in = sampled_flat[in_win_mask]
    det_ids = sampled_in // n_grid
    pos_indices = sampled_in % n_grid
    # Add sub-grid uniform jitter for continuous density histogramming
    x_clicks = x_mm[pos_indices] + rng.uniform(-0.5 * dx_mm, 0.5 * dx_mm, size=sampled_in.size)

    mc_counts = {
        "D1_in_window": int(np.sum(det_ids == 0)),
        "D2_in_window": int(np.sum(det_ids == 1)),
        "D3_in_window": int(np.sum(det_ids == 2)),
        "D4_in_window": int(np.sum(det_ids == 3)),
        "outside_window": int(np.sum(~in_win_mask)),
        "window_mass_exact": in_window_mass,
    }

    # --- Plot Figure 1 ---
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    fig.suptitle(
        "Experiment 1 — Delayed-Choice Quantum Eraser (Kim et al., 2000): Complementary Fringes vs. Exact No-Signaling",
        fontsize=12.8,
        fontweight="bold",
        y=1.02,
    )

    bins = np.linspace(-5.5, 5.5, 111)
    bin_width = bins[1] - bins[0]

    # Panel (a): Eraser coincidence fringes R01 and R02
    ax = axes[0, 0]
    for det_idx, color, label in [(0, PALETTE["navy"], None), (1, PALETTE["coral"], None)]:
        counts, edges = np.histogram(x_clicks[det_ids == det_idx], bins=bins)
        density = counts / (n_photons * bin_width)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.bar(centers, density, width=bin_width * 0.9, color=color, alpha=0.18, edgecolor="none")

    ax.plot(x_mm, r01, color=PALETTE["navy"], lw=2.0, label=r"$R_{01}(x)$ (Eraser $D_1$, $\propto \cos^2$, $V=1.000$)")
    ax.plot(x_mm, r02, color=PALETTE["coral"], lw=2.0, ls="--", label=r"$R_{02}(x)$ (Eraser $D_2$, $\propto \sin^2$, $V=1.000$)")
    ax.set_title("(a) Post-Selected Eraser Subsets ($D_1, D_2$ Coincidences)")
    ax.set_xlabel(r"Signal Detector $D_0$ Position $x$ (mm)")
    ax.set_ylabel(r"Joint Probability Density ($\mathrm{mm}^{-1}$)")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(0, max(np.max(r01), np.max(r02)) * 1.24)
    ax.grid(True)
    ax.legend(loc="upper right")

    # Panel (b): Which-path R03, R04 and Exact Cancellation R01 + R02
    ax = axes[0, 1]
    ax.plot(
        x_mm,
        r01_plus_r02,
        color=PALETTE["teal"],
        lw=2.4,
        label=r"$R_{01}(x) + R_{02}(x) = \frac{1}{2}|\mathcal{A}(x)|^2$ ($V=0$)",
    )
    ax.plot(
        x_mm,
        0.5 * envelope_intensity,
        color="black",
        lw=1.1,
        ls=":",
        label=r"Exact Envelope $\frac{1}{2}\mathrm{sinc}^2$ (max err $<10^{-15}$)",
    )
    ax.plot(
        x_mm,
        r03,
        color=PALETTE["purple"],
        lw=2.0,
        label=r"$R_{03}(x)$ (Which-Path $D_3$, $V=0$)",
    )
    ax.plot(
        x_mm,
        r04,
        color=PALETTE["amber"],
        lw=2.0,
        ls="--",
        label=r"$R_{04}(x)$ (Which-Path $D_4$, $V=0$)",
    )
    ax.set_title("(b) Exact Phase Cancellation & Which-Path Channels")
    ax.set_xlabel(r"Signal Detector $D_0$ Position $x$ (mm)")
    ax.set_ylabel(r"Probability Density ($\mathrm{mm}^{-1}$)")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(0, np.max(r01_plus_r02) * 1.32)
    ax.grid(True)
    ax.legend(loc="upper right")

    # Panel (c): Total Unconditioned Marginal on D0
    ax = axes[1, 0]
    total_counts, edges = np.histogram(x_clicks, bins=bins)
    total_density = total_counts / (n_photons * bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(
        centers,
        total_density,
        width=bin_width * 0.9,
        color=PALETTE["light_gray"],
        edgecolor=PALETTE["slate"],
        linewidth=0.4,
        alpha=0.75,
        label=f"Monte Carlo All $D_0$ Clicks ($N={n_photons:,}$)",
    )
    ax.plot(
        x_mm,
        r0_sum,
        color=PALETTE["navy"],
        lw=2.3,
        label=r"Total Marginal $R_0(x) = \sum_{j=1}^4 R_{0j}(x)$",
    )
    ax.plot(
        x_mm,
        r0_reduced,
        color=PALETTE["coral"],
        lw=1.6,
        ls=":",
        label=r"Reduced State $\mathrm{Tr}_S[M_S(x)\rho_S]$ ($V=0$)",
    )
    ax.set_title(r"(c) Unconditioned Screen $D_0$: Zero Fringes (No Retrocausality)")
    ax.set_xlabel(r"Signal Detector $D_0$ Position $x$ (mm)")
    ax.set_ylabel(r"Marginal Probability Density ($\mathrm{mm}^{-1}$)")
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(0, np.max(r0_sum) * 1.28)
    ax.grid(True)
    ax.legend(loc="upper right")

    # Panel (d): Visibility Bar Chart & Numerical No-Signaling Audit
    ax = axes[1, 1]
    labels = [
        r"$R_{01}$" + "\n(Eraser +)",
        r"$R_{02}$" + "\n(Eraser -)",
        r"$R_{03}$" + "\n(Path A)",
        r"$R_{04}$" + "\n(Path B)",
        r"$R_{01}+R_{02}$" + "\n(Sum)",
        r"$R_0$" + "\n(Total)",
    ]
    vis_vals = [vis_r01, vis_r02, vis_r03, vis_r04, vis_r01_r02, vis_r0_total]
    bar_colors = [
        PALETTE["navy"],
        PALETTE["coral"],
        PALETTE["purple"],
        PALETTE["amber"],
        PALETTE["teal"],
        PALETTE["slate"],
    ]
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, vis_vals, color=bar_colors, width=0.62, edgecolor="#1E293B", linewidth=0.7)
    for bar, v in zip(bars, vis_vals):
        text_y = v + 0.03 if v > 0.1 else 0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            f"{v:.4f}" if v > 1e-6 else "0.0000",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#0F172A",
        )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.32)
    ax.set_ylabel("Envelope-Normalized Fringe Visibility $V$")
    ax.set_title("(d) Fringe Visibility & No-Signaling Precision Audit")
    ax.grid(True, axis="y")

    audit_box = (
        "Exact Machine-Precision Verification:\n"
        f"• max |R01 + R02 - 0.5|A(x)|²| = {err_r01_r02_cancel:.2e}\n"
        f"• max |R0(x) - Tr_S[M_S ρ_S]| = {err_r0_no_signaling:.2e}\n"
        f"• max err (100 random CPTP/POVMs) = {max_cptp_marginal_err:.2e}\n"
        f"• Steering sum ||Σσ_j - ρ_S||_max = {steering_sum_err:.2e}"
    )
    ax.text(
        0.97,
        0.94,
        audit_box,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=PALETTE["bg_tint"], edgecolor="#94A3B8", lw=0.9),
    )

    out_path = FIGURES_DIR / "fig1_dcqe_no_signaling.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "wavelength_nm": wavelength_m * 1e9,
        "slit_sep_mm": slit_sep_m * 1e3,
        "slit_width_mm": slit_width_m * 1e3,
        "screen_dist_m": screen_dist_m,
        "fringe_period_mm": float(fringe_period_mm),
        "central_lobe_half_mm": float(central_lobe_half_mm),
        "visibility_R01": float(vis_r01),
        "visibility_R02": float(vis_r02),
        "visibility_R03": float(vis_r03),
        "visibility_R04": float(vis_r04),
        "visibility_R01_plus_R02": float(vis_r01_r02),
        "visibility_R0_total": float(vis_r0_total),
        "err_povm_completeness": povm_completeness_err,
        "err_steering_sum": steering_sum_err,
        "err_R01_analytic": err_r01_analytic,
        "err_R02_analytic": err_r02_analytic,
        "err_R01_R02_cancellation": err_r01_r02_cancel,
        "err_R0_no_signaling": err_r0_no_signaling,
        "err_R0_vs_envelope": err_r0_vs_envelope,
        "max_err_100_random_cptp": max_cptp_marginal_err,
        "mc_photons": n_photons,
        "mc_counts": mc_counts,
        "figure_path": str(out_path),
    }


# ===========================================================================
# EXPERIMENT 2: Frozen Latent-Space Post-Selection vs. Bell-CHSH Boundary
# ===========================================================================
def run_experiment_2(rng: np.random.Generator) -> dict[str, Any]:
    """Compare classical frozen-weight post-selection with quantum Bell-CHSH."""
    # -----------------------------------------------------------------------
    # Part A: Frozen high-dimensional latent representation & softmax attention
    # -----------------------------------------------------------------------
    d_model = 64        # Latent dimension R^d
    n_classes = 16      # Output vocabulary / semantic categories
    keys_per_class = 8
    n_concepts = n_classes * keys_per_class  # M = 128 > d = 64 (2x geometric superposition)

    # Orthogonal semantic prompt axes e_A, e_B in R^64
    e_a = np.zeros(d_model, dtype=float)
    e_b = np.zeros(d_model, dtype=float)
    e_a[0] = 1.0
    e_b[1] = 1.0

    # Elhage-style quasi-orthogonal feature directions packed into the remaining R^(d-2)
    raw_dirs = rng.normal(size=(n_concepts, d_model - 2))
    raw_dirs /= np.linalg.norm(raw_dirs, axis=1, keepdims=True)
    u_bg = np.zeros((n_concepts, d_model), dtype=float)
    u_bg[:, 2:] = raw_dirs

    # Measure pairwise geometric superposition crosstalk in R^(d-2)
    gram_bg = u_bg @ u_bg.T
    off_diag = gram_bg[~np.eye(n_concepts, dtype=bool)]
    rms_cosine = float(np.sqrt(np.mean(off_diag ** 2)))
    max_abs_cosine = float(np.max(np.abs(off_diag)))
    theoretical_rms = 1.0 / np.sqrt(d_model - 2)

    # Assign each stored concept key to one of the 16 output tokens (8 keys/token)
    class_ids = np.repeat(np.arange(n_classes), keys_per_class)
    v_frozen = np.zeros((n_concepts, n_classes), dtype=float)
    v_frozen[np.arange(n_concepts), class_ids] = 1.0

    # Complementary binary latent features on orthogonal axes e_A and e_B:
    # Even tokens carry (+1 on e_A, -1 on e_B); odd tokens carry (-1 on e_A, +1 on e_B).
    w_a = np.where(class_ids % 2 == 0, 1.0, -1.0)
    w_b = -w_a

    # Assemble frozen key matrix K* (rows unit-normalized in R^64)
    alpha_bg = np.sqrt(0.60)
    beta_sig = np.sqrt(0.20)  # beta_sig^2 * (1^2 + 1^2) = 0.40 => ||k_i||_2 = 1.0 exactly
    k_frozen = alpha_bg * u_bg + beta_sig * (
        np.outer(w_a, e_a) + np.outer(w_b, e_b)
    )
    key_norm_err = float(np.max(np.abs(np.linalg.norm(k_frozen, axis=1) - 1.0)))
    assert key_norm_err < 1e-14

    k_snapshot = k_frozen.copy()
    v_snapshot = v_frozen.copy()

    def attend_distribution(queries: np.ndarray) -> np.ndarray:
        """Compute softmax(q K*^T / sqrt(d)) V* for batch of queries (B, d)."""
        logits = (queries @ k_frozen.T) / np.sqrt(d_model)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return weights @ v_frozen

    # 1. Unconditioned marginal over 20,000 isotropic random prompts
    n_iso_queries = 20_000
    iso_queries = rng.normal(loc=0.0, scale=4.0, size=(n_iso_queries, d_model))
    p_iso_all = attend_distribution(iso_queries)
    p_marginal = np.mean(p_iso_all, axis=0)

    # 2. Targeted complementary prompts q_A and q_B (orthogonal: q_A . q_B = 0)
    # Choose prompt scale so kappa = gamma * beta_sig / sqrt(d) = 2.0 => tanh(2.0) = 0.9640
    kappa = 2.0
    gamma = kappa * np.sqrt(d_model) / beta_sig
    q_a = (gamma * e_a)[None, :]
    q_b = (gamma * e_b)[None, :]
    p_qa = attend_distribution(q_a)[0]
    p_qb = attend_distribution(q_b)[0]
    p_ab_mix = 0.5 * (p_qa + p_qb)

    # 3. Targeted single-concept prompt aligned with superposed key u_bg[0] (class 0)
    # Strong inverse-temperature scale demonstrates sharp entropy collapse while
    # still revealing geometric crosstalk from M=128 > d=64 quasi-orthogonal keys.
    gamma_feat = 7.5 * np.sqrt(d_model) / alpha_bg
    q_feat = (gamma_feat * u_bg[0])[None, :]
    p_feat = attend_distribution(q_feat)[0]

    # Verify frozen weights W* were NEVER altered
    weights_drift = float(
        np.linalg.norm(k_frozen - k_snapshot) + np.linalg.norm(v_frozen - v_snapshot)
    )
    assert weights_drift == 0.0

    # Quantitative entropy, KL divergence, and contrast metrics
    p_uniform = np.full(n_classes, 1.0 / n_classes)
    h_max = float(np.log2(n_classes))  # 4.0 bits
    h_marginal = shannon_entropy_bits(p_marginal)
    h_qa = shannon_entropy_bits(p_qa)
    h_qb = shannon_entropy_bits(p_qb)
    h_ab_mix = shannon_entropy_bits(p_ab_mix)
    h_feat = shannon_entropy_bits(p_feat)

    kl_marginal = kl_divergence_bits(p_marginal, p_uniform)
    kl_qa = kl_divergence_bits(p_qa, p_uniform)
    kl_qb = kl_divergence_bits(p_qb, p_uniform)
    kl_ab_mix = kl_divergence_bits(p_ab_mix, p_uniform)
    tv_marginal = float(0.5 * np.sum(np.abs(p_marginal - p_uniform)))
    err_ab_mix_flat = float(np.max(np.abs(p_ab_mix - p_uniform)))
    assert err_ab_mix_flat < 1e-15, "Complementary prompt mixture did not cancel flatly!"

    def contrast_vis(p: np.ndarray) -> float:
        return float((np.max(p) - np.min(p)) / (np.max(p) + np.min(p)))

    vis_marginal = contrast_vis(p_marginal)
    vis_qa = contrast_vis(p_qa)
    vis_qb = contrast_vis(p_qb)
    vis_ab_mix = contrast_vis(p_ab_mix)
    assert vis_ab_mix < 1e-14

    # -----------------------------------------------------------------------
    # Part B: Falsifiable Bell-CHSH Boundary (Classical LHV vs Quantum |Phi+>)
    # -----------------------------------------------------------------------
    # Exact Pauli matrices and Bell state |Phi+> = (|00> + |11>)/sqrt(2)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    phi_plus = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho_phi_plus = np.outer(phi_plus, phi_plus.conj())

    def spin_op(angle: float) -> np.ndarray:
        return np.cos(angle) * sigma_z + np.sin(angle) * sigma_x

    # 1-parameter CHSH sweep: alpha_0 = theta, alpha_1 = 3*theta, beta_0 = 2*theta, beta_1 = 0
    # E_QM(alpha, beta) = cos(alpha - beta) => S_QM(theta) = 3*cos(theta) - cos(3*theta)
    thetas = np.linspace(0.0, np.pi, 361)
    s_qm_sweep = np.zeros_like(thetas)
    s_lhv_threshold_sweep = np.zeros_like(thetas)
    s_lhv_softmax_sweep = np.zeros_like(thetas)

    # Pre-sample classical isotropic latent variable lambda in [0, 2*pi) for Monte Carlo LHV
    # Use high-precision deterministic quadrature grid (N_lambda = 4096) for exact LHV expectation
    lam_grid = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    beta_soft = 3.0  # Softmax/logistic inverse temperature for smooth classical attention LHV

    def classical_correlator_threshold(a_ang: float, b_ang: float) -> float:
        a_resp = np.where(np.cos(a_ang - lam_grid) >= 0.0, 1.0, -1.0)
        b_resp = np.where(np.cos(b_ang - lam_grid) >= 0.0, 1.0, -1.0)
        return float(np.mean(a_resp * b_resp))

    def classical_correlator_softmax(a_ang: float, b_ang: float) -> float:
        # Local stochastic binary response via softmax/tanh: E[A|x,lambda] = tanh(beta * cos(a - lambda))
        ea = np.tanh(beta_soft * np.cos(a_ang - lam_grid))
        eb = np.tanh(beta_soft * np.cos(b_ang - lam_grid))
        return float(np.mean(ea * eb))

    for idx, th in enumerate(thetas):
        a0, a1, b0, b1 = th, 3.0 * th, 2.0 * th, 0.0
        # Exact quantum expectation via Tr[(A_x (x) B_y) rho_Phi+]
        op_a0, op_a1 = spin_op(a0), spin_op(a1)
        op_b0, op_b1 = spin_op(b0), spin_op(b1)
        e00 = float(np.real(np.trace(np.kron(op_a0, op_b0) @ rho_phi_plus)))
        e01 = float(np.real(np.trace(np.kron(op_a0, op_b1) @ rho_phi_plus)))
        e10 = float(np.real(np.trace(np.kron(op_a1, op_b0) @ rho_phi_plus)))
        e11 = float(np.real(np.trace(np.kron(op_a1, op_b1) @ rho_phi_plus)))
        s_qm_sweep[idx] = e00 + e01 + e10 - e11

        # Classical LHV threshold model (isotropic hidden vector)
        s_lhv_threshold_sweep[idx] = (
            classical_correlator_threshold(a0, b0)
            + classical_correlator_threshold(a0, b1)
            + classical_correlator_threshold(a1, b0)
            - classical_correlator_threshold(a1, b1)
        )

        # Classical smooth softmax-attention LHV model
        s_lhv_softmax_sweep[idx] = (
            classical_correlator_softmax(a0, b0)
            + classical_correlator_softmax(a0, b1)
            + classical_correlator_softmax(a1, b0)
            - classical_correlator_softmax(a1, b1)
        )

    # Exact verification at optimal Tsirelson angle theta = pi/4
    th_opt = np.pi / 4.0
    op_a0, op_a1 = spin_op(th_opt), spin_op(3.0 * th_opt)
    op_b0, op_b1 = spin_op(2.0 * th_opt), spin_op(0.0)
    s_qm_opt = float(
        np.real(
            np.trace(
                (
                    np.kron(op_a0, op_b0)
                    + np.kron(op_a0, op_b1)
                    + np.kron(op_a1, op_b0)
                    - np.kron(op_a1, op_b1)
                )
                @ rho_phi_plus
            )
        )
    )
    tsirelson_bound = 2.0 * np.sqrt(2.0)
    tsirelson_err = abs(s_qm_opt - tsirelson_bound)
    assert tsirelson_err < 1e-14, "Quantum CHSH did not match Tsirelson bound!"

    # Exhaustive verification over all 16 deterministic vertices of the Bell polytope
    vertex_s_vals = []
    for bits in range(16):
        a0_v = 1.0 if (bits & 1) else -1.0
        a1_v = 1.0 if (bits & 2) else -1.0
        b0_v = 1.0 if (bits & 4) else -1.0
        b1_v = 1.0 if (bits & 8) else -1.0
        vertex_s_vals.append(a0_v * b0_v + a0_v * b1_v + a1_v * b0_v - a1_v * b1_v)
    max_polytope_abs_s = float(np.max(np.abs(vertex_s_vals)))
    assert max_polytope_abs_s == 2.0

    # Monte Carlo audit of 5,000 random classical LHV models probing the polytope boundary:
    # Half use dense Dirichlet(1) + Uniform[-1,1]; half use sparse near-extremal
    # Dirichlet(0.15) + Beta(0.15, 0.15) responses mapped to [-1, 1] to stress-test |S| <= 2.
    n_mc_lhv = 5_000
    n_half = n_mc_lhv // 2
    k_latent = 16
    p_lam_dense = rng.dirichlet(np.ones(k_latent), size=n_half)
    ea0_d = rng.uniform(-1.0, 1.0, size=(n_half, k_latent))
    ea1_d = rng.uniform(-1.0, 1.0, size=(n_half, k_latent))
    eb0_d = rng.uniform(-1.0, 1.0, size=(n_half, k_latent))
    eb1_d = rng.uniform(-1.0, 1.0, size=(n_half, k_latent))

    p_lam_sparse = rng.dirichlet(np.full(k_latent, 0.15), size=n_half)
    ea0_s = 2.0 * rng.beta(0.15, 0.15, size=(n_half, k_latent)) - 1.0
    ea1_s = 2.0 * rng.beta(0.15, 0.15, size=(n_half, k_latent)) - 1.0
    eb0_s = 2.0 * rng.beta(0.15, 0.15, size=(n_half, k_latent)) - 1.0
    eb1_s = 2.0 * rng.beta(0.15, 0.15, size=(n_half, k_latent)) - 1.0

    p_lam = np.vstack([p_lam_dense, p_lam_sparse])
    ea0 = np.vstack([ea0_d, ea0_s])
    ea1 = np.vstack([ea1_d, ea1_s])
    eb0 = np.vstack([eb0_d, eb0_s])
    eb1 = np.vstack([eb1_d, eb1_s])
    mc_s_lhv = np.sum(
        p_lam * (ea0 * eb0 + ea0 * eb1 + ea1 * eb0 - ea1 * eb1), axis=1
    )
    max_mc_lhv_abs_s = float(np.max(np.abs(mc_s_lhv)))
    max_threshold_abs_s = float(np.max(np.abs(s_lhv_threshold_sweep)))
    max_softmax_abs_s = float(np.max(np.abs(s_lhv_softmax_sweep)))

    assert max_mc_lhv_abs_s <= 2.0 + 1e-12
    assert max_threshold_abs_s <= 2.0 + 1e-12
    assert max_softmax_abs_s <= 2.0 + 1e-12

    # Werner state hierarchy rho_W(v) = v |Phi+><Phi+| + (1-v) I_4 / 4
    vis_werner = np.linspace(0.0, 1.0, 201)
    s_werner = tsirelson_bound * vis_werner

    # --- Plot Figure 2 ---
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), constrained_layout=True)
    fig.suptitle(
        "Experiment 2 — Frozen Latent-Space Post-Selection (Prompt as Detector) vs. Quantum Bell-CHSH Boundary",
        fontsize=12.6,
        fontweight="bold",
        y=1.02,
    )

    # Panel (a): Part A — Marginal vs Complementary Prompt Post-Selection
    ax = axes[0, 0]
    tokens = np.arange(n_classes)
    w_bar = 0.27
    ax.bar(
        tokens - w_bar,
        p_marginal,
        width=w_bar,
        color=PALETTE["light_gray"],
        edgecolor=PALETTE["slate"],
        linewidth=0.6,
        label=f"Isotropic Marginal $P_{{\\mathrm{{marg}}}}$ ($H={h_marginal:.3f}$ bits)",
    )
    ax.bar(
        tokens,
        p_qa,
        width=w_bar,
        color=PALETTE["navy"],
        alpha=0.92,
        label=f"Prompt $q_A$ ($H={h_qa:.3f}$ bits, $V={vis_qa:.3f}$)",
    )
    ax.bar(
        tokens + w_bar,
        p_qb,
        width=w_bar,
        color=PALETTE["coral"],
        alpha=0.90,
        label=f"Prompt $q_B$ ($H={h_qb:.3f}$ bits, $V={vis_qb:.3f}$)",
    )
    ax.axhline(
        1.0 / n_classes,
        color="black",
        ls=":",
        lw=1.1,
        label=r"Exact Uniform $1/16$ ($= \frac{1}{2}[P(\cdot|q_A)+P(\cdot|q_B)]$)",
    )
    ax.set_title(r"(a) Part A: Frozen $W^*$ ($d=64, M=128$) — Complementary Prompt Post-Selection")
    ax.set_xlabel("Output Semantic Token Index $c$")
    ax.set_ylabel("Output Probability $P(c)$")
    ax.set_xticks(tokens)
    ax.set_ylim(0.0, np.max(p_qa) * 1.35)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=8.0)

    # Panel (b): Entropy Collapse & Superposition Diagnostics
    ax = axes[0, 1]
    regimes = [
        "Uniform\nCeiling",
        "Isotropic\nMarginal",
        r"$\frac{1}{2}(q_A+q_B)$" + "\nMixture",
        "Prompt\n$q_A$",
        "Prompt\n$q_B$",
        "Single-Key\n$q_{\\mathrm{feat}}$",
    ]
    entropies = [h_max, h_marginal, h_ab_mix, h_qa, h_qb, h_feat]
    ent_colors = [
        PALETTE["light_gray"],
        PALETTE["slate"],
        PALETTE["teal"],
        PALETTE["navy"],
        PALETTE["coral"],
        PALETTE["purple"],
    ]
    x_e = np.arange(len(regimes))
    bars = ax.bar(x_e, entropies, color=ent_colors, width=0.58, edgecolor="#1E293B", linewidth=0.7)
    for bar, h_val in zip(bars, entropies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h_val + 0.08,
            f"{h_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
        )
    ax.axhline(h_max, color="#475569", ls="--", lw=1.0, alpha=0.7)
    ax.set_xticks(x_e)
    ax.set_xticklabels(regimes, fontsize=8.5)
    ax.set_ylim(0.0, 5.15)
    ax.set_ylabel("Shannon Entropy $H(P)$ (bits)")
    ax.set_title("(b) Part A: Entropy Reduction Without Weight Mutation ($\\|\\Delta W^*\\|_F = 0$)")
    ax.grid(True, axis="y")

    geom_box = (
        "Geometric Superposition ($\\mathbb{R}^{64}$):\n"
        f"• Pack ratio M/d = {n_concepts}/{d_model} = {n_concepts/d_model:.1f}x\n"
        f"• RMS |cos| = {rms_cosine:.4f} (th. {theoretical_rms:.4f})\n"
        f"• max |0.5(q_A+q_B) - U| = {err_ab_mix_flat:.2e}\n"
        f"• TV(P_marg, U) = {tv_marginal:.4f}"
    )
    ax.text(
        0.97,
        0.96,
        geom_box,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["bg_tint"], edgecolor="#94A3B8", lw=0.85),
    )

    # Panel (c): Part B — Bell-CHSH Angle Sweep
    ax = axes[1, 0]
    theta_deg = np.degrees(thetas)
    ax.axhspan(2.0, tsirelson_bound, color=PALETTE["coral"], alpha=0.10, label=r"Quantum Violation Zone ($2 < |S| \leq 2\sqrt{2}$)")
    ax.axhspan(-tsirelson_bound, -2.0, color=PALETTE["coral"], alpha=0.10)
    ax.axhline(2.0, color="#1E293B", ls="--", lw=1.3, label=r"Classical Bell-CHSH Bound ($|S| = 2$)")
    ax.axhline(-2.0, color="#1E293B", ls="--", lw=1.3)
    ax.axhline(tsirelson_bound, color=PALETTE["coral"], ls=":", lw=1.3, label=r"Tsirelson Bound ($2\sqrt{2} \approx 2.8284$)")
    ax.axhline(-tsirelson_bound, color=PALETTE["coral"], ls=":", lw=1.3)

    ax.plot(theta_deg, s_qm_sweep, color=PALETTE["navy"], lw=2.4, label=r"Quantum $|\Phi^+\rangle$: $S_{\mathrm{QM}}(\theta)$")
    ax.plot(theta_deg, s_lhv_threshold_sweep, color=PALETTE["teal"], lw=2.0, ls="-.", label=r"Optimal Isotropic LHV ($\max|S|=2.000$)")
    ax.plot(theta_deg, s_lhv_softmax_sweep, color=PALETTE["amber"], lw=1.8, ls="--", label=f"Softmax-Attention LHV ($\\max|S|={max_softmax_abs_s:.3f}$)")

    ax.scatter([45.0], [s_qm_opt], color=PALETTE["coral"], s=60, zorder=5, edgecolor="white", linewidth=1.0)
    ax.annotate(
        f"$S(45^\\circ) = {s_qm_opt:.4f}$",
        xy=(45.0, s_qm_opt),
        xytext=(74.0, 2.58),
        fontsize=8.6,
        fontweight="bold",
        color=PALETTE["coral"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=1.1),
    )

    ax.set_title(r"(c) Part B: Bell-CHSH $S(\theta)$ — Classical Post-Selection vs. Entanglement")
    ax.set_xlabel(r"Measurement Angle Parameter $\theta$ (degrees)")
    ax.set_ylabel("CHSH Parameter $S$")
    ax.set_xlim(0, 180)
    ax.set_ylim(-3.15, 3.25)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7.8)

    # Panel (d): Werner State Hierarchy & Random LHV Polytope Audit
    ax = axes[1, 1]
    # Shaded physical regimes for Werner states rho_W(v)
    ax.axvspan(0.0, 1.0 / 3.0, color=PALETTE["light_gray"], alpha=0.45)
    ax.axvspan(1.0 / 3.0, 0.5, color="#DBEAFE", alpha=0.55)
    ax.axvspan(0.5, 1.0 / np.sqrt(2.0), color="#FEF3C7", alpha=0.60)
    ax.axvspan(1.0 / np.sqrt(2.0), 1.0, color="#FEE2E2", alpha=0.60)

    ax.plot(vis_werner, s_werner, color=PALETTE["navy"], lw=2.5, label=r"Werner State $S_{\max}(v) = 2\sqrt{2}\,v$")
    ax.axhline(2.0, color="#1E293B", ls="--", lw=1.3, label=r"Classical LHV Ceiling ($S=2$)")
    ax.axhline(tsirelson_bound, color=PALETTE["coral"], ls=":", lw=1.3, label=r"Tsirelson Limit ($2\sqrt{2}$)")

    ax.axvline(1.0 / 3.0, color=PALETTE["slate"], ls=":", lw=1.0)
    ax.axvline(0.5, color=PALETTE["amber"], ls=":", lw=1.1)
    ax.axvline(1.0 / np.sqrt(2.0), color=PALETTE["coral"], ls=":", lw=1.1)

    ax.text(0.165, 2.74, "Separable\n(Classical)\n$v \\leq 1/3$", ha="center", va="top", fontsize=7.6, color="#334155", fontweight="bold")
    ax.text(0.417, 2.74, "Entangled\nNon-\nSteerable", ha="center", va="top", fontsize=7.4, color="#1E40AF", fontweight="bold")
    ax.text(0.604, 2.74, "Steerable\n(DCQE)\n$v > 1/2$", ha="center", va="top", fontsize=7.6, color="#92400E", fontweight="bold")
    ax.text(0.854, 1.20, "Bell-CHSH\nViolating\n$v > 1/\\sqrt{2}$", ha="center", va="top", fontsize=8.0, color="#991B1B", fontweight="bold")

    lhv_box = (
        "Classical LHV Audit:\n"
        f"• All 16 polytope vertices: max|S| = {max_polytope_abs_s:.4f}\n"
        f"• 5,000 random LHV models: max|S| = {max_mc_lhv_abs_s:.4f}\n"
        f"• Softmax-attn LHV: max|S| = {max_softmax_abs_s:.4f}\n"
        f"• Quantum |Φ+>: S_max = {s_qm_opt:.4f}"
    )
    ax.text(
        0.03,
        0.04,
        lhv_box,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#94A3B8", lw=0.85, alpha=0.95),
    )

    ax.set_title(r"(d) Part B: Werner Hierarchy — Bell $\subset$ Steering $\subset$ Entangled")
    ax.set_xlabel(r"Werner State Visibility $v$")
    ax.set_ylabel(r"Maximum Achievable CHSH $|S|$")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 3.22)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.8, bbox_to_anchor=(0.0, 0.68))

    out_path = FIGURES_DIR / "fig2_latent_postselection_vs_chsh.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "latent_dim_d": d_model,
        "n_superposed_concepts_M": n_concepts,
        "n_output_classes": n_classes,
        "rms_pairwise_cosine": rms_cosine,
        "theoretical_rms_cosine": float(theoretical_rms),
        "max_abs_pairwise_cosine": max_abs_cosine,
        "weights_mutation_frobenius": weights_drift,
        "entropy_uniform_max_bits": h_max,
        "entropy_isotropic_marginal_bits": h_marginal,
        "entropy_prompt_qA_bits": h_qa,
        "entropy_prompt_qB_bits": h_qb,
        "entropy_ab_mixture_bits": h_ab_mix,
        "entropy_single_feature_prompt_bits": h_feat,
        "kl_marginal_vs_uniform_bits": kl_marginal,
        "kl_prompt_qA_vs_uniform_bits": kl_qa,
        "kl_prompt_qB_vs_uniform_bits": kl_qb,
        "kl_ab_mix_vs_uniform_bits": kl_ab_mix,
        "tv_marginal_vs_uniform": tv_marginal,
        "err_ab_mix_cancellation": err_ab_mix_flat,
        "contrast_marginal": vis_marginal,
        "contrast_prompt_qA": vis_qa,
        "contrast_prompt_qB": vis_qb,
        "contrast_ab_mixture": vis_ab_mix,
        "chsh_polytope_max_classical": max_polytope_abs_s,
        "chsh_mc_5000_lhv_max": max_mc_lhv_abs_s,
        "chsh_isotropic_threshold_lhv_max": max_threshold_abs_s,
        "chsh_softmax_attention_lhv_max": max_softmax_abs_s,
        "chsh_quantum_phi_plus_opt": s_qm_opt,
        "chsh_tsirelson_bound": float(tsirelson_bound),
        "chsh_tsirelson_err": float(tsirelson_err),
        "figure_path": str(out_path),
    }


# ===========================================================================
# EXPERIMENT 3: Data Dredging Overfitting & Berkson Collider Bias
# ===========================================================================
def expected_max_normal_approx(m: int) -> float:
    """Bailey & Lopez de Prado (2014) extreme-value approximation for E[max_{1..M} Z_i]."""
    if m <= 1:
        return 0.0
    euler_gamma = 0.5772156649015329
    return float(
        (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / m)
        + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (m * math.e))
    )


def berkson_exact_correlation(tau: float) -> float:
    """Exact conditional Pearson correlation rho(X, Y | X + Y > tau) for X, Y ~ i.i.d. N(0, 1).

    Using orthogonal rotation U = (X+Y)/sqrt(2), V = (X-Y)/sqrt(2), conditioning on
    X + Y > tau is equivalent to U > c = tau / sqrt(2) with V independent of U.
    Then Var(U | U > c) = 1 + c*lambda(c) - lambda(c)^2, where lambda(c) = phi(c)/(1-Phi(c))
    is the inverse Mills ratio, and rho = (Var(U|U>c) - 1) / (Var(U|U>c) + 1).
    """
    c = tau / np.sqrt(2.0)
    sf = float(stats.norm.sf(c))
    pdf = float(stats.norm.pdf(c))
    inv_mills = pdf / sf
    var_u_cond = 1.0 + c * inv_mills - inv_mills ** 2
    return float((var_u_cond - 1.0) / (var_u_cond + 1.0))


def run_experiment_3(rng: np.random.Generator) -> dict[str, Any]:
    """Quantify post-selection overfitting on pure white noise and Berkson collider bias."""
    # -----------------------------------------------------------------------
    # Part A: Data Dredging / Backtest Overfitting on Pure White Noise
    # -----------------------------------------------------------------------
    t_is = 252      # In-Sample length (1 trading year, frozen past)
    t_oos = 252     # Out-of-Sample length (1 trading year, future)
    daily_sigma = 0.01
    m_grid = [1, 10, 100, 1000, 10000]
    n_trials = 60   # Independent Monte Carlo repetitions per M

    # Storage for MC statistics across M
    sr_is_mean = []
    sr_is_std = []
    sr_oos_mean = []
    sr_oos_std = []
    naive_p1_median = []
    naive_p2_median = []
    bonf_p1_median = []
    theory_sr_max = [expected_max_normal_approx(m) for m in m_grid]

    # Also save a canonical realization at M = 10,000 for equity curve visualization
    canonical_winner_is = None
    canonical_winner_oos = None
    canonical_peers_full = None
    canonical_sr_is = 0.0
    canonical_sr_oos = 0.0
    canonical_p1 = 0.0
    canonical_p2 = 0.0

    # To keep runtime fast and exact, simulate trials of M_max = 10,000 and slice prefixes
    # so nested pools M in {1, 10, 100, 1000, 10000} share identical noise draws per trial!
    m_max = max(m_grid)
    sr_is_trials = {m: np.zeros(n_trials) for m in m_grid}
    sr_oos_trials = {m: np.zeros(n_trials) for m in m_grid}
    p1_trials = {m: np.zeros(n_trials) for m in m_grid}
    p2_trials = {m: np.zeros(n_trials) for m in m_grid}

    # Store per-trial M=10,000 winners to pick the median-representative trajectory
    trial_winners_is = []
    trial_winners_oos = []
    trial_peers_full = None

    ann_factor = np.sqrt(252.0)
    df_t = t_is - 1

    for trial in range(n_trials):
        # Pure i.i.d. N(0, sigma^2) daily log-returns: zero true signal, zero drift
        r_is = rng.normal(0.0, daily_sigma, size=(m_max, t_is))
        r_oos = rng.normal(0.0, daily_sigma, size=(m_max, t_oos))

        mean_is = np.mean(r_is, axis=1)
        std_is = np.std(r_is, axis=1, ddof=1)
        sr_is_all = ann_factor * (mean_is / std_is)

        mean_oos = np.mean(r_oos, axis=1)
        std_oos = np.std(r_oos, axis=1, ddof=1)
        sr_oos_all = ann_factor * (mean_oos / std_oos)

        for m in m_grid:
            best_idx = int(np.argmax(sr_is_all[:m]))
            best_sr_is = float(sr_is_all[best_idx])
            best_sr_oos = float(sr_oos_all[best_idx])
            # Since T_IS = 252 and ann_factor = sqrt(252), t_stat == best_sr_is exactly
            t_stat = best_sr_is * np.sqrt(t_is) / ann_factor
            p_one = float(stats.t.sf(t_stat, df=df_t))
            p_two = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))

            sr_is_trials[m][trial] = best_sr_is
            sr_oos_trials[m][trial] = best_sr_oos
            p1_trials[m][trial] = p_one
            p2_trials[m][trial] = p_two

        best_10k = int(np.argmax(sr_is_all))
        trial_winners_is.append(r_is[best_10k].copy())
        trial_winners_oos.append(r_oos[best_10k].copy())
        if trial == 0:
            trial_peers_full = np.concatenate([r_is[:30], r_oos[:30]], axis=1)

    # Select canonical visualization trial closest to the median IS Sharpe at M = 10,000
    med_sr_10k = float(np.median(sr_is_trials[m_max]))
    can_trial = int(np.argmin(np.abs(sr_is_trials[m_max] - med_sr_10k)))
    canonical_winner_is = trial_winners_is[can_trial]
    canonical_winner_oos = trial_winners_oos[can_trial]
    canonical_peers_full = trial_peers_full
    canonical_sr_is = float(sr_is_trials[m_max][can_trial])
    canonical_sr_oos = float(sr_oos_trials[m_max][can_trial])
    canonical_p1 = float(p1_trials[m_max][can_trial])
    canonical_p2 = float(p2_trials[m_max][can_trial])

    summary_by_m = []
    for m, th_val in zip(m_grid, theory_sr_max):
        m_is = float(np.mean(sr_is_trials[m]))
        s_is = float(np.std(sr_is_trials[m], ddof=1))
        m_oos = float(np.mean(sr_oos_trials[m]))
        s_oos = float(np.std(sr_oos_trials[m], ddof=1))
        med_p1 = float(np.median(p1_trials[m]))
        med_p2 = float(np.median(p2_trials[m]))
        med_bonf = float(min(1.0, m * med_p1))

        sr_is_mean.append(m_is)
        sr_is_std.append(s_is)
        sr_oos_mean.append(m_oos)
        sr_oos_std.append(s_oos)
        naive_p1_median.append(med_p1)
        naive_p2_median.append(med_p2)
        bonf_p1_median.append(med_bonf)

        summary_by_m.append(
            {
                "M": m,
                "sr_is_mean": m_is,
                "sr_is_std": s_is,
                "sr_is_theory_blp": th_val,
                "sr_oos_mean": m_oos,
                "sr_oos_std": s_oos,
                "naive_p_one_sided_median": med_p1,
                "naive_p_two_sided_median": med_p2,
                "bonferroni_p_one_sided_median": med_bonf,
            }
        )

    # Sanity assertions on overfitting mirage vs OOS collapse
    assert sr_is_mean[-1] > 3.5, "Expected M=10,000 IS Sharpe > 3.5"
    assert abs(sr_oos_mean[-1]) < 0.35, "Expected M=10,000 OOS Sharpe near 0"
    assert naive_p1_median[-1] < 1e-4, "Expected M=10,000 naive one-sided p < 1e-4"

    # -----------------------------------------------------------------------
    # Part B: Berkson Collider Bias (X _|_ Y, post-select on Z = X + Y > tau)
    # -----------------------------------------------------------------------
    n_col = 50_000
    x_col = rng.normal(0.0, 1.0, size=n_col)
    y_col = rng.normal(0.0, 1.0, size=n_col)
    z_col = x_col + y_col

    r_marginal, p_marginal_col = stats.pearsonr(x_col, y_col)
    tau_highlight = 2.0
    mask_hl = z_col > tau_highlight
    r_hl, p_hl = stats.pearsonr(x_col[mask_hl], y_col[mask_hl])
    r_hl_exact = berkson_exact_correlation(tau_highlight)
    frac_hl = float(np.mean(mask_hl))

    # Threshold sweep tau in [-1.0, 3.0]
    tau_sweep = np.linspace(-1.0, 3.0, 21)
    r_emp_sweep = []
    r_exact_sweep = []
    for tau in tau_sweep:
        m_tau = z_col > tau
        r_e, _ = stats.pearsonr(x_col[m_tau], y_col[m_tau])
        r_emp_sweep.append(float(r_e))
        r_exact_sweep.append(berkson_exact_correlation(float(tau)))

    max_berkson_err = float(np.max(np.abs(np.array(r_emp_sweep) - np.array(r_exact_sweep))))
    assert abs(r_marginal) < 0.02, "Marginal X, Y must be uncorrelated!"
    assert r_hl < -0.65, "Expected strong spurious anti-correlation at tau=2.0!"
    assert abs(r_hl - r_hl_exact) < 0.03, "Empirical Berkson correlation should match theory!"

    # --- Plot Figure 3 ---
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), constrained_layout=True)
    fig.suptitle(
        "Experiment 3 — Post-Selection Illusion: Data Dredging Overfitting & Berkson Collider Bias",
        fontsize=12.6,
        fontweight="bold",
        y=1.02,
    )

    # Panel (a): Data Dredging Scaling IS vs OOS
    ax = axes[0, 0]
    m_arr = np.array(m_grid, dtype=float)
    is_m = np.array(sr_is_mean)
    is_s = np.array(sr_is_std)
    oos_m = np.array(sr_oos_mean)
    oos_s = np.array(sr_oos_std)

    ax.plot(
        m_arr,
        theory_sr_max,
        color="#1E293B",
        ls="--",
        lw=1.4,
        label=r"Bailey–López de Prado Law $\mathbb{E}[\max \widehat{\mathrm{SR}}]$",
    )
    ax.errorbar(
        m_arr,
        is_m,
        yerr=is_s,
        fmt="o-",
        color=PALETTE["coral"],
        lw=2.2,
        capsize=4,
        label=f"Post-Selected In-Sample Sharpe ($R={n_trials}$ trials)",
    )
    ax.errorbar(
        m_arr,
        oos_m,
        yerr=oos_s,
        fmt="s-",
        color=PALETTE["navy"],
        lw=2.2,
        capsize=4,
        label="Same Winner Out-of-Sample Sharpe (Future)",
    )
    ax.axhline(0.0, color=PALETTE["slate"], ls=":", lw=1.1)
    ax.axhline(2.0, color=PALETTE["amber"], ls=":", lw=1.1, label="Institutional Threshold ($\\mathrm{SR}=2.0$)")

    ax.annotate(
        f"$M=10^4$: $\\mathrm{{SR}}_{{\\mathrm{{IS}}}}={is_m[-1]:.2f}$\n(naive $p_{{1}}={naive_p1_median[-1]:.1e}$)",
        xy=(10000, is_m[-1]),
        xytext=(310, 3.88),
        fontsize=8.3,
        color=PALETTE["coral"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=1.1),
    )

    ax.set_xscale("log")
    ax.set_xticks(m_grid)
    ax.set_xticklabels([f"$10^{int(np.log10(m))}$" if m >= 10 else "1" for m in m_grid])
    ax.set_title("(a) Data Dredging on Pure White Noise ($T_{\\mathrm{IS}}=T_{\\mathrm{OOS}}=252$)")
    ax.set_xlabel("Number of Candidate Strategies $M$ (log scale)")
    ax.set_ylabel("Annualized Sharpe Ratio")
    ax.set_ylim(-1.4, 4.75)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.9)

    # Panel (b): Equity Curve Mirage — All R=60 M=10,000 Winners & Ensemble Mean
    ax = axes[0, 1]
    t_axis = np.arange(t_is + t_oos + 1)
    all_winners_full = np.concatenate(
        [np.array(trial_winners_is), np.array(trial_winners_oos)], axis=1
    )  # (n_trials, t_is + t_oos)
    all_cum_winners = np.concatenate(
        [np.zeros((n_trials, 1)), np.cumsum(all_winners_full, axis=1)], axis=1
    ) * 100.0
    mean_cum_winner = np.mean(all_cum_winners, axis=0)
    std_cum_winner = np.std(all_cum_winners, axis=0, ddof=1)

    for tr_i in range(min(25, n_trials)):
        ax.plot(t_axis[: t_is + 1], all_cum_winners[tr_i, : t_is + 1], color=PALETTE["coral"], lw=0.65, alpha=0.22)
        ax.plot(t_axis[t_is:], all_cum_winners[tr_i, t_is:], color=PALETTE["navy"], lw=0.65, alpha=0.22)

    ax.fill_between(
        t_axis[t_is:],
        mean_cum_winner[t_is:] - std_cum_winner[t_is:],
        mean_cum_winner[t_is:] + std_cum_winner[t_is:],
        color=PALETTE["navy"],
        alpha=0.12,
        label=r"OOS Brownian Fan ($\pm 1\sigma$ across 60 winners)",
    )
    ax.plot(
        t_axis[: t_is + 1],
        mean_cum_winner[: t_is + 1],
        color=PALETTE["coral"],
        lw=2.8,
        label=f"Mean Winner IS ($\\overline{{\\mathrm{{SR}}}}_{{\\mathrm{{IS}}}}={sr_is_mean[-1]:.2f},\\;\\mathrm{{med}}\\;p_1={naive_p1_median[-1]:.1e}$)",
    )
    ax.plot(
        t_axis[t_is:],
        mean_cum_winner[t_is:],
        color=PALETTE["navy"],
        lw=2.8,
        label=f"Mean Winner OOS ($\\overline{{\\mathrm{{SR}}}}_{{\\mathrm{{OOS}}}}={sr_oos_mean[-1]:+.2f}$ — Flat Drift)",
    )
    ax.axvline(t_is, color="#0F172A", ls="--", lw=1.4)
    ax.axvspan(t_is, t_is + t_oos, color="#F1F5F9", alpha=0.45)

    y_lo = min(-5.0, float(np.percentile(all_cum_winners[:, -1], 2)) - 8.0)
    y_hi = max(float(np.max(all_cum_winners[:25])) + 24.0, float(np.max(mean_cum_winner)) + 48.0)
    ax.text(
        t_is * 0.5,
        y_hi * 0.96,
        "Frozen Past (In-Sample)\nPost-Selection Mirage",
        ha="center",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color=PALETTE["coral"],
    )
    ax.text(
        t_is + t_oos * 0.5,
        y_hi * 0.96,
        "Unseen Future (Out-of-Sample)\nZero Expected Drift",
        ha="center",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    ax.set_title(r"(b) $M=10^4$ Post-Selected Winners ($R=60$ Trials): Sharp Kink at Boundary")
    ax.set_xlabel("Time Step $t$ (Trading Days)")
    ax.set_ylabel("Cumulative Log-Return (%)")
    ax.set_xlim(0, t_is + t_oos)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7.8)

    # Panel (c): Berkson Collider Bias Scatter Plot
    ax = axes[1, 0]
    # Subsample background scatter for crisp visual density while computing stats on all 50,000
    sub_idx = rng.choice(n_col, size=6000, replace=False)
    x_sub = x_col[sub_idx]
    y_sub = y_col[sub_idx]
    m_sub = (x_sub + y_sub) > tau_highlight

    ax.scatter(
        x_sub[~m_sub],
        y_sub[~m_sub],
        s=7,
        color=PALETTE["light_gray"],
        alpha=0.55,
        edgecolors="none",
        label=f"Unselected ($Z \\leq {tau_highlight:.1f}$); All $N=50k$: $r={r_marginal:+.4f}$",
    )
    ax.scatter(
        x_sub[m_sub],
        y_sub[m_sub],
        s=11,
        color=PALETTE["coral"],
        alpha=0.70,
        edgecolors="none",
        label=f"Post-Selected ($Z > {tau_highlight:.1f}$, {100*frac_hl:.1f}%): $r={r_hl:+.4f}$",
    )

    # Selection boundary X + Y = tau_highlight
    x_line = np.linspace(-2.5, 4.2, 200)
    ax.plot(x_line, tau_highlight - x_line, color="#0F172A", ls="--", lw=1.4, label=f"Collider Cut $X + Y = {tau_highlight:.1f}$")

    # OLS regression line on post-selected subset
    slope, intercept = np.polyfit(x_col[mask_hl], y_col[mask_hl], 1)
    x_reg = np.linspace(np.percentile(x_col[mask_hl], 1), np.percentile(x_col[mask_hl], 99), 100)
    ax.plot(x_reg, slope * x_reg + intercept, color=PALETTE["navy"], lw=2.2, label=f"Spurious Fit ($\\text{{slope}}={slope:.2f}$)")

    ax.set_title(r"(c) Berkson Collider Bias ($X \perp Y \sim \mathcal{N}(0,1)$, $Z=X+Y$)")
    ax.set_xlabel("Independent Cause $X$")
    ax.set_ylabel("Independent Cause $Y$")
    ax.set_xlim(-3.8, 4.2)
    ax.set_ylim(-3.8, 4.2)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=7.9)

    # Panel (d): Berkson Spurious Correlation Sweep vs Exact Inverse-Mills Law
    ax = axes[1, 1]
    ax.plot(
        tau_sweep,
        r_exact_sweep,
        color=PALETTE["navy"],
        lw=2.4,
        label=r"Exact Theory $\rho(\tau) = \frac{v_U(\tau/\sqrt{2}) - 1}{v_U(\tau/\sqrt{2}) + 1}$",
    )
    ax.scatter(
        tau_sweep,
        r_emp_sweep,
        color=PALETTE["coral"],
        s=36,
        zorder=4,
        edgecolor="white",
        linewidth=0.6,
        label=f"Empirical Monte Carlo ($N={n_col:,}$)",
    )
    ax.axhline(0.0, color=PALETTE["slate"], ls=":", lw=1.1, label="True Unconditioned Correlation ($\\rho = 0$)")
    ax.scatter([tau_highlight], [r_hl], color=PALETTE["teal"], s=85, zorder=5, marker="D", edgecolor="#0F172A", linewidth=0.8)
    ax.annotate(
        f"$\\tau={tau_highlight:.1f}$:\n$r={r_hl:.4f}$ (th. {r_hl_exact:.4f})",
        xy=(tau_highlight, r_hl),
        xytext=(0.55, -0.30),
        fontsize=8.4,
        fontweight="bold",
        color=PALETTE["teal"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["teal"], lw=1.2),
    )

    ax.set_title(r"(d) Amplification of Spurious Correlation Under Stricter Post-Selection $\tau$")
    ax.set_xlabel(r"Collider Selection Threshold $\tau$ ($Z = X + Y > \tau$)")
    ax.set_ylabel(r"Conditional Correlation $\rho(X, Y \mid Z > \tau)$")
    ax.set_xlim(-1.1, 3.1)
    ax.set_ylim(-0.92, 0.14)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=8.0)

    out_path = FIGURES_DIR / "fig3_postselection_illusion_overfitting.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "T_in_sample": t_is,
        "T_out_of_sample": t_oos,
        "n_mc_trials": n_trials,
        "data_dredging_by_M": summary_by_m,
        "canonical_10k_sr_is": canonical_sr_is,
        "canonical_10k_sr_oos": canonical_sr_oos,
        "canonical_10k_naive_p_one_sided": canonical_p1,
        "canonical_10k_naive_p_two_sided": canonical_p2,
        "berkson_N": n_col,
        "berkson_r_marginal": float(r_marginal),
        "berkson_p_marginal": float(p_marginal_col),
        "berkson_tau_highlight": tau_highlight,
        "berkson_selected_fraction": frac_hl,
        "berkson_r_cond_empirical": float(r_hl),
        "berkson_r_cond_exact": float(r_hl_exact),
        "berkson_p_cond": float(p_hl),
        "berkson_r_tau_0_exact": berkson_exact_correlation(0.0),
        "berkson_r_tau_2_5_exact": berkson_exact_correlation(2.5),
        "berkson_max_sweep_abs_err": max_berkson_err,
        "figure_path": str(out_path),
    }


# ===========================================================================
# Main entry point
# ===========================================================================
def main() -> None:
    t0 = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    # Dedicated deterministic child generators for independent reproducibility
    ss = np.random.SeedSequence(GLOBAL_SEED)
    rng1, rng2, rng3 = [np.random.default_rng(s) for s in ss.spawn(3)]

    print("=" * 78)
    print("DCQE & AI PhD THESIS — NUMERICAL VERIFICATION SUITE")
    print(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python   : {platform.python_version()} | NumPy {np.__version__} | Matplotlib {matplotlib.__version__}")
    print(f"Seed     : {GLOBAL_SEED}")
    print("=" * 78)

    res1 = run_experiment_1(rng1)
    print("\n[EXPERIMENT 1] Delayed-Choice Quantum Eraser (Kim et al., 2000)")
    print(f"  Fringe period Delta_x         = {res1['fringe_period_mm']:.4f} mm (central lobe half-width = {res1['central_lobe_half_mm']:.4f} mm)")
    print(f"  Visibility V(R01)             = {res1['visibility_R01']:.12f}")
    print(f"  Visibility V(R02)             = {res1['visibility_R02']:.12f}")
    print(f"  Visibility V(R03), V(R04)     = {res1['visibility_R03']:.3e}, {res1['visibility_R04']:.3e}")
    print(f"  Visibility V(R01 + R02)       = {res1['visibility_R01_plus_R02']:.3e}")
    print(f"  Visibility V(R0 total)        = {res1['visibility_R0_total']:.3e}")
    print(f"  POVM completeness err         = {res1['err_povm_completeness']:.3e}")
    print(f"  Steering sum err ||Ssig-rhoS||= {res1['err_steering_sum']:.3e}")
    print(f"  Max |R01+R02 - 0.5|A|^2|      = {res1['err_R01_R02_cancellation']:.3e}")
    print(f"  Max |R0 - Tr_S[M_S rho_S]|    = {res1['err_R0_no_signaling']:.3e}")
    print(f"  Max err (100 random CPTP)     = {res1['max_err_100_random_cptp']:.3e}")
    print(f"  MC Clicks (N={res1['mc_photons']:,})         = {res1['mc_counts']}")
    print(f"  Saved figure                  -> {res1['figure_path']}")

    res2 = run_experiment_2(rng2)
    print("\n[EXPERIMENT 2] Frozen Latent-Space Post-Selection vs. Bell-CHSH Boundary")
    print(
        f"  Superposition geometry        = M={res2['n_superposed_concepts_M']} in R^{res2['latent_dim_d']} "
        f"(RMS|cos|={res2['rms_pairwise_cosine']:.4f}, th={res2['theoretical_rms_cosine']:.4f}, max|cos|={res2['max_abs_pairwise_cosine']:.4f})"
    )
    print(f"  Frozen weight mutation ||dW*||= {res2['weights_mutation_frobenius']:.1f}")
    print(f"  H(Uniform ceiling)            = {res2['entropy_uniform_max_bits']:.4f} bits")
    print(f"  H(Isotropic marginal)         = {res2['entropy_isotropic_marginal_bits']:.4f} bits (TV={res2['tv_marginal_vs_uniform']:.5f}, KL={res2['kl_marginal_vs_uniform_bits']:.3e} bits)")
    print(f"  H(Prompt qA), H(Prompt qB)    = {res2['entropy_prompt_qA_bits']:.4f}, {res2['entropy_prompt_qB_bits']:.4f} bits (V={res2['contrast_prompt_qA']:.4f}, KL={res2['kl_prompt_qA_vs_uniform_bits']:.4f} bits)")
    print(f"  H(0.5*(qA+qB))                = {res2['entropy_ab_mixture_bits']:.4f} bits (max flat err={res2['err_ab_mix_cancellation']:.3e})")
    print(f"  H(Single-concept q_feat)      = {res2['entropy_single_feature_prompt_bits']:.4f} bits")
    print(f"  Bell-CHSH 16 vertices max|S|  = {res2['chsh_polytope_max_classical']:.6f}")
    print(f"  Bell-CHSH 5000 MC LHV max|S|  = {res2['chsh_mc_5000_lhv_max']:.6f}")
    print(f"  Bell-CHSH Isotropic LHV max|S|= {res2['chsh_isotropic_threshold_lhv_max']:.6f}")
    print(f"  Bell-CHSH Softmax LHV max|S|  = {res2['chsh_softmax_attention_lhv_max']:.6f}")
    print(f"  Bell-CHSH Quantum |Phi+> opt  = {res2['chsh_quantum_phi_plus_opt']:.12f} (Tsirelson err={res2['chsh_tsirelson_err']:.3e})")
    print(f"  Saved figure                  -> {res2['figure_path']}")

    res3 = run_experiment_3(rng3)
    print("\n[EXPERIMENT 3] Data Dredging Overfitting & Berkson Collider Bias")
    print("  M        | IS Sharpe (mean +/- sd) | Theory BLP | OOS Sharpe (mean +/- sd) | Med p1 (naive) | Med p2 (naive) | Med p1 (Bonf)")
    for row in res3["data_dredging_by_M"]:
        print(
            f"  {row['M']:<8d} | {row['sr_is_mean']:6.3f} +/- {row['sr_is_std']:5.3f}        | "
            f"{row['sr_is_theory_blp']:10.3f} | {row['sr_oos_mean']:+6.3f} +/- {row['sr_oos_std']:5.3f}         | "
            f"{row['naive_p_one_sided_median']:14.3e} | {row['naive_p_two_sided_median']:14.3e} | {row['bonferroni_p_one_sided_median']:12.4f}"
        )
    print(
        f"  Canonical M=10,000 winner     : SR_IS={res3['canonical_10k_sr_is']:.3f} "
        f"(p1={res3['canonical_10k_naive_p_one_sided']:.3e}, p2={res3['canonical_10k_naive_p_two_sided']:.3e}) "
        f"-> SR_OOS={res3['canonical_10k_sr_oos']:+.3f}"
    )
    print(
        f"  Berkson (N={res3['berkson_N']:,})            : r_marginal={res3['berkson_r_marginal']:+.5f} (p={res3['berkson_p_marginal']:.3f})"
    )
    print(
        f"  Berkson (Z > {res3['berkson_tau_highlight']:.1f}, {100*res3['berkson_selected_fraction']:.2f}%)    : "
        f"r_emp={res3['berkson_r_cond_empirical']:+.5f} vs r_exact={res3['berkson_r_cond_exact']:+.5f} "
        f"(p={res3['berkson_p_cond']:.2e}, max sweep err={res3['berkson_max_sweep_abs_err']:.4f})"
    )
    print(
        f"  Berkson exact reference pts   : rho(tau=0.0)={res3['berkson_r_tau_0_exact']:+.5f}, "
        f"rho(tau=2.0)={res3['berkson_r_cond_exact']:+.5f}, rho(tau=2.5)={res3['berkson_r_tau_2_5_exact']:+.5f}"
    )
    print(f"  Saved figure                  -> {res3['figure_path']}")

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 78)
    print(f"ALL ASSERTIONS PASSED (EXIT 0) in {elapsed:.2f} s.")
    print("=" * 78)

    # Emit exact JSON metrics to stdout block? No, keep stdout crisp; optionally write?
    # Note: exclusive ownership restricts us to verify_dcqe_ai.py, figures/*, and 04_*.md,
    # so we do NOT create extra unrequested files in sim/.
    _ = json.dumps({"exp1": res1, "exp2": res2, "exp3": res3})


if __name__ == "__main__":
    main()
