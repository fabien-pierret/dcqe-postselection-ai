# Dual Independent Agent Formal Audit Record — `lean/DCQEPostSelectionCore.lean`

> **Audit Date:** November 27, 2026  
> **Lean Toolchain:** `leanprover/lean4:v4.36.0` (Apple Silicon ARM64, zero external dependencies)  
> **Audited Artifact:** [`lean/DCQEPostSelectionCore.lean`](./DCQEPostSelectionCore.lean) (`66` machine-checked public theorems, `66 / 66` `#print axioms` checks)  
> **Auditors:** Two independent, read-only adversarial agents:
> - **Agent #1 — Statement Auditor:** *Lean 4 Statement-Gaming, Vacuity & Kernel Soundness Auditor*
> - **Agent #2 — Math Fidelity Referee:** *Mathematical Fidelity, Equation Cross-Check & Anti-Overclaim Referee* (cross-checked verbatim against the three compiled LaTeX manuscripts `Article_1_PRX_Quantum`, `Article_2_Nature_MI`, and `Article_3_ICML_JMLR`)

---

## 1. Final Post-Remediation Verification Status

| Verification Gate | Status | Evidence |
|---|---|---|
| Clean Lean 4.36.0 Typecheck (`lean lean/DCQEPostSelectionCore.lean`) | **`PASS`** | Exit code `0`, `0` errors, `0` warnings |
| Standard Kernel Axioms Only (`66 / 66` public theorems) | **`PASS`** | Exclusively `[propext, Classical.choice, Quot.sound]`; `0` `sorryAx`, `0` `native_decide`, `0` `Lean.trustCompiler`, `0` custom axioms |
| Division-by-Zero Protection (`x / 0 = 0` loophole) | **`PASS`** | Every quotient denominator across all 66 theorems explicitly proven `> 0` |
| Non-Vacuity Witnesses (`100%` conditional theorem coverage) | **`PASS`** | Explicit machine-checked witness theorems for every conditional result across Papers 1, 2, and 3 (including `sips_inf_satisfies_uniqueness_axioms`, anisotropic Jensen concavity, and 5-row classical LHV detection loophole) |
| Ordered-Field Transferability (`ℚ → ℝ`) | **`PASS`** | Uses only ordered-field `Rat` axioms (`div_def`, `inv_pos`, `le_of_lt`, `le_refl`, `mul_nonneg`, `mul_pos`) + `Nat` induction; zero `ℚ`-specific representation properties |
| Epistemic Hygiene (Zero Quantum Woo & Zero Overclaim) | **`PASS`** | Strictly classical statistics, Markov/KV-cache tables, and Euclidean geometry in `ℝ^d`; exact demarcation between Lean 4 algebraic cores and PDF analytical / `sim/` empirical proofs |

---

## 2. Summary of Independent Agent Findings & Remediations Applied

### 2.1. Agent #1 (Statement-Gaming & Kernel Soundness Auditor)

| Finding | Severity | Remediation Applied & Re-Verified |
|---|---|---|
| `linear_minimax_impossibility_general_K` allowed `m = 0` (`K = 1`), where `eps0 < delta` and `delta / 1 ≤ eps0` are mutually contradictory; lacked head-to-head separation against `sipsInf`; 4-pillar witness left disjunction unresolved | `CRITICAL` | Re-indexed structurally to `Fin (m + 2)` (`K = m + 2 ≥ 2`); added `linear_blind_where_sips_vetoes_general_K` proving simultaneous causal pass + single-mode veto vs. linear blindness for every `τ ∈ [ε_0, δ)`; replaced 4-pillar witness with conjunctive `witness_paper3_linear_separation_4pillars` |
| Corollaries 1A & 1B proved one-sided `cycleS s c - 2*Δ_TV ≤ 2` (missing `ratAbs`), and Corollary 1B docstring falsely claimed satisfiability for maximally frustrated `\|δ_i\| = 1` | `CRITICAL` | Added `negOddParity4` and upgraded BOTH `corollary1a_pure_framing_total_absorption` and `corollary1b_edgewise_signaling_dominance` to full two-sided `ratAbs (cycleS s c) - 2 * cycleDeltaTV c ≤ 2`; removed false `\|δ_i\| = 1` claim; added nontrivial odd-parity witness `witness_llm_corollary1b_signaling_dominance` with raw `S = 5/2 > 2` absorbed to `-2 ≤ 2` |
| `manski_rosenbaum_hajek_sharp_bounds` claimed sharpness/attainment in its docstring without proving vertex attainment | `CRITICAL` | Added explicit existential attainment conjuncts at both feasible vertices `(C_lo, E_hi)` and `(C_hi, E_lo)` plus finite-sample `Fin n` theorem `manski_rosenbaum_sample_hajek_bounds` |
| `frechet_amplification_power_law` & `frechet_unbounded_vs_any_constant` allowed degenerate `nu = 0` (and `a_cl ≠ 0` instead of `0 < a_cl`) | `CRITICAL` | Enforced `(hnu : 0 < nu)`, `(ha_cl : 0 < a_cl)`, `(ha_quant : 0 < a_quant)`, `(hkappa : 0 < kappa)`, and added `ratPow_lt_implies_base_lt` (`B^ν < R^ν ⇒ B < R`) |
| `sips_inf_uniqueness` lacked a machine-checked non-vacuity witness showing `sipsInf` satisfies its axioms (and `hsym` was redundant) | `CRITICAL` | Added `sips_inf_satisfies_uniqueness_axioms` + `witness_paper3_sips_uniqueness_nonvacuous`, and dropped redundant `hsym` to strengthen the Uniqueness Theorem |
| 12 of 47 public theorems were absent from `#print axioms` | `MODERATE` | Expanded `#print axioms` footer to **66 / 66 (100%)** of all public theorems (including every witness theorem) |
| 14+ conditional theorems lacked dedicated non-vacuity witnesses | `MODERATE` | Added witnesses covering 100% of conditional theorems across all three papers (including `n = 2` LHV & CbD mixtures, non-multiple Šidák collapse, all Welch bounds on `rationalUNTF_2x4`, and isotropic + anisotropic Spectral Jensen concavity) |
| `mobiusSymDelta0` started post-cancellation and hardcoded `S^{sym} = 4`; `welch_overloaded_forces_crosstalk` was qualitative-only | `MODERATE` | Derived `mobiusSymDelta0FromCycles` & `mobiusSymSFromCycles = 4` directly from explicit forward/reverse `AutoRegEdge` tables; added quantitative `welch_max_coherence_bound` proving `∃ i ≠ j, (M - d)/(d*(M - 1)) ≤ ⟨w_i, w_j⟩^2` |
| Minor strict-positivity conjunct uniformity & `M = 0` guard in `welch_frobenius_bound` | `MINOR` | Added `(hM : 0 < M)` to `welch_frobenius_bound` and explicit `> 0` denominator conjuncts across `welch_offdiag_coherence_bound`, `spectral_jensen_vulnerability_bound`, and `sherman_morrison_and_overload_snr_attenuation` |

### 2.2. Agent #2 (Mathematical Fidelity & Anti-Overclaim Referee)

| Finding | Severity | Remediation Applied & Re-Verified |
|---|---|---|
| **[M1]** Paper 1 docstring wrote `11.757×` instead of exact `(4/π)^{2/3} · 1000^{1/3} ≈ 11.747×` (`11.75×` theory / `11.76×` empirical in Paper 1 Table 2), and needed clearer leading-asymptotic framing | `MODERATE` | Corrected docstring in `frechet_student_t3_c1000_lower_bound` to exact `≈ 11.747×` (`11.75×` theory / `11.76×` empirical) and explicitly scoped the bound to the leading asymptotic factor `κ^{2/3} · 1000^{1/3}` |
| **[M2]** `sidak_c_indexed_collapse` assumed exact divisibility `M = C * k` and omitted the `1 - ε` bound | `MODERATE` | Generalized `sidak_c_indexed_collapse` to arbitrary `M ≥ C * k` (with `0 ≤ F_tau`, `0 < C`), proving both `1 - 1/(1 + k*α) ≤ FPR` and `0 < 1 - ε ≤ FPR`, witnessed at non-multiple `C = 2, k = 20, M = 41` |
| **[M3]** Demarcate i.i.d. max-CDF factorization, Probability Integral Transform, and analytical tail comparison `F_heavy < F_gauss` | `MODERATE` | Added explicit demarcation in both `Paper1_QDSR` docstrings and top-level Scope Statement |
| **[M4]** `cycleDeltaTV` was defined as `cycleDelta0 / 2` instead of proved from probability marginals; `Δ_0 ≤ 2 * Δ_mismatch` unproved; KDL (2015) necessity vs. sufficiency unstated | `MODERATE` | Defined `edgeP2Marginal` and `cycleDeltaTV` directly as probability-scale TV shifts `∑ \|p̃_i - p_i\|`; proved `cycleDelta0_eq_two_mul_cycleDeltaTV` (`Δ_0 = 2 * Δ_TV`) and `coupling_marginal_shift_le_two_mismatch` (`\|E[X] - E[Y]\| ≤ 2 * P(X ≠ Y)`); explicitly documented KDL (2015) necessity-only scope |
| **[M5]** Witness D (`witness_detection_loophole_rational`) only checked `4/(4/5) - 2 = 3` arithmetically without a classical LHV + abstention table | `MODERATE` | Replaced with `witness_detection_loophole_classical_lhv_table`: an explicit 5-row classical LHV table over `{-1, 0, +1}^4` (`η_single = 4/5`, `η_pair = 3/5`, `η_cond = 3/4`) proving `S_post = 10/3 > 2√2` directly from conditional expectations AND proving `\|S_imputed\| ≤ 2` via `bell_chsh_lhv_mixture` under `∅ ↦ +1` imputation |
| **[M6]** Rosenbaum odds-ratio weight bracket (Paper 3 Eq. 6) and sample-level summation unformalized | `MODERATE` | Added `rosenbaum_odds_ratio_weight_bracket` (exact IFF of Paper 3 Eq. 6 with all positive denominators) and `manski_rosenbaum_sample_hajek_bounds` over `Fin n`; demarcated clipped `I_col` (Eq. 7b) as proved in Paper 3 |
| **[M7]** Missing `ρ ≤ 1` Loewner plateau and unstated matrix-analysis demarcation for Paper 3 Theorem 2 | `MODERATE` | Added `sherman_morrison_scalar_id` (general `α < 1`) and `orthonormal_plateau_snr_id` (`ρ ≤ 1` plateau `Δ_mem^2 / a`); added explicit Scope Statement demarcation for matrix spectral decomposition (Eq. 18), Loewner operator monotonicity (Eq. 13), matrix Sherman–Morrison, and Markov/`erf`/Le Cam bounds (Eqs. 14–16) |
| **[m1–m6]** Four equation-number offsets (`Eq. 14→15` P1, `Eq. 4→5` P2, `Prop. 1.4→Proposition 1(4)` P2, `Eq. 14→19/13` P3) + `ℝ` transferability nuance (`Classical.dec`, integer `nu`) | `MINOR` | Corrected every equation reference against the compiled PDFs and documented `Classical.dec` + natural-number `nu` explicitly in the header |

---

## 3. Exact Reproducible Verification Command

Anyone with Lean 4 (`v4.36.0`) can independently verify all `66` machine-checked theorems in ~5 seconds with zero external downloads:

```bash
lean lean/DCQEPostSelectionCore.lean
```

Expected result: exit code `0`, `0` warnings, `0` errors, and `66` `#print axioms` lines reporting exclusively `[propext, Classical.choice, Quot.sound]`.
