# Delayed-Choice Quantum Eraser, Post-Selection, and Artificial Intelligence: Theoretical Foundations & Reproducible Verification Suite

[![Lean 4.36](https://img.shields.io/badge/Lean_4.36-66_Theorems_·_0_sorry-10B981.svg)](./lean/DCQEPostSelectionCore.lean)
[![Dual-Agent Audit](https://img.shields.io/badge/Formal_Audit-2_Independent_Agents-0EA5E9.svg)](./lean/AUDIT_REPORT_2_AGENTS.md)
[![Code & Suite DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591062.svg)](https://doi.org/10.5281/zenodo.24591062)
[![Paper 1 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591340.svg)](https://doi.org/10.5281/zenodo.24591340)
[![Paper 2 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591378.svg)](https://doi.org/10.5281/zenodo.24591378)
[![Paper 3 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591422.svg)](https://doi.org/10.5281/zenodo.24591422)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--1223--2170-A6CE39.svg)](https://orcid.org/0009-0001-1223-2170)

**Author:** Fabien Pierret (`Independent Researcher`)  
**ORCID:** [0009-0001-1223-2170](https://orcid.org/0009-0001-1223-2170)  
**Research Showcase (GitHub Pages · AEO/JSON-LD):** [https://fabien-pierret.github.io/dcqe-postselection-ai/](https://fabien-pierret.github.io/dcqe-postselection-ai/) (`llms.txt` included)  
**Live Edge Auditor:** [https://audit.wallstreetdebug.com](https://audit.wallstreetdebug.com) · **PyPI Package:** [`pip install qdsr-audit`](https://pypi.org/project/qdsr-audit/)  
**WallStreetDebug Ecosystem:** [Hub (`wallstreetdebug.com`)](https://wallstreetdebug.com) · [EvolveGuard (`evolve.wallstreetdebug.com`)](https://evolve.wallstreetdebug.com)  
**Zenodo Software Archive DOI:** [10.5281/zenodo.24591062](https://doi.org/10.5281/zenodo.24591062) *(Concept DOI: `10.5281/zenodo.24591061`)*  
**License:** MIT (code, Lean 4 proofs & scripts) / CC BY 4.0 (preprints & figures)

---

## Overview

This repository provides the complete, self-contained, deterministic verification suite (**12 numerical and live-LLM experiments**, **10 Python scripts**, **9 raw JSON result archives**, **12 publication-grade 300 DPI figures**, and a **Google Colab GPU notebook**) accompanying three original research preprints by **Fabien Pierret** investigating the mathematical boundary between **quantum post-selection / no-signaling** and **classical post-selection in Artificial Intelligence and Large Language Models (LLMs)**:

### 1. Preprints (`papers/`)

1. **Paper 1 (*PRX Quantum* / *Quantum* / *NeurIPS* track)** — [`papers/Article_1_PRX_Quantum.pdf`](./papers/Article_1_PRX_Quantum.pdf) · **Preprint DOI:** [10.5281/zenodo.24591340](https://doi.org/10.5281/zenodo.24591340)  
   *The Quantum Overfitting Amplification Theorem and Heavy-Tail Quantum-Adjusted Deflated Sharpe Ratio ($\mathrm{Q\text{-}DSR}_\nu$)*  
   - Proves that under fixed oracle query budget $C$ on frozen null data, quadratic quantum search (Grover / Dürr–Høyer) amplifies false in-sample discoveries by the universal factor **$2^{1/\alpha}$** for sub-Weibull tails ($\sqrt{2}$ Gaussian, $2$ Laplace) and triggers an unbounded **polynomial explosion $\Theta(\kappa^{2/\nu} C^{1/\nu})$** under Fréchet heavy tails (reaching **$11.76\times$** at $C = 1{,}000$ for Student-$t_3$), while leaving out-of-sample expectation identically zero.
   - Derives the closed-form heavy-tail **$\mathrm{Q\text{-}DSR}_\nu$** restoring exact $\mathrm{FPR} = \alpha$ (vs. $53.4\%\text{--}100\%$ failure of $C$-indexed DSR and $56.9\%$ failure of naive participation-ratio discounting), together with the Landauer $2\times$ thermodynamic selection dissipation bound.

2. **Paper 2 (*Nature Machine Intelligence* / *PRX Quantum* / *NeurIPS* track)** — [`papers/Article_2_Nature_Machine_Intelligence.pdf`](./papers/Article_2_Nature_Machine_Intelligence.pdf) · **Preprint DOI:** [10.5281/zenodo.24591378](https://doi.org/10.5281/zenodo.24591378)  
   *The Contextuality–Signaling Dichotomy in Autoregressive Large Language Models: Causal-Mask Absorption, Semantic Frustration Phase Transition, and Correlated Effective Multiplicity*  
   - Proves the **Causal-Mask Signaling-Absorption Theorem** (explaining live `Qwen2.5-1.5B-Instruct` results: **38/51** natural multi-turn cycles breach raw $s_{\text{odd}} > 2$ up to $3.487$, yet **0/51** violate Contextuality-by-Default after subtracting KV-cache signaling $\Delta_0$, $\max C_{\text{CbD}} = 1.9997$).
   - Proves the exact IFF **Möbius Frustration Phase Transition** and **Symmetrization Law** $\Delta_0^{\text{sym}} = \max(|m_1|, |m_3|) + \max(|m_2|, |m_4|) \le 2$, validated live on frozen `Qwen2.5-1.5B-Instruct` weights (`SHA-256` invariant, $\|\Delta\theta^*\|_2 = 0$): balanced Möbius cycle `M01` reaches $C_{\text{CbD}} = 2.959$ (forward), $3.043$ (reverse), and $3.209$ (symmetrized, surpassing Tsirelson's $2\sqrt{2}$ purely via classical KV-cache access), with **15/15 (100%)** symmetrized Möbius cycles crossing $C_{\text{CbD}} > 2$.
   - Demonstrates live on isolated LLM logits ($0/1{,}050$ violations at $\eta = 1$) that even **benign confidence-threshold abstention ($\eta \le 95\%$)** opens a real-LLM **detection loophole** ($|S| = 2.2756$, closed back to $\le 1.7783$ via deterministic $\emptyset \mapsto +1$ imputation), and proves the universal spectral multiplicity ordering $1 \le M_{\text{eff}}^{\text{PR}} \le M_{\text{eff}}^{\text{Galwey}} \le M$ reducing live prompt-dredging optimism MAE by $2.65\times\text{--}2.79\times$.

3. **Paper 3 (*ICML* / *JMLR* / *Nature Machine Intelligence* track)** — [`papers/Article_3_ICML_JMLR.pdf`](./papers/Article_3_ICML_JMLR.pdf) · **Preprint DOI:** [10.5281/zenodo.24591422](https://doi.org/10.5281/zenodo.24591422)  
   *Minimax Impossibility of Linear Post-Selection Auditors ($\mathrm{SIPS}_\infty$) and Analytical Welch–Superposition Coupling to Membership Inference Attacks ($V_{\mathrm{AI}}^2 + D_{\mathrm{MIA}}^2 \le 1$)*  
   - Proves that over $K$ orthogonal post-selection failure manifolds, **every** convex linear aggregator $\sum_k w_k I_k$ suffers minimax risk $R_{\max} = 1$ (confirmed across 500 Dirichlet weights: min oracle risk $\ge 0.90$ vs. $0.00$ for the unique minimax-optimal $\ell_\infty$ auditor $\mathrm{SIPS}_\infty$), coupled with closed-form **Manski–Rosenbaum** partial-identification bounds verified to $2.22 \times 10^{-16}$ against Charnes–Cooper linear programming.
   - Derives the first closed-form analytical coupling between neural geometric superposition overload $\rho = M/d$ (Welch bound) and white-box Bayes-optimal Membership Inference Attack advantage $D_{\mathrm{MIA}}$ via Loewner monotonicity, Sherman–Morrison rank-one updates ($3.55 \times 10^{-15}$ error), strict spectral concavity, and Le Cam–Bhattacharyya duality—validated on Adam-trained ReLU autoencoders ($-59.4\%$ MIA leakage reduction, $V_{\mathrm{AI}}^2 + D_{\mathrm{MIA}}^2 \le 0.9313 \le 1$).

---

## Machine-Checked Lean 4.36.0 Certificate (`lean/`, 0 `sorry`, Dual-Agent Audited)

[`lean/DCQEPostSelectionCore.lean`](./lean/DCQEPostSelectionCore.lean) (`leanprover/lean4:v4.36.0`, zero external dependencies) formally verifies **66 public theorems** (`66 / 66` checked via `#print axioms`, depending exclusively on `[propext, Classical.choice, Quot.sound]`; zero `sorryAx`, zero `native_decide`, zero custom axioms; every quotient denominator proven `> 0`; explicit non-vacuity witnesses for every conditional theorem), audited by two independent read-only agents in [`lean/AUDIT_REPORT_2_AGENTS.md`](./lean/AUDIT_REPORT_2_AGENTS.md).

**Exact Audited Scope (Deterministic Rational Algebraic Cores over $\mathbb{Q}$, transferring verbatim to $\mathbb{R}$; zero overclaim, zero *quantum woo*):**
- **Paper 1 (`Paper1_QDSR`, 20 theorems):** Leading-order Fréchet polynomial law $(a_{\text{quant}}/a_{\text{cl}})^\nu = \kappa^2 C$, scale cancellation, unbounded growth past any constant $B$, Student-$t_3$ leading-asymptotic witness ($> 11.7\times$ at $C=1{,}000$; exact Eq. 8 value $\approx 11.747\times$), exact rational Bernoulli contraction $(1+nu)(1-u)^n \le 1$, non-asymptotic collapse of both $C$-indexed Bonferroni ($M \ge kC$) and $C$-indexed Šidák ($M \ge Ck$, arbitrary non-multiples) to $\mathrm{FPR} \ge 1 - \varepsilon$, exact $\mathrm{Q\text{-}DSR}_\nu$ monotonicity/calibration, and strict heavy-tail under-coverage of Gaussian $\mathrm{Q\text{-}DSR}$.
- **Paper 2 (`Paper2_CHSH_CbD`, 20 theorems):** Bell–CHSH $|S| \le 2$ on $\{-1,+1\}^4$ and arbitrary finite LHV mixtures; necessity direction of Contextuality-by-Default $|S| - 2\Delta_{\text{mismatch}} \le 2$ on finite couplings plus $|E[X] - E[Y]| \le 2 P(X \neq Y)$; autoregressive causal-mask Lemma 1, exact $\Delta_0 = 2\Delta_{\text{TV}}$, Master Signaling-Absorption Identity (Eq. 5), two-sided Corollaries 1A ($\delta = 0 \Rightarrow |S| - 2\Delta_{\text{TV}} \le 2$) & 1B (edgewise signaling dominance); exact Möbius Symmetrization Law ($S^{\text{sym}} = 4$, $\Delta_0^{\text{sym}} = \max(|m_1|,|m_3|)+\max(|m_2|,|m_4|) < 2$) derived directly from forward/reverse probability tables; plus 4 purely classical rational LLM/LHV witnesses (including a 5-row classical LHV abstention table hitting $S_{\text{post}} = 10/3 > 2\sqrt{2}$ yet closing back to $\le 2$ under $\emptyset \mapsto +1$ imputation).
- **Paper 3 (`Paper3_SIPS_Welch`, 26 theorems):** General $K \ge 2$ minimax blindness of every convex linear auditor ($\delta/K \le \varepsilon_0 < \delta$) + strict head-to-head separation vs. $\mathrm{SIPS}_\infty$; zero minimax risk & full Uniqueness Theorem of $\mathrm{SIPS}_\infty$ (with machine-checked axiom satisfiability); Rosenbaum odds-ratio weight bracket IFF (Eq. 6) + sharp Manski–Rosenbaum Hájek IPSW bounds with vertex attainment & `Fin n` sample summation; arbitrary-dimension $(d, M)$ Frobenius–Welch bound $\|W^\top W\|_F^2 \ge M^2/d$, off-diagonal cross-talk $M(M-d)/d$, quantitative max-coherence Welch bound $(M-d)/(d(M-1))$, impossibility of orthogonality for $M > d$, and exact $2\times 4$ rational UNTF saturation; arbitrary-dimension $d$ spectral Jensen concavity bound (Eq. 19, equality IFF isotropic $\lambda_\ell = M/d$), scalar Sherman–Morrison, $\rho \le 1$ orthonormal SNR plateau, and strict overload SNR attenuation for $\rho \ge 1$.

*Demarcation:* Probabilistic asymptotics (Fisher–Tippett–Gnedenko, Probability Integral Transform, Student-$t_\nu$ tail integrals, Slepian), KDL (2015) maximal-coupling sufficiency, clipped $I_{\text{col}}$ (Eq. 7b), matrix spectral decomposition (Eq. 18), Loewner operator monotonicity for $\rho \le 1$, matrix Sherman–Morrison, Markov/$\mathrm{erf}$/Le Cam bounds (Eqs. 14–16), and the 12 Python/MLX benchmarks are proved analytically in `papers/*.pdf` and reproduced deterministically in `sim/`.

```bash
lean lean/DCQEPostSelectionCore.lean
```

---

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── llms.txt                                      # Machine-readable AEO specification
├── requirements.txt
├── .gitignore
├── colab_gpu_suite.ipynb                         # Google Colab GPU (T4/A100) replication notebook
├── lean/                                         # Zero-dependency Lean 4.36.0 machine-checked proofs
│   ├── lean-toolchain                            # leanprover/lean4:v4.36.0
│   ├── DCQEPostSelectionCore.lean                # 66 public theorems, 0 sorry, standard axioms only
│   └── AUDIT_REPORT_2_AGENTS.md                  # Dual independent agent audit record
├── docs_site/                                    # GitHub Pages SEO/AEO showcase (index.html, llms.txt, robots.txt)
├── papers/                                       # Camera-ready English PDFs (Fabien Pierret)
│   ├── Article_1_PRX_Quantum.pdf
│   ├── Article_2_Nature_Machine_Intelligence.pdf
│   └── Article_3_ICML_JMLR.pdf
└── sim/
    ├── verify_dcqe_ai.py                         # Exp 1–3: DCQE no-signaling, CHSH/Werner, data dredging & Berkson
    ├── exp4_sips_benchmark.py                    # Exp 4: SIPS causal audit benchmark (K=240, DeLong test)
    ├── exp5_shadows_and_quantum_memory.py        # Exp 5: Pauli Classical Shadows & noisy Bell memory threshold
    ├── exp6_barren_plateaus_gcq.py               # Exp 6: Exact Barren Plateaus (2^-n vs 4^-n) & g_CQ
    ├── exp7_jury_traps_loophole_grover.py        # Exp 7: Classical Detection Loophole & unitary Grover dredging
    ├── exp8_real_llm_prompt_dredging.py          # Exp 8: Live Qwen2.5-1.5B prompt dredging & entropy collapse (MLX)
    ├── exp9_real_llm_cbd_signaling.py            # Exp 9: Live Qwen2.5-1.5B isolated CHSH & 36-cycle CbD (MLX)
    ├── exp10_paper1_quantum_overfitting_heavy_tails.py  # Exp 10: Paper 1 heavy-tail amplification & Q-DSR_nu
    ├── exp11_paper2_autoregressive_frustration_transition.py  # Exp 11: Paper 2 Mobius transition, loophole & M_eff (MLX)
    ├── exp12_paper3_duality_and_sips_minimax.py  # Exp 12: Paper 3 SIPS_inf minimax, Manski LP & ReLU AE MIA
    ├── results/                                  # 9 verified raw JSON output archives (exp4–exp12)
    └── figures/                                  # 12 publication-grade 300 DPI PNG figures (fig1–fig12)
```

---

## Reproducibility & Quickstart

All scripts use a fixed deterministic seed (`GLOBAL_SEED = 42`) and run cleanly on **macOS (Apple Silicon ARM64)** and **Linux (x86_64/ARM64)**.

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run All Quantum, Statistical & Theoretical Experiments (`Exp 1–7, 10, 12`, CPU, < 1 minute)

```bash
python3 sim/verify_dcqe_ai.py && \
python3 sim/exp4_sips_benchmark.py && \
python3 sim/exp5_shadows_and_quantum_memory.py && \
python3 sim/exp6_barren_plateaus_gcq.py && \
python3 sim/exp7_jury_traps_loophole_grover.py && \
python3 sim/exp10_paper1_quantum_overfitting_heavy_tails.py && \
python3 sim/exp12_paper3_duality_and_sips_minimax.py
```

### 3. Run Live Real-LLM Experiments (`Exp 8, 9, 11` on `Qwen2.5-1.5B-Instruct-4bit`)

On **Apple Silicon macOS** (requires `pip install mlx mlx-lm`):

```bash
python3 sim/exp9_real_llm_cbd_signaling.py && \
python3 sim/exp11_paper2_autoregressive_frustration_transition.py && \
python3 sim/exp8_real_llm_prompt_dredging.py
```

On **NVIDIA GPU (Google Colab T4 / A100)** via PyTorch & HuggingFace `transformers`, open [`colab_gpu_suite.ipynb`](./colab_gpu_suite.ipynb) directly in Google Colab.

---

## Citation

If you use this codebase, figures, or theoretical results in your research, please cite using [`CITATION.cff`](./CITATION.cff) or the BibTeX entry below:

```bibtex
@software{pierret2026dcqe_postselection_ai,
  author       = {Pierret, Fabien},
  title        = {Delayed-Choice Quantum Eraser, Post-Selection, and Artificial Intelligence: Theoretical Proofs and Reproducible Verification Suite},
  year         = {2026},
  version      = {1.0.0},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24591062},
  url          = {https://doi.org/10.5281/zenodo.24591062},
  license      = {MIT},
  note         = {ORCID: 0009-0001-1223-2170}
}
```
