# Delayed-Choice Quantum Eraser, Post-Selection, and Artificial Intelligence: Theoretical Foundations & Reproducible Verification Suite

[![Code & Suite DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591062.svg)](https://doi.org/10.5281/zenodo.24591062)
[![Paper 1 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591340.svg)](https://doi.org/10.5281/zenodo.24591340)
[![Paper 2 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591378.svg)](https://doi.org/10.5281/zenodo.24591378)
[![Paper 3 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24591422.svg)](https://doi.org/10.5281/zenodo.24591422)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--1223--2170-A6CE39.svg)](https://orcid.org/0009-0001-1223-2170)

**Author:** Fabien Pierret (`Independent Researcher`)  
**ORCID:** [0009-0001-1223-2170](https://orcid.org/0009-0001-1223-2170)  
**Zenodo Software Archive DOI:** [10.5281/zenodo.24591062](https://doi.org/10.5281/zenodo.24591062) *(Concept DOI: `10.5281/zenodo.24591061`)*  
**License:** MIT (code & scripts) / CC BY 4.0 (preprints & figures)

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

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── colab_gpu_suite.ipynb                         # Google Colab GPU (T4/A100) replication notebook
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
