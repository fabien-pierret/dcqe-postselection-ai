#!/usr/bin/env python3
"""Experiment 9 — Real LLM Isolated Bell-CHSH & Multi-Turn KV-Cache Signaling
vs. Contextuality-by-Default (CbD) on Apple Silicon MLX (H2.1 & H2).

Evaluates exact deterministic next-token binary probabilities P("Yes") / P("No")
on ``mlx-community/Qwen2.5-1.5B-Instruct-4bit`` (Metal GPU, zero Monte Carlo error):

1. Exact frozen-weight invariance audit (SHA-256 + L2 norm before/after: ||dtheta*|| = 0).
2. Part A — Isolated Bipartite Bell-CHSH Protocol (eta = 1.0, zero rejection):
   Runs strictly isolated Alice (lambda_k, x) and Bob (lambda_k, y) calls across
   50 shared classical context scenarios lambda_k (x, y in {0, 1}). Computes exact
   local expectations <A>_{x,lambda}, <B>_{y,lambda} in [-1, +1] and verifies
   |S_CHSH| <= 2.000000 across all 50 individual scenarios, the uniform aggregate,
   and 1,000 random/adversarial classical mixtures over the 50 LLM scenarios.
3. Part B — Multi-Turn Sequential Protocol (KV-Cache Signaling Delta_0, QQ-Equality
   q_QQ, and Exact Contextuality-by-Default Witness C_CbD = s_odd - Delta_0):
   Constructs 36 rank-4 cycles of binary Yes/No questions across 3 categories
   (12 cycles each = 144 unique pairs, 288 ordered sequential evaluations):
     (i)   Neutral Independent Factual Controls;
     (ii)  Psychological Framing / Order-Effect Cycles (cf. Wang et al., 2014);
     (iii) Semantic Ambiguity / Anaphora / Polysemy Cycles (cf. Lo et al., 2024).
   For each ordered pair (q_a -> q_b), computes the exact joint distribution
   P(a, b) = P(a | q_a) * P(b | q_a, a, q_b) over all 4 (a, b) in {+1, -1}^2.
   Measures exactly:
     - Sequential KV-cache signaling (marginal inconsistency):
         Delta_0 = sum_{i=1}^4 |<Q_i>_{c_i} - <Q_i>_{c_{i-1}}|
     - QQ-equality residual (Wang & Busemeyer, 2013):
         q_QQ = [P_{AB}(+,+) + P_{AB}(-,-)] - [P_{BA}(+,+) + P_{BA}(-,-)]
     - Exact CbD witness (Dzhafarov & Kujala, 2016):
         C_CbD = s_odd - Delta_0  vs. classical non-contextual threshold n - 2 = 2
   Reports directed forward, directed reverse, and order-symmetrized cycles
   transparently with zero cherry-picking.

Outputs:
- ``sim/results/exp9_llm_cbd_signaling.json``
- ``sim/figures/fig9_real_llm_cbd_signaling.png`` (300 DPI, 4 panels)

Hardware target: Apple Silicon (macOS ARM64, Metal GPU via MLX).
Deterministic global seed: 42.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.utils as mxu
import mlx_lm
import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_DIR = SCRIPT_DIR / "results"

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

SYS_PROMPT = (
    "You are answering a short survey. Answer each question strictly with a single word: "
    "Yes or No. Do not explain."
)


def configure_matplotlib() -> None:
    """Configure publication-grade Matplotlib styling."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 10.9,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.1,
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


def fingerprint_frozen_weights(model: Any) -> dict[str, Any]:
    """Compute exact SHA-256 digest and L2 norm over all model parameter tensors."""
    flat = mxu.tree_flatten(model.parameters())
    hasher = hashlib.sha256()
    sq_sum = 0.0
    total_elems = 0
    for name, arr in flat:
        np_arr = np.asarray(arr)
        hasher.update(name.encode("utf-8"))
        hasher.update(np_arr.tobytes())
        sq_sum += float(np.sum(np_arr.astype(np.float64) ** 2))
        total_elems += int(np_arr.size)
    return {
        "sha256": hasher.hexdigest(),
        "l2_norm": float(math.sqrt(sq_sum)),
        "num_tensors": len(flat),
        "num_stored_elements": total_elems,
    }


def last_token_logits(model: Any, hidden_last: mx.array) -> mx.array:
    """Project only last-token hidden states (B, d) to vocabulary logits (B, V)."""
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(hidden_last)
    return model.lm_head(hidden_last)


def batch_eval_yes_probs(
    model: Any,
    tok: Any,
    conversations: list[list[dict[str, str]]],
    yes_id: int,
    no_id: int,
    batch_size: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact next-token P(Yes | conv) and Yes+No mass over batched conversations."""
    encoded = []
    for conv in conversations:
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        encoded.append(tok.encode(text, add_special_tokens=False))

    n = len(encoded)
    order = sorted(range(n), key=lambda i: len(encoded[i]))
    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    probs_yes = np.zeros(n, dtype=np.float64)
    yn_mass = np.zeros(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        idxs = order[start:end]
        cb = len(idxs)
        max_len = max(len(encoded[i]) for i in idxs)
        arr = np.full((cb, max_len), pad_id, dtype=np.int32)
        lens = np.zeros(cb, dtype=np.int32)
        for row, orig_i in enumerate(idxs):
            seq = encoded[orig_i]
            arr[row, : len(seq)] = seq
            lens[row] = len(seq) - 1

        h = model.model(mx.array(arr))
        last_h = h[mx.arange(cb), mx.array(lens), :]
        logits = last_token_logits(model, last_h).astype(mx.float32)
        lse = mx.logsumexp(logits, axis=-1)
        y_lg = logits[:, yes_id]
        n_lg = logits[:, no_id]
        p_y = mx.sigmoid(y_lg - n_lg)
        mass = mx.exp(y_lg - lse) + mx.exp(n_lg - lse)
        mx.eval(p_y, mass)

        p_np = np.asarray(p_y, dtype=np.float64)
        m_np = np.asarray(mass, dtype=np.float64)
        for row, orig_i in enumerate(idxs):
            probs_yes[orig_i] = p_np[row]
            yn_mass[orig_i] = m_np[row]

    return probs_yes, yn_mass


# ---------------------------------------------------------------------------
# Part A: 50 Shared Classical Contexts for Isolated Bell-CHSH
# ---------------------------------------------------------------------------
def build_50_chsh_scenarios() -> list[dict[str, str]]:
    """Create 50 diverse shared classical context scenarios lambda_k for Bell-CHSH."""
    domains = [
        ("urban transit", "A mid-sized city plans a new light-rail corridor funded by congestion tolls.",
         "Should the city proceed with construction?", "Will daily commuters benefit within five years?",
         "Should outer suburbs help fund the project?", "Is the congestion toll politically sustainable?"),
        ("clinical trial", "An experimental antibiotic shows 15% faster recovery with mild nausea in 8% of patients.",
         "Should regulators grant conditional approval?", "Would you recommend it for moderate infections?",
         "Should hospitals replace standard generics?", "Do benefits clearly outweigh reported side effects?"),
        ("remote work", "A software firm considers shifting permanently from hybrid office schedules to full remote work.",
         "Will overall engineering productivity increase?", "Should the company close its downtown headquarters?",
         "Will junior mentorship suffer significantly?", "Should remote staff receive identical regional pay?"),
        ("nuclear energy", "An industrial region debates extending two 40-year-old nuclear reactors by twenty years.",
         "Should the operating licenses be extended?", "Is this safer than building new gas turbines?",
         "Will electricity prices stabilize for households?", "Should public subsidies cover safety upgrades?"),
        ("ai grading", "A public university tests an automated grading assistant for introductory undergraduate essays.",
         "Should instructors adopt automated first-pass scoring?", "Will grading consistency improve across sections?",
         "Should students be notified when AI assists?", "Could automated scoring penalize original writing?"),
        ("ocean farming", "Coastal planners evaluate expanding offshore kelp and shellfish aquaculture across ten bays.",
         "Should permits be expanded next year?", "Will marine biodiversity improve locally?",
         "Should commercial fishers share the waters?", "Is offshore aquaculture economically resilient?"),
        ("four-day week", "A regional hospital network trials a 32-hour workweek for administrative staff at full pay.",
         "Should the hospital make the schedule permanent?", "Will employee burnout decline substantially?",
         "Could patient billing delays worsen initially?", "Should clinical departments run a similar pilot?"),
        ("central bank", "Inflation falls to 2.7% while unemployment edges up to 4.8% over two consecutive quarters.",
         "Should the central bank cut interest rates?", "Is recession risk currently higher than inflation risk?",
         "Will housing affordability improve next year?", "Should policymakers wait for two more reports?"),
        ("smart meters", "A power utility proposes dynamic hourly electricity pricing paired with home battery rebates.",
         "Should households accept dynamic hourly tariffs?", "Will peak summer grid strain drop meaningfully?",
         "Could low-income renters face unfair bills?", "Should battery rebates target rural feeders first?"),
        ("autonomous vans", "A logistics carrier deploys driverless delivery vans on night routes between regional hubs.",
         "Should highway regulators permit unattended night runs?", "Will collision rates fall below human averages?",
         "Should displaced drivers receive mandatory retraining?", "Will freight costs drop for small merchants?"),
    ]
    # Expand 10 base domains x 5 distinct classical framing/evidence states lambda_k = 50 scenarios
    framings = [
        "Independent audits report strong preliminary compliance and supportive public polling.",
        "Early pilot results are mixed, with sharp disagreement between local stakeholders.",
        "A recent watchdog memo raised serious cost overrun and reliability concerns.",
        "Budget constraints are tight, yet comparable regions reported modest net gains.",
        "Expert panels remain split evenly, recommending cautious phased implementation.",
    ]
    scenarios = []
    idx = 0
    for d_name, base_ctx, a0, a1, b0, b1 in domains:
        for f_idx, frame in enumerate(framings):
            scenarios.append(
                {
                    "id": idx,
                    "domain": f"{d_name}_v{f_idx + 1}",
                    "context": f"{base_ctx} {frame}",
                    "a0": a0,
                    "a1": a1,
                    "b0": b0,
                    "b1": b1,
                }
            )
            idx += 1
    assert len(scenarios) == 50
    return scenarios


def max_odd_signed_sum(correlators: list[float] | np.ndarray) -> tuple[float, list[int]]:
    """Compute exact s_odd = max over all 8 odd-parity sign choices of sum(eps_i * E_i)."""
    e = np.asarray(correlators, dtype=float)
    best_val = -1e9
    best_signs = [1, 1, 1, -1]
    for s0 in (-1, 1):
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                for s3 in (-1, 1):
                    if s0 * s1 * s2 * s3 == -1:
                        val = float(s0 * e[0] + s1 * e[1] + s2 * e[2] + s3 * e[3])
                        if val > best_val:
                            best_val = val
                            best_signs = [s0, s1, s2, s3]
    return best_val, best_signs


# ---------------------------------------------------------------------------
# Part B: 36 Rank-4 Question Cycles (3 Categories x 12 Cycles)
# ---------------------------------------------------------------------------
def build_36_cycles() -> list[dict[str, Any]]:
    """Build 36 distinct rank-4 cycles across 3 theoretically motivated categories."""
    # Category 1: 12 Neutral Independent Factual Controls (unrelated objective facts)
    neutral_cycles = [
        ("N01", ["Is Paris the capital of France?", "Is 17 a prime number?", "Does water boil at 100 degrees Celsius at sea level?", "Is Mars larger than Jupiter?"]),
        ("N02", ["Is Tokyo located in Japan?", "Is 24 divisible by 6?", "Do penguins live naturally in the Arctic?", "Is gold heavier than aluminum?"]),
        ("N03", ["Is the Amazon River in South America?", "Is 2 plus 2 equal to 5?", "Does Earth orbit the Sun?", "Is nitrogen the most abundant gas in Earth's atmosphere?"]),
        ("N04", ["Is Ottawa the capital of Canada?", "Is 81 a perfect square?", "Do whales breathe air using lungs?", "Is Mount Everest located in Europe?"]),
        ("N05", ["Is Cairo located in Egypt?", "Is 15 an even number?", "Does sound travel faster in steel than in air?", "Is diamond harder than quartz?"]),
        ("N06", ["Is Sydney the capital of Australia?", "Is 7 multiplied by 8 equal to 56?", "Do plants use chlorophyll for photosynthesis?", "Is Mercury closer to the Sun than Venus?"]),
        ("N07", ["Is Rome located in Italy?", "Is 100 divisible by 3?", "Is helium lighter than air?", "Do spiders normally have eight legs?"]),
        ("N08", ["Is Lisbon the capital of Portugal?", "Is 13 a prime number?", "Is ice denser than liquid water?", "Does a standard triangle have three sides?"]),
        ("N09", ["Is Nairobi located in Kenya?", "Is 9 squared equal to 81?", "Is copper a good conductor of electricity?", "Do sharks have bones made primarily of cartilage?"]),
        ("N10", ["Is Vienna the capital of Austria?", "Is 45 divisible by 9?", "Does light travel faster than sound?", "Is the Sahara Desert located in Antarctica?"]),
        ("N11", ["Is Madrid located in Spain?", "Is 29 a prime number?", "Is iron attracted to a magnet?", "Does a leap year have 366 days?"]),
        ("N12", ["Is Athens the capital of Greece?", "Is 12 multiplied by 12 equal to 144?", "Is oxygen required for ordinary wood combustion?", "Is the Pacific Ocean smaller than the Atlantic Ocean?"]),
    ]

    # Category 2: 12 Psychological Framing / Order-Effect Cycles (cf. Wang et al., 2014)
    framing_cycles = [
        ("F01_gore_clinton", [
            "In general, do you think Bill Clinton was generally honest and trustworthy?",
            "In general, do you think Al Gore was generally honest and trustworthy?",
            "In general, do you think elected politicians usually keep their promises?",
            "In general, do you think public trust in national leaders is warranted?",
        ]),
        ("F02_privacy_security", [
            "Should governments be allowed to monitor encrypted messages to prevent terror attacks?",
            "Should citizens have an absolute right to private digital communication?",
            "Should technology firms be legally required to build law-enforcement backdoors?",
            "Is personal privacy more important than maximum national security?",
        ]),
        ("F03_taxes_services", [
            "Should income taxes be lowered for middle-class households this year?",
            "Should the government expand free public healthcare and university funding?",
            "Should public debt be reduced even if social budgets shrink?",
            "Are taxpayers currently paying too much relative to public services?",
        ]),
        ("F04_ai_regulation", [
            "Should governments pause deployment of frontier AI systems until safety audits pass?",
            "Should startups face minimal regulation to accelerate AI innovation?",
            "Do the existential risks of advanced AI outweigh its immediate economic gains?",
            "Should open-weight AI models remain freely downloadable without a license?",
        ]),
        ("F05_climate_economy", [
            "Should carbon taxes be raised sharply even if fuel prices rise immediately?",
            "Should economic growth take priority over rapid emissions cuts?",
            "Should gas-powered passenger cars be banned from sale within ten years?",
            "Are current environmental regulations overly burdensome on working families?",
        ]),
        ("F06_free_speech", [
            "Should social platforms strictly remove medical misinformation?",
            "Should peaceful political speech remain uncensored even when factually wrong?",
            "Should independent boards police online speech to protect democracy?",
            "Is censorship by large platforms a greater danger than online falsehoods?",
        ]),
        ("F07_trolley_ethics", [
            "Is it morally acceptable to harm one innocent person to save five others?",
            "Should moral rules forbid intentionally harming an innocent individual under all circumstances?",
            "Should autonomous vehicles be programmed to minimize total casualties at the expense of their passenger?",
            "Does an individual have an inviolable right not to be sacrificed for the greater good?",
        ]),
        ("F08_trade_jobs", [
            "Should a country impose steep tariffs to protect domestic manufacturing jobs?",
            "Does free international trade generally make ordinary consumers better off?",
            "Should critical supply chains be kept domestic even at much higher retail prices?",
            "Do import tariffs usually hurt the economy more than they help?",
        ]),
        ("F09_surveillance_crime", [
            "Should police deploy live facial recognition cameras across public transit stations?",
            "Should facial recognition in public spaces be banned to protect civil liberties?",
            "Would stricter urban surveillance meaningfully deter violent street crime?",
            "Do automated surveillance tools pose an unacceptable risk of wrongful arrest?",
        ]),
        ("F10_work_automation", [
            "Should companies be taxed extra when they replace human workers with robots?",
            "Does workplace automation ultimately create more prosperity than unemployment?",
            "Should governments guarantee a universal basic income funded by automation profits?",
            "Should businesses remain free to automate routine jobs without penalty?",
        ]),
        ("F11_wealth_merit", [
            "Is extreme personal wealth primarily the result of luck and inherited privilege?",
            "Is financial success in modern economies mostly earned through hard work and talent?",
            "Should an annual wealth tax cap billionaire fortunes?",
            "Do high rewards for founders provide an essential incentive for innovation?",
        ]),
        ("F12_journalism_access", [
            "Should foreign journalists from rival nations be allowed to report freely inside the country?",
            "Should domestic journalists be allowed to report freely inside rival nations?",
            "Should states restrict foreign media outlets suspected of spreading propaganda?",
            "Is reciprocal press freedom essential even during geopolitical tensions?",
        ]),
    ]

    # Category 3: 12 Semantic Ambiguity / Polysemy / Anaphora Cycles (cf. Lo et al., 2024)
    semantic_cycles = [
        ("S01_bank", [
            "In 'He sat near the steep bank watching the trout', does 'bank' refer to a financial institution?",
            "In 'The bank approved her loan near the riverfront', does 'bank' refer to a financial institution?",
            "In 'The river bank collapsed after the storm', is 'bank' related to money?",
            "In 'She walked down to the bank before closing time', does 'bank' clearly mean a riverbank?",
        ]),
        ("S02_bat", [
            "In 'The wooden bat cracked on the final pitch', is 'bat' a flying mammal?",
            "In 'A dark bat darted out of the cave at dusk', is 'bat' a flying mammal?",
            "In 'He swung the bat silently in the barn', does 'bat' clearly denote sports equipment?",
            "In 'The bat hung upside down near the dugout', is 'bat' an animal?",
        ]),
        ("S03_crane", [
            "In 'The steel crane lifted girders above the street', is 'crane' a bird?",
            "In 'A white crane waded across the shallow marsh', is 'crane' a bird?",
            "In 'They spotted a tall crane beside the harbor dock', does 'crane' necessarily mean a bird?",
            "In 'The crane extended its long neck toward the nest', is 'crane' a machine?",
        ]),
        ("S04_spring", [
            "In 'Cold water bubbled from the mountain spring', does 'spring' refer to a season?",
            "In 'Blossoms returned early this spring', does 'spring' refer to a season?",
            "In 'The coiled spring snapped inside the clock', is 'spring' a water source?",
            "In 'They camped near the spring before April', does 'spring' clearly mean a metal coil?",
        ]),
        ("S05_pitch", [
            "In 'The violinist adjusted her pitch carefully', does 'pitch' refer to musical frequency?",
            "In 'The startup rehearsed its pitch for investors', does 'pitch' refer to musical frequency?",
            "In 'Black pitch sealed the wooden ship hull', is 'pitch' a sales presentation?",
            "In 'The striker sprinted across the wet pitch', does 'pitch' mean a sports field?",
        ]),
        ("S06_light", [
            "In 'Pack only a light suitcase for the flight', does 'light' mean low weight?",
            "In 'Bright morning light poured through the blinds', does 'light' mean low weight?",
            "In 'The lamp cast a soft light on the desk', is 'light' about illumination?",
            "In 'The fabric felt light in the afternoon sun', does 'light' clearly mean pale color rather than low weight?",
        ]),
        ("S07_charge", [
            "In 'The prosecutor dropped the felony charge', does 'charge' mean electrical quantity?",
            "In 'The battery held its charge overnight', does 'charge' mean electrical energy?",
            "In 'The hotel added a hidden service charge', is 'charge' a financial fee?",
            "In 'The cavalry sounded the charge at dawn', does 'charge' refer to a legal accusation?",
        ]),
        ("S08_seal", [
            "In 'A gray seal barked on the rocky pier', is 'seal' a marine mammal?",
            "In 'Check the rubber seal around the freezer door', is 'seal' a marine mammal?",
            "In 'The royal wax seal authenticated the treaty', is 'seal' an official stamp?",
            "In 'The diver noticed a broken seal near the hull', does 'seal' clearly mean a marine mammal?",
        ]),
        ("S09_match", [
            "In 'She struck a dry match to light the stove', is 'match' a sporting contest?",
            "In 'The tennis match lasted four intense hours', is 'match' a sporting contest?",
            "In 'The two paint swatches were an exact match', does 'match' mean a fire stick?",
            "In 'He watched the match flicker in the wind', does 'match' mean a small flame?",
        ]),
        ("S10_scale", [
            "In 'The fish shed a silver scale on the deck', does 'scale' mean a weighing device?",
            "In 'Place the flour on the kitchen scale', does 'scale' mean a weighing device?",
            "In 'The pianist practiced a minor scale', is 'scale' a sequence of musical notes?",
            "In 'The map uses a one-to-thousand scale', does 'scale' refer to a map ratio?",
        ]),
        ("S11_plant", [
            "In 'The assembly plant employs six hundred technicians', is 'plant' a living organism?",
            "In 'Water the potted plant once a week', is 'plant' a living organism?",
            "In 'Inspectors audited the power plant yesterday', is 'plant' an industrial facility?",
            "In 'The botanist catalogued a rare plant near the factory', is 'plant' a living organism?",
        ]),
        ("S12_anaphora", [
            "In 'The trophy did not fit in the suitcase because it was too big', does 'it' refer to the trophy?",
            "In 'The trophy did not fit in the suitcase because it was too small', does 'it' refer to the trophy?",
            "In 'The council refused the demonstrators a permit because they feared violence', does 'they' refer to the council?",
            "In 'The council refused the demonstrators a permit because they advocated violence', does 'they' refer to the council?",
        ]),
    ]

    all_cycles = []
    for cat, items in [
        ("neutral_factual", neutral_cycles),
        ("framing_order", framing_cycles),
        ("semantic_ambiguity", semantic_cycles),
    ]:
        for cid, qs in items:
            assert len(qs) == 4
            all_cycles.append({"id": cid, "category": cat, "questions": qs})
    assert len(all_cycles) == 36
    return all_cycles


def compute_cycle_cbd_metrics(
    pair_tables: list[dict[str, float]],
) -> dict[str, Any]:
    """Compute exact s_odd, Delta_0, and C_CbD = s_odd - Delta_0 for a 4-cycle.

    ``pair_tables[i]`` contains joint probabilities over (Q_i, Q_{i+1}) in context c_i:
      p_pp, p_pm, p_mp, p_mm (summing to 1.0).
    In context c_i:
      <Q_i>_{c_i}     = (p_pp + p_pm) - (p_mp + p_mm)
      <Q_{i+1}>_{c_i} = (p_pp + p_mp) - (p_pm + p_mm)
      <Q_i Q_{i+1}>   = (p_pp + p_mm) - (p_pm + p_mp)
    Marginal inconsistency Delta_0 = sum_{i=0}^3 |<Q_i>_{c_i} - <Q_i>_{c_{i-1}}|.
    """
    assert len(pair_tables) == 4
    corrs = []
    marg_first = []   # <Q_i> in c_i
    marg_second = []  # <Q_{i+1}> in c_i
    for tbl in pair_tables:
        p_pp, p_pm, p_mp, p_mm = tbl["p_pp"], tbl["p_pm"], tbl["p_mp"], tbl["p_mm"]
        assert abs((p_pp + p_pm + p_mp + p_mm) - 1.0) < 1e-9
        corrs.append((p_pp + p_mm) - (p_pm + p_mp))
        marg_first.append((p_pp + p_pm) - (p_mp + p_mm))
        marg_second.append((p_pp + p_mp) - (p_pm + p_mm))

    # For Q_i, its measurement in c_{i-1} is marg_second[(i - 1) % 4]
    delta_per_var = [abs(marg_first[i] - marg_second[(i - 1) % 4]) for i in range(4)]
    delta_0 = float(sum(delta_per_var))
    s_odd, best_signs = max_odd_signed_sum(corrs)
    c_cbd = float(s_odd - delta_0)
    return {
        "correlators": [float(c) for c in corrs],
        "s_odd": float(s_odd),
        "best_signs": best_signs,
        "delta_0": delta_0,
        "delta_per_var": [float(d) for d in delta_per_var],
        "c_cbd": c_cbd,
        "violates_cbd": bool(c_cbd > 2.0 + 1e-9),
        "exceeds_raw_chsh_2": bool(s_odd > 2.0 + 1e-9),
    }


def plot_figure_9(res: dict[str, Any], out_path: Path) -> None:
    """Render 4-panel publication-grade Figure 9."""
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.9), constrained_layout=True)
    fig.suptitle(
        "Experiment 9 — Real Qwen2.5-1.5B-Instruct (MLX): Isolated Bell-CHSH Bound vs. Multi-Turn KV-Cache Signaling & CbD",
        fontsize=12.1,
        fontweight="bold",
        y=1.02,
    )

    # Panel (a): Part A — Isolated Bipartite Bell-CHSH Across 50 Real LLM Scenarios & 1,000 Mixtures
    ax = axes[0, 0]
    part_a = res["part_a_isolated_chsh"]
    s_ind = np.array(part_a["individual_scenario_s_odd"])
    s_mix = np.array(part_a["random_mixture_s_odd"])

    bins_a = np.linspace(0.0, 2.95, 36)
    ax.hist(
        s_mix, bins=bins_a, density=True, color=PALETTE["navy"], alpha=0.65, edgecolor="white", lw=0.5,
        label=f"1,000 Mixtures of 50 LLM Contexts (max={part_a['max_mixture_s_odd']:.4f})",
    )
    ax.hist(
        s_ind, bins=bins_a, density=True, color=PALETTE["teal"], alpha=0.55, edgecolor="white", lw=0.5,
        label=f"50 Individual Isolated Scenarios (max={part_a['max_individual_s_odd']:.4f})",
    )
    ax.axvline(2.0, color="#0F172A", ls="--", lw=1.6, label="Classical Bell-CHSH Ceiling ($|S|=2.0$)")
    ax.axvline(2.0 * math.sqrt(2.0), color=PALETTE["coral"], ls=":", lw=1.6, label=r"Quantum Tsirelson Bound ($2\sqrt{2}\approx 2.828$)")
    ax.axvspan(2.0, 2.95, color=PALETTE["coral"], alpha=0.09)

    ax.text(
        2.44,
        ax.get_ylim()[1] * 0.72,
        "Quantum\nViolation Zone\n(0 / 1,050\nLLM trials)",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=PALETTE["coral"],
    )
    ax.set_title(r"(a) Part A: Isolated Bipartite CHSH ($\eta = 1$, Zero Context Sharing)")
    ax.set_xlabel(r"Odd-Parity CHSH Functional $s_{\mathrm{odd}}$")
    ax.set_ylabel("Empirical Density")
    ax.set_xlim(0.0, 2.95)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7.6)

    # Panel (b): Part B — KV-Cache Signaling Delta_0 vs s_odd Scatter Across 36 Multi-Turn Cycles
    ax = axes[0, 1]
    cycles = res["part_b_multiturn_cbd"]["cycles"]
    cat_styles = {
        "neutral_factual": ("Neutral Factual (12)", PALETTE["teal"], "o"),
        "framing_order": ("Psychological Framing (12)", PALETTE["coral"], "s"),
        "semantic_ambiguity": ("Semantic Ambiguity (12)", PALETTE["purple"], "^"),
    }

    max_d0 = max(max(c["forward"]["delta_0"] for c in cycles), 0.5)
    d_line = np.linspace(0.0, max_d0 * 1.15 + 0.15, 100)
    # Non-contextuality boundary: s_odd = 2 + Delta_0
    ax.plot(d_line, 2.0 + d_line, color="#0F172A", ls="--", lw=1.5, label=r"CbD Boundary $s_{\mathrm{odd}} = 2 + \Delta_0$")
    ax.axhline(2.0, color=PALETTE["slate"], ls=":", lw=1.2, label=r"Naive CHSH Line $s_{\mathrm{odd}} = 2$ (ignoring $\Delta_0$)")
    ax.fill_between(d_line, 2.0 + d_line, 4.15, color=PALETTE["coral"], alpha=0.09, label=r"CbD-Contextual Zone ($s_{\mathrm{odd}} - \Delta_0 > 2$)")

    for cat_key, (cat_lbl, col, mkr) in cat_styles.items():
        pts = [c for c in cycles if c["category"] == cat_key]
        xs = [c["forward"]["delta_0"] for c in pts]
        ys = [c["forward"]["s_odd"] for c in pts]
        ax.scatter(xs, ys, color=col, marker=mkr, s=56, alpha=0.88, edgecolor="white", lw=0.6, label=cat_lbl, zorder=4)

    ax.set_title(r"(b) Part B: Multi-Turn Directed Cycles — $s_{\mathrm{odd}}$ vs. KV-Cache Signaling $\Delta_0$")
    ax.set_xlabel(r"KV-Cache Marginal Inconsistency (Signaling) $\Delta_0$")
    ax.set_ylabel(r"Cyclic Correlation Functional $s_{\mathrm{odd}}$")
    ax.set_xlim(-0.04, max_d0 * 1.12 + 0.12)
    ax.set_ylim(0.5, 4.10)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel (c): Category Comparison of Delta_0, |q_QQ|, Order-Effect TV, and C_CbD
    ax = axes[1, 0]
    cat_summary = res["part_b_multiturn_cbd"]["by_category"]
    cat_order = ["neutral_factual", "framing_order", "semantic_ambiguity"]
    cat_short = ["Neutral\nFactual", "Psychological\nFraming", "Semantic\nAmbiguity"]

    mean_d0 = [cat_summary[k]["forward_delta_0_mean"] for k in cat_order]
    mean_ord = [cat_summary[k]["pair_order_tv_mean"] for k in cat_order]
    mean_qq = [cat_summary[k]["pair_abs_q_qq_mean"] for k in cat_order]
    x_c = np.arange(len(cat_order))
    w = 0.25

    b1 = ax.bar(x_c - w, mean_d0, width=w, color=PALETTE["coral"], edgecolor="#1E293B", lw=0.7, label=r"Cycle Signaling $\overline{\Delta}_0$")
    b2 = ax.bar(x_c, mean_ord, width=w, color=PALETTE["navy"], edgecolor="#1E293B", lw=0.7, label=r"Pair Order TV $\overline{\delta}_{\mathrm{ord}}$")
    b3 = ax.bar(x_c + w, mean_qq, width=w, color=PALETTE["amber"], edgecolor="#1E293B", lw=0.7, label=r"QQ Residual $\overline{|q_{\mathrm{QQ}}|}$")

    for bars in (b1, b2, b3):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    ax.set_xticks(x_c)
    ax.set_xticklabels(cat_short)
    ax.set_ylabel("Mean Magnitude")
    ax.set_title("(c) Signaling $\\Delta_0$, Order Effects $\\delta_{\\mathrm{ord}}$, and QQ-Equality $|q_{\\mathrm{QQ}}|$ by Category")
    ax.set_ylim(0.0, max(max(mean_d0), max(mean_ord), max(mean_qq)) * 1.38 + 0.08)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left", fontsize=7.9)

    # Panel (d): Distribution of Exact CbD Witness C_CbD = s_odd - Delta_0 Across Forward / Reverse / Symmetrized
    ax = axes[1, 1]
    cbd_fwd = [c["forward"]["c_cbd"] for c in cycles]
    cbd_rev = [c["reverse"]["c_cbd"] for c in cycles]
    cbd_sym = [c["symmetrized"]["c_cbd"] for c in cycles]
    raw_fwd = [c["forward"]["s_odd"] for c in cycles]

    box_data = [raw_fwd, cbd_fwd, cbd_rev, cbd_sym]
    box_labels = [
        "Raw $s_{\\mathrm{odd}}$\n(Forward)",
        "$C_{\\mathrm{CbD}}$\n(Forward)",
        "$C_{\\mathrm{CbD}}$\n(Reverse)",
        "$C_{\\mathrm{CbD}}$\n(Symmetrized)",
    ]
    bp = ax.boxplot(
        box_data,
        tick_labels=box_labels,
        patch_artist=True,
        widths=0.48,
        medianprops=dict(color="#0F172A", linewidth=1.8),
        whiskerprops=dict(color="#475569", linewidth=1.1),
        capprops=dict(color="#475569", linewidth=1.1),
        flierprops=dict(marker="o", markersize=4.5, markerfacecolor=PALETTE["coral"], alpha=0.7),
    )
    box_colors = [PALETTE["light_gray"], PALETTE["navy"], PALETTE["purple"], PALETTE["teal"]]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.72)

    ax.axhline(2.0, color=PALETTE["coral"], ls="--", lw=1.5, label="Non-Contextuality Ceiling ($n - 2 = 2.0$)")
    ax.set_title(r"(d) Exact CbD Witness $C_{\mathrm{CbD}} = s_{\mathrm{odd}} - \Delta_0$ Across All 36 Cycles")
    ax.set_ylabel("Statistic Value")
    all_vals = raw_fwd + cbd_fwd + cbd_rev + cbd_sym
    ax.set_ylim(min(all_vals) - 0.35, max(2.55, max(all_vals) + 0.55))
    ax.grid(True, axis="y")
    ax.legend(loc="lower left", fontsize=7.9)

    ov = res["part_b_multiturn_cbd"]["overall"]
    summary_txt = (
        "36-Cycle CbD Audit (Forward / Rev / Sym):\n"
        f"• Raw s_odd > 2 : {ov['forward_raw_gt_2']}/36 | {ov['reverse_raw_gt_2']}/36 | {ov['symmetrized_raw_gt_2']}/36\n"
        f"• CbD C > 2     : {ov['forward_cbd_violations']}/36 | {ov['reverse_cbd_violations']}/36 | {ov['symmetrized_cbd_violations']}/36\n"
        f"• Max C_CbD     : {ov['forward_c_cbd_max']:.3f} / {ov['reverse_c_cbd_max']:.3f} / {ov['symmetrized_c_cbd_max']:.3f}\n"
        f"• ||Δθ*||_2 = 0.0 (SHA-256 verified)"
    )
    ax.text(
        0.97,
        0.97,
        summary_txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.38", facecolor=PALETTE["bg_tint"], edgecolor="#94A3B8", lw=0.8),
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    rng = np.random.default_rng(GLOBAL_SEED)

    print("=" * 78)
    print("EXPERIMENT 9 — REAL LLM BELL-CHSH & MULTI-TURN CbD / SIGNALING (H2.1 & H2)")
    print(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"MLX      : {mx.__version__} ({mx.default_device()}) | mlx_lm {mlx_lm.__version__}")
    print(f"Model    : {MODEL_ID}")
    print("=" * 78)

    model, tok = mlx_lm.load(MODEL_ID)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    assert yes_id != no_id

    weights_before = fingerprint_frozen_weights(model)
    print(f"[Audit] Weights BEFORE: sha256={weights_before['sha256'][:16]}... L2={weights_before['l2_norm']:.4f}")

    # -----------------------------------------------------------------------
    # Part A: Isolated Bipartite Bell-CHSH over 50 Shared Classical Scenarios
    # -----------------------------------------------------------------------
    scenarios = build_50_chsh_scenarios()
    convs_a = []
    for sc in scenarios:
        for q_key in ("a0", "a1", "b0", "b1"):
            convs_a.append(
                [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": f"Context: {sc['context']}\nQuestion: {sc[q_key]}"},
                ]
            )
    probs_a_flat, mass_a_flat = batch_eval_yes_probs(model, tok, convs_a, yes_id, no_id, batch_size=50)
    # Reshape to (50, 4): columns [p_a0, p_a1, p_b0, p_b1] -> expectations in [-1, +1]
    exp_a = 2.0 * probs_a_flat.reshape(50, 4) - 1.0
    ea0, ea1, eb0, eb1 = exp_a[:, 0], exp_a[:, 1], exp_a[:, 2], exp_a[:, 3]

    # Exact per-scenario isolated CHSH correlators E(x, y | lambda) = <A>_{x,lambda} <B>_{y,lambda}
    e_scen = np.stack([ea0 * eb0, ea0 * eb1, ea1 * eb0, ea1 * eb1], axis=1)  # (50, 4)
    indiv_s_odd = [max_odd_signed_sum(row)[0] for row in e_scen]
    indiv_s_std = np.abs(e_scen[:, 0] + e_scen[:, 1] + e_scen[:, 2] - e_scen[:, 3])

    # Uniform mixture across all 50 scenarios
    e_uniform = np.mean(e_scen, axis=0)
    uniform_s_odd, uniform_signs = max_odd_signed_sum(e_uniform)

    # 1,000 random & near-extremal classical mixtures p(lambda) over the 50 LLM scenarios
    w_dense = rng.dirichlet(np.ones(50), size=500)
    w_sparse = rng.dirichlet(np.full(50, 0.08), size=500)
    w_all = np.vstack([w_dense, w_sparse])  # (1000, 50)
    e_mix = w_all @ e_scen  # (1000, 4)
    mix_s_odd = [max_odd_signed_sum(row)[0] for row in e_mix]

    max_indiv = float(np.max(indiv_s_odd))
    max_mix = float(np.max(mix_s_odd))
    assert max_indiv <= 2.0 + 1e-9, f"Impossible isolated CHSH violation: {max_indiv}"
    assert max_mix <= 2.0 + 1e-9, f"Impossible mixture CHSH violation: {max_mix}"

    print(
        f"[Part A] Isolated Bell-CHSH (50 real scenarios, eta=1.0): "
        f"uniform s_odd={uniform_s_odd:.4f} | max indiv={max_indiv:.6f} | "
        f"max 1000 mixtures={max_mix:.6f} (ALL <= 2.000000, mass={np.mean(mass_a_flat)*100:.1f}%)"
    )

    # -----------------------------------------------------------------------
    # Part B: 36 Rank-4 Multi-Turn Cycles (Exact Joint Tables, Zero MC Error)
    # -----------------------------------------------------------------------
    cycles = build_36_cycles()

    # Collect all unique first-turn prompts (36 * 4 = 144) and second-turn prompts
    # Each cycle has 4 cyclic pairs (q_i, q_{i+1}) in 2 orders x 2 first answers = 16 second-turn calls
    # Total second-turn calls = 36 * 16 = 576; total calls = 144 + 576 = 720
    turn1_convs = []
    turn1_index: dict[tuple[int, int], int] = {}
    for ci, cyc in enumerate(cycles):
        for qi in range(4):
            turn1_index[(ci, qi)] = len(turn1_convs)
            turn1_convs.append(
                [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": cyc["questions"][qi]},
                ]
            )

    turn2_convs = []
    turn2_index: dict[tuple[int, int, int, int], int] = {}
    # key: (cycle_idx, q_first_idx, first_ans_pm1, q_second_idx)
    for ci, cyc in enumerate(cycles):
        qs = cyc["questions"]
        for edge in range(4):
            qa_i = edge
            qb_i = (edge + 1) % 4
            for first_i, second_i in [(qa_i, qb_i), (qb_i, qa_i)]:
                for ans_val, ans_str in [(1, "Yes"), (-1, "No")]:
                    turn2_index[(ci, first_i, ans_val, second_i)] = len(turn2_convs)
                    turn2_convs.append(
                        [
                            {"role": "system", "content": SYS_PROMPT},
                            {"role": "user", "content": qs[first_i]},
                            {"role": "assistant", "content": ans_str},
                            {"role": "user", "content": qs[second_i]},
                        ]
                    )

    t_b = time.perf_counter()
    p1_all, m1_all = batch_eval_yes_probs(model, tok, turn1_convs, yes_id, no_id, batch_size=48)
    p2_all, m2_all = batch_eval_yes_probs(model, tok, turn2_convs, yes_id, no_id, batch_size=48)
    print(
        f"[Part B] Evaluated {len(turn1_convs)} turn-1 + {len(turn2_convs)} turn-2 exact logit calls "
        f"in {time.perf_counter() - t_b:.2f}s (Yes+No mass={np.mean(np.concatenate([m1_all, m2_all]))*100:.1f}%)"
    )

    def ordered_joint_table(ci: int, first_i: int, second_i: int) -> dict[str, float]:
        """Exact joint table P_{first -> second}(first_val, second_val) over {+1, -1}^2."""
        p_a_pos = float(p1_all[turn1_index[(ci, first_i)]])
        p_a_neg = 1.0 - p_a_pos
        p_b_pos_given_apos = float(p2_all[turn2_index[(ci, first_i, 1, second_i)]])
        p_b_pos_given_aneg = float(p2_all[turn2_index[(ci, first_i, -1, second_i)]])
        return {
            "p_pp": p_a_pos * p_b_pos_given_apos,
            "p_pm": p_a_pos * (1.0 - p_b_pos_given_apos),
            "p_mp": p_a_neg * p_b_pos_given_aneg,
            "p_mm": p_a_neg * (1.0 - p_b_pos_given_aneg),
        }

    cycle_results = []
    all_pair_qq = []
    all_pair_order_tv = []

    for ci, cyc in enumerate(cycles):
        fwd_tables = []
        rev_tables = []
        sym_tables = []
        pair_details = []

        for edge in range(4):
            qa_i = edge
            qb_i = (edge + 1) % 4
            # Forward order: Q_edge -> Q_{edge+1}
            t_ab = ordered_joint_table(ci, qa_i, qb_i)
            # Reverse order: Q_{edge+1} -> Q_edge, re-indexed to (Q_edge, Q_{edge+1})!
            t_ba_raw = ordered_joint_table(ci, qb_i, qa_i)
            t_ba_aligned = {
                "p_pp": t_ba_raw["p_pp"],
                "p_pm": t_ba_raw["p_mp"],  # Q_edge=+1, Q_{edge+1}=-1
                "p_mp": t_ba_raw["p_pm"],  # Q_edge=-1, Q_{edge+1}=+1
                "p_mm": t_ba_raw["p_mm"],
            }
            t_sym = {k: 0.5 * (t_ab[k] + t_ba_aligned[k]) for k in ("p_pp", "p_pm", "p_mp", "p_mm")}

            # QQ-equality residual: [P_AB(+,+) + P_AB(-,-)] - [P_BA(+,+) + P_BA(-,-)]
            q_qq = (t_ab["p_pp"] + t_ab["p_mm"]) - (t_ba_raw["p_pp"] + t_ba_raw["p_mm"])
            # Total variation distance between P_{A->B}(a, b) and P_{B->A}(a, b) (aligned)
            order_tv = 0.5 * sum(abs(t_ab[k] - t_ba_aligned[k]) for k in ("p_pp", "p_pm", "p_mp", "p_mm"))

            fwd_tables.append(t_ab)
            rev_tables.append(t_ba_aligned)
            sym_tables.append(t_sym)
            all_pair_qq.append(float(q_qq))
            all_pair_order_tv.append(float(order_tv))
            pair_details.append(
                {
                    "edge": [qa_i, qb_i],
                    "q_qq": float(q_qq),
                    "abs_q_qq": float(abs(q_qq)),
                    "order_tv": float(order_tv),
                }
            )

        fwd_metrics = compute_cycle_cbd_metrics(fwd_tables)
        rev_metrics = compute_cycle_cbd_metrics(rev_tables)
        sym_metrics = compute_cycle_cbd_metrics(sym_tables)

        cycle_results.append(
            {
                "id": cyc["id"],
                "category": cyc["category"],
                "forward": fwd_metrics,
                "reverse": rev_metrics,
                "symmetrized": sym_metrics,
                "pairs": pair_details,
                "mean_abs_q_qq": float(np.mean([p["abs_q_qq"] for p in pair_details])),
                "mean_order_tv": float(np.mean([p["order_tv"] for p in pair_details])),
            }
        )

    # Aggregate statistics by category and overall
    by_category = {}
    for cat in ("neutral_factual", "framing_order", "semantic_ambiguity"):
        sub = [c for c in cycle_results if c["category"] == cat]
        pair_qq_cat = [p["abs_q_qq"] for c in sub for p in c["pairs"]]
        pair_tv_cat = [p["order_tv"] for c in sub for p in c["pairs"]]
        by_category[cat] = {
            "n_cycles": len(sub),
            "forward_s_odd_mean": float(np.mean([c["forward"]["s_odd"] for c in sub])),
            "forward_delta_0_mean": float(np.mean([c["forward"]["delta_0"] for c in sub])),
            "forward_delta_0_max": float(np.max([c["forward"]["delta_0"] for c in sub])),
            "forward_c_cbd_mean": float(np.mean([c["forward"]["c_cbd"] for c in sub])),
            "forward_c_cbd_max": float(np.max([c["forward"]["c_cbd"] for c in sub])),
            "forward_raw_gt_2": int(sum(c["forward"]["exceeds_raw_chsh_2"] for c in sub)),
            "forward_cbd_violations": int(sum(c["forward"]["violates_cbd"] for c in sub)),
            "reverse_delta_0_mean": float(np.mean([c["reverse"]["delta_0"] for c in sub])),
            "reverse_c_cbd_max": float(np.max([c["reverse"]["c_cbd"] for c in sub])),
            "reverse_cbd_violations": int(sum(c["reverse"]["violates_cbd"] for c in sub)),
            "symmetrized_delta_0_mean": float(np.mean([c["symmetrized"]["delta_0"] for c in sub])),
            "symmetrized_c_cbd_max": float(np.max([c["symmetrized"]["c_cbd"] for c in sub])),
            "symmetrized_cbd_violations": int(sum(c["symmetrized"]["violates_cbd"] for c in sub)),
            "pair_order_tv_mean": float(np.mean(pair_tv_cat)),
            "pair_order_tv_max": float(np.max(pair_tv_cat)),
            "pair_abs_q_qq_mean": float(np.mean(pair_qq_cat)),
            "pair_abs_q_qq_max": float(np.max(pair_qq_cat)),
        }

    overall = {
        "n_cycles": len(cycle_results),
        "n_unique_pairs": len(all_pair_qq),
        "forward_s_odd_mean": float(np.mean([c["forward"]["s_odd"] for c in cycle_results])),
        "forward_s_odd_max": float(np.max([c["forward"]["s_odd"] for c in cycle_results])),
        "forward_delta_0_mean": float(np.mean([c["forward"]["delta_0"] for c in cycle_results])),
        "forward_delta_0_max": float(np.max([c["forward"]["delta_0"] for c in cycle_results])),
        "forward_c_cbd_mean": float(np.mean([c["forward"]["c_cbd"] for c in cycle_results])),
        "forward_c_cbd_max": float(np.max([c["forward"]["c_cbd"] for c in cycle_results])),
        "forward_raw_gt_2": int(sum(c["forward"]["exceeds_raw_chsh_2"] for c in cycle_results)),
        "forward_cbd_violations": int(sum(c["forward"]["violates_cbd"] for c in cycle_results)),
        "reverse_raw_gt_2": int(sum(c["reverse"]["exceeds_raw_chsh_2"] for c in cycle_results)),
        "reverse_c_cbd_max": float(np.max([c["reverse"]["c_cbd"] for c in cycle_results])),
        "reverse_cbd_violations": int(sum(c["reverse"]["violates_cbd"] for c in cycle_results)),
        "symmetrized_raw_gt_2": int(sum(c["symmetrized"]["exceeds_raw_chsh_2"] for c in cycle_results)),
        "symmetrized_c_cbd_max": float(np.max([c["symmetrized"]["c_cbd"] for c in cycle_results])),
        "symmetrized_cbd_violations": int(sum(c["symmetrized"]["violates_cbd"] for c in cycle_results)),
        "pair_order_tv_mean": float(np.mean(all_pair_order_tv)),
        "pair_order_tv_max": float(np.max(all_pair_order_tv)),
        "pair_abs_q_qq_mean": float(np.mean(np.abs(all_pair_qq))),
        "pair_abs_q_qq_max": float(np.max(np.abs(all_pair_qq))),
        "pair_q_qq_gt_0_05_count": int(sum(abs(x) > 0.05 for x in all_pair_qq)),
        "pair_order_tv_gt_0_05_count": int(sum(x > 0.05 for x in all_pair_order_tv)),
    }

    # Mann-Whitney U test: does framing/semantic signaling Delta_0 exceed neutral factual control?
    d0_neutral = [c["forward"]["delta_0"] for c in cycle_results if c["category"] == "neutral_factual"]
    d0_framing = [c["forward"]["delta_0"] for c in cycle_results if c["category"] == "framing_order"]
    d0_semantic = [c["forward"]["delta_0"] for c in cycle_results if c["category"] == "semantic_ambiguity"]
    u_fr, p_fr = stats.mannwhitneyu(d0_framing, d0_neutral, alternative="greater")
    u_sm, p_sm = stats.mannwhitneyu(d0_semantic, d0_neutral, alternative="greater")
    overall["mwu_framing_vs_neutral_delta0_p"] = float(p_fr)
    overall["mwu_semantic_vs_neutral_delta0_p"] = float(p_sm)

    # Exact frozen-weight audit AFTER
    weights_after = fingerprint_frozen_weights(model)
    weights_sha_match = weights_before["sha256"] == weights_after["sha256"]
    weights_l2_diff = abs(weights_before["l2_norm"] - weights_after["l2_norm"])
    assert weights_sha_match and weights_l2_diff == 0.0, "Frozen weights were mutated!"
    print(f"[Audit] Weights AFTER : sha256={weights_after['sha256'][:16]}... MATCH={weights_sha_match}, ||dtheta*||=0.0")

    elapsed_total = time.perf_counter() - t0
    results = {
        "experiment": "exp9_real_llm_cbd_signaling",
        "model_id": MODEL_ID,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "seed": GLOBAL_SEED,
        "yes_token_id": int(yes_id),
        "no_token_id": int(no_id),
        "weights_before": weights_before,
        "weights_after": weights_after,
        "weights_sha256_match": weights_sha_match,
        "weights_l2_diff": weights_l2_diff,
        "part_a_isolated_chsh": {
            "n_scenarios": 50,
            "n_random_mixtures": 1000,
            "detection_efficiency_eta": 1.0,
            "mean_yes_no_mass": float(np.mean(mass_a_flat)),
            "uniform_aggregate_correlators": [float(x) for x in e_uniform],
            "uniform_aggregate_s_odd": float(uniform_s_odd),
            "uniform_aggregate_signs": uniform_signs,
            "max_individual_s_odd": max_indiv,
            "max_individual_s_std": float(np.max(indiv_s_std)),
            "max_mixture_s_odd": max_mix,
            "violations_gt_2_count": int(sum(x > 2.0 + 1e-9 for x in indiv_s_odd) + sum(x > 2.0 + 1e-9 for x in mix_s_odd)),
            "individual_scenario_s_odd": [float(x) for x in indiv_s_odd],
            "random_mixture_s_odd": [float(x) for x in mix_s_odd],
        },
        "part_b_multiturn_cbd": {
            "total_turn1_calls": len(turn1_convs),
            "total_turn2_calls": len(turn2_convs),
            "mean_yes_no_mass": float(np.mean(np.concatenate([m1_all, m2_all]))),
            "overall": overall,
            "by_category": by_category,
            "cycles": cycle_results,
        },
        "runtime_seconds": float(elapsed_total),
    }

    json_path = RESULTS_DIR / "exp9_llm_cbd_signaling.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig_path = FIGURES_DIR / "fig9_real_llm_cbd_signaling.png"
    plot_figure_9(results, fig_path)

    print("\n" + "=" * 78)
    print("SUMMARY — EXPERIMENT 9 (REAL Qwen2.5-1.5B-Instruct-4bit)")
    print(f"  Part A Isolated CHSH : max indiv={max_indiv:.6f}, max 1000 mix={max_mix:.6f}, violations={results['part_a_isolated_chsh']['violations_gt_2_count']}/1050")
    print("  Part B Multi-Turn CbD by Category (Forward cycle):")
    for cat in ("neutral_factual", "framing_order", "semantic_ambiguity"):
        s = by_category[cat]
        print(
            f"    {cat:19s} | s_odd={s['forward_s_odd_mean']:.3f} | Delta_0={s['forward_delta_0_mean']:.3f} (max {s['forward_delta_0_max']:.3f}) | "
            f"C_CbD max={s['forward_c_cbd_max']:.3f} | raw>2: {s['forward_raw_gt_2']}/12 | CbD>2 (F/R/S): "
            f"{s['forward_cbd_violations']}/{s['reverse_cbd_violations']}/{s['symmetrized_cbd_violations']} | "
            f"ord_TV={s['pair_order_tv_mean']:.3f} | |q_QQ|={s['pair_abs_q_qq_mean']:.3f}"
        )
    print(
        f"  Overall 36 cycles    : raw s_odd>2 (F/R/S) = {overall['forward_raw_gt_2']}/{overall['reverse_raw_gt_2']}/{overall['symmetrized_raw_gt_2']} | "
        f"CbD>2 (F/R/S) = {overall['forward_cbd_violations']}/{overall['reverse_cbd_violations']}/{overall['symmetrized_cbd_violations']}"
    )
    print(
        f"  Mann-Whitney Delta_0 : framing>neutral p={overall['mwu_framing_vs_neutral_delta0_p']:.3e} | "
        f"semantic>neutral p={overall['mwu_semantic_vs_neutral_delta0_p']:.3e}"
    )
    print(f"  Saved JSON           -> {json_path}")
    print(f"  Saved Figure         -> {fig_path}")
    print(f"ALL ASSERTIONS PASSED (EXIT 0) in {elapsed_total:.2f} s.")
    print("=" * 78)


if __name__ == "__main__":
    main()
