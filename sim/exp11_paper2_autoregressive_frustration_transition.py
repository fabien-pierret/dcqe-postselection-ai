#!/usr/bin/env python3
"""Experiment 11 — Live Real-LLM Empirical Test for Paper 2: Autoregressive
Semantic Frustration Phase Transition, Real-Logit Detection Loophole, and
Spectral Effective Multiplicity M_eff(Sigma_prompt).

Runs three original deterministic experiments directly on the locally cached
Apple Silicon MLX model ``mlx-community/Qwen2.5-1.5B-Instruct-4bit`` (Metal GPU,
exact next-token Yes/No logits, verified SHA-256 weight invariance ||Delta theta*|| = 0):

Part A — Constructive Semantic Frustration Phase Transition (Paper 2 Theorems 1 & 2):
  In Exp 9, 36 standard multi-turn cycles yielded 27/36 raw violations (s_odd > 2,
  up to 3.487) yet 0/36 Contextuality-by-Default (CbD) violations (max C_CbD = 1.9997)
  because KV-cache signaling Delta_0 absorbed the excess.
  Here we construct 30 new rank-4 cycles split into two distinct tiers (15 each):
    - Tier 1 (15 Context-Coupled Mobius-Frustrated Cycles): 4 fixed balanced
      ambiguous questions Q_1..Q_4 queried under 3 positive-consistency contexts
      c_1, c_2, c_3 (B ~ +A) and 1 explicit counterbalance/devil's-advocate
      context c_4 (B ~ -A), satisfying both odd parity frustration and near-zero
      Turn-1 marginals |m_i|.
    - Tier 2 (15 Natural Non-Transitive / Polysemous Cycles, Single Fixed System
      Prompt): 4 naturally coupled questions crafted to probe organic non-transitive
      contrast without changing the system prompt across contexts.
  Directly compares (s_odd, Delta_0, C_CbD = s_odd - Delta_0) across Exp 9 (36
  standard cycles), Tier 2 (15 natural cycles), and Tier 1 (15 Mobius-frustrated
  cycles) in forward, reverse, and order-symmetrized conventions.

Part B — Real-LLM Detection Loophole via Confidence/Refusal Rejection Sampling:
  Evaluates isolated bipartite real Qwen2.5-1.5B responses across K = 80 diverse
  classical context scenarios lambda_k (eta = 1.0 strictly verifies |S_CHSH| <= 2.0).
  Applies both (i) symmetric low-confidence abstention filtering and (ii) setting-
  dependent refusal/rejection sampling across acceptance rates eta in [0.55, 1.00].
  Demonstrates directly on real LLM logits how post-selection on accepted pairs
  creates spurious CHSH violations above 2.0 (even crossing Tsirelson 2*sqrt(2)),
  while loophole-free deterministic imputation (refusal -> +1) restores |S| <= 2.0.

Part C — Spectral Validation of Correlated Prompt Multiplicity M_eff(Sigma_prompt)
  (Paper 2 Theorem 3):
  Evaluates M = 80 real candidate prompts on Qwen2.5-1.5B across 2 distinct
  correlation regimes (high-correlation neutral null items and lower-correlation
  ambiguous/borderline null items, 96 items each -> 24 val / 72 test over 300
  balanced resamples), computes the full exact (80 x 80) prompt correlation
  spectrum {lambda_i}, and compares 4 analytical M_eff estimators:
    - Participation Ratio (PR):    M_PR     = (sum lambda_i)^2 / sum lambda_i^2
    - Galwey square-root rank:     M_Galwey = (sum sqrt(lambda_i))^2 / sum lambda_i
    - Nyholt-Cheverud variance:    M_Nyholt = 1 + (M - 1) * (1 - Var(lambda) / M)
    - Naive uncorrelated count:    M_naive  = M
  against the exact empirical optimism gap Acc_val(p*) - Acc_test(p*) on Qwen2.5-1.5B.

Outputs:
- ``sim/results/exp11_paper2_results.json``
- ``sim/figures/fig11_autoregressive_frustration_transition.png`` (300 DPI, 4 panels)

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
from mlx_lm.models.cache import KVCache, make_prompt_cache
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

SYS_UNIFIED = "Reply strictly with a single word: Yes or No."
SYS_NATURAL = (
    "You are answering a short binary survey. Answer each question strictly with a single "
    "word: Yes or No. Do not explain."
)

# Turn-2 contextual coupling templates for Tier 1 constructive Mobius-frustrated cycles.
# Note: Turn 1 uses the exact bare question Q_i under the unified system prompt (zero Turn-1
# contamination across contexts); Turn 2 embeds the exact question Q_j together with the
# context's alignment (+) or counterbalance (-) conversational rule.
T2_ALIGN = (
    "Your previous reply determines your answer to '{q}': repeat your previous reply."
)
T2_INVERT = (
    "Context (counterbalance mode): Flip your previous Yes/No reply above to the opposite word as your verdict on: {q}"
)


def configure_matplotlib() -> None:
    """Configure publication-grade Matplotlib styling."""
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
            "legend.fontsize": 7.9,
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


def compute_cycle_cbd_metrics(pair_tables: list[dict[str, float]]) -> dict[str, Any]:
    """Compute exact s_odd, Delta_0, and C_CbD = s_odd - Delta_0 for a 4-cycle."""
    corrs = []
    marg_first = []
    marg_second = []
    for tbl in pair_tables:
        p_pp, p_pm, p_mp, p_mm = tbl["p_pp"], tbl["p_pm"], tbl["p_mp"], tbl["p_mm"]
        assert abs((p_pp + p_pm + p_mp + p_mm) - 1.0) < 1e-9
        corrs.append((p_pp + p_mm) - (p_pm + p_mp))
        marg_first.append((p_pp + p_pm) - (p_mp + p_mm))
        marg_second.append((p_pp + p_mp) - (p_pm + p_mm))

    delta_per_var = [abs(marg_first[i] - marg_second[(i - 1) % 4]) for i in range(4)]
    delta_0 = float(sum(delta_per_var))
    s_odd, best_signs = max_odd_signed_sum(corrs)
    c_cbd = float(s_odd - delta_0)
    return {
        "correlators": [float(c) for c in corrs],
        "turn1_marginals": [float(m) for m in marg_first],
        "s_odd": float(s_odd),
        "best_signs": best_signs,
        "delta_0": delta_0,
        "c_cbd": c_cbd,
        "violates_cbd": bool(c_cbd > 2.0 + 1e-9),
        "exceeds_raw_chsh_2": bool(s_odd > 2.0 + 1e-9),
    }


# ---------------------------------------------------------------------------
# Part A: 30 New Rank-4 Frustration Cycles (15 Tier-1 Mobius + 15 Tier-2 Natural)
# ---------------------------------------------------------------------------
def build_30_frustration_cycles() -> list[dict[str, Any]]:
    """Build 15 constructive Mobius-frustrated cycles + 15 natural non-transitive cycles."""
    # Tier 1: 15 Constructive Mobius Parity-Frustrated Cycles (3 alignment edges + 1 inversion edge)
    # Mix of genuinely balanced open-ended opinion/policy questions spanning low to moderate Turn-1
    # polarization so the empirical phase boundary C_CbD(Delta_0, |m|) is directly observable!
    tier1_specs = [
        ("M01_civic_everyday", [
            "Is working from home better than working in an office?",
            "Should movie theaters have a brief intermission?",
            "Should cities allow electric scooters on sidewalks?",
            "Should public libraries remain open on Sundays?",
        ]),
        ("M02_work_balance", [
            "Should remote workers follow fixed core hours instead of fully flexible schedules?",
            "Should companies adopt a four-day workweek with longer daily shifts?",
            "Should annual performance reviews include anonymous peer evaluations?",
            "Should office dress codes remain strictly casual all week?",
        ]),
        ("M03_urban_life", [
            "Should coffee shops play background music during morning hours?",
            "Should municipal parks allow dogs off-leash during early mornings?",
            "Should downtown avenues reserve one lane exclusively for cyclists?",
            "Should city buses run 24 hours a day on weekends?",
        ]),
        ("M04_education_norms", [
            "Should universities record and publish all undergraduate lectures online?",
            "Should high schools begin classes at 9 AM instead of 8 AM?",
            "Should college admissions weigh portfolios equally with standardized exams?",
            "Should public schools offer mandatory personal finance courses?",
        ]),
        ("M05_tech_society", [
            "Should streaming platforms release full seasons at once rather than weekly?",
            "Should websites require explicit age verification before creating an account?",
            "Should smartphones include user-replaceable batteries by regulation?",
            "Should autonomous delivery robots be allowed on neighborhood sidewalks?",
        ]),
        ("M06_culture_media", [
            "Should museums charge small admission fees to fund longer evening hours?",
            "Should book clubs choose contemporary fiction over classic novels?",
            "Should live concerts discourage audiences from filming on phones?",
            "Should cinemas show trailers for fewer than ten minutes before films?",
        ]),
        ("M07_transport_policy", [
            "Should regional trains offer designated silent carriages on every route?",
            "Should airlines board window-seat passengers before aisle-seat passengers?",
            "Should highway tolls vary dynamically with rush-hour traffic levels?",
            "Should suburban commuter parking lots be free for rail pass holders?",
        ]),
        ("M08_food_dining", [
            "Should neighborhood cafes limit laptop use during peak lunch hours?",
            "Should grocery stores phase out printed paper receipts by default?",
            "Should restaurants display calorie counts next to every menu item?",
            "Should farmers markets operate year-round inside covered halls?",
        ]),
        ("M09_science_open", [
            "Should scientific journals publish anonymous peer-review reports alongside papers?",
            "Should preprints be eligible for consideration in national grant applications?",
            "Should publicly funded software always use an open-source license?",
            "Should conference talks be strictly capped at fifteen minutes?",
        ]),
        ("M10_energy_grid", [
            "Should new apartment buildings require rooftop solar panels by code?",
            "Should households receive lower overnight electricity tariffs for EV charging?",
            "Should urban streetlamps dim automatically when sidewalks are empty?",
            "Should utilities prioritize underground power cables in coastal zones?",
        ]),
        ("M11_health_habits", [
            "Should employers subsidize gym memberships for remote staff?",
            "Should pharmacies stay open until midnight in residential districts?",
            "Should hospitals allow flexible visiting hours in non-ICU wards?",
            "Should wearable health trackers integrate directly with primary care clinics?",
        ]),
        ("M12_finance_consumer", [
            "Should retail shops remain legally required to accept physical cash?",
            "Should online subscriptions require one-click cancellation by law?",
            "Should airlines issue automatic cash refunds for three-hour delays?",
            "Should warranty periods on home appliances be extended to five years?",
        ]),
        ("M13_civic_spaces", [
            "Should public libraries lend tools and household appliances alongside books?",
            "Should historic plazas host weekly open-air night markets in summer?",
            "Should coastal boardwalks ban motorized bicycles during weekend afternoons?",
            "Should community gardens reserve plots for local school classes?",
        ]),
        ("M14_digital_work", [
            "Should team meetings default to twenty-five minutes instead of thirty?",
            "Should companies disable non-urgent work email notifications after 7 PM?",
            "Should job postings legally disclose exact salary ranges upfront?",
            "Should coding interviews allow candidates to consult online documentation?",
        ]),
        ("M15_mixed_polarization", [
            "Is autumn a better season than spring?",
            "Should schools teach cursive handwriting today?",
            "Is reading print books better than reading ebooks?",
            "Is living near the sea preferable to living in the mountains?",
        ]),
    ]

    # Tier 2: 15 Natural Non-Transitive / Cyclic Tradeoff Cycles (SINGLE Fixed System Prompt)
    tier2_specs = [
        ("NT01_condorcet_vacation", [
            "Between a lively beach resort and a quiet mountain cabin, do you prefer the beach?",
            "Between a quiet mountain cabin and a historic city tour, do you prefer the mountains?",
            "Between a historic city tour and a scenic rail journey, do you prefer the city tour?",
            "Between a scenic rail journey and a lively beach resort, do you prefer the rail journey?",
        ]),
        ("NT02_condorcet_work", [
            "Between 10% higher salary and an extra week of vacation, do you choose higher salary?",
            "Between an extra week of vacation and full remote flexibility, do you choose extra vacation?",
            "Between full remote flexibility and faster promotion tracks, do you choose remote flexibility?",
            "Between faster promotion tracks and 10% higher salary, do you choose faster promotion?",
        ]),
        ("NT03_condorcet_policy", [
            "Between cutting income taxes and boosting school funding, should taxes be cut first?",
            "Between boosting school funding and expanding rail transit, should schools be funded first?",
            "Between expanding rail transit and lowering public debt, should rail transit come first?",
            "Between lowering public debt and cutting income taxes, should public debt come first?",
        ]),
        ("NT04_condorcet_tech", [
            "Between longer battery life and a lighter phone body, do you prefer longer battery?",
            "Between a lighter phone body and a sharper camera lens, do you prefer a lighter body?",
            "Between a sharper camera lens and a brighter screen, do you prefer a sharper camera?",
            "Between a brighter screen and longer battery life, do you prefer a brighter screen?",
        ]),
        ("NT05_sorites_ambiguity", [
            "Should a project delayed by two days be classified as seriously late?",
            "Should a project delayed by four days be classified as seriously late?",
            "Should a project delayed by six days be classified as seriously late?",
            "Is a two-day delay fundamentally acceptable rather than seriously late?",
        ]),
        ("NT06_framing_flip_ai", [
            "Should AI systems that pass safety tests be deployed quickly in hospitals?",
            "Should hospitals wait several extra years before trusting AI diagnostic aids?",
            "Do the benefits of rapid medical AI outweigh rare algorithmic errors?",
            "Should strict liability pause hospital deployment of new medical AI tools?",
        ]),
        ("NT07_framing_flip_energy", [
            "Should countries build new nuclear plants to cut carbon emissions fast?",
            "Are renewable solar and wind alone sufficient without new nuclear reactors?",
            "Does clean grid reliability require baseload nuclear power alongside renewables?",
            "Should nuclear construction budgets be redirected entirely to grid batteries?",
        ]),
        ("NT08_moral_dilemma_loop", [
            "Should an autonomous car prioritize saving pedestrians over its own passenger?",
            "Would you personally buy a car programmed to sacrifice its passenger in a crash?",
            "Should regulators mandate utilitarian crash algorithms for all self-driving cars?",
            "Does a driver have the right to choose a vehicle that protects its occupants first?",
        ]),
        ("NT09_polysemy_cycle_1", [
            "In 'The board approved the pitch quickly', does 'board' likely mean an executive committee?",
            "In 'The carpenter planed the pitch off the board', does 'pitch' mean tree resin?",
            "In 'The pitch looked muddy beside the wooden board', does 'pitch' mean a playing field?",
            "In 'She read the notice on the board before the pitch', does 'board' mean an executive committee?",
        ]),
        ("NT10_polysemy_cycle_2", [
            "In 'The current rose swiftly along the bank', does 'current' mean flowing water?",
            "In 'The bank reported a surge in current accounts', does 'current' mean financial checking?",
            "In 'The technician checked the current inside the battery bank', does 'bank' mean an array of cells?",
            "In 'A sudden current rattled the bank window', does 'bank' mean a river edge?",
        ]),
        ("NT11_anaphora_twist", [
            "In 'Alice praised Claire because she finished early', does 'she' refer to Claire?",
            "In 'Alice envied Claire because she finished early', does 'she' refer to Claire?",
            "In 'Alice called Claire because she needed advice', does 'she' refer to Alice?",
            "In 'Alice thanked Claire because she had been patient', does 'she' refer to Alice?",
        ]),
        ("NT12_economic_paradox", [
            "Should central banks keep interest rates high to crush inflation permanently?",
            "Should central banks cut rates quickly to protect small-business borrowing?",
            "Is persistent 3.5% inflation more damaging than a mild rise in unemployment?",
            "Should avoiding layoffs take priority over pushing inflation all the way down to 2%?",
        ]),
        ("NT13_governance_loop", [
            "Should major cities ban private cars from historic downtown streets?",
            "Should commuters from outer suburbs retain affordable car access to downtown jobs?",
            "Would pedestrian-only centers revitalize local shops more than parking garages?",
            "Would banning cars downtown unfairly penalize families without nearby subway lines?",
        ]),
        ("NT14_science_funding", [
            "Should national science agencies prioritize basic curiosity-driven research over applied products?",
            "Should taxpayers expect measurable commercial payoffs from most public research grants?",
            "Do major technological breakthroughs usually originate from undirected fundamental science?",
            "Should public R&D budgets shift mostly toward urgent industrial challenges?",
        ]),
        ("NT15_privacy_openness", [
            "Should scientific datasets be published openly even when partial re-identification risk exists?",
            "Should strict individual privacy override open data sharing in biomedical trials?",
            "Does open replication data save more lives than tight access restrictions?",
            "Should patients hold absolute veto power over sharing anonymized clinical records?",
        ]),
    ]

    all_cycles = []
    for cid, qs in tier1_specs:
        assert len(qs) == 4
        all_cycles.append(
            {
                "id": cid,
                "tier": "tier1_mobius_frustrated",
                "questions": qs,
                "sys_prompt": SYS_UNIFIED,
                # Edges 0,1,2 -> alignment context (+); Edge 3 -> inversion context (-)
                "edge_t2_templates": [T2_ALIGN, T2_ALIGN, T2_ALIGN, T2_INVERT],
            }
        )
    for cid, qs in tier2_specs:
        assert len(qs) == 4
        all_cycles.append(
            {
                "id": cid,
                "tier": "tier2_natural_nontransitive",
                "questions": qs,
                "sys_prompt": SYS_NATURAL,
                "edge_t2_templates": ["{q}", "{q}", "{q}", "{q}"],
            }
        )
    assert len(all_cycles) == 30
    return all_cycles


def run_part_a_frustration_cycles(
    model: Any, tok: Any, yes_id: int, no_id: int
) -> dict[str, Any]:
    """Evaluate all 30 new rank-4 cycles on Qwen2.5-1.5B in forward, reverse, and symmetrized modes."""
    cycles = build_30_frustration_cycles()

    # Turn 1 depends ONLY on (cycle, qi) under the single fixed system prompt of that tier!
    turn1_convs: list[list[dict[str, str]]] = []
    turn1_map: dict[tuple[int, int], int] = {}
    turn2_convs: list[list[dict[str, str]]] = []
    turn2_map: dict[tuple[int, int, int, int, int], int] = {}

    for ci, cyc in enumerate(cycles):
        qs = cyc["questions"]
        sys_p = cyc["sys_prompt"]
        t2_tmpls = cyc["edge_t2_templates"]
        for qi in range(4):
            turn1_map[(ci, qi)] = len(turn1_convs)
            turn1_convs.append(
                [
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": qs[qi]},
                ]
            )
        for edge in range(4):
            qa_i = edge
            qb_i = (edge + 1) % 4
            t2_tmpl = t2_tmpls[edge]
            for first_i, second_i in [(qa_i, qb_i), (qb_i, qa_i)]:
                for ans_val, ans_str in [(1, "Yes"), (-1, "No")]:
                    turn2_map[(ci, edge, first_i, ans_val, second_i)] = len(turn2_convs)
                    turn2_convs.append(
                        [
                            {"role": "system", "content": sys_p},
                            {"role": "user", "content": qs[first_i]},
                            {"role": "assistant", "content": ans_str},
                            {"role": "user", "content": t2_tmpl.format(q=qs[second_i])},
                        ]
                    )

    t0 = time.perf_counter()
    p1_all, m1_all = batch_eval_yes_probs(model, tok, turn1_convs, yes_id, no_id, batch_size=48)
    p2_all, m2_all = batch_eval_yes_probs(model, tok, turn2_convs, yes_id, no_id, batch_size=48)
    dt = time.perf_counter() - t0
    print(
        f"[Part A] Evaluated {len(turn1_convs)} turn-1 + {len(turn2_convs)} turn-2 exact logit calls "
        f"in {dt:.2f}s (Yes+No mass={np.mean(np.concatenate([m1_all, m2_all]))*100:.1f}%)"
    )

    def ordered_joint(ci: int, edge: int, first_i: int, second_i: int) -> dict[str, float]:
        p_a_pos = float(p1_all[turn1_map[(ci, first_i)]])
        p_a_neg = 1.0 - p_a_pos
        p_b_pos_ap = float(p2_all[turn2_map[(ci, edge, first_i, 1, second_i)]])
        p_b_pos_an = float(p2_all[turn2_map[(ci, edge, first_i, -1, second_i)]])
        return {
            "p_pp": p_a_pos * p_b_pos_ap,
            "p_pm": p_a_pos * (1.0 - p_b_pos_ap),
            "p_mp": p_a_neg * p_b_pos_an,
            "p_mm": p_a_neg * (1.0 - p_b_pos_an),
            "delta_trans": p_b_pos_ap - p_b_pos_an,
        }

    cycle_out = []
    for ci, cyc in enumerate(cycles):
        fwd_tables, rev_tables, sym_tables = [], [], []
        fwd_contrasts = []
        for edge in range(4):
            qa_i, qb_i = edge, (edge + 1) % 4
            t_ab = ordered_joint(ci, edge, qa_i, qb_i)
            t_ba_raw = ordered_joint(ci, edge, qb_i, qa_i)
            t_ba_aligned = {
                "p_pp": t_ba_raw["p_pp"],
                "p_pm": t_ba_raw["p_mp"],
                "p_mp": t_ba_raw["p_pm"],
                "p_mm": t_ba_raw["p_mm"],
            }
            t_sym = {k: 0.5 * (t_ab[k] + t_ba_aligned[k]) for k in ("p_pp", "p_pm", "p_mp", "p_mm")}
            fwd_tables.append(t_ab)
            rev_tables.append(t_ba_aligned)
            sym_tables.append(t_sym)
            fwd_contrasts.append(float(t_ab["delta_trans"]))

        fwd_m = compute_cycle_cbd_metrics(fwd_tables)
        rev_m = compute_cycle_cbd_metrics(rev_tables)
        sym_m = compute_cycle_cbd_metrics(sym_tables)
        cycle_out.append(
            {
                "id": cyc["id"],
                "tier": cyc["tier"],
                "forward": fwd_m,
                "reverse": rev_m,
                "symmetrized": sym_m,
                "forward_transition_contrasts": fwd_contrasts,
                "mean_abs_turn1_marginal": float(np.mean(np.abs(fwd_m["turn1_marginals"]))),
            }
        )

    by_tier = {}
    for tier in ("tier1_mobius_frustrated", "tier2_natural_nontransitive"):
        sub = [c for c in cycle_out if c["tier"] == tier]
        by_tier[tier] = {
            "n_cycles": len(sub),
            "forward_s_odd_mean": float(np.mean([c["forward"]["s_odd"] for c in sub])),
            "forward_s_odd_max": float(np.max([c["forward"]["s_odd"] for c in sub])),
            "forward_delta_0_mean": float(np.mean([c["forward"]["delta_0"] for c in sub])),
            "forward_c_cbd_mean": float(np.mean([c["forward"]["c_cbd"] for c in sub])),
            "forward_c_cbd_max": float(np.max([c["forward"]["c_cbd"] for c in sub])),
            "forward_raw_gt_2": int(sum(c["forward"]["exceeds_raw_chsh_2"] for c in sub)),
            "forward_cbd_violations": int(sum(c["forward"]["violates_cbd"] for c in sub)),
            "reverse_c_cbd_max": float(np.max([c["reverse"]["c_cbd"] for c in sub])),
            "reverse_cbd_violations": int(sum(c["reverse"]["violates_cbd"] for c in sub)),
            "symmetrized_c_cbd_mean": float(np.mean([c["symmetrized"]["c_cbd"] for c in sub])),
            "symmetrized_c_cbd_max": float(np.max([c["symmetrized"]["c_cbd"] for c in sub])),
            "symmetrized_cbd_violations": int(sum(c["symmetrized"]["violates_cbd"] for c in sub)),
            "mean_abs_turn1_marginal": float(np.mean([c["mean_abs_turn1_marginal"] for c in sub])),
        }

    return {
        "n_cycles_total": len(cycle_out),
        "turn1_calls": len(turn1_convs),
        "turn2_calls": len(turn2_convs),
        "mean_yes_no_mass": float(np.mean(np.concatenate([m1_all, m2_all]))),
        "by_tier": by_tier,
        "cycles": cycle_out,
    }


# ---------------------------------------------------------------------------
# Part B: Real-LLM Detection Loophole via Confidence / Refusal Rejection Sampling
# ---------------------------------------------------------------------------
def run_part_b_real_llm_detection_loophole(
    model: Any, tok: Any, yes_id: int, no_id: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Demonstrate detection-loophole CHSH violations directly on real isolated Qwen2.5-1.5B logits."""
    # Evaluate 80 isolated bipartite scenarios on Qwen2.5-1.5B (4 questions each = 320 calls, ~4s)
    topics = [
        "urban rail expansion", "offshore wind permits", "congestion pricing",
        "four-day hospital workweek", "AI essay grading", "open-weight AI licensing",
        "nuclear reactor extension", "dynamic hourly electricity tariffs",
        "autonomous night freight", "public university tuition reform",
        "municipal fiber broadband", "carbon border tariffs",
        "telemedicine prescriptions", "downtown pedestrianization",
        "automated warehouse taxation", "regional water desalination",
    ]
    framings = [
        "Recent audits found broad support and strong efficiency gains.",
        "Preliminary trials revealed severe cost overruns and public backlash.",
        "Independent panels remain split evenly between clear risks and gains.",
        "Pilot towns saw modest benefits alongside minor administrative friction.",
        "Early advocates cite long-term savings while critics warn of reliability issues.",
    ]
    convs = []
    for tp in topics:
        for fr in framings:
            ctx = f"Policy proposal regarding {tp}. {fr}"
            qs = [
                f"Should regulators approve the {tp} pilot?",
                f"Will ordinary residents benefit from {tp} within three years?",
                f"Should public funds subsidize {tp} implementation?",
                f"Are the long-term risks of {tp} acceptable?",
            ]
            for q in qs:
                convs.append(
                    [
                        {"role": "system", "content": SYS_NATURAL},
                        {"role": "user", "content": f"Context: {ctx}\nQuestion: {q}"},
                    ]
                )
    assert len(convs) == 320  # 80 scenarios x 4 isolated calls

    t0 = time.perf_counter()
    p_flat, m_flat = batch_eval_yes_probs(model, tok, convs, yes_id, no_id, batch_size=64)
    dt = time.perf_counter() - t0
    p_mat = p_flat.reshape(80, 4)  # columns [p_a0, p_a1, p_b0, p_b1]
    e_loc = 2.0 * p_mat - 1.0      # local expectations in [-1, +1]
    conf_loc = np.abs(e_loc)       # local confidence margins in [0, 1]

    # Sample N = 40,000 real isolated LLM trial outcomes from exact Bernoulli(p_mat)
    # Uniformly across the 80 real LLM scenarios (strictly isolated: A_x _|_ B_y | lambda)
    n_samples = 40_000
    scen_ids = rng.integers(0, 80, size=n_samples)
    p_s = p_mat[scen_ids]  # (N, 4)
    u_draw = rng.random(size=(n_samples, 4))
    a0 = np.where(u_draw[:, 0] < p_s[:, 0], 1.0, -1.0)
    a1 = np.where(u_draw[:, 1] < p_s[:, 1], 1.0, -1.0)
    b0 = np.where(u_draw[:, 2] < p_s[:, 2], 1.0, -1.0)
    b1 = np.where(u_draw[:, 3] < p_s[:, 3], 1.0, -1.0)
    c_a0, c_a1, c_b0, c_b1 = conf_loc[scen_ids, 0], conf_loc[scen_ids, 1], conf_loc[scen_ids, 2], conf_loc[scen_ids, 3]

    # Verify exact loophole-free isolated CHSH at eta = 1.0
    e_full = [float(np.mean(a0 * b0)), float(np.mean(a0 * b1)), float(np.mean(a1 * b0)), float(np.mean(a1 * b1))]
    s_full_eta1, _ = max_odd_signed_sum(e_full)
    assert s_full_eta1 <= 2.0 + 1e-9

    # Sweep acceptance efficiency eta in [0.55, 1.00]
    eta_grid = [1.00, 0.95, 0.90, 0.85, 0.8284, 0.78, 0.72, 0.67, 0.60, 0.55]

    # (1) Setting-dependent refusal/guardrail filter acting on REAL Qwen2.5-1.5B outcomes:
    # A local guardrail rejects a fraction (1 - eta) of outputs based strictly on LOCAL
    # information (its own setting x or y, its own output a or b, and local confidence/latent):
    # Specifically, assign each real LLM scenario a deterministic local sign polarity derived
    # from Qwen2.5's own confidence rank, and let local refusal trigger on the disfavored branch
    # for setting x=1 / y=1 (canonical local Pearle/Garg-Mermin refusal mechanism applied to real
    # LLM outputs with exact marginal acceptance rate eta!).
    # Also test (2) Pure confidence-margin abstention (reject lowest 1-eta confidence replies).
    s_post_guardrail = []
    s_closed_guardrail = []
    s_post_conf_only = []
    s_closed_conf_only = []
    eta_realized_list = []

    # Real LLM scenario polarity derived from Qwen2.5-1.5B's own response a0 on each trial
    # Construct a local refusal rule using ONLY Alice's local (x, a_x, c_ax) and Bob's local (y, b_y, c_by):
    # First partition the 80 scenarios evenly into two strata based on Qwen's median confidence on a0
    # (purely a function of shared context lambda_k!).
    med_c0 = float(np.median(conf_loc[:, 0]))
    high_conf_ctx = (conf_loc[scen_ids, 0] >= med_c0)  # shared context bit in {True, False}

    # On the 4 settings, Qwen2.5's raw outcomes have strong positive baseline correlation.
    # For a clean setting-dependent guardrail audit on REAL Qwen outputs:
    # - In setting (x=1), Alice's guardrail abstains with probability (1 - eta) when a1 == +1 and high_conf_ctx
    #   OR symmetrically adjusts to keep exact single-side acceptance rate = eta across all settings!
    # Even simpler and exact: construct the symmetric local refusal mask directly on the real
    # Qwen outcomes (a0, a1, b0, b1) by sorting trials by a local score s_A(x, a_x, lambda) and
    # s_B(y, b_y, lambda) and abstaining on the bottom (1 - eta) fraction locally!
    # Local guardrail scores (strictly local: Alice sees ONLY x, a_x, lambda; Bob sees ONLY y, b_y, lambda):
    # Setting 0 accepts preferentially when response matches context polarity; setting 1 reverses on conflict.
    # Using exact local Pearle/Garg-Mermin refusal probabilities on Qwen's real outcomes:
    u_ref_a0 = rng.random(n_samples)
    u_ref_a1 = rng.random(n_samples)
    u_ref_b0 = rng.random(n_samples)
    u_ref_b1 = rng.random(n_samples)

    # Local latent bias bit s_lam in {-1, +1} derived from Qwen2.5's own context-level a0 expectation sign
    ctx_sign = np.where(e_loc[scen_ids, 0] >= np.median(e_loc[:, 0]), 1.0, -1.0)
    # Local real-LLM outputs used in setting 0 and 1:
    # Alice's local guardrail abstains if her real output disagrees with the local refusal condition
    # (specifically: when eta < 1, Alice rejects a fraction (1-eta) of trials on x=0 if a0*ctx_sign < 0,
    # and on x=1 if a1*ctx_sign < 0; Bob rejects on y=0 if b0*ctx_sign < 0, and on y=1 if b1*ctx_sign > 0!
    # Notice: each party uses ONLY its OWN setting, its OWN real Qwen output, and shared context lambda!).
    for eta in eta_grid:
        # 1. Pure confidence-margin abstention (setting-independent local threshold)
        if eta >= 1.0 - 1e-12:
            acc_ca0 = np.ones(n_samples, dtype=bool)
            acc_ca1 = np.ones(n_samples, dtype=bool)
            acc_cb0 = np.ones(n_samples, dtype=bool)
            acc_cb1 = np.ones(n_samples, dtype=bool)
        else:
            acc_ca0 = c_a0 >= np.quantile(c_a0, 1.0 - eta)
            acc_ca1 = c_a1 >= np.quantile(c_a1, 1.0 - eta)
            acc_cb0 = c_b0 >= np.quantile(c_b0, 1.0 - eta)
            acc_cb1 = c_b1 >= np.quantile(c_b1, 1.0 - eta)

        e_pc = [
            float(np.mean((a0 * b0)[acc_ca0 & acc_cb0])),
            float(np.mean((a0 * b1)[acc_ca0 & acc_cb1])),
            float(np.mean((a1 * b0)[acc_ca1 & acc_cb0])),
            float(np.mean((a1 * b1)[acc_ca1 & acc_cb1])),
        ]
        s_pc, _ = max_odd_signed_sum(e_pc)
        a0_cl, a1_cl = np.where(acc_ca0, a0, 1.0), np.where(acc_ca1, a1, 1.0)
        b0_cl, b1_cl = np.where(acc_cb0, b0, 1.0), np.where(acc_cb1, b1, 1.0)
        e_cc = [
            float(np.mean(a0_cl * b0_cl)),
            float(np.mean(a0_cl * b1_cl)),
            float(np.mean(a1_cl * b0_cl)),
            float(np.mean(a1_cl * b1_cl)),
        ]
        s_cc, _ = max_odd_signed_sum(e_cc)
        s_post_conf_only.append(float(s_pc))
        s_closed_conf_only.append(float(s_cc))

        # 2. Setting-dependent local guardrail / rejection sampling on real Qwen2.5 outputs
        # For each local setting, reject disfavored local outcomes at rate r_rej so overall
        # local acceptance rate equals eta (strictly local: no communication between Alice & Bob!).
        def local_reject_mask(out_vec: np.ndarray, target_sign: np.ndarray, u_vec: np.ndarray, eta_tgt: float) -> np.ndarray:
            if eta_tgt >= 1.0 - 1e-12:
                return np.ones_like(out_vec, dtype=bool)
            disfav = (out_vec * target_sign) < 0.0
            frac_disfav = float(np.mean(disfav))
            if frac_disfav <= 1e-9:
                return u_vec < eta_tgt
            # Rejection prob on disfavored branch to hit overall acceptance = eta_tgt
            p_rej_disfav = min(1.0, (1.0 - eta_tgt) / frac_disfav)
            rem_rej = max(0.0, (1.0 - eta_tgt) - frac_disfav)
            frac_fav = 1.0 - frac_disfav
            p_rej_fav = min(1.0, rem_rej / frac_fav) if frac_fav > 1e-9 else 0.0
            rej = np.where(disfav, u_vec < p_rej_disfav, u_vec < p_rej_fav)
            return ~rej

        # Notice the 4 local target signs: (+ctx_sign, +ctx_sign, +ctx_sign, -ctx_sign)
        # Alice's x=0 prefers +ctx_sign, x=1 prefers +ctx_sign;
        # Bob's y=0 prefers +ctx_sign, y=1 prefers -ctx_sign -> purely local setting-dependent guardrail!
        # Wait: if Alice has identical sign for x=0 and x=1, E00+E01+E10-E11 cancels E01-E11!
        # To get genuine setting-dependent CHSH frustration, we split the 80 scenarios into two
        # balanced halves (using Qwen's median confidence bit high_conf_ctx):
        #   Half 1 (high_conf_ctx=True):  A0=+1, A1=+1, B0=+1, B1=+1 (all +1 -> E00+E01+E10-E11 = +2)
        #   Half 2 (high_conf_ctx=False): A0=+1, A1=-1, B0=+1, B1=-1 (E00=+1, E01=-1, E10=-1, E11=+1)
        # Better: standard Garg-Mermin / Pearle 4-branch local refusal on real LLM outputs:
        # On Half 1: Alice accepts on x=0 (+), x=1 (+); Bob accepts on y=0 (+), y=1 (+)
        # On Half 2: Alice accepts on x=0 (+), x=1 (-); Bob accepts on y=0 (+), y=1 (-)
        # Actually, let's use the exact 4-sector local refusal scheme where each party rejects ONLY
        # on the single frustrated setting pair when its local real Qwen answer is uncertain/disfavored:
        # Sector k in {0, 1, 2, 3} determined purely from shared context lambda (scen_ids % 4):
        sec = scen_ids % 4
        # Desired local accepted outputs per sector (purely local function of setting & shared context!):
        # Setting x=0: prefers +1 on sec {0,1}, -1 on sec {2,3};
        # Setting x=1: prefers +1 on sec {0,2}, -1 on sec {1,3};
        # Setting y=0: prefers +1 on sec {0}, abstains on sec {3}, etc.
        # Cleanest exact local Garg-Mermin refusal on real Qwen2.5 outputs:
        # Alice abstains on x=0 in sector 3 (prob 1-p_keep), on x=1 in sector 1;
        # Bob abstains on y=0 in sector 2, on y=1 in sector 0!
        # Wait: let's check what happens on the 4 sectors for real Qwen outputs filtered to match
        # local guardrail polarities!
        # Let target signs be:
        #   sec 0: a0=+1, b0=+1, a1=+1 (b1 abstains -> avoids -1 penalty on E11!)
        #   sec 1: a0=+1, b1=+1, b0=+1 (a1 abstains -> avoids -1 penalty on E10/E11!)
        #   sec 2: a1=+1, b0=+1, a0=+1 (b1 abstains)
        # Let's check exact Pearle / Garg-Mermin 2-setting local refusal directly:
        # In 2 halves of contexts (half = scen_ids % 2):
        #   - If half == 0: Alice x=0 prefers a0=+1; x=1 abstains at rate (1-q)
        #                   Bob   y=0 prefers b0=+1; y=1 prefers b1=+1 (so E00=+1, E01=+1)
        #   - If half == 1: Alice x=0 abstains at rate (1-q); x=1 prefers a1=+1
        #                   Bob   y=0 prefers b0=+1; y=1 prefers b1=-1 (so E10=+1, E11=-1)
        # CHECK THAT ALGEBRA!!
        #   - On pair (0, 0): both accept on half 0 -> E00 -> +1
        #   - On pair (0, 1): both accept on half 0 -> E01 -> +1
        #   - On pair (1, 0): both accept on half 1 -> E10 -> +1
        #   - On pair (1, 1): both accept on half 1 -> E11 -> -1 !!
        #   - Marginal single-side acceptance rate of Alice on x=0: 0.5*1 + 0.5*q = (1+q)/2 = eta!
        #   - Marginal single-side acceptance rate of Alice on x=1: 0.5*q + 0.5*1 = (1+q)/2 = eta!
        #   - Marginal single-side acceptance rate of Bob on y=0 and y=1: 1.0 (or symmetric sqrt(eta)!)
        #   - Wait: let's apply symmetric acceptance rate = eta on ALL 4 settings (a0, a1, b0, b1):
        #     Use 4 equal sectors (scen_ids % 4):
        #       sec 0: A0 prefers +1, B0 prefers +1; A1 abstains, B1 abstains
        #       sec 1: A0 prefers +1, B1 prefers +1; A1 abstains, B0 abstains
        #       sec 2: A1 prefers +1, B0 prefers +1; A0 abstains, B1 abstains
        #       sec 3: A1 prefers +1, B1 prefers -1; A0 abstains, B0 abstains
        #     At acceptance rate eta in [0.5, 1.0]: on each setting, the party accepts on its 2 active
        #     sectors (50% of trials) + accepts with probability q = 2*eta - 1 on its 2 inactive sectors,
        #     AND filters real Qwen2.5 outcomes locally!
        q_keep = max(0.0, min(1.0, 2.0 * eta - 1.0))

        # Keep real Qwen2.5 outcomes; on active sectors prefer matching Qwen's natural sign or
        # filter strictly via local refusal on inactive sectors (100% real Qwen2.5 outputs a0,a1,b0,b1
        # where we map each sector's natural polarity so Qwen's real local answers align on active sectors!)
        # Even better: flip ONLY the sign convention of the binary label (Yes=+1 vs No=+1, a purely local
        # relabeling per setting & context) so we use 100% REAL Qwen2.5 Bernoulli outcomes without fabricating a bit!
        a0_eff = np.where(sec <= 1, a0, -a0)
        a1_eff = np.where((sec == 2) | (sec == 3), a1, -a1)
        b0_eff = np.where((sec == 0) | (sec == 2), b0, -b0)
        b1_eff = np.where(sec == 1, b1, np.where(sec == 3, -b1, b1))

        #Wait: check that a0_eff is STILL a 100% local function of (x=0, a0, lambda)!
        # Let's check what happens at eta=1.0 (no abstention):
        # Since a0,a1,b0,b1 are drawn from real Qwen2.5's isolated probabilities, ANY local relabeling
        # MUST STILL satisfy |S| <= 2.0 at eta = 1.0! Let's assert that explicitly!
        acc_g_a0 = np.where((sec == 0) | (sec == 1), True, u_ref_a0 < q_keep)
        acc_g_a1 = np.where((sec == 2) | (sec == 3), True, u_ref_a1 < q_keep)
        acc_g_b0 = np.where((sec == 0) | (sec == 2), True, u_ref_b0 < q_keep)
        acc_g_b1 = np.where((sec == 1) | (sec == 3), True, u_ref_b1 < q_keep)

        # Post-selected CHSH conditioned on both Alice & Bob accepting (collider!)
        e00_p = float(np.mean((a0_eff * b0_eff)[acc_g_a0 & acc_g_b0]))
        e01_p = float(np.mean((a0_eff * b1_eff)[acc_g_a0 & acc_g_b1]))
        e10_p = float(np.mean((a1_eff * b0_eff)[acc_g_a1 & acc_g_b0]))
        e11_p = float(np.mean((a1_eff * b1_eff)[acc_g_a1 & acc_g_b1]))
        s_pg = float(e00_p + e01_p + e10_p - e11_p)

        # Loophole-free deterministic imputation (refusal -> +1) on the EXACT SAME trials
        a0_lf = np.where(acc_g_a0, a0_eff, 1.0)
        a1_lf = np.where(acc_g_a1, a1_eff, 1.0)
        b0_lf = np.where(acc_g_b0, b0_eff, 1.0)
        b1_lf = np.where(acc_g_b1, b1_eff, 1.0)
        e_lf = [
            float(np.mean(a0_lf * b0_lf)),
            float(np.mean(a0_lf * b1_lf)),
            float(np.mean(a1_lf * b0_lf)),
            float(np.mean(a1_lf * b1_lf)),
        ]
        s_cg, _ = max_odd_signed_sum(e_lf)

        if eta >= 1.0 - 1e-12:
            assert abs(s_pg) <= 2.0 + 1e-9, f"eta=1 violation: {s_pg}"
        assert s_cg <= 2.0 + 1e-9, f"Loophole-free violation: {s_cg}"

        s_post_guardrail.append(abs(s_pg))
        s_closed_guardrail.append(float(s_cg))
        eta_realized_list.append(
            float(0.25 * (np.mean(acc_g_a0) + np.mean(acc_g_a1) + np.mean(acc_g_b0) + np.mean(acc_g_b1)))
        )

    print(
        f"[Part B] Real-LLM Detection Loophole (80 Qwen scenarios, 40k trials, {dt:.2f}s): "
        f"eta=1.0 -> S_post={s_post_guardrail[0]:.4f} | eta=0.55 -> S_post={s_post_guardrail[-1]:.4f} "
        f"vs Loophole-Free max={max(s_closed_guardrail):.4f} (conf-only max={max(s_post_conf_only):.4f})"
    )

    return {
        "n_scenarios": 80,
        "n_mc_trials": n_samples,
        "mean_yes_no_mass": float(np.mean(m_flat)),
        "mean_local_confidence": float(np.mean(conf_loc)),
        "eta_grid": eta_grid,
        "eta_realized": eta_realized_list,
        "s_post_setting_guardrail": s_post_guardrail,
        "s_closed_setting_guardrail": s_closed_guardrail,
        "s_post_confidence_only": s_post_conf_only,
        "s_closed_confidence_only": s_closed_conf_only,
    }


# ---------------------------------------------------------------------------
# Part C: Spectral Validation of Correlated Effective Multiplicity M_eff(Sigma)
# ---------------------------------------------------------------------------
def expected_max_z(m_eff: float) -> float:
    """Extreme-value expectation E[max_{1..M_eff} Z_i]."""
    if m_eff <= 1.0:
        return 0.0
    euler_gamma = 0.5772156649015329
    return float(
        (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / m_eff)
        + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (m_eff * math.e))
    )


def build_80_diverse_prompts() -> list[dict[str, str]]:
    """Build M = 80 diverse real prompts spanning wide polarity/style diversity."""
    personas = [
        ("neutral", "You are a neutral evaluator."),
        ("optimist", "You are an optimistic reader who highlights positives."),
        ("skeptic", "You are a skeptical auditor who flags any doubt."),
        ("balanced", "You are a pragmatic decision analyst."),
    ]
    instructions = [
        ("yes_no_pos", "Does this statement lean favorable? Reply Yes or No."),
        ("approve", "Would you endorse this observation? Reply Yes or No."),
        ("accept", "Does this pass an optimistic threshold? Reply Yes or No."),
        ("confirm", "Is the net impression encouraging? Reply Yes or No."),
        ("rate_up", "Would you classify this as positive (Yes) or not (No)?"),
    ]
    few_shots = [
        ("zero", ""),
        ("fs_pn", "Ex1: 'Quiet room, solid finish.' -> Yes\nEx2: 'Sleek look, weak latch.' -> No\n"),
        ("fs_np", "Ex1: 'Sleek look, weak latch.' -> No\nEx2: 'Quiet room, solid finish.' -> Yes\n"),
        ("fs_amb", "Ex1: 'Modest gain, high cost.' -> No\nEx2: 'Slow start, steady value.' -> Yes\n"),
    ]
    prompts = []
    for pn, ps in personas:
        for iname, itxt in instructions:
            for fn, ftxt in few_shots:
                prompts.append(
                    {
                        "name": f"{pn}__{iname}__{fn}",
                        "prefix": f"<|im_start|>system\n{ps} {itxt}<|im_end|>\n<|im_start|>user\n{ftxt}Text: ",
                        "suffix_tail": "\nAnswer:<|im_end|>\n<|im_start|>assistant\n",
                    }
                )
    assert len(prompts) == 80
    return prompts


def run_part_c_spectral_meff(
    model: Any, tok: Any, yes_id: int, no_id: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Validate spectral M_eff(Sigma_prompt) formulas against live Qwen2.5-1.5B optimism gaps."""
    prompts = build_80_diverse_prompts()

    # Build 2 null-label datasets of 96 items each (n_val=24, n_test=72) with DIFFERENT prompt correlation:
    # Regime C1 (High correlation): 96 neutral factual items (prompts strongly agree -> high rho_bar)
    subj = [
        "The ferry", "The archive", "The committee", "The clerk", "The station",
        "The bakery", "The curator", "The inspector", "The utility", "The office",
        "The press", "The lab",
    ]
    act = [
        "updated the schedule on Tuesday", "counted four crates before noon",
        "checked the pressure gauge after lunch", "filed the ledger in cabinet two",
        "issued a routine memo at midday", "logged the room temperature at ten",
        "moved two folders to desk three", "scheduled the annual check for May",
    ]
    items_high_corr = [f"{s} {a}." for s in subj for a in act]
    assert len(items_high_corr) == 96

    # Regime C2 (Lower correlation): 96 borderline mixed-valence items (prompts disagree strongly!)
    positives = [
        "fast service", "clean lobby", "warm welcome", "sturdy frame",
        "clear screen", "quiet engine", "rich flavor", "crisp sound",
        "prompt reply", "tidy cabin", "smooth ride", "sharp lens",
    ]
    negatives = [
        "steep price", "tiny portions", "awkward strap", "short cable",
        "dim corners", "slow checkout", "stiff seats", "noisy vent",
    ]
    items_low_corr = [f"It offers {p}, though marred by {n}." for p in positives for n in negatives]
    assert len(items_low_corr) == 96

    all_items = items_high_corr + items_low_corr  # 192 concise items
    n_prompts = len(prompts)
    probs_all = np.zeros((n_prompts, len(all_items)), dtype=np.float64)

    t0 = time.perf_counter()
    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    for pi, p_cfg in enumerate(prompts):
        prefix_ids = tok.encode(p_cfg["prefix"], add_special_tokens=False)
        suf_enc = [tok.encode(it + p_cfg["suffix_tail"], add_special_tokens=False) for it in all_items]
        order = sorted(range(len(all_items)), key=lambda i: len(suf_enc[i]))

        base_cache = make_prompt_cache(model)
        _ = model.model(mx.array([prefix_ids], dtype=mx.int32), cache=base_cache)
        mx.eval([c.state for c in base_cache])

        for start in range(0, len(all_items), 96):
            end = min(len(all_items), start + 96)
            idxs = order[start:end]
            cb = len(idxs)
            max_l = max(len(suf_enc[i]) for i in idxs)
            arr = np.full((cb, max_l), pad_id, dtype=np.int32)
            lens = np.zeros(cb, dtype=np.int32)
            for row, orig_i in enumerate(idxs):
                seq = suf_enc[orig_i]
                arr[row, : len(seq)] = seq
                lens[row] = len(seq) - 1

            chunk_cache = []
            for c in base_cache:
                nc = KVCache()
                nc.keys = mx.repeat(c.keys[:, :, : c.offset, :], cb, axis=0)
                nc.values = mx.repeat(c.values[:, :, : c.offset, :], cb, axis=0)
                nc.offset = c.offset
                chunk_cache.append(nc)

            h = model.model(mx.array(arr), cache=chunk_cache)
            last_h = h[mx.arange(cb), mx.array(lens), :]
            lg = last_token_logits(model, last_h).astype(mx.float32)
            p_y = mx.sigmoid(lg[:, yes_id] - lg[:, no_id])
            mx.eval(p_y)
            p_np = np.asarray(p_y, dtype=np.float64)
            for row, orig_i in enumerate(idxs):
                probs_all[pi, orig_i] = p_np[row]

    dt = time.perf_counter() - t0
    print(f"[Part C] Evaluated {n_prompts} prompts x {len(all_items)} items in {dt:.2f}s")

    def analyze_spectral_regime(probs_sub: np.ndarray, label_name: str) -> dict[str, Any]:
        # Independent random balanced null labels (48 pos, 48 neg)
        y_null = np.array([1] * 48 + [0] * 48, dtype=np.int32)
        rng.shuffle(y_null)
        preds = (probs_sub >= 0.5).astype(np.int32)
        correct = (preds == y_null[None, :]).astype(np.float64)

        # Correlation matrix of continuous centered logits/probs vs binary correctness
        # Use continuous centered probabilities (standard Gaussian-process score correlation Sigma_prompt)
        # AND binary correctness correlation for full transparency
        p_cent = probs_sub - np.mean(probs_sub, axis=1, keepdims=True)
        p_std = np.std(probs_sub, axis=1, keepdims=True)
        p_std = np.where(p_std < 1e-9, 1.0, p_std)
        p_norm = p_cent / p_std
        sigma_prob = (p_norm @ p_norm.T) / probs_sub.shape[1]
        np.fill_diagonal(sigma_prob, 1.0)
        eigs_full = np.maximum(np.linalg.eigvalsh(sigma_prob), 0.0)
        off_d = sigma_prob[~np.eye(n_prompts, dtype=bool)]
        mean_rho = float(np.mean(off_d))

        m_grid = [1, 5, 10, 20, 40, 80]
        n_val = 24
        n_res = 300
        pos_idx = np.where(y_null == 1)[0]
        neg_idx = np.where(y_null == 0)[0]

        splits = []
        perms = []
        for _ in range(n_res):
            ps = rng.permutation(pos_idx)
            ns = rng.permutation(neg_idx)
            splits.append((np.concatenate([ps[:12], ns[:12]]), np.concatenate([ps[12:], ns[12:]])))
            perms.append(rng.permutation(n_prompts))

        emp_gap = []
        emp_val = []
        emp_test = []
        pred_pr = []
        pred_galwey = []
        pred_nyholt = []
        pred_naive = []
        m_pr_list, m_gal_list, m_nyh_list = [], [], []

        for m in m_grid:
            # Compute spectral M_eff averaged over 20 random subsets of size m (exact for m=80)
            pr_vals, gal_vals, nyh_vals = [], [], []
            n_sub = 1 if m == n_prompts else 20
            for s_i in range(n_sub):
                sub_idx = perms[s_i][:m]
                sub_sig = sigma_prob[np.ix_(sub_idx, sub_idx)]
                ev = np.maximum(np.linalg.eigvalsh(sub_sig), 0.0)
                sum_ev = float(np.sum(ev))
                sum_ev2 = float(np.sum(ev ** 2))
                m_pr = (sum_ev ** 2) / sum_ev2 if sum_ev2 > 0 else 1.0
                m_gal = (float(np.sum(np.sqrt(ev))) ** 2) / sum_ev if sum_ev > 0 else 1.0
                m_nyh = 1.0 + (m - 1.0) * max(0.0, 1.0 - float(np.var(ev)) / m) if m > 1 else 1.0
                pr_vals.append(max(1.0, min(float(m), m_pr)))
                gal_vals.append(max(1.0, min(float(m), m_gal)))
                nyh_vals.append(max(1.0, min(float(m), m_nyh)))

            m_pr_m = float(np.mean(pr_vals))
            m_gal_m = float(np.mean(gal_vals))
            m_nyh_m = float(np.mean(nyh_vals))
            m_pr_list.append(m_pr_m)
            m_gal_list.append(m_gal_m)
            m_nyh_list.append(m_nyh_m)

            # Empirical val, test, and optimism gap across 300 balanced splits
            v_arr = np.zeros(n_res)
            t_arr = np.zeros(n_res)
            for r in range(n_res):
                v_i, t_i = splits[r]
                p_sub = perms[r][:m]
                va = np.mean(correct[p_sub][:, v_i], axis=1)
                ta = np.mean(correct[p_sub][:, t_i], axis=1)
                bi = int(np.argmax(va))
                v_arr[r] = va[bi]
                t_arr[r] = ta[bi]

            emp_val.append(float(np.mean(v_arr)))
            emp_test.append(float(np.mean(t_arr)))
            emp_gap.append(float(np.mean(v_arr - t_arr)))

            # Analytical extreme-value optimism prediction: sigma_bin * E[max Z_{M_eff}]
            # Scale by empirical per-prompt validation accuracy std (~sqrt(0.25 / n_val) * sqrt(1 - rho_eff))
            se_null = math.sqrt(0.25 / n_val)
            pred_pr.append(se_null * expected_max_z(m_pr_m))
            pred_galwey.append(se_null * expected_max_z(m_gal_m))
            pred_nyholt.append(se_null * expected_max_z(m_nyh_m))
            pred_naive.append(se_null * expected_max_z(float(m)))

        # MAE of each predictor vs empirical optimism gap across M in {5..80}
        eg = np.array(emp_gap[1:])
        mae_pr = float(np.mean(np.abs(np.array(pred_pr[1:]) - eg)))
        mae_gal = float(np.mean(np.abs(np.array(pred_galwey[1:]) - eg)))
        mae_nyh = float(np.mean(np.abs(np.array(pred_nyholt[1:]) - eg)))
        mae_naive = float(np.mean(np.abs(np.array(pred_naive[1:]) - eg)))

        return {
            "name": label_name,
            "mean_pairwise_prob_corr": mean_rho,
            "top_5_eigenvalues": [float(x) for x in sorted(eigs_full, reverse=True)[:5]],
            "m_grid": m_grid,
            "m_pr": m_pr_list,
            "m_galwey": m_gal_list,
            "m_nyholt": m_nyh_list,
            "emp_val_acc": emp_val,
            "emp_test_acc": emp_test,
            "emp_optimism_gap": emp_gap,
            "pred_gap_pr": pred_pr,
            "pred_gap_galwey": pred_galwey,
            "pred_gap_nyholt": pred_nyholt,
            "pred_gap_naive": pred_naive,
            "mae_gap_pr": mae_pr,
            "mae_gap_galwey": mae_gal,
            "mae_gap_nyholt": mae_nyh,
            "mae_gap_naive": mae_naive,
        }

    reg_high = analyze_spectral_regime(probs_all[:, :96], "high_corr_neutral_null")
    reg_low = analyze_spectral_regime(probs_all[:, 96:], "low_corr_borderline_null")

    print(
        f"[Part C] High-corr null (rho={reg_high['mean_pairwise_prob_corr']:.2f}): "
        f"M=80 -> M_PR={reg_high['m_pr'][-1]:.1f}, M_Gal={reg_high['m_galwey'][-1]:.1f}, "
        f"emp gap={reg_high['emp_optimism_gap'][-1]*100:+.2f}pp | MAE(PR/Gal/Nyh/Naive)="
        f"{reg_high['mae_gap_pr']*100:.2f}/{reg_high['mae_gap_galwey']*100:.2f}/{reg_high['mae_gap_nyholt']*100:.2f}/{reg_high['mae_gap_naive']*100:.2f}pp"
    )
    print(
        f"[Part C] Low-corr null  (rho={reg_low['mean_pairwise_prob_corr']:.2f}): "
        f"M=80 -> M_PR={reg_low['m_pr'][-1]:.1f}, M_Gal={reg_low['m_galwey'][-1]:.1f}, "
        f"emp gap={reg_low['emp_optimism_gap'][-1]*100:+.2f}pp | MAE(PR/Gal/Nyh/Naive)="
        f"{reg_low['mae_gap_pr']*100:.2f}/{reg_low['mae_gap_galwey']*100:.2f}/{reg_low['mae_gap_nyholt']*100:.2f}/{reg_low['mae_gap_naive']*100:.2f}pp"
    )

    return {
        "n_prompts": n_prompts,
        "n_items_per_regime": 96,
        "n_val": 24,
        "n_test": 72,
        "n_resamples": 300,
        "high_corr_regime": reg_high,
        "low_corr_regime": reg_low,
    }


def plot_figure_11(res: dict[str, Any], exp9_cycles: list[dict[str, Any]], out_path: Path) -> None:
    """Render 4-panel publication-grade Figure 11."""
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0), constrained_layout=True)
    fig.suptitle(
        "Experiment 11 (Paper 2 Live Verification on Qwen2.5-1.5B MLX) — Frustration Phase Transition, Detection Loophole & Spectral M_eff",
        fontsize=11.8,
        fontweight="bold",
        y=1.02,
    )

    part_a = res["part_a_frustration"]
    t1 = [c for c in part_a["cycles"] if c["tier"] == "tier1_mobius_frustrated"]
    t2 = [c for c in part_a["cycles"] if c["tier"] == "tier2_natural_nontransitive"]

    # Panel (a): Phase-Space Scatter (Delta_0, s_odd) comparing Exp 9 vs Tier 2 vs Tier 1
    ax = axes[0, 0]
    all_d0 = (
        [c["forward"]["delta_0"] for c in exp9_cycles]
        + [c["forward"]["delta_0"] for c in t2]
        + [c["forward"]["delta_0"] for c in t1]
    )
    max_d0 = max(all_d0) if all_d0 else 3.5
    d_line = np.linspace(0.0, max_d0 * 1.08 + 0.1, 100)
    ax.plot(d_line, 2.0 + d_line, color="#0F172A", ls="--", lw=1.6, label=r"CbD Phase Boundary $s_{\mathrm{odd}} = 2 + \Delta_0$")
    ax.axhline(2.0, color=PALETTE["slate"], ls=":", lw=1.1, label=r"Naive CHSH Line $s_{\mathrm{odd}} = 2$")
    ax.fill_between(d_line, 2.0 + d_line, 4.25, color=PALETTE["coral"], alpha=0.10, label=r"CbD-Contextual Phase ($C_{\mathrm{CbD}} > 2$)")

    ax.scatter(
        [c["forward"]["delta_0"] for c in exp9_cycles],
        [c["forward"]["s_odd"] for c in exp9_cycles],
        color=PALETTE["slate"], marker="o", s=34, alpha=0.55, edgecolor="white", lw=0.4,
        label=f"Exp 9 Standard (36 cycles, 0/36 CbD>2)", zorder=3,
    )
    ax.scatter(
        [c["forward"]["delta_0"] for c in t2],
        [c["forward"]["s_odd"] for c in t2],
        color=PALETTE["teal"], marker="^", s=55, alpha=0.85, edgecolor="white", lw=0.5,
        label=f"Exp 11 Tier 2 Natural (15 cycles, {part_a['by_tier']['tier2_natural_nontransitive']['forward_cbd_violations']}/15 CbD>2)", zorder=4,
    )
    ax.scatter(
        [c["forward"]["delta_0"] for c in t1],
        [c["forward"]["s_odd"] for c in t1],
        color=PALETTE["coral"], marker="D", s=62, alpha=0.92, edgecolor="#0F172A", lw=0.6,
        label=f"Exp 11 Tier 1 Mobius (15 cycles, {part_a['by_tier']['tier1_mobius_frustrated']['forward_cbd_violations']}/15 CbD>2)", zorder=5,
    )

    ax.set_title(r"(a) Part A: Semantic Frustration Phase Transition on Qwen2.5-1.5B")
    ax.set_xlabel(r"KV-Cache Signaling (Marginal Inconsistency) $\Delta_0$")
    ax.set_ylabel(r"Cyclic Correlation Functional $s_{\mathrm{odd}}$")
    ax.set_xlim(-0.08, max_d0 * 1.06 + 0.15)
    ax.set_ylim(0.5, 4.22)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=7.3)

    # Panel (b): Boxplot Comparison of C_CbD Across Exp 9, Tier 2, and Tier 1 (Forward & Symmetrized)
    ax = axes[0, 1]
    exp9_fwd = [c["forward"]["c_cbd"] for c in exp9_cycles]
    t2_fwd = [c["forward"]["c_cbd"] for c in t2]
    t1_fwd = [c["forward"]["c_cbd"] for c in t1]
    t1_sym = [c["symmetrized"]["c_cbd"] for c in t1]

    bp = ax.boxplot(
        [exp9_fwd, t2_fwd, t1_fwd, t1_sym],
        tick_labels=[
            "Exp 9\nStandard\n(Fwd, n=36)",
            "Tier 2\nNatural\n(Fwd, n=15)",
            "Tier 1\nMobius\n(Fwd, n=15)",
            "Tier 1\nMobius\n(Sym, n=15)",
        ],
        patch_artist=True,
        widths=0.48,
        medianprops=dict(color="#0F172A", lw=1.8),
        whiskerprops=dict(color="#475569", lw=1.1),
        capprops=dict(color="#475569", lw=1.1),
        flierprops=dict(marker="o", ms=4.5, markerfacecolor=PALETTE["coral"], alpha=0.75),
    )
    for patch, col in zip(bp["boxes"], [PALETTE["light_gray"], PALETTE["teal"], PALETTE["coral"], PALETTE["purple"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.76)

    ax.axhline(2.0, color="#0F172A", ls="--", lw=1.5, label="Non-Contextuality Ceiling ($C_{\\mathrm{CbD}} = 2.0$)")
    ax.axhline(2.0 * math.sqrt(2.0), color=PALETTE["amber"], ls=":", lw=1.3, label=r"Tsirelson Reference ($2\sqrt{2} \approx 2.828$)")
    ax.set_title(r"(b) Part A: Exact CbD Witness $C_{\mathrm{CbD}} = s_{\mathrm{odd}} - \Delta_0$ Across Regimes")
    ax.set_ylabel(r"CbD Contextuality Witness $C_{\mathrm{CbD}}$")
    all_c = exp9_fwd + t2_fwd + t1_fwd + t1_sym
    ax.set_ylim(min(-2.5, min(all_c) - 0.6), max(4.35, max(all_c) + 0.55))
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", fontsize=7.5)

    bt1 = part_a["by_tier"]["tier1_mobius_frustrated"]
    bt2 = part_a["by_tier"]["tier2_natural_nontransitive"]
    txt_b = (
        "Live Qwen2.5-1.5B Phase Summary:\n"
        f"• Exp 9 Std   : max C={max(exp9_fwd):.3f} (0/36 > 2)\n"
        f"• Tier 2 Nat  : max C={bt2['forward_c_cbd_max']:.3f} ({bt2['forward_cbd_violations']}/15 > 2)\n"
        f"• Tier 1 Mob  : max C={bt1['forward_c_cbd_max']:.3f} ({bt1['forward_cbd_violations']}/15 Fwd)\n"
        f"• Tier 1 Sym  : max C={bt1['symmetrized_c_cbd_max']:.3f} ({bt1['symmetrized_cbd_violations']}/15 Sym)"
    )
    ax.text(
        0.025, 0.975, txt_b, transform=ax.transAxes, ha="left", va="top",
        fontsize=7.4, family="monospace",
        bbox=dict(boxstyle="round,pad=0.33", facecolor=PALETTE["bg_tint"], edgecolor="#94A3B8", lw=0.8),
    )

    # Panel (c): Part B — Real-LLM Detection Loophole vs Loophole-Free Imputation
    ax = axes[1, 0]
    pb = res["part_b_detection_loophole"]
    eta_arr = np.array(pb["eta_realized"]) * 100.0
    eta_th = np.linspace(0.55, 1.0, 200)
    s_th = np.minimum(4.0, 4.0 / eta_th - 2.0)

    ax.plot(eta_th * 100.0, s_th, color="#1E293B", ls="--", lw=1.4, label=r"Pearle/Garg–Mermin Limit $\min(4, \frac{4}{\eta}-2)$")
    ax.plot(eta_arr, pb["s_post_setting_guardrail"], "o-", color=PALETTE["coral"], lw=2.3, ms=5.5,
            label="Real Qwen2.5 + Setting-Dep. Guardrail (Post-Selected)")
    ax.plot(eta_arr, pb["s_closed_setting_guardrail"], "s-", color=PALETTE["navy"], lw=2.2, ms=5.0,
            label=r"Same Qwen2.5 Outputs Loophole-Free ($\emptyset \mapsto +1$)")
    ax.plot(eta_arr, pb["s_post_confidence_only"], "^--", color=PALETTE["teal"], lw=1.7, ms=4.8,
            label="Real Qwen2.5 + Confidence-Only Abstention")
    ax.axhline(2.0, color="#0F172A", ls=":", lw=1.2, label="Bell-CHSH Bound ($|S|=2.0$)")
    ax.axhline(2.0 * math.sqrt(2.0), color=PALETTE["amber"], ls=":", lw=1.2, label=r"Tsirelson Bound ($2\sqrt{2}$)")

    ax.set_title("(c) Part B: Real-LLM Detection Loophole Under Refusal Filtering")
    ax.set_xlabel(r"Marginal Acceptance Efficiency $\eta$ (%)")
    ax.set_ylabel("Measured CHSH Parameter $|S|$")
    ax.set_xlim(53.0, 102.0)
    ax.set_ylim(0.5, 4.25)
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=7.3)

    # Panel (d): Part C — Spectral Effective Multiplicity M_eff(Sigma_prompt) Validation
    ax = axes[1, 1]
    pc = res["part_c_spectral_meff"]
    rh = pc["high_corr_regime"]
    rl = pc["low_corr_regime"]
    mg = np.array(rh["m_grid"], dtype=float)

    ax.plot(mg, np.array(rl["emp_optimism_gap"]) * 100.0, "o-", color=PALETTE["coral"], lw=2.4, ms=5.5,
            label=f"Empirical Gap — Borderline Null ($\\bar{{\\rho}}={rl['mean_pairwise_prob_corr']:.2f}$)")
    ax.plot(mg, np.array(rl["pred_gap_galwey"]) * 100.0, "--", color=PALETTE["coral"], lw=1.7,
            label=f"Galwey Spectral Bound ($M_{{\\mathrm{{Gal}}}}={rl['m_galwey'][-1]:.1f}$, MAE={rl['mae_gap_galwey']*100:.1f}pp)")
    ax.plot(mg, np.array(rh["emp_optimism_gap"]) * 100.0, "s-", color=PALETTE["navy"], lw=2.4, ms=5.5,
            label=f"Empirical Gap — Neutral Null ($\\bar{{\\rho}}={rh['mean_pairwise_prob_corr']:.2f}$)")
    ax.plot(mg, np.array(rh["pred_gap_pr"]) * 100.0, "--", color=PALETTE["navy"], lw=1.7,
            label=f"PR Spectral Bound ($M_{{\\mathrm{{PR}}}}={rh['m_pr'][-1]:.1f}$, MAE={rh['mae_gap_pr']*100:.1f}pp)")
    ax.plot(mg, np.array(rl["pred_gap_naive"]) * 100.0, ":", color="#0F172A", lw=1.6,
            label=f"Naive Uncorrelated ($M=80$, MAE={rh['mae_gap_naive']*100:.1f}/{rl['mae_gap_naive']*100:.1f}pp)")

    ax.set_xscale("log")
    ax.set_xticks(mg)
    ax.set_xticklabels([str(int(x)) for x in mg])
    ax.set_title(r"(d) Part C: Spectral $M_{\mathrm{eff}}(\Sigma_{\mathrm{prompt}})$ vs. Live Qwen2.5 Optimism Gap")
    ax.set_xlabel("Candidate Prompt Pool Size $M$ (log scale)")
    ax.set_ylabel(r"Optimism Gap $\mathrm{Acc}_{\mathrm{val}} - \mathrm{Acc}_{\mathrm{test}}$ (pp)")
    ax.set_ylim(-1.0, max(np.max(rl["pred_gap_naive"]) * 100.0 + 4.0, 28.0))
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.3)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    rng = np.random.default_rng(GLOBAL_SEED)

    print("=" * 78)
    print("EXPERIMENT 11 — PAPER 2 LIVE VERIFICATION ON Qwen2.5-1.5B-Instruct (MLX)")
    print(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"MLX      : {mx.__version__} ({mx.default_device()}) | mlx_lm {mlx_lm.__version__}")
    print(f"Model    : {MODEL_ID}")
    print("=" * 78)

    model, tok = mlx_lm.load(MODEL_ID)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]

    weights_before = fingerprint_frozen_weights(model)
    print(f"[Audit] Weights BEFORE: sha256={weights_before['sha256'][:16]}...")

    # Load Exp 9 standard cycles for direct comparison in Figure 11
    exp9_json_path = RESULTS_DIR / "exp9_llm_cbd_signaling.json"
    with open(exp9_json_path, "r", encoding="utf-8") as f:
        exp9_data = json.load(f)
    exp9_cycles = exp9_data["part_b_multiturn_cbd"]["cycles"]

    # Run all 3 parts
    part_a = run_part_a_frustration_cycles(model, tok, yes_id, no_id)
    part_b = run_part_b_real_llm_detection_loophole(model, tok, yes_id, no_id, rng)
    part_c = run_part_c_spectral_meff(model, tok, yes_id, no_id, rng)

    weights_after = fingerprint_frozen_weights(model)
    weights_sha_match = weights_before["sha256"] == weights_after["sha256"]
    weights_l2_diff = abs(weights_before["l2_norm"] - weights_after["l2_norm"])
    assert weights_sha_match and weights_l2_diff == 0.0, "Frozen weights mutated!"
    print(f"[Audit] Weights AFTER : sha256={weights_after['sha256'][:16]}... MATCH={weights_sha_match}, ||dtheta*||=0.0")

    elapsed = time.perf_counter() - t0
    results = {
        "experiment": "exp11_paper2_autoregressive_frustration_transition",
        "model_id": MODEL_ID,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "seed": GLOBAL_SEED,
        "weights_before": weights_before,
        "weights_after": weights_after,
        "weights_sha256_match": weights_sha_match,
        "weights_l2_diff": weights_l2_diff,
        "exp9_reference_summary": {
            "n_cycles": len(exp9_cycles),
            "forward_raw_gt_2": exp9_data["part_b_multiturn_cbd"]["overall"]["forward_raw_gt_2"],
            "forward_cbd_violations": exp9_data["part_b_multiturn_cbd"]["overall"]["forward_cbd_violations"],
            "forward_c_cbd_max": exp9_data["part_b_multiturn_cbd"]["overall"]["forward_c_cbd_max"],
        },
        "part_a_frustration": part_a,
        "part_b_detection_loophole": part_b,
        "part_c_spectral_meff": part_c,
        "runtime_seconds": float(elapsed),
    }

    json_path = RESULTS_DIR / "exp11_paper2_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig_path = FIGURES_DIR / "fig11_autoregressive_frustration_transition.png"
    plot_figure_11(results, exp9_cycles, fig_path)

    print("\n" + "=" * 78)
    print("SUMMARY — EXPERIMENT 11 (PAPER 2 LIVE VERIFICATION ON Qwen2.5-1.5B)")
    for tier in ("tier1_mobius_frustrated", "tier2_natural_nontransitive"):
        s = part_a["by_tier"][tier]
        print(
            f"  [Part A] {tier:27s} | s_odd_max={s['forward_s_odd_max']:.3f} | "
            f"Delta0_mean={s['forward_delta_0_mean']:.3f} | C_CbD max (F/R/S)="
            f"{s['forward_c_cbd_max']:.3f}/{s['reverse_c_cbd_max']:.3f}/{s['symmetrized_c_cbd_max']:.3f} | "
            f"CbD>2 (F/R/S)={s['forward_cbd_violations']}/{s['reverse_cbd_violations']}/{s['symmetrized_cbd_violations']} out of 15"
        )
    print(
        f"  [Part B] Detection Loophole : eta=1.0 S_post={part_b['s_post_setting_guardrail'][0]:.4f} | "
        f"eta=0.8284 S_post={part_b['s_post_setting_guardrail'][4]:.4f} | "
        f"eta=0.55 S_post={part_b['s_post_setting_guardrail'][-1]:.4f} | "
        f"Loophole-Free max={max(part_b['s_closed_setting_guardrail']):.4f}"
    )
    print(f"  Saved JSON   -> {json_path}")
    print(f"  Saved Figure -> {fig_path}")
    print(f"ALL ASSERTIONS PASSED (EXIT 0) in {elapsed:.2f} s.")
    print("=" * 78)


if __name__ == "__main__":
    main()
