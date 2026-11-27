#!/usr/bin/env python3
"""Experiment 8 — Real LLM Prompt Dredging & Frozen-Weight Entropy Contrast (H2.4).

Runs real deterministic next-token logit evaluations on the locally cached
Apple Silicon MLX model ``mlx-community/Qwen2.5-1.5B-Instruct-4bit`` (Metal GPU):

1. Exact frozen-weight invariance audit: computes full SHA-256 hash and L2 norm
   over all 732 parameter tensors before and after all 57,600 evaluations
   (asserting ||Delta theta*|| = 0 identically).
2. Systematic grid of M = 200 real candidate prompts (5 personas x 5 instructions
   x 4 few-shot configurations/orderings x 2 delimiters, cf. Lu et al., 2022)
   evaluated via exact binary next-token probability P("Yes") / (P("Yes") + P("No"))
   on two strictly controlled balanced 144-item datasets (n_val = 24 frozen past,
   n_test = 120 locked future):
   - Regime 1 (Null Signal): 144 neutral/ambiguous factual statements with
     independent random 50/50 labels (true Bayes accuracy = 50.0%). Quantifies
     how post-selecting the best prompt among M in {1, 5, 10, 25, 50, 100, 200}
     inflates validation accuracy along the extreme-value scaling law while test
     accuracy remains locked at ~50%.
   - Regime 2 (Real Subtle NLP Task): 144 original nuanced/subtle sentiment
     statements (72 subtly positive, 72 subtly negative). Measures genuine prompt
     sensitivity spread, optimism gap Acc_val(p*) - Acc_test(p*) vs. M, effective
     number of independent prompts M_eff from inter-prompt correlation, and
     calibration via a Deflated Prompt Score (extreme-value & Hoeffding bounds).
3. Frozen-weight vocabulary entropy contrast: compares full 151,936-token next-token
   Shannon entropy and binary Yes/No entropy across unconstrained neutral prompts,
   neutral binary prompts, and strongly directive prompts at ||Delta theta*|| = 0.

Outputs:
- ``sim/results/exp8_llm_prompt_dredging.json``
- ``sim/figures/fig8_real_llm_prompt_dredging.png`` (300 DPI, 4 panels)

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


def configure_matplotlib() -> None:
    """Configure publication-grade Matplotlib styling."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.8,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
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


# ---------------------------------------------------------------------------
# Controlled Datasets (144 items each, concise and genuinely distinct)
# ---------------------------------------------------------------------------
def build_null_signal_dataset(rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """Build 144 neutral/ambiguous statements with independent random 50/50 labels.

    Labels are strictly balanced in both val (first 24: 12 pos, 12 neg) and
    test (last 120: 60 pos, 60 neg), completely independent of text content.
    """
    subjects = [
        "The regional train", "The library committee", "The afternoon ferry",
        "The municipal archive", "The building inspector", "The warehouse clerk",
        "The printing press", "The weather station", "The harbor office",
        "The botany journal", "The transit authority", "The local bakery",
        "The postal van", "The engineering team", "The museum curator",
        "The water utility", "The forestry service", "The acoustic lab",
    ]
    actions = [
        "updated the quarterly timetable on Tuesday",
        "moved three filing cabinets to the east wing",
        "logged the temperature reading near midday",
        "issued a standard revision to section four",
        "counted the inventory crates before noon",
        "scheduled a routine inspection for Thursday",
        "archived the paper ledger in folder nine",
        "checked the calibration dial after lunch",
    ]
    # 18 * 8 = 144 unique neutral factual statements with zero sentiment signal
    texts = [f"{s} {a}." for s in subjects for a in actions]
    assert len(texts) == 144 and len(set(texts)) == 144

    # Shuffle sentence order deterministically, then assign balanced independent random labels
    order = rng.permutation(144)
    texts = [texts[i] for i in order]
    val_labels = np.array([1] * 12 + [0] * 12, dtype=np.int32)
    test_labels = np.array([1] * 60 + [0] * 60, dtype=np.int32)
    rng.shuffle(val_labels)
    rng.shuffle(test_labels)
    labels = np.concatenate([val_labels, test_labels])
    return texts, labels


def build_subtle_sentiment_dataset(rng: np.random.Generator) -> tuple[list[str], np.ndarray]:
    """Build 144 original nuanced/subtle sentiment sentences (72 pos, 72 neg).

    Intentionally understated, hedged, or mixed-clause so real prompt wording
    sensitivity produces realistic accuracy variance across M = 200 prompts.
    """
    subtle_pos = [
        "Despite the slow opening chapter, the novel leaves a lingering warmth.",
        "The cafe looks plain outside, yet the broth carries surprising depth.",
        "It took two attempts to assemble, though the frame feels sturdy now.",
        "The seats are slightly narrow, still the acoustics make every note shine.",
        "Not the flashiest phone on the shelf, but its battery outlasts expectations.",
        "She spoke softly at first before winning over the entire seminar room.",
        "The trail starts muddy, however the ridge view rewards the climb.",
        "Parking was inconvenient, yet the host handled our table with grace.",
        "The interface seems dated, nonetheless every workflow completes reliably.",
        "He arrived five minutes late and still clarified the hardest equation.",
        "The soup looked modest, though the herbs balanced it remarkably well.",
        "Prices rose slightly this season, yet the craftsmanship justifies the cost.",
        "The first draft felt rough, still the revised conclusion lands cleanly.",
        "Rain soaked our jackets, but the guide kept spirits genuinely high.",
        "The hotel lobby is small, however the rooms are spotless and quiet.",
        "It lacks luxury trim, yet the engine runs smoothly on steep hills.",
        "The play runs long, though the final act earns its applause.",
        "Waiting in line tested our patience, still the pastry tasted unforgettable.",
        "The manual is dense, yet the troubleshooting diagram saves hours.",
        "Our flight faced headwinds, however the crew made the cabin restful.",
        "The fabric wrinkles easily, yet the cut flatters almost every frame.",
        "Lighting in the gallery is dim, still the brushwork commands attention.",
        "The lecture moved quickly, though her analogies made the proof click.",
        "It costs a bit more, yet the warranty covers every realistic mishap.",
        "The cover design is understated, however the essays inside feel sharp.",
        "Delivery took an extra day, still the packaging protected the glass.",
        "The sauce needed salt, yet the roasted vegetables carried the plate.",
        "Their booth was tucked in back, though their demo outperformed rivals.",
        "The forecast looked bleak, yet the clouds parted just before kickoff.",
        "His tone sounded dry, still his feedback rescued our project timeline.",
        "The bike feels heavy uphill, however it glides effortlessly on gravel.",
        "Setup required patience, yet the soundstage fills the living room nicely.",
        "The decor is sparse, still the kitchen delivers honest seasonal flavors.",
        "Our seats sat near the aisle, yet the sightline remained unobstructed.",
        "The patch notes were brief, however the update fixed every crash.",
        "The coat feels stiff initially, still it blocks freezing wind completely.",
        "The plot takes detours, yet the characters grow convincingly by the end.",
        "Service paused briefly during rush hour, though the manager apologized warmly.",
        "The camera struggles at midnight, still daylight portraits look crisp.",
        "The workshop ran overtime, yet every attendee left with working code.",
        "The coffee tastes bold, however the finish remains remarkably smooth.",
        "Check-in felt formal, still the staff remembered our morning preference.",
        "The sequel feels quieter, yet its emotional core resonates more deeply.",
        "The desk took effort to lift, though it never wobbles under load.",
        "Some jokes miss the mark, still the ensemble chemistry carries the show.",
        "The route winds through traffic, yet the coastal stretch feels serene.",
        "The battery charges slowly, however standby drain is barely noticeable.",
        "The crust looked pale, yet the crumb had an airy artisan chew.",
        "Her slides had minimal graphics, still her argument persuaded the panel.",
        "The cabin stays cool at dawn, yet the fireplace warms it fast.",
        "The lens weighs quite a bit, however its sharpness justifies carrying it.",
        "The soundtrack stays subdued, still it elevates every quiet scene.",
        "We waited twenty minutes, yet the mechanic solved the rattle permanently.",
        "The book avoids easy drama, though its restraint makes it memorable.",
        "The chair lacks armrests, still its lumbar support eases long afternoons.",
        "The wine opens tightly, however twenty minutes of air unlocks rich fruit.",
        "The gym is compact, yet every machine is maintained to perfection.",
        "The game starts unforgiving, still mastering its rhythm feels deeply rewarding.",
        "The courtyard looked narrow, yet morning sunlight turns it inviting.",
        "His report was blunt, however its recommendations doubled our throughput.",
        "The ferry rocked gently, still the crossing felt peaceful and scenic.",
        "The pen looks ordinary, yet the ink flows without a single skip.",
        "The album skips radio hooks, though repeated listens reveal intricate craft.",
        "The soup arrived piping hot, and the subtle ginger brightened each spoonful.",
        "The mattress feels firm at first, still my back felt rested by dawn.",
        "The map looks crowded, yet the color coding makes transfers intuitive.",
        "The garden looked overgrown in spring, however summer blooms transformed it.",
        "The watch has no smart features, still its dial reads effortlessly.",
        "The debate grew tense, yet both speakers reached a constructive compromise.",
        "The keyboard sounds quiet, still its tactile feedback makes typing brisk.",
        "The museum skips spectacle, however its curated letters feel deeply moving.",
        "The umbrella looks slim, yet it held firm through sideways gusts.",
    ]

    subtle_neg = [
        "The trailer promised tension, yet the movie drifts without real stakes.",
        "The dining room looks elegant, however the pasta arrived lukewarm and bland.",
        "It boasts sleek metal casing, still the hinge already creaks after a week.",
        "The speaker sounded confident, though his slides dodged the core objection.",
        "The brochure highlights mountain views, yet our window faced a brick wall.",
        "Packaging looked premium, however the headphones hissed whenever silence returned.",
        "She greeted us politely, still the appointment felt rushed and dismissive.",
        "The opening track hooks you, yet the remaining songs blur into filler.",
        "The resort sits beside the bay, though construction noise lasted all afternoon.",
        "The coat looks tailored online, however the sleeves bind across the shoulders.",
        "The app launches quickly, still routine exports freeze without an error message.",
        "He quoted impressive figures, yet none of the footnotes matched the table.",
        "The bakery smells wonderful, however the croissant tasted stale and dry.",
        "The sofa looks plush in photos, still the cushions flatten within minutes.",
        "Their pitch sounded ambitious, yet the prototype failed basic stress checks.",
        "The waiter smiled constantly, however our main course never reached the table.",
        "The cover promises a thriller, still half the chapters stall in small talk.",
        "The battery reads full at noon, yet it drops abruptly before dinner.",
        "The room appeared tidy at glance, however dust coated every high shelf.",
        "The manual lists twenty features, still the core timer drifts by minutes.",
        "The museum looks grand outside, yet most galleries were closed for renovation.",
        "The coffee pours rich crema, however a burnt bitterness lingers afterward.",
        "The jacket feels featherlight, still light drizzle soaked straight through the seams.",
        "The lecture cited famous names, yet it left the central paradox untouched.",
        "Our guide spoke enthusiastically, however we spent two hours stuck in gift shops.",
        "The monitor displays vivid colors, still dead pixels appeared in the corner.",
        "The sauce looked glossy, yet salt overwhelmed every other ingredient.",
        "The game renders gorgeous skies, however repetitive missions drain the fun.",
        "The landlord seemed accommodating, still minor repairs waited for three months.",
        "The novel opens with wit, yet the ending collapses into rushed cliches.",
        "The desk looks minimalist, however the laminate chips at the slightest bump.",
        "The airline advertised extra legroom, still the tray table barely folded flat.",
        "The tea smells floral, yet the flavor turns watery after one sip.",
        "The tutorial looks clean, however deprecated commands break on step two.",
        "The theater renovated its lobby, still the projector stayed noticeably dim.",
        "The shoes felt soft in store, yet the heel blistered within two blocks.",
        "The report used polished charts, however the sample excluded every failure.",
        "The kettle boils rapidly, still the lid leaks steam onto your knuckles.",
        "The wine label looks prestigious, yet the palate tastes thin and sour.",
        "The workshop advertised hands-on practice, however we only watched prerecorded clips.",
        "The blanket looks thick, still loose threads shed across the entire bed.",
        "The neighborhood stays quiet, yet the thin walls carry every hallway footstep.",
        "The update redesigned every icon, however battery life dropped noticeably afterward.",
        "The salad looked fresh, yet soggy greens sat hidden beneath the garnish.",
        "The play featured renowned actors, still the pacing dragged past intermission.",
        "The backpack carries many pockets, however the main zipper snags on every pull.",
        "The doctor ran thorough tests, still nobody called back with the findings.",
        "The bicycle shifts crisply, yet the brakes squeal loudly in dry weather.",
        "The essay reads smoothly, however its thesis contradicts its own evidence.",
        "The toaster gleams on the counter, still it burns one side while leaving bread pale.",
        "The hotel offered complimentary breakfast, yet the trays stood empty by eight.",
        "The pen feels balanced, however smudges appear across every turned page.",
        "The documentary filmed rare footage, still the narration feels awkwardly scripted.",
        "The fan looks whisper-quiet, yet a metallic rattle hums at low speed.",
        "The course promised mentorship, however questions went unanswered for weeks.",
        "The pizza crust looked blistered, yet the center turned soggy immediately.",
        "The phone takes bright snapshots, still skin tones look artificially waxy.",
        "The park offers wide lawns, however overflowing bins line the main walkway.",
        "The album hired top producers, yet every melody feels strangely hollow.",
        "The chair swivels effortlessly, still the gas lift sinks every ten minutes.",
        "The seminar gathered senior experts, however the panel repeated textbook basics.",
        "The sweater feels cozy initially, yet pilling covered the cuffs after one wash.",
        "The train runs on schedule, still broken ventilation made the carriage stifling.",
        "The glaze looked glossy, however the cake inside crumbled completely dry.",
        "The router claims whole-home range, yet signal drops two rooms away.",
        "The memoir shares vivid anecdotes, still the reflection stays frustratingly shallow.",
        "The umbrella opens with one click, however a modest breeze bent two ribs.",
        "The exhibit displayed rare maps, yet tiny faded labels prevented reading them.",
        "The cream smells pleasant, still it leaves a sticky residue for hours.",
        "The software promises automation, however manual fixes consume half the week.",
        "The resort pool looked inviting, yet icy water kept every guest ashore.",
        "The sequel raised the budget, still the script lost all original charm.",
    ]

    assert len(subtle_pos) == 72 and len(set(subtle_pos)) == 72
    assert len(subtle_neg) == 72 and len(set(subtle_neg)) == 72

    pos_idx = rng.permutation(72)
    neg_idx = rng.permutation(72)

    # First 24 = strictly balanced val (12 pos + 12 neg); remaining 120 = balanced test (60 + 60)
    val_items = [(subtle_pos[i], 1) for i in pos_idx[:12]] + [(subtle_neg[i], 0) for i in neg_idx[:12]]
    test_items = [(subtle_pos[i], 1) for i in pos_idx[12:]] + [(subtle_neg[i], 0) for i in neg_idx[12:]]
    rng.shuffle(val_items)
    rng.shuffle(test_items)
    all_items = val_items + test_items
    texts = [t for t, _ in all_items]
    labels = np.array([y for _, y in all_items], dtype=np.int32)
    return texts, labels


# ---------------------------------------------------------------------------
# Systematic Grid of M = 200 Real Candidate Prompts
# ---------------------------------------------------------------------------
def build_200_prompts() -> list[dict[str, str]]:
    """Construct M = 200 distinct real prompts (5 x 5 x 4 x 2 factorial grid)."""
    personas = [
        ("neutral", "You are a helpful text classifier."),
        ("strict_critic", "You are a skeptical, demanding quality reviewer."),
        ("optimist", "You are a generous, appreciative reader who notices bright sides."),
        ("linguist", "You are an academic linguist analyzing pragmatic polarity."),
        ("concise_judge", "You are a strict binary judge."),
    ]
    instructions = [
        ("overall_pos", "Decide if the statement expresses overall positive sentiment. Reply Yes or No."),
        ("favorable", "Is the writer's final impression favorable? Answer strictly Yes or No."),
        ("recommend", "Would this remark indicate satisfaction rather than disappointment? Answer Yes or No."),
        ("tone_up", "Does the sentence end on an approving or encouraging note? Output Yes or No."),
        ("valence", "Classify whether the net emotional valence is positive (Yes) or negative (No)."),
    ]
    few_shots = [
        ("zero_shot", ""),
        (
            "2shot_pos_neg",
            "Ex1: 'Small room, yet spotless and restful.' -> Yes\n"
            "Ex2: 'Nice view, yet noisy and tiring.' -> No\n",
        ),
        (
            "2shot_neg_pos",
            "Ex1: 'Nice view, yet noisy and tiring.' -> No\n"
            "Ex2: 'Small room, yet spotless and restful.' -> Yes\n",
        ),
        (
            "2shot_subtle_np",
            "Ex1: 'Polished lobby, still the meal tasted bland.' -> No\n"
            "Ex2: 'Plain storefront, though the bread was superb.' -> Yes\n",
        ),
    ]
    delimiters = [
        ("colon", "Statement: ", "\nPositive (Yes/No):"),
        ("brackets", "Input: [", "]\nLabel:"),
    ]

    prompts = []
    idx = 0
    for p_name, p_sys in personas:
        for i_name, i_txt in instructions:
            for f_name, f_txt in few_shots:
                for d_name, d_open, d_close in delimiters:
                    prefix = (
                        f"<|im_start|>system\n{p_sys} {i_txt}<|im_end|>\n"
                        f"<|im_start|>user\n{f_txt}{d_open}"
                    )
                    suffix_tail = f"{d_close}<|im_end|>\n<|im_start|>assistant\n"
                    prompts.append(
                        {
                            "id": idx,
                            "name": f"{p_name}__{i_name}__{f_name}__{d_name}",
                            "persona": p_name,
                            "instruction": i_name,
                            "few_shot": f_name,
                            "delimiter": d_name,
                            "prefix": prefix,
                            "suffix_tail": suffix_tail,
                        }
                    )
                    idx += 1
    assert len(prompts) == 200
    return prompts


# ---------------------------------------------------------------------------
# Fast Exact KV-Cache Prefix Evaluation on Metal GPU
# ---------------------------------------------------------------------------
def evaluate_prompt_over_items(
    model: Any,
    tok: Any,
    prefix_text: str,
    suffix_tail: str,
    items: list[str],
    yes_id: int,
    no_id: int,
    chunk_size: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact P(Yes | prompt, item) and Yes+No vocab mass for all items.

    Uses single prefix KV-cache prefill + length-sorted right-padded causal
    chunks (zero mathematical leakage from right-padding due to causal mask).
    """
    prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
    # Encode all suffixes and sort by length to minimize wasted right-padding
    suf_encoded = [tok.encode(item + suffix_tail, add_special_tokens=False) for item in items]
    n_items = len(items)
    order = sorted(range(n_items), key=lambda i: len(suf_encoded[i]))

    # Prefill shared prefix ONCE
    base_cache = make_prompt_cache(model)
    _ = model.model(mx.array([prefix_ids], dtype=mx.int32), cache=base_cache)
    mx.eval([c.state for c in base_cache])

    pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    probs_yes = np.zeros(n_items, dtype=np.float64)
    yn_mass = np.zeros(n_items, dtype=np.float64)

    for start in range(0, n_items, chunk_size):
        end = min(n_items, start + chunk_size)
        idxs = order[start:end]
        cb = len(idxs)
        max_len = max(len(suf_encoded[i]) for i in idxs)
        arr = np.full((cb, max_len), pad_id, dtype=np.int32)
        lens = np.zeros(cb, dtype=np.int32)
        for row, orig_i in enumerate(idxs):
            seq = suf_encoded[orig_i]
            arr[row, : len(seq)] = seq
            lens[row] = len(seq) - 1

        chunk_cache = []
        for c in base_cache:
            nc = KVCache()
            k_cur = c.keys[:, :, : c.offset, :]
            v_cur = c.values[:, :, : c.offset, :]
            nc.keys = mx.repeat(k_cur, cb, axis=0)
            nc.values = mx.repeat(v_cur, cb, axis=0)
            nc.offset = c.offset
            chunk_cache.append(nc)

        h = model.model(mx.array(arr), cache=chunk_cache)
        last_h = h[mx.arange(cb), mx.array(lens), :]
        logits = last_token_logits(model, last_h).astype(mx.float32)

        # Log-softmax for exact Yes+No probability mass and normalized P(Yes)
        lse = mx.logsumexp(logits, axis=-1)
        y_lg = logits[:, yes_id]
        n_lg = logits[:, no_id]
        mass = mx.exp(y_lg - lse) + mx.exp(n_lg - lse)
        p_y = mx.sigmoid(y_lg - n_lg)
        mx.eval(p_y, mass)

        p_y_np = np.asarray(p_y, dtype=np.float64)
        mass_np = np.asarray(mass, dtype=np.float64)
        for row, orig_i in enumerate(idxs):
            probs_yes[orig_i] = p_y_np[row]
            yn_mass[orig_i] = mass_np[row]

    return probs_yes, yn_mass


def measure_entropy_contrast(
    model: Any, tok: Any, sample_items: list[str], yes_id: int, no_id: int
) -> dict[str, Any]:
    """Measure full-vocabulary and binary Shannon entropy across prompt regimes at ||dtheta*||=0."""
    prompt_regimes = {
        "unconstrained_open": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nComment briefly on this remark: ",
            "<|im_end|>\n<|im_start|>assistant\n",
        ),
        "neutral_binary": (
            "<|im_start|>system\nDecide if the statement expresses positive sentiment. Reply Yes or No.<|im_end|>\n"
            "<|im_start|>user\nStatement: ",
            "\nAnswer:<|im_end|>\n<|im_start|>assistant\n",
        ),
        "directive_positive": (
            "<|im_start|>system\nYou are an enthusiastic optimist who almost always finds silver linings. "
            "Respond with a single word: Yes or No.<|im_end|>\n"
            "<|im_start|>user\nDoes this show anything positive? Statement: ",
            "\nOne-word answer (Yes/No):<|im_end|>\n<|im_start|>assistant\n",
        ),
        "directive_strict": (
            "<|im_start|>system\nYou are an ultra-strict quality auditor who flags any imperfection as unacceptable. "
            "Respond with a single word: Yes or No.<|im_end|>\n"
            "<|im_start|>user\nIs this completely flawless? Statement: ",
            "\nOne-word answer (Yes/No):<|im_end|>\n<|im_start|>assistant\n",
        ),
    }

    out = {}
    for regime_name, (pref, tail) in prompt_regimes.items():
        h_vocab_list = []
        h_bin_list = []
        p_yes_list = []
        yn_mass_list = []
        for item in sample_items:
            ids = tok.encode(pref + item + tail, add_special_tokens=False)
            h = model.model(mx.array([ids], dtype=mx.int32))
            lg = last_token_logits(model, h[:, -1, :])[0].astype(mx.float32)
            log_p = lg - mx.logsumexp(lg)
            p_full = mx.exp(log_p)
            h_v = -mx.sum(p_full * log_p) / math.log(2.0)
            y_lg = lg[yes_id]
            n_lg = lg[no_id]
            p_y = mx.sigmoid(y_lg - n_lg)
            mass = mx.exp(log_p[yes_id]) + mx.exp(log_p[no_id])
            mx.eval(h_v, p_y, mass)

            pv = float(p_y.item())
            pv_clip = min(max(pv, 1e-12), 1.0 - 1e-12)
            h_b = -(pv_clip * math.log2(pv_clip) + (1.0 - pv_clip) * math.log2(1.0 - pv_clip))
            h_vocab_list.append(float(h_v.item()))
            h_bin_list.append(h_b)
            p_yes_list.append(pv)
            yn_mass_list.append(float(mass.item()))

        out[regime_name] = {
            "mean_vocab_entropy_bits": float(np.mean(h_vocab_list)),
            "std_vocab_entropy_bits": float(np.std(h_vocab_list, ddof=1)),
            "mean_binary_entropy_bits": float(np.mean(h_bin_list)),
            "mean_p_yes": float(np.mean(p_yes_list)),
            "mean_yes_no_mass": float(np.mean(yn_mass_list)),
        }
    return out


# ---------------------------------------------------------------------------
# Post-Selection Scaling & Deflated Prompt Score Analytics
# ---------------------------------------------------------------------------
def expected_max_z(m_eff: float) -> float:
    """Bailey-Lopez de Prado extreme-value expectation E[max_{1..M} Z_i]."""
    if m_eff <= 1.0:
        return 0.0
    euler_gamma = 0.5772156649015329
    return float(
        (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / m_eff)
        + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (m_eff * math.e))
    )


def estimate_effective_trials(corr_matrix: np.ndarray, m: int) -> float:
    """Estimate effective number of independent prompts M_eff from pairwise correlation.

    Uses the Nyholt / Gao eigenvalue-variance effective test formula on the
    (M x M) prompt correlation matrix: M_eff = 1 + (M - 1) * (1 - Var(eig) / M).
    """
    if m <= 1:
        return 1.0
    sub = corr_matrix[:m, :m]
    eigs = np.linalg.eigvalsh(sub)
    eigs = np.maximum(eigs, 0.0)
    var_eigs = float(np.var(eigs))
    m_eff = 1.0 + (m - 1.0) * max(0.0, 1.0 - var_eigs / m)
    return float(max(1.0, min(float(m), m_eff)))


def analyze_dredging_curve(
    probs_matrix: np.ndarray,
    labels: np.ndarray,
    m_grid: list[int],
    n_val: int,
    rng: np.random.Generator,
    n_resamples: int = 200,
) -> dict[str, Any]:
    """Compute canonical split + 200-resample dredging scaling curves for a (200, 144) regime."""
    preds = (probs_matrix >= 0.5).astype(np.int32)
    correct = (preds == labels[None, :]).astype(np.float64)  # (200, 144)

    # Full-dataset accuracy per prompt (for sensitivity distribution)
    acc_all = np.mean(correct, axis=1)
    acc_val_can = np.mean(correct[:, :n_val], axis=1)
    acc_test_can = np.mean(correct[:, n_val:], axis=1)

    # Inter-prompt correlation on correctness indicators (for M_eff)
    # Guard constant rows by adding tiny jitter only for correlation matrix stability
    c_centered = correct - np.mean(correct, axis=1, keepdims=True)
    stds = np.std(correct, axis=1, keepdims=True)
    valid_rows = (stds[:, 0] > 1e-9)
    if np.sum(valid_rows) >= 2:
        normed = c_centered[valid_rows] / stds[valid_rows]
        corr_mat = (normed @ normed.T) / correct.shape[1]
        off_diag = corr_mat[~np.eye(corr_mat.shape[0], dtype=bool)]
        mean_pairwise_corr = float(np.mean(off_diag))
    else:
        corr_mat = np.eye(probs_matrix.shape[0])
        mean_pairwise_corr = 0.0

    # Prepare balanced random splits (24 val: 12 pos, 12 neg; 120 test: 60 pos, 60 neg)
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    n_val_half = n_val // 2

    val_means = []
    val_stds = []
    test_means = []
    test_stds = []
    gap_means = []
    gap_stds = []
    deflated_ev_means = []
    hoeffding_lb_means = []
    can_val_list = []
    can_test_list = []
    m_eff_list = []

    n_prompts = probs_matrix.shape[0]
    # Pre-generate 200 balanced splits and prompt permutations so nested M values are coupled
    split_val_masks = []
    prompt_perms = []
    for _ in range(n_resamples):
        p_shuf = rng.permutation(pos_indices)
        n_shuf = rng.permutation(neg_indices)
        v_idx = np.concatenate([p_shuf[:n_val_half], n_shuf[:n_val_half]])
        t_idx = np.concatenate([p_shuf[n_val_half:], n_shuf[n_val_half:]])
        split_val_masks.append((v_idx, t_idx))
        prompt_perms.append(rng.permutation(n_prompts))

    # Empirical val std across prompts for deflated prompt score
    for m in m_grid:
        # Canonical split using first m prompts
        best_can = int(np.argmax(acc_val_can[:m]))
        can_val_list.append(float(acc_val_can[best_can]))
        can_test_list.append(float(acc_test_can[best_can]))

        m_eff = estimate_effective_trials(corr_mat, min(m, corr_mat.shape[0]))
        m_eff_list.append(m_eff)

        v_scores = np.zeros(n_resamples)
        t_scores = np.zeros(n_resamples)
        defl_scores = np.zeros(n_resamples)
        hoeff_scores = np.zeros(n_resamples)

        ez = expected_max_z(m_eff)
        hoeff_pen = math.sqrt(math.log(max(m, 1) / 0.05) / (2.0 * n_val)) if m > 1 else 0.0

        for r in range(n_resamples):
            v_idx, t_idx = split_val_masks[r]
            p_sub = prompt_perms[r][:m]
            v_acc_m = np.mean(correct[p_sub][:, v_idx], axis=1)
            t_acc_m = np.mean(correct[p_sub][:, t_idx], axis=1)
            best_local = int(np.argmax(v_acc_m))
            v_best = float(v_acc_m[best_local])
            t_best = float(t_acc_m[best_local])
            v_scores[r] = v_best
            t_scores[r] = t_best

            # Extreme-value Deflated Prompt Score:
            # Estimate binomial sampling SE at observed accuracy (~sqrt(p(1-p)/n_val))
            se_bin = math.sqrt(max(v_best * (1.0 - v_best), 0.05) / n_val)
            defl_scores[r] = max(0.0, v_best - se_bin * ez)
            hoeff_scores[r] = max(0.0, v_best - hoeff_pen)

        gaps = v_scores - t_scores
        val_means.append(float(np.mean(v_scores)))
        val_stds.append(float(np.std(v_scores, ddof=1)))
        test_means.append(float(np.mean(t_scores)))
        test_stds.append(float(np.std(t_scores, ddof=1)))
        gap_means.append(float(np.mean(gaps)))
        gap_stds.append(float(np.std(gaps, ddof=1)))
        deflated_ev_means.append(float(np.mean(defl_scores)))
        hoeffding_lb_means.append(float(np.mean(hoeff_scores)))

    # Canonical M=200 naive binomial p-value vs H0: acc <= 0.5
    best_can_200 = int(np.argmax(acc_val_can))
    k_val_best = int(round(acc_val_can[best_can_200] * n_val))
    p_val_naive = float(stats.binomtest(k_val_best, n_val, 0.5, alternative="greater").pvalue)
    p_val_bonf = float(min(1.0, 200.0 * p_val_naive))
    k_test_best = int(round(acc_test_can[best_can_200] * (len(labels) - n_val)))
    p_test_best = float(
        stats.binomtest(k_test_best, len(labels) - n_val, 0.5, alternative="greater").pvalue
    )

    return {
        "m_grid": m_grid,
        "m_eff_by_m": m_eff_list,
        "mean_pairwise_prompt_corr": mean_pairwise_corr,
        "all_prompts_acc_summary": {
            "min": float(np.min(acc_all)),
            "p25": float(np.percentile(acc_all, 25)),
            "median": float(np.median(acc_all)),
            "mean": float(np.mean(acc_all)),
            "std": float(np.std(acc_all, ddof=1)),
            "p75": float(np.percentile(acc_all, 75)),
            "max": float(np.max(acc_all)),
        },
        "resample_val_acc_mean": val_means,
        "resample_val_acc_std": val_stds,
        "resample_test_acc_mean": test_means,
        "resample_test_acc_std": test_stds,
        "optimism_gap_mean": gap_means,
        "optimism_gap_std": gap_stds,
        "deflated_ev_score_mean": deflated_ev_means,
        "hoeffding_lb_mean": hoeffding_lb_means,
        "canonical_val_acc_by_m": can_val_list,
        "canonical_test_acc_by_m": can_test_list,
        "canonical_m200_best_prompt_id": best_can_200,
        "canonical_m200_val_hits": f"{k_val_best}/{n_val}",
        "canonical_m200_val_acc": float(acc_val_can[best_can_200]),
        "canonical_m200_test_hits": f"{k_test_best}/{len(labels) - n_val}",
        "canonical_m200_test_acc": float(acc_test_can[best_can_200]),
        "canonical_m200_naive_p_val": p_val_naive,
        "canonical_m200_bonf_p_val": p_val_bonf,
        "canonical_m200_p_test": p_test_best,
        "all_prompts_acc_raw": acc_all.tolist(),
    }


def plot_figure_8(res: dict[str, Any], out_path: Path) -> None:
    """Render 4-panel publication-grade Figure 8."""
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.9), constrained_layout=True)
    fig.suptitle(
        "Experiment 8 — Real LLM Prompt Dredging on Frozen Qwen2.5-1.5B-Instruct (Apple Silicon MLX, H2.4)",
        fontsize=12.4,
        fontweight="bold",
        y=1.02,
    )

    m_arr = np.array(res["regime1_null"]["m_grid"], dtype=float)
    n_val = res["n_val"]

    # Panel (a): Regime 1 — Null Signal (Random Balanced Labels)
    ax = axes[0, 0]
    r1 = res["regime1_null"]
    v1_m = np.array(r1["resample_val_acc_mean"]) * 100.0
    v1_s = np.array(r1["resample_val_acc_std"]) * 100.0
    t1_m = np.array(r1["resample_test_acc_mean"]) * 100.0
    t1_s = np.array(r1["resample_test_acc_std"]) * 100.0
    d1_m = np.array(r1["deflated_ev_score_mean"]) * 100.0
    m_eff1 = np.array(r1["m_eff_by_m"])
    # Theoretical binomial extreme-value null curve using M_eff
    th_null = (0.5 + np.array([expected_max_z(me) for me in m_eff1]) * math.sqrt(0.25 / n_val)) * 100.0

    ax.errorbar(
        m_arr, v1_m, yerr=v1_s, fmt="o-", color=PALETTE["coral"], lw=2.2, capsize=3.5,
        label=f"Post-Selected Val Acc ($n_{{\\mathrm{{val}}}}={n_val}$, 200 splits)",
    )
    ax.plot(
        m_arr, th_null, color="#1E293B", ls="--", lw=1.4,
        label=r"Null EV Law $50\% + \frac{1}{2\sqrt{n}}\mathbb{E}[\max Z_{M_{\mathrm{eff}}}]$",
    )
    ax.plot(
        m_arr, d1_m, color=PALETTE["teal"], ls="-.", lw=1.9, marker="^", ms=4.5,
        label="Deflated Prompt Score (EV-corrected)",
    )
    ax.errorbar(
        m_arr, t1_m, yerr=t1_s, fmt="s-", color=PALETTE["navy"], lw=2.2, capsize=3.5,
        label=f"Locked Test Acc ($n_{{\\mathrm{{test}}}}={res['n_test']}$)",
    )
    ax.axhline(50.0, color=PALETTE["slate"], ls=":", lw=1.1, label="True Bayes Ceiling (50.0%)")
    ax.set_xscale("log")
    ax.set_xticks(m_arr)
    ax.set_xticklabels([str(int(x)) for x in m_arr])
    ax.set_title(f"(a) Regime 1: Pure Null Signal (Random Labels, $\\bar{{\\rho}}={r1['mean_pairwise_prompt_corr']:.2f}$)")
    ax.set_xlabel("Candidate Prompt Pool Size $M$ (log scale)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_ylim(38.0, 82.0)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=7.6)

    # Panel (b): Regime 2 — Real Subtle Sentiment Task
    ax = axes[0, 1]
    r2 = res["regime2_subtle_sentiment"]
    v2_m = np.array(r2["resample_val_acc_mean"]) * 100.0
    v2_s = np.array(r2["resample_val_acc_std"]) * 100.0
    t2_m = np.array(r2["resample_test_acc_mean"]) * 100.0
    t2_s = np.array(r2["resample_test_acc_std"]) * 100.0
    d2_m = np.array(r2["deflated_ev_score_mean"]) * 100.0

    ax.errorbar(
        m_arr, v2_m, yerr=v2_s, fmt="o-", color=PALETTE["coral"], lw=2.2, capsize=3.5,
        label=f"Post-Selected Val Acc ($n_{{\\mathrm{{val}}}}={n_val}$)",
    )
    ax.errorbar(
        m_arr, t2_m, yerr=t2_s, fmt="s-", color=PALETTE["navy"], lw=2.2, capsize=3.5,
        label=f"Locked Test Acc ($n_{{\\mathrm{{test}}}}={res['n_test']}$)",
    )
    ax.plot(
        m_arr, d2_m, color=PALETTE["teal"], ls="-.", lw=2.0, marker="^", ms=4.5,
        label="Deflated Prompt Score (EV-corrected)",
    )
    ax.fill_between(m_arr, t2_m, v2_m, color=PALETTE["coral"], alpha=0.11, label="Optimism Gap ($\\mathrm{Acc}_{\\mathrm{val}} - \\mathrm{Acc}_{\\mathrm{test}}$)")

    gap200 = v2_m[-1] - t2_m[-1]
    ax.annotate(
        f"$M=200$:\nVal = {v2_m[-1]:.1f}%\nTest = {t2_m[-1]:.1f}%\nGap = +{gap200:.1f} pp",
        xy=(200, 0.5 * (v2_m[-1] + t2_m[-1])),
        xytext=(5.5, 76.5),
        fontsize=8.0,
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PALETTE["bg_tint"], edgecolor="#94A3B8", lw=0.8),
        arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=1.1),
    )

    ax.set_xscale("log")
    ax.set_xticks(m_arr)
    ax.set_xticklabels([str(int(x)) for x in m_arr])
    ax.set_title(f"(b) Regime 2: Real Subtle Sentiment ($M=200$ Prompts, $\\bar{{\\rho}}={r2['mean_pairwise_prompt_corr']:.2f}$)")
    ax.set_xlabel("Candidate Prompt Pool Size $M$ (log scale)")
    ax.set_ylabel("Classification Accuracy (%)")
    y_min_b = max(35.0, min(float(np.min(d2_m)), float(np.min(t2_m - t2_s))) - 10.0)
    y_max_b = min(103.0, float(np.max(v2_m + v2_s)) + 6.0)
    ax.set_ylim(y_min_b, y_max_b)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=7.6)

    # Panel (c): Distribution of Full-Dataset Accuracies Across All 200 Real Prompts
    ax = axes[1, 0]
    acc1 = np.array(r1["all_prompts_acc_raw"]) * 100.0
    acc2 = np.array(r2["all_prompts_acc_raw"]) * 100.0
    bins_c = np.linspace(35, 98, 32)
    ax.hist(
        acc1, bins=bins_c, color=PALETTE["slate"], alpha=0.62, edgecolor="white", linewidth=0.6,
        label=f"Regime 1 Null (mean={np.mean(acc1):.1f}%, sd={np.std(acc1, ddof=1):.1f}%)",
    )
    ax.hist(
        acc2, bins=bins_c, color=PALETTE["navy"], alpha=0.72, edgecolor="white", linewidth=0.6,
        label=f"Regime 2 Subtle Sentiment (mean={np.mean(acc2):.1f}%, sd={np.std(acc2, ddof=1):.1f}%)",
    )
    ax.axvline(50.0, color=PALETTE["coral"], ls="--", lw=1.3, label="Random Chance (50.0%)")
    ax.set_title("(c) Prompt Sensitivity Spread Across All $M=200$ Prompts (Full 144 Items)")
    ax.set_xlabel("Full-Dataset Accuracy of Individual Prompt (%)")
    ax.set_ylabel("Number of Prompts")
    ax.set_xlim(35, 100)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=7.9)

    # Panel (d): Frozen-Weight Entropy Contrast (Real 1.5B LLM, ||Delta theta*|| = 0)
    ax = axes[1, 1]
    ec = res["entropy_contrast"]
    order_keys = ["unconstrained_open", "neutral_binary", "directive_positive", "directive_strict"]
    labels_d = [
        "Open-Ended\nNeutral",
        "Binary\nNeutral",
        "Directive\nOptimist",
        "Directive\nStrict",
    ]
    h_vocab = [ec[k]["mean_vocab_entropy_bits"] for k in order_keys]
    h_bin = [ec[k]["mean_binary_entropy_bits"] for k in order_keys]
    x_d = np.arange(len(order_keys))
    w = 0.35
    b1 = ax.bar(x_d - w / 2, h_vocab, width=w, color=PALETTE["navy"], edgecolor="#1E293B", lw=0.7, label="Full Vocab Entropy $H_{\\mathrm{vocab}}$ (151,936 tokens)")
    b2 = ax.bar(x_d + w / 2, h_bin, width=w, color=PALETTE["amber"], edgecolor="#1E293B", lw=0.7, label="Binary Yes/No Entropy $H_{\\mathrm{bin}}$ (max 1.0 bit)")

    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 0.08,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )

    ax.set_xticks(x_d)
    ax.set_xticklabels(labels_d)
    ax.set_ylabel("Next-Token Shannon Entropy (bits)")
    ax.set_title(r"(d) Real LLM Entropy Collapse Under Frozen Weights ($\|\Delta\theta^*\|_2 = 0$)")
    ax.set_ylim(0, max(h_vocab) * 1.40 + 0.5)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=7.8)

    audit_txt = (
        f"Frozen-Weight Audit ({MODEL_ID.split('/')[-1]}):\n"
        f"• Tensors: {res['weights_before']['num_tensors']} ({res['weights_before']['num_stored_elements']:,} elems)\n"
        f"• SHA-256 identical: {res['weights_sha256_match']}\n"
        f"• ||Δθ*||_2 = {res['weights_l2_diff']:.1f} | Total evals: {res['total_forward_evals']:,}"
    )
    ax.text(
        0.97,
        0.74,
        audit_txt,
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
    print("EXPERIMENT 8 — REAL LLM PROMPT DREDGING ON APPLE SILICON MLX (H2.4)")
    print(f"Platform : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"MLX      : {mx.__version__} ({mx.default_device()}) | mlx_lm {mlx_lm.__version__}")
    print(f"Model    : {MODEL_ID}")
    print("=" * 78)

    model, tok = mlx_lm.load(MODEL_ID)
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    assert yes_id != no_id

    # 1. Exact frozen-weight fingerprint BEFORE all evaluations
    weights_before = fingerprint_frozen_weights(model)
    print(f"[Audit] Weights BEFORE: sha256={weights_before['sha256'][:16]}... L2={weights_before['l2_norm']:.4f}")

    # 2. Build datasets & M = 200 candidate prompts
    null_texts, null_labels = build_null_signal_dataset(rng)
    sent_texts, sent_labels = build_subtle_sentiment_dataset(rng)
    prompts = build_200_prompts()
    all_texts = null_texts + sent_texts  # 288 items evaluated jointly per shared prompt prefix!

    n_prompts = len(prompts)
    n_null = len(null_texts)
    n_sent = len(sent_texts)
    n_val = 24
    n_test = 120

    probs_null = np.zeros((n_prompts, n_null), dtype=np.float64)
    probs_sent = np.zeros((n_prompts, n_sent), dtype=np.float64)
    yn_masses = np.zeros((n_prompts, n_null + n_sent), dtype=np.float64)

    t_eval = time.perf_counter()
    for i, p_cfg in enumerate(prompts):
        p_all, m_all = evaluate_prompt_over_items(
            model=model,
            tok=tok,
            prefix_text=p_cfg["prefix"],
            suffix_tail=p_cfg["suffix_tail"],
            items=all_texts,
            yes_id=yes_id,
            no_id=no_id,
            chunk_size=96,
        )
        probs_null[i] = p_all[:n_null]
        probs_sent[i] = p_all[n_null:]
        yn_masses[i] = m_all
        if (i + 1) % 40 == 0 or i == 0:
            elapsed = time.perf_counter() - t_eval
            rate = (i + 1) / elapsed
            eta = (n_prompts - i - 1) / rate if rate > 0 else 0.0
            print(
                f"  Prompt {i + 1:3d}/{n_prompts} ({elapsed:5.1f}s elapsed, ETA {eta:4.1f}s) | "
                f"Null acc={np.mean((probs_null[i]>=0.5)==null_labels)*100:5.1f}% | "
                f"SubtleSent acc={np.mean((probs_sent[i]>=0.5)==sent_labels)*100:5.1f}% | "
                f"Yes+No mass={np.mean(m_all)*100:5.1f}%"
            )

    # 3. Entropy contrast on 24 subtle sentences across 4 prompt regimes
    entropy_contrast = measure_entropy_contrast(model, tok, sent_texts[:24], yes_id, no_id)

    # 4. Exact frozen-weight fingerprint AFTER all evaluations
    weights_after = fingerprint_frozen_weights(model)
    weights_sha_match = weights_before["sha256"] == weights_after["sha256"]
    weights_l2_diff = abs(weights_before["l2_norm"] - weights_after["l2_norm"])
    assert weights_sha_match and weights_l2_diff == 0.0, "Frozen weights were mutated!"
    print(f"[Audit] Weights AFTER : sha256={weights_after['sha256'][:16]}... MATCH={weights_sha_match}, ||dtheta*||=0.0")

    # 5. Analyze post-selection dredging curves across M in {1, 5, 10, 25, 50, 100, 200}
    m_grid = [1, 5, 10, 25, 50, 100, 200]
    r1_stats = analyze_dredging_curve(probs_null, null_labels, m_grid, n_val=n_val, rng=rng, n_resamples=200)
    r2_stats = analyze_dredging_curve(probs_sent, sent_labels, m_grid, n_val=n_val, rng=rng, n_resamples=200)

    # Add best prompt metadata for transparency
    r1_stats["canonical_m200_best_prompt_name"] = prompts[r1_stats["canonical_m200_best_prompt_id"]]["name"]
    r2_stats["canonical_m200_best_prompt_name"] = prompts[r2_stats["canonical_m200_best_prompt_id"]]["name"]

    # Few-shot order sensitivity check (Lu et al., 2022): compare 2shot_pos_neg vs 2shot_neg_pos
    acc_sent_all = np.array(r2_stats["all_prompts_acc_raw"])
    pos_neg_accs = [acc_sent_all[p["id"]] for p in prompts if p["few_shot"] == "2shot_pos_neg"]
    neg_pos_accs = [acc_sent_all[p["id"]] for p in prompts if p["few_shot"] == "2shot_neg_pos"]
    order_abs_diffs = np.abs(np.array(pos_neg_accs) - np.array(neg_pos_accs))

    elapsed_total = time.perf_counter() - t0
    results = {
        "experiment": "exp8_real_llm_prompt_dredging",
        "model_id": MODEL_ID,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm.__version__,
        "seed": GLOBAL_SEED,
        "yes_token_id": int(yes_id),
        "no_token_id": int(no_id),
        "n_prompts": n_prompts,
        "n_val": n_val,
        "n_test": n_test,
        "total_forward_evals": int(n_prompts * len(all_texts) + 4 * 24),
        "mean_yes_no_vocab_mass": float(np.mean(yn_masses)),
        "weights_before": weights_before,
        "weights_after": weights_after,
        "weights_sha256_match": weights_sha_match,
        "weights_l2_diff": weights_l2_diff,
        "regime1_null": r1_stats,
        "regime2_subtle_sentiment": r2_stats,
        "few_shot_order_sensitivity": {
            "mean_2shot_pos_neg_acc": float(np.mean(pos_neg_accs)),
            "mean_2shot_neg_pos_acc": float(np.mean(neg_pos_accs)),
            "mean_abs_order_shift": float(np.mean(order_abs_diffs)),
            "max_abs_order_shift": float(np.max(order_abs_diffs)),
        },
        "entropy_contrast": entropy_contrast,
        "runtime_seconds": float(elapsed_total),
    }

    json_path = RESULTS_DIR / "exp8_llm_prompt_dredging.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fig_path = FIGURES_DIR / "fig8_real_llm_prompt_dredging.png"
    plot_figure_8(results, fig_path)

    print("\n" + "=" * 78)
    print("SUMMARY — EXPERIMENT 8 (REAL Qwen2.5-1.5B-Instruct-4bit)")
    print(f"  Mean Yes+No vocab mass : {results['mean_yes_no_vocab_mass']*100:.2f}%")
    print("  [Regime 1 — Null Signal (Random Balanced Labels)]")
    for idx_m, m in enumerate(m_grid):
        print(
            f"    M={m:3d} (M_eff={r1_stats['m_eff_by_m'][idx_m]:5.1f}) | "
            f"Val={r1_stats['resample_val_acc_mean'][idx_m]*100:5.2f}% +/- {r1_stats['resample_val_acc_std'][idx_m]*100:4.2f}% | "
            f"Test={r1_stats['resample_test_acc_mean'][idx_m]*100:5.2f}% +/- {r1_stats['resample_test_acc_std'][idx_m]*100:4.2f}% | "
            f"Gap={r1_stats['optimism_gap_mean'][idx_m]*100:+5.2f}pp | "
            f"Deflated={r1_stats['deflated_ev_score_mean'][idx_m]*100:5.2f}%"
        )
    print(
        f"    Canonical M=200 best: Val={r1_stats['canonical_m200_val_hits']} ({r1_stats['canonical_m200_val_acc']*100:.2f}%, "
        f"naive p={r1_stats['canonical_m200_naive_p_val']:.3e}, Bonf p={r1_stats['canonical_m200_bonf_p_val']:.3f}) -> "
        f"Test={r1_stats['canonical_m200_test_hits']} ({r1_stats['canonical_m200_test_acc']*100:.2f}%)"
    )

    print("  [Regime 2 — Real Subtle Sentiment Task]")
    s2 = r2_stats["all_prompts_acc_summary"]
    print(
        f"    200-prompt spread (full 144): min={s2['min']*100:.1f}%, med={s2['median']*100:.1f}%, "
        f"mean={s2['mean']*100:.1f}% +/- {s2['std']*100:.1f}%, max={s2['max']*100:.1f}%"
    )
    for idx_m, m in enumerate(m_grid):
        print(
            f"    M={m:3d} (M_eff={r2_stats['m_eff_by_m'][idx_m]:5.1f}) | "
            f"Val={r2_stats['resample_val_acc_mean'][idx_m]*100:5.2f}% +/- {r2_stats['resample_val_acc_std'][idx_m]*100:4.2f}% | "
            f"Test={r2_stats['resample_test_acc_mean'][idx_m]*100:5.2f}% +/- {r2_stats['resample_test_acc_std'][idx_m]*100:4.2f}% | "
            f"Gap={r2_stats['optimism_gap_mean'][idx_m]*100:+5.2f}pp | "
            f"Deflated={r2_stats['deflated_ev_score_mean'][idx_m]*100:5.2f}%"
        )
    print("  [Entropy Contrast at ||Delta theta*|| = 0]")
    for k, v in entropy_contrast.items():
        print(
            f"    {k:20s}: H_vocab={v['mean_vocab_entropy_bits']:.3f} bits | "
            f"H_bin={v['mean_binary_entropy_bits']:.3f} bits | P(Yes)={v['mean_p_yes']:.3f} | "
            f"Yes+No mass={v['mean_yes_no_mass']*100:.1f}%"
        )
    print(f"  Saved JSON   -> {json_path}")
    print(f"  Saved Figure -> {fig_path}")
    print(f"ALL ASSERTIONS PASSED (EXIT 0) in {elapsed_total:.2f} s.")
    print("=" * 78)


if __name__ == "__main__":
    main()
