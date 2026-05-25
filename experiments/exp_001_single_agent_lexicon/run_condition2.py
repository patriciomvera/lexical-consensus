"""
exp_001_single_agent_lexicon/run_condition2.py
----------------------------------------------
Condition 2 (Inverse Grounding) test on the exp_001b agent (10 seeds).

FRAMING HYPOTHESIS:
    If Carroll labels are genuinely grounded in the learned perceptual
    category structure, the agent should not only name unseen images
    correctly (Condition 1), but also retrieve unseen images from labels
    under distractor pressure (Condition 2).

AGENT USED:
    exp_001b — 10 seeds per category. This agent has full Condition 1
    accuracy (1.000) but not yet fully converged centroids (shift at
    seed 10: slithy 0.012, mimsy 0.010, vorpal 0.007; shift declines
    further through seed 15). We test at the point of behavioral
    competence, not maximal representational stability.

DIFFICULTY LEVELS:
    easy    Correct image + 2 distractors from UNTRAINED categories
            (airplane, deer). 20 trials per Carroll label.
    medium  Correct image + 2 distractors from OTHER trained categories
            (e.g. slithy trial: correct frog + mimsy + vorpal distractor).
    hard    Correct image + 2 distractors selected as the images
            geometrically closest to the correct centroid among the
            wrong trained categories (hard negatives).

TASK (all levels):
    Given Carroll label "slithy", which of these 3 images IS a slithy?
    Agent uses: argmax_i similarity(image_i, centroid_slithy)

IMAGE POOLS:
    Seeds (0-9):       used to build the 10-seed lexicon (exp_001b replica)
    C2 pool (10-29):   used as correct-image candidates for Condition 2
    C1 test (30-49):   used in Condition 1 only — not touched here
    Airplane (0-19):   fresh sample (seed 142) for easy distractors
    Deer (0-19):       fresh sample (seed 142) for easy distractors

Usage:
    python -m experiments.exp_001_single_agent_lexicon.run_condition2 --level easy
    python -m experiments.exp_001_single_agent_lexicon.run_condition2 --level medium
    python -m experiments.exp_001_single_agent_lexicon.run_condition2 --level hard
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.perception import PerceptionLayer   # noqa: E402
from src.agents.lexicon import LexiconLayer          # noqa: E402


# ---------------------------------------------------------------------------
# Constants — must stay in sync with run.py for the lexicon rebuild to be
# identical to exp_001b.
# ---------------------------------------------------------------------------

# Trained categories + tutor label assignment (agents see only Carroll labels)
CATEGORIES = {
    "frog":  {"cifar_idx": 6, "carroll_label": "slithy"},
    "horse": {"cifar_idx": 7, "carroll_label": "mimsy"},
    "ship":  {"cifar_idx": 8, "carroll_label": "vorpal"},
}

N_TOTAL_PER_CATEGORY = 50   # must match run.py to preserve RNG state
N_SEEDS              = 10   # exp_001b
C2_START             = N_SEEDS                              # 10
C2_END               = N_TOTAL_PER_CATEGORY - 20            # 30
N_C2_PER_CATEGORY    = C2_END - C2_START                    # 20
N_TRIALS_PER_LABEL   = 20   # equals N_C2_PER_CATEGORY
RANDOM_SEED          = 42   # must match run.py
DISTRACTOR_SEED      = RANDOM_SEED + 100   # separate, for easy distractors

EASY_DISTRACTOR_CATEGORIES = {
    "airplane": 0,
    "deer":     4,
}

CARROLL_LABELS = [CATEGORIES[k]["carroll_label"] for k in CATEGORIES]
CAT_TO_CARROLL = {k: v["carroll_label"] for k, v in CATEGORIES.items()}
CARROLL_TO_CAT = {v["carroll_label"]: k for k, v in CATEGORIES.items()}

SUCCESS_THRESHOLD = 0.70

LEVELS = ("easy", "medium", "hard")
RESULTS_DIR_MAP = {
    "easy":   "exp_001b_condition2_easy",
    "medium": "exp_001b_condition2_medium",
    "hard":   "exp_001b_condition2_hard",
}

# Centroid shift data at seeds 10 and 15 from centroid_drift.csv
# Used to document the behavioral saturation vs representational convergence gap
DRIFT_AT_SEED_10 = {"slithy": 0.012379, "mimsy": 0.009732, "vorpal": 0.006556}
DRIFT_AT_SEED_15 = {"slithy": 0.004116, "mimsy": 0.002800, "vorpal": 0.002843}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_images() -> tuple[dict[str, list], dict[str, list]]:
    """
    Load images for trained categories (seeds + C2 pool) and easy
    distractor categories (airplane, deer).

    For trained categories, we call rng.choice(size=50) to consume exactly
    the same random state as run.py — ensuring seeds 0-9 are identical to
    exp_001b. Only the first 30 images (seeds + C2 pool) are loaded into
    memory; the last 20 (C1 test) are skipped.

    Returns:
        trained — cat_name -> list of 30 PIL Images (indices 0-29)
                  images[0:10]  = seeds
                  images[10:30] = C2 pool
        distractor — cat_name -> list of 20 PIL Images
    """
    import torchvision

    print("[Data] Loading CIFAR-10 from cache...")
    dataset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / ".cache" / "cifar10"),
        train=True,
        download=False,
        transform=None,
    )

    # Main RNG: must replicate run.py state exactly
    rng = np.random.default_rng(RANDOM_SEED)

    trained: dict[str, list] = {}
    for cat_name, cat_info in CATEGORIES.items():
        cidx     = cat_info["cifar_idx"]
        all_idxs = [i for i, (_, lbl) in enumerate(dataset) if lbl == cidx]
        # Choose 50 (same as run.py) to preserve RNG state
        chosen   = rng.choice(all_idxs, size=N_TOTAL_PER_CATEGORY, replace=False)
        # Load only first 30; labels dropped immediately
        trained[cat_name] = [dataset[int(i)][0] for i in chosen[:C2_END]]
        print(f"  [{cat_name:5s}] {N_SEEDS} seeds + {N_C2_PER_CATEGORY} C2 images")

    # Separate RNG for distractor categories
    rng_dist = np.random.default_rng(DISTRACTOR_SEED)
    distractor: dict[str, list] = {}
    for cat_name, cidx in EASY_DISTRACTOR_CATEGORIES.items():
        all_idxs = [i for i, (_, lbl) in enumerate(dataset) if lbl == cidx]
        chosen   = rng_dist.choice(all_idxs, size=20, replace=False)
        distractor[cat_name] = [dataset[int(i)][0] for i in chosen]
        print(f"  [{cat_name:8s}] 20 easy distractor images")

    return trained, distractor


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_all(
    trained: dict[str, list],
    distractor: dict[str, list],
    perception: PerceptionLayer,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Encode all images.

    Returns:
        trained_emb   — cat_name -> ndarray (30, 384)
        distractor_emb — cat_name -> ndarray (20, 384)
    """
    all_batches = list(trained.items()) + list(distractor.items())
    n_total = sum(len(v) for _, v in all_batches)
    print(f"\n[Encoding] {n_total} images with DINOv2-small...")

    trained_emb:    dict[str, np.ndarray] = {}
    distractor_emb: dict[str, np.ndarray] = {}
    done = 0

    for cat_name, imgs in all_batches:
        vecs = []
        for img in imgs:
            vecs.append(perception.encode(img))
            done += 1
            if done % 30 == 0 or done == n_total:
                print(f"  {done}/{n_total}")
        arr = np.array(vecs)
        if cat_name in trained:
            trained_emb[cat_name] = arr
        else:
            distractor_emb[cat_name] = arr

    return trained_emb, distractor_emb


# ---------------------------------------------------------------------------
# Lexicon rebuild — exact replica of exp_001b
# ---------------------------------------------------------------------------

def rebuild_lexicon(trained_emb: dict[str, np.ndarray]) -> LexiconLayer:
    """
    Rebuild the 10-seed lexicon from exp_001b using the same seeds (0-9).
    This is the agent under test for Condition 2.
    """
    lexicon = LexiconLayer(agent_id="agent_00")
    for cat_name, cat_info in CATEGORIES.items():
        label = cat_info["carroll_label"]
        for i in range(N_SEEDS):
            lexicon.assign(trained_emb[cat_name][i], label, round_number=0)
    print("\n[Lexicon] Rebuilt exp_001b agent (10 seeds per category).")
    for lbl in CARROLL_LABELS:
        e = lexicon.get_entry(lbl)
        print(f"  {lbl:8s}: {len(e.embeddings)} embeddings, cohesion={e.confidence:.3f}")
    return lexicon


# ---------------------------------------------------------------------------
# Hard-negative ranking
# ---------------------------------------------------------------------------

def rank_hard_negatives(
    lexicon: LexiconLayer,
    trained_emb: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """
    For each Carroll label, rank the C2 pool images of WRONG categories
    by descending cosine similarity to the target centroid.

    Returns:
        hard_neg[target_label][distractor_cat] -> ndarray of shape (20, 384)
        sorted so that index 0 is the hardest (most similar) negative.
    """
    hard_neg: dict[str, dict[str, np.ndarray]] = {}

    for cat_name, cat_info in CATEGORIES.items():
        label   = cat_info["carroll_label"]
        centroid = lexicon.get_entry(label).centroid
        hard_neg[label] = {}

        for other_cat, other_info in CATEGORIES.items():
            if other_cat == cat_name:
                continue
            pool = trained_emb[other_cat][C2_START:C2_END]   # (20, 384)
            sims = pool @ centroid                             # (20,)
            order = np.argsort(-sims)                         # descending
            hard_neg[label][other_cat] = pool[order]

    return hard_neg


# ---------------------------------------------------------------------------
# Trial assembly
# ---------------------------------------------------------------------------

def _make_trial(
    label: str,
    correct_emb: np.ndarray,
    correct_id: str,
    d1_emb: np.ndarray,
    d1_id: str,
    d1_cat: str,
    d2_emb: np.ndarray,
    d2_id: str,
    d2_cat: str,
    extra: dict | None = None,
) -> dict:
    """Score one Condition 2 trial and return result dict."""
    centroid = None  # filled by caller; kept as sentinel

    return {
        "label":           label,
        "correct_id":      correct_id,
        "distractor_ids":  [d1_id, d2_id],
        "distractor_cats": {d1_id: d1_cat, d2_id: d2_cat},
        "_emb": {
            correct_id: correct_emb,
            d1_id:      d1_emb,
            d2_id:      d2_emb,
        },
        **(extra or {}),
    }


def assemble_trials(
    level: str,
    trained_emb: dict[str, np.ndarray],
    distractor_emb: dict[str, np.ndarray],
    hard_neg: dict[str, dict[str, np.ndarray]] | None,
) -> list[dict]:
    """Build all raw trial dicts for the requested level."""
    trials = []

    for cat_name, cat_info in CATEGORIES.items():
        label = cat_info["carroll_label"]
        c2_pool = trained_emb[cat_name][C2_START:C2_END]   # (20, 384) — correct images

        # Distractor category names for this label
        other_cats = [c for c in CATEGORIES if c != cat_name]

        for i in range(N_TRIALS_PER_LABEL):
            correct_emb = c2_pool[i]
            correct_id  = f"{cat_name}_c2_{i}"

            if level == "easy":
                d_cats = list(EASY_DISTRACTOR_CATEGORIES.keys())
                d1_emb, d1_id = distractor_emb[d_cats[0]][i], f"{d_cats[0]}_{i}"
                d2_emb, d2_id = distractor_emb[d_cats[1]][i], f"{d_cats[1]}_{i}"
                d1_cat, d2_cat = d_cats[0], d_cats[1]

            elif level == "medium":
                d1_pool = trained_emb[other_cats[0]][C2_START:C2_END]
                d2_pool = trained_emb[other_cats[1]][C2_START:C2_END]
                d1_emb, d1_id = d1_pool[i], f"{other_cats[0]}_c2_{i}"
                d2_emb, d2_id = d2_pool[i], f"{other_cats[1]}_c2_{i}"
                d1_cat, d2_cat = other_cats[0], other_cats[1]

            elif level == "hard":
                # Hard negatives pre-ranked by similarity to this label's centroid
                hn1 = hard_neg[label][other_cats[0]]   # sorted desc by similarity
                hn2 = hard_neg[label][other_cats[1]]
                d1_emb, d1_id = hn1[i], f"{other_cats[0]}_hn_{i}"
                d2_emb, d2_id = hn2[i], f"{other_cats[1]}_hn_{i}"
                d1_cat, d2_cat = other_cats[0], other_cats[1]

            else:
                raise ValueError(f"Unknown level: {level}")

            trials.append(_make_trial(
                label=label,
                correct_emb=correct_emb, correct_id=correct_id,
                d1_emb=d1_emb, d1_id=d1_id, d1_cat=d1_cat,
                d2_emb=d2_emb, d2_id=d2_id, d2_cat=d2_cat,
            ))

    return trials


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_trials(trials: list[dict], lexicon: LexiconLayer) -> list[dict]:
    """
    Score every trial: compute similarities, pick best candidate, record
    retrieval margin and entropy.
    """
    scored = []

    for t in trials:
        label    = t["label"]
        centroid = lexicon.get_entry(label).centroid

        # Similarities of all 3 candidates to the target centroid
        cands   = list(t["_emb"].items())   # [(id, emb), ...]
        sim_map = {cid: float(np.dot(emb, centroid)) for cid, emb in cands}

        selected_id = max(sim_map, key=sim_map.get)
        correct     = selected_id == t["correct_id"]

        sim_correct = sim_map[t["correct_id"]]
        sim_distractors = [sim_map[did] for did in t["distractor_ids"]]
        sim_best_distractor = max(sim_distractors)

        retrieval_margin = float(sim_correct - sim_best_distractor)

        # Entropy of softmax over 3 similarities
        sims_arr = np.array(list(sim_map.values()))
        sims_arr = sims_arr - sims_arr.max()
        probs    = np.exp(sims_arr) / np.exp(sims_arr).sum()
        entropy  = float(-np.sum(probs * np.log2(probs + 1e-12)))

        result = {
            "label":             label,
            "correct_id":        t["correct_id"],
            "distractor_ids":    t["distractor_ids"],
            "distractor_cats":   t["distractor_cats"],
            "selected_id":       selected_id,
            "correct":           correct,
            "sim_correct":       round(sim_correct, 4),
            "sim_best_distractor": round(sim_best_distractor, 4),
            "retrieval_margin":  round(retrieval_margin, 4),
            "entropy_bits":      round(entropy, 4),
        }

        # If wrong, record which distractor was selected and from what category
        if not correct:
            result["error_selected_cat"] = t["distractor_cats"][selected_id]

        scored.append(result)

    return scored


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def aggregate(scored: list[dict], level: str) -> dict:
    n_total   = len(scored)
    n_correct = sum(1 for r in scored if r["correct"])
    accuracy  = n_correct / n_total

    per_label_counts: dict[str, dict] = {}
    for r in scored:
        lbl = r["label"]
        per_label_counts.setdefault(lbl, {"correct": 0, "total": 0, "failures": []})
        per_label_counts[lbl]["total"] += 1
        if r["correct"]:
            per_label_counts[lbl]["correct"] += 1
        else:
            per_label_counts[lbl]["failures"].append({
                "correct_id":     r["correct_id"],
                "selected_id":    r["selected_id"],
                "selected_cat":   r.get("error_selected_cat", "unknown"),
                "retrieval_margin": r["retrieval_margin"],
                "entropy_bits":   r["entropy_bits"],
            })

    per_label_accuracy = {
        lbl: round(d["correct"] / d["total"], 4)
        for lbl, d in per_label_counts.items()
    }
    failure_cases = {
        lbl: d["failures"]
        for lbl, d in per_label_counts.items()
        if d["failures"]
    }

    margins   = [r["retrieval_margin"] for r in scored]
    entropies = [r["entropy_bits"]     for r in scored]
    mean_margin  = float(np.mean(margins))
    mean_entropy = float(np.mean(entropies))

    result = {
        "level":                   level,
        "n_trials":                n_total,
        "n_correct":               n_correct,
        "overall_accuracy":        round(accuracy, 4),
        "per_label_accuracy":      per_label_accuracy,
        "min_per_label_accuracy":  round(min(per_label_accuracy.values()), 4),
        "mean_retrieval_margin":   round(mean_margin, 4),
        "mean_entropy_bits":       round(mean_entropy, 4),
        "success_threshold":       SUCCESS_THRESHOLD,
        "pass":                    accuracy >= SUCCESS_THRESHOLD,
        "failure_cases":           failure_cases,
    }

    if level == "hard":
        n_hard_neg_errors = n_total - n_correct
        result["hard_negative_error_rate"] = round(n_hard_neg_errors / n_total, 4)
        # Break down errors by which distractor category was selected
        error_by_cat: dict[str, int] = {}
        for r in scored:
            if not r["correct"]:
                cat = r.get("error_selected_cat", "unknown")
                error_by_cat[cat] = error_by_cat.get(cat, 0) + 1
        result["hard_negative_errors_by_distractor_category"] = error_by_cat

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_margin_distribution(scored: list[dict], level: str, out_path: Path) -> None:
    """
    Histogram of retrieval margins per label.
    Positive margin = correct; negative = error.
    Vertical line at 0 marks the decision boundary.
    """
    fig, axes = plt.subplots(1, len(CARROLL_LABELS), figsize=(12, 4), sharey=True)

    COLORS = {"slithy": "#52B788", "mimsy": "#E76F51", "vorpal": "#457B9D"}

    for ax, lbl in zip(axes, CARROLL_LABELS):
        margins = [r["retrieval_margin"] for r in scored if r["label"] == lbl]
        ax.hist(margins, bins=10, color=COLORS[lbl], alpha=0.85, edgecolor="white")
        ax.axvline(0, color="crimson", linewidth=1.5, linestyle="--", label="decision boundary")
        ax.set_title(f"{lbl}\n({CARROLL_TO_CAT[lbl]})", fontsize=11)
        ax.set_xlabel("Retrieval margin", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Condition 2 retrieval margin — {level.upper()}\n"
        f"(positive = correct retrieval, negative = failure)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    config: dict,
    metrics: dict,
    level: str,
    out_path: Path,
) -> None:
    verdict   = "PASS" if metrics["pass"] else "FAIL"
    n_seeds   = config["n_seeds"]
    n_trials  = metrics["n_trials"]

    per_label_rows = "\n".join(
        f"| {CARROLL_TO_CAT.get(lbl, lbl):5s} | {lbl:8s} | {acc:.4f} | "
        f"{'PASS' if acc >= SUCCESS_THRESHOLD else 'FAIL'} |"
        for lbl, acc in metrics["per_label_accuracy"].items()
    )

    # Failure summary
    all_failures = []
    for lbl, cases in metrics["failure_cases"].items():
        for f in cases:
            all_failures.append(
                f"| {lbl:8s} | {f['correct_id']:20s} | {f['selected_id']:20s} | "
                f"{f['selected_cat']:8s} | {f['retrieval_margin']:+.4f} |"
            )
    failures_section = (
        "\n".join(all_failures)
        if all_failures else
        "*(no failures)*"
    )

    hard_neg_section = ""
    if level == "hard":
        hn_rate = metrics.get("hard_negative_error_rate", 0)
        hn_by_cat = metrics.get("hard_negative_errors_by_distractor_category", {})
        cat_rows  = "\n".join(
            f"| {cat:8s} | {count:3d} | {count/n_trials:.3f} |"
            for cat, count in hn_by_cat.items()
        ) or "*(none)*"
        hard_neg_section = textwrap.dedent(f"""
    ## Hard-Negative Error Analysis

    **Hard-negative error rate:** {hn_rate:.4f}
    (= failure rate; all distractors are hard negatives at this level)

    | Distractor category | Errors | Rate |
    |---|---|---|
    {cat_rows}

    A higher rate for one distractor category than another indicates
    which wrong-category centroid is geometrically closest to the
    target — the boundary most likely to fail in exp_002.
    """)

    # Distractor description per level
    distractor_desc = {
        "easy":   "airplane and deer — untrained CIFAR-10 categories with "
                  "no visual overlap with frog/horse/ship",
        "medium": "the other two trained categories (e.g. for slithy: "
                  "mimsy distractor + vorpal distractor)",
        "hard":   "the 2 images from wrong trained categories with HIGHEST "
                  "cosine similarity to the target centroid (hard negatives, "
                  "pre-ranked per centroid)",
    }[level]

    report = textwrap.dedent(f"""\
    # Experiment 001b — Condition 2 ({level.upper()})

    **Date:** {config['date']}
    **Agent:** exp_001b — {n_seeds} seeds per category
    **Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
    **Level:** {level.upper()}
    **Distractors:** {distractor_desc}
    **Trials per Carroll label:** {config['n_trials_per_label']}
    **Total trials:** {n_trials}

    ---

    ## Framing Hypothesis

    > If Carroll labels are genuinely grounded in the learned perceptual
    > category structure, the agent should not only name unseen images
    > correctly (Condition 1), but also retrieve unseen images from labels
    > under distractor pressure (Condition 2).

    Condition 1 established behavioral competence at 10 seeds (accuracy = 1.000).
    Condition 2 tests whether that competence reflects genuine grounding —
    a structured internal representation that supports retrieval — or only
    directional naming from a good-enough centroid.

    ---

    ## Behavioral Saturation vs Representational Convergence

    exp_001b (10 seeds) achieved full Condition 1 accuracy but not fully
    converged centroids. Centroid shift at seed 10 vs seed 15:

    | Carroll label | Shift at seed 10 | Shift at seed 15 | Still moving? |
    |---|---|---|---|
    | slithy | 0.012379 | 0.004116 | yes |
    | mimsy  | 0.009732 | 0.002800 | yes |
    | vorpal | 0.006556 | 0.002843 | yes |

    This test is run on the 10-seed agent intentionally: we are asking
    whether behavioral competence (Condition 1 saturation) implies
    representational grounding (Condition 2 performance), even when the
    centroid has not fully converged. If Condition 2 also passes at 10
    seeds, the grounding is robust to centroid imprecision. If it fails,
    the centroid shift between seed 10 and seed 15 is the likely cause.

    ---

    ## Verdict: {verdict}

    Overall accuracy: **{metrics['overall_accuracy']:.4f}** (threshold: > {SUCCESS_THRESHOLD})

    ---

    ## Per-Label Accuracy

    | Category | Carroll label | Accuracy | Result |
    |---|---|---|---|
    {per_label_rows}

    ---

    ## Retrieval Metrics

    | Metric | Value |
    |---|---|
    | Mean retrieval margin (sim_correct − sim_best_distractor) | {metrics['mean_retrieval_margin']:.4f} |
    | Mean entropy over candidates H(image\\|label) | {metrics['mean_entropy_bits']:.4f} bits |

    **Retrieval margin** is positive when the correct image is closer to the
    target centroid than the best distractor. Negative margin = the agent
    would fail. See margin_distribution.png for the full distribution.

    **Entropy** measures how diffuse the similarity distribution is across
    the 3 candidates. At 3 candidates, maximum entropy = 1.585 bits.
    Lower entropy = more decisive retrieval.
    {hard_neg_section}

    ## Failure Cases

    | Label | Correct image | Selected (wrong) | Wrong cat | Margin |
    |---|---|---|---|---|
    {failures_section}

    ---

    ## Interpretation

    {"The agent successfully retrieves correct images under distractor pressure at the " + level + " difficulty level. Carroll labels are grounded in the centroid structure, not just memorised naming associations." if metrics["pass"] else "The agent fails to reliably retrieve correct images at the " + level + " difficulty level. Naming competence (Condition 1) does not guarantee retrieval competence (Condition 2) at this level. The centroid at 10 seeds may not be stable enough for reliable inverse grounding under this distractor pressure."}
    """)

    out_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(level: str) -> None:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got '{level}'")

    results_dir = PROJECT_ROOT / "results" / RESULTS_DIR_MAP[level]
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Condition 2 — {level.upper()}")
    print(f"Agent: exp_001b ({N_SEEDS} seeds) | {N_TRIALS_PER_LABEL} trials per label")
    print("=" * 60)

    config = {
        "experiment":          f"exp_001b_condition2_{level}",
        "date":                datetime.now().strftime("%Y-%m-%d"),
        "agent":               "exp_001b",
        "n_seeds":             N_SEEDS,
        "encoder":             "facebook/dinov2-small",
        "dataset":             "CIFAR-10",
        "label_assignment":    CAT_TO_CARROLL,
        "level":               level,
        "n_trials_per_label":  N_TRIALS_PER_LABEL,
        "c2_image_pool":       f"indices [{C2_START}:{C2_END}] of the 50 loaded per category",
        "distractor_seed":     DISTRACTOR_SEED,
        "random_seed":         RANDOM_SEED,
        "success_threshold":   SUCCESS_THRESHOLD,
        "categories_ground_truth": {
            cat: info["cifar_idx"] for cat, info in CATEGORIES.items()
        },
    }

    with open(results_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n[Config] Written.")

    # Load + encode
    trained_imgs, distractor_imgs = load_all_images()
    perception = PerceptionLayer(device="cpu")
    trained_emb, distractor_emb = encode_all(trained_imgs, distractor_imgs, perception)

    # Rebuild lexicon
    lexicon = rebuild_lexicon(trained_emb)

    # Precompute hard negatives (needed only for hard, but cheap to compute always)
    hard_neg = rank_hard_negatives(lexicon, trained_emb)
    if level == "hard":
        print("\n[Hard negatives] Pre-ranked by similarity to each centroid.")
        for lbl in CARROLL_LABELS:
            for other_cat, pool in hard_neg[lbl].items():
                centroid = lexicon.get_entry(lbl).centroid
                top_sim  = float(np.dot(pool[0], centroid))
                print(f"  {lbl} <- {other_cat}: hardest distractor sim = {top_sim:.4f}")

    # Assemble and score trials
    print(f"\n[Trials] Assembling {level.upper()} trials...")
    trials = assemble_trials(level, trained_emb, distractor_emb, hard_neg)
    scored = score_trials(trials, lexicon)

    # Metrics
    metrics = aggregate(scored, level)

    print(f"\n  Overall accuracy:        {metrics['overall_accuracy']:.4f}  "
          f"(threshold: >{SUCCESS_THRESHOLD})")
    print(f"  Mean retrieval margin:   {metrics['mean_retrieval_margin']:.4f}")
    print(f"  Mean entropy:            {metrics['mean_entropy_bits']:.4f} bits")
    print(f"\n  Per-label:")
    for lbl, acc in metrics["per_label_accuracy"].items():
        cat    = CARROLL_TO_CAT.get(lbl, lbl)
        status = "PASS" if acc >= SUCCESS_THRESHOLD else "FAIL"
        n_fail = len(metrics["failure_cases"].get(lbl, []))
        print(f"    {lbl:8s} ({cat:8s}): {acc:.4f}  {status}  "
              f"({n_fail} failure{'s' if n_fail != 1 else ''})")

    if level == "hard":
        print(f"\n  Hard-negative error rate: {metrics['hard_negative_error_rate']:.4f}")

    print(f"\n{'='*60}")
    print(f"  VERDICT: {'PASS' if metrics['pass'] else 'FAIL'}")
    print(f"{'='*60}")

    # Artifacts
    print(f"\n[Artifacts]")

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: metrics.json")

    plot_margin_distribution(scored, level, results_dir / "margin_distribution.png")
    write_report(config, metrics, level, results_dir / "report.md")

    print(f"\n[Done] All artifacts in: {results_dir}")
    if not metrics["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Condition 2 (inverse grounding) test on exp_001b agent"
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=LEVELS,
        help="Distractor difficulty: easy, medium, or hard",
    )
    args = parser.parse_args()
    run(args.level)
