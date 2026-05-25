"""
exp_002_grounding_controls/run.py
----------------------------------
Five falsification conditions that attempt to break the exp_001 result.

Each condition removes or corrupts one specific component of the
perception-label-category relationship. If exp_001 results were
genuine grounding, these conditions should degrade performance
predictably and systematically.

  A  Random label control      — scramble label assignments across seeds
  B  Random embedding control  — replace DINOv2 embeddings with random vectors
  C  Permuted embedding control — frog seeds get ship embs, etc. (seeds only)
  D  OOV rejection             — all candidates are from untrained categories
  E  Harder category set       — cat / dog / deer instead of frog / horse / ship

Acceptance criteria (conditions must reach expected degradation to confirm
that the exp_001 result was not an artifact):
  A  accuracy <= 0.45
  B  accuracy <= 0.45
  C  accuracy <= 0.50
  D  AUROC >= 0.80
  E  document degradation (target accuracy >= 0.70 still passing is interesting)

Usage:
    python -m experiments.exp_002_grounding_controls.run --condition a
    python -m experiments.exp_002_grounding_controls.run --condition b
    python -m experiments.exp_002_grounding_controls.run --condition c
    python -m experiments.exp_002_grounding_controls.run --condition d
    python -m experiments.exp_002_grounding_controls.run --condition e
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.perception import PerceptionLayer  # noqa: E402
from src.agents.lexicon import LexiconLayer         # noqa: E402


# ---------------------------------------------------------------------------
# Shared constants — kept in sync with exp_001
# ---------------------------------------------------------------------------

TRAINED_CATS = {
    "frog":  {"cifar_idx": 6, "carroll_label": "slithy"},
    "horse": {"cifar_idx": 7, "carroll_label": "mimsy"},
    "ship":  {"cifar_idx": 8, "carroll_label": "vorpal"},
}
HARDER_CATS = {
    # Visually similar neighbours in CIFAR-10 space
    "cat":  {"cifar_idx": 3, "carroll_label": "slithy"},
    "dog":  {"cifar_idx": 5, "carroll_label": "mimsy"},
    "deer": {"cifar_idx": 4, "carroll_label": "vorpal"},
}
OOV_CATS = {
    # Untrained categories for Condition D
    "airplane":    0,
    "deer":        4,
    "automobile":  1,
}

N_TOTAL     = 50   # images drawn per category (must match exp_001)
N_SEEDS     = 10   # mirrors exp_001b
C2_START    = 10
C2_END      = 30
TEST_START  = 30

RANDOM_SEED     = 42   # image sampling — must match exp_001
SCRAMBLE_SEED   = 99   # for conditions A and B
OOV_SEED        = 242  # for OOV image sampling in condition D

RESULTS_BASE = PROJECT_ROOT / "results" / "exp_002_grounding_controls"

THRESHOLDS = {
    "a_accuracy":  0.45,
    "b_accuracy":  0.45,
    "c_accuracy":  0.50,
    "d_auroc":     0.80,
    "e_accuracy":  0.70,   # stricter: we want to see degradation but not total collapse
}


# ---------------------------------------------------------------------------
# Shared: data loading
# ---------------------------------------------------------------------------

def load_cifar(n_per_cat: int = N_TOTAL) -> object:
    import torchvision
    return torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / ".cache" / "cifar10"),
        train=True, download=False, transform=None,
    )


def sample_category(dataset, cifar_idx: int, n: int, rng: np.random.Generator):
    """Draw n images from a CIFAR-10 category, RNG advancing forward."""
    all_idx = [i for i, (_, lbl) in enumerate(dataset) if lbl == cifar_idx]
    chosen  = rng.choice(all_idx, size=n, replace=False)
    return [dataset[int(i)][0] for i in chosen]


def load_trained_images(dataset) -> dict[str, list]:
    """Return 50 images per trained category, RNG=42 (mirrors exp_001b)."""
    rng = np.random.default_rng(RANDOM_SEED)
    imgs = {}
    for cat_name, info in TRAINED_CATS.items():
        imgs[cat_name] = sample_category(dataset, info["cifar_idx"], N_TOTAL, rng)
    return imgs


# ---------------------------------------------------------------------------
# Shared: encoding
# ---------------------------------------------------------------------------

def encode_dict(perception: PerceptionLayer,
                images_by_cat: dict[str, list],
                label: str = "") -> dict[str, np.ndarray]:
    """Encode all images in the dict; return cat_name -> ndarray (n, 384)."""
    total = sum(len(v) for v in images_by_cat.values())
    if label:
        print(f"[Encode] {label}: {total} images...")
    embs = {}
    done = 0
    for cat, imgs in images_by_cat.items():
        vecs = []
        for img in imgs:
            vecs.append(perception.encode(img))
            done += 1
            if done % 40 == 0 or done == total:
                print(f"  {done}/{total}")
        embs[cat] = np.array(vecs)
    return embs


# ---------------------------------------------------------------------------
# Shared: lexicon building
# ---------------------------------------------------------------------------

def build_lexicon(seed_assignments: list[tuple[np.ndarray, str]],
                  agent_id: str = "ctrl") -> LexiconLayer:
    """
    Build a LexiconLayer from a list of (embedding, label) pairs.
    Calling code controls which embeddings and labels are paired.
    """
    lexicon = LexiconLayer(agent_id=agent_id)
    for i, (emb, label) in enumerate(seed_assignments):
        lexicon.assign(emb, label, round_number=i)
    return lexicon


def standard_seed_assignments(embs: dict[str, np.ndarray],
                               cat_info: dict) -> list[tuple[np.ndarray, str]]:
    """Normal exp_001b assignments: first N_SEEDS embeddings, correct labels."""
    pairs = []
    for seed_idx in range(N_SEEDS):
        for cat_name, info in cat_info.items():
            label = info["carroll_label"]
            pairs.append((embs[cat_name][seed_idx], label))
    return pairs


# ---------------------------------------------------------------------------
# Shared: Condition 1 evaluation
# ---------------------------------------------------------------------------

def eval_condition1(lexicon: LexiconLayer,
                    test_embs: dict[str, np.ndarray],
                    cat_info: dict) -> dict:
    """
    Classify all test images (indices TEST_START–N_TOTAL).
    Returns per-label accuracy, overall accuracy, mean margin.
    """
    labels_ordered = [info["carroll_label"] for info in cat_info.values()]
    n_per_cat = test_embs[list(test_embs)[0]].shape[0]

    correct_total = 0
    total = 0
    per_label_correct: dict[str, int] = {lbl: 0 for lbl in labels_ordered}
    per_label_total:   dict[str, int] = {lbl: 0 for lbl in labels_ordered}
    margins = []

    for cat_name, info in cat_info.items():
        true_label = info["carroll_label"]
        for emb in test_embs[cat_name]:
            pred_label, _ = lexicon.classify(emb)
            # compute margin = sim(correct) - max(sim(others))
            sims = {}
            for lbl in labels_ordered:
                entry = lexicon.get_entry(lbl)
                if entry and entry.centroid is not None:
                    sims[lbl] = float(np.dot(emb, entry.centroid))
                else:
                    sims[lbl] = -1.0
            if true_label in sims:
                other_sims = [v for k, v in sims.items() if k != true_label]
                margin = sims[true_label] - max(other_sims) if other_sims else 0.0
                margins.append(margin)
            hit = (pred_label == true_label)
            correct_total += int(hit)
            per_label_correct[true_label] += int(hit)
            per_label_total[true_label]   += 1
            total += 1

    per_label_acc = {
        lbl: per_label_correct[lbl] / per_label_total[lbl]
        for lbl in labels_ordered
    }
    return {
        "overall_accuracy": correct_total / total,
        "per_label_accuracy": per_label_acc,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "n_test": total,
    }


# ---------------------------------------------------------------------------
# Shared: Condition 2 medium evaluation
# ---------------------------------------------------------------------------

def eval_condition2_medium(lexicon: LexiconLayer,
                            c2_embs: dict[str, np.ndarray],
                            cat_info: dict) -> dict:
    """
    For each label, 20 trials: correct image + 2 distractors from the other
    trained categories' C2 pools. Distractor selection is deterministic:
    trial i uses c2_pool[i] of each distractor category.
    """
    labels_ordered = [info["carroll_label"] for info in cat_info.values()]
    cat_names      = list(cat_info.keys())
    n_trials       = C2_END - C2_START   # 20

    correct_total = 0
    total = 0
    per_label_correct: dict[str, int] = {lbl: 0 for lbl in labels_ordered}
    margins = []

    for q_idx, (cat_name, info) in enumerate(cat_info.items()):
        query_label = info["carroll_label"]
        distractor_cats = [c for c in cat_names if c != cat_name]

        centroid = lexicon.ground(query_label)
        if centroid is None:
            continue

        for i in range(n_trials):
            correct_emb = c2_embs[cat_name][i]
            dist_embs   = [c2_embs[dc][i % c2_embs[dc].shape[0]] for dc in distractor_cats]
            all_embs    = [correct_emb] + dist_embs

            sims = [float(np.dot(e, centroid)) for e in all_embs]
            best = int(np.argmax(sims))
            hit  = (best == 0)
            margin = sims[0] - max(sims[1:])
            margins.append(margin)
            correct_total += int(hit)
            per_label_correct[query_label] += int(hit)
            total += 1

    per_label_acc = {
        lbl: per_label_correct[lbl] / n_trials
        for lbl in labels_ordered
    }
    return {
        "overall_accuracy": correct_total / total,
        "per_label_accuracy": per_label_acc,
        "mean_margin": float(np.mean(margins)),
        "n_trials": total,
    }


# ---------------------------------------------------------------------------
# Shared: plots
# ---------------------------------------------------------------------------

def plot_confusion(matrix, labels, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_margin_hist(margins_per_label: dict[str, list], path: Path) -> None:
    fig, axes = plt.subplots(1, len(margins_per_label), figsize=(4 * len(margins_per_label), 3))
    if len(margins_per_label) == 1:
        axes = [axes]
    for ax, (lbl, m) in zip(axes, margins_per_label.items()):
        ax.hist(m, bins=10, color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(lbl); ax.set_xlabel("margin")
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CONDITION A — Random label control
# ---------------------------------------------------------------------------

def run_condition_a(out_dir: Path, perception: PerceptionLayer) -> None:
    """
    Scramble Carroll label assignments across all seed images (seed=99).
    Each centroid becomes a mixture of embeddings from multiple categories.
    Expected: accuracy ≈ chance (0.33).
    """
    print("=" * 60)
    print("Condition A — Random Label Control")
    print("=" * 60)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cifar()
    imgs    = load_trained_images(dataset)
    print("[Data] Loaded.")

    embs = encode_dict(perception, imgs, "Condition A")

    # Build scrambled assignments: pool all (emb, label) pairs for seeds,
    # shuffle the labels with rng=99.
    seed_embs  = [(embs[c][i], TRAINED_CATS[c]["carroll_label"])
                  for c in TRAINED_CATS for i in range(N_SEEDS)]
    rng_s = np.random.default_rng(SCRAMBLE_SEED)
    all_labels = [p[1] for p in seed_embs]
    rng_s.shuffle(all_labels)
    scrambled = [(seed_embs[i][0], all_labels[i]) for i in range(len(seed_embs))]

    lexicon = build_lexicon(scrambled, agent_id="ctrl_a")
    print("[Lexicon] Built with scrambled labels.")
    for lbl in [v["carroll_label"] for v in TRAINED_CATS.values()]:
        e = lexicon.get_entry(lbl)
        if e:
            print(f"  {lbl}: {len(e.embeddings)} embeddings, cohesion={e.confidence:.3f}")

    test_embs = {c: embs[c][TEST_START:] for c in TRAINED_CATS}
    c2_embs   = {c: embs[c][C2_START:C2_END] for c in TRAINED_CATS}

    c1 = eval_condition1(lexicon, test_embs, TRAINED_CATS)
    c2 = eval_condition2_medium(lexicon, c2_embs, TRAINED_CATS)

    # Confusion matrix (Condition 1)
    labels_ord = [TRAINED_CATS[c]["carroll_label"] for c in TRAINED_CATS]
    n = len(labels_ord)
    matrix = [[0]*n for _ in range(n)]
    for cat_name, info in TRAINED_CATS.items():
        true_lbl = info["carroll_label"]
        ti = labels_ord.index(true_lbl)
        for emb in test_embs[cat_name]:
            pred, _ = lexicon.classify(emb)
            if pred in labels_ord:
                pi = labels_ord.index(pred)
                matrix[ti][pi] += 1

    plot_confusion(matrix, labels_ord, out_dir / "confusion_c1.png")

    margins_c2 = {lbl: [] for lbl in labels_ord}
    for cat_name, info in TRAINED_CATS.items():
        lbl = info["carroll_label"]
        centroid = lexicon.ground(lbl)
        if centroid is None:
            continue
        dist_cats = [c for c in TRAINED_CATS if c != cat_name]
        for i in range(C2_END - C2_START):
            correct_e = c2_embs[cat_name][i]
            dist_e    = [c2_embs[dc][i] for dc in dist_cats]
            all_e     = [correct_e] + dist_e
            sims      = [float(np.dot(e, centroid)) for e in all_e]
            margins_c2[lbl].append(sims[0] - max(sims[1:]))
    plot_margin_hist(margins_c2, out_dir / "margin_distribution_c2.png")

    pass_c1 = c1["overall_accuracy"] <= THRESHOLDS["a_accuracy"]
    pass_c2 = c2["overall_accuracy"] <= THRESHOLDS["a_accuracy"]
    verdict = "PASS (degraded as expected)" if (pass_c1 and pass_c2) else "FAIL (did not degrade)"

    metrics = {
        "condition": "A_random_labels",
        "scramble_seed": SCRAMBLE_SEED,
        "condition1": c1,
        "condition2_medium": c2,
        "thresholds": THRESHOLDS,
        "pass_c1_degraded": pass_c1,
        "pass_c2_degraded": pass_c2,
        "verdict": verdict,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    config = {
        "condition": "A", "description": "Random label control",
        "n_seeds": N_SEEDS, "scramble_seed": SCRAMBLE_SEED,
        "random_seed": RANDOM_SEED, "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    _write_report_abc(out_dir, "A", "Random Label Control",
                      "Scrambled Carroll label assignments (seed=99) across all 30 seed images. "
                      "Each centroid is built from a mixture of embeddings drawn from multiple "
                      "real categories. The agent perceives correctly but associates arbitrary labels.",
                      "≈ chance (0.33) — labels carry no category-consistent signal.",
                      c1, c2, verdict, labels_ord, matrix)

    print(f"\n[Done] Condition A: {verdict}")
    print(f"  C1 accuracy: {c1['overall_accuracy']:.4f}  C2 accuracy: {c2['overall_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# CONDITION B — Random embedding control
# ---------------------------------------------------------------------------

def run_condition_b(out_dir: Path, perception: PerceptionLayer) -> None:
    """
    Replace all DINOv2 embeddings (seeds and test) with independent random
    unit vectors (seed=99). Labels and category-image pairing preserved.
    Expected: accuracy ≈ chance.
    """
    print("=" * 60)
    print("Condition B — Random Embedding Control")
    print("=" * 60)
    out_dir.mkdir(parents=True, exist_ok=True)

    # We don't need to encode actual images — generate random unit vectors.
    rng_b = np.random.default_rng(SCRAMBLE_SEED)
    DIM   = 384

    def rand_embs(n):
        v = rng_b.standard_normal((n, DIM))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / norms

    seed_embs = {c: rand_embs(N_SEEDS) for c in TRAINED_CATS}
    test_embs = {c: rand_embs(N_TOTAL - TEST_START) for c in TRAINED_CATS}
    c2_embs   = {c: rand_embs(C2_END - C2_START)    for c in TRAINED_CATS}

    pairs = []
    for seed_idx in range(N_SEEDS):
        for cat_name, info in TRAINED_CATS.items():
            pairs.append((seed_embs[cat_name][seed_idx], info["carroll_label"]))

    lexicon = build_lexicon(pairs, agent_id="ctrl_b")
    print("[Lexicon] Built with random embeddings.")

    labels_ord = [TRAINED_CATS[c]["carroll_label"] for c in TRAINED_CATS]
    c1 = eval_condition1(lexicon, test_embs, TRAINED_CATS)
    c2 = eval_condition2_medium(lexicon, c2_embs, TRAINED_CATS)

    matrix = [[0]*3 for _ in range(3)]
    for cat_name, info in TRAINED_CATS.items():
        ti = labels_ord.index(info["carroll_label"])
        for emb in test_embs[cat_name]:
            pred, _ = lexicon.classify(emb)
            if pred in labels_ord:
                matrix[ti][labels_ord.index(pred)] += 1

    plot_confusion(matrix, labels_ord, out_dir / "confusion_c1.png")

    margins_c2 = {lbl: [] for lbl in labels_ord}
    for cat_name, info in TRAINED_CATS.items():
        lbl = info["carroll_label"]
        centroid = lexicon.ground(lbl)
        if centroid is None:
            continue
        dist_cats = [c for c in TRAINED_CATS if c != cat_name]
        for i in range(C2_END - C2_START):
            all_e  = [c2_embs[cat_name][i]] + [c2_embs[dc][i] for dc in dist_cats]
            sims   = [float(np.dot(e, centroid)) for e in all_e]
            margins_c2[lbl].append(sims[0] - max(sims[1:]))
    plot_margin_hist(margins_c2, out_dir / "margin_distribution_c2.png")

    pass_c1 = c1["overall_accuracy"] <= THRESHOLDS["b_accuracy"]
    pass_c2 = c2["overall_accuracy"] <= THRESHOLDS["b_accuracy"]
    verdict = "PASS (degraded as expected)" if (pass_c1 and pass_c2) else "FAIL (did not degrade)"

    metrics = {
        "condition": "B_random_embeddings",
        "random_embedding_seed": SCRAMBLE_SEED,
        "condition1": c1,
        "condition2_medium": c2,
        "thresholds": THRESHOLDS,
        "pass_c1_degraded": pass_c1,
        "pass_c2_degraded": pass_c2,
        "verdict": verdict,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    config = {
        "condition": "B", "description": "Random embedding control",
        "n_seeds": N_SEEDS, "embedding_seed": SCRAMBLE_SEED,
        "random_seed": RANDOM_SEED, "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    _write_report_abc(out_dir, "B", "Random Embedding Control",
                      "All DINOv2 embeddings (seeds and test images) replaced with independent "
                      "random unit vectors (seed=99). Labels and category assignments preserved. "
                      "Centroids are means of random vectors, carrying no visual structure.",
                      "≈ chance (0.33) — no visual structure in embedding space.",
                      c1, c2, verdict, labels_ord, matrix)

    print(f"\n[Done] Condition B: {verdict}")
    print(f"  C1 accuracy: {c1['overall_accuracy']:.4f}  C2 accuracy: {c2['overall_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# CONDITION C — Permuted embedding control
# ---------------------------------------------------------------------------

def run_condition_c(out_dir: Path, perception: PerceptionLayer) -> None:
    """
    Permute seed embeddings across categories (frog seeds get ship embeddings,
    horse seeds get frog embeddings, ship seeds get horse embeddings).
    Test images keep their real DINOv2 embeddings.
    Expected: systematic misclassification — centroids are cross-domain,
    test embeddings don't match their learned label's centroid.
    """
    print("=" * 60)
    print("Condition C — Permuted Embedding Control")
    print("=" * 60)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cifar()
    imgs    = load_trained_images(dataset)
    embs    = encode_dict(perception, imgs, "Condition C")

    # Cyclic permutation: frog → ship embs, horse → frog embs, ship → horse embs
    cat_order = list(TRAINED_CATS.keys())   # ["frog", "horse", "ship"]
    # source_of[cat_name] = which category's real embeddings the seeds will receive
    #   frog  <- ship  (index 2)
    #   horse <- frog  (index 0)
    #   ship  <- horse (index 1)
    permuted_sources = {
        cat_order[0]: cat_order[2],   # frog  <- ship
        cat_order[1]: cat_order[0],   # horse <- frog
        cat_order[2]: cat_order[1],   # ship  <- horse
    }

    pairs = []
    for seed_idx in range(N_SEEDS):
        for cat_name, info in TRAINED_CATS.items():
            label  = info["carroll_label"]
            source = permuted_sources[cat_name]
            # This cat's seed gets embeddings from source category
            emb_permuted = embs[source][seed_idx]
            pairs.append((emb_permuted, label))

    lexicon = build_lexicon(pairs, agent_id="ctrl_c")
    print("[Lexicon] Built with permuted seed embeddings.")
    print("  Permutation: frog<-ship, horse<-frog, ship<-horse")

    # Test set uses REAL embeddings
    test_embs = {c: embs[c][TEST_START:] for c in TRAINED_CATS}
    c2_embs   = {c: embs[c][C2_START:C2_END] for c in TRAINED_CATS}

    labels_ord = [TRAINED_CATS[c]["carroll_label"] for c in TRAINED_CATS]
    c1 = eval_condition1(lexicon, test_embs, TRAINED_CATS)
    c2 = eval_condition2_medium(lexicon, c2_embs, TRAINED_CATS)

    matrix = [[0]*3 for _ in range(3)]
    for cat_name, info in TRAINED_CATS.items():
        ti = labels_ord.index(info["carroll_label"])
        for emb in test_embs[cat_name]:
            pred, _ = lexicon.classify(emb)
            if pred in labels_ord:
                matrix[ti][labels_ord.index(pred)] += 1

    plot_confusion(matrix, labels_ord, out_dir / "confusion_c1.png")

    margins_c2 = {lbl: [] for lbl in labels_ord}
    for cat_name, info in TRAINED_CATS.items():
        lbl      = info["carroll_label"]
        centroid = lexicon.ground(lbl)
        if centroid is None:
            continue
        dist_cats = [c for c in TRAINED_CATS if c != cat_name]
        for i in range(C2_END - C2_START):
            all_e  = [c2_embs[cat_name][i]] + [c2_embs[dc][i] for dc in dist_cats]
            sims   = [float(np.dot(e, centroid)) for e in all_e]
            margins_c2[lbl].append(sims[0] - max(sims[1:]))
    plot_margin_hist(margins_c2, out_dir / "margin_distribution_c2.png")

    pass_c1 = c1["overall_accuracy"] <= THRESHOLDS["c_accuracy"]
    pass_c2 = c2["overall_accuracy"] <= THRESHOLDS["c_accuracy"]
    verdict = "PASS (degraded as expected)" if (pass_c1 and pass_c2) else "FAIL (did not degrade)"

    metrics = {
        "condition": "C_permuted_embeddings",
        "permutation": {"frog": "ship_embs", "horse": "frog_embs", "ship": "horse_embs"},
        "condition1": c1,
        "condition2_medium": c2,
        "thresholds": THRESHOLDS,
        "pass_c1_degraded": pass_c1,
        "pass_c2_degraded": pass_c2,
        "verdict": verdict,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    config = {
        "condition": "C", "description": "Permuted embedding control",
        "permutation": "frog←ship, horse←frog, ship←horse (seeds only)",
        "n_seeds": N_SEEDS, "random_seed": RANDOM_SEED,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    _write_report_abc(out_dir, "C", "Permuted Embedding Control",
                      "Seed embeddings cyclically permuted across categories: frog seeds receive "
                      "ship embeddings, horse seeds receive frog embeddings, ship seeds receive "
                      "horse embeddings. Each Carroll label's centroid is computed from the wrong "
                      "visual category. Test images retain their real DINOv2 embeddings.",
                      "Systematic misclassification — centroids and test images live in different "
                      "regions of the embedding space.",
                      c1, c2, verdict, labels_ord, matrix)

    print(f"\n[Done] Condition C: {verdict}")
    print(f"  C1 accuracy: {c1['overall_accuracy']:.4f}  C2 accuracy: {c2['overall_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# CONDITION D — OOV rejection
# ---------------------------------------------------------------------------

def run_condition_d(out_dir: Path, perception: PerceptionLayer) -> None:
    """
    Build the exp_001b lexicon normally. Present retrieval trials where
    ALL candidates are from untrained categories (airplane, deer, automobile).
    Agent must reject the trial (return NONE) based on a similarity threshold
    calibrated on a validation set, not the test set.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    print("=" * 60)
    print("Condition D — OOV Rejection")
    print("=" * 60)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cifar()

    # Trained images (same sampling as exp_001b)
    trained_imgs = load_trained_images(dataset)
    # OOV images with separate RNG
    rng_oov = np.random.default_rng(OOV_SEED)
    oov_imgs = {cat: sample_category(dataset, idx, 20, rng_oov)
                for cat, idx in OOV_CATS.items()}

    print(f"[Data] {20} images × {len(OOV_CATS)} OOV categories loaded.")

    trained_embs = encode_dict(perception, trained_imgs, "Trained categories")
    oov_embs     = encode_dict(perception, oov_imgs,     "OOV categories")

    # Build exp_001b lexicon
    pairs = standard_seed_assignments(trained_embs, TRAINED_CATS)
    lexicon = build_lexicon(pairs, agent_id="ctrl_d")
    labels_ord = [TRAINED_CATS[c]["carroll_label"] for c in TRAINED_CATS]
    centroids  = {lbl: lexicon.ground(lbl) for lbl in labels_ord}

    def max_sim_any(emb):
        """Max cosine similarity between emb and any learned centroid."""
        return max(float(np.dot(emb, centroids[lbl]))
                   for lbl in labels_ord if centroids[lbl] is not None)

    # --- Threshold calibration ---
    # Positives: seed images [0:N_SEEDS] of each trained category
    # Negatives: first 10 images of each OOV category
    val_pos_scores = [max_sim_any(trained_embs[c][i])
                      for c in TRAINED_CATS for i in range(N_SEEDS)]
    val_neg_scores = [max_sim_any(oov_embs[c][i])
                      for c in OOV_CATS for i in range(10)]

    all_val_scores = val_pos_scores + val_neg_scores
    all_val_labels = [1]*len(val_pos_scores) + [0]*len(val_neg_scores)

    # Youden's J = sensitivity + specificity - 1 = TPR - FPR
    fpr_v, tpr_v, thresh_v = roc_curve(all_val_labels, all_val_scores)
    j_scores = tpr_v - fpr_v
    best_t_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresh_v[best_t_idx])
    print(f"[Threshold] Optimal T={optimal_threshold:.4f}  "
          f"(TPR={tpr_v[best_t_idx]:.3f}, FPR={fpr_v[best_t_idx]:.3f})")

    # --- OOV rejection trials ---
    # Each trial: 3 OOV candidates (1 from each OOV cat), query = some Carroll label
    # Correct answer: NONE
    oov_cats = list(OOV_CATS.keys())
    n_trials_per_label = 10  # 10 trials × 3 labels = 30 total OOV trials
    total_oov_trials   = 0
    correct_rejections = 0
    false_accepts      = 0
    trial_scores       = []   # (max_score, rejected_bool)

    for lbl in labels_ord:
        centroid = centroids[lbl]
        if centroid is None:
            continue
        for i in range(n_trials_per_label):
            cands = [oov_embs[oc][i + 10] for oc in oov_cats]   # use last 10 (val used first 10)
            sims  = [float(np.dot(e, centroid)) for e in cands]
            max_s = max(sims)
            rejected = (max_s < optimal_threshold)
            correct_rejections += int(rejected)
            false_accepts      += int(not rejected)
            total_oov_trials   += 1
            trial_scores.append((max_s, int(rejected)))

    oov_rejection_acc = correct_rejections / total_oov_trials
    false_accept_rate = false_accepts / total_oov_trials

    # --- AUROC: trained C2 pool vs all OOV images ---
    trained_c2_scores = [max_sim_any(trained_embs[c][i])
                         for c in TRAINED_CATS for i in range(C2_START, C2_END)]
    oov_all_scores    = [max_sim_any(oov_embs[c][i])
                         for c in OOV_CATS for i in range(20)]
    auroc_scores  = trained_c2_scores + oov_all_scores
    auroc_labels_ = [1]*len(trained_c2_scores) + [0]*len(oov_all_scores)
    auroc = float(roc_auc_score(auroc_labels_, auroc_scores))
    fpr_t, tpr_t, _ = roc_curve(auroc_labels_, auroc_scores)

    # False reject rate: trained test images that would be wrongly rejected
    trained_test_scores = [max_sim_any(trained_embs[c][i])
                           for c in TRAINED_CATS for i in range(TEST_START, N_TOTAL)]
    false_reject_rate = float(np.mean([s < optimal_threshold for s in trained_test_scores]))

    pass_d = auroc >= THRESHOLDS["d_auroc"]
    verdict = ("PASS (AUROC meets threshold)" if pass_d
               else "FAIL (AUROC below threshold)")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Score distribution
    ax = axes[0]
    ax.hist(trained_c2_scores, bins=15, alpha=0.6, color="steelblue",  label="trained C2")
    ax.hist(oov_all_scores,    bins=15, alpha=0.6, color="tomato",     label="OOV")
    ax.axvline(optimal_threshold, color="black", linestyle="--", label=f"T={optimal_threshold:.3f}")
    ax.set_xlabel("max cosine similarity to any centroid")
    ax.set_title("Score distributions: trained vs OOV")
    ax.legend()

    # ROC curve
    ax = axes[1]
    ax.plot(fpr_t, tpr_t, color="steelblue", lw=2, label=f"AUROC={auroc:.3f}")
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC: trained vs OOV")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "oov_detection.png", dpi=120)
    plt.close(fig)

    metrics = {
        "condition": "D_oov_rejection",
        "optimal_threshold": optimal_threshold,
        "oov_rejection_accuracy": oov_rejection_acc,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "auroc": auroc,
        "n_oov_trials": total_oov_trials,
        "n_trained_test": len(trained_test_scores),
        "thresholds": THRESHOLDS,
        "pass_auroc": pass_d,
        "verdict": verdict,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    config = {
        "condition": "D", "description": "OOV rejection",
        "n_seeds": N_SEEDS, "random_seed": RANDOM_SEED, "oov_seed": OOV_SEED,
        "oov_categories": list(OOV_CATS.keys()),
        "threshold_calibration": "val_pos=seeds[0:10] of trained; val_neg=OOV[0:10]",
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    _write_report_d(out_dir, metrics)

    print(f"\n[Done] Condition D: {verdict}")
    print(f"  AUROC={auroc:.4f}  OOV rejection acc={oov_rejection_acc:.4f}  "
          f"FAR={false_accept_rate:.4f}  FRR={false_reject_rate:.4f}")


# ---------------------------------------------------------------------------
# CONDITION E — Harder category set
# ---------------------------------------------------------------------------

def run_condition_e(out_dir: Path, perception: PerceptionLayer) -> None:
    """
    Replace frog/horse/ship with cat/dog/deer — visually similar categories
    with less distinct boundaries in DINOv2 embedding space.
    Run exp_001b protocol (10 seeds, C1 + C2 medium).
    """
    print("=" * 60)
    print("Condition E — Harder Category Set (cat / dog / deer)")
    print("=" * 60)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_cifar()
    rng_e   = np.random.default_rng(RANDOM_SEED)
    imgs    = {c: sample_category(dataset, info["cifar_idx"], N_TOTAL, rng_e)
               for c, info in HARDER_CATS.items()}
    print("[Data] Loaded.")

    embs = encode_dict(perception, imgs, "Condition E")

    seed_embs = {c: embs[c][:N_SEEDS]            for c in HARDER_CATS}
    test_embs = {c: embs[c][TEST_START:]          for c in HARDER_CATS}
    c2_embs   = {c: embs[c][C2_START:C2_END]      for c in HARDER_CATS}

    pairs = standard_seed_assignments(seed_embs, HARDER_CATS)
    lexicon = build_lexicon(pairs, agent_id="ctrl_e")
    labels_ord = [HARDER_CATS[c]["carroll_label"] for c in HARDER_CATS]

    print("[Lexicon] Built.")
    for lbl in labels_ord:
        e = lexicon.get_entry(lbl)
        if e:
            print(f"  {lbl} ({[c for c,i in HARDER_CATS.items() if i['carroll_label']==lbl][0]}): "
                  f"cohesion={e.confidence:.3f}")

    c1 = eval_condition1(lexicon, test_embs, HARDER_CATS)
    c2 = eval_condition2_medium(lexicon, c2_embs, HARDER_CATS)

    matrix = [[0]*3 for _ in range(3)]
    for cat_name, info in HARDER_CATS.items():
        ti = labels_ord.index(info["carroll_label"])
        for emb in test_embs[cat_name]:
            pred, _ = lexicon.classify(emb)
            if pred in labels_ord:
                matrix[ti][labels_ord.index(pred)] += 1

    plot_confusion(matrix, labels_ord, out_dir / "confusion_c1.png")

    margins_c2 = {lbl: [] for lbl in labels_ord}
    for cat_name, info in HARDER_CATS.items():
        lbl      = info["carroll_label"]
        centroid = lexicon.ground(lbl)
        if centroid is None:
            continue
        dist_cats = [c for c in HARDER_CATS if c != cat_name]
        for i in range(C2_END - C2_START):
            all_e  = [c2_embs[cat_name][i]] + [c2_embs[dc][i] for dc in dist_cats]
            sims   = [float(np.dot(e, centroid)) for e in all_e]
            margins_c2[lbl].append(sims[0] - max(sims[1:]))
    plot_margin_hist(margins_c2, out_dir / "margin_distribution_c2.png")

    pass_c1 = c1["overall_accuracy"] >= THRESHOLDS["e_accuracy"]
    pass_c2 = c2["overall_accuracy"] >= THRESHOLDS["e_accuracy"]
    verdict = "PASS (grounding holds on harder set)" if (pass_c1 and pass_c2) else "PARTIAL / FAIL (degraded below threshold)"

    metrics = {
        "condition": "E_harder_categories",
        "categories": {c: {"cifar_idx": i["cifar_idx"], "carroll_label": i["carroll_label"]}
                       for c, i in HARDER_CATS.items()},
        "condition1": c1,
        "condition2_medium": c2,
        "thresholds": THRESHOLDS,
        "pass_c1": pass_c1,
        "pass_c2": pass_c2,
        "verdict": verdict,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    config = {
        "condition": "E", "description": "Harder category set (cat/dog/deer)",
        "categories": {c: HARDER_CATS[c]["cifar_idx"] for c in HARDER_CATS},
        "n_seeds": N_SEEDS, "random_seed": RANDOM_SEED,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    _write_report_e(out_dir, metrics, labels_ord, matrix)

    print(f"\n[Done] Condition E: {verdict}")
    print(f"  C1 accuracy: {c1['overall_accuracy']:.4f}  C2 accuracy: {c2['overall_accuracy']:.4f}")


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _write_report_abc(out_dir, cond, title, description, expected,
                      c1, c2, verdict, labels, matrix):
    rows = "\n".join(
        f"| {labels[i]} | {' | '.join(str(matrix[i][j]) for j in range(len(labels)))} |"
        for i in range(len(labels))
    )
    header = "| True \\ Pred | " + " | ".join(labels) + " |"
    sep    = "|" + "|".join(["---"]*(len(labels)+1)) + "|"

    c1_table = "\n".join(
        f"| {lbl} | {acc:.4f} |"
        for lbl, acc in c1["per_label_accuracy"].items()
    )
    c2_table = "\n".join(
        f"| {lbl} | {acc:.4f} |"
        for lbl, acc in c2["per_label_accuracy"].items()
    )

    text = f"""# Condition {cond} — {title}

**Experiment:** exp_002_grounding_controls
**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Verdict:** {verdict}

---

## Design

{description}

**Expected result:** {expected}

---

## Condition 1 Results (Naming)

| Metric | Value |
|--------|-------|
| Overall accuracy | {c1['overall_accuracy']:.4f} |
| Mean margin | {c1['mean_margin']:.4f} |
| Threshold (≤ for degradation) | {THRESHOLDS.get(f'{cond.lower()}_accuracy', 'N/A')} |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
{c1_table}

Confusion matrix:

{header}
{sep}
{rows}

---

## Condition 2 Medium Results (Inverse Grounding)

| Metric | Value |
|--------|-------|
| Overall accuracy | {c2['overall_accuracy']:.4f} |
| Mean margin | {c2['mean_margin']:.4f} |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
{c2_table}

---

## Interpretation

{"The accuracy fell within the expected degraded range, confirming that the component under test is necessary for grounding." if "PASS" in verdict else "The accuracy did NOT fall within the expected degraded range. This is an unexpected finding requiring investigation."}
"""
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(text)


def _write_report_d(out_dir, m):
    text = f"""# Condition D — OOV Rejection

**Experiment:** exp_002_grounding_controls
**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Verdict:** {m['verdict']}

---

## Design

Agent has learned slithy / mimsy / vorpal from 10 seeds each.
All retrieval trial candidates are from untrained categories
(airplane, deer, automobile). The correct answer is NONE.

The agent rejects a trial if `max_similarity(candidate, centroid) < T`,
where T is calibrated on a validation set (seeds [0:10] of trained
categories + OOV[0:10]) using Youden's J criterion.

**Expected:** OOV rejection accuracy > 0.80, AUROC ≥ 0.80

---

## Threshold Calibration

| Parameter | Value |
|-----------|-------|
| Optimal threshold (T) | {m['optimal_threshold']:.4f} |
| Calibration set | trained seeds [0:9] (pos) + OOV[0:9] (neg) |

---

## OOV Rejection Metrics

| Metric | Value |
|--------|-------|
| OOV rejection accuracy | {m['oov_rejection_accuracy']:.4f} |
| False accept rate (FAR) | {m['false_accept_rate']:.4f} |
| False reject rate (FRR) | {m['false_reject_rate']:.4f} |
| AUROC | {m['auroc']:.4f} |
| AUROC threshold | {THRESHOLDS['d_auroc']} |

---

## Interpretation

{"AUROC above threshold confirms that the learned centroid geometry provides a usable signal for distinguishing in-vocabulary from out-of-vocabulary images, even without explicit negative training. The agent rejects OOV candidates based solely on the weakness of their similarity to any learned centroid." if "PASS" in m['verdict'] else "AUROC below threshold — the centroid geometry does not reliably separate in-vocabulary from OOV images. This may indicate centroid spread is too large or the OOV categories are visually similar to trained ones."}
"""
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(text)


def _write_report_e(out_dir, m, labels, matrix):
    rows = "\n".join(
        f"| {labels[i]} | {' | '.join(str(matrix[i][j]) for j in range(len(labels)))} |"
        for i in range(len(labels))
    )
    header = "| True \\ Pred | " + " | ".join(labels) + " |"
    sep    = "|" + "|".join(["---"]*(len(labels)+1)) + "|"

    c1   = m["condition1"]
    c2   = m["condition2_medium"]
    cats = ", ".join(f"{c} → {i['carroll_label']}" for c, i in HARDER_CATS.items())

    text = f"""# Condition E — Harder Category Set

**Experiment:** exp_002_grounding_controls
**Date:** {datetime.now().strftime("%Y-%m-%d")}
**Verdict:** {m['verdict']}

---

## Design

Replace frog/horse/ship with cat/dog/deer — visually similar categories
with overlapping boundary regions in DINOv2 embedding space.
Protocol identical to exp_001b: 10 seeds per category, Condition 1 + Condition 2 medium.

Label assignment: {cats}

**Expected:** degraded but measurable performance. This locates where the
system starts to fail as category boundaries tighten.

---

## Condition 1 Results (Naming)

| Metric | Value |
|--------|-------|
| Overall accuracy | {c1['overall_accuracy']:.4f} |
| Mean margin | {c1['mean_margin']:.4f} |
| Pass threshold | ≥ {THRESHOLDS['e_accuracy']} |

Confusion matrix:

{header}
{sep}
{rows}

---

## Condition 2 Medium Results (Inverse Grounding)

| Metric | Value |
|--------|-------|
| Overall accuracy | {c2['overall_accuracy']:.4f} |
| Mean margin | {c2['mean_margin']:.4f} |

---

## Comparison with exp_001b (frog/horse/ship)

| Metric | exp_001b | Condition E | Delta |
|--------|----------|-------------|-------|
| C1 overall accuracy | 1.000 | {c1['overall_accuracy']:.4f} | {c1['overall_accuracy'] - 1.0:+.4f} |
| C2 overall accuracy | 1.000 | {c2['overall_accuracy']:.4f} | {c2['overall_accuracy'] - 1.0:+.4f} |

---

## Interpretation

{"Performance holds on the harder set. Cat/dog/deer may be less separable in DINOv2 space than expected, or 10 seeds is sufficient to establish robust boundaries even for visually similar categories." if "PASS" in m['verdict'] else "Performance degraded below threshold on the harder category set. This documents the boundary condition for the grounding mechanism — it succeeds on frog/horse/ship but fails when visual category overlap increases."}
"""
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CONDITION_MAP = {
    "a": ("condition_a_random_labels",       run_condition_a),
    "b": ("condition_b_random_embeddings",   run_condition_b),
    "c": ("condition_c_permuted_embeddings", run_condition_c),
    "d": ("condition_d_oov_rejection",       run_condition_d),
    "e": ("condition_e_harder_categories",   run_condition_e),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_002_grounding_controls: falsification conditions A–E"
    )
    parser.add_argument("--condition", required=True, choices=list(CONDITION_MAP),
                        help="Which condition to run (a/b/c/d/e)")
    args = parser.parse_args()

    dir_name, run_fn = CONDITION_MAP[args.condition]
    out_dir = RESULTS_BASE / dir_name

    perception = PerceptionLayer()
    run_fn(out_dir, perception)


if __name__ == "__main__":
    main()
