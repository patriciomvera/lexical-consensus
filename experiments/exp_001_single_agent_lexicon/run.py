"""
exp_001_single_agent_lexicon/run.py
------------------------------------
Experiment 001: Single Agent Lexicon.

Tests whether a single LearnerAgent can acquire and apply a Carroll
vocabulary through tutor instruction alone — before introducing the
complexity of multi-agent consensus.

Label assignment (tutor-defined, agents never see category names):
    frog  -> slithy
    horse -> mimsy
    ship  -> vorpal

Three variants differing only in seed count:
    001a  n_seeds = 5   -> results/exp_001a_seed_05/
    001b  n_seeds = 10  -> results/exp_001b_seed_10/
    001c  n_seeds = 15  -> results/exp_001c_seed_15/

The test set (20 images per category) is FIXED and identical across
all three variants. Seeds are always drawn from the first n_seeds
images; test images are always the last 20.

Success thresholds (all must pass):
    overall accuracy           >= 0.70
    per-label minimum accuracy >= 0.60
    mean margin (d2 - d1)      >= 0.05

Usage:
    python -m experiments.exp_001_single_agent_lexicon.run --n-seeds 5
    python -m experiments.exp_001_single_agent_lexicon.run --n-seeds 10
    python -m experiments.exp_001_single_agent_lexicon.run --n-seeds 15
"""

from __future__ import annotations

import argparse
import csv
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
# Fixed constants — changing any of these invalidates the comparison across
# variants, so they are intentionally not CLI arguments.
# ---------------------------------------------------------------------------

# Tutor label assignment: category -> Carroll label
# Agents see only the Carroll labels; category names never leave this dict.
CATEGORIES = {
    "frog":  {"cifar_idx": 6, "carroll_label": "slithy"},
    "horse": {"cifar_idx": 7, "carroll_label": "mimsy"},
    "ship":  {"cifar_idx": 8, "carroll_label": "vorpal"},
}

N_TOTAL_PER_CATEGORY = 50  # must match exp_000 sampling
N_TEST_PER_CATEGORY  = 20  # last 20 images — fixed across all variants
# seeds = images[0 : n_seeds], test = images[30:50] — no overlap for n_seeds <= 15
TEST_START = N_TOTAL_PER_CATEGORY - N_TEST_PER_CATEGORY   # 30

RANDOM_SEED = 42

THRESHOLDS = {
    "accuracy":              0.70,
    "per_label_min_accuracy": 0.60,
    "mean_margin":            0.05,
}

# Variants: n_seeds -> (letter, results-dir-suffix)
VARIANTS = {
    5:  ("a", "exp_001a_seed_05"),
    10: ("b", "exp_001b_seed_10"),
    15: ("c", "exp_001c_seed_15"),
}

# Derived lookups
CARROLL_LABELS   = [CATEGORIES[k]["carroll_label"] for k in CATEGORIES]
CAT_TO_CARROLL   = {k: v["carroll_label"] for k, v in CATEGORIES.items()}
CARROLL_TO_CAT   = {v["carroll_label"]: k for k, v in CATEGORIES.items()}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cifar10_splits(
    n_seeds: int,
) -> tuple[dict[str, list], dict[str, list]]:
    """
    Load CIFAR-10 for frog / horse / ship using the same deterministic
    sampling as exp_000 (seed 42, same category iteration order).

    Returns:
        seed_images — cat_name -> list of n_seeds PIL Images (unlabeled)
        test_images — cat_name -> list of 20 PIL Images  (fixed, unlabeled)
    """
    import torchvision

    print("[Data] Loading CIFAR-10 from cache...")
    dataset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / ".cache" / "cifar10"),
        train=True,
        download=False,
        transform=None,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    seed_images: dict[str, list] = {}
    test_images: dict[str, list] = {}

    for cat_name, cat_info in CATEGORIES.items():
        cifar_idx = cat_info["cifar_idx"]
        all_indices = [i for i, (_, lbl) in enumerate(dataset) if lbl == cifar_idx]
        chosen = rng.choice(all_indices, size=N_TOTAL_PER_CATEGORY, replace=False)

        # Labels are discarded immediately — only PIL images travel forward
        all_imgs = [dataset[int(i)][0] for i in chosen]

        seed_images[cat_name] = all_imgs[:n_seeds]
        test_images[cat_name] = all_imgs[TEST_START:]

        print(f"  [{cat_name:5s}] {n_seeds:2d} seeds | {N_TEST_PER_CATEGORY} test images")

    return seed_images, test_images


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_all(
    seed_images: dict[str, list],
    test_images: dict[str, list],
    perception: PerceptionLayer,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Encode seeds and test images.

    Returns:
        seed_emb — cat_name -> ndarray (n_seeds, 384)
        test_emb — cat_name -> ndarray (20, 384)
    """
    n_seeds_total = sum(len(v) for v in seed_images.values())
    n_test_total  = sum(len(v) for v in test_images.values())
    n_total       = n_seeds_total + n_test_total

    print(f"\n[Encoding] {n_total} images with DINOv2-small "
          f"({n_seeds_total} seeds + {n_test_total} test)...")

    seed_emb: dict[str, np.ndarray] = {}
    test_emb: dict[str, np.ndarray] = {}
    done = 0

    for cat_name in CATEGORIES:
        imgs = seed_images[cat_name] + test_images[cat_name]
        vecs = []
        for img in imgs:
            vecs.append(perception.encode(img))
            done += 1
            if done % 30 == 0 or done == n_total:
                print(f"  {done}/{n_total}")
        vecs = np.array(vecs)
        n_s = len(seed_images[cat_name])
        seed_emb[cat_name] = vecs[:n_s]
        test_emb[cat_name] = vecs[n_s:]

    return seed_emb, test_emb


# ---------------------------------------------------------------------------
# Lexicon building with centroid drift tracking
# ---------------------------------------------------------------------------

def build_lexicon_track_drift(
    seed_emb: dict[str, np.ndarray],
    n_seeds: int,
) -> tuple[LexiconLayer, list[dict]]:
    """
    Build LexiconLayer by adding seed embeddings one at a time.
    Records the centroid shift (cosine distance) after each addition.

    Adding seeds in order: seed_1, seed_2, ..., seed_n for each category,
    interleaved across categories to simulate realistic tutor instruction.

    Drift record schema (one row per seed addition):
        category, carroll_label, seed_n, n_embeddings, centroid_shift

    centroid_shift is empty string for seed_n=1 (no prior centroid to compare).
    """
    lexicon = LexiconLayer(agent_id="agent_00")
    drift_records: list[dict] = []

    for seed_idx in range(n_seeds):
        for cat_name, cat_info in CATEGORIES.items():
            label = cat_info["carroll_label"]
            emb   = seed_emb[cat_name][seed_idx]

            # Snapshot centroid before adding this seed
            entry_before = lexicon.get_entry(label)
            centroid_before = (
                entry_before.centroid.copy()
                if entry_before is not None and entry_before.centroid is not None
                else None
            )

            lexicon.assign(emb, label, round_number=0)

            centroid_after = lexicon.get_entry(label).centroid

            if centroid_before is not None:
                # Both vectors are unit-normalised; cosine dist = 1 - dot
                shift: float | None = float(1.0 - np.dot(centroid_before, centroid_after))
            else:
                shift = None  # first seed: no prior centroid

            drift_records.append({
                "category":      cat_name,
                "carroll_label": label,
                "seed_n":        seed_idx + 1,
                "n_embeddings":  seed_idx + 1,
                "centroid_shift": shift,
            })

    return lexicon, drift_records


# ---------------------------------------------------------------------------
# Inference (read-only — lexicon is NOT updated during testing)
# ---------------------------------------------------------------------------

def classify_test_set(
    lexicon: LexiconLayer,
    test_emb: dict[str, np.ndarray],
) -> list[dict]:
    """
    Classify every test image without modifying the lexicon.

    Per-image output:
        category          — ground-truth category name
        true_label        — ground-truth Carroll label
        predicted_label   — nearest-centroid Carroll label
        correct           — bool
        margin            — cosine_dist(2nd nearest) - cosine_dist(nearest)
        entropy_bits      — H(label|image) in bits, using softmax over similarities
        distances         — dict label->cosine_distance
    """
    results = []

    for cat_name, cat_info in CATEGORIES.items():
        true_label = cat_info["carroll_label"]

        for emb in test_emb[cat_name]:
            # Distances and similarities to all three centroids
            distances: dict[str, float] = {}
            similarities: dict[str, float] = {}

            for lbl in CARROLL_LABELS:
                entry = lexicon.get_entry(lbl)
                if entry and entry.centroid is not None:
                    sim  = float(np.dot(emb, entry.centroid))
                    distances[lbl]    = 1.0 - sim
                    similarities[lbl] = sim

            if not distances:
                continue

            predicted_label = min(distances, key=distances.get)

            sorted_dists = sorted(distances.values())
            margin = float(sorted_dists[1] - sorted_dists[0]) if len(sorted_dists) > 1 else 0.0

            # H(label|image): entropy of softmax over similarities
            sims_arr = np.array([similarities[l] for l in CARROLL_LABELS])
            sims_arr = sims_arr - sims_arr.max()   # numerical stability
            exp_s  = np.exp(sims_arr)
            probs  = exp_s / exp_s.sum()
            entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

            results.append({
                "category":         cat_name,
                "true_label":       true_label,
                "predicted_label":  predicted_label,
                "correct":          predicted_label == true_label,
                "margin":           margin,
                "entropy_bits":     entropy,
                "distances":        distances,
            })

    return results


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(test_results: list[dict]) -> dict:
    n_total   = len(test_results)
    n_correct = sum(1 for r in test_results if r["correct"])
    accuracy  = n_correct / n_total if n_total > 0 else 0.0

    # Per-label accuracy
    per_label_counts: dict[str, dict] = {}
    for r in test_results:
        tl = r["true_label"]
        if tl not in per_label_counts:
            per_label_counts[tl] = {"correct": 0, "total": 0}
        per_label_counts[tl]["total"] += 1
        if r["correct"]:
            per_label_counts[tl]["correct"] += 1

    per_label_accuracy = {
        lbl: round(d["correct"] / d["total"], 4)
        for lbl, d in per_label_counts.items()
    }
    min_per_label = min(per_label_accuracy.values()) if per_label_accuracy else 0.0

    # Confusion matrix (row = true, col = predicted)
    label_idx = {l: i for i, l in enumerate(CARROLL_LABELS)}
    cm = np.zeros((len(CARROLL_LABELS), len(CARROLL_LABELS)), dtype=int)
    for r in test_results:
        ti = label_idx[r["true_label"]]
        pi = label_idx[r["predicted_label"]]
        cm[ti][pi] += 1

    # Scalar metrics
    margins   = [r["margin"]      for r in test_results]
    entropies = [r["entropy_bits"] for r in test_results]
    mean_margin  = float(np.mean(margins))
    mean_entropy = float(np.mean(entropies))

    # Pass/fail per criterion
    pass_acc    = accuracy         >= THRESHOLDS["accuracy"]
    pass_pl     = min_per_label    >= THRESHOLDS["per_label_min_accuracy"]
    pass_margin = mean_margin      >= THRESHOLDS["mean_margin"]

    return {
        "n_test_images":          n_total,
        "overall_accuracy":       round(accuracy, 4),
        "per_label_accuracy":     per_label_accuracy,
        "min_per_label_accuracy": round(min_per_label, 4),
        "mean_margin":            round(mean_margin, 4),
        "mean_entropy_bits":      round(mean_entropy, 4),
        "confusion_matrix": {
            "labels": CARROLL_LABELS,
            "matrix": cm.tolist(),
        },
        "thresholds": THRESHOLDS,
        "pass_accuracy":              pass_acc,
        "pass_per_label_accuracy":    pass_pl,
        "pass_mean_margin":           pass_margin,
        "pass":                       pass_acc and pass_pl and pass_margin,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm_data: dict, out_path: Path) -> None:
    labels = cm_data["labels"]
    cm     = np.array(cm_data["matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title("Confusion matrix — Condition 1 (naming)", fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# CSV artifact
# ---------------------------------------------------------------------------

def write_centroid_drift_csv(drift_records: list[dict], out_path: Path) -> None:
    fields = ["category", "carroll_label", "seed_n", "n_embeddings", "centroid_shift"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in drift_records:
            shift = rec["centroid_shift"]
            writer.writerow({
                "category":       rec["category"],
                "carroll_label":  rec["carroll_label"],
                "seed_n":         rec["seed_n"],
                "n_embeddings":   rec["n_embeddings"],
                "centroid_shift": "" if shift is None else f"{shift:.6f}",
            })
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    config: dict,
    metrics: dict,
    drift_records: list[dict],
    out_path: Path,
) -> None:
    n_seeds = config["n_seeds"]
    variant = config["variant"]
    verdict = "PASS" if metrics["pass"] else "FAIL"

    criteria_rows = "\n".join([
        f"| Overall accuracy | {metrics['overall_accuracy']:.4f} | >= {THRESHOLDS['accuracy']} | {'PASS' if metrics['pass_accuracy'] else 'FAIL'} |",
        f"| Min per-label accuracy | {metrics['min_per_label_accuracy']:.4f} | >= {THRESHOLDS['per_label_min_accuracy']} | {'PASS' if metrics['pass_per_label_accuracy'] else 'FAIL'} |",
        f"| Mean margin | {metrics['mean_margin']:.4f} | >= {THRESHOLDS['mean_margin']} | {'PASS' if metrics['pass_mean_margin'] else 'FAIL'} |",
    ])

    per_label_rows = "\n".join(
        f"| {CARROLL_TO_CAT.get(lbl, lbl):5s} | {lbl:8s} | {acc:.4f} | {'PASS' if acc >= THRESHOLDS['per_label_min_accuracy'] else 'FAIL'} |"
        for lbl, acc in metrics["per_label_accuracy"].items()
    )

    # Centroid drift summary per label: mean, max, final shift
    drift_by_label: dict[str, list[float]] = {}
    for rec in drift_records:
        lbl = rec["carroll_label"]
        if rec["centroid_shift"] is not None:
            drift_by_label.setdefault(lbl, []).append(rec["centroid_shift"])

    drift_rows = "\n".join(
        f"| {lbl:8s} | {np.mean(shifts):.6f} | {np.max(shifts):.6f} | {shifts[-1]:.6f} |"
        for lbl, shifts in drift_by_label.items()
        if shifts
    )

    # Confusion matrix as plain text grid
    cm_labels = metrics["confusion_matrix"]["labels"]
    cm_matrix = metrics["confusion_matrix"]["matrix"]
    cm_header = "  " + "  ".join(f"{l:8s}" for l in cm_labels)
    cm_rows   = "\n".join(
        f"  {cm_labels[i]:8s}  " + "  ".join(f"{v:6d}  " for v in row)
        for i, row in enumerate(cm_matrix)
    )

    status_narrative = (
        "The agent correctly acquires Carroll labels from the seed set and "
        "generalises to held-out images. All three success criteria pass."
        if metrics["pass"] else
        "The agent does not yet reach all required thresholds at this seed count. "
        "See per-criterion breakdown above for which criterion(a) failed."
    )

    report = textwrap.dedent(f"""\
    # Experiment 001{variant} — Single Agent Lexicon (n_seeds = {n_seeds})

    **Date:** {config['date']}
    **Encoder:** {config['encoder']}
    **Dataset:** {config['dataset']}
    **Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
    **Seeds per category:** {n_seeds}
    **Test images per category:** {config['n_test_per_category']} (fixed: same images across 001a/b/c)
    **Random seed:** {config['random_seed']}

    ---

    ## Verdict: {verdict}

    {status_narrative}

    ---

    ## Success Criteria

    | Criterion | Value | Threshold | Result |
    |---|---|---|---|
    {criteria_rows}

    ---

    ## Per-Label Accuracy

    | Category | Carroll label | Accuracy | Result |
    |---|---|---|---|
    {per_label_rows}

    ---

    ## Information-Theoretic Metrics

    **Mean H(label | image):** {metrics['mean_entropy_bits']:.4f} bits
    Entropy of the softmax label distribution per image, averaged over the
    test set. Maximum for 3 balanced classes = 1.585 bits. Lower = the
    classifier is more decisive about each image.

    **Mean margin (d₂ − d₁):** {metrics['mean_margin']:.4f}
    Average gap between the nearest and second-nearest centroid distance.
    Higher = clearer decision boundary at classification time.

    ---

    ## Centroid Drift

    Cosine distance of the centroid before and after each seed addition.
    Full trajectory in centroid_drift.csv.

    | Carroll label | Mean shift | Max shift | Shift at final seed |
    |---|---|---|---|
    {drift_rows}

    The drift curve measures how quickly the centroid stabilises as seeds
    accumulate. A large shift at seed 2 followed by small shifts at seeds
    3-N indicates the centroid converged early. Persistent large shifts
    indicate the seed images themselves are highly variable.

    ---

    ## Confusion Matrix

    Rows = true Carroll label. Columns = predicted Carroll label.
    See confusion_matrix.png for the visual.

    {cm_header}
    {cm_rows}

    Total test images: {metrics['n_test_images']}
    Correctly classified: {int(round(metrics['overall_accuracy'] * metrics['n_test_images']))}

    ---

    ## Interpretation

    With {n_seeds} seed image{'s' if n_seeds != 1 else ''} per category, the centroid-based lexicon
    {"achieves all required thresholds. The geometric substrate established in exp_000 is sufficient for single-agent Carroll label acquisition at this seed count." if metrics['pass'] else "does not yet achieve all required thresholds. Increasing n_seeds is the first corrective action — the perceptual substrate is viable (exp_000 PASS), so the failure is in centroid estimation quality, not perception."}
    """)

    out_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(n_seeds: int) -> None:
    if n_seeds not in VARIANTS:
        raise ValueError(f"n_seeds must be one of {list(VARIANTS.keys())}, got {n_seeds}")

    variant_letter, results_dir_name = VARIANTS[n_seeds]
    results_dir = PROJECT_ROOT / "results" / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Experiment 001{variant_letter} — Single Agent Lexicon")
    print(f"n_seeds = {n_seeds} per category | test = {N_TEST_PER_CATEGORY} per category")
    print("=" * 60)

    config = {
        "experiment":             f"exp_001{variant_letter}_seed_{n_seeds:02d}",
        "variant":                variant_letter,
        "date":                   datetime.now().strftime("%Y-%m-%d"),
        "encoder":                "facebook/dinov2-small",
        "dataset":                "CIFAR-10",
        "label_assignment":       CAT_TO_CARROLL,
        "n_seeds":                n_seeds,
        "n_test_per_category":    N_TEST_PER_CATEGORY,
        "n_total_per_category":   N_TOTAL_PER_CATEGORY,
        "random_seed":            RANDOM_SEED,
        "test_slice":             f"[{TEST_START}:{N_TOTAL_PER_CATEGORY}]",
        "thresholds":             THRESHOLDS,
        # Ground truth lives only here. Agents see Carroll labels, not category names.
        "categories_ground_truth": {
            cat: info["cifar_idx"] for cat, info in CATEGORIES.items()
        },
    }

    # Write config immediately — always on disk before any computation
    with open(results_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\n[Config] Written.")

    # 1. Load data
    seed_images, test_images = load_cifar10_splits(n_seeds)

    # 2. Encode
    perception = PerceptionLayer(device="cpu")
    seed_emb, test_emb = encode_all(seed_images, test_images, perception)
    print(f"\n[Encoding] Done. Shapes: "
          f"seeds {list(seed_emb.values())[0].shape[0]}×{len(CATEGORIES)}, "
          f"test {list(test_emb.values())[0].shape[0]}×{len(CATEGORIES)}")

    # 3. Build lexicon (tracking centroid drift)
    print("\n[Lexicon] Building from seed set...")
    lexicon, drift_records = build_lexicon_track_drift(seed_emb, n_seeds)
    for lbl in CARROLL_LABELS:
        entry = lexicon.get_entry(lbl)
        print(f"  {lbl:8s}: {len(entry.embeddings)} embeddings, "
              f"cohesion={entry.confidence:.3f}")

    # 4. Test (lexicon read-only)
    print("\n[Testing] Classifying held-out test set...")
    test_results = classify_test_set(lexicon, test_emb)

    # 5. Metrics
    metrics = aggregate_metrics(test_results)

    print(f"\n  Overall accuracy:    {metrics['overall_accuracy']:.4f}  "
          f"(>= {THRESHOLDS['accuracy']})")
    print(f"  Min per-label acc.:  {metrics['min_per_label_accuracy']:.4f}  "
          f"(>= {THRESHOLDS['per_label_min_accuracy']})")
    print(f"  Mean margin:         {metrics['mean_margin']:.4f}  "
          f"(>= {THRESHOLDS['mean_margin']})")
    print(f"  Mean entropy:        {metrics['mean_entropy_bits']:.4f} bits")
    print(f"\n  Per-label:")
    for lbl, acc in metrics["per_label_accuracy"].items():
        cat    = CARROLL_TO_CAT.get(lbl, lbl)
        status = "PASS" if acc >= THRESHOLDS["per_label_min_accuracy"] else "FAIL"
        print(f"    {lbl:8s} ({cat:5s}): {acc:.4f}  {status}")

    print(f"\n{'='*60}")
    print(f"  VERDICT: {'PASS' if metrics['pass'] else 'FAIL'}")
    print(f"{'='*60}")

    # 6. Write artifacts
    print(f"\n[Artifacts]")
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: metrics.json")

    plot_confusion_matrix(metrics["confusion_matrix"], results_dir / "confusion_matrix.png")
    write_centroid_drift_csv(drift_records, results_dir / "centroid_drift.csv")
    write_report(config, metrics, drift_records, results_dir / "report.md")

    print(f"\n[Done] All artifacts in: {results_dir}")
    print(f"       Gate result: exp_001{variant_letter} {'PASS' if metrics['pass'] else 'FAIL'}")

    if not metrics["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 001: Single Agent Lexicon — Carroll label acquisition"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        required=True,
        choices=[5, 10, 15],
        metavar="{5,10,15}",
        help="Number of seed images per category (5 = 001a, 10 = 001b, 15 = 001c)",
    )
    args = parser.parse_args()
    run(args.n_seeds)
