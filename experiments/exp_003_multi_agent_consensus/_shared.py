"""
_shared.py — shared constants, data loading, and utilities for exp_003.

Imported by run_003a.py (consensus with feedback) and run_003b.py (no-feedback
baseline). Both experiments use identical data, agent setup, and metrics; they
differ only in whether the consensus feedback loop updates agent lexicons.

Data split — same RNG as exp_001 (seed=42), same 50 images per category:
    Agent 00 seeds:   indices [0:5]   per category
    Agent 01 seeds:   indices [5:10]  per category
    Agent 02 seeds:   indices [10:15] per category
    Interaction pool: indices [15:30] per category  (45 images total)
    Held-out pool:    indices [30:50] per category  (60 images total)
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torchvision  # noqa: E402


# ─── Constants ────────────────────────────────────────────────────────────────

CATEGORIES = {
    "frog":  {"cifar_idx": 6, "carroll_label": "slithy"},
    "horse": {"cifar_idx": 7, "carroll_label": "mimsy"},
    "ship":  {"cifar_idx": 8, "carroll_label": "vorpal"},
}

AGENT_SEED_SLICES: dict[str, tuple[int, int]] = {
    "agent_00": (0,  5),
    "agent_01": (5,  10),
    "agent_02": (10, 15),
}

INTERACTION_SLICE    = (15, 30)
HELD_OUT_SLICE       = (30, 50)
N_TOTAL_PER_CATEGORY = 50
N_AGENTS             = 3
RANDOM_SEED          = 42

CARROLL_LABELS = ["slithy", "mimsy", "vorpal"]
CAT_TO_CARROLL = {k: v["carroll_label"] for k, v in CATEGORIES.items()}
CARROLL_TO_CAT = {v["carroll_label"]: k for k, v in CATEGORIES.items()}

CONFIDENCE_THRESHOLD  = 0.30
MAX_ROUNDS            = 100
STOP_UNANIMOUS_ROUNDS = 5   # consecutive 100%-unanimous rounds required to stop

SUCCESS_CRITERIA = {
    "inter_agent_agreement": 0.90,
    "consensus_accuracy":    0.90,
    "per_agent_accuracy":    0.85,
    "label_coverage":        3,
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_all_images() -> dict[str, list]:
    """
    Load 50 images per category from CIFAR-10, reproducing the exact RNG
    sequence from exp_001 (seed=42, same category iteration order).

    Returns cat_name -> list of 50 PIL Images (no labels attached).
    """
    cache_dir = PROJECT_ROOT / ".cache" / "cifar10"
    dataset = torchvision.datasets.CIFAR10(
        root=str(cache_dir), train=True, download=False, transform=None
    )
    rng = np.random.default_rng(RANDOM_SEED)

    all_images: dict[str, list] = {}
    for cat_name, cat_info in CATEGORIES.items():
        cifar_idx = cat_info["cifar_idx"]
        indices   = [i for i, (_, lbl) in enumerate(dataset) if lbl == cifar_idx]
        chosen    = rng.choice(indices, size=N_TOTAL_PER_CATEGORY, replace=False)
        all_images[cat_name] = [dataset[int(i)][0] for i in chosen]
        print(f"  [{cat_name:5s}] {N_TOTAL_PER_CATEGORY} images loaded")

    return all_images


def encode_all_images(
    all_images: dict[str, list],
    perception,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Encode all 150 images upfront with DINOv2. No re-encoding during rounds.

    Returns:
        embeddings   — img_hash -> unit-normed 384-d vector
        ground_truth — img_hash -> true Carroll label
    """
    embeddings:   dict[str, np.ndarray] = {}
    ground_truth: dict[str, str]        = {}

    n_total = N_TOTAL_PER_CATEGORY * len(CATEGORIES)
    done    = 0
    print(f"\n[Encoding] {n_total} images with DINOv2-small...")

    for cat_name, imgs in all_images.items():
        label = CAT_TO_CARROLL[cat_name]
        for idx, img in enumerate(imgs):
            h = f"{cat_name}_{idx:03d}"
            embeddings[h]   = perception.encode(img)
            ground_truth[h] = label
            done += 1
            if done % 30 == 0 or done == n_total:
                print(f"  {done}/{n_total}")

    return embeddings, ground_truth


def build_pools() -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, str]],
]:
    """
    Build interaction pool, held-out pool, and per-agent seed assignments.

    Returns:
        interaction_hashes — img_hash -> cat_name  (45 images total)
        held_out_hashes    — img_hash -> cat_name  (60 images total)
        agent_seed_hashes  — agent_id -> {img_hash -> cat_name}
    """
    interaction_hashes: dict[str, str] = {}
    held_out_hashes:    dict[str, str] = {}
    agent_seed_hashes:  dict[str, dict[str, str]] = {aid: {} for aid in AGENT_SEED_SLICES}

    for cat_name in CATEGORIES:
        i_start, i_end = INTERACTION_SLICE
        h_start, h_end = HELD_OUT_SLICE

        for idx in range(i_start, i_end):
            interaction_hashes[f"{cat_name}_{idx:03d}"] = cat_name
        for idx in range(h_start, h_end):
            held_out_hashes[f"{cat_name}_{idx:03d}"] = cat_name

        for agent_id, (s_start, s_end) in AGENT_SEED_SLICES.items():
            for idx in range(s_start, s_end):
                agent_seed_hashes[agent_id][f"{cat_name}_{idx:03d}"] = cat_name

    return interaction_hashes, held_out_hashes, agent_seed_hashes


# ─── Labeling (read-only) ─────────────────────────────────────────────────────

def label_pool(
    agents,
    pool_hashes: dict[str, str],
    embeddings: dict[str, np.ndarray],
) -> dict[str, dict[str, str]]:
    """
    Each agent classifies every image in pool_hashes.
    Read-only: does NOT modify any agent lexicon.

    Returns agent_id -> {img_hash -> predicted_label}.
    """
    result: dict[str, dict[str, str]] = {}
    for agent in agents:
        labels: dict[str, str] = {}
        for img_hash in pool_hashes:
            emb = embeddings[img_hash]
            if agent.lexicon.is_empty():
                label = "UNCERTAIN"
            else:
                label, conf = agent.lexicon.classify(emb)
                if conf < CONFIDENCE_THRESHOLD:
                    label = "UNCERTAIN"
            labels[img_hash] = label
        result[agent.agent_id] = labels
    return result


# ─── Consensus computation ────────────────────────────────────────────────────

def compute_consensus(
    agent_labels: dict[str, dict[str, str]],
) -> dict[str, dict]:
    """
    Compute per-image consensus from agent label assignments.

    Majority = >= 2/3 of agents agree on the same label.
    With 3 agents, majority requires at least 2 to agree.

    Returns img_hash -> {
        agent_labels:   {agent_id: label},
        label_counts:   {label: count},
        majority_label: str or None,
        unanimous:      bool  (3/3 same label),
        unresolved:     bool  (no majority — all three different),
    }
    """
    all_hashes: set[str] = set()
    for labels in agent_labels.values():
        all_hashes.update(labels.keys())

    result: dict[str, dict] = {}
    for img_hash in all_hashes:
        votes = {
            aid: labels[img_hash]
            for aid, labels in agent_labels.items()
            if img_hash in labels
        }
        n_voters     = len(votes)
        label_counts = Counter(votes.values())

        majority_label = None
        for lbl, count in label_counts.most_common(1):
            if n_voters > 0 and count / n_voters >= 2 / 3:
                majority_label = lbl

        n_distinct = len(label_counts)
        result[img_hash] = {
            "agent_labels":   votes,
            "label_counts":   dict(label_counts),
            "majority_label": majority_label,
            "unanimous":      (majority_label is not None and n_distinct == 1),
            "unresolved":     (majority_label is None),
        }

    return result


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_interaction_metrics(
    consensus: dict[str, dict],
    ground_truth: dict[str, str],
    stable_counts: dict[str, tuple],
) -> tuple[dict, dict[str, tuple]]:
    """
    Compute interaction-pool metrics for one round.
    Updates and returns stable_counts (img_hash -> (majority_label, consecutive_rounds)).

    stable_label_count = images where the same majority label has held for >= 3 rounds.
    """
    n_images = len(consensus)
    if n_images == 0:
        return {}, stable_counts

    n_majority  = sum(1 for c in consensus.values() if c["majority_label"] is not None)
    n_unanimous = sum(1 for c in consensus.values() if c["unanimous"])

    majority_correct = sum(
        1 for img_hash, c in consensus.items()
        if c["majority_label"] is not None
        and c["majority_label"] == ground_truth.get(img_hash)
    )
    consensus_accuracy = majority_correct / n_majority if n_majority > 0 else 0.0

    # Shannon entropy of per-image label distribution across agents
    entropies = []
    for c in consensus.values():
        counts = list(c["label_counts"].values())
        total  = sum(counts)
        if total == 0:
            continue
        probs = np.array(counts, dtype=float) / total
        h = -float(np.sum(probs * np.log2(probs + 1e-12)))
        entropies.append(h)
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    # Label coverage and balance
    majority_labels_used = {
        c["majority_label"] for c in consensus.values()
        if c["majority_label"] is not None
    }
    label_coverage = len(majority_labels_used)

    label_balance = {
        lbl: sum(1 for c in consensus.values() if c["majority_label"] == lbl) / n_images
        for lbl in CARROLL_LABELS
    }

    # Stable label count: same majority for >= 3 consecutive rounds
    new_stable: dict[str, tuple] = {}
    for img_hash, c in consensus.items():
        ml = c["majority_label"]
        prev_label, prev_count = stable_counts.get(img_hash, (None, 0))
        if ml is not None and ml == prev_label:
            new_stable[img_hash] = (ml, prev_count + 1)
        elif ml is not None:
            new_stable[img_hash] = (ml, 1)
        else:
            new_stable[img_hash] = (None, 0)

    stable_label_count = sum(1 for _, cnt in new_stable.values() if cnt >= 3)

    metrics = {
        "majority_agreement":   round(n_majority  / n_images, 4),
        "unanimous_agreement":  round(n_unanimous / n_images, 4),
        "consensus_accuracy":   round(consensus_accuracy, 4),
        "mean_entropy":         round(mean_entropy, 4),
        "stable_label_count":   stable_label_count,
        "label_coverage":       label_coverage,
        "label_balance_slithy": round(label_balance["slithy"], 4),
        "label_balance_mimsy":  round(label_balance["mimsy"],  4),
        "label_balance_vorpal": round(label_balance["vorpal"], 4),
    }
    return metrics, new_stable


def evaluate_held_out(
    agents,
    held_out_hashes: dict[str, str],
    embeddings: dict[str, np.ndarray],
    ground_truth: dict[str, str],
) -> dict:
    """
    Classify held-out images with each agent (read-only, no lexicon update).

    Returns:
        per_agent_accuracy:          {agent_id: float}
        held_out_consensus_accuracy: float  (majority label vs ground truth)
        inter_agent_agreement:       float  (fraction of images with majority)
    """
    agent_labels: dict[str, dict[str, str]] = {}
    for agent in agents:
        labels: dict[str, str] = {}
        for img_hash in held_out_hashes:
            emb = embeddings[img_hash]
            if agent.lexicon.is_empty():
                label = "UNCERTAIN"
            else:
                label, conf = agent.lexicon.classify(emb)
                if conf < CONFIDENCE_THRESHOLD:
                    label = "UNCERTAIN"
            labels[img_hash] = label
        agent_labels[agent.agent_id] = labels

    per_agent_accuracy: dict[str, float] = {}
    for agent_id, labels in agent_labels.items():
        correct = sum(
            1 for img_hash, lbl in labels.items()
            if lbl == ground_truth.get(img_hash)
        )
        per_agent_accuracy[agent_id] = round(correct / len(held_out_hashes), 4)

    ho_consensus   = compute_consensus(agent_labels)
    n_held_out     = len(held_out_hashes)
    n_majority_ho  = sum(1 for c in ho_consensus.values() if c["majority_label"] is not None)

    majority_correct_ho = sum(
        1 for img_hash, c in ho_consensus.items()
        if c["majority_label"] is not None
        and c["majority_label"] == ground_truth.get(img_hash)
    )
    held_out_consensus_accuracy = (
        majority_correct_ho / n_majority_ho if n_majority_ho > 0 else 0.0
    )
    inter_agent_agreement = n_majority_ho / n_held_out if n_held_out > 0 else 0.0

    return {
        "per_agent_accuracy":          per_agent_accuracy,
        "held_out_consensus_accuracy": round(held_out_consensus_accuracy, 4),
        "inter_agent_agreement":       round(inter_agent_agreement, 4),
    }


# ─── Centroid drift ───────────────────────────────────────────────────────────

def snapshot_centroids(agents) -> dict[str, dict[str, np.ndarray | None]]:
    """Capture current centroids: agent_id -> {label -> centroid_or_None}."""
    snap: dict[str, dict[str, np.ndarray | None]] = {}
    for agent in agents:
        snap[agent.agent_id] = {}
        for lbl in CARROLL_LABELS:
            entry = agent.lexicon.get_entry(lbl)
            if entry is not None and entry.centroid is not None:
                snap[agent.agent_id][lbl] = entry.centroid.copy()
            else:
                snap[agent.agent_id][lbl] = None
    return snap


def compute_centroid_drift(
    prev_snap: dict[str, dict[str, np.ndarray | None]],
    curr_snap: dict[str, dict[str, np.ndarray | None]],
    round_num: int,
) -> list[dict]:
    """
    Cosine distance between consecutive round centroids, per agent per label.
    0.0 when centroid is absent or unchanged.
    """
    records: list[dict] = []
    for agent_id in prev_snap:
        for lbl in CARROLL_LABELS:
            prev = prev_snap[agent_id].get(lbl)
            curr = curr_snap[agent_id].get(lbl)
            shift = float(1.0 - np.dot(prev, curr)) if (prev is not None and curr is not None) else 0.0
            records.append({
                "round":          round_num,
                "agent_id":       agent_id,
                "label":          lbl,
                "centroid_shift": round(max(0.0, shift), 6),
            })
    return records


# ─── CSV writers ──────────────────────────────────────────────────────────────

def write_metrics_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path.name}")


def write_drift_csv(records: list[dict], out_path: Path) -> None:
    if not records:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "agent_id", "label", "centroid_shift"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  Saved: {out_path.name}")


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_convergence_curve(rows: list[dict], title_suffix: str, out_path: Path) -> None:
    rounds    = [r["round"]               for r in rows]
    majority  = [r["majority_agreement"]  for r in rows]
    unanimous = [r["unanimous_agreement"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, majority,  label="Majority (≥ 2/3)", lw=2)
    ax.plot(rounds, unanimous, label="Unanimous (3/3)",  lw=2, linestyle="--")
    ax.axhline(0.90, color="gray", linestyle=":", lw=1, label="Target 0.90")
    ax.set_xlabel("Round")
    ax.set_ylabel("Fraction of images")
    ax.set_title(f"Convergence — inter-agent agreement [{title_suffix}]")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_entropy_curve(rows: list[dict], title_suffix: str, out_path: Path) -> None:
    rounds  = [r["round"]        for r in rows]
    entropy = [r["mean_entropy"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, entropy, lw=2, color="purple")
    ax.axhline(0.0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean entropy (bits)")
    ax.set_title(f"Label distribution entropy [{title_suffix}]")
    max_h = max(entropy) if entropy else 1.585
    ax.set_ylim(-0.05, max_h * 1.15 + 0.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_per_agent_accuracy(rows: list[dict], title_suffix: str, out_path: Path) -> None:
    rounds    = [r["round"] for r in rows]
    agent_ids = [f"agent_{i:02d}" for i in range(N_AGENTS)]
    colors    = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for agent_id, color in zip(agent_ids, colors):
        key = f"per_agent_accuracy_{agent_id}"
        if rows and key in rows[0]:
            acc = [r[key] for r in rows]
            ax.plot(rounds, acc, label=agent_id, lw=2, color=color)

    if rows and "held_out_consensus_accuracy" in rows[0]:
        cons = [r["held_out_consensus_accuracy"] for r in rows]
        ax.plot(rounds, cons, label="Consensus", lw=2.5, color="black", linestyle="--")

    ax.axhline(0.85, color="gray", linestyle=":", lw=1, label="Target 0.85")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy (held-out)")
    ax.set_title(f"Per-agent and consensus accuracy [{title_suffix}]")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
