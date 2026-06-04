"""
_shared.py — shared constants, data, and helpers for exp_005b.

This module owns:
  - the experiment configuration (agents, splits, hyperparameters)
  - data loading (delegated to exp_003 shared so the seed/split is identical)
  - the consensus-and-centroid mechanics, rewritten on top of LexicalAdapter

Why a separate _shared from exp_003:
  - centroids in exp_005b live in projected space z = normalize(W @ x), so we
    cannot reuse LexiconLayer (which builds centroids over raw DINOv2 input)
  - we also need to maintain *raw* centroids in parallel for the
    raw-vs-projected metrics — and raw centroids are a function of which
    images each agent has accepted, not just the embeddings
  - re-implementing the centroid + classify + consensus pipeline as small
    pure functions keeps the W-training round loop in run.py readable and
    keeps the implementation in one place (no inheritance, no hidden state)

Conditions A / B / C share the entire pipeline. The only thing that varies
between them is the W-update step at the end of each round (apply_feedback_*).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_003_multi_agent_consensus"))

# Reuse exp_003's data loading verbatim — exp_005b must use exactly the same
# images and splits so any metric difference between exp_005 and exp_005b is
# attributable to the adapter, not to a different sample.
from _shared import (  # noqa: E402
    CATEGORIES, CARROLL_LABELS, CAT_TO_CARROLL,
    AGENT_SEED_SLICES, INTERACTION_SLICE, HELD_OUT_SLICE,
    load_all_images, encode_all_images, build_pools,
)


# ─── exp_005b configuration ───────────────────────────────────────────────────

N_AGENTS              = 3
AGENT_IDS             = [f"agent_{i:02d}" for i in range(N_AGENTS)]
MAX_ROUNDS            = 25
CONFIDENCE_THRESHOLD  = 0.30
EMBEDDING_DIM         = 384

# Adapter init: identity + small Gaussian noise. Small enough that agents
# start within ~3-4 cosine-distance units of each other (epsilon=0.01 yields
# ||W - I||_F ~= 3.84 with dim=384).
ADAPTER_EPSILON = 0.01
RANDOM_SEED     = 42

# Training hyperparameters per the exp_005b spec
LEARNING_RATE   = 1e-3
LAMBDA_REJECT   = 0.5
BETA_FROBENIUS  = 0.1
REJECT_MARGIN   = 0.2

CONDITIONS = ["frozen", "consensus", "random"]
CONDITION_LABELS = {
    "frozen":    "A — Frozen (control)",
    "consensus": "B — Consensus-trained",
    "random":    "C — Random-feedback",
}

SUCCESS_CRITERIA = {
    "held_out_accuracy":  0.90,
    "label_coverage":     3,
    "max_label_fraction": 0.50,
}


# ─── Encoded-data convenience ─────────────────────────────────────────────────

@dataclass
class EncodedData:
    """Per-experiment encoded data — computed once, shared across conditions."""
    embeddings:         dict[str, np.ndarray]   # img_hash -> raw DINOv2 (384,)
    ground_truth:       dict[str, str]          # img_hash -> true Carroll label
    interaction_hashes: dict[str, str]          # img_hash -> cat_name
    held_out_hashes:    dict[str, str]
    agent_seed_hashes:  dict[str, dict[str, str]]  # agent_id -> {img_hash: cat_name}

    @property
    def all_hashes(self) -> list[str]:
        out: set[str] = set()
        out.update(self.embeddings.keys())
        return sorted(out)


def load_encoded_data(perception) -> EncodedData:
    """
    Load CIFAR-10, encode with DINOv2, and split into seed / interaction /
    held-out pools — using the same RNG and slices as exp_003 / exp_005.
    """
    print("\n[Data] Loading CIFAR-10...")
    all_images = load_all_images()
    embeddings, ground_truth = encode_all_images(all_images, perception)
    interaction_hashes, held_out_hashes, agent_seed_hashes = build_pools()
    return EncodedData(
        embeddings=embeddings,
        ground_truth=ground_truth,
        interaction_hashes=interaction_hashes,
        held_out_hashes=held_out_hashes,
        agent_seed_hashes=agent_seed_hashes,
    )


# ─── Per-agent state (per condition) ──────────────────────────────────────────

@dataclass
class AgentState:
    """
    All per-agent mutable state for a single condition.

    accepted_by_label maps each Carroll label to the set of image hashes that
    this agent has so far accepted as members of that label's cluster. The
    seed images go in immediately (round 0); interaction images get added
    each round whenever consensus accepts them.

    Centroids are recomputed from accepted_by_label after each round —
    importantly, AFTER the W-update step so that the next round uses
    centroids consistent with the updated projection.
    """
    agent_id:           str
    adapter:            torch.nn.Module                       # LexicalAdapter
    accepted_by_label:  dict[str, set[str]] = field(default_factory=dict)
    projected_centroid: dict[str, np.ndarray] = field(default_factory=dict)
    raw_centroid:       dict[str, np.ndarray] = field(default_factory=dict)


def seed_agent_state(
    state: AgentState,
    seed_hashes: dict[str, str],
) -> None:
    """
    Inject this agent's tutor seeds into accepted_by_label.

    seed_hashes maps img_hash -> cat_name; we convert each to its Carroll
    label and add the image to the corresponding cluster.
    """
    for img_hash, cat_name in seed_hashes.items():
        label = CAT_TO_CARROLL[cat_name]
        state.accepted_by_label.setdefault(label, set()).add(img_hash)


# ─── Projection helpers (NumPy in, NumPy out) ─────────────────────────────────

def project_embedding(adapter, x: np.ndarray) -> np.ndarray:
    """Apply current W to a single raw DINOv2 vector, return unit-normed numpy."""
    return adapter.project_numpy(x)


def project_many(adapter, embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Project every embedding once, return img_hash -> z vector.

    Doing this once per round (instead of per-classify) keeps the round loop
    O(n_images) instead of O(n_images * n_agents * n_labels).
    """
    with torch.no_grad():
        keys = list(embeddings.keys())
        X = np.stack([embeddings[k] for k in keys], axis=0)
        Xt = torch.from_numpy(X.astype(np.float32))
        Zt = adapter(Xt)
        Z = Zt.cpu().numpy()
    return {k: Z[i] for i, k in enumerate(keys)}


# ─── Centroid construction ────────────────────────────────────────────────────

def rebuild_centroids(
    state: AgentState,
    embeddings: dict[str, np.ndarray],
    projected: dict[str, np.ndarray],
) -> None:
    """
    Recompute both projected and raw centroids from accepted_by_label.

    projected_centroid[label] = normalize(mean(z_i for i in accepted))
        — what the agent uses for its own classification next round.

    raw_centroid[label] = normalize(mean(x_i for i in accepted))
        — used by the raw-vs-projected metric so we can show that the same
        underlying accepted-image set behaves differently in the two spaces.

    Both are kept in sync (same accepted set, different feature space).
    Empty clusters yield no entry (handled by the classifier).
    """
    state.projected_centroid = {}
    state.raw_centroid       = {}
    for label, img_hashes in state.accepted_by_label.items():
        if not img_hashes:
            continue
        Z = np.stack([projected[h]  for h in img_hashes], axis=0)
        X = np.stack([embeddings[h] for h in img_hashes], axis=0)
        z_mean = Z.mean(axis=0)
        x_mean = X.mean(axis=0)
        zn = float(np.linalg.norm(z_mean))
        xn = float(np.linalg.norm(x_mean))
        if zn > 1e-9:
            state.projected_centroid[label] = (z_mean / zn).astype(np.float32)
        if xn > 1e-9:
            state.raw_centroid[label] = (x_mean / xn).astype(np.float32)


# ─── Classification in projected space ────────────────────────────────────────

def classify_projected(
    z: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> tuple[str, float, float]:
    """
    Nearest-centroid classification in projected space with the same
    UNCERTAIN logic as LexiconLayer.classify():
      - confidence = 1 - best_distance
      - if margin (second-best - best distance) < 0.1, scale confidence by margin/0.1

    The caller is responsible for applying CONFIDENCE_THRESHOLD to convert
    a low-confidence prediction into UNCERTAIN.
    """
    if not centroids:
        return "UNCERTAIN", 0.0, 0.0

    distances = {
        label: max(0.0, 1.0 - float(np.dot(z, c)))
        for label, c in centroids.items()
    }
    sorted_d = sorted(distances.values())
    best_label = min(distances, key=distances.get)
    best_d     = distances[best_label]
    confidence = max(0.0, 1.0 - best_d)
    margin = (sorted_d[1] - sorted_d[0]) if len(sorted_d) > 1 else 1.0
    if margin < 0.1:
        confidence *= margin / 0.1
    return best_label, confidence, margin


def label_pool(
    states: list[AgentState],
    pool_hashes: dict[str, str],
    projected_per_agent: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict]]:
    """
    Per-agent labeling of a pool of images.

    projected_per_agent[agent_id] is the agent's full projection dict for
    every image hash (cached per round to avoid re-projecting per image).

    Returns: agent_id -> {img_hash -> {label, confidence, margin}}.
    """
    out: dict[str, dict[str, dict]] = {}
    for state in states:
        proj_map = projected_per_agent[state.agent_id]
        cents    = state.projected_centroid
        details: dict[str, dict] = {}
        if not cents:
            for img_hash in pool_hashes:
                details[img_hash] = {"label": "UNCERTAIN", "confidence": 0.0, "margin": 0.0}
        else:
            for img_hash in pool_hashes:
                z = proj_map[img_hash]
                label, conf, margin = classify_projected(z, cents)
                if conf < CONFIDENCE_THRESHOLD:
                    label = "UNCERTAIN"
                details[img_hash] = {
                    "label":      label,
                    "confidence": round(conf, 6),
                    "margin":     round(margin, 6),
                }
        out[state.agent_id] = details
    return out


# ─── Consensus ────────────────────────────────────────────────────────────────

def compute_consensus(
    agent_labels: dict[str, dict[str, str]],
) -> dict[str, dict]:
    """
    Same majority-rule consensus as exp_003. With 3 agents, "majority"
    means at least 2 agree (>= 2/3). UNCERTAIN counts as its own label and
    cannot win consensus unless all three return UNCERTAIN.
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
        majority_label: Optional[str] = None
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


# ─── Round metrics ────────────────────────────────────────────────────────────

def round_metrics(
    consensus_interaction: dict[str, dict],
    consensus_held:        dict[str, dict],
    ground_truth:          dict[str, str],
) -> dict:
    """
    Compute per-round summary metrics — interaction agreement, held-out
    consensus accuracy, mean entropy, label coverage and balance.
    """
    n_int = len(consensus_interaction)
    n_majority  = sum(1 for c in consensus_interaction.values() if c["majority_label"] is not None)
    n_unanimous = sum(1 for c in consensus_interaction.values() if c["unanimous"])

    # Entropy of the 3-agent vote distribution per interaction image
    entropies: list[float] = []
    for c in consensus_interaction.values():
        counts = list(c["label_counts"].values())
        total  = sum(counts)
        if total == 0:
            continue
        probs = np.array(counts, dtype=float) / total
        entropies.append(-float(np.sum(probs * np.log2(probs + 1e-12))))
    mean_entropy = float(np.mean(entropies)) if entropies else 0.0

    # Label coverage: how many distinct Carroll labels won majority
    majority_carroll = {
        c["majority_label"] for c in consensus_interaction.values()
        if c["majority_label"] in CARROLL_LABELS
    }
    label_coverage = len(majority_carroll)
    label_balance = {
        lbl: round(
            sum(1 for c in consensus_interaction.values() if c["majority_label"] == lbl) / max(n_int, 1),
            4,
        )
        for lbl in CARROLL_LABELS
    }

    # Held-out: consensus accuracy + inter-agent agreement
    n_held       = len(consensus_held)
    n_maj_held   = sum(1 for c in consensus_held.values() if c["majority_label"] is not None)
    correct_held = sum(
        1 for h, c in consensus_held.items()
        if c["majority_label"] is not None
        and c["majority_label"] == ground_truth.get(h)
    )

    return {
        "majority_agreement":          round(n_majority  / max(n_int, 1), 4),
        "unanimous_agreement":         round(n_unanimous / max(n_int, 1), 4),
        "mean_entropy":                round(mean_entropy, 4),
        "label_coverage":              label_coverage,
        "label_balance_slithy":        label_balance.get("slithy", 0.0),
        "label_balance_mimsy":         label_balance.get("mimsy",  0.0),
        "label_balance_vorpal":        label_balance.get("vorpal", 0.0),
        "held_out_consensus_accuracy": round(correct_held / max(n_maj_held, 1), 4),
        "held_out_inter_agent_agreement": round(n_maj_held / max(n_held, 1), 4),
    }


# ─── Centroid accumulation ────────────────────────────────────────────────────

def accumulate_accepted(
    states: list[AgentState],
    detailed_interaction: dict[str, dict[str, dict]],
    consensus: dict[str, dict],
) -> None:
    """
    For each agent, add interaction images to accepted_by_label whenever the
    agent's own label matched the majority label for that image. This is the
    same rule as exp_003a's apply_feedback — accepted means "your label won".

    Note: this mutates the agents' accepted_by_label sets but does NOT
    rebuild the centroids. Centroid rebuilding happens after the W-update,
    so it sees the updated W.
    """
    for state in states:
        details = detailed_interaction[state.agent_id]
        for img_hash, det in details.items():
            agent_lbl = det["label"]
            if agent_lbl not in CARROLL_LABELS:
                continue
            cons     = consensus.get(img_hash, {})
            majority = cons.get("majority_label")
            if majority is not None and agent_lbl == majority:
                state.accepted_by_label.setdefault(agent_lbl, set()).add(img_hash)


# ─── CSV helpers ──────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path, fieldnames: Optional[list[str]] = None) -> None:
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return
    fns = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path.name}")
