"""
run.py — exp_006_regional_divergence main runner.

Runs all three regimes (vervet / raven / latin) sequentially with the
same encoded data, the same cluster-specific seed assignments, and
identical centroid-accumulation rules. The independent variables across
regimes are network topology (which agents talk to which) and
transmission noise σ.

Architecture decision (recorded in CLAUDE.md after exp_005b):
the LexicalAdapter is NOT used here. Centroids live in raw DINOv2
space. Divergence — if it emerges — must be attributable to input
distribution + topology + noise, not to a per-agent learned
projection.

Output layout (under results/exp_006_regional_divergence/):
  config.yaml
  within_cluster_metrics.csv
  between_cluster_metrics.csv
  centroid_vectors.npz                  — every (regime, agent, label, round)
  cluster_centroid_vectors.npz          — cluster-collective centroids per round
  ledger_events.csv                     — assignment-level log
  (alignment_summary.json + plots produced by metrics_and_plots.py)

Run:
    python -m experiments.exp_006_regional_divergence.run
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.perception import PerceptionLayer  # noqa: E402

from experiments.exp_006_regional_divergence._shared import (  # noqa: E402
    ALL_AGENT_IDS, CLUSTERS, REGIMES, REGIME_CONFIG,
    CARROLL_LABELS, CAT_TO_CARROLL,
    CLUSTER_IMAGE_SLICES, HELD_OUT_SLICE,
    AGENT_SEED_SLICES_PER_CLUSTER, INTERACTION_SLICE_PER_CLUSTER,
    MAX_ROUNDS, CONFIDENCE_THRESHOLD, RANDOM_SEED, SUCCESS_CRITERIA,
    load_all_images, encode_all_images, build_pools, agent_cluster,
    noisy_centroid, classify_with_centroids, compute_consensus,
    cluster_centroid, pairwise_mean_distance, write_csv,
)
from experiments.exp_006_regional_divergence.topology import (  # noqa: E402
    consensus_groups_for_round, is_bridge_round, ConsensusGroup,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_006_regional_divergence"


# ─── Per-agent state ──────────────────────────────────────────────────────────

class AgentState:
    """
    One agent's mutable state for one regime.

    centroids = clean per-label centroids (mean of accepted embeddings,
    L2 normalized). Transmission noise is applied at classification time,
    not stored — the underlying centroid never holds noise so it can be
    cleanly compared across regimes for the divergence metric.
    """
    __slots__ = ("agent_id", "cluster_id", "accepted_by_label", "centroids")

    def __init__(self, agent_id: str):
        self.agent_id   = agent_id
        self.cluster_id = agent_cluster(agent_id)
        self.accepted_by_label: dict[str, set[str]] = {}
        self.centroids:         dict[str, np.ndarray] = {}


def seed_agent_state(state: AgentState, seed_hashes: dict[str, str]) -> None:
    """Inject this agent's tutor seeds (img_hash -> cat_name) into accepted_by_label."""
    for img_hash, cat_name in seed_hashes.items():
        label = CAT_TO_CARROLL[cat_name]
        state.accepted_by_label.setdefault(label, set()).add(img_hash)


def rebuild_centroids(state: AgentState, embeddings: dict[str, np.ndarray]) -> None:
    """centroid_label = normalize(mean(x_i for i in accepted_by_label[label]))."""
    state.centroids = {}
    for label, img_hashes in state.accepted_by_label.items():
        if not img_hashes:
            continue
        X = np.stack([embeddings[h] for h in img_hashes], axis=0)
        m = X.mean(axis=0)
        n = float(np.linalg.norm(m))
        if n > 1e-9:
            state.centroids[label] = (m / n).astype(np.float32)


# ─── Per-round per-agent noisy centroid cache ────────────────────────────────

def build_noisy_centroids(
    states: dict[str, AgentState],
    sigma:  float,
    rng:    np.random.Generator,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Refresh noise per agent per label once at the start of every round.

    Doing this once per round (not per classification) reflects the model
    that an agent has a single noisy recall of its centroid for the
    duration of the round; per-image fresh noise would model something
    different (per-perception noise) and isn't what the spec asks for.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for agent_id, state in states.items():
        out[agent_id] = {
            label: noisy_centroid(c, sigma, rng)
            for label, c in state.centroids.items()
        }
    return out


# ─── Group consensus + accept rule ────────────────────────────────────────────

def run_consensus_group(
    group:              ConsensusGroup,
    states:             dict[str, AgentState],
    noisy_per_agent:    dict[str, dict[str, np.ndarray]],
    embeddings:         dict[str, np.ndarray],
    image_categories:   dict[str, str],
    regime:             str,
    round_num:          int,
) -> list[dict]:
    """
    Classify, vote, and update accepted_by_label for one consensus group.

    Returns a list of ledger rows — one per (agent, image) pair in this
    group — for the diagnostics log. Mutates states in place.

    A 6-agent bridge group uses the same 2/3 majority threshold as a
    3-agent cluster group, so a bridge requires 4 of 6. This keeps the
    "majority" definition consistent across group sizes.
    """
    # Step 1: every agent classifies every image with its NOISY centroid
    detailed: dict[str, dict[str, dict]] = {}
    for agent_id in group.agent_ids:
        noisy_c = noisy_per_agent[agent_id]
        agent_dets: dict[str, dict] = {}
        for img_hash in group.image_hashes:
            label, conf, margin = classify_with_centroids(embeddings[img_hash], noisy_c)
            if conf < CONFIDENCE_THRESHOLD:
                label = "UNCERTAIN"
            agent_dets[img_hash] = {"label": label, "confidence": round(conf, 6), "margin": round(margin, 6)}
        detailed[agent_id] = agent_dets

    # Step 2: consensus within this group only
    agent_labels = {aid: {h: d["label"] for h, d in dets.items()} for aid, dets in detailed.items()}
    consensus    = compute_consensus(agent_labels, majority_threshold=2 / 3)

    # Step 3: ACCEPTED images get added to accepted_by_label.
    # REJECTED / UNRESOLVED do not modify state (exp_003 convention).
    for agent_id in group.agent_ids:
        for img_hash, det in detailed[agent_id].items():
            agent_lbl = det["label"]
            if agent_lbl not in CARROLL_LABELS:
                continue
            majority = consensus[img_hash]["majority_label"]
            if majority is not None and agent_lbl == majority:
                states[agent_id].accepted_by_label.setdefault(agent_lbl, set()).add(img_hash)

    # Step 4: emit ledger rows for this group (includes scope so we can
    # tell intra-cluster from bridge events in post-hoc analysis)
    rows: list[dict] = []
    for agent_id in group.agent_ids:
        for img_hash, det in detailed[agent_id].items():
            majority = consensus[img_hash]["majority_label"]
            status   = ("unresolved" if majority is None
                        else "accepted" if det["label"] == majority
                        else "rejected")
            rows.append({
                "experiment_id":   f"exp_006_{regime}",
                "regime":          regime,
                "round":           round_num,
                "scope":           group.scope,
                "agent_id":        agent_id,
                "cluster_id":      states[agent_id].cluster_id,
                "image_id":        img_hash,
                "true_category":   image_categories.get(img_hash, ""),
                "predicted_label": det["label"],
                "confidence":      det["confidence"],
                "margin":          det["margin"],
                "status":          status,
                "pool":            "interaction",
            })
    return rows


# ─── Held-out evaluation (shared across all 6, clean centroids) ──────────────

def evaluate_held_out(
    states:          dict[str, AgentState],
    held_out_hashes: dict[str, str],
    embeddings:      dict[str, np.ndarray],
    ground_truth:    dict[str, str],
    regime:          str,
    round_num:       int,
) -> tuple[dict, list[dict]]:
    """
    Every agent classifies every held-out image with its CLEAN centroid
    (no transmission noise — held-out is an evaluation, not a training
    interaction). We compute three accuracies:

      - per-cluster consensus accuracy: each cluster does its own 3-agent
        majority and is scored on ground truth
      - shared consensus accuracy: all 6 agents combined under the 2/3
        threshold — measures whether the regime preserves cross-cluster
        agreement on a common test set
      - between_cluster_label_agreement: fraction of held-out images on
        which the two clusters' majority labels match each other
    """
    per_agent_labels: dict[str, dict[str, str]] = {}
    ledger_rows:      list[dict] = []
    for agent_id, state in states.items():
        labels: dict[str, str] = {}
        for img_hash in held_out_hashes:
            label, conf, margin = classify_with_centroids(embeddings[img_hash], state.centroids)
            if conf < CONFIDENCE_THRESHOLD:
                label = "UNCERTAIN"
            labels[img_hash] = label
            ledger_rows.append({
                "experiment_id":   f"exp_006_{regime}",
                "regime":          regime,
                "round":           round_num,
                "scope":           "held_out",
                "agent_id":        agent_id,
                "cluster_id":      state.cluster_id,
                "image_id":        img_hash,
                "true_category":   ground_truth.get(img_hash, ""),
                "predicted_label": label,
                "confidence":      round(conf, 6),
                "margin":          round(margin, 6),
                "status":          "held_out",
                "pool":            "held_out",
            })
        per_agent_labels[agent_id] = labels

    # Per-cluster consensus
    cluster_accuracies:    dict[str, float] = {}
    cluster_majority_maps: dict[str, dict[str, str]] = {}
    for cluster_id, agent_ids in CLUSTERS.items():
        sub_labels = {aid: per_agent_labels[aid] for aid in agent_ids}
        cons       = compute_consensus(sub_labels, majority_threshold=2 / 3)
        n_majority = sum(1 for c in cons.values() if c["majority_label"] is not None)
        correct    = sum(
            1 for h, c in cons.items()
            if c["majority_label"] is not None
            and c["majority_label"] == ground_truth.get(h)
        )
        cluster_accuracies[cluster_id]    = round(correct / max(n_majority, 1), 4)
        cluster_majority_maps[cluster_id] = {h: c["majority_label"] for h, c in cons.items()}

    # Shared all-6 consensus
    shared_cons = compute_consensus(per_agent_labels, majority_threshold=2 / 3)
    n_shared_maj = sum(1 for c in shared_cons.values() if c["majority_label"] is not None)
    n_shared_corr = sum(
        1 for h, c in shared_cons.items()
        if c["majority_label"] is not None
        and c["majority_label"] == ground_truth.get(h)
    )
    shared_acc = round(n_shared_corr / max(n_shared_maj, 1), 4)

    # Between-cluster label agreement: fraction of held-out images where the
    # two clusters' majority labels match (both must have a majority).
    m1 = cluster_majority_maps["cluster_1"]
    m2 = cluster_majority_maps["cluster_2"]
    common = [h for h in held_out_hashes if m1.get(h) is not None and m2.get(h) is not None]
    n_agree = sum(1 for h in common if m1[h] == m2[h])
    btw_agreement = round(n_agree / max(len(common), 1), 4)

    return ({
        "per_cluster_held_out_accuracy":     cluster_accuracies,
        "shared_held_out_consensus_accuracy": shared_acc,
        "between_cluster_label_agreement":   btw_agreement,
    }, ledger_rows)


# ─── Per-round divergence metrics ────────────────────────────────────────────

def compute_round_metrics(
    states:    dict[str, AgentState],
    regime:    str,
    round_num: int,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    """
    Emit:
      - within_rows: per (cluster, label) intra-cluster pairwise mean distance
      - between_rows: per label, between-cluster centroid distance and
                      lexical divergence (between - mean within)
      - cluster_vecs: per (cluster, label) cluster-collective centroid, keyed
                      `{regime}_{cluster_id}_{label}_{round}` for the .npz file
    """
    # Build cluster-collective centroids per label
    cluster_collectives: dict[str, dict[str, np.ndarray]] = {}
    for cluster_id, agent_ids in CLUSTERS.items():
        agent_cs = [states[aid].centroids for aid in agent_ids]
        per_label: dict[str, np.ndarray] = {}
        for label in CARROLL_LABELS:
            v = cluster_centroid(agent_cs, label)
            if v is not None:
                per_label[label] = v
        cluster_collectives[cluster_id] = per_label

    # Intra-cluster pairwise mean distance
    within_rows: list[dict] = []
    within_per_label: dict[str, dict[str, float]] = {}     # cluster -> {label: dist}
    for cluster_id, agent_ids in CLUSTERS.items():
        within_per_label[cluster_id] = {}
        for label in CARROLL_LABELS:
            vecs = [states[aid].centroids[label] for aid in agent_ids if label in states[aid].centroids]
            d    = pairwise_mean_distance(vecs)
            within_per_label[cluster_id][label] = d
            within_rows.append({
                "regime":     regime,
                "round":      round_num,
                "cluster_id": cluster_id,
                "label":      label,
                "n_agents_with_centroid": len(vecs),
                "within_cluster_distance": round(d, 8),
            })

    # Between-cluster distance + divergence
    between_rows: list[dict] = []
    for label in CARROLL_LABELS:
        c1 = cluster_collectives["cluster_1"].get(label)
        c2 = cluster_collectives["cluster_2"].get(label)
        if c1 is None or c2 is None:
            continue
        btw = max(0.0, 1.0 - float(np.dot(c1, c2)))
        mean_within = (
            within_per_label["cluster_1"][label]
            + within_per_label["cluster_2"][label]
        ) / 2.0
        divergence = btw - mean_within
        between_rows.append({
            "regime":                       regime,
            "round":                        round_num,
            "label":                        label,
            "between_cluster_distance":     round(btw, 8),
            "mean_within_cluster_distance": round(mean_within, 8),
            "lexical_divergence":           round(divergence, 8),
            "is_bridge_round":              is_bridge_round(regime, round_num),
        })

    # Cluster vectors npz keys
    cluster_vecs: dict[str, np.ndarray] = {}
    for cluster_id, per_label in cluster_collectives.items():
        for label, vec in per_label.items():
            cluster_vecs[f"{regime}_{cluster_id}_{label}_{round_num}"] = vec.astype(np.float32)

    return within_rows, between_rows, cluster_vecs


# ─── Per-regime round loop ────────────────────────────────────────────────────

def run_regime(
    regime: str,
    pools,                       # WorldPools
    embeddings: dict[str, np.ndarray],
    ground_truth: dict[str, str],
    base_rng: np.random.Generator,
) -> dict:
    """
    Drive a full set of rounds for one regime. Returns a bundle of artifact
    rows for the main writer.

    The noise RNG is seeded deterministically per regime so vervet (sigma=0)
    is trivially reproducible and raven / latin produce the same sequence
    of noise draws on every run.
    """
    cfg = REGIME_CONFIG[regime]
    sigma = cfg["sigma"]
    print(f"\n[{regime}] sigma={sigma}, bridge_every={cfg['bridge_every_n_rounds']} — {cfg['description']}")

    states: dict[str, AgentState] = {aid: AgentState(aid) for aid in ALL_AGENT_IDS}
    for cluster_id, cluster_pool in pools.cluster_pools.items():
        for agent_id, seed_hashes in cluster_pool.agent_seed_hashes.items():
            seed_agent_state(states[agent_id], seed_hashes)
    for state in states.values():
        rebuild_centroids(state, embeddings)

    # Deterministic noise RNG per regime
    noise_rng = np.random.default_rng(hash(regime) & 0xFFFFFFFF)

    # Build a lookup from img_hash to its true category (across all pools)
    image_categories: dict[str, str] = {}
    for cp in pools.cluster_pools.values():
        for h in cp.interaction_hashes:
            image_categories[h] = ground_truth.get(h, "")
    for h in pools.held_out_hashes:
        image_categories[h] = ground_truth.get(h, "")

    round_rows:      list[dict] = []
    within_rows:     list[dict] = []
    between_rows:    list[dict] = []
    ledger_rows:     list[dict] = []
    centroid_vecs:   dict[str, np.ndarray] = {}
    cluster_vecs:    dict[str, np.ndarray] = {}

    for round_num in range(1, MAX_ROUNDS + 1):
        # Fresh noisy centroids for the round
        noisy = build_noisy_centroids(states, sigma, noise_rng)

        # Each consensus group in this round
        for group in consensus_groups_for_round(regime, round_num, pools):
            ledger_rows.extend(
                run_consensus_group(
                    group, states, noisy, embeddings,
                    image_categories, regime, round_num,
                )
            )

        # Rebuild clean centroids after this round's accepts
        for state in states.values():
            rebuild_centroids(state, embeddings)

        # Held-out evaluation with clean centroids
        held_metrics, held_ledger = evaluate_held_out(
            states, pools.held_out_hashes, embeddings, ground_truth, regime, round_num,
        )
        ledger_rows.extend(held_ledger)

        # Per-round divergence metrics
        wr, br, cv = compute_round_metrics(states, regime, round_num)
        within_rows.extend(wr)
        between_rows.extend(br)
        cluster_vecs.update(cv)

        # Per-agent centroid snapshots
        for agent_id, state in states.items():
            for label, vec in state.centroids.items():
                centroid_vecs[f"{regime}_{agent_id}_{label}_{round_num}"] = vec.astype(np.float32)

        # Per-round summary row
        mean_div = (
            sum(r["lexical_divergence"] for r in br) / len(br) if br else 0.0
        )
        mean_btw = (
            sum(r["between_cluster_distance"] for r in br) / len(br) if br else 0.0
        )
        mean_within = (
            sum(r["within_cluster_distance"] for r in wr) / len(wr) if wr else 0.0
        )
        row: dict = {
            "regime":                              regime,
            "round":                               round_num,
            "is_bridge_round":                     is_bridge_round(regime, round_num),
            "mean_within_cluster_distance":        round(mean_within, 6),
            "mean_between_cluster_distance":       round(mean_btw, 6),
            "mean_lexical_divergence":             round(mean_div, 6),
            "shared_held_out_consensus_accuracy":  held_metrics["shared_held_out_consensus_accuracy"],
            "between_cluster_label_agreement":     held_metrics["between_cluster_label_agreement"],
            **{f"per_cluster_acc_{cid}": acc
               for cid, acc in held_metrics["per_cluster_held_out_accuracy"].items()},
        }
        round_rows.append(row)

        print(
            f"  R{round_num:3d} | "
            f"div={mean_div:+.4f} "
            f"btw={mean_btw:.4f} "
            f"win={mean_within:.4f} | "
            f"held_shared={held_metrics['shared_held_out_consensus_accuracy']:.3f} "
            f"btw_agree={held_metrics['between_cluster_label_agreement']:.3f} "
            f"{'[BRIDGE]' if is_bridge_round(regime, round_num) else ''}"
        )

    return {
        "regime":           regime,
        "round_rows":       round_rows,
        "within_rows":      within_rows,
        "between_rows":     between_rows,
        "ledger_rows":      ledger_rows,
        "centroid_vecs":    centroid_vecs,
        "cluster_vecs":     cluster_vecs,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment":     "exp_006_regional_divergence",
        "date":           datetime.now().strftime("%Y-%m-%d"),
        "encoder":        "facebook/dinov2-small (frozen)",
        "adapter":        "none (per exp_005b decision)",
        "dataset":        "CIFAR-10",
        "label_assignment": CAT_TO_CARROLL,
        "n_agents":       len(ALL_AGENT_IDS),
        "clusters":       {k: list(v) for k, v in CLUSTERS.items()},
        "cluster_image_slices":          {k: list(v) for k, v in CLUSTER_IMAGE_SLICES.items()},
        "held_out_slice":                list(HELD_OUT_SLICE),
        "agent_seed_slices_per_cluster": [list(s) for s in AGENT_SEED_SLICES_PER_CLUSTER],
        "interaction_slice_per_cluster": list(INTERACTION_SLICE_PER_CLUSTER),
        "regimes":        REGIME_CONFIG,
        "max_rounds":     MAX_ROUNDS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "random_seed":    RANDOM_SEED,
        "success_criteria": SUCCESS_CRITERIA,
    }
    with open(RESULTS_DIR / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("[Config] Written.")

    perception = PerceptionLayer(device="cpu")
    print("\n[Data] Loading CIFAR-10 ...")
    all_images = load_all_images()
    embeddings, ground_truth = encode_all_images(all_images, perception)

    pools = build_pools()
    base_rng = np.random.default_rng(RANDOM_SEED)

    all_round_rows:     list[dict] = []
    all_within_rows:    list[dict] = []
    all_between_rows:   list[dict] = []
    all_ledger_rows:    list[dict] = []
    all_centroid_vecs:  dict[str, np.ndarray] = {}
    all_cluster_vecs:   dict[str, np.ndarray] = {}

    for regime in REGIMES:
        bundle = run_regime(regime, pools, embeddings, ground_truth, base_rng)
        all_round_rows.extend(bundle["round_rows"])
        all_within_rows.extend(bundle["within_rows"])
        all_between_rows.extend(bundle["between_rows"])
        all_ledger_rows.extend(bundle["ledger_rows"])
        all_centroid_vecs.update(bundle["centroid_vecs"])
        all_cluster_vecs.update(bundle["cluster_vecs"])

    # Write artifacts
    print(f"\n[Artifacts] -> {RESULTS_DIR}")
    write_csv(all_round_rows,    RESULTS_DIR / "round_log_by_regime.csv")
    write_csv(all_within_rows,   RESULTS_DIR / "within_cluster_metrics.csv")
    write_csv(all_between_rows,  RESULTS_DIR / "between_cluster_metrics.csv")
    write_csv(all_ledger_rows,   RESULTS_DIR / "ledger_events.csv")

    np.savez(RESULTS_DIR / "centroid_vectors.npz",         **all_centroid_vecs)
    print("  Saved: centroid_vectors.npz")
    np.savez(RESULTS_DIR / "cluster_centroid_vectors.npz", **all_cluster_vecs)
    print("  Saved: cluster_centroid_vectors.npz")

    print("\n[Done] Round loop complete. Run metrics_and_plots.py next.")


if __name__ == "__main__":
    run()
