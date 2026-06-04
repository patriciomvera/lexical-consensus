"""
run.py — exp_005b: three-condition language-conditioned-geometry experiment.

Three parallel conditions, identical except for the W-update rule:

  A — frozen     W never changes. Reproduces exp_005 under the new architecture
                 and serves as the baseline curve for the alignment plots.
  B — consensus  W is trained via Adam on cosine-distance losses driven by
                 the real ACCEPTED / REJECTED consensus events. This is the
                 hypothesis condition: does consensus feedback reshape
                 categorical geometry when there is a mechanism for it?
  C — random     Same number of gradient steps as B, but the target labels
                 are sampled uniformly from CARROLL_LABELS. Attribution
                 control: any alignment B achieves over C is attributable to
                 consensus *structure*, not merely to gradient training.

All three conditions share data, seed splits, encoder, and centroid-accumulation
rule. Per-condition agents start from the same W (same RNG seed per agent)
so any divergence after round 1 is attributable to the W-update rule alone.

Output layout (results/exp_005b_language_conditioned_geometry/):
  config.yaml                          — full experiment configuration
  adapter_weights_initial.npz          — every agent x condition W at t=0
  adapter_weights_final.npz            — every agent x condition W at t=end
  projected_centroid_vectors.npz       — per (condition, agent, label, round)
  raw_centroid_vectors.npz             — same, but in raw DINOv2 space
  round_log_by_condition.csv           — every per-round metric (for plots)
  adapter_drift.csv                    — frobenius diagnostics per round
  accepted_images.csv                  — which images each agent accepted
  ledger_events.csv                    — assignment-level log (for Neo4j)

Run:
    python -m experiments.exp_005b_language_conditioned_geometry.run
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.lexical_adapter import LexicalAdapter        # noqa: E402
from src.agents.perception      import PerceptionLayer       # noqa: E402

from experiments.exp_005b_language_conditioned_geometry._shared import (  # noqa: E402
    AGENT_IDS, CARROLL_LABELS, CAT_TO_CARROLL,
    AGENT_SEED_SLICES, INTERACTION_SLICE, HELD_OUT_SLICE,
    CONDITIONS, CONDITION_LABELS,
    MAX_ROUNDS, ADAPTER_EPSILON, RANDOM_SEED,
    LEARNING_RATE, LAMBDA_REJECT, BETA_FROBENIUS, REJECT_MARGIN,
    EMBEDDING_DIM, SUCCESS_CRITERIA,
    AgentState, EncodedData, load_encoded_data, seed_agent_state,
    project_many, rebuild_centroids,
    label_pool, compute_consensus, round_metrics, accumulate_accepted,
    write_csv,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_005b_language_conditioned_geometry"


# ─── Condition initialization ─────────────────────────────────────────────────

def init_states_for_condition(condition: str) -> list[AgentState]:
    """
    Build one AgentState per agent for this condition.

    Each agent's adapter is initialized with a deterministic per-agent RNG
    seed (RANDOM_SEED + agent_index) — so agent_00 has the same starting W
    in every condition. This is essential: with random per-condition inits
    we could not separate "B converged because of consensus signal" from
    "B happened to start closer to the truth".
    """
    states: list[AgentState] = []
    for idx, agent_id in enumerate(AGENT_IDS):
        torch.manual_seed(RANDOM_SEED + idx)
        np.random.seed(RANDOM_SEED + idx)
        adapter = LexicalAdapter(dim=EMBEDDING_DIM, epsilon=ADAPTER_EPSILON)
        states.append(AgentState(agent_id=agent_id, adapter=adapter))
    return states


# ─── W-update rules per condition ─────────────────────────────────────────────

def apply_w_update_consensus(
    state: AgentState,
    optimizer: torch.optim.Optimizer,
    detailed_interaction: dict[str, dict],
    consensus: dict[str, dict],
    embeddings: dict[str, np.ndarray],
) -> None:
    """
    Condition B's update. For each interaction image, one Adam step on:

        loss = loss_accept           if agent_label == majority (ACCEPTED)
             | lambda * loss_reject  if agent_label != majority (REJECTED)
             + beta * ||W - I||_F^2  (always)

      loss_accept  = cosine_distance(z, centroid_majority_label)
      loss_reject  = max(0, margin - cosine_distance(z, centroid_rejected_label))

    The reject loss is a hinge — it stops applying force once z is further
    than `margin` from the rejected centroid. Without the hinge, REJECTED
    feedback would push z arbitrarily far and the geometry would distort
    without bound.

    UNRESOLVED (no majority) images are skipped — there's no consensus
    signal to act on. UNCERTAIN agent labels are also skipped — the agent
    declined to vote and gets no gradient update.

    centroids used in the loss are the projected centroids computed at the
    END of the previous round (or at seed-injection time for round 1).
    They are detached from autograd so we never backpropagate through other
    images' contributions to the centroid.
    """
    centroids = {
        lbl: torch.from_numpy(c.astype(np.float32))
        for lbl, c in state.projected_centroid.items()
    }
    if not centroids:
        return  # no centroids yet, nothing to push toward or away from

    eye = torch.eye(EMBEDDING_DIM)

    details = detailed_interaction[state.agent_id]
    for img_hash, det in details.items():
        agent_lbl = det["label"]
        if agent_lbl == "UNCERTAIN":
            continue
        cons     = consensus.get(img_hash, {})
        majority = cons.get("majority_label")
        if majority is None or majority == "UNCERTAIN":
            continue  # unresolved or all-uncertain — no signal

        x = torch.from_numpy(embeddings[img_hash].astype(np.float32))

        optimizer.zero_grad()
        z = state.adapter(x)

        if agent_lbl == majority:
            target = centroids.get(majority)
            if target is None:
                continue
            loss = 1.0 - torch.dot(z, target)
        else:
            # REJECTED: push z away from the centroid of the agent's own label
            target = centroids.get(agent_lbl)
            if target is None:
                continue
            dist = 1.0 - torch.dot(z, target)
            loss = LAMBDA_REJECT * torch.clamp(REJECT_MARGIN - dist, min=0.0)

        reg = BETA_FROBENIUS * torch.norm(state.adapter.W - eye) ** 2
        (loss + reg).backward()
        optimizer.step()


def apply_w_update_random(
    state: AgentState,
    optimizer: torch.optim.Optimizer,
    detailed_interaction: dict[str, dict],
    embeddings: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> None:
    """
    Condition C's update. Same number of gradient steps as B (one per
    interaction image per agent), but each image's "target" Carroll label
    is drawn uniformly at random — independent of what the consensus
    actually decided.

    Only the ACCEPTED-style update is applied (pull z toward the random
    centroid). The reject branch is not used because there's no signal
    being negated — the random label IS the noise. Matching B's step count
    is what the spec requires, not matching B's per-step composition; this
    keeps the comparison clean ("did the structure of B's signal matter,
    or just the volume of training?").
    """
    centroids = {
        lbl: torch.from_numpy(c.astype(np.float32))
        for lbl, c in state.projected_centroid.items()
    }
    if not centroids:
        return

    eye = torch.eye(EMBEDDING_DIM)

    details = detailed_interaction[state.agent_id]
    for img_hash, det in details.items():
        if det["label"] == "UNCERTAIN":
            # Skip UNCERTAIN to match B (UNCERTAIN agents got no update in B)
            continue
        random_label = CARROLL_LABELS[int(rng.integers(0, len(CARROLL_LABELS)))]
        target = centroids.get(random_label)
        if target is None:
            continue

        x = torch.from_numpy(embeddings[img_hash].astype(np.float32))
        optimizer.zero_grad()
        z = state.adapter(x)
        loss = 1.0 - torch.dot(z, target)
        reg  = BETA_FROBENIUS * torch.norm(state.adapter.W - eye) ** 2
        (loss + reg).backward()
        optimizer.step()


# ─── Single-condition round loop ──────────────────────────────────────────────

def run_condition(
    condition: str,
    data: EncodedData,
) -> dict:
    """
    Run all rounds for one condition. Returns a bundle of artifacts:
        round_rows           — list of per-round metric dicts
        adapter_drift_rows   — list of per-(round, agent) frobenius rows
        projected_centroids  — {f'{condition}_{agent}_{label}_{round}': vec384}
        raw_centroids        — same, raw DINOv2 space
        ledger_rows          — assignment-level log for Neo4j
        accepted_rows        — final accepted_by_label snapshot
        states               — final AgentState list (for adapter saving)
        initial_W_per_agent  — initial W copies for the npz file
    """
    print(f"\n[{CONDITION_LABELS[condition]}] starting ...")
    states = init_states_for_condition(condition)

    initial_W_per_agent: dict[str, np.ndarray] = {
        s.agent_id: s.adapter.W.detach().cpu().numpy().copy()
        for s in states
    }

    # Per-agent Adam optimizers — only used by conditions B and C.
    optimizers: dict[str, torch.optim.Optimizer] = {
        s.agent_id: torch.optim.Adam(s.adapter.parameters(), lr=LEARNING_RATE)
        for s in states
    }

    # Seed each agent with its tutor images, project, build centroids
    for state in states:
        seed_agent_state(state, data.agent_seed_hashes[state.agent_id])
    projected_per_agent: dict[str, dict[str, np.ndarray]] = {
        s.agent_id: project_many(s.adapter, data.embeddings) for s in states
    }
    for state in states:
        rebuild_centroids(state, data.embeddings, projected_per_agent[state.agent_id])

    round_rows:         list[dict] = []
    drift_rows:         list[dict] = []
    ledger_rows:        list[dict] = []
    projected_vectors:  dict[str, np.ndarray] = {}
    raw_vectors:        dict[str, np.ndarray] = {}

    # Per-agent RNGs for C — independent of any other randomness so changing
    # the loop order in A or B doesn't shift C's random draws.
    random_rngs = {
        s.agent_id: np.random.default_rng(RANDOM_SEED + 1000 + idx)
        for idx, s in enumerate(states)
    }

    for round_num in range(1, MAX_ROUNDS + 1):
        # 1) label interaction pool with current (W, centroids)
        detailed_int = label_pool(states, data.interaction_hashes, projected_per_agent)
        agent_labels_int = {
            aid: {h: d["label"] for h, d in dets.items()}
            for aid, dets in detailed_int.items()
        }
        consensus_int = compute_consensus(agent_labels_int)

        # 2) ledger rows for this round (interaction)
        for aid, dets in detailed_int.items():
            for img_hash, det in dets.items():
                majority = consensus_int[img_hash]["majority_label"]
                status   = ("unresolved" if majority is None
                            else "accepted" if det["label"] == majority
                            else "rejected")
                ledger_rows.append({
                    "experiment_id":   f"exp_005b_{condition}",
                    "condition":       condition,
                    "round":           round_num,
                    "agent_id":        aid,
                    "image_id":        img_hash,
                    "true_category":   data.interaction_hashes.get(img_hash, ""),
                    "predicted_label": det["label"],
                    "confidence":      det["confidence"],
                    "margin":          det["margin"],
                    "status":          status,
                    "pool":            "interaction",
                })

        # 3) accumulate accepted images (mutates state.accepted_by_label)
        accumulate_accepted(states, detailed_int, consensus_int)

        # 4) update W per condition rule
        if condition == "consensus":
            for state in states:
                apply_w_update_consensus(
                    state, optimizers[state.agent_id], detailed_int, consensus_int,
                    data.embeddings,
                )
        elif condition == "random":
            for state in states:
                apply_w_update_random(
                    state, optimizers[state.agent_id], detailed_int, data.embeddings,
                    random_rngs[state.agent_id],
                )
        # condition == "frozen" : no W update

        # 5) re-project and rebuild centroids using updated W
        projected_per_agent = {
            s.agent_id: project_many(s.adapter, data.embeddings) for s in states
        }
        for state in states:
            rebuild_centroids(state, data.embeddings, projected_per_agent[state.agent_id])

        # 6) snapshot centroid vectors (after W update — that's what next round uses)
        for state in states:
            for label in CARROLL_LABELS:
                if label in state.projected_centroid:
                    key = f"{condition}_{state.agent_id}_{label}_{round_num}"
                    projected_vectors[key] = state.projected_centroid[label].astype(np.float32)
                if label in state.raw_centroid:
                    key = f"{condition}_{state.agent_id}_{label}_{round_num}"
                    raw_vectors[key] = state.raw_centroid[label].astype(np.float32)

        # 7) held-out evaluation (no feedback, no accumulation)
        detailed_held = label_pool(states, data.held_out_hashes, projected_per_agent)
        agent_labels_held = {
            aid: {h: d["label"] for h, d in dets.items()}
            for aid, dets in detailed_held.items()
        }
        consensus_held = compute_consensus(agent_labels_held)

        for aid, dets in detailed_held.items():
            for img_hash, det in dets.items():
                majority = consensus_held[img_hash]["majority_label"]
                status   = ("unresolved" if majority is None
                            else "accepted" if det["label"] == majority
                            else "rejected")
                ledger_rows.append({
                    "experiment_id":   f"exp_005b_{condition}",
                    "condition":       condition,
                    "round":           round_num,
                    "agent_id":        aid,
                    "image_id":        img_hash,
                    "true_category":   data.held_out_hashes.get(img_hash, ""),
                    "predicted_label": det["label"],
                    "confidence":      det["confidence"],
                    "margin":          det["margin"],
                    "status":          status,
                    "pool":            "held_out",
                })

        # 8) per-round metric row
        metrics = round_metrics(consensus_int, consensus_held, data.ground_truth)
        # Per-agent held-out accuracy (no consensus aggregation)
        per_agent_acc: dict[str, float] = {}
        for aid, labels in agent_labels_held.items():
            n = len(data.held_out_hashes)
            correct = sum(
                1 for h, lbl in labels.items()
                if lbl == data.ground_truth.get(h)
            )
            per_agent_acc[aid] = round(correct / n, 4) if n > 0 else 0.0

        row: dict = {
            "condition": condition,
            "round":     round_num,
            **metrics,
            **{f"per_agent_accuracy_{aid}": acc for aid, acc in per_agent_acc.items()},
        }
        round_rows.append(row)

        # 9) per-agent adapter drift this round
        for state in states:
            drift_rows.append({
                "condition":                       condition,
                "round":                           round_num,
                "agent_id":                       state.agent_id,
                "frobenius_distance_to_identity": round(state.adapter.frobenius_distance_to_identity(), 6),
                "frobenius_drift_from_initial":   round(state.adapter.frobenius_drift_from_initial(),   6),
            })

        print(
            f"  R{round_num:3d} | "
            f"major={metrics['majority_agreement']:.3f} "
            f"unan={metrics['unanimous_agreement']:.3f} "
            f"H={metrics['mean_entropy']:.3f} | "
            f"held_acc={metrics['held_out_consensus_accuracy']:.3f} "
            f"cov={metrics['label_coverage']} | "
            f"||W-W0||F=[" +
            ", ".join(
                f"{s.adapter.frobenius_drift_from_initial():.3f}" for s in states
            ) + "]"
        )

    # Final accepted-images snapshot for diagnostics
    accepted_rows: list[dict] = []
    for state in states:
        for label, hashes in state.accepted_by_label.items():
            for h in sorted(hashes):
                accepted_rows.append({
                    "condition": condition,
                    "agent_id":  state.agent_id,
                    "label":     label,
                    "image_id":  h,
                })

    return {
        "round_rows":           round_rows,
        "drift_rows":           drift_rows,
        "ledger_rows":          ledger_rows,
        "projected_vectors":    projected_vectors,
        "raw_vectors":          raw_vectors,
        "accepted_rows":        accepted_rows,
        "states":               states,
        "initial_W_per_agent":  initial_W_per_agent,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment":          "exp_005b_language_conditioned_geometry",
        "date":                datetime.now().strftime("%Y-%m-%d"),
        "encoder":             "facebook/dinov2-small (frozen)",
        "adapter":             "LexicalAdapter (W = I + epsilon*N(0,1), trainable per agent)",
        "dataset":             "CIFAR-10",
        "label_assignment":    CAT_TO_CARROLL,
        "agent_seed_slices":   {k: list(v) for k, v in AGENT_SEED_SLICES.items()},
        "interaction_slice":   list(INTERACTION_SLICE),
        "held_out_slice":      list(HELD_OUT_SLICE),
        "n_agents":            len(AGENT_IDS),
        "conditions":          CONDITIONS,
        "max_rounds":          MAX_ROUNDS,
        "adapter_epsilon":     ADAPTER_EPSILON,
        "learning_rate":       LEARNING_RATE,
        "lambda_reject":       LAMBDA_REJECT,
        "beta_frobenius":      BETA_FROBENIUS,
        "reject_margin":       REJECT_MARGIN,
        "random_seed":         RANDOM_SEED,
        "success_criteria":    SUCCESS_CRITERIA,
    }
    with open(RESULTS_DIR / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("[Config] Written.")

    # Encode all images once — shared across conditions
    perception = PerceptionLayer(device="cpu")
    data = load_encoded_data(perception)

    # Run each condition sequentially. They share data but maintain
    # independent agent states (adapter, accepted_by_label, centroids).
    all_round_rows:        list[dict] = []
    all_drift_rows:        list[dict] = []
    all_ledger_rows:       list[dict] = []
    all_accepted_rows:     list[dict] = []
    all_projected_vectors: dict[str, np.ndarray] = {}
    all_raw_vectors:       dict[str, np.ndarray] = {}
    all_initial_W:         dict[str, np.ndarray] = {}
    all_final_W:           dict[str, np.ndarray] = {}

    for condition in CONDITIONS:
        bundle = run_condition(condition, data)
        all_round_rows.extend(bundle["round_rows"])
        all_drift_rows.extend(bundle["drift_rows"])
        all_ledger_rows.extend(bundle["ledger_rows"])
        all_accepted_rows.extend(bundle["accepted_rows"])
        all_projected_vectors.update(bundle["projected_vectors"])
        all_raw_vectors.update(bundle["raw_vectors"])
        for aid, w in bundle["initial_W_per_agent"].items():
            all_initial_W[f"{condition}_{aid}"] = w
        for state in bundle["states"]:
            all_final_W[f"{condition}_{state.agent_id}"] = (
                state.adapter.W.detach().cpu().numpy().copy()
            )

    # Write artifacts
    print(f"\n[Artifacts] -> {RESULTS_DIR}")

    np.savez(
        RESULTS_DIR / "adapter_weights_initial.npz", **all_initial_W,
    )
    print("  Saved: adapter_weights_initial.npz")
    np.savez(
        RESULTS_DIR / "adapter_weights_final.npz", **all_final_W,
    )
    print("  Saved: adapter_weights_final.npz")
    np.savez(
        RESULTS_DIR / "projected_centroid_vectors.npz", **all_projected_vectors,
    )
    print("  Saved: projected_centroid_vectors.npz")
    np.savez(
        RESULTS_DIR / "raw_centroid_vectors.npz", **all_raw_vectors,
    )
    print("  Saved: raw_centroid_vectors.npz")

    write_csv(all_round_rows,    RESULTS_DIR / "round_log_by_condition.csv")
    write_csv(all_drift_rows,    RESULTS_DIR / "adapter_drift.csv")
    write_csv(all_ledger_rows,   RESULTS_DIR / "ledger_events.csv")
    write_csv(all_accepted_rows, RESULTS_DIR / "accepted_images.csv")

    print("\n[Done] Round loop complete. Run metrics_and_plots.py next.")


if __name__ == "__main__":
    run()
