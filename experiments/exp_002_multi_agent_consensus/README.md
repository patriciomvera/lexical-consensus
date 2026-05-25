# Experiment 002 — Multi-Agent Consensus

**Status:** pending (blocked on exp_001 PASS)
**Prerequisite:** exp_001_single_agent_lexicon must pass both conditions

---

## Purpose

Test whether a network of agents can converge on a shared Carroll
vocabulary through the three-layer architecture and distributed
consensus mechanism.

This is the first experiment to engage Layer 3 (ConsensusLedger).
The single-agent behavior is already validated by exp_001; this
experiment introduces the social dynamics: label competition, voting,
stabilization, and rejection.

---

## Design

**Setup:**
- 3 LearnerAgents (identical architecture, independent lexicons)
- 1 Tutor (scripted, same seed set as exp_001)
- 3 Carroll labels: `vorpal`, `slithy`, `mimsy`
- 3 CIFAR-10 categories (those that passed exp_000)
- 25 consensus rounds

**Protocol:**
1. Tutor presents seed images to all agents simultaneously
2. Each round: agents label a new image independently, submit to ledger
3. Ledger computes agreement; labels above 70% threshold stabilize
4. Labels below 20% agreement are rejected
5. After 25 rounds: measure convergence, label stability, and both conditions

---

## Success Criteria

**Convergence:** all 3 labels reach >= 70% agreement by round 25.

**Condition 1 — Naming (post-consensus):**
Agents label held-out images with >= 0.80 accuracy using stabilized labels.

**Condition 2 — Inverse grounding (post-consensus):**
Agents identify correct image from label alone at >= 0.70 accuracy.

**Convergence round per label:**
The round at which each label first reaches and holds 70% agreement.
Lower is better — this is the primary efficiency metric.

---

## Metrics

| Metric | Description |
|---|---|
| Convergence round per label | First round with stable >= 70% agreement |
| Agreement trajectory | Per-label agreement % per round (time series) |
| Naming accuracy (post-consensus) | Condition 1 on held-out images |
| Inverse grounding accuracy | Condition 2 from label string to image |
| Inter-agent centroid distance | L2 distance between agents' centroids per label |
| Rejection rate | % of label proposals rejected per round |

---

## Configuration

| Parameter | Value |
|---|---|
| Number of agents | 3 |
| Carroll labels | vorpal, slithy, mimsy |
| CIFAR-10 categories | from exp_000 results |
| Images per category | 50 |
| Seed images (tutor) | 5 per category |
| Rounds | 25 |
| Convergence threshold | 0.70 |
| Rejection threshold | 0.20 |
| Random seed | 42 |

---

## Artifacts

All outputs written to `results/exp_002_multi_agent_consensus/`:

| File | Contents |
|---|---|
| `config.yaml` | Full experiment configuration |
| `metrics.json` | Convergence rounds, accuracy, agreement trajectories |
| `report.md` | Human-readable summary and interpretation |
| `agreement_curves.png` | Per-label agreement % over rounds |
| `centroid_trajectories.json` | Per-agent per-label centroid per round |

---

## Connection to Subsequent Experiments

A successful exp_002 is the first result of the paper. All subsequent
experiments (Neo4j integration, centroid drift, regional divergence)
build on the baseline established here.

```
exp_002 PASS  ->  exp_003 (Neo4j + Shannon metrics) may begin
exp_002 FAIL  ->  diagnose ledger logic; do not proceed to Neo4j
```
