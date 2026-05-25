# Experiment 001 — Single Agent Lexicon

**Status:** pending (blocked on exp_000 PASS)
**Prerequisite:** exp_000_embedding_separability must produce silhouette > 0.3

---

## Purpose

Verify that a single LearnerAgent can acquire and apply a Carroll
vocabulary through tutor instruction alone — before introducing the
complexity of multi-agent consensus.

This experiment isolates the agent architecture (Layers 1 and 2) from
the consensus mechanism (Layer 3). If a single agent cannot learn to
classify consistently, adding more agents will not fix it.

---

## Design

**Setup:**
- 1 LearnerAgent
- 1 Tutor (scripted, deterministic)
- 3 Carroll labels: `vorpal`, `slithy`, `mimsy`
- 3 CIFAR-10 categories (those that passed exp_000)
- 50 images per category; 5 seed images per category shown to tutor

**Protocol:**
1. Tutor presents 5 seed images per category with Carroll label
2. Agent builds centroids from seed set
3. Agent is tested on 45 held-out images per category
4. Agent is tested on Condition 2 (inverse grounding)

---

## Success Criteria

**Condition 1 — Naming accuracy >= 0.80**
Agent correctly labels held-out images at 80% accuracy or above.

**Condition 2 — Inverse grounding accuracy >= 0.70**
Given only the string "vorpal" (or "slithy" or "mimsy"), the agent
identifies the correct image from a candidate set at 70% accuracy or
above.

Both conditions must pass. Condition 2 is the harder, more important
one — it distinguishes genuine grounding from memorization.

---

## Metrics

| Metric | Description |
|---|---|
| Classification accuracy | % correct labels on held-out images, per category |
| Label-image entropy H(label\|image) | Uncertainty in label assignment per image |
| Centroid stability | L2 distance between centroid after 5 seeds vs. 50 seeds |
| Confusion matrix | Per-category mis-labeling rates |
| Inverse grounding accuracy | % correct image identification from label only |

---

## Configuration

| Parameter | Value |
|---|---|
| Carroll labels | vorpal, slithy, mimsy |
| CIFAR-10 categories | from exp_000 results |
| Images per category | 50 |
| Seed images (tutor) | 5 per category |
| Random seed | 42 |
| Naming threshold | 0.80 |
| Inverse grounding threshold | 0.70 |

---

## Artifacts

All outputs written to `results/exp_001_single_agent_lexicon/`:

| File | Contents |
|---|---|
| `config.yaml` | Full experiment configuration |
| `metrics.json` | Accuracy, entropy, confusion matrix, PASS/FAIL flags |
| `report.md` | Human-readable summary |
| `centroid_positions.json` | Per-label centroid vectors (for inspection) |

---

## Connection to Subsequent Experiments

```
exp_001 PASS (both conditions)  ->  proceed to exp_002_multi_agent_consensus
exp_001 PASS (Condition 1 only) ->  debug Condition 2 before proceeding
exp_001 FAIL                    ->  debug agent architecture; do not proceed
```
