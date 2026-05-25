# Experiment 000 — Embedding Separability

**Status:** pending implementation
**Gate:** must PASS before exp_001 may run

---

## Purpose

This is a standalone, falsifiable experiment — not a warm-up step.

It asks one question: **Do the CIFAR-10 visual categories that will be
used in subsequent experiments form separable clusters in DINOv2 (small)
embedding space?**

If they do not, the centroid-based lexicon (Layer 2) has no geometric
substrate. Labels assigned to centroids would overlap in embedding
space, making consistent label retrieval impossible. No amount of
tuning the consensus mechanism can fix a failed perceptual substrate.

---

## Success Criterion

**Silhouette score > 0.3** across the selected CIFAR-10 categories.

This threshold is conservative. A score of 0.3 means clusters are
discernible but not perfectly separated — sufficient for the
nearest-centroid assignment in the lexicon to work above chance.

If the score falls below 0.3, the experiment FAILS and the next step
is to investigate which category pairs are most confused, then either:
- Select different CIFAR-10 categories with better separation
- Evaluate a different encoder (DINOv2-base instead of small)

---

## Script

`diagnostic.py` — main and only script for this experiment.

```
python -m experiments.exp_000_embedding_separability.diagnostic
```

---

## Configuration

| Parameter | Value |
|---|---|
| Encoder | DINOv2-small (frozen) |
| Dataset | CIFAR-10 |
| Images per category | 50 |
| Categories | TBD — chosen for semantic distance |
| Random seed | 42 |
| Silhouette threshold | 0.3 |

---

## Artifacts

All outputs are written to `results/exp_000_embedding_separability/`:

| File | Contents |
|---|---|
| `config.yaml` | Full experiment configuration |
| `metrics.json` | Per-pair intra/inter-class distances, silhouette score, PASS/FAIL flag |
| `report.md` | Human-readable summary with interpretation |
| `umap_projection.png` | 2D UMAP projection of all embeddings, colored by category |

---

## Connection to Subsequent Experiments

```
exp_000 PASS  ->  proceed to exp_001_single_agent_lexicon
exp_000 FAIL  ->  stop, diagnose, do not build agent infrastructure
```

This gate exists because the agent architecture assumes separability.
Verifying it empirically before building the agent is basic scientific
discipline — not a precaution.
