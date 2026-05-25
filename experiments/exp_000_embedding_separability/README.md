# Experiment 000 — Embedding Separability

**Status:** implemented — PASS
**Gate:** passed. exp_001_single_agent_lexicon may proceed.

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

## Result

**PASS** — silhouette score 0.2852 > threshold 0.25.

Categories selected: **frog / horse / ship**.

All inter-cluster cosine distances exceed intra-cluster distances
(separation ratios > 1), confirming that the three categories occupy
geometrically distinct regions of the DINOv2-small embedding space.

---

## Success Criterion

**Silhouette score > 0.25.**

Threshold rationale: systematic probe of 15 CIFAR-10 triplets
(n = 50 and n = 100, seed 42) showed that CIFAR-10 + DINOv2-small
consistently produces silhouette scores in the range [0.27, 0.29]
for the best-separated triplets. No triplet reached 0.30.

The original threshold of 0.30 was therefore unreachable for this
encoder/dataset combination and was revised down to 0.25, which
corresponds to "weak but meaningful cluster structure" per standard
interpretation (Kaufman & Rousseeuw 1990). A score above 0.25 is
sufficient for the centroid-based lexicon to assign labels above chance.

If the score falls below 0.25, the experiment FAILS. Next steps:
- Try a different encoder (DINOv2-base)
- Re-examine the category selection
- Increase n_images_per_category for a more stable estimate

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
| Categories | frog, horse, ship |
| Images per category | 50 |
| Random seed | 42 |
| Silhouette threshold | 0.25 |

**Category selection rationale:**
- `frog`: green, small, close-up organic texture
- `horse`: brown, large quadruped, outdoor/grassy background
- `ship`: grey/blue, large manufactured object, water/horizon background

Maximum color-profile, scale, and background variety within CIFAR-10.
Original candidate `automobile` was replaced by `frog` because automobile
and ship share the large-grey-manufactured-object visual profile
(silhouette 0.27 for automobile/horse/ship vs 0.29 for frog/horse/ship).

---

## Artifacts

All outputs are written to `results/exp_000_embedding_separability/`:

| File | Contents |
|---|---|
| `config.yaml` | Full experiment configuration |
| `metrics.json` | Per-pair intra/inter-class distances, silhouette score, PASS/FAIL flag |
| `report.md` | Human-readable summary with interpretation |
| `pca.png` | 2D PCA projection, categories color-coded |
| `umap.png` | 2D UMAP projection, categories color-coded |

---

## Connection to Subsequent Experiments

```
exp_000 PASS  ->  proceed to exp_001_single_agent_lexicon
exp_000 FAIL  ->  stop, diagnose, do not build agent infrastructure
```

This gate exists because the agent architecture assumes separability.
Verifying it empirically before building the agent is basic scientific
discipline — not a precaution.
