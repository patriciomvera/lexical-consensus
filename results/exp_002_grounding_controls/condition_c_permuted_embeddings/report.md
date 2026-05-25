# Condition C — Permuted Embedding Control

**Experiment:** exp_002_grounding_controls
**Date:** 2026-05-25
**Verdict:** PASS (degraded as expected)

---

## Design

Seed embeddings cyclically permuted across categories: frog seeds receive ship embeddings, horse seeds receive frog embeddings, ship seeds receive horse embeddings. Each Carroll label's centroid is computed from the wrong visual category. Test images retain their real DINOv2 embeddings.

**Expected result:** Systematic misclassification — centroids and test images live in different regions of the embedding space.

---

## Condition 1 Results (Naming)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.0000 |
| Mean margin | -0.4252 |
| Threshold (≤ for degradation) | 0.5 |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
| slithy | 0.0000 |
| mimsy | 0.0000 |
| vorpal | 0.0000 |

Confusion matrix:

| True \ Pred | slithy | mimsy | vorpal |
|---|---|---|---|
| slithy | 0 | 20 | 0 |
| mimsy | 0 | 0 | 20 |
| vorpal | 20 | 0 | 0 |

---

## Condition 2 Medium Results (Inverse Grounding)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.0000 |
| Mean margin | -0.3825 |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
| slithy | 0.0000 |
| mimsy | 0.0000 |
| vorpal | 0.0000 |

---

## Interpretation

The accuracy fell within the expected degraded range, confirming that the component under test is necessary for grounding.
