# Condition B — Random Embedding Control

**Experiment:** exp_002_grounding_controls
**Date:** 2026-05-25
**Verdict:** PASS (degraded as expected)

---

## Design

All DINOv2 embeddings (seeds and test images) replaced with independent random unit vectors (seed=99). Labels and category assignments preserved. Centroids are means of random vectors, carrying no visual structure.

**Expected result:** ≈ chance (0.33) — no visual structure in embedding space.

---

## Condition 1 Results (Naming)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.3000 |
| Mean margin | -0.0371 |
| Threshold (≤ for degradation) | 0.45 |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
| slithy | 0.3500 |
| mimsy | 0.4000 |
| vorpal | 0.1500 |

Confusion matrix:

| True \ Pred | slithy | mimsy | vorpal |
|---|---|---|---|
| slithy | 7 | 6 | 7 |
| mimsy | 6 | 8 | 6 |
| vorpal | 8 | 9 | 3 |

---

## Condition 2 Medium Results (Inverse Grounding)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.2500 |
| Mean margin | -0.0447 |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
| slithy | 0.3000 |
| mimsy | 0.2000 |
| vorpal | 0.2500 |

---

## Interpretation

The accuracy fell within the expected degraded range, confirming that the component under test is necessary for grounding.
