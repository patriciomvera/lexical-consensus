# Condition A — Random Label Control

**Experiment:** exp_002_grounding_controls
**Date:** 2026-05-25
**Verdict:** FAIL (did not degrade)

---

## Design

Scrambled Carroll label assignments (seed=99) across all 30 seed images. Each centroid is built from a mixture of embeddings drawn from multiple real categories. The agent perceives correctly but associates arbitrary labels.

**Expected result:** ≈ chance (0.33) — labels carry no category-consistent signal.

---

## Condition 1 Results (Naming)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.5500 |
| Mean margin | 0.0185 |
| Threshold (≤ for degradation) | 0.45 |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
| slithy | 0.6000 |
| mimsy | 0.9000 |
| vorpal | 0.1500 |

Confusion matrix:

| True \ Pred | slithy | mimsy | vorpal |
|---|---|---|---|
| slithy | 12 | 0 | 8 |
| mimsy | 0 | 18 | 2 |
| vorpal | 14 | 3 | 3 |

---

## Condition 2 Medium Results (Inverse Grounding)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.4333 |
| Mean margin | -0.0130 |

Per-label accuracy:

| Label | Accuracy |
|-------|----------|
| slithy | 0.2000 |
| mimsy | 0.9000 |
| vorpal | 0.2000 |

---

## Interpretation

The accuracy did NOT fall within the expected degraded range. This is an unexpected finding requiring investigation.
