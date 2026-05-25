# Condition E — Harder Category Set

**Experiment:** exp_002_grounding_controls
**Date:** 2026-05-25
**Verdict:** PASS (grounding holds on harder set)

---

## Design

Replace frog/horse/ship with cat/dog/deer — visually similar categories
with overlapping boundary regions in DINOv2 embedding space.
Protocol identical to exp_001b: 10 seeds per category, Condition 1 + Condition 2 medium.

Label assignment: cat → slithy, dog → mimsy, deer → vorpal

**Expected:** degraded but measurable performance. This locates where the
system starts to fail as category boundaries tighten.

---

## Condition 1 Results (Naming)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.9500 |
| Mean margin | 0.1926 |
| Pass threshold | ≥ 0.7 |

Confusion matrix:

| True \ Pred | slithy | mimsy | vorpal |
|---|---|---|---|
| slithy | 20 | 0 | 0 |
| mimsy | 2 | 17 | 1 |
| vorpal | 0 | 0 | 20 |

---

## Condition 2 Medium Results (Inverse Grounding)

| Metric | Value |
|--------|-------|
| Overall accuracy | 0.8667 |
| Mean margin | 0.1930 |

---

## Comparison with exp_001b (frog/horse/ship)

| Metric | exp_001b | Condition E | Delta |
|--------|----------|-------------|-------|
| C1 overall accuracy | 1.000 | 0.9500 | -0.0500 |
| C2 overall accuracy | 1.000 | 0.8667 | -0.1333 |

---

## Interpretation

Performance holds on the harder set. Cat/dog/deer may be less separable in DINOv2 space than expected, or 10 seeds is sufficient to establish robust boundaries even for visually similar categories.
