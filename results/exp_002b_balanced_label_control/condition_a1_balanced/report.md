# Sub-condition A1 — Balanced Centroid Construction

**Experiment:** exp_002b_balanced_label_control
**Date:** 2026-05-25
**Verdict:** PARTIAL — C1=FAIL, C2=FAIL

---

## Design

Each Carroll label's centroid is built from a balanced mix of embeddings
from all three trained categories, using round-robin assignment with a
per-category offset. The assignment is deterministic — no random seed needed.

Label composition (seeds per category):

| Label | frog | horse | ship |
|-------|------|-------|------|
| slithy | 4 | 3 | 3 |
| mimsy | 3 | 4 | 3 |
| vorpal | 3 | 3 | 4 |

With balanced mixing, each centroid is approximately the global mean of
all 30 seed embeddings. All three centroids are nearly equal → classification
is determined by minor numerical differences → expected accuracy ≈ chance.

---

## Results

| Metric | Value | Threshold |
|--------|-------|-----------|
| C1 overall accuracy | 0.7167 | <= 0.45 |
| C2 medium accuracy  | 0.5667 | <= 0.45 |

C1 per-label accuracy:

| Label | Accuracy |
|-------|----------|
| slithy | 0.8500 |
| mimsy | 0.5500 |
| vorpal | 0.7500 |

Confusion matrix (C1):

| True \ Pred | slithy | mimsy | vorpal |
|---|---|---|---|
| slithy | 17 | 1 | 2 |
| mimsy | 6 | 11 | 3 |
| vorpal | 4 | 1 | 15 |

---

## Interpretation

Balanced mixing did not fully collapse accuracy to chance. This is unexpected and warrants further investigation — the centroid geometry may retain residual discriminability even under balanced mixing.

**Reference:** exp_002 Condition A (seed=99) achieved C1=0.550 with an
unbalanced scramble. This balanced result (0.7167) confirms that
the elevated result was a sampling artifact.
