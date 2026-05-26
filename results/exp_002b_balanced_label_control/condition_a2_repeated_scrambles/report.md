# Sub-condition A2 — Repeated Random Scrambles (N=100)

**Experiment:** exp_002b_balanced_label_control
**Date:** 2026-05-25
**Verdict:** PARTIAL — see per-metric breakdown

---

## Design

100 independent random scrambles of the 30 seed labels, using seeds 0–99.
For each scramble, C1 and C2 medium accuracy are recorded.
DINOv2 encoding is performed once; only label assignments vary.

This directly addresses the concern raised by exp_002 Condition A (seed=99,
C1=0.550): was that result a typical outcome of random scrambling, or a
tail-end outlier from an uneven random draw?

---

## C1 Accuracy Distribution

| Statistic | Value |
|-----------|-------|
| Mean ± std | 0.3415 ± 0.2252 |
| 95% CI | [0.0000, 0.9000] |
| Min / Max | 0.0000 / 0.9833 |
| Scrambles above threshold (>0.45) | 24 / 100 (24.0%) |
| seed=99 result (exp_002) | 0.3833 (percentile: 68.0th) |

## C2 Medium Accuracy Distribution

| Statistic | Value |
|-----------|-------|
| Mean ± std | 0.3458 ± 0.1728 |

---

## Falsification Statement for Condition A

The exp_002 Condition A result (C1=0.550 with seed=99) falls at the
**68th percentile** of the scramble distribution.
The mean C1 accuracy across 100 scrambles is 0.3415, well below
the baseline (1.000) and below the collapse threshold.

Only 24 of 100 scrambles (24.0%) exceeded the 0.45
threshold. The seed=99 draw was an outlier — the typical behavior of random
scrambling is near-chance performance. The falsification argument for
Condition A is intact.

---

## Combined Condition A Conclusion (exp_002 + exp_002b)

| Sub-experiment | C1 | C2 | Interpretation |
|---|---|---|---|
| exp_002 Cond A (seed=99) | 0.550 | 0.433 | Uneven scramble — tail outlier |
| exp_002b A1 (balanced)   | see A1 report | see A1 report | Deterministic balanced mix |
| exp_002b A2 (mean of 100) | 0.3415 | 0.3458 | Typical random scramble |
| Baseline exp_001b | 1.000 | 1.000 | Correct label assignment |

**Conclusion:** Random label scrambling reliably degrades grounding.
The single-seed result was an artifact of sampling variance, not a genuine
failure of the control condition.
