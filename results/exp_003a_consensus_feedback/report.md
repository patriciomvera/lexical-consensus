# Experiment 003a — Multi-Agent Consensus with Feedback

**Date:** 2026-05-28
**Encoder:** facebook/dinov2-small
**Dataset:** CIFAR-10
**Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
**Feedback rule:** ACCEPTED → centroid update | REJECTED/UNRESOLVED → no update

**Seed split:**
- agent_00: indices [0:5]   per category (5 seeds × 3 labels = 15 total)
- agent_01: indices [5:10]  per category
- agent_02: indices [10:15] per category

**Interaction pool:** [15:30] per category (45 images)
**Held-out pool:**    [30:50] per category (60 images)
**Total rounds run:** 6

---

## Verdict: PASS

---

## Final-Round Metrics

| Metric | Value | Target | Result |
|---|---|---|---|
| Held-out inter-agent agreement | 1.0000 | >= 0.90 | PASS |
| Held-out consensus accuracy | 1.0000 | >= 0.90 | PASS |
| Label coverage | 3 | == 3 | PASS |
| Consensus beats individual | True | True | PASS |

**Interaction pool (final round):**
- Majority agreement:  1.0000
- Unanimous agreement: 0.9778
- Consensus accuracy:  0.9778
- Mean entropy:        0.0204 bits
- Stable label count:  45

---

## Per-Agent Accuracy — Held-out (final round)

| Agent | Accuracy |
|---|---|
| agent_00 | 1.0000 |
| agent_01 | 1.0000 |
| agent_02 | 1.0000 |
| Consensus | 1.0000 |

---

## Label Balance — Final Round (interaction pool)

| Label | Category | Fraction as majority |
|---|---|---|
| slithy | frog  | 0.3333 |
| mimsy  | horse | 0.3111  |
| vorpal | ship  | 0.3333 |

Expected: ~0.333 each (15 images per category in interaction pool).

---

## Key Question

Did consensus feedback accelerate agreement and improve accuracy
beyond what shared DINOv2 perception alone produces?

Compare with: results/exp_003b_no_feedback_baseline/report.md
Full comparison: results/exp_003_comparison/summary_report.md
