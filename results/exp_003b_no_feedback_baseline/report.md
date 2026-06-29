# Experiment 003b — No-Feedback Baseline

**Date:** 2026-05-28
**Encoder:** facebook/dinov2-small
**Dataset:** CIFAR-10
**Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
**Feedback rule:** NONE — lexicons frozen after seed phase

**Seed split (identical to 003a):**
- agent_00: indices [0:5]   per category
- agent_01: indices [5:10]  per category
- agent_02: indices [10:15] per category

**Interaction pool:** [15:30] per category (45 images)
**Held-out pool:**    [30:50] per category (60 images)
**Total rounds run:** 100

---

## Round-by-round stability

Lexicons do not update after seeding, so metrics should be identical
across all rounds. Observed accuracy range across rounds:
**0.0000** (flat as expected)

---

## Baseline Metrics (constant across all rounds)

| Metric | Value |
|---|---|
| Majority agreement (interaction) | 1.0000 |
| Unanimous agreement (interaction) | 0.9333 |
| Consensus accuracy (interaction) | 0.9556 |
| Held-out inter-agent agreement | 1.0000 |
| Held-out consensus accuracy | 0.9833 |
| Label coverage | 3 |
| Mean entropy | 0.0612 bits |

---

## Per-Agent Accuracy — Held-out

| Agent | Round 1 | Final |
|---|---|---|
| agent_00 | 0.9833 | 0.9833 |
| agent_01 | 1.0000 | 1.0000 |
| agent_02 | 0.9667 | 0.9667 |
| Consensus | 0.9833 | 0.9833 |

---

## Interpretation

These are the agreement and accuracy levels achievable by three agents
with disjoint 5-seed training sets and shared DINOv2 perception, with
NO communication or feedback mechanism.

Any improvement seen in 003a above these values is attributable to the
consensus feedback loop, not to the shared perceptual substrate alone.

Full comparison: results/exp_003_comparison/summary_report.md
