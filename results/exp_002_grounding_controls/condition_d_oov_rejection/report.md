# Condition D — OOV Rejection

**Experiment:** exp_002_grounding_controls
**Date:** 2026-05-25
**Verdict:** PASS (AUROC meets threshold)

---

## Design

Agent has learned slithy / mimsy / vorpal from 10 seeds each.
All retrieval trial candidates are from untrained categories
(airplane, deer, automobile). The correct answer is NONE.

The agent rejects a trial if `max_similarity(candidate, centroid) < T`,
where T is calibrated on a validation set (seeds [0:10] of trained
categories + OOV[0:10]) using Youden's J criterion.

**Expected:** OOV rejection accuracy > 0.80, AUROC ≥ 0.80

---

## Threshold Calibration

| Parameter | Value |
|-----------|-------|
| Optimal threshold (T) | 0.5289 |
| Calibration set | trained seeds [0:9] (pos) + OOV[0:9] (neg) |

---

## OOV Rejection Metrics

| Metric | Value |
|--------|-------|
| OOV rejection accuracy | 1.0000 |
| False accept rate (FAR) | 0.0000 |
| False reject rate (FRR) | 0.3000 |
| AUROC | 0.9642 |
| AUROC threshold | 0.8 |

---

## Interpretation

AUROC above threshold confirms that the learned centroid geometry provides a usable signal for distinguishing in-vocabulary from out-of-vocabulary images, even without explicit negative training. The agent rejects OOV candidates based solely on the weakness of their similarity to any learned centroid.
