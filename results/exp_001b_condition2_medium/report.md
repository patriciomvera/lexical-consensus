    # Experiment 001b — Condition 2 (MEDIUM)

    **Date:** 2026-05-25
    **Agent:** exp_001b — 10 seeds per category
    **Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
    **Level:** MEDIUM
    **Distractors:** the other two trained categories (e.g. for slithy: mimsy distractor + vorpal distractor)
    **Trials per Carroll label:** 20
    **Total trials:** 60

    ---

    ## Framing Hypothesis

    > If Carroll labels are genuinely grounded in the learned perceptual
    > category structure, the agent should not only name unseen images
    > correctly (Condition 1), but also retrieve unseen images from labels
    > under distractor pressure (Condition 2).

    Condition 1 established behavioral competence at 10 seeds (accuracy = 1.000).
    Condition 2 tests whether that competence reflects genuine grounding —
    a structured internal representation that supports retrieval — or only
    directional naming from a good-enough centroid.

    ---

    ## Behavioral Saturation vs Representational Convergence

    exp_001b (10 seeds) achieved full Condition 1 accuracy but not fully
    converged centroids. Centroid shift at seed 10 vs seed 15:

    | Carroll label | Shift at seed 10 | Shift at seed 15 | Still moving? |
    |---|---|---|---|
    | slithy | 0.012379 | 0.004116 | yes |
    | mimsy  | 0.009732 | 0.002800 | yes |
    | vorpal | 0.006556 | 0.002843 | yes |

    This test is run on the 10-seed agent intentionally: we are asking
    whether behavioral competence (Condition 1 saturation) implies
    representational grounding (Condition 2 performance), even when the
    centroid has not fully converged. If Condition 2 also passes at 10
    seeds, the grounding is robust to centroid imprecision. If it fails,
    the centroid shift between seed 10 and seed 15 is the likely cause.

    ---

    ## Verdict: PASS

    Overall accuracy: **1.0000** (threshold: > 0.7)

    ---

    ## Per-Label Accuracy

    | Category | Carroll label | Accuracy | Result |
    |---|---|---|---|
    | frog  | slithy   | 1.0000 | PASS |
| horse | mimsy    | 1.0000 | PASS |
| ship  | vorpal   | 1.0000 | PASS |

    ---

    ## Retrieval Metrics

    | Metric | Value |
    |---|---|
    | Mean retrieval margin (sim_correct − sim_best_distractor) | 0.3511 |
    | Mean entropy over candidates H(image\|label) | 1.5563 bits |

    **Retrieval margin** is positive when the correct image is closer to the
    target centroid than the best distractor. Negative margin = the agent
    would fail. See margin_distribution.png for the full distribution.

    **Entropy** measures how diffuse the similarity distribution is across
    the 3 candidates. At 3 candidates, maximum entropy = 1.585 bits.
    Lower entropy = more decisive retrieval.


    ## Failure Cases

    | Label | Correct image | Selected (wrong) | Wrong cat | Margin |
    |---|---|---|---|---|
    *(no failures)*

    ---

    ## Interpretation

    The agent successfully retrieves correct images under distractor pressure at the medium difficulty level. Carroll labels are grounded in the centroid structure, not just memorised naming associations.
