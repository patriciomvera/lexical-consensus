    # Experiment 001b — Single Agent Lexicon (n_seeds = 10)

    **Date:** 2026-05-25
    **Encoder:** facebook/dinov2-small
    **Dataset:** CIFAR-10
    **Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
    **Seeds per category:** 10
    **Test images per category:** 20 (fixed: same images across 001a/b/c)
    **Random seed:** 42

    ---

    ## Verdict: PASS

    The agent correctly acquires Carroll labels from the seed set and generalises to held-out images. All three success criteria pass.

    ---

    ## Success Criteria

    | Criterion | Value | Threshold | Result |
    |---|---|---|---|
    | Overall accuracy | 1.0000 | >= 0.7 | PASS |
| Min per-label accuracy | 1.0000 | >= 0.6 | PASS |
| Mean margin | 0.4045 | >= 0.05 | PASS |

    ---

    ## Per-Label Accuracy

    | Category | Carroll label | Accuracy | Result |
    |---|---|---|---|
    | frog  | slithy   | 1.0000 | PASS |
| horse | mimsy    | 1.0000 | PASS |
| ship  | vorpal   | 1.0000 | PASS |

    ---

    ## Information-Theoretic Metrics

    **Mean H(label | image):** 1.5510 bits
    Entropy of the softmax label distribution per image, averaged over the
    test set. Maximum for 3 balanced classes = 1.585 bits. Lower = the
    classifier is more decisive about each image.

    **Mean margin (d₂ − d₁):** 0.4045
    Average gap between the nearest and second-nearest centroid distance.
    Higher = clearer decision boundary at classification time.

    ---

    ## Centroid Drift

    Cosine distance of the centroid before and after each seed addition.
    Full trajectory in centroid_drift.csv.

    | Carroll label | Mean shift | Max shift | Shift at final seed |
    |---|---|---|---|
    | slithy   | 0.053224 | 0.192971 | 0.012379 |
| mimsy    | 0.033875 | 0.108748 | 0.009732 |
| vorpal   | 0.043953 | 0.188094 | 0.006556 |

    The drift curve measures how quickly the centroid stabilises as seeds
    accumulate. A large shift at seed 2 followed by small shifts at seeds
    3-N indicates the centroid converged early. Persistent large shifts
    indicate the seed images themselves are highly variable.

    ---

    ## Confusion Matrix

    Rows = true Carroll label. Columns = predicted Carroll label.
    See confusion_matrix.png for the visual.

      slithy    mimsy     vorpal  
      slithy        20         0         0  
  mimsy          0        20         0  
  vorpal         0         0        20  

    Total test images: 60
    Correctly classified: 60

    ---

    ## Interpretation

    With 10 seed images per category, the centroid-based lexicon
    achieves all required thresholds. The geometric substrate established in exp_000 is sufficient for single-agent Carroll label acquisition at this seed count.
