    # Experiment 001a — Single Agent Lexicon (n_seeds = 5)

    **Date:** 2026-05-25
    **Encoder:** facebook/dinov2-small
    **Dataset:** CIFAR-10
    **Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
    **Seeds per category:** 5
    **Test images per category:** 20 (fixed: same images across 001a/b/c)
    **Random seed:** 42

    ---

    ## Verdict: PASS

    The agent correctly acquires Carroll labels from the seed set and generalises to held-out images. All three success criteria pass.

    ---

    ## Success Criteria

    | Criterion | Value | Threshold | Result |
    |---|---|---|---|
    | Overall accuracy | 0.9833 | >= 0.7 | PASS |
| Min per-label accuracy | 0.9500 | >= 0.6 | PASS |
| Mean margin | 0.3766 | >= 0.05 | PASS |

    ---

    ## Per-Label Accuracy

    | Category | Carroll label | Accuracy | Result |
    |---|---|---|---|
    | frog  | slithy   | 0.9500 | PASS |
| horse | mimsy    | 1.0000 | PASS |
| ship  | vorpal   | 1.0000 | PASS |

    ---

    ## Information-Theoretic Metrics

    **Mean H(label | image):** 1.5554 bits
    Entropy of the softmax label distribution per image, averaged over the
    test set. Maximum for 3 balanced classes = 1.585 bits. Lower = the
    classifier is more decisive about each image.

    **Mean margin (d₂ − d₁):** 0.3766
    Average gap between the nearest and second-nearest centroid distance.
    Higher = clearer decision boundary at classification time.

    ---

    ## Centroid Drift

    Cosine distance of the centroid before and after each seed addition.
    Full trajectory in centroid_drift.csv.

    | Carroll label | Mean shift | Max shift | Shift at final seed |
    |---|---|---|---|
    | slithy   | 0.096079 | 0.192971 | 0.041525 |
| mimsy    | 0.063638 | 0.108748 | 0.030675 |
| vorpal   | 0.085057 | 0.188094 | 0.026239 |

    The drift curve measures how quickly the centroid stabilises as seeds
    accumulate. A large shift at seed 2 followed by small shifts at seeds
    3-N indicates the centroid converged early. Persistent large shifts
    indicate the seed images themselves are highly variable.

    ---

    ## Confusion Matrix

    Rows = true Carroll label. Columns = predicted Carroll label.
    See confusion_matrix.png for the visual.

      slithy    mimsy     vorpal  
      slithy        19         0         1  
  mimsy          0        20         0  
  vorpal         0         0        20  

    Total test images: 60
    Correctly classified: 59

    ---

    ## Interpretation

    With 5 seed images per category, the centroid-based lexicon
    achieves all required thresholds. The geometric substrate established in exp_000 is sufficient for single-agent Carroll label acquisition at this seed count.
