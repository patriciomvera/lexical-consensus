    # Experiment 000 — Embedding Separability Report

    **Date:** 2026-05-25
    **Encoder:** facebook/dinov2-small
    **Dataset:** CIFAR-10
    **Categories:** frog, horse, ship
    **N per category:** 50
    **Random seed:** 42

    ---

    ## Verdict: PASS — suitable for exp_001, with moderate separation

    The selected categories ARE separable in DINOv2-small embedding space. exp_001_single_agent_lexicon may proceed.

    ---

    ## Key Metrics

    | Metric | Value | Threshold | Result |
    |---|---|---|---|
    | Silhouette score | 0.2826 | > 0.25 | PASS |
    | Davies-Bouldin index | 2.2224 | lower is better | — |

    **Silhouette score** measures how well each point fits its own cluster
    vs the nearest other cluster. Range: [-1, 1]. Values above 0.3 indicate
    meaningful separation.

    **Davies-Bouldin index** measures the average ratio of intra-cluster
    scatter to inter-cluster distance. Lower values indicate tighter,
    more separated clusters.

    ---

    ## Intra-Cluster Cosine Distance (lower = tighter clusters)

    | Category | Mean cosine distance |
    |---|---|
    | frog | 0.6962 |
| horse | 0.5725 |
| ship | 0.6266 |

    ---

    ## Inter-Cluster Cosine Distance and Separation Ratio

    | Pair | Inter-cluster distance | Separation ratio (inter/intra) |
    |---|---|---|
    | frog vs horse | 0.9008 | 1.42 |
| frog vs ship | 0.8947 | 1.3528 |
| horse vs ship | 0.8940 | 1.4911 |

    A separation ratio > 1 indicates that inter-cluster distance exceeds
    average intra-cluster distance, suggesting usable separation, though
    not guaranteeing absence of overlap.

    ---

    ## Most Confused Pair

    **horse vs ship** has the lowest inter-cluster cosine distance,
    meaning these two categories are hardest to distinguish in DINOv2
    embedding space.

    This is acceptable — silhouette score still passes the threshold.

    ---

    ## Interpretation

    The centroid-based lexicon in Layer 2 assigns a Carroll label to an
    image by finding the nearest centroid in DINOv2 embedding space. For
    this to work reliably, images of different categories must occupy
    distinct regions of that space.

    A silhouette score of 0.2826 confirms that the three categories form geometrically distinct regions. The lexicon has a viable substrate to operate on.

    The perceptual substrate is viable, but not trivial.
