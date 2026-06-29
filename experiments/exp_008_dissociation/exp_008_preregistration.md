# exp_008 — Perceptual vs Semantic Dissociation: Pre-Registration

**Status:** PRE-REGISTERED SPEC — frozen before Phase 3 data collection.
**Date registered:** 2026-06-08
**Author:** Anonymous

---

This document is frozen at the time of implementation and must not be edited
after Phase 3 data collection begins. It contains the full experimental
specification including pre-registered decision rules (§8).

For the complete specification, see the experiment design provided in the
project chat log on 2026-06-08.

## Summary

This experiment resolves an ambiguity in exp_007b/007b_extension_clean:

> The C1 gradient (native≈0.943 → near=0.847 → mid=0.654 → far=0.530) was
> interpreted as perceptual geometry constraining lexical acquisition.
> A reviewer can counter: concept tiers were defined using inter-centroid
> cosine distance in DINOv2-small, and C1 was measured in the same DINOv2
> space. The gradient may be a measurement tautology.

We introduce an external semantic distance measure (WordNet Wu-Palmer) and
identify dissociation cases where perceptual and semantic coherence disagree.
If C1 tracks d_perc even in dissociation cases, the gradient has genuine
empirical content beyond circularity.

## Dataset: CIFAR-100

100 fine-grained classes × 600 images per class (500 train + 100 test).
Episode pool: 200 images per class from train split, seed=42.

## Phases

1. **Phase 1** — Compute perceptual (DINOv2) and semantic (WordNet) distance matrices.
   Checkpoint: review Pearson r(d_perc, d_sem) before continuing.
2. **Phase 2** — Classify 4,950 pairs into Q1/Q2/Q3/Q4 quadrants; sample 25 per quadrant.
3. **Phase 3** — C1 acquisition episodes (3-way, 5-shot, 30 episodes per pair).
   Learners: centroid, exemplar_knn_k3, logreg, random.
4. **Phase 4** — Multiple regression, dissociation subset analysis, figures, report.

## Pre-Registered Decision Rules

**Rule A — Perception governs** (circularity closed): ALL of the following:
1. In Model PS, β_perc significant (p<0.05) and negative.
2. In Model PS, β_sem not significant (p≥0.10) OR partial_R²_sem < partial_R²_perc / 3.
3. In Q3+Q4: Spearman(C1, d_perc) negative, |ρ|>0.30, stronger than |ρ_sem|.
4. Quadrant boxplot: Q4 median C1 > Q3 median C1.

**Rule B — Ambiguous** (high collinearity or weak dissociation):
Neither A nor C criteria are met.

**Rule C — Semantics wins / gradient breaks**: ANY of:
1. β_perc NOT significant but β_sem IS in Model PS.
2. |ρ_sem| > |ρ_perc| + 0.15 in Q3+Q4 dissociation subset.
3. Q3 median C1 > Q4 median C1 (opposite of perceptual prediction).

## Execution Order

```
python src/phase1/embed_cifar100.py
python src/phase1/compute_perceptual_distances.py
python src/phase1/compute_semantic_distances.py
python src/phase1/correlation_diagnostic.py      ← STOP. Review correlation.
python src/phase2/classify_quadrants.py
python src/phase2/sample_pairs.py
python src/phase3/run_episodes.py
python src/phase4/regression_analysis.py
python src/phase4/dissociation_analysis.py
python src/phase4/generate_figures.py
python src/phase4/generate_report.py
```

## Relation to Paper

- Rule A → new subsection §3.X with quadrant boxplot as main-text figure.
- Rule B → paragraph in Limitations; schedule cross-encoder replication.
- Rule C → revise claim from "perceptual geometry" to "concept coherence".

All three outcomes are publishable. A is strongest. B is honest. C is
essential correction protecting the program from a faulty premise.
