# exp_007 — Concept Carving (prototype vs. episodic dissociation)

**Status:** scaffolded (skeletons only — no logic implemented yet).

## Why this experiment exists

The Round-1 reviewer raised two criticisms. One is cheap (scale: 3 classes, no
seeds, no CIs, no baselines). The other can kill the paper:

> the learner is a centroid classifier over an already-separable DINOv2 space;
> "learning" reduces to averaging features of categories the encoder already
> knew → it assigns a pointer to a pre-existing cluster rather than acquiring a
> concept.

Scaling to 10/20-way on CIFAR only answers the cheap criticism with more
statistical power over the same conceptual void. This experiment attacks the
real one:

> **Can an agent acquire a concept whose extension does NOT coincide with a
> native DINOv2 cluster?**

If yes, acquisition happened (the lexical layer carved a new region). If no, we
map the true boundary of grounding over frozen perception. Either outcome is a
strong, honest result — and it is the same testbed as the prototype (centroid)
vs. episodic (exemplar) distinction in the theoretical framework.

Organizing question: **which kind of lexical mechanism works under which kind of
concept?**

## Honesty constraint (write into the paper)

This experiment does **not** rescue the grand Vera et al. thesis (acquisition >
imitation as a fundamental intelligence measure). It defeats the "pointer, not
acquisition" objection and maps the prototype/episodic boundary. State this in
Limitations.

## Hypotheses (falsification criteria in the full plan)

- **H1 — dissociation.** Centroid accuracy decreases monotonically as native
  separability NS(c) drops; exemplar/k-NN hold. Supported iff the prototype-
  failure gap `Acc(exemplar) − Acc(centroid)` has a 95% paired-bootstrap CI
  excluding 0 and Wilcoxon p < 0.05 on cross-cutting / disjunctive concepts.
- **H2 — non-triviality.** On native concepts, no learner dissociation
  (centroid ≈ exemplar ≈ baselines).
- **H3 — genuine carving.** At least one learner acquires low-NS / low-NMI
  concepts above chance (CI excludes chance).
- **H4 — no identity cheating** (cross-cutting only). A learner that succeeds on
  an attribute concept does so with low NMI(prediction, object_category).

## Design — a 2-D grid

| | centroid | kcentroid | exemplar_knn | logreg | linsvm | random |
|---|---|---|---|---|---|---|
| **NATIVE** (007a) | H2 baseline / Round-1 replication with full stats |
| **DISJUNCTIVE** (007b) | primary payoff — central figure + paired test (H1, H3) |
| **CROSS_CUTTING** (007c) | needs synthetic shapes×colors; H4 detector |

Axis B spans the prototype → episodic continuum: `centroid` (pure prototype) →
`kcentroid` (multi-anchor bridge) → `exemplar_knn` (pure episodic), plus the
reviewer's `logreg` / `linsvm` linear baselines and a `random` floor.

## Execution order (plan §11)

1. Learner interface (`src/learners/base.py`) — **scaffolded.**
2. **007a NATIVE** — replicate Round 1, full stats, all learners (H2).
3. **007b DISJUNCTIVE** — ≥5 unions spanning an NS range, central figure +
   paired exemplar-vs-centroid test. No new data.
4. Falsification controls (random-label, permuted-binding, OOV) on 007b.
5. **007c CROSS_CUTTING** — generate `data/synthetic_shapes/`, run the substrate
   diagnostic first, then attribute concepts + H4 detector.

## Where the code lives

```
src/learners/            # the learner zoo (base + 6 learners)         [scaffolded]
src/eval/                # concepts, descriptors, episodes, metrics,
                         # stats, falsification                        [scaffolded]
data/synthetic_shapes/   # shapes×colors generator (exp_007c)          [scaffolded]
experiments/exp_007_concept_carving/
    _shared.py           # grid config + embedding cache               [scaffolded]
    run_exp007.py        # condition loop -> ledger                    [scaffolded]
    analyze_exp007.py    # aggregates, CIs, tables, figures            [scaffolded]
results/exp_007/         # ledger.jsonl, aggregate.csv, paired_tests.csv,
                         # descriptors.csv, cheating_detector.csv, figures
```

## Deliverables (plan §12)

- `ledger.jsonl` — per-episode events.
- `aggregate.csv` — mean/std/CI95/n_episodes/chance per (concept_type, learner, N, K).
- `paired_tests.csv` — exemplar vs centroid (and baselines): diff, CI95, Wilcoxon p, effect size.
- `descriptors.csv` — NS_silhouette, compactness, centroid_lands_outside, NMI_native per concept.
- `cheating_detector.csv` — NMI(prediction, object_category) per cross-cutting condition.
- **Figure 1**: Accuracy vs NS(c), one curve per learner (the central image).
- **Figure 2**: prototype-failure-gap boxplots by concept type.
- **Figure 3**: accuracy vs NMI(prediction, object_category) scatter (H4).

## Compute (plan §10)

All learning (centroid / kcentroid / k-NN / logreg / SVM) runs on CPU over
precomputed 384-d embeddings — fast, fits 16 GB, no GPU. The only heavy step is
extracting DINOv2 embeddings once (CIFAR subset for 007a/b; ~3.2k synthetic
images for 007c), which is cached to `.cache/embeddings_exp007/`.

> **Note on numbering.** This redefines the `exp_007` slot from the old
> "Test 2, full paper" placeholder. The CLAUDE.md roadmap should be updated to
> point exp_007 at concept carving once implementation begins.
