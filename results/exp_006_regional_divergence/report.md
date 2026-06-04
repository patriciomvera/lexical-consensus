# Experiment 006 — Regional Divergence

**Date:** 2026-06-04
**Encoder:** `facebook/dinov2-small` (frozen)
**Architecture:** centroid only — no adapter (per exp_005b decision)
**Dataset:** CIFAR-10 — frog (slithy) / horse (mimsy) / ship (vorpal), 80 images per category
**Agents:** 6, partitioned into two clusters of three (cluster_1 = agents 00/01/02, cluster_2 = agents 03/04/05)
**Image distribution:** cluster_1 sees indices [0:25] per category, cluster_2 sees [25:50], shared held-out [50:80]
**Rounds:** 25 per regime

---

## Framing

> exp_006 tests whether systematically different input distributions across
> isolated agent sub-populations produce regional lexical divergence —
> the computational analogue of Latin vulgar splitting into the Romance
> languages. Given exp_005's finding that DINOv2 geometric consistency
> dominates inter-agent alignment, this experiment asks whether
> input-distribution differences and network isolation are sufficient
> to overcome that shared perceptual anchor.

---

## Verdict

**Shared DINOv2 perception is a strong universal anchor that resists
linguistic divergence even under maximal isolation, high noise, and
disjoint input distributions.** The most isolated regime (latin)
develops a measurably elevated between-cluster centroid distance
relative to the fully-connected baseline (latin 0.078 vs vervet 0.005),
but that elevation is *static, not growing* — the per-label
divergence_growth_rate is **negative** in every label of every regime.
Across all three regimes, held-out cross-cluster label agreement stays
above 0.97, and per-cluster held-out accuracy stays above 0.97.

In the language of the spec's critical control question — "does shared
perception prevent divergence even under isolation?" — the answer is
yes, with a caveat: shared perception prevents *runaway* divergence;
it does not prevent a moderate, stable inter-cluster offset arising
from cluster-specific seeding. The two clusters keep using the same
Carroll labels for the same images. Same word, same meaning.

---

## Three-regime comparison (mean final between-cluster distance)

| Regime | σ | Bridge | Mean between-cluster d | Mean growth rate | Held-out btw_agreement |
|---|---|---|---|---|---|
| vervet | 0.00 | every round   | **0.0051** | −0.000009 | 0.989 |
| raven  | 0.05 | every 5th     | 0.0056     | −0.002044 | 0.989 |
| latin  | 0.10 | never (after seeding) | **0.0785** | −0.001471 | 0.978 |

Three observations follow immediately:

1. **Vervet collapses inter-cluster distance to noise levels** (~0.005)
   within the first round and stays there. With every agent voting in
   every round, the two cluster-collective centroids are functionally
   the same vector.
2. **Raven's bridge mechanism works exactly as designed.** Between-cluster
   distance grows to ≈ 0.07 over rounds 1–4, then **crashes to ≈ 0.008
   at round 5** (the first bridge round) and stabilizes near vervet
   levels for the remainder. Subsequent bridges (rounds 10, 15, 20, 25)
   show small additional re-alignments.
3. **Latin retains a moderate inter-cluster offset (~0.08) but does not
   diverge further.** The growth rate is negative — latin's
   between-cluster distance actually shrinks from ≈ 0.12 at round 1 to
   ≈ 0.078 at round 25. Isolation + σ=0.1 is not enough to overcome
   the DINOv2 anchor.

---

## Per-label divergence at round 25

| Regime | slithy | mimsy | vorpal |
|---|---|---|---|
| vervet | 0.0055 | 0.0036 | 0.0061 |
| raven  | 0.0070 | 0.0038 | 0.0061 |
| latin  | **0.1111** | 0.0585 | 0.0658 |

Within-cluster pairwise distances (mean of two clusters) at round 25:

| Regime | slithy | mimsy | vorpal |
|---|---|---|---|
| vervet | 0.0131 | 0.0111 | 0.0132 |
| raven  | 0.0152 | 0.0119 | 0.0132 |
| latin  | 0.1348 | 0.0566 | 0.0734 |

Note that for latin, the *within-cluster* distances are roughly the
same magnitude as the *between-cluster* distances. With σ=0.1, agents
within a cluster end up nearly as far apart from each other as the two
clusters' collectives are from each other — noise dominates internal
agreement. This makes the "lexical divergence" metric (between −
within) close to zero or slightly negative in latin, despite the
between-cluster distance being substantial in absolute terms.

---

## Answers to the spec's six questions

### 1. Does between-cluster distance grow under isolation (latin)?

**No.** Latin's mean between-cluster distance starts at ≈ 0.12 (round 1)
and *decreases* to ≈ 0.078 by round 25. The least-squares slope of
between-cluster distance vs round is **negative** for every Carroll
label in latin (slithy −0.0012, mimsy −0.0007, vorpal −0.0025). The
initial elevation comes from disjoint seeding, not from accumulating
drift. Over rounds, both clusters' centroids are slowly pulled back
toward the underlying DINOv2 attractors as more interaction images get
accepted — the shared encoder is the strong attractor.

### 2. Does the weak bridge (raven) pull divergence back periodically?

**Yes, and the effect is sharp.** Between-cluster distance climbs to
≈ 0.07 over rounds 1–4 (pre-bridge), then drops by an order of magnitude
to ≈ 0.008 at round 5 (the first bridge round). Subsequent bridges
(10, 15, 20, 25) keep the metric pinned near vervet levels. The pullback
is visible in both `divergence_curves.png` (sharp step down at round 5)
and `divergence_comparison.png` (raven's orange line drops to overlap
vervet's green line at round 5). Raven looks like latin for 4 rounds
and like vervet for the next 21.

### 3. Does full connectivity (vervet) prevent divergence entirely?

**Yes.** Vervet's mean between-cluster distance is ≈ 0.005 from round 2
onward — the floor set by the small remaining inter-agent variation in
which interaction images each agent accumulates over the deterministic
rounds. With σ=0 and every round acting as a bridge, the two cluster-
collective centroids are functionally identical vectors.

### 4. CRITICAL: Does shared DINOv2 perception prevent divergence even under isolation?

**Yes, with one important caveat.**

The strong reading: the three-regime ordering of mean final between-
cluster distance (latin 0.078 > raven 0.006 ≈ vervet 0.005) does show a
measurable signal from isolation + noise + input-distribution differences.
The success criterion `latin − vervet ≥ 0.005` is satisfied
(0.073 ≫ 0.005). Topology and noise demonstrably matter.

The weaker reading — which is the one that matters for the linguistic
analogy: latin does NOT diverge over time. It starts elevated from
seeding and then slowly converges back toward vervet. Even maximally
isolated noisy clusters trained on disjoint image subsets of the same
underlying CIFAR-10 categories end up pointing at the same regions of
DINOv2 embedding space for the same Carroll labels.

The PCA scatter (`cluster_centroid_pca.png`) makes this visceral. In
all three regimes the three labels occupy three well-separated regions
of the centroid PCA, and within each region cluster_1 (blue) and
cluster_2 (red) markers are tightly overlapping. There is no
"different region of label space" for the two clusters in any regime,
including latin.

### 5. Do isolated clusters maintain held-out accuracy?

**Yes.** Per-cluster held-out accuracy at round 25:

| Regime | cluster_1 | cluster_2 | shared 6-agent consensus |
|---|---|---|---|
| vervet | 0.989 | 0.978 | 0.989 |
| raven  | 0.989 | 0.978 | 0.989 |
| latin  | 0.989 | 0.967 | 0.989 |

Latin loses one additional held-out image relative to vervet/raven in
cluster_2 (0.967 vs 0.978), consistent with a single sample of noise
making the difference under σ=0.10 — well within single-sample
fluctuation on a 90-image set. The shared 6-agent consensus accuracy
is identical across regimes at 0.989. Divergence, to the extent it
exists, does not degrade either cluster's ability to label the shared
test set.

### 6. Same word, different meaning?

**No, not in any meaningful sense.** Cross-cluster label agreement on
the 90-image shared held-out at round 25:

| Regime | between_cluster_label_agreement |
|---|---|
| vervet | 0.989 (89/90) |
| raven  | 0.989 (89/90) |
| latin  | 0.978 (88/90) |

Latin's two isolated clusters end with 88 of 90 held-out images receiving
the same Carroll label from both clusters' majority votes. The single
"contested" image (relative to vervet) is the only place where the two
clusters mean different things by the same word. By any reasonable
linguistic standard, they speak the same language.

---

## Why the Latin regime did not produce Latin → Romance

The intuition behind the experiment is that geographically isolated
populations of Latin speakers diverged over centuries into French,
Spanish, Italian, etc. — same lexical ancestry, different meanings and
forms after long isolation. exp_006 sets up a computational analogue,
and the analogue fails. Why?

The likely answer: the categorical attractors in DINOv2's embedding
space are **too strong relative to the noise budget**. CIFAR-10's
frog / horse / ship classes occupy three well-separated regions of
the 384-dimensional manifold. Different image instances of the same
class (cluster 1's frogs versus cluster 2's frogs) live very close to
each other in that manifold; they are not different enough to anchor
different categorical concepts even when the clusters never
communicate.

Real Latin → Romance involved more than spatial isolation. It involved
local phonological shifts, exposure to substrate languages, contact
with non-Latin neighbors (Frankish, Visigothic, Arabic), and changing
referents (new objects, new social structures). All of these are
ways the *input distribution* could diverge in ways that *DINOv2-on-
CIFAR-10* simply cannot capture. The result here suggests that
modeling regional divergence in artificial agents requires either:

- a perceptual encoder whose categorical structure is *not* uniform
  across populations (each cluster's perception itself shaped by local
  input), or
- visual sub-types within a category that are far enough apart in
  embedding space to anchor distinct sub-concepts (e.g. one cluster
  only seeing tree frogs, the other only seeing bullfrogs), or
- a much longer time horizon and a substantially larger noise budget
  than σ = 0.1.

That this experiment fails to produce divergence is itself the
significant finding: it operationalizes the conditions under which a
shared perceptual substrate prevents linguistic drift, which is
exactly the boundary the theoretical framework (Sapir-Whorf, weak
form) wants to characterize.

---

## Success criteria

| Criterion | vervet | raven | latin |
|---|---|---|---|
| within_cluster_accuracy ≥ 0.90 | PASS | PASS | PASS |
| shared_held_out_accuracy ≥ 0.80 | PASS (0.989) | PASS (0.989) | PASS (0.989) |
| divergence_signal (latin − vervet ≥ 0.005) | — | — | PASS (0.073) |

The third row is on the spec's terms: the difference is well above
threshold. But as the report emphasizes, the *direction of change*
(negative growth rate everywhere) is the more substantive finding.

---

## Artifacts

- `config.yaml` — full experiment configuration
- `round_log_by_regime.csv` — per-round summary, three regimes × 25 rounds
- `within_cluster_metrics.csv` — per (regime, cluster, label, round)
- `between_cluster_metrics.csv` — per (regime, label, round)
- `ledger_events.csv` — assignment-level log (for Neo4j upload)
- `centroid_vectors.npz` / `cluster_centroid_vectors.npz` — per-agent and per-cluster centroids
- `divergence_summary.json` — single-file summary with success-criteria evaluation
- `query_divergence_trajectory.csv` — flattened trajectory for downstream tooling
- `divergence_curves.png` — per-regime, per-label between-cluster distance over rounds
- `divergence_comparison.png` — three-regime mean between-cluster distance
- `cluster_centroid_pca.png` — 2D PCA of cluster-collective centroids by regime

---

## Closing

> If divergence emerges, it demonstrates that meaning can fragment
> across isolated populations even when perception is shared — the
> conditions under which one language becomes many. If divergence
> does not emerge, it demonstrates that shared perceptual grounding
> acts as a universal anchor resisting linguistic drift — a finding
> equally significant for understanding when languages stay unified.

exp_006 lands on the second alternative. The three experiments
exp_005, exp_005b, and exp_006 together tell a consistent story:
shared DINOv2 perception dominates inter-agent alignment, and neither
gradient-based adapter training (exp_005b) nor input-distribution
differences + isolation + noise (exp_006) was sufficient to break
it. The theoretical framework's prediction that shared visual
perception could anchor lexical consensus across distributed agent
populations is empirically supported across multiple perturbations.

The next test of the framework would change the perceptual substrate
itself — either by giving different agent sub-populations different
encoders, or by training a shared encoder on regionally-distinct
visual input — to ask whether perceptual divergence is the
prerequisite for lexical divergence. That is the experiment exp_007
(or its successor) should pursue.
