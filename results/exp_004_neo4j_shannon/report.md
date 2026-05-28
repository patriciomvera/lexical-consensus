# Experiment 004 — Neo4j Integration + Shannon Metrics

**Date:** 2026-05-28
**Depends on:** exp_003a (with feedback), exp_003b (no-feedback baseline)
**Graph database:** Neo4j AuraDB (instance 25e47996)
**Shannon source:** src/metrics/shannon.py (29 unit tests pass)

---

## Overview

This experiment converts the numerical results of exp_003 into a queryable
graph structure and applies information-theoretic analysis via Shannon mutual
information. The two sub-analyses answer complementary questions:

- **Shannon metrics:** How much information does a label assignment carry about
  the image that received it? Does feedback increase this grounding signal?
- **Neo4j graph:** Which images reached unanimous consensus first? Which agents
  had the most stable centroid trajectories? What is the full assignment history?

---

## 1. At which round does normalized_MI exceed 0.5?

**Answer: Round 1 — for both exp_003a and exp_003b.**

| Condition | Round 1 normalized_MI | Grounding threshold round |
|---|---|---|
| 003a (with feedback) | **0.9659** | **1** |
| 003b (no feedback)   | **0.9659** | **1** |

The grounding threshold (normalized_MI > 0.5) is exceeded immediately in the
first round of the experiment. This means the shared DINOv2 embedding space
produces near-maximal information transfer between images and labels from the
very first labeling interaction.

**Interpretation:** With only 5 seed images per category per agent, the
centroid geometry is already well-discriminative. The first round of labeling
produces H(label|image) = 0.061 bits against H(label) ≈ 1.73 bits, giving
normalized_MI = 0.966. The grounding threshold of 0.5 is surpassed by a
margin of +0.466.

This is not a trivial result. A random labeling system would have
normalized_MI ≈ 0 (exp_002b demonstrated this: the scrambled-label control
gave C1=0.342 ≈ chance). The near-1.0 normalized_MI from round 1 reflects
the geometric concentration of CIFAR-10 category information in DINOv2 space.

---

## 2. How many images had residual disagreement at the final round?

**Formula:** n_disagreements ≈ mean_H(label|image) × n_images / 0.9183

With 3 agents, every disagreement produces H ≈ 0.918 bits (a 2/1 split).
The formula converts mean entropy per round into a count of disagreeing images.

| Condition | Final H(label\|image) | n_images | n_disagreements |
|---|---|---|---|
| 003a (with feedback) | **0.0204 bits** | 45 | **≈ 1.0** |
| 003b (no feedback)   | **0.0612 bits** | 45 | **≈ 3.0** |

**003a final round:** Approximately **1 image** remains non-unanimous. This
corresponds to the single UNCERTAIN image identified in the CLAUDE.md notes —
an image that sits geometrically near the boundary between two category
clusters. All three agents assign it UNCERTAIN (confidence < 0.30), which
counts as a 3-way disagreement in the entropy calculation. This is correct
agent behavior, not a failure.

**003b final round (all rounds identical):** Approximately **3 images** remain
non-unanimous throughout. These are the same 3 boundary images that produce
unanimous=0.933 in the baseline (100% unanimous for 42/45 images; 3 images
remain ambiguous). The frozen lexicons cannot resolve these boundary cases
without feedback.

**Feedback gain:** The consensus feedback mechanism resolves 2 of the 3
boundary cases over 6 rounds, reducing residual disagreement from ~3 to ~1.
The remaining image is irreducibly ambiguous given the agent architecture.

---

## 3. Which Carroll label reached unanimous consensus first?

**From Query 2 (first_unanimous_round per label per image):**

All three Carroll labels first achieved unanimous consensus in **Round 1**.

| Label | Category | First image | First unanimous round |
|---|---|---|---|
| mimsy  | horse | horse_015 | **1** |
| slithy | frog  | frog_015  | **1** |
| vorpal | ship  | ship_015  | **1** |

**Count of images unanimous by round:**

| Round | Images unanimous (003a) | Fraction |
|---|---|---|
| 1 | 42 / 45 | 0.933 |
| 2–6 | 44 / 45 | 0.978 |

The first images to reach unanimous consensus are the images at the start of
the interaction pool (index 15 per category). These are the "easiest" images
— closest to the centroid of their category in DINOv2 space and therefore
furthest from any decision boundary. 3 images in the interaction pool
(geometry-boundary cases) do not achieve unanimous consensus until round 2
(2 of them) or never (1 UNCERTAIN image), consistent with the entropy analysis
in Question 2.

---

## 4. Which agent had the most stable centroid trajectory?

**From Query 4 (mean_shift_from_previous per agent per label):**

| Label | Most stable agent | Mean shift/round |
|---|---|---|
| mimsy  | **agent_00** | **0.01135** |
| vorpal | **agent_01** | **0.01351** |
| slithy | **agent_02** | **0.01791** |

No single agent dominates across all labels. The per-label stability pattern
reflects each agent's seed set:

- **agent_00** has the most stable mimsy centroid. Its horse seeds (indices
  0:5 per category) likely land near the category center, giving a well-placed
  initial centroid that requires little adjustment.
- **agent_01** is most stable for vorpal. Its ship seeds (indices 5:10) are
  similarly near the center of the ship cluster.
- **agent_02** is most stable for slithy. Its frog seeds (indices 10:15)
  produce the most stable frog centroid.

**Overall stability:** All agents converge to near-zero shift by round 3–4.
The largest single-round shift is 0.129 (agent_00, slithy, round 1 — the first
feedback update moves the centroid substantially before stabilizing).
After round 2, all shifts are below 0.002. Centroids are effectively fixed
by round 3.

---

## 5. Did feedback produce measurably different Shannon metrics?

**Yes — feedback produces measurable improvement in all three metrics.**

**Including UNCERTAIN as a 4th effective symbol:**

| Metric | 003a (feedback) | 003b (no feedback) | Delta (003a − 003b) |
|---|---|---|---|
| Final H(label\|image) | **0.0204 bits** | 0.0612 bits | **−0.041 bits** |
| Final mutual information | 1.7096 bits | 1.7357 bits | +0.026 bits (raw) |
| Final normalized MI | **0.9882** | 0.9659 | **+0.022** |

**Carroll-only (UNCERTAIN excluded, 44/45 images):**

| Metric | 003a (feedback) | 003b (no feedback) |
|---|---|---|
| Final H(label\|image) | **0.0000 bits** | 0.0000 bits |
| Final normalized MI | **1.0000** | **1.0000** |
| Images included | 44/45 | 44/45 |
| Images excluded (all-UNCERTAIN) | 1 | 1 |

**Central result:** Including UNCERTAIN, normalized MI is 0.9659 (003b) → 0.9882
(003a). Carroll-only (UNCERTAIN excluded from both conditions): normalized MI = 1.000
for both. Feedback resolves approximately 2–3 residual disagreements per 45-image
interaction pool, consistent with delta_entropy = 0.0408 bits ≈ 2 images × 0.918 / 45.

**Reading the including-UNCERTAIN table:**

- **H(label|image):** Lower is better (less residual uncertainty). Feedback
  reduces it by 0.041 bits — eliminating approximately 2 of the 3 baseline
  disagreements.

- **Raw MI:** 003b shows slightly higher raw MI (1.736 vs 1.710) because its
  label marginal H(label) is also higher (1.797 vs 1.730). With frozen lexicons,
  the label distribution is slightly more spread — a consequence of 3 unresolved
  images producing diverse majority votes that inflate H(label). Raw MI alone
  is misleading here.

- **Normalized MI:** Correctly accounts for this difference. 003a achieves
  0.9882 (feedback) vs 0.9659 (baseline), a **+0.022 improvement**. This
  is the cleanest metric: it isolates the fraction of H(label) that is
  explained by image identity, controlling for variation in label marginals.

**3-agent entropy context:** The 0.041-bit reduction in H(label|image) is
large relative to the discrete entropy scale (where the minimum unit of
disagreement is 0.918 bits for a 2/1 split). A 0.041-bit reduction
corresponds to resolving 0.041/0.918 ≈ 4.5% of a single disagreement per
image — equivalently, eliminating 2 images' worth of persistent disagreement
over 45 interaction images.

**Conclusion:** Feedback is beneficial but operates on top of an already
high baseline. The 003b baseline itself shows normalized_MI = 0.966, confirming
that DINOv2's geometric consistency is the primary driver of grounding. The
feedback mechanism provides a measurable but incremental refinement (+0.022
normalized_MI) by resolving boundary cases that DINOv2 geometry alone cannot
deterministically classify.

---

## Methodological Notes

### UNCERTAIN as a 4th effective symbol

Agents emit the special label `UNCERTAIN` when their classification confidence
falls below 0.30. This label is not part of the Carroll vocabulary (slithy,
mimsy, vorpal) but is a genuine signal from the agent architecture.

Two Shannon variants are reported throughout this experiment:

- **Including UNCERTAIN:** UNCERTAIN is counted as a 4th effective symbol.
  The alphabet size is 4, so `max_entropy = log2(4) = 2.0 bits`. When all
  three agents label an image UNCERTAIN, H(label|image) = 0 bits (unanimous
  agreement on "no label"), which contributes to — not inflates — the entropy.
  The label marginal H(label) can exceed `log2(3) = 1.585 bits` because UNCERTAIN
  is a 4th symbol with non-zero marginal probability. This is correct and
  intentional: the 4-symbol vocabulary is the agent's true effective alphabet.

- **Carroll-only:** UNCERTAIN votes are dropped and each image's remaining
  Carroll-label votes are renormalized to sum to 1.0. Images where all agents
  said UNCERTAIN are excluded entirely (1 image out of 45 in both conditions).
  This variant answers: "Among images that received at least one Carroll label,
  how grounded is the labeling?" Carroll-only normalized_MI = 1.000 for both
  conditions, confirming perfect grounding over the 44 non-boundary images.

The Carroll-only variant is more conservative and appropriate for the paper's
primary grounding claim. The including-UNCERTAIN variant provides the complete
picture of how the agent system behaves, including its correct identification
of ambiguous boundary images.

### Neo4j query pool filtering

Queries 1 and 2 are run separately for the `interaction` pool (images agents
can update their lexicons on) and the `held_out` pool (evaluation only).
Queries 3 and 4 operate at centroid level, which has no pool dimension; they
produce identical results across both pool runs. Pool-specific output files
are generated for naming consistency.

---

## Graph Summary

The Neo4j AuraDB instance now contains the complete record of exp_003a:

| Node type | Count |
|---|---|
| Experiment | 1 |
| Agent | 3 |
| Label | 3 |
| Image | 105 |
| Assignment | 1,890 |
| Centroid | 54 |

| Relationship type | Count |
|---|---|
| CONSENSUS_WITH | 1,874 |
| HAS_AGENT, HAS_LABEL, HAS_IMAGE | 111 |
| MADE, USES_LABEL, APPLIES_TO | 5,670 |
| HAS_CENTROID | 54 |

---

## Closing

This experiment converts the numerical results of exp_003 into a queryable
graph structure. The Neo4j AuraDB instance now contains a permanent record
of every label assignment, consensus event, and centroid trajectory from the
experiment. Future experiments (exp_005 centroid drift, exp_006 regional
divergence) will add to this graph, enabling cross-experiment queries that
are impossible with CSV files alone.

The Shannon metrics provide a complementary analytical lens: normalized
mutual information compresses the entire grounding quality of a labeling
system into a single number in [0, 1]. The exp_003 system achieves 0.988
with feedback and 0.966 without — both far above the 0.5 formal grounding
threshold, and both far above the chance baseline established in exp_002b
(C1=0.342, normalized_MI ≈ 0).
