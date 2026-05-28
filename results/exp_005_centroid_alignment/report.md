# Experiment 005 — Centroid Alignment and the Sapir-Whorf Analogue

**Date:** 2026-05-28
**Depends on:** exp_003a (with feedback), exp_003b (no-feedback baseline)
**Source:** experiments/exp_005_centroid_alignment/
**Centroid reconstruction:** interaction-pool accepted images, cumulative per round

---

## Central Question

exp_003 established that agents reach lexical consensus (agreement on Carroll
labels per image). exp_005 asks a different question:

> **Does shared labeling cause agents to organize perceptual space more
> similarly over time?**

An agent can say "slithy" while its centroid occupies a different region of
DINOv2 space than another agent's slithy centroid. The formal framing tested:

> "We test whether shared labels act as attractors over agent-specific
> category centroids — an operational analogue of weak Sapir-Whorf: shared
> lexical feedback induces measurable convergence in category representations."

---

## Centroid Reconstruction Method

Full 384-dim centroid vectors are not stored in ledger_events.csv. They are
reconstructed from scratch:

For each (experiment_id, agent_id, label, round):
- Collect all interaction-pool images (`pool="interaction"`, `status="accepted"`)
  assigned to that label by that agent, cumulative up to that round.
- Compute the mean of their DINOv2 unit-normed embeddings.
- L2-normalize the result.

**Critical constraint:** Only interaction-pool images are used. Held-out images
never updated centroids during the experiment. Seed images (indices 0–14 per
category) are not in the ledger and are excluded by design — this measures
only the feedback-driven portion of each agent's centroid.

**003b (no-feedback baseline):** Lexicons are frozen after seeding; the same
images are accepted in every round. The cumulative set stabilizes at round 1
and produces identical centroid vectors across all 100 rounds. The sanity check
confirms: shift between round 1 and round 50 = 0.0 for all (agent, label).

---

## 1. Does inter-agent centroid distance decrease with feedback?

| Label | 003a Round 1 | 003a Final (R6) | 003b Final (R100) | Alignment gain |
|---|---|---|---|---|
| **slithy** | 0.00911 | 0.00420 | 0.00911 | **+0.00491** |
| **mimsy**  | 0.00000 | 0.00000 | 0.00000 | **0.00000** |
| **vorpal** | 0.00000 | 0.00000 | 0.00000 | **0.00000** |

**mimsy and vorpal (ceiling effect):** Inter-agent pairwise cosine distance is
exactly 0.0 from round 1 in both conditions. All three agents independently
build identical interaction-pool centroids for horse and ship: each correctly
classifies the same 14 interaction-pool images of each category from the very
first round. Since the image sets are identical, the means are identical, and
the cosine distance between centroids is 0.0. There is nothing for feedback to
improve — the representations are maximally aligned before feedback begins.

**slithy (frog):** Non-zero initial distance (0.00911 mean) because agent_01
initially has 14 accepted frog images while agents 00 and 02 have 15 and 14
different images respectively. Feedback causes agent_01's missing image to be
accepted at round 2, resolving the agent_00–agent_01 gap. The 003b distance
stays flat at 0.00911 across all 100 rounds — the frozen baseline never
resolves this gap.

---

## 2. Does inter-agent centroid distance remain flat without feedback?

**Yes — for slithy, the control behaves exactly as predicted.**

| Label | 003b Round 1 | 003b Round 100 | Change |
|---|---|---|---|
| slithy | 0.00911 | 0.00911 | 0.000 (flat) |
| mimsy  | 0.00000 | 0.00000 | 0.000 (floor) |
| vorpal | 0.00000 | 0.00000 | 0.000 (floor) |

The 003b baseline is flat for all labels. For slithy, this confirms that the
representational gap observed at round 1 is persistent without feedback: agents
whose seed-driven centroids classify slightly different subsets of the frog
interaction pool cannot resolve the disagreement without the consensus signal.

---

## 3. Which Carroll label shows the greatest alignment gain?

**slithy** is the only label with non-zero alignment gain (+0.00491). Mimsy
and vorpal are already at the representational floor (d=0.0) from round 1, so
they cannot gain further.

The AUC of the distance curve captures speed of alignment (total area under
the curve across rounds):

| Label | AUC 003a | AUC 003b | AUC reduction |
|---|---|---|---|
| slithy | 0.030 | 0.055 | **−0.025 (feedback faster)** |
| mimsy  | 0.000 | 0.000 | 0.000 |
| vorpal | 0.000 | 0.000 | 0.000 |

Feedback reduces the slithy AUC by 45%: 003a converges faster (the gap closes
at round 2) compared to 003b (which stays elevated throughout all 6 rounds of
comparison).

---

## 4. Which agent pair converges most?

For slithy (the only label with non-zero distances):

| Agent pair | Round 1 | Round 2 | Round 6 |
|---|---|---|---|
| agent_00 – agent_01 | 0.00585 | **0.00000** | 0.00000 |
| agent_00 – agent_02 | 0.00631 | 0.00631 | 0.00631 |
| agent_01 – agent_02 | 0.01518 | 0.00631 | 0.00631 |

**agent_00 – agent_01** converges completely (d=0) at round 2. This is because
agent_01's missing frog image (`frog_026`, the one that agent_01 initially
labels UNCERTAIN) reaches majority consensus in round 2 and is added to
agent_01's centroid — making it identical to agent_00's set of 15 images.

**agent_00 – agent_02** and **agent_01 – agent_02** stabilize at d=0.00631
but do not converge further. Agent_02 is missing a *different* frog image than
the one agent_01 resolves. That image never achieves majority acceptance (agent_02
consistently labels it UNCERTAIN, and it is not in the majority vote set). The
gap persists at a fixed 0.00631 throughout both conditions.

For mimsy and vorpal: all pairs are at d=0.0 from round 1 — no differentiation
is possible.

---

## 5. Is alignment_gain positive overall?

**Formally: H0 is not rejected.**

| Metric | Value |
|---|---|
| Mean alignment_gain | **+0.00164** |
| Labels with gain > 0 | 1/3 (slithy only) |
| H1 supported (all > 0) | **False** |

The stopping rule specified: H1 requires alignment_gain > 0 for ALL three
labels. Only slithy satisfies this. Mimsy and vorpal both show gain = 0 because
the no-feedback baseline already achieves perfect representational alignment
(d=0.0 from round 1). H1 is not supported.

**Interpretation:** This null result is itself informative. The feedback
mechanism cannot improve what is already perfect. The DINOv2 geometry is
consistent enough that agents converge independently to identical category
representations for 2/3 of the Carroll vocabulary, before any feedback signal
is applied. The Sapir-Whorf prediction — that shared labels act as attractors
over representations — is not supported here because there is no representational
gap for labels to close.

The difference-in-differences for slithy (+0.00491) confirms that whatever
convergence occurs for slithy is attributable to feedback, not shared perception
alone. But this affects only one label and one agent pair.

---

## 6. Does representational convergence occur before or after lexical agreement?

| Label | Round lexical consensus | Round repr. convergence | Order |
|---|---|---|---|
| slithy | **−1** (never) | **2** | representational_first |
| mimsy  | **−1** (never) | **1** | representational_first |
| vorpal | **4**          | **1** | representational_first |

**Representational convergence precedes lexical agreement for all labels.**

- **mimsy, vorpal:** Centroids are perfectly aligned from round 1 (d=0 always).
  Representational convergence is immediate. Lexical consensus requires that all
  15 images in the category receive unanimous correct labels — mimsy never
  achieves this (due to the boundary UNCERTAIN image), and vorpal achieves it
  only at round 4, three rounds after the geometry was already fully aligned.

- **slithy:** Representational convergence (50% reduction in mean pairwise
  distance) occurs at round 2. Lexical consensus (all 15 frog images unanimously
  labeled "slithy") is never reached — one boundary image consistently receives
  UNCERTAIN from one or more agents throughout all 6 rounds.

**Theoretical interpretation:** The categorical geometry stabilizes before the
labeling system reaches full consistency. The agents' perceptual representations
converge faster than their label assignments because the DINOv2 embedding space
provides an implicit shared prior. Labels track geometry; they do not lead it.
This is the inverse of the weak Sapir-Whorf prediction: in this system,
geometry drives labels, not the other way around.

---

## Formal Hypothesis Test Summary

**H₀:** Consensus feedback does not reduce inter-agent centroid distance
relative to the no-feedback baseline.

**H₁:** Consensus feedback reduces inter-agent centroid distance relative to
the no-feedback baseline.

**Result:** H₀ is not rejected. Two of three Carroll labels (mimsy, vorpal)
show zero alignment gain because both conditions achieve perfect representational
alignment from round 1. The third label (slithy) shows a small positive gain
(+0.00491) attributable to feedback resolving one agent pair's gap. The overall
mean alignment gain is +0.00164, positive but driven by a single label.

The null result has a structural explanation: the feedback mechanism can only
improve representational alignment when there is a representational gap to close.
DINOv2's geometric consistency leaves no gap for 2/3 of the vocabulary.

---

## Neo4j Graph Additions

| Entity | Count added |
|---|---|
| exp_003b Centroid nodes (rounds 1–6) | 54 |
| exp_003b Agent nodes | 3 |
| PROXIMITY_TO edges (003a) | 54 |
| PROXIMITY_TO edges (003b) | 54 |
| Centroid properties updated | 108 |

New Centroid node properties: `mean_pairwise_distance`, `distance_to_agent_00`,
`distance_to_agent_01`, `distance_to_agent_02`, `vector_hash`, `vector_path`.

Queries 5–7 confirm Neo4j returns consistent results with the Python-computed
values. Query 7 (Neo4j) alignment_gain: slithy = +0.00409, mimsy = 0.000,
vorpal = 0.000. The small difference from the Python value (+0.00491) is
expected: Query 7 averages cosine_distance across all rounds rather than
comparing only final-round values.

---

## Closing

exp_005 tests whether the lexical consensus mechanism in exp_003 produces
representational alignment beyond surface label agreement. A positive
alignment_gain would constitute an operational analogue of weak Sapir-Whorf:
shared labels acting as attractors over agent-specific category representations.

**The data does not support this prediction.** For 2/3 labels (mimsy, vorpal),
agents' category centroids are already perfectly aligned (d=0) from the first
interaction, independent of feedback. For slithy, feedback provides marginal
improvement (+0.00491) by resolving one specific inter-agent gap that frozen
lexicons cannot close. Representational convergence systematically precedes
lexical agreement across all labels — geometry organizes first, labels follow.

The primary driver of shared perceptual representations in this system is
DINOv2's frozen embedding geometry, not the consensus feedback mechanism.
Feedback's contribution is lexical (improving unanimous agreement from 0.933
to 0.978, as established in exp_003) rather than representational.

Future experiment exp_006 will test whether this alignment breaks down under
regional divergence conditions — the computational analogue of Latin vulgar
splitting into Romance languages.

---

## Note on exp_005b (not implemented)

A matched-update control would verify that convergence is driven by
consensus-structured feedback rather than by additional exposure to images.
In exp_005b, agents receive the same number of updates as in 003a but with
randomized feedback (no real consensus signal). If 003a converges more than
exp_005b, the result is attributable to consensus structure, not update count.

In the present case, the representational null result makes exp_005b less
urgent: since feedback does not produce measurable representational convergence
for 2/3 labels, there is no effect to attribute. exp_005b would be most
informative if the slithy result (+0.00491) is of theoretical interest — it
would test whether that small gain is driven by consensus structure or simply
by additional image exposure.
