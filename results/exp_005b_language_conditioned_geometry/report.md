# Experiment 005b — Language-Conditioned Geometry

**Date:** 2026-06-04
**Encoder:** `facebook/dinov2-small` (frozen)
**Adapter:** `LexicalAdapter` (`W = I + ε·N(0,1)`, per-agent, trainable)
**Dataset:** CIFAR-10 — frog (slithy) / horse (mimsy) / ship (vorpal)
**Agents:** 3, with disjoint 5-seed slices per category (same splits as exp_003 / exp_005)
**Rounds:** 25 per condition
**Hyperparameters:** lr = 1e-3 (Adam), λ = 0.5, β = 0.1, margin = 0.2, ε = 0.01

---

## Framing

> exp_005 established that consensus feedback does not substantially reshape
> representational geometry under a frozen centroid architecture — because
> the architecture had no mechanism for representational rewriting.
> exp_005b introduces a mutable lexical adapter per agent and tests whether
> consensus feedback can act as an attractor over agent-specific categorical
> geometry when such a mechanism exists.
> This constitutes a proper operational test of weak Sapir-Whorf in this system.

---

## Verdict

**H1 not supported.**

Consensus-driven adapter training (Condition B) does **not** reduce
inter-agent projected centroid distance more than the frozen control
(Condition A), and does **not** reduce it more than the random-feedback
attribution control (Condition C). The random control in fact produces
the lowest final projected distance of any condition.

This is a clean null result and a more falsifying one than exp_005 — the
architectural change worked exactly as designed (the adapter is mutable;
gradients flow; W drifts as expected), but the consensus signal does not
act as the attractor over geometry the weak Sapir-Whorf claim would
predict.

---

## Final-round results

| Condition | mean final d (projected) | vs. frozen | vs. random | mean ‖W − W₀‖_F | held-out acc | label coverage |
|---|---|---|---|---|---|---|
| A — Frozen      | **0.0557** | +0.0000 | +0.0164 | 0.000 | 1.000 | 3 |
| B — Consensus   | 0.0581     | +0.0024 | +0.0188 | 4.068 | 0.983 | 3 |
| C — Random      | **0.0393** | −0.0164 | +0.0000 | 4.086 | 0.983 | 3 |

Lower projected distance = more inter-agent alignment.
Negative "vs. X" = better aligned than the comparison condition.

### Per-label projected distances (round-1 → round-25)

| Label  | A round-1 → final          | B round-1 → final          | C round-1 → final          |
|---|---|---|---|
| slithy | 0.0636 → **0.0603**        | 0.0766 → **0.0327**        | 0.0904 → **0.0282**        |
| mimsy  | 0.0515 → **0.0546**        | 0.1641 → **0.0546**        | 0.1583 → **0.0428**        |
| vorpal | 0.0552 → **0.0524**        | 0.1537 → **0.0870**        | 0.1336 → **0.0469**        |

Per-label outcome at the final round (lowest distance wins):
- **slithy:** C < B < A
- **mimsy:** C < A ≈ B
- **vorpal:** C < A < B

C is the consistent winner across all three labels. B sometimes matches A and
sometimes does worse than A; never the best.

### Raw DINOv2 distances barely move

In raw (un-projected) DINOv2 space, the mean pairwise distance across all
labels and rounds stays within ≈ 0.001 of its initial value in every
condition. The encoder is genuinely frozen — anything we see in projected
space is the work of W, not a quiet contamination of the perceptual layer.

---

## Answers to the spec's six questions

### 1. Does consensus-trained adapter (B) reduce projected inter-agent distance more than frozen adapter (A)?

**No.** Mean final projected distance is 0.0581 in B versus 0.0557 in A — B is
slightly *worse* aligned than A at the end of 25 rounds. The literal
`alignment_gain = d_round1 − d_final` rollup is positive for B (≈ +0.073)
purely because B's first Adam step inflates `d_round1` from ≈ 0.06 to
≈ 0.13; subsequent rounds recover from that perturbation but never settle
below A. On the spec's H1 criterion — *projected distance decreases more
than Condition A* — B does not meet the criterion when measured against
the pre-training baseline.

### 2. Does consensus-trained adapter (B) reduce projected inter-agent distance more than random-feedback (C)?

**No, the reverse.** C's final mean projected distance (0.0393) is *lower*
than B's (0.0581). On every Carroll label, C ends with smaller pairwise
distance than B. The `consensus_specific_gain` field is negative for all
three labels of B (slithy −0.018, mimsy −0.006, vorpal −0.020). Random
gradient signal aligns agents more than consensus signal does, given equal
number of training steps and identical centroid-accumulation rules.

### 3. Does raw DINOv2 distance remain flat in all conditions?

**Yes, effectively.** Per-condition mean raw distances drift by less than
0.005 across 25 rounds in every condition. The small movement that does
occur is driven by which interaction images each agent accepts (different
W → different consensus → marginally different accepted sets); the
encoder weights themselves are never touched. The raw row in
`raw_vs_projected_alignment.png` is essentially three overlapping flat
lines.

### 4. Does held-out accuracy remain ≥ 0.90 after adapter training?

**Yes — comfortably.** Final held-out consensus accuracy is 1.000 in A and
0.983 in both B and C. That 0.017 difference corresponds to a single
held-out image (out of 60) being assigned UNCERTAIN under the trained
adapters where the frozen adapter committed to a label. All three
conditions exceed the 0.90 success threshold; the task is preserved.

### 5. Which adapter shows the most drift from identity?

The three trained adapters drift to a near-identical steady-state of
`‖W − W₀‖_F ≈ 4.07` (B) and `4.09` (C). At ε = 0.01 the initialization
itself sits at ≈ 3.84 from identity, so the trained adapters add only
≈ 0.2–0.3 units of additional drift before equilibrium between the β·‖W − I‖²
regularizer and the loss-driven push takes hold. Per-agent variation is
small (range ≈ 0.025) in both B and C. The frozen adapter is flat at 0 by
construction.

### 6. Is there a geometry-accuracy tradeoff?

**Yes, but small.** B and C both lose one held-out image relative to A
(0.983 vs 1.000) while moving projected geometry substantially (gain in
the spec-literal sense ≈ +0.07–0.09). The tradeoff exists but is in the
range of single-sample noise on a 60-image held-out set; not enough to
prefer A on accuracy grounds alone, not enough to preserve B's H1 claim
either.

---

## Why does random feedback align agents more than consensus feedback?

The most interesting finding is not the null H1 — it is the
*direction* of the null: C beats B. Two mechanisms plausibly explain it.

1. **The β·‖W − I‖² regularizer is the dominant signal.** Each Adam step
   on B and C is composed of (i) a per-image cosine-distance pull toward
   some centroid and (ii) a global penalty pulling W back toward the
   identity. The regularizer's gradient magnitude (≈ 2β·‖W − I‖_F
   ≈ 0.77 at initialization) dwarfs the cosine-distance gradient on a
   single image. W's steady-state is set mostly by the regularizer; the
   per-image signal only nudges the *direction* of small departures from I.

2. **Random feedback averages agent-specific noise; consensus preserves
   it.** Under C, every agent gets a different random target per image,
   but on expectation the targets sample uniformly over the three label
   centroids — and after enough rounds, every agent's W is pulled in the
   same average direction, which is "toward the mean of all centroids."
   Under B, each agent's gradient is keyed to *its own* accepted-image
   set and its own label assignments. Disjoint seeds → different
   accepted sets → different per-agent pull directions. The structure of
   consensus actually *preserves* agent-specific geometric bias rather
   than washing it out.

That second point is the substantive finding: consensus signal is
*more individualized* than random signal, and individualization is the
opposite of inter-agent alignment. Random feedback wins on alignment
precisely because it has no structure to distinguish agents.

---

## What this means for the weak Sapir-Whorf claim in this system

Weak Sapir-Whorf, operationally re-stated for exp_005b, was:

> *Shared lexical feedback reshapes the agent-specific categorical
> geometry in the mutable projection space.*

The experiment shows: shared lexical feedback *can* reshape W (the
adapter mutates, drift > 0, the projected centroid distances move),
but the reshaping does not pull agents toward shared geometry any
more than untrained or randomly-trained adapters do. Under the current
hyperparameters and DINOv2's already strongly-consistent embedding
geometry, the consensus signal does not function as an attractor.

This is not a refutation of weak Sapir-Whorf as a linguistic
hypothesis. It is a refutation of one specific computational
operationalization of it — the one in which a per-agent linear
adapter, regularized toward identity, is trained on per-image cosine
losses against majority-vote consensus. Other operationalizations
(non-linear adapters, smaller β, removed regularizer, larger initial
ε for bigger starting gap, multi-step inner loops per round) might
behave differently. Those are ablations for a follow-up, not
re-runs of exp_005b.

The structural finding from exp_005 still stands and is reinforced
here: DINOv2's geometric consistency dominates inter-agent alignment.
Both the seed-driven baseline (exp_005) and now the adapter-trained
experiments (exp_005b) tell the same story from different angles.

---

## Artifacts

- `config.yaml` — full experimental configuration
- `round_log_by_condition.csv` — every per-round metric, three conditions
- `adapter_drift.csv` — ‖W − I‖_F and ‖W − W₀‖_F per (condition, agent, round)
- `inter_agent_distances_projected.csv` — pairwise distances in projected space
- `inter_agent_distances_raw.csv` — same in raw DINOv2 space
- `held_out_accuracy_by_round.csv` — held-out task metric over rounds
- `accepted_images.csv` — final accepted-image snapshot per (condition, agent, label)
- `ledger_events.csv` — assignment-level log (for Neo4j upload)
- `alignment_summary.json` — single-file summary with success-criteria evaluation
- `adapter_weights_initial.npz` / `adapter_weights_final.npz` — W matrices
- `projected_centroid_vectors.npz` / `raw_centroid_vectors.npz` — centroids per round
- `projected_alignment_curve.png` — main result
- `raw_vs_projected_alignment.png` — raw flatness confirmation
- `adapter_drift.png` — W trajectory
- `accuracy_by_condition.png` — task preservation

---

## Looking forward

> exp_006 (regional divergence) will use the architecture established
> in exp_005b — with or without the adapter depending on results — to
> test whether shared labels break down under conditions analogous to
> geographic isolation, the computational equivalent of Latin vulgar
> splitting into Romance languages.

Given exp_005b's null, the adapter is unlikely to be the load-bearing
component of exp_006. The substantive question for exp_006 is whether
*differing input distributions* across regional sub-populations break
the DINOv2-driven geometric consensus that exp_003 / exp_005 / exp_005b
have repeatedly shown to be the dominant signal. If divergence
emerges in exp_006, it will come from the data-distribution difference,
not from the adapter mechanism — exp_005b has now ruled the adapter
out as a reliable alignment driver under the regularization regime
explored here.
