# CLAUDE.md — Project Context for Claude Code

This file provides context for any AI assistant (Claude Code, Cursor, etc.) working on this repository. Read this before suggesting code changes.

---

## What This Project Is

**Lexical Consensus** is an empirical experiment that tests whether artificial agents can construct a shared vocabulary for visual concepts from scratch — through instruction and distributed consensus alone, without pre-existing labels.

It is the empirical validation of **Test 1** from the theoretical paper:

> Vera, P., Moya, P., & Barraza, L. (20xx). *Rethinking the Evaluating Framework for Natural Language Understanding in AI Systems: Language Acquisition as a Core for Future Metrics.*

The paper proposes language acquisition (not behavioral imitation) as a more fundamental measure of machine intelligence than the Turing Test. This repository builds the experiment that tests that claim.

---

## Repository Maintainer

**Patricio Vera** — only active developer on this implementation. Pedro Moya and Lisa Barraza are co-authors of the theoretical paper, not contributors to this codebase.

Patricio works at a slow, deliberate pace — this project is built incrementally over time. Do not assume continuous engagement; design every module to be self-contained and well-documented so context can be picked up after gaps.

---

## Architecture in One Page

Three-layer agent design:

```
LAYER 1 — Perception (DINOv2, frozen)
  image -> 384-dim vector
  Identical across agents. No labels. No category bias.
  
LAYER 2 — Lexicon (learned during experiment)
  vector <-> artificial label (bidirectional)
  Starts empty. Built through tutor instruction and consensus.
  Implementation: centroid-based nearest-neighbor.

LAYER 3 — Consensus (in src/consensus/)
  Labels compete across agents.
  Stable labels: >= 70% agent agreement.
  Discarded labels: < 20% agreement.
```

**Critical design decision:** the agent does NOT use LLMs for image classification. Classification is purely geometric in DINOv2 embedding space. This eliminates the contamination problem — there is no "previous knowledge" to suppress because the perceptual layer never had labels.

**Artificial vocabulary:** Lewis Carroll's invented words from *Jabberwocky* (vorpal, slithy, mimsy, borogove, tulgey, frumious, manxome, galumphing, uffish, jubjub). Chosen because they are phonotactically valid, semantically empty, and symbolically resonant with the experiment's themes.

**LexicalAdapter (exp_005b onward):** an agent-specific trainable linear
projection `z_i = normalize(W_i @ x)` over the frozen DINOv2 embedding `x`.
Lives in `src/agents/lexical_adapter.py`. Introduced in exp_005b to give
consensus feedback a mechanism for actual representational rewriting — the
gradient-free architecture used in exp_001 through exp_005 only let centroids
change through which images were accepted, not through how agents transform
perceptual space. The adapter is initialized as `I + epsilon * N(0, 1)` so
agents start near identical, and trained via Adam on cosine-distance losses
driven by consensus ACCEPT / REJECT events. DINOv2 itself is still frozen —
only `W_i` is learned, and it is per-agent. Earlier experiments do not use
the adapter and remain bit-for-bit reproducible.

---

## Two Conditions That Must Pass

The experiment is only successful if BOTH conditions hold:

**Condition 1 — Naming (perception -> label):**
An agent sees a new image and consistently assigns the correct Carroll label, matching what other agents assign.

**Condition 2 — Inverse grounding (label -> perception):**
Given only the string "vorpal", an agent can identify which image in a candidate set corresponds to that concept. This tests whether the agent built a real internal representation or just memorized image-label pairs.

These two conditions are operationalized in `src/agents/learner_agent.py` as `label_image()` and `test_inverse_grounding()` respectively.

---

## Theoretical Framework

The experiment connects several historical threads. If you're suggesting changes, be aware these connections are deliberate:

- **Shannon (1948):** consensus = error correction; grounding = mutual information between perception and label
- **Carroll (1872):** vocabulary is semantically empty by design — meaning must be constructed, not inherited
- **Steels (1996-2015):** lexical consensus emerges in robotic agents — same dynamics, different substrate
- **Sapir-Whorf:** centroid drift over time measures categorical deformation under linguistic pressure

Full details in README.md sections "Theoretical Extensions".

---

## Roadmap — Where We Are

Note on numbering: exp_002_grounding_controls and exp_002b were added after
the original plan to strengthen the falsification argument. This shifted
the original exp_002 (multi-agent consensus) to exp_003, and all subsequent
numbers forward by one. The table below reflects the actual execution order.

| Experiment | Status | Description |
|---|---|---|
| exp_000 | PASS | Embedding separability: frog/horse/ship in DINOv2 space, silhouette=0.283 |
| exp_001a/b/c | PASS | Single-agent naming curve: 5/10/15 seeds → accuracy 1.000 at 10 seeds |
| exp_001b_condition2 | PASS | Inverse grounding: easy/medium/hard — all levels pass (hard=1.000) |
| exp_002_grounding_controls | PASS | Falsification: conditions A–E all degrade as predicted |
| exp_002b_balanced_label_control | PASS | Closes condition A — population mean C1=0.342 ≈ chance |
| exp_003a_consensus_feedback | PASS | Multi-agent consensus with feedback: converged round 6, held=1.000 |
| exp_003b_no_feedback_baseline | PASS | Control: unanimous=0.933, held=0.983 — DINOv2 alignment precedes feedback |
| exp_004_neo4j_shannon | PASS | Neo4j integration + Shannon metrics — grounding threshold reached round 1 in both 003a (NMI=0.988) and 003b (NMI=0.966) |
| exp_005_centroid_alignment | PASS (null) | Centroid alignment — H1 not supported. DINOv2 geometry, not feedback, drives convergence. Seed-divergence floor is permanent under gradient-free architecture. |
| exp_005b_language_conditioned_geometry | PASS (null) | LexicalAdapter (first gradient component), 3 conditions × 25 rounds. H1 not supported: B (consensus) final projected distance 0.058 vs A (frozen) 0.056 vs C (random) 0.039. Random training aligns agents MORE than consensus, because consensus signal is more individualized per agent. |
| exp_006_regional_divergence | Pending | Vervet/raven/Latin experiment (was exp_005) |
| exp_007_test2_paper | Pending | Test 2, full paper (was exp_006) |

**Each experiment gates the next. Always check the roadmap in README.md before suggesting features that belong to later experiments.** A common failure mode is jumping ahead and breaking the incremental structure.

---

## Empirical Findings

### exp_005b — Language-conditioned geometry (first gradient experiment)

Three conditions × 3 agents × 25 rounds. The LexicalAdapter is mutable but
regularized toward identity (β = 0.1) and trained per accepted/rejected image
with Adam (lr = 1e-3, λ = 0.5, margin = 0.2).

**Final mean projected inter-agent centroid distance:**
- A frozen:    0.0557 (held-out acc 1.000, drift 0.000)
- B consensus: 0.0581 (held-out acc 0.983, drift 4.068)
- C random:    0.0393 (held-out acc 0.983, drift 4.086)

**H1 not supported.** Consensus-driven W training does NOT reduce projected
distance more than frozen baseline (B is slightly worse than A on the
final-state metric), and does NOT reduce it more than random feedback
(C beats B on every label).

**Most informative finding:** C beats B. Random feedback aligns agents MORE
than consensus feedback. Two contributing mechanisms:
1. The β·‖W − I‖² regularizer dominates the gradient signal per step (its
   gradient norm ≈ 0.77 at init vs the per-image cosine loss gradient on a
   single image, which is much smaller).
2. Consensus signal is structured per-agent (each agent's gradient depends
   on its own accepted set and label assignments); random signal is uniform
   over labels and averages out agent-specific bias. Individualization is
   the opposite of alignment.

**Methodological note:** The H1 criteria in `alignment_summary.json` are
evaluated on FINAL projected distance rather than the spec-literal
`alignment_gain = d_round1 − d_final`. The spec-literal metric overstates
convergence because Adam's first step inflates d_round1 in B and C; the
"gain" is then mostly recovery from that perturbation. Both quantities are
reported; the report explains the choice.

**Raw DINOv2 distances** stay within ≈ 0.005 across all conditions, all
rounds — confirms the encoder is genuinely frozen and the observed
projected-space movement is the work of W.

**Implication for exp_006:** the adapter is unlikely to be the load-bearing
component. The substantive question for exp_006 is whether differing
*input distributions* across regional sub-populations can break the
DINOv2-driven geometric consensus that exp_003/005/005b have repeatedly
shown to dominate inter-agent alignment.

### exp_003 — Multi-agent consensus results

**003a (with feedback):** 3 agents with disjoint 5-seed training sets.
Majority agreement = 1.000 from round 1. Unanimous agreement rose from
0.933 (round 1) to 0.978 (rounds 2–6) and held. Held-out consensus
accuracy = 1.000 from round 1. Convergence criterion met at round 6.

**003b (no-feedback baseline):** Identical setup, lexicons frozen after
seeding. Majority = 1.000, unanimous = 0.933, held-out = 0.983. Flat
across all 100 rounds — confirmed deterministic.

**Feedback gains (003a vs 003b):**
- Unanimous agreement: +0.045 (0.978 vs 0.933)
- Held-out accuracy: +0.017 (1.000 vs 0.983)
- Mean entropy: −0.041 bits (0.020 vs 0.061)

**Key structural finding:** The baseline (003b) itself shows excellent
agreement (majority=1.000, held=0.983) with NO feedback. This confirms
that DINOv2's embedding space is geometrically consistent enough that
independently trained agents naturally converge on the same Carroll label
assignments from disjoint 5-seed sets. The feedback mechanism then
provides measurable refinement (+0.017 accuracy, +0.045 unanimity).

**UNCERTAIN images:** 1 interaction-pool image is persistently non-unanimous
because all 3 agents assign it UNCERTAIN (low margin → confidence < 0.30).
This is correct agent behavior for a genuinely ambiguous image (near the
geometric boundary between two category clusters). It is not a failure.

**Stopping criterion note:** The unanimity threshold was relaxed from 1.0 to
0.95 to account for permanently UNCERTAIN boundary images. This is
methodologically correct: requiring 1.0 conflates ambiguity with failure.

### exp_002b — A1 sensitivity finding

The A1 "balanced" centroid construction (round-robin with 4:3 per-category imbalance)
produced C1=0.717, not chance-level as expected. Root cause: with N_SEEDS=10 and 3 labels,
the round-robin produces a diagonal imbalance where each label gets one extra seed from
a different category. In DINOv2's 384-dimensional space, a single additional seed is
enough to shift the centroid measurably toward the favored category.

**Implication:** The category signal in DINOv2 space is so concentrated that even a
4:3 bias creates detectable discriminability. For a clean random label control, use
N_SEEDS divisible by 3, or report population-level mean from N ≥ 30 scrambles.

This is a positive finding: the grounding mechanism extracts signal efficiently.

### exp_001 — Condition 2 asymmetry (OOV vs. trained distractors)

EASY level (OOV distractors: airplane, deer) scored **0.950** — 3 failures across 60 trials.
MEDIUM and HARD (same-category trained distractors) scored **1.000** — zero failures.

**Interpretation:** The agent discriminates better *within* trained category space than against untrained distractors. OOV categories occupy unexpected regions of the embedding manifold; the agent's centroid geometry does not model them and therefore cannot sharply reject them. This is not a weakness of grounding — it is evidence of proper grounding: the agent only knows what it was taught.

This asymmetry is expected under genuine representational learning and would not appear under a memorization strategy. Document in the paper as a positive structural finding.

---

## Coding Principles

1. **Every module must be self-documenting.** Patricio may not touch this code for weeks. Comments explain *why*, not just *what*.

2. **No silent failures.** Log everything. Errors that pass silently are worse than crashes.

3. **Reproducibility is non-negotiable.** Every experiment run logs: random seed, model versions, full configuration, all interactions. See VVUQ section in PROTOCOL.md.

4. **Do not introduce LLMs into the classification pipeline.** The architecture is intentionally LLM-free for classification to eliminate label contamination. LLMs may appear in future phases for natural-language description generation, but not for the core consensus loop. The `LexicalAdapter` (exp_005b onward) is the only gradient-trained component in the project, and even it is a single linear layer per agent over the frozen DINOv2 output — not a deep network and not an LLM.

5. **Carroll vocabulary is fixed.** Do not propose replacing vorpal/slithy/mimsy with "cleaner" alternatives. The contamination concern is documented as a known limitation with a three-condition control experiment in PROTOCOL.md section 8.

6. **Statistical rigor over speed.** Confidence intervals, bootstrapping, multiple seeds. A fast result without error bars is worthless for this project.

---

## Hardware Reality

Patricio's local machine: Windows HP laptop, i7-1165G7, 16GB RAM, no dedicated GPU.

- DINOv2-small CAN run locally on CPU but slowly
- Real experiments will run on Modal (cloud GPU)
- Do not propose architectures that require local GPU
- All code must work on CPU for development, then scale up

---

## File Map

```
lexical-consensus/
|-- README.md                 # Full project narrative + theoretical extensions
|-- PROTOCOL.md               # Formal experiment protocol (citable)
|-- CLAUDE.md                 # This file
|-- requirements.txt          # Python dependencies
|-- setup.ps1                 # Windows directory setup
|
|-- docs/
|   `-- experiment_summary.md # One-page summary for external review
|
|-- src/
|   |-- agents/
|   |   |-- base_agent.py     # Abstract agent interface
|   |   |-- perception.py     # DINOv2 wrapper (Layer 1, frozen)
|   |   |-- lexicon.py        # Centroid-based lexicon (Layer 2)
|   |   |-- lexical_adapter.py # Per-agent trainable projection (exp_005b+)
|   |   `-- learner_agent.py  # Full three-layer agent
|   |
|   |-- consensus/
|   |   `-- ledger.py         # Consensus mechanism (Layer 3)
|   |
|   |-- dataset/
|   |   `-- artificial_vocab.py  # Carroll vocabulary definitions
|   |
|   |-- metrics/              # exp_004 (Shannon metrics)
|   |-- graph/                # exp_004 (Neo4j)
|   `-- utils/
|
|-- experiments/
|   |-- exp_000_embedding_separability/        # Gate: CIFAR-10 separable in DINOv2?
|   |-- exp_001_single_agent_lexicon/          # Naming acquisition curve + inverse grounding
|   |-- exp_002_grounding_controls/            # Falsification conditions A–E
|   |-- exp_002b_balanced_label_control/       # Closes condition A
|   |-- exp_002_multi_agent_consensus/         # Old placeholder (README only, no code)
|   |-- exp_003_multi_agent_consensus/         # 003a (feedback) + 003b (baseline) + compare.py
|   |-- exp_004_neo4j_shannon/                 # Shannon metrics + Neo4j replay
|   |-- exp_005_centroid_alignment/            # Gradient-free Sapir-Whorf test (null)
|   `-- exp_005b_language_conditioned_geometry/ # First gradient experiment — 3 conditions
|
|-- tests/
|   |-- agents/test_lexical_adapter.py
|   `-- metrics/test_shannon.py
|-- notebooks/                # Analysis
`-- results/                  # Experiment outputs (gitignored)
```

---

## Next Concrete Steps

**Current task: exp_004_neo4j_shannon** (was exp_003 in the original plan — numbering shifted when grounding controls were added).

exp_003 is complete (both 003a and 003b). The multi-agent consensus mechanism
works. Key results:

- 003a: converged at round 6. Unanimous=0.978, held-out accuracy=1.000.
- 003b: unanimous=0.933, held-out=0.983 (stable — feedback adds +0.045/+0.017).
- DINOv2's geometric consistency is so strong that baseline agreement (003b)
  is already 0.983 without any feedback. This is a positive structural finding.
- The consensus feedback mechanism provides measurable refinement on top of
  an already-functional shared vocabulary.

The two-condition architecture (003a vs 003b) is the correct design for
isolating the feedback effect. Results are clean and interpretable for the paper.

---

## What NOT to Do

- Do not refactor for performance before correctness is established
- Do not add dependencies not in `requirements.txt` without justification
- Do not propose deep learning training of any component — DINOv2 is frozen by design
- Do not add LLM calls inside the classification loop
- Do not skip writing tests for the consensus mechanism — it is the hardest component to verify
- Do not push results to GitHub — only commit code and documentation

---

*If anything in this document conflicts with what Patricio says in chat, ask him to clarify rather than assuming. This document captures intent at a point in time; chat captures current intent.*
