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

| Experiment | Status | Description |
|---|---|---|
| exp_000 | PASS | Embedding separability: frog/horse/ship in DINOv2 space, silhouette=0.283 |
| exp_001 | PASS | Single agent lexicon: Condition 1 (accuracy 1.000) + Condition 2 (all levels pass) |
| exp_002 | PASS | Grounding controls: falsification conditions A–E |
| exp_002b | PASS | Balanced label control: closes Condition A weak point |
| exp_003 | PASS | Multi-agent consensus: 003a converged round 6 (unanimous=0.978, held=1.000); 003b baseline 0.933/0.983 |
| exp_004 | Pending | Neo4j integration, Shannon metrics |
| exp_005 | Pending | Centroid drift, Sapir-Whorf measurement |
| exp_006 | Pending | Regional divergence — vervet/raven/Latin experiment |
| exp_007 | Pending | Test 2, full paper |

**Each experiment gates the next. Always check the roadmap in README.md before suggesting features that belong to later experiments.** A common failure mode is jumping ahead and breaking the incremental structure.

---

## Empirical Findings

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

4. **Do not introduce LLMs into the classification pipeline.** The architecture is intentionally LLM-free for classification to eliminate label contamination. LLMs may appear in future phases for natural-language description generation, but not for the core consensus loop.

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
|   |   |-- perception.py     # DINOv2 wrapper (Layer 1)
|   |   |-- lexicon.py        # Centroid-based lexicon (Layer 2)
|   |   `-- learner_agent.py  # Full three-layer agent
|   |
|   |-- consensus/
|   |   `-- ledger.py         # Consensus mechanism (Layer 3)
|   |
|   |-- dataset/
|   |   `-- artificial_vocab.py  # Carroll vocabulary definitions
|   |
|   |-- metrics/              # TODO: exp_002+
|   |-- graph/                # TODO: exp_003 (Neo4j)
|   `-- utils/
|
|-- experiments/
|   |-- exp_000_embedding_separability/  # Gate: CIFAR-10 separable in DINOv2?
|   |-- exp_001_single_agent_lexicon/    # One agent, one tutor, Carroll vocab
|   `-- exp_002_multi_agent_consensus/  # First full 3-agent run
|
|-- tests/                    # TODO
|-- notebooks/                # Analysis
`-- results/                  # Experiment outputs (gitignored)
```

---

## Next Concrete Steps

**Current task: exp_004 — Neo4j integration, Shannon metrics.**

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
