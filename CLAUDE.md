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

| Phase | Status | Description |
|---|---|---|
| 1 | In progress | Foundation: structure, single agent, basic metrics |
| 2 | Pending | Multi-agent consensus, first full experiment |
| 3 | Pending | Neo4j integration, Shannon metrics |
| 4 | Pending | Centroid drift, Sapir-Whorf measurement |
| 5 | Pending | Regional divergence — vervet/raven/Latin experiment |
| 6 | Pending | Test 2, full paper |

**Always check the roadmap in README.md before suggesting features that belong to later phases.** A common failure mode is jumping ahead and breaking the incremental structure.

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
|   |-- metrics/              # TODO: Phase 2-3
|   |-- graph/                # TODO: Phase 3 (Neo4j)
|   `-- utils/
|
|-- experiments/
|   |-- exp_001_baseline/     # First multi-agent run
|   `-- exp_002_control_conditions/  # Carroll vs ad-hoc vs numeric
|
|-- tests/                    # TODO
|-- notebooks/                # Analysis
`-- results/                  # Experiment outputs (gitignored)
```

---

## Next Concrete Steps (Phase 1 Completion)

1. **Diagnostic script:** verify CIFAR-10 categories form separable clusters in DINOv2 embedding space. If they don't, the experiment cannot work.
2. **Update run.py:** the current `experiments/exp_001_baseline/run.py` uses an older LLM-based design and must be rewritten for the three-layer DINOv2 architecture.
3. **First test run:** 3 agents, 3 Carroll labels, 25 rounds, synthetic dataset (3 classes from CIFAR-10).
4. **Metric: Convergence round per label.**

Each step should be a separate commit.

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
