# Experiment Summary — Lexical Consensus
*One-page overview for evaluation and comparison*

---

## The Question

Can a network of AI agents build a shared vocabulary for visual concepts from scratch — using only instruction and consensus — without any pre-existing labels?

---

## The Setup

- **Visual encoder:** DINOv2 (frozen, no labels, no categories — pure visual geometry)
- **Agents:** N independent instances, each starting with an empty vocabulary
- **Tutor:** A human who assigns invented labels to seed images using only verbal description
- **Vocabulary:** 10 invented words from Lewis Carroll's *Jabberwocky* (vorpal, slithy, mimsy, borogove, tulgey, frumious, manxome, galumphing, uffish, jubjub)
- **Dataset:** CIFAR-10 images with all original labels removed

---

## How It Works

1. Tutor shows agents a few images and names them: *"This is a vorpal. A vorpal is round and smooth."*
2. Each agent encodes the image with DINOv2 and stores the visual embedding under that label
3. Agents independently label new images by finding the nearest centroid in their learned vocabulary
4. A consensus mechanism compares labels across agents — labels that reach 70% agreement stabilize, labels below 20% are discarded
5. Agents receive feedback on their own labels only — not what others said
6. The cycle repeats for 25 rounds

---

## What Success Looks Like

**Condition 1 — Naming:** agents consistently assign the same Carroll label to new images of the same visual category (not seen during instruction)

**Condition 2 — Inverse grounding:** given only the string "vorpal", an agent can identify which image in a set corresponds to that concept — the label activates the internal visual representation

Both conditions must pass for the experiment to validate Test 1 of Vera et al.

---

## What Is Being Measured

| Metric | Description |
|---|---|
| Convergence round | When does the first stable label emerge? |
| Full convergence rate | What % of categories reach consensus? |
| Generalization accuracy | Does the label transfer to unseen images? |
| Centroid drift | Does the concept's internal representation shift over time? |
| Shannon entropy | Does label entropy reduce as expected across rounds? |
| Inter-agent centroid distance | Do agents align their internal representations, not just their labels? |

---

## Why This Design

**DINOv2 instead of LLMs for classification:** eliminates the core contamination problem — a language model asked to "forget" its labels cannot, but DINOv2 never had labels to begin with. The perceptual layer is genuinely clean.

**Carroll vocabulary:** phonotactically valid, semantically empty, public domain. Any pre-training contamination is linguistic, not visual, and is quantified through a three-condition control experiment (Carroll / ad-hoc invented / numeric labels).

**Consensus inspired by distributed ledgers:** labels survive by majority agreement across independent agents, not by any single authority. This mirrors how words stabilize in natural language communities.

---

## What This Would Prove

If the experiment succeeds, it provides empirical evidence that:

1. Language acquisition — not behavioral imitation — is a measurable property of AI systems
2. Shared meaning can emerge from perception and social consensus alone, without pre-defined ontologies
3. The Turing Test measures the wrong thing — a system can pass it without this capacity, and fail it while having it

This directly validates Test 1 of the Vera et al. theoretical framework and challenges the Turing Test as the primary benchmark for machine intelligence.

---

## Theoretical Connections

| Thinker | Contribution | Connection to this experiment |
|---|---|---|
| **Shannon (1948)** | Information is entropy reduction | Consensus is error correction; grounding is mutual information between perception and label |
| **Carroll (1872)** | Semantically empty but syntactically valid words | The experimental vocabulary — meaning must be built, not inherited |
| **Steels (1996–2015)** | Lexical consensus emerges in robotic agents | Same dynamics, different substrate — LLMs instead of custom robots, DINOv2 instead of cameras |
| **Vera et al. (20xx)** | Language acquisition as AI benchmark | This experiment is the empirical validation of that theoretical claim |

---

## Complexity and Timeline

| Phase | Content | Effort |
|---|---|---|
| 1 | Single agent learns Carroll labels from tutor instruction | Low |
| 2 | Multi-agent consensus, first full experiment | Medium |
| 3 | Neo4j integration, Shannon metrics, visualization | Medium |
| 4 | Centroid drift, Sapir-Whorf measurement | Medium |
| 5 | Regional divergence — vervet / raven / Latin experiment | High |
| 6 | Test 2, full paper | High |

Phase 1 and 2 are sufficient for a first publishable result.

---

## Hardware Requirements

- **Local development:** any modern laptop (CPU only, DINOv2-small)
- **Experiment execution:** Modal or Vast.ai (GPU cloud, low cost)
- **Storage:** Neo4j for graph analysis (free community edition)

---

*Full documentation: README.md, PROTOCOL.md*
*Repository: github.com/[username]/lexical-consensus*
