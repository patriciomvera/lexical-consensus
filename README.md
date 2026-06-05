# Lexical Consensus — Emergent Language Acquisition in Multi-Agent LLM Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-blue.svg)]()
[![Paper: Vera et al. 20xx](https://img.shields.io/badge/Paper-Vera%20et%20al.-green.svg)](docs/paper_reference.md)

> *We are not testing whether agents know the word.
> We are testing whether agents can make a word become knowable.*

---

## What This Is

This repository contains the empirical implementation of **Test 1** from the framework proposed in:

> Vera, P., Moya, P., & Barraza, L. (20xx). *Rethinking the Evaluating Framework for Natural Language Understanding in AI Systems: Language Acquisition as a Core for Future Metrics.* Journal of Experimental & Theoretical Artificial Intelligence.

That paper argues that **language acquisition** — not imitation — is the right proxy for measuring machine intelligence. This project operationalizes that argument with a reproducible experiment.

---

## The Core Question

**Can a network of LLM agents converge on a shared, stable lexicon for novel visual concepts through verbal instruction alone — without access to pre-existing labels?**

This is Test 1 from the theoretical framework: an AI must acquire vocabulary for its environment through direct verbal instruction, mirroring early-stage language acquisition in a human child (~3 years old, per Moravec's paradox).

---

## The Experiment — Plain Language

1. We take a set of images and **strip all existing labels** from them.
2. A human instructor (the "tutor") assigns **artificial, invented labels** to a seed set of images using only verbal/textual description. No formal definitions. No ground truth leaked.
3. A network of independent LLM agents learns from the tutor's labels and from each other.
4. A **consensus mechanism** — inspired by distributed ledger principles — determines which labels stabilize across the network and which are rejected.
5. We measure: convergence speed, label stability, drift over time, and generalization to unseen instances.

The key constraint: agents are forbidden from using their pre-trained knowledge of image labels. All grounding must come from the instruction process itself.

---

## The Carroll Framework — Following the White Rabbit

> *"'When I use a word,' Humpty Dumpty said, in rather a scornful tone, 'it means just what I choose it to mean — neither more nor less.'*
> *'The question is,' said Alice, 'whether you can make words mean so many different things.'*
> *'The question is,' said Humpty Dumpty, 'which is to be master — that's all.'"*
> — Lewis Carroll, Through the Looking-Glass (1871)

Lewis Carroll — Charles Dodgson, mathematician and logician at Oxford — wrote "Jabberwocky" in 1872 as a poem full of invented words that mean nothing. And yet readers immediately build intuitions about them. "Vorpal" sounds sharp and precise. "Slithy" sounds smooth and sinuous. "Mimsy" sounds small and frail. Nobody explains this — readers infer it from sound, grammar, and context alone.

That is exactly the mechanism this experiment tests. The agents receive labels with no prior semantic content. All meaning must emerge from perception and consensus — exactly as Humpty Dumpty describes, except the agents must agree with each other, not just decree.

**The entire artificial vocabulary in this experiment consists of Carroll's invented words.**

This choice is not decorative. It is scientifically precise:

- The words are phonotactically valid English — agents can use them in sentences without syntactic friction
- They carry no semantic grounding in any language — no pre-training can contaminate their meaning
- They were constructed by a logician — their internal structure is deliberate, not random
- They are culturally resonant — making results communicable to general audiences without simplification

**The symbolic layers run deep:**

*Following the White Rabbit* — the tutor is the White Rabbit. Agents enter a world where the rules of language must be built from scratch, not inherited.

*Through the Looking-Glass* — Condition 2 of the experiment is literally a mirror: image → label, then label → image. The bidirectional grounding test is Carroll's looking-glass made computational.

*The Cheshire Cat* — the consensus that persists after individual agents are removed. The label that remains stable when the agent that coined it is gone.

*The vorpal blade went snicker-snack* — the label that cuts the continuous embedding space into discrete categories. Clean, decisive, irreversible.

**For general audiences, the experiment can be described in one sentence:**

> *Lewis Carroll invented "vorpal" in 1872 — a word with no meaning, only sound and feeling. One hundred and fifty years later, we asked artificial agents to agree on what "vorpal" means by looking at images together. This experiment is about how meaning is born.*

---

## Why This Matters

Current AI evaluation benchmarks measure **performance on known tasks**. They do not measure the capacity to *acquire new grounding* from minimal input — which is arguably the more fundamental intelligence property.

This experiment is a minimal but falsifiable test of that capacity. A positive result (stable lexical convergence from artificial labels) would provide empirical support for language acquisition as an AI evaluation axis. A negative result would be equally informative.

---

## Relation to Steels (1996–2015)

This experiment is directly inspired by Luc Steels' **Talking Heads experiment**, where physically embodied robotic agents developed shared lexicons through interaction. The key differences in our work:

| Steels (1996–2015) | This work (design) | Experimental result |
|---|---|---|
| Physical robots with cameras | LLM agents via API → DINOv2 + centroid | DINOv2 dominates alignment |
| Simple reinforcement rules | Distributed consensus mechanism | Consensus refines but does not drive |
| Small populations (2–10) | Scalable N-agent architecture | Tested with 3 and 6 agents |
| Custom agent architecture | Commodity LLMs → Frozen encoder + learnable lexicon | Three-layer architecture validated |
| Labels for real objects | Artificial labels (Carroll vocabulary) | Carroll labels acquired bidirectionally |

This shift from embodied robots to commodity LLMs tests whether the consensus dynamics Steels observed generalize to a fundamentally different substrate.

---

## Repository File Structure

```
lexical-consensus/
│
├── docs/                              # Theoretical background and protocol documentation
│   └── experiment_summary.md          # One-page summary for external review
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py              # Abstract agent interface
│   │   ├── perception.py              # DINOv2 wrapper (Layer 1, frozen)
│   │   ├── lexicon.py                 # Centroid-based lexicon (Layer 2)
│   │   ├── lexical_adapter.py         # Per-agent trainable projection (exp_005b+)
│   │   └── learner_agent.py           # Full three-layer agent
│   │
│   ├── consensus/
│   │   └── ledger.py                  # Consensus mechanism (Layer 3)
│   │
│   ├── dataset/
│   │   └── artificial_vocab.py        # Carroll vocabulary definitions
│   │
│   ├── graph/
│   │   ├── neo4j_logger.py            # Parallel Neo4j logging alongside ledger
│   │   └── replay_exp003.py           # exp_003a/b graph replay for Neo4j
│   │
│   ├── metrics/
│   │   └── shannon.py                 # Shannon entropy and NMI per round
│   │
│   └── utils/
│
├── experiments/
│   ├── exp_000_embedding_separability/  # DINOv2 substrate gate — PASS
│   ├── exp_001_single_agent_lexicon/    # Naming + inverse grounding — PASS
│   ├── exp_002_grounding_controls/      # Falsification conditions A–E — PASS
│   ├── exp_002b_balanced_label_control/ # Condition A closure — PASS
│   ├── exp_002_multi_agent_consensus/   # Placeholder (README only, superseded by exp_003)
│   ├── exp_003_multi_agent_consensus/   # 003a (feedback) + 003b (baseline) — PASS
│   ├── exp_004_neo4j_shannon/           # Shannon metrics + Neo4j replay — PASS
│   ├── exp_005_centroid_alignment/      # Passive Sapir-Whorf test — PASS (null)
│   ├── exp_005b_language_conditioned_geometry/  # Active Sapir-Whorf, LexicalAdapter — PASS (null)
│   └── exp_006_regional_divergence/    # Three biological regimes — PASS (anchor)
│
├── tests/
│   ├── agents/test_lexical_adapter.py
│   └── metrics/test_shannon.py
│
├── notebooks/                         # Analysis and visualization
├── scripts/                           # Utility scripts
│
├── results/                           # Experiment outputs (gitignored except structure)
│   ├── exp_000_embedding_separability/
│   ├── exp_001a_seed_05/
│   ├── exp_001b_seed_10/
│   ├── exp_001b_condition2_easy/
│   ├── exp_001b_condition2_medium/
│   ├── exp_001b_condition2_hard/
│   ├── exp_001c_seed_15/
│   ├── exp_002_grounding_controls/
│   ├── exp_002b_balanced_label_control/
│   ├── exp_003a_consensus_feedback/
│   ├── exp_003b_no_feedback_baseline/
│   ├── exp_003_comparison/
│   ├── exp_004_neo4j_shannon/
│   ├── exp_005_centroid_alignment/
│   ├── exp_005b_language_conditioned_geometry/
│   └── exp_006_regional_divergence/
│
├── README.md                          # This file
├── PROTOCOL.md                        # Formal experiment protocol (citable)
├── CLAUDE.md                          # AI assistant context and coding principles
├── SETUP.md                           # Environment setup instructions
├── requirements.txt                   # Python dependencies
└── .gitignore
```

---

## Completed Experiments

Results are stored locally in `results/`. Experiments are gated — each must pass before the next begins.

**Note on experiment numbering:** The original plan had exp_002 = multi-agent consensus and exp_003 = Neo4j. Between exp_001 and the multi-agent run, two additional falsification experiments (exp_002_grounding_controls and exp_002b_balanced_label_control) were added to strengthen the scientific argument. This shifted multi-agent consensus to exp_003 and all subsequent numbers forward by one. The roadmap below reflects this reconciled numbering.

---

### exp_000 — Embedding Separability
**Status:** PASS
**Result:** Silhouette score 0.2826 on frog/horse/ship. DINOv2-small provides viable perceptual substrate with moderate separation. Visual UMAP projection confirms cluster structure.
**Artifacts:** `results/exp_000_embedding_separability/`
**Key files:** `config.yaml`, `metrics.json`, `pca.png`, `umap.png`, `report.md`

---

### exp_001 — Single Agent Lexicon
**Status:** PASS (all sub-experiments)
**Sub-experiments:**
- exp_001a (5 seeds): C1 accuracy 0.983, 1 error (slithy→vorpal)
- exp_001b (10 seeds): C1 accuracy 1.000, C2 accuracy 1.000 (all levels)
- exp_001c (15 seeds): C1 accuracy 1.000
- exp_001b_condition2_easy: accuracy 0.950 (3 failures, OOV distractors)
- exp_001b_condition2_medium: accuracy 1.000
- exp_001b_condition2_hard: accuracy 1.000

**Result:** Single agent acquires Carroll labels from minimal instruction and generalizes bidirectionally. Behavioral saturation at 10 seeds; centroid drift continues through 15 seeds. OOV asymmetry (easy < hard) is a positive structural finding: the agent only knows what it was taught.
**Artifacts:** `results/exp_001a_seed_05/`, `results/exp_001b_seed_10/`, `results/exp_001c_seed_15/`, `results/exp_001b_condition2_{easy,medium,hard}/`

---

### exp_002 — Grounding Controls (Falsification)
**Status:** PASS (4/5 conditions clean; Condition A closed by exp_002b)
**Conditions:**
- A — Random labels: C1=0.550 (partial), C2=0.433 (collapsed)
- B — Random embeddings: C1=0.300, C2=0.250 (chance)
- C — Permuted embeddings: C1=0.000, C2=0.000 (complete collapse)
- D — OOV rejection: AUROC=0.964
- E — Harder categories (cat/dog/deer): C1=0.950, C2=0.867

**Result:** The grounding effect disappears when perception, binding, or label consistency is broken. Falsification argument complete.
**Artifacts:** `results/exp_002_grounding_controls/`, `results/exp_002b_balanced_label_control/`

---

### exp_003 — Multi-Agent Consensus
**Status:** PASS
**Sub-experiments:**
- exp_003a (with feedback): consensus accuracy 1.000 in 6 rounds; unanimous=0.978
- exp_003b (no feedback): baseline unanimous=0.933, held=0.983 (flat from round 1)

**Result:** Three agents with disjoint seed sets converge on shared Carroll vocabulary. Feedback provides measurable refinement (+0.017 consensus accuracy, +0.045 unanimity) on top of an already high perceptual baseline. DINOv2 geometric consistency is the primary driver — 003b achieves 0.983 held-out accuracy with zero feedback.
**Artifacts:** `results/exp_003a_consensus_feedback/`, `results/exp_003b_no_feedback_baseline/`, `results/exp_003_comparison/`

---

### exp_004 — Neo4j Integration + Shannon Metrics
**Status:** PASS
**Result:** Normalized MI reaches 0.966 (no feedback) to 0.988 (with feedback) from round 1. Grounding threshold (normalized_MI > 0.5) exceeded immediately. Shannon analysis confirms high information transfer between images and Carroll labels. Complete experiment graph stored in Neo4j AuraDB.
**Artifacts:** `results/exp_004_neo4j_shannon/`
**Key files:** `entropy_curve.png`, `shannon_metrics_003a.json`, `shannon_metrics_003b.json`, `query_1–4_label_assignments*.csv`

---

### exp_005 — Centroid Alignment (Sapir-Whorf Test, Passive)
**Status:** PASS — H1 not supported (informative null)
**Result:** Consensus feedback does not substantially reshape representational geometry under frozen centroid architecture. Alignment gain only for slithy (+0.004) and vorpal (+0.001). Geometry drives labels, not the reverse. Two reconstructions reported (interaction-only and full operational) to eliminate floor-effect artifacts.
**Artifacts:** `results/exp_005_centroid_alignment/`
**Key files:** `alignment_summary.json`, `centroid_alignment_curve.png`, `alignment_gain_by_label.png`

---

### exp_005b — Language-Conditioned Geometry (Sapir-Whorf Test, Active)
**Status:** PASS — H1 not supported (clean null)
**Sub-experiments:**
- Condition A (frozen adapter): final projected distance 0.056, held-out acc 1.000
- Condition B (consensus feedback): final projected distance 0.058, held-out acc 0.983
- Condition C (random feedback): final projected distance 0.039, held-out acc 0.983

**Result:** Even with a mutable LexicalAdapter (trainable W per agent), consensus feedback does not act as a representational attractor. Random feedback (Condition C) aligned agents more than consensus (Condition B) — the β-regularizer dominates gradient magnitude, and consensus signal is individualized per agent rather than averaging toward shared representations.
**Artifacts:** `results/exp_005b_language_conditioned_geometry/`

---

### exp_006 — Regional Divergence (Three Biological Regimes)
**Status:** PASS — divergence hypothesis not supported (strong anchoring result)
**Regimes:**
- vervet (σ=0, full network): final between-cluster distance 0.005 (flat from round 2)
- raven (σ=0.05, bridge every 5 rounds): 0.006 (climbs to ≈0.07 pre-bridge, crashes at round 5)
- latin (σ=0.10, fully isolated): 0.078 (starts at 0.12, slowly converges — growth rate NEGATIVE)

**Result:** Shared DINOv2 perception resists linguistic divergence even under maximal isolation, high transmission noise, and disjoint input distributions. Latin clusters do not diverge — they start elevated from disjoint seeding and slowly converge as shared DINOv2 attractors pull them in. Cross-cluster held-out label agreement: 0.978 in latin, 0.989 in vervet/raven.
**Artifacts:** `results/exp_006_regional_divergence/`

---

### Transversal Finding

> Shared perceptual grounding is a stronger stabilizer than regional isolation is a destabilizer.
> The consensus mechanism coordinates decisions (lexical agreement) and resolves boundary cases
> (grounding), but does not reorganize internal representations. DINOv2's geometric consistency
> is the universal anchor across all three perturbation classes: gradient-free centroid feedback
> (exp_005), gradient-based per-agent linear adapter (exp_005b), and topology + input-distribution
> + transmission noise (exp_006).

---

## Roadmap

Experiments are numbered sequentially and gated: each must pass before
the next begins. The experiment directory and its results directory share
the same name (`exp_NNN_description`).

### Completed

- [x] exp_000 — Embedding separability diagnostic
- [x] exp_001 — Single agent lexicon (naming + inverse grounding)
- [x] exp_002 — Grounding controls (falsification)
- [x] exp_002b — Balanced label control (closes Condition A)
- [x] exp_003 — Multi-agent consensus (feedback + no-feedback baseline)
- [x] exp_004 — Neo4j integration + Shannon metrics
- [x] exp_005 — Centroid alignment (passive Sapir-Whorf test)
- [x] exp_005b — Language-conditioned geometry (active Sapir-Whorf test)
- [x] exp_006 — Regional divergence (three biological regimes)

### Next

- [ ] First empirical paper draft (based on exp_000–006)
- [ ] exp_007 — Test 2: second language acquisition (future)
- [ ] exp_008 — Interactive visualization: Game of Life (future)

---

## Theoretical Extensions — Open Research Questions

These are research directions that emerged during the design of the experiment and will be pursued incrementally. They are documented here so the intellectual thread is not lost between sessions.

---

### 1. Centroid Drift as a Measure of Categorical Deformation (Sapir-Whorf)

DINOv2 is frozen — low-level perception does not change. But the **lexicon centroids** (the learned label representations in embedding space) do move over time as agents accumulate new instances under each label.

This centroid drift is a measurable, operationalized form of the **Sapir-Whorf hypothesis**: the claim that language influences cognition, not just the reverse. In human studies, this is notoriously hard to test cleanly because you cannot freeze a person's visual cortex. Here, the separation is exact:

- **Perceptual space** (DINOv2): fixed, identical across agents
- **Categorical space** (lexicon centroids): dynamic, shaped by consensus

The empirical question becomes:
> *Does the consensus process cause agents to draw category boundaries in increasingly similar ways — even when the underlying perceptual space is identical?*

If centroids converge across agents: the shared label is aligning internal representations — evidence for Sapir-Whorf weak form.
If centroids diverge despite label consensus: agents share a surface label over internally distinct concepts — evidence against, and equally interesting.

**Implementation target:** Track centroid position per label per agent per round in Neo4j. Measure inter-agent centroid distance over time.

---

### 2. Regional Divergence — From Latin Vulgar to Romance Languages

If the agent network is partitioned into **weakly connected subgraphs** (clusters with limited inter-cluster communication), the same initial label may drift toward different centroids in each cluster — driven by local consensus dynamics.

This is computationally analogous to how *Latin vulgar* diverged into the Romance languages: the underlying perceptual and conceptual substrate was shared, but geographically isolated communities developed different consensus attractors for the same concept. "Caballus" became "cheval" in Gaul, "caballo" in Iberia, "cavallo" in Italy — same origin, different centroid trajectories.

The graph topology of agent communication **is** the geography. Edge density between clusters controls how much a label "travels" across regions.

**Measurable predictions:**
- Clusters with dense internal communication → fast local convergence, possible divergence from other clusters
- Bridge agents (high betweenness centrality) → label transfer between regions
- Centroid distance between clusters grows monotonically if inter-cluster communication is cut after initial grounding

**Implementation target:** Parameterize network topology in `config.yaml`. Run experiments with fully connected, clustered, and sparse topologies. Compare centroid trajectories across conditions.

---

### 3. Graph Database — Neo4j Integration

All of the above requires a temporal graph model. Neo4j will store:

```
(:Agent {id, cluster})
(:Label {name, round_created})
(:Centroid {vector, round, agent_id})
(:Image {hash})

(:Agent)-[:ASSIGNED {round, confidence}]->(:Label)
(:Label)-[:APPLIED_TO {round}]->(:Image)
(:Label)-[:HAS_CENTROID {round}]->(:Centroid)
(:Agent)-[:TRANSFERRED_TO {round, label}]->(:Agent)
(:Label)-[:COMPETED_WITH {winner, round}]->(:Label)
```

This enables Cypher queries for:
- Label propagation paths between agents
- Centroid drift trajectory per label per cluster
- Identification of influential agents (PageRank on transfer graph)
- Detection of regional divergence (community detection on consensus graph)
- Animated visualization of lexical emergence over rounds

**Implementation target:** `src/graph/neo4j_logger.py` — parallel logging to Neo4j alongside the existing ledger, so the experiment runs unchanged and the graph is a read layer.

---

### 4. Broader Implications — Edge Computing and Robotics

If the experiment succeeds, the architecture has direct applications for **edge AI**:

- DINOv2-small runs on a Raspberry Pi 5 or Jetson Nano
- The lexicon is just a dictionary of centroids — kilobytes, not gigabytes
- Consensus is distributed — no node needs global visibility
- Vocabulary adapts to local environment without cloud dependency

A robot deployed in an unknown environment could build a shared vocabulary with other robots in the same space through the same consensus mechanism demonstrated here — grounding novel concepts from minimal human instruction, without retraining.

This is qualitatively different from current edge AI, where intelligence lives in the cloud and the robot is an actuator.

---

### 5. Label Transmission — Text Now, Audio Later

Labels are currently transmitted as plain strings between agents — a perfect, lossless copy. This is intentional for Phase 1 and 2: it isolates the consensus dynamics from transmission noise, making the experiment cleaner and more defensible for the first paper.

However, perfect transmission prevents drift. In natural language, labels are never copied exactly — they are perceived, processed, and reproduced with small variations that accumulate over time and across communities. This is the mechanism behind phonetic change and, ultimately, language divergence.

**The planned transition:**

| Phase | Transmission mode | Drift mechanism |
|---|---|---|
| 1–3 | Plain string | None — label is a fixed identifier |
| 5 | Centroid vector + Gaussian noise | Controlled semantic drift |
| Future | Audio synthesis + recognition | Natural acoustic drift |

The Gaussian noise approach is the mathematically clean intermediate step. Instead of transmitting the string `"vorpal"`, an agent transmits its **centroid vector** for that label with added noise:

```python
transmitted = centroid("vorpal") + N(0, σ²)
```

The receiving agent maps this noisy vector back to its nearest label. If σ is small, transmission is nearly perfect. As σ grows, centroids in isolated clusters begin drifting in different directions — producing regional divergence without changing the surface string.

Crucially, the noise parameter σ does not need to be arbitrary. It can be **calibrated against real phonetic drift rates** measured in historical linguistics corpora, connecting the experiment quantitatively to empirical language change data.

The audio transition is kept for a future experiment because it adds acoustic synthesis and recognition infrastructure without changing the underlying dynamics. The conceptual move is the same — noise enters at transmission. The text-to-audio upgrade is clean when it comes.

---

### 6. Three Biological Regimes — From Vervet Monkeys to Romance Languages

The combination of network topology and transmission noise produces three distinct communication regimes, each mapping to a known biological system:

```
Fully connected network + zero noise     →  fixed signals     →  vervet monkeys
Clustered network + low noise            →  local drift       →  ravens / cetaceans
Isolated clusters + moderate noise       →  divergence        →  Latin → Romance languages
```

**Fixed signal systems (vervet monkeys, most species):**
Most animal communication systems transmit labels with near-perfect fidelity. The alarm call for "eagle" is the same call across all individuals, with no creative recombination. These are labels, not language — they cannot drift because there is no mechanism for variation accumulation.

**Drift-capable systems (ravens, cetaceans, some primates):**
Ravens have regional dialects — calls in one population differ measurably from those in another. This implies their labeling system has a drift component absent in most species. The condition for this appears to be: populations large enough that not all individuals interact directly, combined with sufficient social transmission fidelity to maintain local consensus while allowing inter-regional divergence.

**Divergence systems (humans):**
Full linguistic divergence — the Latin vulgar case — requires isolated clusters with enough internal density to maintain local stability, combined with sufficient noise or time for centroids to migrate to distinct attractors. The surface label ("caballus") can remain recognizable across regions while the internal representation (centroid position in conceptual space) diverges significantly.

**The experimental prediction:**
Network topology alone — without changing the agents — should reproduce all three regimes:

```python
# Experiment configurations
FULLY_CONNECTED   = {"topology": "complete",  "sigma": 0.0}   # vervet
CLUSTERED_WEAK    = {"topology": "clustered", "sigma": 0.05}  # raven
CLUSTERED_ISOLATED = {"topology": "isolated", "sigma": 0.10}  # latin
```

If the experiment confirms this, the consensus mechanism is not just a test of AI language acquisition — it is a **minimal formal model of how communication systems evolve across biological substrates**, from fixed signals to full linguistic divergence.

This would be a result that speaks beyond AI — to linguistics, cognitive science, and evolutionary biology simultaneously.

**Implementation target:** Phase 5 — parameterize topology and σ in `config.yaml`, run all three conditions, compare centroid divergence trajectories in Neo4j.

---

### 7. Connection to Shannon's Mathematical Theory of Communication

Shannon (1948) built a mathematical theory of symbol transmission. His definition of information is purely statistical — entropy H(X) measures uncertainty in a distribution without any regard for what the symbols *mean*. He explicitly stated that the semantic aspects of communication are irrelevant to the engineering problem he was solving.

His co-author Weaver identified this as a deliberate scope limitation, distinguishing three levels of the communication problem:
- **Level A — Technical:** How accurately can symbols be transmitted? *(Shannon solved this)*
- **Level B — Semantic:** Do the transmitted symbols convey the intended meaning? *(left open)*
- **Level C — Effectiveness:** Does the meaning produce the desired effect? *(left open)*

This experiment lives precisely in the gap Shannon left open. The tools he developed for Level A are directly applicable to measure what happens at Level B — which is what makes the connection theoretically productive rather than merely analogical.

---

**Concrete connections:**

**The consensus mechanism is error correction.**
Shannon proved that any noisy channel has a maximum capacity C, and that reliable transmission below that limit is possible through redundancy. The consensus mechanism in this experiment *is* error correction — when 70% of agents converge on "vorpal", they create sufficient redundancy to make the label robust against individual perturbations. The convergence threshold is essentially an error correction parameter, and can be analyzed as such.

**The experiment is entropy reduction.**
Before consensus, the distribution of labels for a given image has high entropy — many labels competing with similar probabilities. After convergence, entropy approaches zero. The entire experiment can be described as a controlled entropy reduction process in label space:

```
H(label | image) before consensus  →  high
H(label | image) after consensus   →  ≈ 0
```

This is directly measurable per round in Neo4j using the label distribution in the ledger.

**Mutual information as a grounding metric.**
The success of Condition 1 (naming) can be expressed as the mutual information between the embedding space and the label space:

```
I(image ; label) = H(label) - H(label | image)
```

If I is high, labels are informative with respect to perception — grounding worked. If I is low, labels are arbitrary with respect to perception — the experiment failed. This is the grounding metric expressed in Shannon's terms, connecting it to a 75-year-old theoretical framework.

**Gaussian noise and channel capacity.**
When Phase 5 introduces σ on centroid transmission, the channel capacity becomes:

```
C = (1/2) log(1 + SNR)    where SNR = ||centroid||² / σ²
```

The three biological regimes map to distinct operating points on the channel capacity curve. The vervet operates near a perfect channel. The Latin divergence case operates near the breakdown threshold where information can no longer be transmitted reliably between clusters — and linguistic divergence emerges precisely at that boundary.

---

**The fundamental tension — and why it matters:**

Shannon measures information as statistical surprise. A symbol is informative if it is unexpected. But in this experiment, the label "vorpal" has *zero* entropy — and therefore zero Shannon information — at the moment of full convergence. Every agent uses it, nothing is surprising.

And yet that is precisely the moment when the system has learned something. When the label has *meaning*.

This inverts the intuitive relationship: **learning produces entropy reduction, but Shannon would measure that as less information.** What Shannon's framework cannot capture is semantic information — the relationship between a symbol and the perceptual representation it grounds.

Stated precisely: **Shannon measures whether the channel works. This experiment measures whether what travels through the channel means something.**

This positions lexical-consensus as an empirical investigation of Level B — the semantic level that Weaver identified in 1949 as the next unsolved problem. Shannon's metrics (entropy, mutual information, channel capacity) are the measurement tools. The question being answered is the one he deliberately left open.

---

**Implementation target:** Add entropy and mutual information metrics to `src/metrics/` computed per round from the ledger distribution. Plot entropy reduction curves alongside convergence rate — they should be inversely correlated, and the shape of that correlation is a result.

---

### 8. A New Benchmark for Machine Intelligence

If the experiment validates Test 1 of Vera et al., the broader claim follows: **language acquisition from minimal instruction through social consensus is a more fundamental measure of machine intelligence than behavioral imitation** (the Turing Test).

A system that passes this test has demonstrated:
- Grounding: connecting symbols to perceptual experience
- Generalization: applying learned labels to unseen instances
- Social coordination: aligning internal representations through consensus
- Bidirectionality: recovering perceptual representation from label alone (Test 1, Condition 2)

None of these are measured by the Turing Test. Together, they constitute the first layer of what would eventually become a genuine world model — the naming of the world, before the modeling of its causal structure.

---

## Neo4j Graph Status

The Neo4j AuraDB instance contains the complete experimental record for experiments that ran with live graph logging:

| Content | Count |
|---|---|
| Nodes | 2,023+ |
| Relationships | 7,610+ |
| Experiments logged | exp_003a, exp_005, exp_005b, exp_006 |

**Node types:** Experiment, Agent, Label, Image, Assignment, Centroid, Adapter

**Relationship types:** HAS_AGENT, HAS_LABEL, HAS_IMAGE, MADE, USES_LABEL, APPLIES_TO, HAS_CENTROID, CONSENSUS_WITH, PROXIMITY_TO, PROJECTED_PROXIMITY_TO, HAS_ADAPTER

Replay scripts in `src/graph/replay_exp003.py` reconstruct the full exp_003a assignment graph from ledger logs. `src/graph/neo4j_logger.py` provides the parallel logging interface used in all subsequent experiments.

---

## Reproducibility Commitment

Every experiment in this repository follows the **VVUQ guidelines** (Verification, Validation & Uncertainty Quantification) referenced in the originating paper (Adams, 2012; Coveney & Highfield, 2021). Specifically:

- All random seeds are logged and fixed per experiment
- All results include confidence intervals
- All experimental configurations are versioned in `config.yaml` files
- First positive results require independent confirmation before claims are made

---

## Authors

**Patricio Vera** — Neurocreaciones, Las Condes, Santiago, Chile *(repository maintainer)*

Pedro Moya and Lisa Barraza are co-authors of the originating theoretical paper (Vera et al.) but are not active contributors to this implementation.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Citation

If you use this repository, please cite both the theoretical paper and this implementation:

```bibtex
@article{vera2024rethinking,
  title={Rethinking the Evaluating Framework for Natural Language Understanding 
         in AI Systems: Language Acquisition as a Core for Future Metrics},
  author={Vera, Patricio and Moya, Pedro and Barraza, Lisa},
  journal={Journal of Experimental \& Theoretical Artificial Intelligence},
  year={20xx}
}

@software{vera2024lexical,
  title={Lexical Consensus: Emergent Language Acquisition
         in Multi-Agent Systems with Frozen Visual Perception},
  author={Vera, Patricio},
  year={2024--2026},
  url={https://github.com/patriciomvera/lexical-consensus}
}
```
