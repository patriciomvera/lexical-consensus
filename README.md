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

| Steels (1996–2015) | This work |
|---|---|
| Physical robots with cameras | LLM agents via API |
| Simple reinforcement rules | Distributed consensus mechanism |
| Small populations (2–10 agents) | Scalable N-agent architecture |
| Custom agent architecture | Commodity LLMs (GPT-4, Claude, etc.) |
| Labels for real objects | Artificial labels on stripped datasets |

This shift from embodied robots to commodity LLMs tests whether the consensus dynamics Steels observed generalize to a fundamentally different substrate.

---

## Repository Structure

```
lexical-consensus/
│
├── docs/                        # Theoretical background and protocol documentation
│   ├── protocol.md              # Formal experiment protocol (citable)
│   ├── metrics.md               # Definition of all metrics used
│   ├── artificial_lexicon.md    # The invented label vocabulary
│   └── paper_reference.md       # Link to the originating theoretical paper
│
├── src/
│   ├── agents/                  # LLM agent implementations
│   │   ├── base_agent.py        # Abstract agent interface
│   │   ├── tutor_agent.py       # The human instructor interface
│   │   └── learner_agent.py     # The learning agent (LLM-backed)
│   │
│   ├── consensus/               # The distributed consensus mechanism
│   │   ├── ledger.py            # Label ledger (who said what, when)
│   │   ├── voting.py            # Weighted voting and threshold logic
│   │   └── drift_detector.py    # Detects label instability over time
│   │
│   ├── dataset/                 # Dataset handling
│   │   ├── loader.py            # Load images, strip labels
│   │   ├── seed_set.py          # Define the tutor's initial instruction set
│   │   └── artificial_vocab.py  # The artificial label definitions
│   │
│   ├── metrics/                 # Measurement and evaluation
│   │   ├── convergence.py       # Convergence speed metrics
│   │   ├── stability.py         # Label stability over time
│   │   ├── generalization.py    # Generalization to unseen instances
│   │   └── efficiency_index.py  # The composite efficiency metric (per paper)
│   │
│   └── utils/                   # Shared utilities
│       ├── logger.py            # Structured experiment logging
│       ├── config.py            # Configuration management
│       └── api_client.py        # LLM API abstraction layer
│
├── experiments/
│   └── exp_001_baseline/        # First experiment: baseline consensus dynamics
│       ├── config.yaml          # Experiment parameters
│       ├── run.py               # Entry point
│       └── README.md            # What this specific experiment tests
│
├── tests/                       # Unit and integration tests
├── notebooks/                   # Analysis and visualization notebooks
├── results/                     # Experiment outputs (gitignored except structure)
│
├── README.md                    # This file
├── PROTOCOL.md                  # Short-form protocol for reproducibility
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT
```

---

## Roadmap

### Phase 1 — Foundation (current)
- [ ] Repository structure and documentation
- [ ] Dataset pipeline (load, strip labels, define seed set)
- [ ] Tutor interface (human → artificial label injection)
- [ ] Single learner agent (3-layer architecture: DINOv2 + lexicon + consensus)
- [ ] Diagnostic script: are CIFAR-10 clusters separable in DINOv2 embedding space?

### Phase 2 — Multi-Agent Consensus
- [ ] N-agent network with shared PerceptionLayer
- [ ] Consensus ledger (voting, threshold, stability window)
- [ ] First multi-agent experiment (exp_001_baseline)
- [ ] Basic metrics: convergence rate, stability index, generalization accuracy

### Phase 3 — Neo4j Integration and Shannon Metrics
- [ ] Graph model: agents, labels, centroids, transfers
- [ ] Parallel logging to Neo4j alongside ledger
- [ ] Cypher queries for label propagation and centroid drift
- [ ] Animated visualization of lexical emergence per round
- [ ] Shannon metrics per round: H(label|image), I(image;label), entropy reduction curve

### Phase 4 — Centroid Drift and Sapir-Whorf Test
- [ ] Track centroid trajectory per label per agent per round
- [ ] Measure inter-agent centroid distance over time
- [ ] Test: does consensus cause categorical alignment beyond label agreement?

### Phase 5 — Regional Divergence and Biological Regimes
- [ ] Implement Gaussian noise on centroid transmission (parameter σ)
- [ ] Parameterize network topology (fully connected / clustered / isolated)
- [ ] Run three experimental conditions: vervet / raven / latin
- [ ] Measure centroid divergence between clusters over time in Neo4j
- [ ] Compare drift rates against historical phonetic change data (calibration)
- [ ] Audio transmission design (architecture only — implementation deferred)

### Phase 6 — Test 2 and Paper
- [ ] Extend to second language acquisition using Test 1 lexicon as base
- [ ] Full statistical analysis (confidence intervals, bootstrapping, VVUQ)
- [ ] Reproducibility confirmation with independent seeds
- [ ] Empirical paper draft

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
  title={Lexical Consensus: Emergent Language Acquisition in Multi-Agent LLM Systems},
  author={Vera, Patricio and Moya, Pedro and Barraza, Lisa},
  year={2024},
  url={https://github.com/neurocreaciones/lexical-consensus}
}
```
