# Novel-Word Learning over Frozen Perception — Experimental Code

[![Status: Complete](https://img.shields.io/badge/Status-Complete-green.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![License: See note](https://img.shields.io/badge/License-See%20note-lightgrey.svg)](#license)

---

## What This Is

This repository contains an experimental framework that tests whether artificial agents can acquire, stabilize, and bidirectionally use a novel grounded lexicon over a frozen perceptual substrate. Agents receive opaque invented labels for visual categories and must construct shared meaning from minimal instruction and mutual agreement — without access to any pre-existing semantic knowledge about those labels. The framework treats language acquisition, rather than behavioral imitation, as the object of evaluation. The full experimental program spans nine families of experiments (exp_000 through exp_008); additional robustness analyses run on the released code and data are in [`revision_analyses/`](revision_analyses/).

## Papers

This repository accompanies two preprints by Patricio M. Vera:

1. **Lexical Consensus: Grounded Word Learning and Shared Meaning in Artificial Agents** (2026). [arXiv:2606.22207](https://arxiv.org/abs/2606.22207). The artificial-learner experiments in this repository (`exp_000`–`exp_008` and `revision_analyses/`) are reported there.
2. **Visual Similarity in Adult Word Extension and Artificial Word Learning** (2026). [arXiv:2611.14116](https://arxiv.org/abs/2611.14116). A continuation of that program: a reanalysis of public human word-extension data together with further learner analyses. Scripts and notes are in [`human_data_analyses/`](human_data_analyses/). Companion datasets: [`MrPatoVera/lexical-consensus-data`](https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data) (CC BY 4.0) and [`MrPatoVera/lexical-consensus-data-cifar10h`](https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data-cifar10h) (CC BY-NC-SA 4.0). Collection: [Lexical Consensus](https://huggingface.co/collections/MrPatoVera/lexical-consensus-6af31e7f9870b8e83e668c49).

---

## The Carroll Framework — Following the White Rabbit

> *"'When I use a word,' Humpty Dumpty said, in rather a scornful tone, 'it means just what I choose it to mean — neither more nor less.'*
> *'The question is,' said Alice, 'whether you can make words mean so many different things.'*
> *'The question is,' said Humpty Dumpty, 'which is to be master — that's all.'"*
> — Lewis Carroll, *Through the Looking-Glass* (1871)

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

## Key Finding

Lexical acquisition over frozen perception is not arbitrary set learning. It follows a **monotonic perceptual-coherence gradient**: concepts whose positive examples form a tight, coherent cluster in DINOv2 embedding space are acquired reliably (native C1 ≈ 0.943), while concepts defined as arbitrary unions of perceptually distant classes degrade toward chance (far C1 ≈ 0.530, chance = 0.333). The four tiers of this gradient — native → near-disjunctive → mid-disjunctive → far-disjunctive — are documented in exp_007b_extension_clean (centroid learner: 0.943 → 0.847 → 0.654 → 0.530; all other supervised learners follow the same monotonic ordering).

A pre-registered dissociation experiment (exp_008) rules out the competing explanation that this gradient is a measurement tautology arising from evaluating C1 in the same DINOv2 space used to define concept tiers. Using an independent semantic distance measure (WordNet Wu-Palmer similarity over CIFAR-100) to identify pairs where perceptual and semantic coherence disagree, the regression confirms:

- **Perceptual distance governs C1** (Model PS, centroid learner): β_perc = −0.212, partial R² = 0.245, p = 1.87 × 10⁻⁷
- **Semantic distance is non-significant**: β_sem = −0.013, partial R² = 0.002, p = 0.660

The pre-registered decision Rule A is triggered: perception governs, semantics adds nothing, and the gradient has genuine empirical content beyond circularity.

---

## Architecture

The experimental framework is organized into four layers:

**Layer 1 — Frozen perceptual encoder**
`facebook/dinov2-small`, 384-dimensional CLS token embeddings, L2-normalized. Shared and frozen across all agents. No labels are ever passed to the encoder; it has no category knowledge to contaminate the lexical learning process.

**Layer 2 — Learnable lexical layer**
Maps between embeddings and Carroll labels bidirectionally. Default implementation: centroid-based nearest-neighbor (one centroid per label, updated by ACCEPT events). Additional learner variants: multi-centroid (k-centroid), exemplar k-NN (k = 3), logistic regression, linear SVM, and random baseline. Introduced in exp_007 to separate the contribution of learning mechanism from perceptual structure. A per-agent trainable linear projection (`LexicalAdapter`, introduced in exp_005b) is available as an optional additive gradient component but is not the default; DINOv2 remains frozen throughout.

**Layer 3 — Multi-agent consensus (optional)**
Label competition across agents is mediated by majority agreement: a label is ACCEPTED when ≥ τ = 2/3 of the agent population agrees (as implemented in `experiments/exp_003_multi_agent_consensus/_shared.py` and `run_003a.py`). The standalone `src/consensus/ledger.py` module is not used by the experiment scripts. Active in exp_003 onward; disabled in baseline conditions (exp_003b) and single-agent experiments.

**Layer 4 — Measurement layer**
All experiments report at minimum: Condition 1 (C1) naming accuracy, Condition 2 (C2) inverse grounding retrieval, and confidence margins. Consensus experiments additionally report: convergence round, unanimity trajectory, held-out accuracy, Shannon entropy (NMI), and centroid drift. The dissociation experiment (exp_008) adds: partial R² decomposition, likelihood ratio tests, and quadrant-stratified dissociation analysis (Q1–Q4 defined by median split of d_perc × d_sem).

**Artificial vocabulary**: Labels are Lewis Carroll's invented words from *Jabberwocky* (vorpal, slithy, mimsy, borogove, tulgey, frumious, manxome, galumphing, uffish, jubjub). These are phonotactically valid but semantically empty. No text encoder is ever applied to the labels; they function as opaque identifiers. Pre-trained model knowledge of English cannot contaminate their meaning because they have no meaning to contaminate.

---

## Experiment Inventory

Each experiment family is gated — it must pass before the next begins. Outcomes listed below are final; intermediate runs and sensitivity analyses are documented in per-experiment `report.md` files.

| Experiment | Purpose | Main outcome |
|---|---|---|
| [exp_000](experiments/exp_000_embedding_separability/) | Perceptual substrate diagnostic | DINOv2-small provides moderate category separability (silhouette = 0.283 on frog/horse/ship) |
| [exp_001](experiments/exp_001_single_agent_lexicon/) | Single-agent lexical acquisition | Carroll labels acquired from as few as 10 examples; C1 accuracy 1.000 at 10 seeds |
| [exp_001_C2](experiments/exp_001_single_agent_lexicon/) | Bidirectional grounding | Labels support image retrieval, not only naming; OOV asymmetry confirms genuine grounding |
| [exp_002](experiments/exp_002_grounding_controls/) | Grounding falsification controls | Grounding collapses when perception, label binding, or embedding structure is broken |
| [exp_002b](experiments/exp_002b_balanced_label_control/) | Balanced-label control | Random-label artifacts are unstable under repeated scrambles; population mean C1 ≈ 0.342 |
| [exp_003](experiments/exp_003_multi_agent_consensus/) | Multi-agent consensus | Three agents with disjoint seed sets converge on shared lexical usage; consensus refines but does not drive alignment |
| [exp_004](experiments/exp_004_neo4j_shannon/) | Shannon and graph logging | Label assignments show high information transfer (NMI = 0.966–0.988 from round 1); graph stored in Neo4j |
| [exp_005](experiments/exp_005_centroid_alignment/) | Passive centroid alignment | Consensus feedback does not substantially reshape centroid geometry under frozen architecture |
| [exp_005b](experiments/exp_005b_language_conditioned_geometry/) | Active language-conditioned geometry | Trainable linear adapters (LexicalAdapter) do not make consensus a representational attractor; random feedback aligns agents more than consensus |
| [exp_006](experiments/exp_006_regional_divergence/) | Regional divergence | Shared DINOv2 perception resists divergence under topology manipulation, transmission noise, and disjoint input distributions |
| [exp_007a](experiments/exp_007_concept_carving/) | Native concept baseline | Native concepts (coherent DINOv2 clusters) acquired reliably across all learner types; H2 holds |
| [exp_007b](experiments/exp_007_concept_carving/) | Far-disjunctive concept stress test | Arbitrary unions of perceptually distant classes degrade toward chance in C1; frozen DINOv2 is a hard acquisition ceiling |
| [exp_007b_extension](experiments/exp_007b_extension/) | Near/mid/far concept-carving gradient | C1 follows a monotonic perceptual-coherence gradient across four tiers; C2 episodic advantage is robust in disjunctive regime |
| [exp_007b_clean](experiments/exp_007b_extension_clean/) | Homogeneous C2 pool re-run | C2 retrieval gap is disjunctive-specific; NMI_native = 0.357 is a mathematical constant, not a discriminative gate |
| [exp_008](experiments/exp_008_dissociation/) | Perceptual vs. semantic dissociation | Perceptual distance governs C1 (partial R² = 0.245, p = 1.87 × 10⁻⁷); semantic distance non-significant (partial R² = 0.002, p = 0.660) |

---

## Repository Structure

The actual directory layout is:

```
./
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py              # Abstract agent interface
│   │   ├── perception.py              # DINOv2 wrapper (Layer 1, frozen)
│   │   ├── lexicon.py                 # Centroid-based lexicon (Layer 2)
│   │   ├── lexical_adapter.py         # Per-agent trainable projection (exp_005b+)
│   │   └── learner_agent.py           # Full three-layer agent
│   ├── consensus/
│   │   └── ledger.py                  # Consensus mechanism (Layer 3)
│   ├── dataset/
│   │   └── artificial_vocab.py        # Carroll vocabulary definitions
│   ├── eval/
│   │   ├── concepts.py                # Concept tier definitions (native/near/mid/far)
│   │   ├── descriptors.py             # Pair descriptors for concept carving
│   │   ├── episodes.py                # Paired episode generation
│   │   ├── falsification.py           # Grounding control conditions
│   │   ├── harness.py                 # Experiment harness
│   │   ├── metrics.py                 # C1/C2 accuracy, confidence margins
│   │   └── stats.py                   # Bootstrap CI, Wilcoxon, paired tests
│   ├── learners/
│   │   ├── base.py                    # Learner interface
│   │   ├── centroid.py                # Centroid nearest-neighbor
│   │   ├── exemplar_knn.py            # Exemplar k-NN (k=3)
│   │   ├── kcentroid.py               # Multi-centroid
│   │   ├── logreg.py                  # Logistic regression
│   │   ├── linsvm.py                  # Linear SVM
│   │   └── random_baseline.py         # Random chance baseline
│   ├── graph/
│   │   ├── neo4j_logger.py            # Neo4j AuraDB logging
│   │   └── replay_exp003.py           # Consensus graph replay
│   └── metrics/
│       └── shannon.py                 # Shannon entropy and NMI
│
├── experiments/
│   ├── exp_000_embedding_separability/ # Perceptual substrate gate
│   ├── exp_001_single_agent_lexicon/   # Naming curve + inverse grounding (C2)
│   ├── exp_002_grounding_controls/     # Falsification conditions A–E
│   ├── exp_002b_balanced_label_control/ # Condition A closure
│   ├── exp_003_multi_agent_consensus/  # 003a (feedback) + 003b (no-feedback baseline)
│   ├── exp_004_neo4j_shannon/          # Shannon metrics + Neo4j replay
│   ├── exp_005_centroid_alignment/     # Passive Sapir-Whorf test
│   ├── exp_005b_language_conditioned_geometry/ # Active Sapir-Whorf, LexicalAdapter
│   ├── exp_006_regional_divergence/    # Three biological regimes (vervet/raven/latin)
│   ├── exp_007_concept_carving/        # 007a native baseline + 007b disjunctive stress test
│   ├── exp_007b_extension/             # Near/mid/far gradient; C2 pool sensitivity
│   ├── exp_007b_extension_clean/       # Homogeneous all-tier re-run; NMI_native audit
│   └── exp_008_dissociation/           # Perceptual vs. semantic dissociation (pre-registered)
│       ├── src/
│       │   ├── phase1/                 # DINOv2 and WordNet distance matrices
│       │   ├── phase2/                 # Quadrant classification and pair sampling
│       │   ├── phase3/                 # C1 acquisition episodes (12,000 total)
│       │   └── phase4/                 # OLS regression, figures, decision rules, report
│       ├── config/
│       │   ├── exp_008_config.yaml
│       │   └── wordnet_mappings.json
│       └── exp_008_preregistration.md  # Frozen pre-registration spec
│
├── results/                            # Experiment outputs (gitignored except structure)
│   ├── exp_000_embedding_separability/
│   ├── exp_001{a,b,c}_seed_{05,10,15}/
│   ├── exp_001b_condition2_{easy,medium,hard}/
│   ├── exp_002_grounding_controls/
│   ├── exp_002b_balanced_label_control/
│   ├── exp_003{a,b}_*/
│   ├── exp_003_comparison/
│   ├── exp_004_neo4j_shannon/
│   ├── exp_005_centroid_alignment/
│   ├── exp_005b_language_conditioned_geometry/
│   ├── exp_006_regional_divergence/
│   ├── exp_007/
│   ├── exp_007b_extension/
│   ├── exp_007b_extension_clean/
│   └── exp_008_dissociation/
│       ├── phase1/                     # Distance matrices, correlation report
│       ├── phase2/                     # Quadrant counts, sampled pairs
│       ├── phase3/                     # Episode results (12,000 rows)
│       ├── phase4/                     # regression_table.csv, 5 figures, raw OLS summaries
│       ├── report.md                   # Original run report
│       ├── exp_008_report_corrected.md # Report with VIF errata noted
│       ├── exp_008_decision.md         # Pre-registered rule outcome
│       └── exp_008_regression_table_corrected.csv  # VIF-corrected table (VIF = 1.0105)
│
├── tests/
│   ├── agents/test_lexical_adapter.py
│   ├── eval/                           # Concept carving, episode, metric, pipeline tests
│   ├── learners/                       # Learner interface and exemplar weighting tests
│   └── metrics/test_shannon.py
│
├── revision_analyses/                  # Robustness analyses r01–r07b (see its README)
│
├── docs/
│   └── experiment_summary.md           # One-page summary for external review
│
├── README.md                           # This file
├── PROTOCOL.md                         # Formal experiment protocol (citable)
├── SETUP.md                            # Environment setup instructions
└── requirements.txt                    # Python dependencies
```

---

## Pre-Registration

The dissociation experiment (exp_008) was pre-registered before Phase 3 data collection began. Decision rules A, B, and C — covering all three possible outcomes (perception governs / ambiguous / semantics wins) — were written in full and frozen in [`experiments/exp_008_dissociation/exp_008_preregistration.md`](experiments/exp_008_dissociation/exp_008_preregistration.md) on 2026-06-08.

The pre-registered data collection and analysis proceeded without modification. Rule A was triggered on the first and only run (all four criteria met; see [`results/exp_008_dissociation/exp_008_decision.md`](results/exp_008_dissociation/exp_008_decision.md)).

**VIF correction (documented errata):** The original Phase 4 run (`phase4/regression_table.csv`, timestamp 20260608_173255) reported VIF = 9.693 for both predictors in Model PS. This value arose from a statsmodels intercept artifact: the auxiliary VIF regression was fit without an intercept, inflating the condition number. The corrected VIF, fit with an intercept, is **1.0105** — indicating negligible multicollinearity (confirmed by the Phase 1 correlation diagnostic: Pearson r = 0.155, Spearman ρ = 0.089 between d_perc and d_sem). The correction does not affect any regression coefficients, standard errors, p-values, partial R² values, or Rule A/B/C decision outcomes. A corrected regression table is archived at [`results/exp_008_dissociation/exp_008_regression_table_corrected.csv`](results/exp_008_dissociation/exp_008_regression_table_corrected.csv). The errata are also noted in [`results/exp_008_dissociation/exp_008_report_corrected.md`](results/exp_008_dissociation/exp_008_report_corrected.md) and in [`results/exp_008_dissociation/phase4/ERRATA.md`](results/exp_008_dissociation/phase4/ERRATA.md). The full record, including the spurious Rule B trigger that the original VIF would have produced and its removal after correction, is documented in the errata files above.

---

## Reproducibility

**Environment setup:**

```bash
# CPU-only installation (local development — no GPU required):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# For NLTK WordNet (exp_008 only):
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**Running an experiment family independently:**
Each experiment folder under `experiments/` is self-contained. Example:

```bash
# Re-run the dissociation experiment (requires CIFAR-100 download on first run):
cd experiments/exp_008_dissociation
python src/phase1/embed_cifar100.py
python src/phase1/compute_perceptual_distances.py
python src/phase1/compute_semantic_distances.py
python src/phase1/correlation_diagnostic.py   # review before continuing
python src/phase2/classify_quadrants.py
python src/phase2/sample_pairs.py
python src/phase3/run_episodes.py
python src/phase4/regression_analysis.py
python src/phase4/dissociation_analysis.py
python src/phase4/generate_figures.py
python src/phase4/generate_report.py
```

For cloud GPU runs (concept-carving experiments with many learners), Modal is used as the compute backend. Local CPU is sufficient for development and for all experiments up through exp_006.

**Key design choices that support reproducibility:**

- DINOv2 embeddings are cached to disk on first computation and reused across runs.
- No data augmentation is applied at any stage.
- Concept-carving experiments (exp_007 onward) use **paired episodes**: identical support/query splits are shared across all learners within a run, so all within-run learner comparisons are matched and support paired statistical testing (Wilcoxon signed-rank, directional bootstrap CI).
- Every run logs: master random seed, model version, full configuration YAML, and per-episode results.
- Selected consensus runs are additionally logged to Neo4j AuraDB as a temporal graph of agent interactions (exp_003, exp_004, exp_005b, exp_006).

**Tests:**

```bash
pytest tests/
```

---

## Datasets

**CIFAR-10** (experiments exp_000 through exp_007b_clean):
Three categories used in the core naming experiments: frog, horse, ship (exp_000–exp_006). The concept-carving experiments (exp_007a/b) draw from all 10 CIFAR-10 classes to construct native, near, mid, and far concept tiers. Downloaded via `torchvision.datasets.CIFAR10`. Original category labels are stripped from the agent-facing protocol; agents receive only embeddings and Carroll-vocabulary labels.

**CIFAR-100** (exp_008):
100 fine-grained classes × 600 images per class (500 train / 100 test). Used in the dissociation experiment to sample class pairs stratified by the joint distribution of perceptual distance (DINOv2 cosine) and semantic distance (WordNet Wu-Palmer similarity). Downloaded via `torchvision.datasets.CIFAR100`. As with CIFAR-10, original labels are not passed to the agent layer; the agents receive only embeddings.

WordNet class-to-synset mappings used for Wu-Palmer computation are archived at `experiments/exp_008_dissociation/config/wordnet_mappings.json`.

---

## Robustness Analyses

The folder [`revision_analyses/`](revision_analyses/) contains additional analysis scripts (r01–r07b) that re-use the committed code and results: an exp_008 semantic-measure and bootstrap check (r01), multiple-comparison corrections (r02), embedding-cache rebuild (r03), multi-seed re-runs of exp_007b_extension_clean (r04, r06), tier-distance tabulation (r05), and a partial-correlation check and observed quadrant-median difference for exp_008 (r07, r07b). Small text/CSV outputs are in `revision_analyses/out/`. See [`revision_analyses/README.md`](revision_analyses/README.md) for usage.

---


## Citation

If you use this code or the companion datasets, please cite the relevant paper(s). A machine-readable [`CITATION.cff`](CITATION.cff) is included at the repository root. The preferred citation in that file is the 2026 follow-up paper; the original Lexical Consensus preprint is listed under `references`.

```bibtex
@misc{vera2026visualsimilarity,
  title         = {Visual Similarity in Adult Word Extension and Artificial Word Learning},
  author        = {Vera, Patricio M.},
  year          = {2026},
  eprint        = {2611.14116},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2611.14116},
  url           = {https://arxiv.org/abs/2611.14116}
}
```

```bibtex
@misc{vera2026lexical,
  title         = {Lexical Consensus: Grounded Word Learning and Shared Meaning in Artificial Agents},
  author        = {Vera, Patricio M.},
  year          = {2026},
  eprint        = {2606.22207},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2606.22207},
  url           = {https://arxiv.org/abs/2606.22207}
}
```

---
## License

**No LICENSE file is present at the repository root.** The prior README referenced an MIT license, but the file is absent. A license file will be added in a later release.
