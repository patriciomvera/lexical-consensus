# PROTOCOL.md — Formal Experiment Protocol
## Lexical Consensus: Test 1 Implementation

**Version:** 0.1.0  
**Status:** Draft  
**Authors:** Anonymous.  
**Last updated:** 2026

---

## 1. Objective

To test whether a network of LLM-based agents can acquire and stabilize a shared artificial lexicon for novel visual concepts, using only progressive verbal instruction from a human tutor — without access to pre-existing label knowledge.

This constitutes a minimal, reproducible instantiation of **Test 1** from Vera et al. (2023):

> *"The first proposed test for machine intelligence should assess whether a machine, through direct verbal instruction from a human teacher, can describe its surroundings without any preloaded data sets or algorithms."*

---

## 2. Definitions

**Tutor:** The human instructor. Provides initial label assignments via textual description only. Does not provide formal definitions, category names, or any label that exists in standard image datasets.

**Learner Agent:** An LLM instance (GPT-4, Claude, or equivalent) operating under a system prompt that prohibits use of prior label knowledge. Receives tutor instructions and inter-agent communications only.

**Artificial Lexicon:** A set of invented, phonotactically valid but semantically empty labels (e.g., *vorpal*, *clinth*, *draven*) assigned by the tutor to visual concept categories. Full vocabulary defined in `docs/artificial_lexicon.md`.

**Consensus Ledger:** A shared data structure recording all label assignments made by all agents, with timestamps and confidence weights.

**Convergence:** The state where ≥ 70% of active agents assign the same artificial label to the same image category, sustained over ≥ 10 consecutive interaction rounds.

**Label Drift:** A measurable shift in dominant label assignment for a given category across agents over time.

---

## 3. Constraints

The following constraints are inviolable across all experiments:

1. **No ground-truth leakage.** Original dataset labels are stripped before any agent sees the data. Labels are never revealed during the experiment.

2. **Tutor instruction only.** The only source of grounding is the tutor's verbal/textual description. Example: *"The vorpal is the round object that holds liquid."*

3. **No inter-agent shared memory.** Agents communicate only through the consensus ledger — they do not share internal states, embeddings, or model weights.

4. **Fixed context window.** Each agent interaction uses a context window capped at a defined token limit, to prevent unbounded knowledge accumulation that would not reflect real acquisition dynamics.

5. **Logged and versioned.** Every interaction is logged with timestamp, agent ID, label assigned, confidence score, and the image hash. Logs are immutable after writing.

---

## 4. Procedure

### 4.1 Dataset Preparation

1. Select source dataset (default: CIFAR-10 subset, 10 categories, 100 images per category).
2. Strip all original labels. Store label mapping in an encrypted file inaccessible to agents.
3. Assign artificial labels to categories (tutor action, offline, before experiment begins).
4. Define **seed set**: 5 images per category that the tutor will use for initial instruction.
5. Define **test set**: 20 images per category, never shown during instruction, used for generalization measurement only.

### 4.2 Tutor Initialization

The tutor provides verbal descriptions for each seed image using only the artificial label and observable properties. Example format:

```
"This is a [ARTIFICIAL_LABEL]. It is [observable property 1]. It is [observable property 2]. 
It is not [contrast property]."
```

No synonyms, no reference to real-world category names, no analogies to known concepts.

### 4.3 Agent Learning Phase

1. Each Learner Agent receives tutor seed descriptions sequentially (not all at once).
2. After each description, the agent is shown a new image and asked: *"What is this? Use only the vocabulary you have been taught."*
3. The agent's response is logged to the consensus ledger with a confidence score (derived from the model's own uncertainty, if accessible, or from inter-agent agreement).

### 4.4 Consensus Phase

1. After each round of individual labeling, the consensus mechanism runs:
   - Compute the distribution of labels assigned to each image across all agents.
   - Labels meeting the convergence threshold (≥ 70% agreement) are marked as **stable**.
   - Labels below a rejection threshold (< 20% agreement) are marked as **discarded**.
   - Agents receive feedback: which labels are stable, which are discarded. No label content from other agents is revealed — only the stability status of their own labels.

2. Agents adjust their labeling behavior based on stability feedback.

3. The cycle repeats for a minimum of 25 rounds.

### 4.5 Generalization Test

After convergence or at round 25 (whichever comes first):

1. Show each agent the **test set** (images never seen during training).
2. Ask each agent to label each test image using the acquired vocabulary.
3. Measure agreement with the tutor's intended artificial labels (ground truth, now revealed for scoring only).

---

## 5. Metrics

All metrics are defined formally in `docs/metrics.md`. Summary:

| Metric | What It Measures | Expected Range |
|---|---|---|
| **Convergence Round** | Round at which first stable label emerges | 1–25 |
| **Full Convergence Rate** | % of categories that reach stable consensus | 0–100% |
| **Label Stability Index** | Variance of dominant label across last 10 rounds | 0–1 |
| **Generalization Accuracy** | Agreement with tutor intent on test set | 0–100% |
| **Drift Score** | Net label change from round 1 to round 25 | 0–N |
| **Efficiency Index** | Composite score (per Vera et al. framework) | 0–1 |

All results are reported as mean ± 95% confidence interval over N independent runs.

---

## 6. Reproducibility Requirements

Following VVUQ guidelines (Adams, 2012):

- Random seed fixed and logged per experiment run
- All LLM API calls logged with model version, temperature, and full prompt
- Results stored in versioned JSON files under `results/`
- Any anomalous agent behavior logged as an outlier event
- A positive convergence result requires confirmation by at least one independent run with different random seed before any claim is made

---

## 7. Null Hypothesis

> *H₀: A network of LLM agents exposed to artificial label instruction will not achieve stable lexical convergence above chance level (defined as > 1/N where N = vocabulary size).*

A result that rejects H₀ constitutes empirical support for Test 1 of the Vera et al. framework.

---

## 8. Known Limitations

1. **Pre-training contamination — visual.** LLMs and vision models have seen CIFAR/ImageNet images during training. The three-layer architecture (frozen DINOv2 + empty lexicon + consensus) mitigates this at the perceptual level: DINOv2 produces geometry, not categories. Future work (Test 2) will use procedurally generated images for a fully clean baseline.

2. **Pre-training contamination — lexical (Carroll vocabulary).** The artificial vocabulary used in this experiment consists of invented words from Lewis Carroll's *Jabberwocky* (1872) and the Alice universe. These texts are in the public domain and are almost certainly present in the training corpora of any LLM used in the pipeline. A model may carry latent linguistic associations for words like "vorpal" (sharp, cutting) or "slithy" (smooth, sinuous) that could influence its labeling behavior independent of tutor instruction.

   This is a known and documented risk. It is not fatal for the following reasons:

   - DINOv2 (the perceptual layer) has no exposure to text and carries no Carroll associations. The contamination risk is confined to any LLM component in the consensus pipeline.
   - Carroll associations are *linguistic*, not *visual*. The experiment assigns Carroll labels to specific CIFAR-10 visual categories that may or may not align with those associations. Tension between prior linguistic association and tutor instruction is itself a measurable variable.
   - No vocabulary is perfectly neutral for an LLM. Models generalize by phonetics and morphology — even entirely invented strings like "clinth" or "sulox" may trigger analogical associations. Carroll words are not uniquely contaminated; they are more *transparently* contaminated, which is scientifically preferable.

   **Mitigation — three control conditions:**

   To quantify the effect of lexical pre-training contamination, the following control experiment will be run in parallel with the main Carroll condition:

   | Condition | Vocabulary | Expected contamination |
   |---|---|---|
   | **A — Carroll** | vorpal, slithy, mimsy, borogove... | Moderate — public domain literary associations |
   | **B — Ad-hoc invented** | clinth, draven, sulox, merith... | Low — no known prior associations |
   | **C — Numeric labels** | label_0, label_1, label_2... | Minimal — purely indexical, no semantic surface |

   If all three conditions produce equivalent consensus dynamics, lexical contamination did not materially affect the result — and Condition A is the most publishable and communicable. If conditions differ, the differential *is* the result: a direct empirical measure of how much prior linguistic knowledge influences visual concept formation. Both outcomes are informative and publishable.

   This control structure is declared in advance — not added post-hoc — in compliance with VVUQ pre-registration principles.

3. **Consensus threshold is operationally chosen.** The 70% convergence threshold is not derived from language acquisition theory or biological data. It is a starting parameter. Sensitivity analysis across thresholds (50%, 60%, 70%, 80%) will be reported in the full results.

4. **Single modality.** This experiment uses visual input only. The full theoretical framework (Vera et al.) envisions multimodal grounding including embodiment, affect, and social context. Visual-only grounding is an acknowledged simplification and a starting point, not a final claim.

5. **No survival pressure.** Unlike Steels' embodied agents, our agents have no ecological incentive to communicate accurately. The consensus mechanism provides social pressure as a proxy, but the biological motivation for communication accuracy is absent. This gap between experiment and theory is acknowledged and will be addressed in the discussion section of the empirical paper.

---

## 9. Version History

| Version | Date | Change |
|---|---|---|
| 0.1.0 | 2024 | Initial draft |
| 0.2.0 | 2024 | Added Carroll vocabulary, three control conditions, expanded limitations |

