# Experiment 001 — Baseline Consensus Dynamics

The first end-to-end experiment of the lexical-consensus project.

---

## Goal

Test whether a small network of agents (N=3) can converge on a shared
artificial vocabulary for a minimal set of visual categories (3 CIFAR-10
classes) through the three-layer architecture:

```
DINOv2 (perception) -> Lexicon (centroids) -> Consensus (voting)
```

---

## Sequence of Steps

This experiment must be built incrementally. Each step is a separate commit.

### Step 1 — Diagnostic
File: `diagnostic.py`

Verify that CIFAR-10 categories form separable clusters in DINOv2 embedding
space. If they don't, the experiment cannot work — we need to know this
before investing time in the full multi-agent loop.

**Success criterion:** silhouette score > 0.3 across selected categories.

### Step 2 — Single Agent Learning
File: `step_2_single_agent.py` (to be created)

Verify that a single LearnerAgent can:
- Receive tutor instructions for 3 Carroll labels on 3 CIFAR-10 categories
- Correctly classify new images of the same categories
- Pass Condition 2 (inverse grounding) on a small test set

This isolates the agent architecture before adding consensus complexity.

### Step 3 — Multi-Agent Consensus
File: `run.py` (to be created)

Full experiment: 3 agents, 3 categories, 25 rounds, with full consensus loop.

### Step 4 — Metrics and Analysis
Files: in `notebooks/`

Compute convergence metrics, generate plots, document findings.

---

## Configuration

To be defined in `config.yaml` once Step 3 is reached. Initial parameters:

| Parameter | Value |
|---|---|
| Number of agents | 3 |
| Carroll labels | vorpal, slithy, mimsy |
| CIFAR-10 categories | TBD by diagnostic |
| Images per category | 50 |
| Seed images per category | 5 |
| Rounds | 25 |
| Convergence threshold | 0.70 |
| Rejection threshold | 0.20 |
| Random seed | 42 |

---

## Expected Outputs

- `results/exp_001_diagnostic.json` — cluster separability metrics
- `results/exp_001_baseline_runN.json` — per-run results (N = seed)
- `results/exp_001_baseline_summary.json` — aggregated across runs
- `notebooks/exp_001_analysis.ipynb` — plots and discussion
