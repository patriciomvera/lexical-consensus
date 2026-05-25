# results/

This directory holds the output artifacts from each experiment run.
It is **gitignored** except for this file — results are not committed
to version control. They are reproducible from code + config.

---

## Directory Convention

Each experiment, when run, creates a subdirectory named after it:

```
results/
  exp_000_embedding_separability/
  exp_001_single_agent_lexicon/
  exp_002_multi_agent_consensus/
  ...
```

The directory name must match the corresponding experiment directory
in `experiments/` exactly.

---

## Required Artifacts per Experiment

Every experiment directory must contain at minimum:

| File | Description |
|---|---|
| `config.yaml` | Complete configuration used for this run (all parameters, seeds, model versions). Must be sufficient to reproduce the run exactly. |
| `metrics.json` | All quantitative results: scores, pass/fail flags, per-label breakdowns, confidence intervals where applicable. |
| `report.md` | Human-readable summary: what was tested, what the results mean, PASS or FAIL verdict with justification. |

Additional artifacts (plots, centroid dumps, per-round logs) are
encouraged and should be named descriptively.

---

## Why Not Git?

Results are gitignored for three reasons:
1. Binary files (plots, numpy arrays) bloat the repository
2. Results are derived from code — committing them creates a
   maintenance burden (code changes must be re-run and re-committed)
3. The paper will cite specific run IDs, not file diffs

The config.yaml inside each results directory IS the record of what
was run. If results need to be shared, archive the whole directory.

---

## Versioning

If the same experiment is run multiple times (e.g. with different seeds
for reproducibility confirmation), use a suffix:

```
results/
  exp_002_multi_agent_consensus/          <- canonical run (seed=42)
  exp_002_multi_agent_consensus_seed7/    <- confirmation run
  exp_002_multi_agent_consensus_seed13/   <- confirmation run
```

Aggregated results across seeds go in a separate `_summary/` directory:

```
results/
  exp_002_multi_agent_consensus_summary/
```
