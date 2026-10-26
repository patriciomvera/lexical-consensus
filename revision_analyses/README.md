# Robustness analyses (r01–r07b)

Additional analyses run on the released code and committed results. They do not modify any existing file;
all outputs go to `revision_analyses/out/`. Run every script **from the repository root**, e.g.
`python revision_analyses/r02_multiplicity.py`.

Requirements: the packages in `requirements.txt` plus `nltk` with the `wordnet`, `omw-1.4` and `wordnet_ic`
corpora (`python -c "import nltk; [nltk.download(c) for c in ('wordnet','omw-1.4','wordnet_ic')]"`).

| Script | What it does | Inputs | Outputs (in `out/`) |
|---|---|---|---|
| `r01_exp008_semantics_bootstrap.py` | exp_008: checks that 1 − Wu-Palmer with the frozen WordNet mapping reproduces the committed d_sem matrix; reproduces the P+S regression; pairs bootstrap (10,000 resamples) of β_perc, β_sem, Q3∪Q4 Spearman ρ and Q4−Q3 median difference; re-runs the regression under alternative semantic measures (path, LCH, Lin/JCN with Brown IC) and a sense-corrected mapping for 21 classes. | `results/exp_008_dissociation/phase1–3` | `r01_exp008_semantics_bootstrap.json`, `r01_wordnet_sense_corrections.csv` |
| `r02_multiplicity.py` | Holm and Benjamini–Hochberg corrections over the paired Wilcoxon families of exp_007b_extension_clean (all 96 tests; the 4 hard-pool exemplar-vs-centroid tests) and over the exp_008 centroid tests. Exact p values are recomputed from the per-episode ledger. | `results/exp_007b_extension_clean/`, `results/exp_008_dissociation/phase4/` | `r02_multiplicity.json`, `r02_paired_tests_adjusted.csv` |
| `r03_build_cache.py` | Rebuilds the DINOv2-small embedding cache `.cache/embeddings_exp007/cifar_all10.npz` through the repository's own `load_or_build_embeddings` (CIFAR-10 download on first run). Needed by r04 and r05. | CIFAR-10 | `.cache/` |
| `r04_run_clean_seed.py` | Re-runs `experiments/exp_007b_extension_clean/run_clean.py` unchanged except for the master seed and output path. `episodes` mode changes episode sampling and learner seeds; `full` mode also re-draws and re-encodes the 200-image-per-class pool. Usage: `python revision_analyses/r04_run_clean_seed.py <seed> [episodes\|full]`. | r03 cache | `out/r04_seeds/<mode>_seed_<s>/` (large; not committed) |
| `r05_tier_distances.py` | Recomputes the 10×10 CIFAR-10 inter-category distances, checks them against `results/exp_007b_extension/concept_gate.csv`, and tabulates all 45 pairs by tier band. | r03 cache | `r05_tier_distances.json`, `r05_tier_pairs.csv` |
| `r06_seed_summary.py` | Summarizes the r04 runs: C1 gradient monotonicity and hard-pool C2 exemplar-vs-centroid gaps per seed, with mean ± SD across seeds. | r04 outputs | `r06_seed_summary.json` |
| `r07_partial_corr_q34.py` | exp_008: partial (Spearman and Pearson) correlation of C1 accuracy with semantic distance given perceptual distance, within Q3∪Q4 (original and sense-corrected quadrants) and over all 100 pairs. | `results/exp_008_dissociation/` | `r07_partial_corr_q34.json`, `r07_partial_corr_q34.log` |
| `r07b_observed_q4q3.py` | exp_008: observed Q4−Q3 median C1 difference for the centroid learner. | `results/exp_008_dissociation/phase3/` | `r07b_observed_q4q3.log` (stdout) |

Seeds used for r04/r06: 42 (reproduces the committed ledger for all non-random learners), 0, 1, 2, 3.
Reproduce r04→r06 with:

```bash
python revision_analyses/r03_build_cache.py
for s in 42 0 1 2 3; do python revision_analyses/r04_run_clean_seed.py $s episodes; done
for s in 0 1 2 3; do python revision_analyses/r04_run_clean_seed.py $s full; done
python revision_analyses/r06_seed_summary.py
```
