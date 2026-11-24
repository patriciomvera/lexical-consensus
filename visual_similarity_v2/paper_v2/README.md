# Visual Similarity, version 2: manuscript, post hoc analyses and build

This directory contains everything used to produce version 2 of *Visual Similarity in Adult Word Extension and Artificial Word Learning* ([arXiv:2611.14116](https://arxiv.org/abs/2611.14116)): the post hoc analyses of Appendix E, the scripts that write every number, table and figure of the manuscript, the step-by-step text build, the LaTeX source as submitted to arXiv, the final PDF and the two automated external reviews of an earlier draft.

Use a clean checkout of [`a945ed2`](https://github.com/patriciomvera/lexical-consensus/commit/a945ed2cbdd9365fe346b262763f0262eaf686ea) for `REPO`. Some scripts read frozen inputs directly from that checkout and others use `git show` at the same commit. The internally frozen plans and their outputs are in [`../encoder_equivalence_v2/`](../encoder_equivalence_v2/) and [`../agents_xtmem_v2/`](../agents_xtmem_v2/). See [`../README.md`](../README.md) and [`../RESULTS.md`](../RESULTS.md).

## Contents

| Path | What it is |
| --- | --- |
| [`Lexical_Consensus_v2.pdf`](Lexical_Consensus_v2.pdf) | Final manuscript (34 pages). The text is identical to a compilation of `arxiv_source/`. |
| [`arxiv_source/`](arxiv_source/) | LaTeX source as submitted to arXiv (comments stripped with `scripts/strip_comments.py`; includes `main.bbl`, biblatex 3.3). Compiles with `pdflatex main.tex` (×3), or with `biber` from `references.bib`. |
| [`arxiv_abstract.txt`](arxiv_abstract.txt) | Shorter abstract used for the arXiv abstract field (the field allows at most 1,920 characters). |
| [`posthoc_v2c/`](posthoc_v2c/) | Post hoc analyses B1–B4 (exploratory, outside every frozen plan): scripts, outputs, logs and the deviation log [`DESVIACIONES_posthoc_v2c.md`](posthoc_v2c/DESVIACIONES_posthoc_v2c.md). |
| [`scripts/`](scripts/) | Scripts that write the LaTeX macros, tables and figures (`generated/`, `manuscrito_v2f/generated/`, `manuscrito_v2f/tables/`). Each asserts its checks and records the source of every value. |
| `v2c_build/` … `v2f_build/` | Text build: a snapshot of the previous version plus a script that applies the edits of each round with guards (every replaced passage must exist exactly once, untouched passages are compared). |
| `manuscrito_v2c/` … `manuscrito_v2f/` | `build.sh` of each round. Only `manuscrito_v2f/build.sh` (the final version) is meant to be run; it also checks, by sha256, that the v2c, v2d and v2e texts are regenerated exactly. `manuscrito_v2f/` also holds the static inputs (figures, bibliography, v1 number macros, turtle photograph and its attribution). |
| [`freeze_copy/DIFF_agentes_propuesta.patch`](freeze_copy/DIFF_agentes_propuesta.patch) | The copy of the historical patch whose sha256 (`21125d6a…f87d`) is listed in `FREEZE_agentes_v2.txt`. It differs from the published file only by a 5-line header (Appendix A, "Checksum of a historical patch"). |
| [`reviews/`](reviews/) | Automated external review of the earlier draft (DeepReview). |

## Post hoc analyses (Appendix E)

Added after two automated external reviews of an earlier draft (paperreview.ai and DeepReview) and approved by the author on 24-11-2026. Seed 42 throughout; models that do not meet the convergence criterion of `lib_glmm.R` are reported as not converged. Results are reported whether or not they are favorable.

| Analysis | Script | Output | Manuscript |
| --- | --- | --- | --- |
| P01: per-trial nearest-exemplar and mean-exemplar distances and teaching-set dispersion for the 10 representations (centroid distance reproduces the frozen `h4_long_<ID>.csv`, max abs. difference 0) | `scripts/p01_prep_exemplar.py` | `data/ex_long_<ID>.csv` (local, see below), [`data/p01_check.json`](posthoc_v2c/data/p01_check.json) | Input to B1–B3 |
| B1: by-image random slopes for distance in the 8-item positive control; superordinate-only model with image slopes | `scripts/p02_image_slopes.R` | `out/p02_<ID>.csv` | Table 13 (frozen model refit, by-image slopes, SD of image slopes), §2.4, §6, §7.1, Appendix E |
| B2: mean- and nearest-exemplar models against the centroid (AIC, slopes); condition given distance | `scripts/p03_exemplar.R` | `out/p03_<ID>.csv` | Table 13 (ΔAIC columns, χ² for condition), §2.4 (14,755 = smallest χ²(3) for condition given mean-exemplar distance, supervised ResNet-50) |
| B3: distances of superordinate test items by teaching condition | `scripts/p05_sup_distances.py` | `out/fig_sup_distances.pdf`, `out/p05_*.csv` | Figure 8 |
| B4: H2 estimates by representation and minimum detectable effect (n = 61, 32) | `scripts/p04_h2_power.py` | `out/p04_h2_estimates.csv`, `out/p04_h2_mde.csv` | Table 14 |

Table 13 and Table 14 are written by `scripts/make_numbers_v2c.py` from these outputs (`tables/tab_posthoc_v2c.tex`, `tables/tab_h2post_v2c.tex`), and the Table 13 column header and note are set by `v2f_build/fix_appendix_tables_v2f.py`. The intervals of the frozen model in Table 13 come from the post hoc refit and may differ from Table 10 in the second decimal.

## How to reproduce

Requirements used: Python 3.13 with NumPy 2.5, pandas 3.1, SciPy 1.18, Matplotlib 3.11; R 4.5 with lme4 1.1-37; TeX Live 2025 with latexmk and biber 2.20.

```sh
# 1. A checkout of this repository at the frozen commit (make_forest_plot.py asserts HEAD = a945ed2)
git worktree add ../lc_a945ed2 a945ed2
export REPO="$(cd ../lc_a945ed2 && pwd)" PY=python3

# 2. Third-party adult data (not redistributed here): Lewis & Frank (2018), CC BY 4.0
#    all_raw_A.csv from OSF https://osf.io/yekhj/ (https://osf.io/download/qyzcg/, md5 e42712b8a15187983620a1d9f6edb5e9)
#    experiment_key.csv from https://github.com/mllewis/XTMEM at commit 220ed17ec2b85d30a4053711209252fdbc3cb166
#    (see human_data_analyses/data/download_data.sh)
export RAW=/path/to/all_raw_A.csv KEY=/path/to/experiment_key.csv

# 3. Post hoc analyses. Regenerate the per-trial tables if any are missing or empty.
#    Existing R fits are reused. Set FORCE=1 to refit them as well.
bash visual_similarity_v2/paper_v2/posthoc_v2c/scripts/run_posthoc.sh

# 4. Numbers, tables, figures, text and PDF of the final version (all checks asserted)
bash visual_similarity_v2/paper_v2/manuscrito_v2f/build.sh
```

`build.sh` regenerates every macro and table, rebuilds the texts of v2c, v2d and v2e from their snapshots and checks their sha256, applies the v2f edits, compiles and fails on overfull boxes, undefined references, "??" or Type 3 fonts. A clean run on a copy of this directory reproduced every generated file byte for byte and a PDF whose text is identical to `Lexical_Consensus_v2.pdf`. To produce the arXiv bundle, copy the files listed in `manuscrito_v2f/main.fls` and run `python scripts/strip_comments.py <bundle_dir>`.

Internal message identifiers and an assistant-name guard were removed from comments in the build snapshots before publication; the sha256 values checked by the build scripts are those of the published snapshots.

## Data and licenses

- Adult trial data: Lewis & Frank (2018), XTMEM, CC BY 4.0. The per-trial post hoc tables (`ex_long_<ID>.csv`, about 21 MB each) are not stored in git; step 3 regenerates them deterministically.
- Since 24-11-2026 the per-trial tables are also on Hugging Face, in [`MrPatoVera/lexical-consensus-data`](https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data) under `v2_encoders/` (CC BY 4.0, Parquet with an added `representation` column, values otherwise identical): the frozen centroid tables `h4_long_<ID>` in `v2_encoders/trials_centroid/` (config `v2_trials_centroid`) and the post hoc exemplar tables `ex_long_<ID>` in `v2_encoders/trials_exemplar/` (config `v2_trials_exemplar`). The same folder holds the XTMEM image embeddings, the adult and agent results, the matched-stimulus outputs and the post hoc outputs. Agent result files that contain values derived from CIFAR-10H (`H_<ID>.json` and six region files) are in [`MrPatoVera/lexical-consensus-data-cifar10h`](https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data-cifar10h) under `v2_encoders/` (CC BY-NC-SA 4.0).
- THINGS, ImageNet-HSJ (CC BY 4.0) and CIFAR-10H (CC BY-NC-SA 4.0): see `human_data_analyses/data/README.md` and the Hugging Face datasets [`MrPatoVera/lexical-consensus-data`](https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data) and [`MrPatoVera/lexical-consensus-data-cifar10h`](https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data-cifar10h).
- Figure 6 photograph: *Florida Box Turtle* by Jonathan Zander (Digon3), CC BY-SA 3.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Florida_Box_Turtle_Digon3.jpg); illustrative only, not the experimental stimulus. See `manuscrito_v2f/figs/ATTRIBUTION_Florida_Box_Turtle_Digon3.txt`.
