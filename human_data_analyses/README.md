# Human-data analyses (secondary data)

Pre-specified analyses that compare a few-shot learner's word-learning results with **secondary human data collected by other teams**: THINGS odd-one-out, ImageNet-HSJ, CIFAR-10H, Lewis & Frank (2018), and Glasgow Norms. We collected no new human data. All third-party data remain the work of their original authors (see `data/README.md`).

## Contents
- `plan_analisis_datos_humanos_FROZEN_2026-10-30_ANON.md`, `PLAN_HASH.txt`: the analysis plan, frozen before any analysis was run (H1–H6), with its SHA-256.
- `DESVIACIONES.md`: registered deviations from the plan.
- `RESULTADOS.md`: results per hypothesis.
- `REVISION_DETALLADA.md`: audit.
- `COMPARACION_CONCLUSIONES.md`: comparison with the source papers' conclusions.
- `EXPLORATORIO_categorias_amplias.md`: **exploratory, post hoc**.
- `scripts/`: Python and R. Run in numeric order, with `_paths.py` for local paths.
- `out/`: aggregated results (JSON and logs).
- `figs/`, including `figs/comparativos/` and `figs/comparativos/exploratorio/`.
- `data/`: processed, redistributable inputs, plus `README.md` and `download_data.sh` for the rest.

The documentation is in Spanish. Code comments and figures are in English.

These analyses are reported in Vera (2026), *Visual Similarity in Adult Word Extension and Artificial Word Learning*, [arXiv:2611.14116](https://arxiv.org/abs/2611.14116).

## Reproduce
1. Python 3.10+ with numpy, pandas, scipy, statsmodels, matplotlib, torch, torchvision. R with lme4 (for H4).
2. `bash data/download_data.sh`
3. Set the environment variables `LC_ROOT`, `MAIN_REPO` (the main project repository, needed for the DINOv2 PerceptionLayer and the exp_007/exp_008 results) and `HDA_CACHE`.
4. Embedding steps: `scripts/h3_embed_c10test.py`, `h5_embed_c100_dino.py`, `h6_embed_cvcl.py`. These take hours on CPU.
5. Analyses: `00_mappings.py` … `10_exploratorio_categorias_amplias.py`, `h4_prep.py`, then the H4 GLMMs in R: `04c_h4_person_re.R` (primary H4 estimate, random effects by person; `04_h4_glmm.R` is the superseded fit grouped by exp_subid) and `04d_h4_firstsession_interaction.R` (exploratory distance × condition interaction; `04b_h4_interaction.R` is superseded). Run `08_holm.py` and `09_figs_comparativos.py` after the GLMMs, since they read `out/h4_person_re.txt`. `out/` already contains our outputs.

Bootstrap seeds are fixed in each script.

## Licenses
- Code: MIT (`LICENSE`).
- Third-party data keep their own licenses (see `data/README.md`), as verified on the primary sources:

| Dataset | License | Source | In this repo |
|---|---|---|---|
| Lewis & Frank (2018), XTMEM | CC BY 4.0 on the authors' OSF project; the GitHub repo has no LICENSE file | https://osf.io/yekhj/ , https://github.com/mllewis/XTMEM | Not redistributed (raw data and all per-participant/per-trial derivatives are rebuilt locally by `download_data.sh`, `h4_prep.py`, `h4_first_session.py`) |
| THINGS odd-one-out (Hebart et al., 2023) | CC BY 4.0, with exceptions for some images (LICENSE.txt) | https://osf.io/f5rn6/ | Mappings and distances only; no images |
| ImageNet-HSJ (Roads & Love, 2021) | CC BY 4.0 | https://osf.io/cn2s3/ | `hsj_trials.csv.gz` (CIFAR-mapped classes) |
| Glasgow Norms (Scott et al., 2019) | Article published under CC BY 4.0 | https://doi.org/10.3758/s13428-018-1099-3 | `aoa_glasgow` column |
| wordfreq (Speer, 2022) | Data CC BY-SA 4.0 (code Apache 2.0) | https://github.com/rspeer/wordfreq | `zipf` computed at runtime by `05_h5.py` (not shipped) |
| CIFAR-10H (Peterson et al., 2019) | CC BY-NC-SA 4.0 | https://github.com/jcpeterson/cifar-10h | 45 class-pair aggregates (`d_c10h`) — **share-alike, non-commercial** |
| CVCL model (Vong et al., 2024) | MIT | https://github.com/wkvong/multimodal-baby | No weights |

- Share-alike: wordfreq frequencies are not shipped (computed at runtime); files containing the `d_c10h` column (`pairs_c10.csv`, `tiers_c10.csv`, `h6_pairs_c10.csv`) are distributed under CC BY-NC-SA 4.0. The MIT license covers only the code.
- `data/h4_first_session.csv` (a per-participant XTMEM derivative) was included in earlier commits and has been removed from the tree; it remains in the git history.
