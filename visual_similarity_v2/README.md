# Visual Similarity, version 2

This directory collects the evidence for the second version of *Visual Similarity in Adult Word Extension and Artificial Word Learning*. It contains two internally frozen analyses and an exploratory comparison in which adults and agents saw the same images. The adult data throughout come from Lewis and Frank (2018). The study reuses the same 520 adults across encoders.

[Results and conclusions in English](RESULTS.md) · [Three-page comparison](exhibits/adults_and_agents_by_encoder.pdf) · [Manuscript table in LaTeX](exhibits/word_extension_table.tex)

## Evidence

| Analysis | What varies | Where to find it |
| --- | --- | --- |
| Adult multi-encoder reanalysis | The encoder used to estimate visual distance for a fixed set of human choices | [Plan, scripts, data and logs](encoder_equivalence_v2/) · [Positive-control fits](encoder_equivalence_v2/encoder_equivalence/results/poscontrol.csv) · [Primary decisions](encoder_equivalence_v2/encoder_equivalence/results/regions/) |
| Original CIFAR agent tasks | The frozen encoder used by each lexical learner | [Scripts, tests and logs](encoder_equivalence_v2/encoder_equivalence/agents/) · [Per-encoder outputs](encoder_equivalence_v2/encoder_equivalence/agents/out/results/) · [Decision tables](encoder_equivalence_v2/encoder_equivalence/agents/out/results/regions/) |
| Exploratory matched-stimulus task | Agent decisions under the same teaching sets and test photographs shown to adults | [English protocol](agents_xtmem_v2/protocol.md) · [Item decisions, rankings and three-word task](agents_xtmem_v2/outputs/) · [Input hashes](agents_xtmem_v2/outputs/run.json) |

## Manuscript version 2, post hoc analyses and build

[`paper_v2/`](paper_v2/) contains the final manuscript of version 2 ([PDF](paper_v2/Lexical_Consensus_v2.pdf), [LaTeX source as submitted to arXiv](paper_v2/arxiv_source/)), the exploratory post hoc analyses of Appendix E (Tables 13 and 14, Figure 8), the scripts that write every number, table and figure, and the build that reproduces the PDF from commit `a945ed2`. See [`paper_v2/README.md`](paper_v2/README.md).

## Manuscript exhibits

The PDF has three pages. The first compares extension across teaching conditions. The second shows broad teaching separately for animals, vegetables and vehicles. The third plots the adult and CIFAR distance coefficients on separate scales. The LaTeX table contains the values from the first page and uses the `booktabs` package. In a LaTeX manuscript, include it with `\input{path/to/word_extension_table.tex}`.

| Data behind the exhibits | Columns |
| --- | --- |
| [Teaching conditions](exhibits/teaching_conditions.csv) | Responses or decisions in the numerator and denominator, candidate counts and percentages by condition |
| [Broad teaching by domain](exhibits/broad_teaching_by_domain.csv) | The same quantities stratified by taught domain |
| [Distance effects](exhibits/distance_effects.csv) | Adult log-odds slopes, CIFAR partial correlations and their respective 95% intervals |

To regenerate all exhibits from a repository checkout, run the following from the repository root with Python, NumPy, pandas and Matplotlib installed.

```sh
python visual_similarity_v2/exhibits/build.py
```

## Provenance and reuse

| Source | Git commit | Contribution |
| --- | --- | --- |
| [PR #2](https://github.com/patriciomvera/lexical-consensus/pull/2) | [23e4082](https://github.com/patriciomvera/lexical-consensus/commit/23e4082b0bd7a33889714d0d179fa5a12791fb6c) | Adult analysis, signed-off internal plan, freeze manifest and derived trial data |
| [PR #3](https://github.com/patriciomvera/lexical-consensus/pull/3) | [5d2e9c1](https://github.com/patriciomvera/lexical-consensus/commit/5d2e9c15a87d547b8ee5f95c18e0e96f817d5c6e) | CIFAR addendum, internally frozen scripts, output tables, deviations and logs |
| [Matched-stimulus branch](https://github.com/patriciomvera/lexical-consensus/tree/b3db91c641d6ac34a8650d2184bb24742e8915ab/agents_xtmem_v2) | [b3db91c](https://github.com/patriciomvera/lexical-consensus/commit/b3db91c641d6ac34a8650d2184bb24742e8915ab) | Exploratory item-level analyses with the original learners and an explicit one-word acceptance rule |

The archival packages retain their original languages, filenames and checksums. The freeze manifests and deviation log record what was run, including run-machine paths and external caches. The large CIFAR embedding caches were omitted from the source PR. The repository contains their manifest, the fitted outputs and episode or pair records. The previous root-level path `encoder_equivalence_v2/` resolves here through a compatibility symlink for existing scripts.

The adult data are derived from the Lewis and Frank XTMEM project, distributed under CC BY 4.0. Cite Lewis and Frank and the original paper when using these results. [Original data and license](https://osf.io/yekhj/) · [Stimulus protocol](https://github.com/mllewis/XTMEM/blob/220ed17ec2b85d30a4053711209252fdbc3cb166/experiments/exp1/js/exp1.js) · [Visual Similarity v1](https://arxiv.org/abs/2611.14116)
