# Data

## Included (redistribution allowed)
| File(s) | Content | Source and license |
|---|---|---|
| `hsj_trials.csv.gz`, `map_hsj_v1.json`, `d_hsj*` columns | ImageNet-HSJ trials restricted to the CIFAR-mapped classes; mapping | Roads & Love (2021), CVPR. https://osf.io/cn2s3/ . **CC BY 4.0** |
| `map_things_v1.json`, `d_things*` columns | Distances from the THINGS SPoSE 66-d embedding; mapping | Hebart et al. (2023), eLife. https://osf.io/f5rn6/ . **CC BY 4.0** |
| `aoa_glasgow` column (`h5_*_glasgow.csv`) | Age of acquisition | Scott et al. (2019), Behavior Research Methods, Glasgow Norms. **CC BY 4.0** |
| `zipf` column (`h5_*_glasgow.csv`) | Word frequency | wordfreq (Speer, 2022). **CC BY-SA 4.0**: these columns are under share-alike |
| `d_c10h` column (`pairs_c10.csv`, `tiers_c10.csv`, `h6_pairs_c10.csv`) | Symmetrised class-confusion, **aggregated over 10,000 images into 45 class pairs** | Peterson et al. (2019), ICCV, CIFAR-10H. **CC BY-NC-SA 4.0**: inherited. Only aggregates are included |
| `pairs_*.csv`, `sampled_100.csv`, `tiers_c10.csv`, `cvcl_*.csv`, `exp007b_clean_ledger.jsonl` | Our learner results and DINOv2 and CVCL distances | This project (MIT). CVCL model: Vong et al. (2024), Science, https://github.com/wkvong/multimodal-baby (MIT). Weights not included |

## Excluded (fetch with `download_data.sh`)
| Excluded | Why | How to rebuild |
|---|---|---|
| `xtmem/all_raw_A.csv`, `xtmem/experiment_key.csv` (Lewis & Frank 2018 raw data) | Raw third-party data. Redistribution terms of the OSF/GitHub release not confirmed. Stimulus images have undocumented provenance | https://github.com/mllewis/XTMEM (commit 220ed17), OSF https://osf.io/zcbp7/ . Then `scripts/h4_prep.py` |
| `h4_long.csv` (per-trial Lewis & Frank + DINOv2 distances) | Per-subject copy of the above raw data | `scripts/h4_prep.py` (needs XTMEM images) |
| `h3_c10test_margin.csv` (per-image CIFAR-10H entropy) | Per-item CIFAR-10H derivative (NC-SA). Only the aggregate in `out/h3_results.json` is kept | `scripts/h3_embed_c10test.py`, `scripts/03_h3.py` |
| Overextension pairs (Ferreira Pinto & Xu) | No declared license. Not used in any analysis | — |
| CIFAR-10/100, ImageNet images; embedding caches; CVCL weights | Not ours; large | torchvision; `scripts/*embed*.py` |

## Citations
- Hebart, M. N., et al. (2023). THINGS-data. *eLife*, 12, e82580.
- Roads, B. D., & Love, B. C. (2021). Enriching ImageNet with human similarity judgments and psychological embeddings. *CVPR*.
- Peterson, J. C., Battleday, R. M., Griffiths, T. L., & Russakovsky, O. (2019). Human uncertainty makes classification more robust. *ICCV*.
- Lewis, M., & Frank, M. C. (2018). Still suspicious: The suspicious-coincidence effect revisited. *Psychological Science*, 29(12).
- Xu, F., & Tenenbaum, J. B. (2007). Word learning as Bayesian inference. *Psychological Review*, 114(2).
- Scott, G. G., Keitel, A., Becirspahic, M., Yao, B., & Sereno, S. C. (2019). The Glasgow Norms. *Behavior Research Methods*, 51.
- Speer, R. (2022). wordfreq (v3.0). Zenodo. doi:10.5281/zenodo.7199437
- Vong, W. K., Wang, W., Orhan, A. E., & Lake, B. M. (2024). Grounded language acquisition through the eyes and ears of a single child. *Science*, 383.
