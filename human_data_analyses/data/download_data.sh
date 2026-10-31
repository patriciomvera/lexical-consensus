#!/usr/bin/env bash
# Fetch third-party data that is NOT redistributed in this repository.
# Places files under $LC_ROOT/human_data_scouting and $LC_ROOT/XTMEM (default LC_ROOT = parent of this repo).
set -euo pipefail
ROOT="${LC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"; SC="$ROOT/human_data_scouting"; mkdir -p "$SC"
# CIFAR-10H (CC BY-NC-SA 4.0) - Peterson et al. 2019
curl -L -o "$SC/cifar10h-counts.npy" https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-counts.npy
# Lewis & Frank 2018 / XTMEM (data and stimuli; images are not redistributed here). Analyses used commit 220ed17ec2b85d30a4053711209252fdbc3cb166
[ -d "$ROOT/XTMEM" ] || git clone https://github.com/mllewis/XTMEM "$ROOT/XTMEM"
git -C "$ROOT/XTMEM" checkout 220ed17ec2b85d30a4053711209252fdbc3cb166
mkdir -p "$(dirname "$0")/xtmem"
echo "Copy all_raw_A.csv and experiment_key.csv from $ROOT/XTMEM (see data/README.md) into data/xtmem/, then run scripts/h4_prep.py."
# THINGS odd-one-out / SPoSE (CC BY 4.0) and ImageNet-HSJ (CC BY 4.0): processed versions are included in data/.
# Raw sources, only needed to rebuild data/map_*.json and data/hsj_trials.csv.gz:
#   THINGS: https://osf.io/f5rn6/   (concepts-metadata_things.tsv, spose_embedding_66d_sorted.txt)
#   ImageNet-HSJ v0.3.1: https://osf.io/download/zg9hx/
# CIFAR-10/100 and ImageNet images: torchvision.datasets (download=True); not redistributed.
