#!/usr/bin/env bash
# Runs the v2c post hoc analyses (B1-B4) from the frozen files. Skips an output that already exists unless FORCE=1.
set -euo pipefail
H="$(cd "$(dirname "$0")/.." && pwd)"; REPO="${REPO:?set REPO to a checkout of this repository at commit a945ed2 (see paper_v2/README.md)}"; PY="${PY:-python3}"
RAW="${RAW:?set RAW to all_raw_A.csv of Lewis & Frank 2018 (OSF md5 e42712b8a15187983620a1d9f6edb5e9)}"
KEY="${KEY:?set KEY to experiment_key.csv from XTMEM commit 220ed17}"
IDS="R0_dinov2s O1_dinov2b R1_cvcl E1_siglip O2_openclip E2_vitsup E3_rn50sup E4_mae F1_rn50rand0 F2_pixels"
mkdir -p "$H/data" "$H/out" "$H/logs"; cd "$H"
prepare_tables=${FORCE:-0}
for id in $IDS; do
  if [ ! -s "data/ex_long_$id.csv" ]; then
    prepare_tables=1
    break
  fi
done
if [ "$prepare_tables" = 1 ] || [ ! -s data/p01_check.json ]; then
  stage=$(mktemp -d data/.p01.XXXXXX)
  trap 'rm -rf -- "$stage"' EXIT
  "$PY" scripts/p01_prep_exemplar.py "$REPO" "$RAW" "$KEY" "$stage" > logs/p01.log 2>&1
  mv "$stage"/ex_long_*.csv data/
  mv "$stage/p01_check.json" data/
fi
jobs_list=""
for id in $IDS; do
  [ -f out/p03_$id.csv ] && [ "${FORCE:-0}" != 1 ] || jobs_list+="p03_exemplar.R $id"$'\n'
  [ -f out/p02_$id.csv ] && [ "${FORCE:-0}" != 1 ] || jobs_list+="p02_image_slopes.R $id"$'\n'
done
printf "%s" "$jobs_list" | xargs -r -P "${NPAR:-4}" -L 1 bash -c 'Rscript scripts/$0 scripts/lib_ph.R data out $1 > logs/${0%.R}_$1.log 2>&1'
"$PY" scripts/p04_h2_power.py "$REPO" out > logs/p04.log 2>&1
"$PY" scripts/p05_sup_distances.py data out out/fig_sup_distances.pdf > logs/p05.log 2>&1
echo "post hoc outputs complete"
