#!/usr/bin/env bash
# WEEKEND RUN (Sat 21 / Sun 22-Nov-2026), ONLY after FREEZE_encoders_v2.txt exists and verifies. Order = plan §10.3.
# Usage: bash run_weekend.sh saturday | sunday     (each step logs to encoder_equivalence/logs/; stops at the first error, except S6)
# The 32 h CPU cap of plan §9.2 is NOT enforced here: watch it by hand (log timestamps) and cut S1 if exceeded (S9 now costs no extra fits) (REVISION3 minor).
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd); B=$(dirname "$S"); H=/workspace/lexical-consensus-openmind/human_data_analyses
PY=/workspace/rv_venv/bin/python; R=/tmp/renv/bin/Rscript; DATA=$B/data; RES=$B/results; L=$B/logs
KEY=$H/data/xtmem/experiment_key.csv; RAW=$H/data/xtmem/all_raw_A.csv; V1LONG=$H/data/h4_long.csv; FS=$H/data/h4_first_session.csv; EMB=$B/out; GRP=$B/out
if [ -n "${WEEKEND_SYNTH_DIR:-}" ]; then   # ONLY for the synthetic rehearsal (tests/): every path points to synthetic data
  T=$WEEKEND_SYNTH_DIR; case "$T" in /tmp/synth*) ;; *) echo "synthetic dir must be /tmp/synth*"; exit 2;; esac
  DATA=$T/w_data; RES=$T/w_results; L=$T/w_logs; KEY=$T/raw/experiment_key_SYNTH.csv; RAW=$T/raw/all_raw_A_SYNTH.csv
  V1LONG=$T/v1like/h4_long_SYNTH_v1code.csv; FS=$T/raw/h4_first_session_SYNTH.csv; EMB=$T/emb; GRP=$T/w_results; fi
P10="R0_dinov2s O1_dinov2b R1_cvcl E1_siglip O2_openclip E2_vitsup E3_rn50sup E4_mae F1_rn50rand0 F2_pixels"
EXTRA="E4s_mae_cls F1_rn50rand1 F1_rn50rand2"
[ -n "${WEEKEND_SYNTH_DIR:-}" ] && [ -n "${WEEKEND_SYNTH_P10:-}" ] && P10=$WEEKEND_SYNTH_P10   # synthetic rehearsal only: fewer encoders to save time
mkdir -p $DATA $RES $L
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a $L/run_weekend.log; }
par3() { # par3 <out_prefix> <items> <variant> <suffix|-> <first_session|-> <fit:ladder|L1..L4> <ids...>: <=3 R processes in parallel (§8.13)
  local pre=$1 it=$2 va=$3 su=$4 fs=$5 fi=$6; shift 6; local ids=($@) n=$# pids=()
  for g in 0 1 2; do local sel=(); for ((i=g; i<n; i+=3)); do sel+=(${ids[$i]}); done; [ ${#sel[@]} -gt 0 ] || continue
    $R $S/05_twostage_stage1.R $DATA $KEY ${pre}_g$g.csv $it $va $su $fs $fi "${sel[@]}" > $L/$(basename $pre)_g$g.log 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || { log "stage-1 process failed ($pre)"; exit 1; }; done; }
step0() {
  if [ -n "${WEEKEND_SYNTH_DIR:-}" ]; then log "0. SYNTHETIC rehearsal: freeze/sha checks skipped"; return; fi
  log "0. checks"; [ -f $B/FREEZE_encoders_v2.txt ] || { log "NO FREEZE MANIFEST: stop"; exit 1; }
  $PY $S/09_freeze.py --verify | tee -a $L/run_weekend.log
  [ -x $R ] && $R -e 'suppressMessages(library(lme4)); cat("lme4", as.character(packageVersion("lme4")), "\n")' || { log "/tmp/renv missing: rebuild R env before continuing"; exit 1; }
  [ -d /tmp/XTMEM/experiments/image_stim/images ] || log "WARNING /tmp/XTMEM missing: needed only for S6 (01b); regenerate with $H/data/download_data.sh"
  [ -f $RAW ] && [ -f $V1LONG ] || { log "v1 data missing: run $H/data/download_data.sh and scripts/h4_prep.py"; exit 1; }
  sha256sum $V1LONG | tee -a $L/run_weekend.log | grep -q '^d9d30f9d.*394dd1 ' || { log "h4_long.csv sha256 != d9d30f9d…394dd1: stop"; exit 1; }
  grep MemAvailable /proc/meminfo | tee -a $L/run_weekend.log; }
select_structure() {  # plan §5.2: ONE common rung for all encoders (STRUCTURE_SCOPE=common, Patricio 18-11 12:14), chosen by convergence only
  local files=($RES/stage1_8_R0.csv $RES/stage1_8_rest_g*.csv $RES/stage1_24_g*.csv) k rc
  for k in 1 2 3 4 5 6 7 8; do
    rc=0; $PY $S/05b_select_structure.py $RES/structure "${files[@]}" > $L/05b_select_iter$k.log 2>&1 || rc=$?
    [ $rc -eq 0 ] && { cat $RES/structure_status.json | tee -a $L/run_weekend.log; return 0; }
    [ $rc -eq 4 ] && { cat $RES/structure_DEVIATION.txt | tee -a $L/run_weekend.log; log "DEVIATION: no common stage-1 structure (plan §5.2): stop, no fallback"; exit 1; }
    [ $rc -eq 3 ] || { cat $L/05b_select_iter$k.log; log "05b failed"; exit 1; }
    local lines=(); mapfile -t lines < <($PY -c "import pandas as p; d=p.read_csv('$RES/structure_needs.csv')
for (i, r), g in d.groupby(['items', 'rung']): print(i, r, ' '.join(sorted(set(g.encoder))))")
    for ln in "${lines[@]}"; do set -- $ln; local it=$1 rg=$2; shift 2; log "   pass 2 (iter $k): items=$it rung=$rg: $*"
      par3 $RES/stage1_pass2_${it}_${rg}_i$k $it primary - - $rg "$@"; files+=($RES/stage1_pass2_${it}_${rg}_i${k}_g*.csv); done
  done; log "05b: selection did not finish after 8 iterations: stop"; exit 1; }
rung8() {  # common 8-item rung fixed on Saturday (plan §5.2). REVISION3b: missing/null -> log the deviation and fail (no silent L4)
  local r; r=$($PY -c "import json,sys; r=json.load(open('$RES/structure_status.json'))['8']['common_rung']; print(r) if r in ('L1','L2','L3','L4') else sys.exit(4)" 2>/dev/null) \
    || { log "DEVIATION: no common 8-item stage-1 structure in $RES/structure_status.json (plan §5.2): stop, no fallback" >&2; return 1; }
  echo "$r"; }
saturday() {
  step0
  log "1. h4_prep_encoders + pipeline control (G0)"
  $PY $S/h4_prep_encoders.py --raw $RAW --key $KEY --emb-dir $EMB --out-dir $DATA $P10 $EXTRA > $L/01_prep.log 2>&1
  $PY $S/h4_prep_encoders.py --check $DATA/h4_long_R0_dinov2s.csv $V1LONG $RES/pipeline_control.json > $RES/pipeline_control.log || { cat $RES/pipeline_control.log; log "G0: pipeline control FAILED: stop everything"; exit 1; }
  cat $RES/pipeline_control.log | tee -a $L/run_weekend.log
  log "2. 02_equivalence (descriptive table + geometry rule at cuts 0.3/0.5/0.7 + calibration per cut)"; if [ -n "${WEEKEND_SYNTH_DIR:-}" ]; then $PY $S/02_equivalence.py $DATA $GRP $EMB > $L/02.log 2>&1; else $PY $S/02_equivalence.py $DATA > $L/02.log 2>&1; fi
  log "3. timing: R0, 8 items, ladder pass 1 (secs column) -> update budget only"; $R $S/05_twostage_stage1.R $DATA $KEY $RES/stage1_8_R0.csv 8 primary - - ladder R0_dinov2s > $L/stage1_8_R0.log 2>&1
  log "4. stage 1, 8 items, ladder pass 1"; par3 $RES/stage1_8_rest 8 primary - - ladder ${P10#R0_dinov2s }
  log "5. stage 1, 24 items, ladder pass 1"; par3 $RES/stage1_24 24 primary - - ladder $P10
  log "5b. common structure (05b, convergence only; pass 2 fits where needed)"; select_structure
  log "6. positive control (before regions: 08 uses it)"; $R $S/07_pooled_glmm.R $DATA $KEY $RES/poscontrol.csv poscontrol 8 - $P10 > $L/poscontrol.log 2>&1
  log "7. stage 2 (primary = common rung; per_encoder_rung and S9_mixed descriptive) + regions (primary only)"
  $PY $S/06_stage2_permutation.py $RES/stage2_primary.csv $RES/structure_primary.csv $RES/structure_per_encoder.csv $RES/structure_S9_mixed.csv
  $PY $S/08_regions.py --stage2 $RES/stage2_primary.csv --poscontrol $RES/poscontrol.csv --grouping $GRP/equivalence_grouping.json --pipeline $RES/pipeline_control.json --out-dir $RES/regions
  log "Saturday done"; }
s6() {  # S6, common preprocessing; run in a subshell at the very end of Sunday. errexit is off inside "( s6 ) || ...", hence the && chain.
  [ -n "${WEEKEND_SYNTH_DIR:-}" ] && [ -n "${WEEKEND_SYNTH_S6_FAIL:-}" ] && return 1   # rehearsal of the non-blocking failure
  local RG; RG=$(rung8) || return 1                                                       # REVISION3b: no common rung -> S6 fails (logged)
  if [ -n "${WEEKEND_SYNTH_DIR:-}" ]; then for i in $P10; do cp $EMB/emb_$i.npz $EMB/emb_${i}_s6common.npz || return 1; done
  else $PY $S/01b_embed_common.py > $L/01b.log 2>&1 || return 1; fi
  $PY $S/h4_prep_encoders.py --raw $RAW --key $KEY --emb-dir $EMB --out-dir $DATA --suffix _s6common $P10 >> $L/01_prep.log 2>&1 &&
  par3 $RES/stage1_S6 8 primary _s6common - $RG $P10 &&
  $PY $S/06_stage2_permutation.py $RES/stage2_S6.csv $RES/stage1_S6_g*.csv; }
sunday() {
  step0
  log "8. S1"; $R $S/07_pooled_glmm.R $DATA $KEY $RES/S1_24.csv S1 24 - $P10 > $L/S1_24.log 2>&1; $R $S/07_pooled_glmm.R $DATA $KEY $RES/S1_8.csv S1 8 - $P10 > $L/S1_8.log 2>&1
  log "S2"; $R $S/07_pooled_glmm.R $DATA $KEY $RES/S2.csv S2 8 - $P10 > $L/S2.log 2>&1
  log "S3"; $R $S/07_pooled_glmm.R $DATA $KEY $RES/S3.csv S3 8 - $P10 > $L/S3.log 2>&1
  local RG; RG=$(rung8) || exit 1; log "S4/S5/S7/S8 at the common 8-item rung $RG (fixed on Saturday by 05b; not re-selected)"
  log "S4"; par3 $RES/stage1_S4 8 first_session - $FS $RG $P10
  log "S5"; par3 $RES/stage1_S5 8 rank - - $RG $P10
  log "S7"; $R $S/05_twostage_stage1.R $DATA $KEY $RES/stage1_S7.csv 8 primary - - $RG F1_rn50rand1 F1_rn50rand2 > $L/S7.log 2>&1
  log "S8"; $R $S/05_twostage_stage1.R $DATA $KEY $RES/stage1_S8.csv 8 primary - - $RG E4s_mae_cls > $L/S8.log 2>&1
  $PY $S/06_stage2_permutation.py $RES/stage2_sensitivities.csv $RES/stage1_S4_g*.csv $RES/stage1_S5_g*.csv $RES/stage1_S7.csv $RES/stage1_S8.csv
  log "S9 (descriptive; Patricio 12:12): mixed fit = first converged rung per experiment, already in stage2_primary.csv (variant S9_mixed); no extra fits"
  $PY $S/08_regions.py --stage2 $RES/stage2_primary.csv --poscontrol $RES/poscontrol.csv --grouping $GRP/equivalence_grouping.json \
      --pipeline $RES/pipeline_control.json --out-dir $RES/regions_with_S9 --stage2-s9 $RES/stage2_primary.csv
  log "9. Family A: F0 (Mb condition:order coefficients are in S1_*.csv)"
  $R $S/07_pooled_glmm.R $DATA $KEY $RES/F0_24.csv F0 24 - R0_dinov2s > $L/F0_24.log 2>&1; $R $S/07_pooled_glmm.R $DATA $KEY $RES/F0_8.csv F0 8 - R0_dinov2s > $L/F0_8.log 2>&1
  log "S6 (last; REVISION3 I2: its failure does not stop anything else)"; ( s6 ) || log "S6 FAILED (non-blocking): see $L/01b.log; reported as a deviation"
  log "10. write RESULTADOS_encoders.md + DESVIACIONES entries by hand from results/; notify Evaristo. Nothing is published."; }
"${1:?saturday|sunday}"
