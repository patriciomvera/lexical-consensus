#!/usr/bin/env bash
# AGENTS ADDENDUM RUN (v2 of arXiv 2611.14116), ONLY after encoder_equivalence/FREEZE_agentes_v2.txt exists and verifies.
# Usage: bash run_agents.sh primary | sensitivities      (each step logs to agents/logs/; stops at the FIRST error, like run_weekend.sh)
# primary       = step 0 checks -> parity on the 45 L&F stimuli -> pipeline control 1 (T with R0 from the v1 cache; FAIL = stop: GA0)
#                 -> CIFAR embeddings -> §3.6 dispersion (before any accuracy) -> T, D, D-todos -> H1/H2/H3 -> H4 per encoder -> 14 + fig
# sensitivities = optional, in REVERSE of the pre-declared cut order (§6.9: cut first T-sem full, then H5, F1 seeds 1-2, E4s, T-sem
#                 episodes), so that cutting from the end of the run follows §6.9. D-todos is in 'primary' and is never cut.
# COMPUTE (REVISION5 C5, declared before the freeze): primary ~13-28 core-h, ~3.5-7 h wall, 3 processes (<= 1.5 GB each), local CPU,
#   no GPU/API, $0. CAP: 14 h of wall clock from the start of 'primary' (start time in $L/run_start_epoch). The rate (img/min) is
#   checked automatically 15 min after step 3 starts (< 50 % of the measured rate -> logged; re-estimate by hand). If the cap is
#   exceeded: a deviation is logged and cuts follow risk 9: FIRST the sensitivities (skipped entirely), THEN H4 per encoder (each
#   remaining H4 fit is skipped -> "inconcluso (técnico)" in 14). D-todos and the rest of 'primary' are never cut.
# Backups outside /tmp and outside out/ (a box restart on 22-11 wiped /tmp, out/ and .cache/): embeddings + SHA256SUMS after step 3
#   and results at the end -> $BK; /tmp/XTMEM and /tmp/renv are restored/rebuilt in step 0 if missing.
set -euo pipefail
S=$(cd "$(dirname "$0")" && pwd); AG=$(dirname "$S"); EE=$(dirname "$AG")
PY=/workspace/rv_venv/bin/python; R=/tmp/renv/bin/Rscript; L=$AG/logs; LONG=$EE/data; RES=$AG/out/results
if [ -n "${AGENTS_SYNTH_DIR:-}" ]; then   # ONLY for the synthetic rehearsal (tests/): every data path points to synthetic data
  case "$AGENTS_SYNTH_DIR" in /tmp/synth_agents*) ;; *) echo "synthetic dir must be /tmp/synth_agents*"; exit 2;; esac
  L=$AGENTS_SYNTH_DIR/logs; LONG=$AGENTS_SYNTH_DIR/human/data; RES=$AGENTS_SYNTH_DIR/out/results; fi
P10="R0_dinov2s O1_dinov2b R1_cvcl E1_siglip O2_openclip E2_vitsup E3_rn50sup E4_mae F1_rn50rand0 F2_pixels"
[ -n "${AGENTS_SYNTH_DIR:-}" ] && [ -n "${AGENTS_SYNTH_P10:-}" ] && P10=$AGENTS_SYNTH_P10   # synthetic rehearsal only
DALL_MAX=""; [ -n "${AGENTS_SYNTH_DIR:-}" ] && [ -n "${AGENTS_SYNTH_DALL_MAX:-}" ] && DALL_MAX="--max-pairs $AGENTS_SYNTH_DALL_MAX"
LIMIT=""; [ -n "${AGENTS_SYNTH_DIR:-}" ] && [ -n "${AGENTS_SYNTH_LIMIT:-}" ] && LIMIT="--limit $AGENTS_SYNTH_LIMIT"
export AGENT_TORCH_THREADS=${AGENT_TORCH_THREADS:-2}; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SIMENV="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"   # simulations single-threaded (as r04 / v1 T), 3 processes in parallel
BK=/workspace/agents_respaldo; [ -n "${AGENTS_SYNTH_DIR:-}" ] && BK=$AGENTS_SYNTH_DIR/respaldo
CAP_S=$((14*3600)); [ -n "${AGENTS_SYNTH_DIR:-}" ] && [ -n "${AGENTS_SYNTH_CAP_S:-}" ] && CAP_S=$AGENTS_SYNTH_CAP_S   # synthetic test of the cut
XTMEM_COMMIT=220ed17ec2b85d30a4053711209252fdbc3cb166
mkdir -p $L $RES
cap_exceeded() { [ -f $L/run_start_epoch ] && [ $(( $(date +%s) - $(cat $L/run_start_epoch) )) -ge $CAP_S ]; }
backup() { # backup <src dir> <name>: copy + SHA256SUMS outside /tmp and outside out/
  mkdir -p $BK/$2 && cp -a $1/. $BK/$2/ && (cd $BK/$2 && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS) && log "   backup: $1 -> $BK/$2 ($(wc -l < $BK/$2/SHA256SUMS) files, SHA256SUMS)"; }
rate_check() { # 15 min after step 3 starts: measured img/min per encoder vs the rates measured 22-11 (1 process, 2 threads)
  sleep ${AGENTS_RATE_CHECK_S:-900}; $PY - "$L" <<'PYEOF' 2>&1 | tee -a $L/run_agents.log
import sys, re, glob
exp = {"R0_dinov2s": 70, "O1_dinov2b": 20, "R1_cvcl": 40, "E1_siglip": 26, "O2_openclip": 154, "E2_vitsup": 30, "E3_rn50sup": 52, "E4_mae": 30, "F1_rn50rand0": 51, "F2_pixels": 1600}
last = {}
for f in glob.glob(sys.argv[1] + "/embed_g*.log"):
    for l in open(f):
        m = re.match(r"\[(\S+) (\S+)\]\s+(\d+)/(\d+) images, (\d+) img/min", l)
        if m: last[m.group(1)] = int(m.group(5))
print("[rate check 15 min] img/min:", last or "no chunk finished yet")
for e, v in last.items():
    if e in exp and v < 0.5 * 60 * exp[e]: print(f"[rate check] {e}: {v} img/min < 50 % of {60 * exp[e]}: LOG IT and re-estimate (cap 14 h)")
PYEOF
}
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a $L/run_agents.log; }
par3() { # par3 <tag> <command template with {ID}> <ids...>: <= 3 processes in parallel, each runs its ids sequentially; any failure stops
  local tag=$1 tpl=$2; shift 2; local ids=($@) n=$# pids=()
  for g in 0 1 2; do local sel=(); for ((i=g; i<n; i+=3)); do sel+=(${ids[$i]}); done; [ ${#sel[@]} -gt 0 ] || continue
    ( for id in "${sel[@]}"; do eval "${tpl//\{ID\}/$id}" >> $L/${tag}_g$g.log 2>&1 || { echo "FAILED $id" >> $L/${tag}_g$g.log; exit 1; }; done ) & pids+=($!); done
  for p in "${pids[@]}"; do wait $p || { log "$tag: a process failed (see $L/${tag}_g*.log): stop"; exit 1; }; done; }
step0() {
  if [ -n "${AGENTS_SYNTH_DIR:-}" ]; then log "0. SYNTHETIC rehearsal: freeze/sha checks skipped"; return; fi
  if [ ! -x $R ]; then log "/tmp/renv missing: rebuilding R 4.5.3 + lme4 2.0.6 (conda-forge, micromamba)"
    ( cd /tmp && curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest -o mm.tar.bz2 && $PY -c "import tarfile; tarfile.open('mm.tar.bz2','r:bz2').extract('bin/micromamba','.')" \
      && ./bin/micromamba create -y -q -p /tmp/renv -c conda-forge r-base=4.5.3 r-lme4=2.0.6 ) >> $L/run_agents.log 2>&1 || { log "R rebuild failed: stop"; exit 1; }; fi
  if [ ! -d /tmp/XTMEM/experiments/image_stim/images ]; then
    if [ -d $BK/XTMEM/experiments/image_stim/images ]; then cp -a $BK/XTMEM /tmp/XTMEM && log "/tmp/XTMEM restored from $BK/XTMEM"
    else git clone -q https://github.com/mllewis/XTMEM /tmp/XTMEM && git -C /tmp/XTMEM checkout -q $XTMEM_COMMIT && log "/tmp/XTMEM cloned at $XTMEM_COMMIT"; fi; fi
  [ "$(git -C /tmp/XTMEM rev-parse HEAD)" = "$XTMEM_COMMIT" ] || { log "/tmp/XTMEM not at $XTMEM_COMMIT: stop"; exit 1; }
  log "0. checks"; [ -f $EE/FREEZE_agentes_v2.txt ] || { log "NO FREEZE_agentes_v2.txt: stop"; exit 1; }
  $PY $S/15_freeze_agents.py --verify | tee -a $L/run_agents.log
  $PY $EE/scripts/09_freeze.py --verify | tee -a $L/run_agents.log
  [ -x $R ] && $R -e 'suppressMessages(library(lme4)); cat("lme4", as.character(packageVersion("lme4")), "\n")' | tee -a $L/run_agents.log || { log "/tmp/renv missing: rebuild R env"; exit 1; }
  grep MemAvailable /proc/meminfo | tee -a $L/run_agents.log; df -h $AG | tail -1 | tee -a $L/run_agents.log; }
primary() {
  step0; [ -f $L/run_start_epoch ] || date +%s > $L/run_start_epoch; log "   start $(date -d @$(cat $L/run_start_epoch) '+%F %T %Z'); cap $((CAP_S/3600)) h wall"

  log "1. parity of the encoders with the frozen human embeddings (45 L&F stimuli; |diff| <= 1e-5): FAIL = stop"
  if [ -n "${AGENTS_SYNTH_DIR:-}" ] && [ "${AGENTS_SYNTH_SKIP_PARITY:-}" = 1 ]; then log "   (synthetic rehearsal: parity skipped; covered by tests/test_embed_agents.py)"
  else par3 parity "$PY $S/10_agent_embed.py --parity {ID}" $P10; fi
  log "2. pipeline control 1 (§3.5): T with R0 from the v1 CIFAR-10 cache vs the v1 ledger"
  $SIMENV $PY $S/11_agent_sims.py --analysis T --encoder R0_dinov2s > $L/T_R0_control.log 2>&1
  $PY -c "import json,sys; c=json.load(open('$RES/control_T_exact.json')); print('control_T ok =', c['ok'], '| rule', c['rule_used']); sys.exit(0 if c['ok'] else 1)" | tee -a $L/run_agents.log \
    || { log "GA0: pipeline control 1 FAILED: agents stop here (nothing new was computed with other encoders)"; exit 1; }
  log "3. CIFAR embeddings (c100train, c10pool, c10test; existing caches reused; E4s from the E4 pass)"
  rate_check & RC=$!
  par3 embed "$PY $S/10_agent_embed.py --encoder {ID} --set c100train $LIMIT && $PY $S/10_agent_embed.py --encoder {ID} --set c10pool $LIMIT && $PY $S/10_agent_embed.py --encoder {ID} --set c10test $LIMIT" $P10
  kill $RC 2>/dev/null || true
  if [ -n "${AGENTS_SYNTH_DIR:-}" ]; then backup $AGENTS_SYNTH_DIR/out/emb emb; else backup $AG/out/emb emb; fi
  log "4. §3.6 dispersion of d_perc on the 100 v1 pairs (BEFORE any accuracy)"; $PY $S/11_agent_sims.py --analysis dispersion --encoder all > $L/dispersion.log 2>&1
  log "5. T (seed 42; R0 already done in step 2), D (100 v1 pairs) and D-todos (4,950 pairs, mandatory)"
  par3 T "$SIMENV $PY $S/11_agent_sims.py --analysis T --encoder {ID}" ${P10#R0_dinov2s }
  par3 D "$SIMENV $PY $S/11_agent_sims.py --analysis D --encoder {ID}" $P10
  $PY -c "import json; c=json.load(open('$RES/control_c100_recode.json')); print('control_c100:', c['verdict'])" | tee -a $L/run_agents.log
  par3 Dall "$SIMENV $PY $S/11_agent_sims.py --analysis Dall --encoder {ID} $DALL_MAX" $P10
  log "6. H1/H2/H3"; par3 H "$SIMENV $PY $S/12_agent_h1h2h3.py --encoder {ID}" $P10
  log "7. H4 per encoder (people; replica of v1 H4, 24 items)"
  par3 H4 "if cap_exceeded; then log 'H4 {ID}: CAP of 14 h exceeded: H4 cut for this encoder (risk 9) -> technical; LOG A DEVIATION'; else OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $R $S/13_h4_by_encoder.R $LONG $RES/h4_{ID}.csv {ID}; fi" $P10
  $PY -c "import pandas as p,glob; fs=[f for f in sorted(glob.glob('$RES/h4_*.csv')) if not f.endswith('h4_by_encoder.csv')]; p.concat([p.read_csv(f) for f in fs]).to_csv('$RES/h4_by_encoder.csv', index=False) if fs else None"
  log "8. regions, GA, families, agents x humans table (14) and fig x1 (14b)"; $PY $S/14_agent_regions.py > $L/14.log 2>&1; $PY $S/14b_fig_x1.py >> $L/14.log 2>&1
  backup $RES results
  log "primary done. Next: sensitivities (optional), then RESULTADOS_agentes.md + DESVIACIONES by hand; notify Evaristo. Nothing is published."; }
sensitivities() {
  step0
  if cap_exceeded; then log "CAP of 14 h exceeded: sensitivities cut entirely (risk 9: first cut). LOG A DEVIATION."; return; fi
  cap_exceeded && { log "CAP of 14 h exceeded before S-a: remaining sensitivities cut (risk 9). LOG A DEVIATION."; return; }
  log "S-a. T-sem 'episodes' (seeds 0-3, fixed pool)"; for s in 0 1 2 3; do par3 Tsem_e$s "$SIMENV $PY $S/11_agent_sims.py --analysis T --encoder {ID} --seed $s" $P10; done
  cap_exceeded && { log "CAP of 14 h exceeded before S-b: remaining sensitivities cut (risk 9). LOG A DEVIATION."; return; }
  log "S-b. E4s (CLS of MAE; embeddings from the E4 pass)"; for a in T D Dall; do $SIMENV $PY $S/11_agent_sims.py --analysis $a --encoder E4s_mae_cls $([ $a = Dall ] && echo $DALL_MAX) >> $L/E4s.log 2>&1; done
  $SIMENV $PY $S/12_agent_h1h2h3.py --encoder E4s_mae_cls >> $L/E4s.log 2>&1
  cap_exceeded && { log "CAP of 14 h exceeded before S-c: remaining sensitivities cut (risk 9). LOG A DEVIATION."; return; }
  log "S-c. F1 seeds 1-2"; par3 F1s "$PY $S/10_agent_embed.py --encoder {ID} --set c100train $LIMIT && $PY $S/10_agent_embed.py --encoder {ID} --set c10pool $LIMIT" F1_rn50rand1 F1_rn50rand2
  par3 F1sim "$SIMENV $PY $S/11_agent_sims.py --analysis T --encoder {ID} && $SIMENV $PY $S/11_agent_sims.py --analysis D --encoder {ID} && $SIMENV $PY $S/11_agent_sims.py --analysis Dall --encoder {ID} $DALL_MAX && $SIMENV $PY $S/12_agent_h1h2h3.py --encoder {ID}" F1_rn50rand1 F1_rn50rand2
  cap_exceeded && { log "CAP of 14 h exceeded before S-d: remaining sensitivities cut (risk 9). LOG A DEVIATION."; return; }
  log "S-d. H5 (optional, descriptive)"; par3 H5 "$SIMENV $PY $S/12_agent_h1h2h3.py --encoder {ID} --h5" $P10
  cap_exceeded && { log "CAP of 14 h exceeded before S-e: remaining sensitivities cut (risk 9). LOG A DEVIATION."; return; }
  log "S-e. T-sem 'full' (re-drawn pools with default_rng(s), seeds 0-3; R0 encoded by 10_agent_embed.py like the others: the r04 caches were lost on 22-11)"
  for s in 0 1 2 3; do par3 Tsem_f$s "$PY $S/10_agent_embed.py --encoder {ID} --set c10pool_seed$s $LIMIT && $SIMENV $PY $S/11_agent_sims.py --analysis T --encoder {ID} --seed $s --pool c10pool_seed$s" $P10; done
  $PY $S/14_agent_regions.py > $L/14_sens.log 2>&1
  log "sensitivities done"; }
"${1:?primary|sensitivities}"
