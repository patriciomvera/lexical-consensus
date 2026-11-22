#!/usr/bin/env bash
# END-TO-END on SYNTHETIC data only (tests/make_synthetic.py): unit tests + run_weekend.sh itself in synthetic mode (saturday, sunday with
# a forced S6 failure that must not stop anything). Usage: bash tests/run_e2e_synthetic.sh /tmp/synth_enc_test   (never point it at real data)
set -euo pipefail
T=${1:?synthetic dir}; S=$(cd "$(dirname "$0")/.." && pwd); PY=/workspace/rv_venv/bin/python
case "$T" in /tmp/synth*) ;; *) echo "refusing: synthetic dir must be /tmp/synth*"; exit 2;; esac
rm -rf "$T"; $PY $S/tests/make_synthetic.py $T 15
$PY $S/tests/test_prep.py $T
for t in test_stage2 test_regions test_freeze test_select_structure; do $PY $S/tests/$t.py; done
P3="${WEEKEND_SYNTH_P10:-R0_dinov2s O1_dinov2b F2_pixels}"
WEEKEND_SYNTH_DIR=$T WEEKEND_SYNTH_P10="$P3" bash $S/run_weekend.sh saturday
WEEKEND_SYNTH_DIR=$T WEEKEND_SYNTH_P10="$P3" WEEKEND_SYNTH_S6_FAIL=1 bash $S/run_weekend.sh sunday
WEEKEND_SYNTH_DIR=$T WEEKEND_SYNTH_P10="$P3" bash $S/run_weekend.sh s6
echo "E2E SYNTHETIC OK"
