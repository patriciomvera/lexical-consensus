#!/usr/bin/env bash
# v2f (= v2e + corrections of the final internal review; no new computation): regenerates every number, table and figure from the frozen files (repo checked out at a945ed2), runs the post hoc
# analyses (posthoc_v2c, exploratory, outside the frozen plans), applies the v2c text edits to the v2b snapshot, and compiles.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$HERE/.."; REPO="${REPO:?set REPO to a checkout of this repository at commit a945ed2 (see paper_v2/README.md)}"; B="$ROOT/v2c_build"
FREEZE_COPY="${FREEZE_COPY:-$ROOT/freeze_copy/DIFF_agentes_propuesta.patch}"
PY="${PY:-python3}"
"$PY" "$ROOT/scripts/make_encoder_tables.py" "$REPO"
"$PY" "$ROOT/scripts/make_tabla3b_imagenes.py" "$REPO"
"$PY" "$ROOT/scripts/make_numbers_v2.py" "$REPO" "$FREEZE_COPY"
"$PY" "$ROOT/scripts/make_forest_plot.py" "$REPO"
FOREST_MARK='$^{a}$' FOREST_OUT=fig_forest_adults_v2c.pdf "$PY" "$ROOT/scripts/make_forest_plot.py" "$REPO"
"$PY" "$ROOT/scripts/fix_v1_number_format.py" "$HERE/generated/numbers.tex" "$HERE/generated/numbers_v1_fmt.tex"
cp "$ROOT/generated/numbers_v2_text.tex" "$HERE/generated/"
cp "$ROOT/generated/fig_forest_adults_v2c.pdf" "$HERE/figs/"
mkdir -p "$HERE/tables"
for t in tab3_matched tab3b_imagenes_por_encoder tab3c_tortuga_por_contexto; do cp "$ROOT/generated/$t.tex" "$HERE/tables/"; done
for t in tab1_adults tab2a_agents_distance tab2b_agents_human tab_agents_compact tab_h1_compact; do cp "$ROOT/generated/${t}_v2b.tex" "$HERE/tables/$t.tex"; done
sed 's/DINOv2-small (reference, v1)/DINOv2-small (reference)/' "$ROOT/generated/tab_checkpoints.tex" > "$HERE/tables/tab_checkpoints.tex"
grep -q "DINOv2-small (reference)" "$HERE/tables/tab_checkpoints.tex"
# post hoc analyses (B1-B4): skips outputs that already exist unless FORCE=1
# v2d runs no analysis: it reads the post hoc outputs produced for v2c
for id in R0_dinov2s O1_dinov2b R1_cvcl E1_siglip O2_openclip E2_vitsup E3_rn50sup E4_mae F1_rn50rand0 F2_pixels; do for k in p02 p03; do test -s "$ROOT/posthoc_v2c/out/${k}_$id.csv"; done; done
for f in p04_h2_estimates.csv p04_h2_mde.csv p05_support_dispersion.csv p05_sup_distance_summary.csv fig_sup_distances.pdf; do test -s "$ROOT/posthoc_v2c/out/$f"; done
cp "$ROOT/posthoc_v2c/out/fig_sup_distances.pdf" "$HERE/figs/"
MS_DIR=manuscrito_v2f "$PY" "$ROOT/scripts/make_numbers_v2c.py" "$REPO"
MS_DIR=manuscrito_v2f "$PY" "$ROOT/scripts/make_numbers_v2d.py" "$REPO"
MS_DIR=manuscrito_v2f "$PY" "$ROOT/scripts/make_numbers_v2f.py"
"$PY" "$B/fix_tables_v2c.py" "$HERE/tables"
sed -i 's/^\\noindent\\resizebox{\\linewidth}{!}{%$/\\noindent\\adjustbox{max width=\\linewidth}{%/' "$HERE"/tables/*.tex
if grep -q "resizebox" "$HERE"/tables/*.tex; then echo "resizebox left in tables" >&2; exit 1; fi
# text: v2b snapshot + asserted v2c edits; new appendices
echo "e1e7a25fb9143d2daf426c0507f0e0cd9cc2f276ca53384bd76ce8b7d4cc0697  $B/base_main_v2b.tex" | sha256sum -c -
"$PY" "$B/apply_v2c_edits.py" "$B/base_main_v2b.tex" "$HERE/main_v2c_regen.tex"
D="$ROOT/v2d_build"; echo "cd88ad2f02b4dea14be8a64d238da24d49de416a27b88fcff8ab08e1ec504992  $HERE/main_v2c_regen.tex" | sha256sum -c -
cmp "$HERE/main_v2c_regen.tex" "$D/base_main_v2c.tex" && rm "$HERE/main_v2c_regen.tex"
"$PY" "$D/apply_v2d_edits.py" "$D/base_main_v2c.tex" "$HERE/main_v2d_regen.tex"
E="$ROOT/v2e_build"; echo "9129ab0e54bde96534430e5d2f84d4e18b5aa67fbccd59bae13fdf3f82d04362  $HERE/main_v2d_regen.tex" | sha256sum -c -
rm "$HERE/main_v2d_regen.tex"
"$PY" "$E/apply_v2e_edits.py" "$E/base_main_v2d.tex" "$HERE/main_v2e_regen.tex"
F="$ROOT/v2f_build"; echo "49ac4512b6bd9e3181a4b51d183865e2a1cdc38491c8cb537f2668288379843c  $HERE/main_v2e_regen.tex" | sha256sum -c -
rm "$HERE/main_v2e_regen.tex"
"$PY" "$F/apply_v2f_edits.py" "$F/base_main_v2e.tex" "$HERE/main.tex"
cat "$B/appendix_rules_v2c.tex" "$B/appendix_posthoc_v2c.tex" > "$HERE/appendix_rules_posthoc_v2c.tex"
"$PY" "$F/fix_appendix_tables_v2f.py" "$HERE"
cd "$HERE" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
# checks
LOG="$HERE/main.log"
! grep -q "Overfull" "$LOG" || { echo "overfull boxes" >&2; exit 1; }
! grep -qi "undefined" "$LOG" || { echo "undefined references or citations" >&2; exit 1; }
! pdftotext "$HERE/main.pdf" - | grep -q "??" || { echo "?? in PDF" >&2; exit 1; }
! pdffonts "$HERE/main.pdf" | grep -q "Type 3" || { echo "Type 3 fonts" >&2; exit 1; }
echo "v2f build OK: $(pdfinfo "$HERE/main.pdf" | awk '/Pages/{print $2}') pages"
