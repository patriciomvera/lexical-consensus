# Comparación de encoders (v2 de arXiv 2611.14116): scripts, pruebas y ejecución del fin de semana

**Estado:** escritos el 18-11-2026 y probados **solo con datos sintéticos** (`tests/`). No se ajustó ningún GLMM con datos reales,
no se leyó `h4_long.csv`, `all_raw_A.csv` ni ninguna respuesta de participantes, y **no se congeló** (`09_freeze.py` no se ha ejecutado
y se niega a correr antes del viernes 20-11-2026 y sin la revisión de Ariadna de estos scripts).

## Scripts (orden de uso)
| Script | Qué hace | Plan |
|---|---|---|
| `01_embed.py` | embeddings de los 45 estímulos (ya corrido 18-11 10:15–10:19) | §3 |
| `00_vision_params_and_checks.py` | parámetros y control R0 frente a la v1 (ya corrido) | §3.2 |
| `01b_embed_common.py` | embeddings con preprocesamiento común para S6; prueba de humo 18-11 12:06 en `/tmp/s6smoke` (R0 coseno mín. 0,9999953; F1 y F2 idénticos) | §5.4 S6 |
| `h4_prep_encoders.py` | `h4_long_<ID>.csv` con la misma lógica, filas y orden que `h4_prep.py`; `--check` = control de tubería G0 (casi-empates < 1e−6 agrupados, BC3; escribe `pipeline_control.json`) | §5.1 |
| `02_equivalence.py` | parte A descriptiva; parte B: r de zd (8 ítems, parcializando nivel), UPGMA con cortes 0,3/0,5/0,7, calibración por corte, perfil | §6 |
| `lib_glmm.R` | lectura, mapeo de condiciones, z por encoder y conjunto de ítems, control de convergencia | §4, §5 |
| `05_twostage_stage1.R` | etapa 1: pendiente de zd por experimento (contr.sum); `fit` = `ladder` (pasada 1: L1 → L4 hasta el primero que converge) o un peldaño fijo | §5.2 |
| `05b_select_structure.py` | peldaño **común** a todos los encoders (`STRUCTURE_SCOPE = "common"`, confirmado por Patricio 18-11 12:14), solo por convergencia; pide las pasadas 2; escribe primario, `per_encoder_rung` y `S9_mixed` (descriptivos) y `structure.csv`; **sin estructura común sale con código 4 y `structure_DEVIATION.txt`, sin caer a L4** (REVISION3b) | §5.2, S9 |
| `06_stage2_permutation.py` | etapa 2: Δ, permutación exacta 924, IC por inversión, TOST, Welch; tabla tipo de ajuste × orden (`*_fit_type_by_order.csv`) | §5.2 |
| `07_pooled_glmm.R` | control positivo, S1 (04f), S2, S3, F0 | §5.3, §5.4 |
| `08_regions.py` | regiones (con BC1), Holm por δ con k y excluidos de cada familia, G0–G4 en la grilla δ × corte, perfiles, pisos; `--stage2-s9` añade el rótulo descriptivo "sensible al tipo de ajuste" | §7, S9 |
| `09_freeze.py` | manifiesto `FREEZE_encoders_v2.txt` (`--freeze`) y verificación (`--verify`) | §10.2 |
| `run_weekend.sh` | orden del sábado y domingo (§10.3); `WEEKEND_SYNTH_DIR=/tmp/synth…` solo para el ensayo sintético | §10.3 |

## Lista de verificación de Ariadna (REVISION2, "Scripts de análisis") y dónde se cumple
1. **Prep.** (BC3 de REVISION3: el orden de rangos agrupa casi-empates < 1e−6; ruido 1e−8, casi-empate invertido y borde de redondeo PASAN; intercambio real con brecha 5e−6 FALLA.) `tests/test_prep.py` ejecuta el bucle de `h4_prep.py` **desde su propio texto fuente** sobre datos sintéticos y compara: columnas, filas y orden idénticos, d idéntica (diferencia 0). Los npz se leen por nombre de imagen (`names`). G0: `--check` exige filas idénticas, |Δd| ≤ 1e−5 y el mismo orden de rangos (falla con +2e−5 y con filas reordenadas). REVISION3b: G0 también falla si el grupo de casi-empates más ancho de la v1 mide ≥ 1e−6 (prueba con cinco valores a 4e−7: ancho 1,6e−6, FALLA). z: `lib_glmm.R::add_zd`, por encoder y conjunto de ítems, sobre todos los experimentos; `02` lee con `usecols` sin `selected`.
2. **Etapa 1.** `contrasts(dd$condition) <- contr.sum(4)` dentro de cada experimento; `read_long` falla si una condición no se mapea (`stopifnot`), y la salida guarda n, personas, el peldaño (`rung` L1–L4), `converged` (criterio de `lib_glmm.R::ladder_converged`: sin error, código 0, sin avisos de convergencia de lme4, no singular con tol 1e−4), advertencias, singularidad y código del optimizador. Se guarda el β̂ de `zd`. La elección del peldaño (`05b`) usa solo `converged` (`tests/test_select_structure.py`, 8 pruebas: todo L1; un piso difícil baja a todos a L2 con dos pasadas; encoder sin peldaño fuera de la elección y evaluado con el común; la elección no cambia si cambian los β̂; constante fija sin opción; REVISION3b: sin estructura común → código 4 sin tablas, conjunto elegible vacío → sin estructura, y `run_weekend.sh rung8` con peldaño nulo → falla y registra la desviación, nunca imprime L4).
3. **Etapa 2.** `COMBOS` = las 924 asignaciones (prueba: 924 filas únicas con 6 "3-1"); Δ = 3-1 menos 1-3; p con ≥ (tolerancia 1e−12); IC por inversión en grilla de paso 0,001 que cubre Δ ± 3·(rango + 1), con banderas `ci*_at_grid_edge` y `ci*_contiguous`; TOST = máx. de los dos p desplazados. `tests/test_stage2.py` contrasta todo contra fuerza bruta con Δ conocido.
4. **Regiones.** Precedencia [no interpretable] → 1 → 2 → 3 → 4 → 5 por δ; familias Holm de 2·k; G0–G4 en orden; perfiles por encoder, por agrupamiento y global (grilla δ × corte); sin opciones para elegir δ, corte ni encoders (`tests/test_regions.py`, 20 pruebas, incluidos los dos ejemplos de §7.1, un encoder que falla el control, R0 que falla, R0 "inconcluso (técnico)" → G0 "sin lectura: técnico" en las 9 celdas y nunca G1 (REVISION3b), R0 técnico solo con 24 ítems, y calibración que pasa en un corte y falla en otro).
5. **`07`.** Fórmulas copiadas de §5.3 (= C6 de 04c, con su fallback), §5.4 y 04f (Mbase, Mb, Mc con contrastes de tratamiento, como en 04f).
6. **Congelamiento.** `09_freeze.py --freeze --scripts-review <archivo>` escribe el sha256 del plan, de las revisiones, de todo `scripts/` (incluidas las pruebas y este README), de los snapshots del plan, de `encoder_specs.json` y de los 13 npz.

## Pruebas sintéticas (reproducibles)
```
/workspace/rv_venv/bin/python tests/test_stage2.py
/workspace/rv_venv/bin/python tests/test_regions.py
/workspace/rv_venv/bin/python tests/test_freeze.py
/workspace/rv_venv/bin/python tests/test_select_structure.py
bash tests/run_e2e_synthetic.sh /tmp/synth_enc_test                 # genera datos sintéticos y corre todo de punta a punta
WEEKEND_SYNTH_DIR=/tmp/synth_enc_test bash run_weekend.sh saturday   # ensayo del orquestador con datos sintéticos
WEEKEND_SYNTH_DIR=/tmp/synth_enc_test WEEKEND_SYNTH_P10="R0_dinov2s O1_dinov2b F2_pixels" WEEKEND_SYNTH_S6_FAIL=1 bash run_weekend.sh sunday   # S6 falla a propósito: el resto sigue
WEEKEND_SYNTH_DIR=/tmp/synth_enc_test WEEKEND_SYNTH_P10="R0_dinov2s O1_dinov2b F2_pixels" bash run_weekend.sh s6
```

## Fin de semana (solo después del congelamiento; plan §10.3)
Antes de empezar: `/tmp/renv` (R 4.5.3 + lme4 2.0.6) y `/tmp/XTMEM` están en `/tmp` y pueden haberse borrado. `run_weekend.sh` lo comprueba
(paso 0) y se detiene si falta R; las imágenes solo hacen falta para S6. `09_freeze.py --verify` debe dar OK; `h4_long.csv` debe tener el sha256 `d9d30f9d…394dd1`.
- **Sábado** (`bash run_weekend.sh saturday`): 0 verificación → 1 prep + G0 (si falla, se detiene todo) → 2 `02` → 3 tiempo de un encoder (columna `secs`; solo actualiza el presupuesto) → 4 etapa 1 con 8 ítems (pasada 1) → 5 etapa 1 con 24 ítems (pasada 1) → 5b `05b` + pasadas 2 (peldaño común; si no hay, se detiene y registra la desviación) → 6 control positivo → 7 etapa 2 y `08_regions.py`.
- **Domingo** (`bash run_weekend.sh sunday`): S1 → S5, S7, S8 (S4, S5, S7, S8 con el peldaño común de 8 ítems), regiones con S9 (ajuste mixto, sin ajustes extra), familia A (F0) y **S6 al final** (si falla, no detiene nada y se registra). Tope de 32 h de CPU **vigilado a mano** (el script no lo aplica); si se supera, se corta en orden inverso de ejecución y se registra. Después, a mano: `RESULTADOS_encoders.md`, entradas de `DESVIACIONES` y aviso a Evaristo. No se publica nada sin Patricio.
