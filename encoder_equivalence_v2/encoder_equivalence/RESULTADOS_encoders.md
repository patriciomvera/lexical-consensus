# Resultados de la comparación de encoders v2 (parte humana)

Documento interno. No es una publicación. Compilación fáctica de lo que produjo la tubería congelada el 21 y el 22 de noviembre de 2026 (hora de Santiago, zona America/Santiago, UTC−3). No interpreta los resultados, no propone títulos ni encuadres, y no añade conclusiones distintas de las que escribieron las reglas de decisión del plan al ser aplicadas por los scripts congelados. Don Patricio revisa e interpreta.

Los números se copian de los archivos citados, con la grafía del archivo (se mantiene el punto decimal de los CSV y de los JSON). Si un dato no está o es ambiguo, se declara en el mismo apartado.

## Índice

1. Procedencia, congelamiento y tiempos de ejecución
2. Desviaciones registradas
3. Control de tubería (G0)
4. Agrupamiento de geometría (salida de `02_equivalence.py`, leída por `08_regions.py`)
5. Peldaños de efectos aleatorios
6. Control positivo por encoder
7. Análisis primario: regiones por encoder y conclusiones globales G0–G4 (transcripción de `08_regions.py`)
8. Análisis primario: valores numéricos de la etapa 2 y de la tabla de regiones
9. Sensibilidades S1–S8
10. S9 (ajuste mixto, descriptivo) y marcas ya calculadas por `08_regions.py`
11. Familia A frente a F0
12. S6 (preprocesamiento común)
13. Advertencia Pandas4Warning (`fillna`)
14. Completitud y archivos citados

---

## 1. Procedencia, congelamiento y tiempos de ejecución

- **Trabajo:** comparación de encoders v2, parte humana, de arXiv 2611.14116 (Lexical Consensus, Open Mind).
- **Plan congelado:** `propuesta_comparacion_encoders_v2_2026-11-18.md`, sha256 `cfaefee5ab8ebfa764c65b6c3a28ae9d0eb23ddc43f4746c6eeb99ed9793fec4` (hash escrito en `FREEZE_encoders_v2.txt` y recomputado sobre el archivo).
- **Manifiesto de congelamiento:** `FREEZE_encoders_v2.txt`. El propio manifiesto tiene sha256 `975aab4cc922a73b3b531fb22f26ce87145143c89a8b8ea6d44430a00894b606` (citado también en `FREEZE_encoders_v2_complemento.txt`). Cabecera del manifiesto: «FREEZE encoders v2 (arXiv 2611.14116) - internal freeze, NOT an external preregistration»; «time: 2026-11-20 09:15:04 America/Santiago (UTC-0300)»; «files: 47».
- **Complemento** (archivos de revisión que `09_freeze.py` no lista; el manifiesto no se modifica): `FREEZE_encoders_v2_complemento.txt`, escrito el 2026-11-20 a las 09:15:09, hora de Santiago.
- **Visto bueno de Ariadna (parte humana):** `VISTOBUENO_ariadna_parte_humana_2026-11-18.md`, sha256 `2dd7c109baa302cb1afd0982f71fec6ef9943c446d29912e0402762beac05a78` (también listado en el manifiesto).
- **Verificación al arrancar cada jornada** (`scripts/09_freeze.py --verify`), copiada de los logs:
  - `logs/run_saturday_2026-11-21.log` (21-11-2026, 08:25:17, hora de Santiago): `VERIFY 47 files: OK; files not in manifest: none`; a continuación, `lme4 2.0.6`.
  - `logs/run_sunday_2026-11-21.log` (21-11-2026, 14:57:14, hora de Santiago): `VERIFY 47 files: OK; files not in manifest: none`; a continuación, `lme4 2.0.6`.
- **Nada se publica** (paso 10 del plan y última línea del log del domingo).

### 1.1 Tiempos de la corrida de sábado (21-11-2026, hora de Santiago)

Fuente: `logs/run_saturday_2026-11-21.log`.

| marca en el log | texto del log |
|---|---|
| 2026-11-21 08:25:17 -03 | 0. checks |
| 2026-11-21 08:25:19 -03 | 1. h4_prep_encoders + pipeline control (G0) |
| (misma marca, línea siguiente) | PIPELINE_CONTROL … -> PASS (texto completo en el §3) |
| 2026-11-21 08:25:47 -03 | 2. 02_equivalence (descriptive table + geometry rule at cuts 0.3/0.5/0.7 + calibration per cut) |
| 2026-11-21 13:26:16 -03 | 3. timing: R0, 8 items, ladder pass 1 (secs column) -> update budget only |
| 2026-11-21 13:27:28 -03 | 4. stage 1, 8 items, ladder pass 1 |
| 2026-11-21 13:31:05 -03 | 5. stage 1, 24 items, ladder pass 1 |
| 2026-11-21 13:57:47 -03 | 5b. common structure (05b, convergence only; pass 2 fits where needed) |
| 2026-11-21 13:57:48 -03 | pass 2 (iter 1): items=8 rung=L2: E3_rn50sup O1_dinov2b R0_dinov2s R1_cvcl |
| 2026-11-21 13:59:28 -03 | pass 2 (iter 1): items=8 rung=L3: F1_rn50rand0 |
| 2026-11-21 13:59:57 -03 | pass 2 (iter 1): items=8 rung=L4: E4_mae F2_pixels |
| 2026-11-21 14:00:21 -03 | pass 2 (iter 1): items=24 rung=L3: E3_rn50sup E4_mae F1_rn50rand0 F2_pixels O1_dinov2b R1_cvcl |
| 2026-11-21 14:08:23 -03 | pass 2 (iter 2): items=8 rung=L4: E1_siglip E2_vitsup E3_rn50sup F1_rn50rand0 O1_dinov2b O2_openclip R0_dinov2s R1_cvcl |
| 2026-11-21 14:09:19 -03 | pass 2 (iter 2): items=24 rung=L4: E4_mae |
| 2026-11-21 14:11:20 -03 | pass 2 (iter 3): items=24 rung=L4: E1_siglip E2_vitsup E3_rn50sup F1_rn50rand0 F2_pixels O1_dinov2b O2_openclip R0_dinov2s R1_cvcl |
| 2026-11-21 14:16:03 -03 | 6. positive control (before regions: 08 uses it) |
| 2026-11-21 14:24:59 -03 | 7. stage 2 (primary = common rung; per_encoder_rung and S9_mixed descriptive) + regions (primary only) |
| 2026-11-21 14:25:06 -03 | Saturday done |

El log del sábado también registra, entre los pasos 7 y «Saturday done», la advertencia Pandas4Warning de `06_stage2_permutation.py` y de `08_regions.py`, la línea `wrote …/results/stage2_primary.csv 60 rows`, y la transcripción de `results/regions/regions_report.md`. Esa advertencia se describe en el §13. No hay línea `structure_DEVIATION` en este log.

### 1.2 Tiempos de la corrida de domingo (21-11 y 22-11-2026, hora de Santiago)

Fuentes: `logs/run_sunday_2026-11-21.log`, `logs/watch_s2_pause.log` y `DESVIACIONES_encoders_v2.md`.

| marca en el log o en DESVIACIONES | texto |
|---|---|
| 2026-11-21 14:57:14 -03 | 0. checks (VERIFY 47 files: OK; files not in manifest: none) |
| 2026-11-21 14:57:15 -03 | 8. S1 |
| 2026-11-21 21:53:14 -03 | S2  (en el log: S1 ha terminado y se escribe la marca de inicio de S2) |
| 2026-11-21 21:53:15 -0300 | PAUSED at S2 start: bash 2371120 and children 2522704  (logs/watch_s2_pause.log) |
| 2026-11-22 08:37:47 -0300 | Reanudada (SIGCONT), según DESVIACIONES_encoders_v2.md, entrada 2 |
| 2026-11-22 09:06:58 -03 | S3 |
| 2026-11-22 09:28:29 -03 | S4/S5/S7/S8 at the common 8-item rung L4 (fixed on Saturday by 05b; not re-selected) |
| 2026-11-22 09:28:29 -03 | S4 |
| 2026-11-22 09:29:30 -03 | S5 |
| 2026-11-22 09:30:40 -03 | S7 |
| 2026-11-22 09:31:18 -03 | S8 |
| (tras S8, sin marca de reloj propia) | Pandas4Warning en 06_stage2_permutation.py; wrote …/results/stage2_sensitivities.csv 23 rows |
| 2026-11-22 09:31:38 -03 | S9 (descriptive; Patricio 12:12): mixed fit = first converged rung per experiment, already in stage2_primary.csv (variant S9_mixed); no extra fits |
| (tras S9, sin marca de reloj propia) | Pandas4Warning en 08_regions.py (líneas 145 y 149); transcripción de results/regions_with_S9/regions_report.md |
| 2026-11-22 09:31:39 -03 | 9. Family A: F0 (Mb condition:order coefficients are in S1_*.csv) |
| 2026-11-22 09:32:50 -03 | S6 (last; REVISION3 I2: its failure does not stop anything else) |
| (tras S6, sin marca de reloj propia) | wrote …/results/stage2_S6.csv 10 rows |
| 2026-11-22 09:35:56 -03 | 10. write RESULTADOS_encoders.md + DESVIACIONES entries by hand from results/; notify Evaristo. Nothing is published. |

No aparece la línea `S6 FAILED` en `logs/run_sunday_2026-11-21.log`.

## 2. Desviaciones registradas

Se transcribe, sin cambios, el archivo `DESVIACIONES_encoders_v2.md` tal como estaba al escribir este documento:

# Desviaciones: comparación de encoders v2 (parte humana, plan cfaefee5, congelamiento 975aab4c)

1. **S1 superó el tope de 32 h de CPU (§9.2).** Por decisión de Patricio del 21-11-2026 a las 18:16 (hora de Santiago), S1 (S1_24 y S1_8) se dejó terminar aunque superara el tope. S1 terminó el 21-11 a las 21:53:14, y la corrida se pausó con SIGSTOP al comenzar S2, a las 21:53:15. CPU medida del domingo hasta la pausa: 43,305 h (bash 2371120 y sus descendientes). El sábado se estimó en unas 11–13 h a partir de lecturas parciales, sin contabilidad de procesos. Total aproximado: 54–56 h. El plan estimaba ~7,5 h de CPU para S1; la diferencia se debe sobre todo a que R usa varios hilos por ajuste (el reloj de S1, 6,9 h, sí calzó con la estimación).
2. **Sensibilidades restantes sin recorte (contra §9.2).** Por decisión de Patricio del 21-11-2026 a las 22:04 (hora de Santiago), S2–S8, S9, F0 y S6 se corren completas, sin el recorte en orden inverso que pide §9.2. La decisión se tomó antes de leer ningún resultado: no se había abierto ningún archivo de results/. La corrida se reanudó con SIGCONT el 22-11-2026 (hora registrada abajo). No cambia ningún script, dato ni análisis congelado.
   Reanudada: 2026-11-22 08:37:47 -0300.
3. **Cierre del domingo y CPU total.** La etapa terminó el 22-11-2026 a las 09:35:56 (hora de Santiago), sin fallas ni structure_DEVIATION.txt; S6 también terminó. Desde la reanudación (08:37:47) tardó 58 min. CPU medida del domingo: 48,876 h a las 09:34:42, poco antes del final; el valor final exacto no quedó disponible, y se estima en ~49 h. Sábado: ~11–13 h (estimación). Total del fin de semana: ~60–62 h, frente al tope de 32 h. Advertencia Pandas4Warning (fillna) en 06_stage2_permutation.py y 08_regions.py, igual que el sábado; no detuvo la corrida.

## 3. Control de tubería (G0)

El plan (§5.1 y §7.4) hace del control de tubería una condición de G0. El resultado se lee de `results/pipeline_control.json` (y no de un buscado en el log). Contenido completo de ese JSON:

```json
{"ok": true, "rows_identical": true, "max_abs_diff": 0.0, "tol": 1e-05, "near_tie": 1e-06, "same_rank_order": true, "near_tie_group_max_width": 5.960464477539063e-08, "near_tie_groups_ok": true}
```

Contenido completo de `results/pipeline_control.log`:

```
PIPELINE_CONTROL rows_identical=True max_abs_diff=0 tol=1e-05 same_rank_order(near-ties<1e-06 merged)=True near_tie_group_max_width=5.96e-08 (<1e-06: True) -> PASS
```

`scripts/08_regions.py` toma `pipeline_ok` de `ok` en el JSON. En `results/regions/regions_report.md` y en `results/regions_with_S9/regions_report.md` la primera viñeta dice: «Control de tubería: PASA».

## 4. Agrupamiento de geometría (salida de `02_equivalence.py`, leída por `08_regions.py`)

El paso 2 del sábado escribió, entre otros, `out/equivalence_grouping.json`, `out/equivalence_descriptive_long.csv`, `out/zd_corr_items8_partial_level.csv`, `out/zd_corr_items24_partial_level_x_domain.csv`, `out/distance_scale.csv` y `figs/encoder_equivalence_descriptive.png`. Las matrices descriptivas de correlación no se transcriben aquí; están en los CSV citados. `08_regions.py` lee el objeto `grouping` de `out/equivalence_grouping.json` y escribe su campo `profile` en el informe de regiones («Perfil del agrupamiento de geometría: depende del corte»).

Valores del objeto `grouping` en `out/equivalence_grouping.json`:

- **Regla** (`grouping.rule`): UPGMA on 1 - r (8 items, partial level), cuts 0.3/0.5/0.7, calibration per cut
- **Perfil** (`grouping.profile`): depende del corte
- **Calibración y lectura por corte** (campos `calibration`, `n_trained_groups`, `reading`, `partition`):
| corte | O1_with_R0 | floors_apart | discriminates | n_trained_groups | reading | partición |
|---|---|---|---|---|---|---|
| 0.3 | true | false | false | 4 | sin grupos | {E1_siglip, E3_rn50sup, O2_openclip}; {E2_vitsup, O1_dinov2b, R0_dinov2s}; {E4_mae, F2_pixels}; {F1_rn50rand0}; {R1_cvcl} |
| 0.5 | true | false | false | 3 | sin grupos | {E1_siglip, E3_rn50sup, O2_openclip}; {E2_vitsup, O1_dinov2b, R0_dinov2s}; {E4_mae, F1_rn50rand0, F2_pixels, R1_cvcl} |
| 0.7 | true | false | false | 2 | sin grupos | {E1_siglip, E2_vitsup, E3_rn50sup, O1_dinov2b, O2_openclip, R0_dinov2s}; {E4_mae, F1_rn50rand0, F2_pixels, R1_cvcl} |

- **Etiquetas de grupo por encoder y corte** (`grouping.cuts.*.labels`):
| encoder | corte 0.3 | corte 0.5 | corte 0.7 |
|---|---|---|---|
| R0_dinov2s | 4 | 2 | 2 |
| O1_dinov2b | 4 | 2 | 2 |
| R1_cvcl | 2 | 1 | 1 |
| E1_siglip | 5 | 3 | 2 |
| O2_openclip | 5 | 3 | 2 |
| E2_vitsup | 4 | 2 | 2 |
| E3_rn50sup | 5 | 3 | 2 |
| E4_mae | 1 | 1 | 1 |
| F1_rn50rand0 | 3 | 1 | 1 |
| F2_pixels | 1 | 1 | 1 |

- **ref90_vs_R0** (`grouping.ref90_vs_R0`):
| encoder | ref90_vs_R0 |
|---|---|
| R0_dinov2s | true |
| O1_dinov2b | true |
| R1_cvcl | false |
| E1_siglip | false |
| O2_openclip | false |
| E2_vitsup | false |
| E3_rn50sup | false |
| E4_mae | false |
| F1_rn50rand0 | false |
| F2_pixels | false |

- **linkage** (`grouping.linkage`; copiado como listas):
  - [0.0, 1.0, 0.0714, 2.0]
  - [5.0, 10.0, 0.1523, 3.0]
  - [3.0, 4.0, 0.1709, 2.0]
  - [7.0, 9.0, 0.2023, 2.0]
  - [6.0, 12.0, 0.2909, 3.0]
  - [2.0, 8.0, 0.4631, 2.0]
  - [13.0, 15.0, 0.4747, 4.0]
  - [11.0, 14.0, 0.5102, 6.0]
  - [16.0, 17.0, 0.9174, 10.0]

## 5. Peldaños de efectos aleatorios

La elección la hizo `scripts/05b_select_structure.py` el sábado (paso 5b), solo por convergencia. El log del domingo, al comenzar S4–S8, registra que esas sensibilidades usan el peldaño común de 8 ítems `L4`, fijado el sábado y no vuelto a elegir.

Peldaño común y peldaño propio, copiados de `results/structure_structure.csv` (los mismos valores aparecen en `results/structure_status.json`):

| items | scope | common_rung | encoder | own_rung | in_selection |
|---|---|---|---|---|---|
| 8 | common | L4 | R0_dinov2s | L2 | True |
| 8 | common | L4 | O1_dinov2b | L2 | True |
| 8 | common | L4 | R1_cvcl | L2 | True |
| 8 | common | L4 | E1_siglip | L1 | True |
| 8 | common | L4 | O2_openclip | L1 | True |
| 8 | common | L4 | E2_vitsup | L1 | True |
| 8 | common | L4 | E3_rn50sup | L2 | True |
| 8 | common | L4 | E4_mae | L4 | True |
| 8 | common | L4 | F1_rn50rand0 | L3 | True |
| 8 | common | L4 | F2_pixels | L4 | True |
| 24 | common | L4 | R0_dinov2s | L1 | True |
| 24 | common | L4 | O1_dinov2b | L3 | True |
| 24 | common | L4 | R1_cvcl | L3 | True |
| 24 | common | L4 | E1_siglip | L1 | True |
| 24 | common | L4 | O2_openclip | L1 | True |
| 24 | common | L4 | E2_vitsup | L1 | True |
| 24 | common | L4 | E3_rn50sup | L3 | True |
| 24 | common | L4 | E4_mae | L4 | True |
| 24 | common | L4 | F1_rn50rand0 | L3 | True |
| 24 | common | L4 | F2_pixels | L3 | True |

Primer peldaño que convergió en cada experimento (pasada 1), copiado de `results/structure_structure.csv` (columnas `first_rung_exp1` … `first_rung_exp12`):

| encoder | items | first_rung_exp1 | first_rung_exp2 | first_rung_exp3 | first_rung_exp4 | first_rung_exp5 | first_rung_exp6 | first_rung_exp7 | first_rung_exp8 | first_rung_exp9 | first_rung_exp10 | first_rung_exp11 | first_rung_exp12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L2 | L1 | L1 |
| O1_dinov2b | 8 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L2 | L1 | L1 |
| R1_cvcl | 8 | L2 | L1 | L1 | L2 | L1 | L1 | L1 | L1 | L1 | L2 | L2 | L2 |
| E1_siglip | 8 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 |
| O2_openclip | 8 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 |
| E2_vitsup | 8 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 |
| E3_rn50sup | 8 | L1 | L2 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L2 | L1 | L1 |
| E4_mae | 8 | L1 | L1 | L4 | L1 | L1 | L1 | L1 | L1 | L1 | L3 | L1 | L3 |
| F1_rn50rand0 | 8 | L1 | L2 | L1 | L3 | L1 | L1 | L1 | L1 | L3 | L2 | L2 | L1 |
| F2_pixels | 8 | L1 | L2 | L4 | L3 | L2 | L1 | L1 | L1 | L3 | L3 | L1 | L2 |
| R0_dinov2s | 24 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 |
| O1_dinov2b | 24 | L1 | L3 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L3 | L1 | L1 |
| R1_cvcl | 24 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L3 | L1 | L1 |
| E1_siglip | 24 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 |
| O2_openclip | 24 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 |
| E2_vitsup | 24 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 |
| E3_rn50sup | 24 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L1 | L3 | L1 | L1 |
| E4_mae | 24 | L1 | L3 | L1 | L3 | L2 | L1 | L1 | L2 | L1 | L3 | L2 | L3 |
| F1_rn50rand0 | 24 | L2 | L2 | L1 | L3 | L2 | L1 | L1 | L2 | L3 | L3 | L2 | L3 |
| F2_pixels | 24 | L1 | L3 | L3 | L3 | L2 | L1 | L1 | L2 | L3 | L3 | L3 | L3 |

En `results/structure_status.json`, `excluded_from_selection` es `[]` para 8 y para 24 ítems, y `scope` es `common` en ambos conjuntos. No existe el archivo `results/structure_DEVIATION.txt`.

En `results/structure_primary.csv` (240 filas: 10 encoders × 2 conjuntos de ítems × 12 experimentos), la columna `variant` es `primary` en todas las filas y la columna `re_used` es `L4` en todas las filas.

En `results/stage2_primary_fit_type_by_order.csv`, variante `primary` (20 filas), `n_L4_1-3` = 6 y `n_L4_3-1` = 6 en todas las filas, y las columnas `n_L1_*`, `n_L2_*`, `n_L3_*` y `n_failed_*` son 0.

## 6. Control positivo por encoder

Modelo y criterio, según el plan §5.3 y el encabezado de `scripts/07_pooled_glmm.R`: con 8 ítems, `selected ~ zd + level + condition + (1 + zd | person) + (1 | image)` (= C6); se exige β(zd) < 0 con un IC de Wald al 95 % que excluya 0. El plan indica que, si un encoder no lo cumple, su familia B no se interpreta; que los pisos se reportan pero el control no funciona como puerta; y que si falla R0 se detiene todo. `08_regions.py` escribe `pasa` / `NO pasa` / `no es puerta (piso)` en la columna `control_positivo` a partir de `gate_pass`.

Filas con `term = zd` en `results/poscontrol.csv` (una por encoder):

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs | gate_pass | zd_lo | zd_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | poscontrol_slope | 8 | zd | -3.15671633846993 | 0.0712580286474666 | -44.2997988912532 | 0 |  | FALSE | 57600 | TRUE | -3.29637950822828 | -3.01705316871157 |
| O1_dinov2b | poscontrol_slope | 8 | zd | -2.84793145160832 | 0.0652405074704201 | -43.652809612181 | 0 |  | FALSE | 57600 | TRUE | -2.97580049658346 | -2.72006240663318 |
| R1_cvcl | poscontrol_slope | 8 | zd | -0.615194351444276 | 0.0460303687521598 | -13.3649668278056 | 9.68781885447289e-41 |  | FALSE | 57600 | TRUE | -0.705412216393607 | -0.524976486494945 |
| E1_siglip | poscontrol_slope | 8 | zd | -3.88739727697533 | 0.0877682553567168 | -44.2916093202009 | 0 |  | FALSE | 57600 | TRUE | -4.05941989646041 | -3.71537465749025 |
| O2_openclip | poscontrol_slope | 8 | zd | -3.99311924987378 | 0.0834175184308162 | -47.869072647918 | 0 |  | FALSE | 57600 | TRUE | -4.15661458167789 | -3.82962391806968 |
| E2_vitsup | poscontrol_slope | 8 | zd | -4.29586609188264 | 0.0848819454302559 | -50.6098920106912 | 0 |  | FALSE | 57600 | TRUE | -4.46223164786363 | -4.12950053590164 |
| E3_rn50sup | poscontrol_slope | 8 | zd | -2.64754952959673 | 0.0650147676736978 | -40.7222793271292 | 0 |  | FALSE | 57600 | TRUE | -2.77497613270042 | -2.52012292649305 |
| E4_mae | poscontrol_noslope | 8 | zd | -0.198244319600915 | 0.0504167220659668 | -3.9321144151642 | 8.42019703912009e-05 |  | FALSE | 57600 | TRUE | -0.297059279068776 | -0.0994293601330547 |
| F1_rn50rand0 | poscontrol_slope | 8 | zd | -1.10238632044123 | 0.0574710408047583 | -19.1815965920345 | 5.27359027651127e-82 |  | FALSE | 57600 | TRUE | -1.21502749057259 | -0.989745150309872 |
| F2_pixels | poscontrol_slope | 8 | zd | 0.115952675577598 | 0.0395417313949472 | 2.93241270645056 | 0.00336339460854474 |  | FALSE | 57600 | FALSE | 0.0384523061571448 | 0.193453044998051 |

Nota sobre E4_mae: en `results/poscontrol.csv` este encoder aparece solo con `model = poscontrol_noslope` (7 filas). Los demás encoders aparecen solo con `model = poscontrol_slope` (7 filas cada uno). El script congelado `scripts/07_pooled_glmm.R` declara, en su encabezado, un fallback como en 04c: `(1|person)+(1|image)` si el ajuste con pendiente aleatoria es singular o no converge, y que ese fallback queda registrado. El criterio de puerta escrito en el mismo script es `beta(zd) < 0` y que el IC de Wald al 95 % excluya 0 (`gate_pass <- (zr$estimate < 0) && (hi < 0)`). `scripts/08_regions.py` (`load_pc`) lee la fila con `term == zd` y toma `gate_pass` de esa fila; para E4_mae esa fila es la de `poscontrol_noslope`.

En `logs/poscontrol.log` cada encoder tiene una línea «… poscontrol items=8 done in … s» (E4_mae: 35.0 s) y el archivo cierra con `wrote …/results/poscontrol.csv 70 rows`. El campo `warn` está vacío en las 70 filas; `singular` es `FALSE` en las 70 filas.

## 7. Análisis primario: regiones por encoder y conclusiones globales G0–G4 (transcripción de `08_regions.py`)

El plan (§7 y §10.3, paso 7 del sábado) encarga las regiones y la conclusión global a `scripts/08_regions.py`. Lo que sigue es la transcripción literal, sin cambios, de `results/regions/regions_report.md` (marca de tiempo del archivo: 21-11-2026, 14:25, hora de Santiago):

# Regiones y conclusión global (plan §7; generado por 08_regions.py)

- Control de tubería: PASA. Perfil del agrupamiento de geometría: depende del corte.
- Referencia descriptiva, no confirmatoria (I-b): Δ en dos etapas de R0 con 8 ítems = 0.019709506140741517 [IC 95 % -0.204779; 0.274221].
- Lectura de R0 con 24 ítems (REVISION3b): inconcluso.

## Tabla única (§7.2): región con δ = 0,05 / 0,10 / 0,20 y con 24 ítems

| encoder | control_positivo | familia_holm | region_d0.05 | region_d0.10 | region_d0.20 | region_24 | perfil |
|---|---|---|---|---|---|---|---|
| R0_dinov2s | pasa | R0: referencia (no entra en Holm; G1) | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso |
| O1_dinov2b | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso |
| R1_cvcl | pasa | familia entrenados | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20) |
| E1_siglip | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso |
| O2_openclip | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso |
| E2_vitsup | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso |
| E3_rn50sup | pasa | familia entrenados | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20) |
| E4_mae | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso |
| F1_rn50rand0 | no es puerta (piso) | familia pisos | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso |
| F2_pixels | no es puerta (piso) | familia pisos | inconcluso | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05); menor que δ (equivalente con margen δ) (δ = 0.10, 0.20) |

## Familias Holm (§7.3; iguales con los tres δ)

- **entrenados:** k = 7 encoders, 14 pruebas por δ; incluidos: O1_dinov2b, R1_cvcl, E1_siglip, O2_openclip, E2_vitsup, E3_rn50sup, E4_mae; excluidos: ninguno.
- **pisos:** k = 2 encoders, 4 pruebas por δ; incluidos: F1_rn50rand0, F2_pixels; excluidos: ninguno.

## Conclusión global por (δ, corte)

| delta | cut | conclusion | detalle | trained_family_size |
|---|---|---|---|---|
| 0.05 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.05 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.05 | 0.7 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.7 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.7 | G1 | R0 en 'inconcluso' | 14 |

**Perfil global:** robusta: G1

## Pisos

| delta | family_size | lectura |
|---|---|---|
| 0.05 | 4 | inconcluso |
| 0.1 | 4 | inconcluso |
| 0.2 | 4 | inconcluso |

## 8. Análisis primario: valores numéricos de la etapa 2 y de la tabla de regiones

En `results/stage2_primary.csv` hay 60 filas: variantes `primary`, `per_encoder_rung` y `S9_mixed` (20 filas cada una: 10 encoders × {8, 24} ítems). En las 60 filas, `status = ok`, `n_exp = 12`, `suffix` está vacío, `ci95_contiguous = True`, `ci95_at_grid_edge = False`, `ci90_contiguous = True` y `ci90_at_grid_edge = False`.

### 8.1 Variante `primary` (la que usa `08_regions.py` para las regiones del §7)

Valores copiados de `results/stage2_primary.csv`, filas con `variant = primary`:

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | primary |  | 12 | ok | 0.019709506140741517 | -0.4664332924026316 | -0.4861427985433731 | 0.8701298701298701 | 0.566017316017316 | 0.43506493506493504 | -0.204779 | 0.274221 | True | False | -0.188779 | 0.219221 | True | False | 0.41233766233766234 | 0.20995670995670995 | 0.06277056277056277 | 0.8528183424870859 | 5.15260513490606 | -0.23775688939703105 | 0.2771759016785141 |
| O1_dinov2b | 8 | primary |  | 12 | ok | -0.06252793792186948 | -0.20309252749609086 | -0.14056458957422138 | 0.6493506493506493 | 0.3246753246753247 | 0.6764069264069265 | -0.369664 | 0.244336 | True | False | -0.345664 | 0.220336 | True | False | 0.5422077922077922 | 0.38311688311688313 | 0.14285714285714285 | 0.6544490011304713 | 5.169806037219722 | -0.3978842470180155 | 0.2728283711742765 |
| R1_cvcl | 8 | primary |  | 12 | ok | 0.06538407185209291 | -0.2081306207958841 | -0.273514692647977 | 0.25757575757575757 | 0.8722943722943723 | 0.12878787878787878 | -0.057869 | 0.187131 | True | False | -0.034869 | 0.161131 | True | False | 0.6028138528138528 | 0.26731601731601734 | 0.0183982683982684 | 0.2618098222559345 | 8.306256796791628 | -0.05905544994933409 | 0.18982359365351992 |
| E1_siglip | 8 | primary |  | 12 | ok | -0.04398131521277637 | -1.4174121554382548 | -1.3734308402254785 | 0.7467532467532467 | 0.37337662337662336 | 0.6277056277056277 | -0.352378 | 0.269622 | True | False | -0.296378 | 0.209622 | True | False | 0.48917748917748916 | 0.354978354978355 | 0.14935064935064934 | 0.7583144748129481 | 9.784097429386227 | -0.3545918664616068 | 0.26662923603605404 |
| O2_openclip | 8 | primary |  | 12 | ok | -0.057093100930458496 | -1.393135599176052 | -1.3360424982455934 | 0.6428571428571429 | 0.32142857142857145 | 0.6796536796536796 | -0.284165 | 0.208835 | True | False | -0.252165 | 0.148835 | True | False | 0.5216450216450217 | 0.3874458874458874 | 0.12987012987012986 | 0.6328310721184006 | 6.5537885585217195 | -0.3303891322642305 | 0.2162029304033135 |
| E2_vitsup | 8 | primary |  | 12 | ok | -0.03377055754553421 | -0.45184421850129625 | -0.41807366095576204 | 0.8008658008658008 | 0.4004329004329004 | 0.6006493506493507 | -0.317971 | 0.265029 | True | False | -0.283971 | 0.208029 | True | False | 0.4512987012987013 | 0.31277056277056275 | 0.10606060606060606 | 0.7956698352358718 | 5.630695680212448 | -0.3430130982366141 | 0.27547198314554566 |
| E3_rn50sup | 8 | primary |  | 12 | ok | -0.02552765438240856 | -0.9879997187809871 | -0.9624720643985786 | 0.7142857142857143 | 0.35714285714285715 | 0.6439393939393939 | -0.172212 | 0.126788 | True | False | -0.148212 | 0.106788 | True | False | 0.37337662337662336 | 0.16233766233766234 | 0.008658008658008658 | 0.7216468732663126 | 7.231559368159705 | -0.1874134413315448 | 0.13635813256672769 |
| E4_mae | 8 | primary |  | 12 | ok | 0.4626714760011359 | -0.7810047056813518 | -1.2436761816824877 | 0.17316017316017315 | 0.9145021645021645 | 0.08658008658008658 | -0.227101 | 1.179899 | True | False | -0.115101 | 1.033899 | True | False | 0.8928571428571429 | 0.8668831168831169 | 0.79004329004329 | 0.17535351746130598 | 8.730488556860223 | -0.2502932137708529 | 1.1756361657731247 |
| F1_rn50rand0 | 8 | primary |  | 12 | ok | -0.04646227329427677 | -0.42759618431617447 | -0.3811339110218977 | 0.6103896103896104 | 0.3051948051948052 | 0.6958874458874459 | -0.243163 | 0.147837 | True | False | -0.216163 | 0.114837 | True | False | 0.48917748917748916 | 0.275974025974026 | 0.06277056277056277 | 0.6095068266426835 | 9.710117889677846 | -0.2434135980153684 | 0.15048905142681487 |
| F2_pixels | 8 | primary |  | 12 | ok | 0.011831747062335474 | -0.13551391442344138 | -0.14734566148577685 | 0.7380952380952381 | 0.6320346320346321 | 0.36904761904761907 | -0.065716 | 0.088284 | True | False | -0.050716 | 0.072284 | True | False | 0.14393939393939395 | 0.016233766233766232 | 0.0010822510822510823 | 0.7386040564873767 | 9.611582238251094 | -0.06533419180200452 | 0.08899768592667547 |
| R0_dinov2s | 24 | primary |  | 12 | ok | 0.26097602151935506 | -2.0792282143135545 | -2.3402042358329096 | 0.09740259740259741 | 0.9523809523809523 | 0.048701298701298704 | -0.061995 | 0.599005 | True | False | 0.007005 | 0.524005 | True | False | 0.920995670995671 | 0.8571428571428571 | 0.6677489177489178 | 0.103550441464701 | 7.2186282162131725 | -0.06833701331047004 | 0.5902890563491802 |
| O1_dinov2b | 24 | primary |  | 12 | ok | 0.3180325426462671 | -2.109953784781585 | -2.427986327427852 | 0.09956709956709957 | 0.9512987012987013 | 0.049783549783549784 | -0.093925 | 0.735075 | True | False | 0.002075 | 0.640075 | True | False | 0.922077922077922 | 0.8766233766233766 | 0.7445887445887446 | 0.11174328603303474 | 7.699700400874926 | -0.09318073131536125 | 0.7292458166078954 |
| R1_cvcl | 24 | primary |  | 12 | ok | 0.14109549978318192 | -2.713122409005985 | -2.854217908789167 | 0.3116883116883117 | 0.8452380952380952 | 0.15584415584415584 | -0.178392 | 0.451608 | True | False | -0.105392 | 0.363608 | True | False | 0.7467532467532467 | 0.6222943722943723 | 0.33116883116883117 | 0.32028130464795107 | 8.52362899300422 | -0.16395610105178016 | 0.446147100618144 |
| E1_siglip | 24 | primary |  | 12 | ok | 0.40420773994579706 | -3.1904778286166287 | -3.5946855685624257 | 0.2943722943722944 | 0.8538961038961039 | 0.1471861471861472 | -0.393906 | 1.236094 | True | False | -0.235906 | 1.058094 | True | False | 0.8181818181818182 | 0.7813852813852814 | 0.7002164502164502 | 0.29941467474698197 | 6.33110091036604 | -0.4599984548604622 | 1.2684139347520564 |
| O2_openclip | 24 | primary |  | 12 | ok | 0.33481022295729623 | -3.2746924979955954 | -3.6095027209528916 | 0.36363636363636365 | 0.8192640692640693 | 0.18181818181818182 | -0.513417 | 1.157583 | True | False | -0.331417 | 0.997583 | True | False | 0.7813852813852814 | 0.7337662337662337 | 0.645021645021645 | 0.3771157041782212 | 6.797627210069346 | -0.5082169296687348 | 1.1778373755833274 |
| E2_vitsup | 24 | primary |  | 12 | ok | 0.5159409556551604 | -2.903542693740825 | -3.4194836493959855 | 0.11688311688311688 | 0.9426406926406926 | 0.05844155844155844 | -0.202687 | 1.235313 | True | False | -0.043687 | 1.087313 | True | False | 0.9274891774891775 | 0.9025974025974026 | 0.8387445887445888 | 0.13377574768780812 | 6.634791712001824 | -0.20646058535686873 | 1.2383424966671894 |
| E3_rn50sup | 24 | primary |  | 12 | ok | 0.2546675783780534 | -2.494423642289623 | -2.7490912206676765 | 0.17316017316017315 | 0.9145021645021645 | 0.08658008658008658 | -0.14562 | 0.66238 | True | False | -0.07262 | 0.57838 | True | False | 0.8712121212121212 | 0.8106060606060606 | 0.6233766233766234 | 0.18315618477717394 | 7.19783793246414 | -0.1519628302966049 | 0.6612979870527118 |
| E4_mae | 24 | primary |  | 12 | ok | 0.45321081225947424 | -6.05248985214664 | -6.505700664406114 | 0.5497835497835498 | 0.7261904761904762 | 0.2748917748917749 | -0.975047 | 2.061953 | True | False | -0.743047 | 1.680953 | True | False | 0.6991341991341992 | 0.6807359307359307 | 0.6331168831168831 | 0.5233450239639835 | 8.432184189989432 | -1.1026439284164136 | 2.009065552935362 |
| F1_rn50rand0 | 24 | primary |  | 12 | ok | 0.043846222194020346 | -2.1195541537866665 | -2.163400375980687 | 0.683982683982684 | 0.6590909090909091 | 0.341991341991342 | -0.178596 | 0.277404 | True | False | -0.144596 | 0.231404 | True | False | 0.4805194805194805 | 0.29653679653679654 | 0.08116883116883117 | 0.6814212810093624 | 9.264885618923069 | -0.18921086798568246 | 0.2769033123737231 |
| F2_pixels | 24 | primary |  | 12 | ok | 0.04646552785080016 | -1.4619017967525616 | -1.5083673246033618 | 0.38961038961038963 | 0.8062770562770563 | 0.19480519480519481 | -0.062793 | 0.160207 | True | False | -0.042793 | 0.135207 | True | False | 0.474025974025974 | 0.15043290043290045 | 0.006493506493506494 | 0.3753033129217673 | 8.147966846862007 | -0.06742924841005657 | 0.1603603041116569 |

### 8.2 Columnas numéricas de `results/regions/regions_table.csv`

Estas columnas son las que `08_regions.py` copió o calculó (Holm) a partir de la etapa 2 primaria y del control positivo. Las etiquetas de región ya están en el informe del §7. Las celdas vacías de Holm en R0_dinov2s están vacías en el CSV (R0 no entra en Holm).

| encoder | status | delta_hat | ci95_lo | ci95_hi | ci90_lo | ci90_hi | p_opposite | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_ci95_lo | welch_ci95_hi | holm_tost_d0.05 | holm_opposite_d0.05 | holm_tost_d0.10 | holm_opposite_d0.10 | holm_tost_d0.20 | holm_opposite_d0.20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | ok | 0.019709506140741517 | -0.204779 | 0.274221 | -0.188779 | 0.219221 | 0.566017316017316 | 0.41233766233766234 | 0.20995670995670995 | 0.06277056277056277 | -0.23775688939703105 | 0.2771759016785141 |  |  |  |  |  |  |
| O1_dinov2b | ok | -0.06252793792186948 | -0.369664 | 0.244336 | -0.345664 | 0.220336 | 0.3246753246753247 | 0.5422077922077922 | 0.38311688311688313 | 0.14285714285714285 | -0.3978842470180155 | 0.2728283711742765 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| R1_cvcl | ok | 0.06538407185209291 | -0.057869 | 0.187131 | -0.034869 | 0.161131 | 0.8722943722943723 | 0.6028138528138528 | 0.26731601731601734 | 0.0183982683982684 | -0.05905544994933409 | 0.18982359365351992 | 1.0 | 1.0 | 1.0 | 1.0 | 0.2391774891774892 | 1.0 |
| E1_siglip | ok | -0.04398131521277637 | -0.352378 | 0.269622 | -0.296378 | 0.209622 | 0.37337662337662336 | 0.48917748917748916 | 0.354978354978355 | 0.14935064935064934 | -0.3545918664616068 | 0.26662923603605404 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| O2_openclip | ok | -0.057093100930458496 | -0.284165 | 0.208835 | -0.252165 | 0.148835 | 0.32142857142857145 | 0.5216450216450217 | 0.3874458874458874 | 0.12987012987012986 | -0.3303891322642305 | 0.2162029304033135 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| E2_vitsup | ok | -0.03377055754553421 | -0.317971 | 0.265029 | -0.283971 | 0.208029 | 0.4004329004329004 | 0.4512987012987013 | 0.31277056277056275 | 0.10606060606060606 | -0.3430130982366141 | 0.27547198314554566 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| E3_rn50sup | ok | -0.02552765438240856 | -0.172212 | 0.126788 | -0.148212 | 0.106788 | 0.35714285714285715 | 0.37337662337662336 | 0.16233766233766234 | 0.008658008658008658 | -0.1874134413315448 | 0.13635813256672769 | 1.0 | 1.0 | 1.0 | 1.0 | 0.12121212121212122 | 1.0 |
| E4_mae | ok | 0.4626714760011359 | -0.227101 | 1.179899 | -0.115101 | 1.033899 | 0.9145021645021645 | 0.8928571428571429 | 0.8668831168831169 | 0.79004329004329 | -0.2502932137708529 | 1.1756361657731247 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| F1_rn50rand0 | ok | -0.04646227329427677 | -0.243163 | 0.147837 | -0.216163 | 0.114837 | 0.3051948051948052 | 0.48917748917748916 | 0.275974025974026 | 0.06277056277056277 | -0.2434135980153684 | 0.15048905142681487 | 0.9783549783549783 | 0.9155844155844155 | 0.827922077922078 | 0.827922077922078 | 0.18831168831168832 | 0.6103896103896104 |
| F2_pixels | ok | 0.011831747062335474 | -0.065716 | 0.088284 | -0.050716 | 0.072284 | 0.6320346320346321 | 0.14393939393939395 | 0.016233766233766232 | 0.0010822510822510823 | -0.06533419180200452 | 0.08899768592667547 | 0.5757575757575758 | 0.9783549783549783 | 0.06493506493506493 | 0.827922077922078 | 0.004329004329004329 | 0.6320346320346321 |

### 8.3 Variante descriptiva `per_encoder_rung` (no entra en las regiones, plan §5.2)

Valores copiados de `results/stage2_primary.csv`, filas con `variant = per_encoder_rung`:

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | per_encoder_rung |  | 12 | ok | -0.3844898115064337 | -1.8935481343216285 | -1.5090583228151948 | 0.06277056277056277 | 0.031385281385281384 | 0.9696969696969697 | -0.7855 | 0.0195 | True | False | -0.6965 | -0.0765 | True | False | 0.9588744588744589 | 0.9383116883116883 | 0.8603896103896104 | 0.0522667504965275 | 7.462388185996454 | -0.7738582264391076 | 0.004878603426240302 |
| O1_dinov2b | 8 | per_encoder_rung |  | 12 | ok | -0.6521191875154587 | -1.4822383179037333 | -0.8301191303882746 | 0.010822510822510822 | 0.005411255411255411 | 0.9956709956709957 | -1.091487 | -0.211487 | True | False | -1.002487 | -0.288487 | True | False | 0.9924242424242424 | 0.9891774891774892 | 0.9794372294372294 | 0.009149129171858937 | 9.237647655272337 | -1.0996249628375139 | -0.20461341219340362 |
| R1_cvcl | 8 | per_encoder_rung |  | 12 | ok | -0.03146932244303785 | -0.38601598363407286 | -0.354546661191035 | 0.7857142857142857 | 0.39285714285714285 | 0.6082251082251082 | -0.273451 | 0.191549 | True | False | -0.220451 | 0.157549 | True | False | 0.43614718614718617 | 0.26731601731601734 | 0.0670995670995671 | 0.774447309534704 | 6.1045032853375805 | -0.2874545225386801 | 0.2245158776526044 |
| E1_siglip | 8 | per_encoder_rung |  | 12 | ok | -0.2504724433360428 | -3.240688683355829 | -2.990216240019786 | 0.4393939393939394 | 0.2196969696969697 | 0.7813852813852814 | -0.906327 | 0.393673 | True | False | -0.785327 | 0.271673 | True | False | 0.7316017316017316 | 0.6872294372294372 | 0.564935064935065 | 0.41727798752624734 | 8.951794464939006 | -0.9173194327695273 | 0.41637454609744173 |
| O2_openclip | 8 | per_encoder_rung |  | 12 | ok | -0.13434210835302807 | -2.535923611884695 | -2.4015815035316668 | 0.551948051948052 | 0.275974025974026 | 0.7251082251082251 | -0.577975 | 0.303025 | True | False | -0.509975 | 0.229025 | True | False | 0.6461038961038961 | 0.5703463203463204 | 0.3722943722943723 | 0.5357472117544192 | 7.063185255818458 | -0.6215564413664945 | 0.3528722246604384 |
| E2_vitsup | 8 | per_encoder_rung |  | 12 | ok | -0.6844568289567996 | -2.04867840639652 | -1.3642215774397206 | 0.0367965367965368 | 0.0183982683982684 | 0.9826839826839827 | -1.382562 | -0.052562 | True | False | -1.233562 | -0.154562 | True | False | 0.9761904761904762 | 0.9653679653679653 | 0.9372294372294372 | 0.05011964121948533 | 8.92784099164095 | -1.3693571457175047 | 0.0004434878039054624 |
| E3_rn50sup | 8 | per_encoder_rung |  | 12 | ok | -0.23303235064860828 | -2.1630937463697486 | -1.9300613957211403 | 0.3528138528138528 | 0.1764069264069264 | 0.8246753246753247 | -0.771237 | 0.304763 | True | False | -0.661237 | 0.204763 | True | False | 0.7738095238095238 | 0.7067099567099567 | 0.5573593073593074 | 0.3538595896731038 | 9.775246666946442 | -0.7681565628808912 | 0.3020918615836746 |
| E4_mae | 8 | per_encoder_rung |  | 12 | ok | 0.4626714760011359 | -0.7810047056813518 | -1.2436761816824877 | 0.17316017316017315 | 0.9145021645021645 | 0.08658008658008658 | -0.227101 | 1.179899 | True | False | -0.115101 | 1.033899 | True | False | 0.8928571428571429 | 0.8668831168831169 | 0.79004329004329 | 0.17535351746130598 | 8.730488556860223 | -0.2502932137708529 | 1.1756361657731247 |
| F1_rn50rand0 | 8 | per_encoder_rung |  | 12 | ok | -0.05911260613262559 | -0.7131404293054909 | -0.6540278231728653 | 0.7683982683982684 | 0.3841991341991342 | 0.6168831168831169 | -0.465934 | 0.293066 | True | False | -0.384934 | 0.239066 | True | False | 0.5183982683982684 | 0.4134199134199134 | 0.21536796536796537 | 0.7470728126312393 | 7.89428091953999 | -0.4681810499587712 | 0.34995583769352 |
| F2_pixels | 8 | per_encoder_rung |  | 12 | ok | 0.011831747062335474 | -0.13551391442344138 | -0.14734566148577685 | 0.7380952380952381 | 0.6320346320346321 | 0.36904761904761907 | -0.065716 | 0.088284 | True | False | -0.050716 | 0.072284 | True | False | 0.14393939393939395 | 0.016233766233766232 | 0.0010822510822510823 | 0.7386040564873767 | 9.611582238251094 | -0.06533419180200452 | 0.08899768592667547 |
| R0_dinov2s | 24 | per_encoder_rung |  | 12 | ok | 0.4110574248406449 | -3.3887226633892236 | -3.7997800882298685 | 0.21645021645021645 | 0.8928571428571429 | 0.10822510822510822 | -0.322794 | 1.126206 | True | False | -0.175794 | 1.000206 | True | False | 0.8647186147186147 | 0.8344155844155844 | 0.7402597402597403 | 0.23612978541482244 | 6.348560330290509 | -0.3476043010851103 | 1.1697191507664 |
| O1_dinov2b | 24 | per_encoder_rung |  | 12 | ok | 0.3802857559024604 | -3.346092478704272 | -3.7263782346067322 | 0.3463203463203463 | 0.827922077922078 | 0.17316017316017315 | -0.555356 | 1.309644 | True | False | -0.358356 | 1.086644 | True | False | 0.7954545454545454 | 0.7554112554112554 | 0.6774891774891775 | 0.36563905203599645 | 6.766187939978021 | -0.5537187299669453 | 1.314290241771866 |
| R1_cvcl | 24 | per_encoder_rung |  | 12 | ok | 0.3791497026296584 | -3.3140962550374584 | -3.693245957667117 | 0.17316017316017315 | 0.9145021645021645 | 0.08658008658008658 | -0.257768 | 1.011232 | True | False | -0.132768 | 0.859232 | True | False | 0.8798701298701299 | 0.8474025974025974 | 0.7391774891774892 | 0.20082198506556156 | 7.256172127536491 | -0.25357915207352355 | 1.0118785573328404 |
| E1_siglip | 24 | per_encoder_rung |  | 12 | ok | 1.0284754900542188 | -5.8251370799052475 | -6.853612569959466 | 0.25757575757575757 | 0.8722943722943723 | 0.12878787878787878 | -1.058667 | 3.075333 | True | False | -0.617667 | 2.715333 | True | False | 0.8560606060606061 | 0.841991341991342 | 0.8062770562770563 | 0.29335998702480215 | 6.253148728687513 | -1.1442023701267092 | 3.201153350235147 |
| O2_openclip | 24 | per_encoder_rung |  | 12 | ok | 0.9233837866514971 | -4.955694404665041 | -5.879078191316538 | 0.23160173160173161 | 0.8852813852813853 | 0.11580086580086581 | -0.81009 | 2.62091 | True | False | -0.45009 | 2.32591 | True | False | 0.8722943722943723 | 0.8560606060606061 | 0.8235930735930735 | 0.2562761138141692 | 6.1457427081002365 | -0.871654471746796 | 2.71842204504979 |
| E2_vitsup | 24 | per_encoder_rung |  | 12 | ok | 1.0041299507014205 | -4.655254777001935 | -5.6593847277033555 | 0.18181818181818182 | 0.9101731601731602 | 0.09090909090909091 | -0.590246 | 2.658754 | True | False | -0.326246 | 2.331754 | True | False | 0.9015151515151515 | 0.895021645021645 | 0.8712121212121212 | 0.19498306492414944 | 6.217214289610578 | -0.6736094898010772 | 2.681869391203918 |
| E3_rn50sup | 24 | per_encoder_rung |  | 12 | ok | 0.9328304137629204 | -3.9770781029764612 | -4.909908516739382 | 0.2012987012987013 | 0.9004329004329005 | 0.10064935064935066 | -0.655558 | 2.587442 | True | False | -0.415558 | 2.203442 | True | False | 0.8917748917748918 | 0.8755411255411255 | 0.8538961038961039 | 0.21884284993340372 | 6.542276412915665 | -0.712414523771719 | 2.57807535129756 |
| E4_mae | 24 | per_encoder_rung |  | 12 | ok | 0.45321081225947424 | -6.05248985214664 | -6.505700664406114 | 0.5497835497835498 | 0.7261904761904762 | 0.2748917748917749 | -0.975047 | 2.061953 | True | False | -0.743047 | 1.680953 | True | False | 0.6991341991341992 | 0.6807359307359307 | 0.6331168831168831 | 0.5233450239639835 | 8.432184189989432 | -1.1026439284164136 | 2.009065552935362 |
| F1_rn50rand0 | 24 | per_encoder_rung |  | 12 | ok | -0.032731613304168405 | -3.0269701895901218 | -2.9942385762859534 | 0.8116883116883117 | 0.40584415584415584 | 0.5952380952380952 | -0.317214 | 0.260786 | True | False | -0.264214 | 0.197786 | True | False | 0.4491341991341991 | 0.30952380952380953 | 0.11147186147186147 | 0.8049590149826287 | 9.0561411968234 | -0.32364790297415896 | 0.25818467636582215 |
| F2_pixels | 24 | per_encoder_rung |  | 12 | ok | 0.1398569442307842 | -2.3827040498412972 | -2.5225609940720815 | 0.08874458874458875 | 0.9567099567099567 | 0.044372294372294376 | -0.027816 | 0.303184 | True | False | 0.013184 | 0.258184 | True | False | 0.8820346320346321 | 0.7067099567099567 | 0.21103896103896103 | 0.08648470053988959 | 7.554703897679303 | -0.025560071358997433 | 0.3052739598205658 |

## 9. Sensibilidades S1–S8

El plan (§5.4 y §7.4) establece que S1–S9 y la familia A no cambian la región, y que si contradicen la conclusión se informa (línea 339 del plan). `08_regions.py` no se aplicó a S1, S2, S3, S4, S5, S7, S8, F0 ni S6: solo se corrió el sábado sobre el primario (`results/regions/`) y el domingo otra vez con S9 (`results/regions_with_S9/`). Este documento no asigna regiones a S1–S8, F0 ni S6. Se transcriben las tablas que sí produjo la tubería.

### 9.1 S1 (GLMM publicado Mb y Mc, 24 y 8 ítems)

El plan §5.4 pide, para S1, `zd:order3-1`, LR χ²(1) y χ²(3). Las tablas completas (todos los términos de Mb y Mc) están en `results/S1_24.csv` y `results/S1_8.csv` (350 filas cada uno; 10 encoders). En ambos archivos el campo `warn` está vacío en todas las filas. `nobs` es 172800 en `S1_24.csv` y 57600 en `S1_8.csv`. `singular` es `FALSE` en las filas de los modelos Mb y Mc, y vale `NA` en las filas cuyo `term` es `LR_chi2(1)` o `LR_chi2(3)`.

**8 ítems** (`results/S1_8.csv`). Términos copiados: `S1_Mb` / `zd:order3-1`; `S1_Mb_vs_Mbase` / `LR_chi2(1)` (el estadístico está en `estimate` y el p en `p`; `se`, `z` y `singular` valen `NA` en el CSV); `S1_Mc` / `zd:order3-1`; `S1_Mc_vs_Mb` / `LR_chi2(3)`.

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mb | 8 | zd:order3-1 | 0.0507459871147492 | 0.0589847264787656 | 0.860324191433144 | 0.389610361267392 |  | FALSE | 57600 |
| O1_dinov2b | S1_Mb | 8 | zd:order3-1 | 0.0119469916230766 | 0.0555407690758964 | 0.215103100332498 | 0.829686936767973 |  | FALSE | 57600 |
| R1_cvcl | S1_Mb | 8 | zd:order3-1 | -0.0575343967824318 | 0.0628803661087328 | -0.914981898848095 | 0.360201132865319 |  | FALSE | 57600 |
| E1_siglip | S1_Mb | 8 | zd:order3-1 | 0.00631892333233533 | 0.0783910688557334 | 0.0806076945316861 | 0.935753946769907 |  | FALSE | 57600 |
| O2_openclip | S1_Mb | 8 | zd:order3-1 | 0.0644112717369619 | 0.0824038609757534 | 0.781653565430804 | 0.434418197773403 |  | FALSE | 57600 |
| E2_vitsup | S1_Mb | 8 | zd:order3-1 | 0.0306781946400253 | 0.0739231881691324 | 0.41500096789434 | 0.678141194574905 |  | FALSE | 57600 |
| E3_rn50sup | S1_Mb | 8 | zd:order3-1 | 0.0897618201155627 | 0.0737860354560663 | 1.21651501616466 | 0.223788794349524 |  | FALSE | 57600 |
| E4_mae | S1_Mb | 8 | zd:order3-1 | -0.114264391115321 | 0.095535494177543 | -1.19604124204322 | 0.231680465269207 |  | FALSE | 57600 |
| F1_rn50rand0 | S1_Mb | 8 | zd:order3-1 | -0.132488921723883 | 0.0639267067975093 | -2.07251285669303 | 0.0382176385114608 |  | FALSE | 57600 |
| F2_pixels | S1_Mb | 8 | zd:order3-1 | -0.00579002313542157 | 0.0500436861735584 | -0.115699373450248 | 0.90789080283264 |  | FALSE | 57600 |

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 0.748780063422601 | NA | NA | 0.386862744963945 |  | NA | 57600 |
| O1_dinov2b | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 0.0479628488428716 | NA | NA | 0.826646759651746 |  | NA | 57600 |
| R1_cvcl | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 0.847897368614213 | NA | NA | 0.357147843264201 |  | NA | 57600 |
| E1_siglip | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 0.07040065781257 | NA | NA | 0.790754313853656 |  | NA | 57600 |
| O2_openclip | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 0.615832180010329 | NA | NA | 0.432600303796173 |  | NA | 57600 |
| E2_vitsup | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 0.172149057973002 | NA | NA | 0.678208891908389 |  | NA | 57600 |
| E3_rn50sup | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 1.48283091765552 | NA | NA | 0.223332123724871 |  | NA | 57600 |
| E4_mae | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 1.68239503030418 | NA | NA | 0.19460651308355 |  | NA | 57600 |
| F1_rn50rand0 | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | 4.48128288029693 | NA | NA | 0.034267989412845 |  | NA | 57600 |
| F2_pixels | S1_Mb_vs_Mbase | 8 | LR_chi2(1) | -0.316823402135924 | NA | NA | 1 |  | NA | 57600 |

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mc | 8 | zd:order3-1 | -0.382412744883853 | 0.0799997652177805 | -4.7801733397945 | 1.75144120001336e-06 |  | FALSE | 57600 |
| O1_dinov2b | S1_Mc | 8 | zd:order3-1 | -0.417421019858338 | 0.0714687523866158 | -5.84060874045018 | 5.20104237433795e-09 |  | FALSE | 57600 |
| R1_cvcl | S1_Mc | 8 | zd:order3-1 | -0.49653830913722 | 0.0855776504665775 | -5.80219609243822 | 6.54519525380649e-09 |  | FALSE | 57600 |
| E1_siglip | S1_Mc | 8 | zd:order3-1 | -0.83843913985218 | 0.119338905994089 | -7.02569822362632 | 2.12998379531342e-12 |  | FALSE | 57600 |
| O2_openclip | S1_Mc | 8 | zd:order3-1 | -0.594365526754019 | 0.117363570649909 | -5.06431018980317 | 4.0988179177703e-07 |  | FALSE | 57600 |
| E2_vitsup | S1_Mc | 8 | zd:order3-1 | -0.698986894181474 | 0.10096102285243 | -6.92333411878314 | 4.41135640260667e-12 |  | FALSE | 57600 |
| E3_rn50sup | S1_Mc | 8 | zd:order3-1 | -0.562065983554224 | 0.0992176648942421 | -5.66497895463817 | 1.47042582549029e-08 |  | FALSE | 57600 |
| E4_mae | S1_Mc | 8 | zd:order3-1 | -2.24832319752663 | 0.219044545653273 | -10.2642281770645 | 1.02134415481989e-24 |  | FALSE | 57600 |
| F1_rn50rand0 | S1_Mc | 8 | zd:order3-1 | -0.980114776034046 | 0.122061153025464 | -8.02970274932253 | 9.77091118672104e-16 |  | FALSE | 57600 |
| F2_pixels | S1_Mc | 8 | zd:order3-1 | -0.678777042496934 | 0.117456841164769 | -5.77894855476952 | 7.51689048564337e-09 |  | FALSE | 57600 |

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 79.5976223690595 | NA | NA | 3.74406054859366e-17 |  | NA | 57600 |
| O1_dinov2b | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 98.4686311852056 | NA | NA | 3.31697818015579e-21 |  | NA | 57600 |
| R1_cvcl | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 82.3801301597923 | NA | NA | 9.471360400982e-18 |  | NA | 57600 |
| E1_siglip | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 149.796434983062 | NA | NA | 2.91527133460534e-32 |  | NA | 57600 |
| O2_openclip | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 107.448065495901 | NA | NA | 3.88575420530799e-23 |  | NA | 57600 |
| E2_vitsup | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 143.896324558325 | NA | NA | 5.46087592559799e-31 |  | NA | 57600 |
| E3_rn50sup | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 126.833366458497 | NA | NA | 2.60270989860918e-27 |  | NA | 57600 |
| E4_mae | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 198.614309934863 | NA | NA | 8.40555233582237e-43 |  | NA | 57600 |
| F1_rn50rand0 | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 97.5659259593194 | NA | NA | 5.18563934404192e-21 |  | NA | 57600 |
| F2_pixels | S1_Mc_vs_Mb | 8 | LR_chi2(3) | 58.8650124048218 | NA | NA | 1.02728756734259e-12 |  | NA | 57600 |

**24 ítems** (`results/S1_24.csv`), mismos términos:

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mb | 24 | zd:order3-1 | 0.204059697411178 | 0.04103941057062 | 4.97228626273847 | 6.61679037631903e-07 |  | FALSE | 172800 |
| O1_dinov2b | S1_Mb | 24 | zd:order3-1 | 0.165887772332774 | 0.0385920257339173 | 4.29849869702436 | 1.71958911903731e-05 |  | FALSE | 172800 |
| R1_cvcl | S1_Mb | 24 | zd:order3-1 | 0.201190238470561 | 0.0558280627234791 | 3.60374744628113 | 0.000313661848428609 |  | FALSE | 172800 |
| E1_siglip | S1_Mb | 24 | zd:order3-1 | 0.235884412742741 | 0.0596939701960802 | 3.95156180712923 | 7.76428131022164e-05 |  | FALSE | 172800 |
| O2_openclip | S1_Mb | 24 | zd:order3-1 | 0.318372758784357 | 0.0634810239289064 | 5.01524296679443 | 5.29664895452628e-07 |  | FALSE | 172800 |
| E2_vitsup | S1_Mb | 24 | zd:order3-1 | 0.380474459249501 | 0.0606804360877441 | 6.27013389784037 | 3.60737761305895e-10 |  | FALSE | 172800 |
| E3_rn50sup | S1_Mb | 24 | zd:order3-1 | 0.314333336032625 | 0.0586031174235487 | 5.36376475948897 | 8.1505038326825e-08 |  | FALSE | 172800 |
| E4_mae | S1_Mb | 24 | zd:order3-1 | -0.0413862316000523 | 0.0549075179498921 | -0.753744353147065 | 0.451002743767436 |  | FALSE | 172800 |
| F1_rn50rand0 | S1_Mb | 24 | zd:order3-1 | -0.10514956825011 | 0.0334823808361671 | -3.14044478391838 | 0.00168691503799284 |  | FALSE | 172800 |
| F2_pixels | S1_Mb | 24 | zd:order3-1 | 0.0141335876082879 | 0.0232770082537062 | 0.60719090074806 | 0.543724230160278 |  | FALSE | 172800 |

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 24.5859182660133 | NA | NA | 7.10691842446903e-07 |  | NA | 172800 |
| O1_dinov2b | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 18.3612749493914 | NA | NA | 1.82734397536629e-05 |  | NA | 172800 |
| R1_cvcl | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 12.9481753710934 | NA | NA | 0.000320233458778358 |  | NA | 172800 |
| E1_siglip | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 15.5577915148679 | NA | NA | 8.00211815964515e-05 |  | NA | 172800 |
| O2_openclip | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 25.0714639961152 | NA | NA | 5.52443727645971e-07 |  | NA | 172800 |
| E2_vitsup | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 39.3143372862833 | NA | NA | 3.60782438209104e-10 |  | NA | 172800 |
| E3_rn50sup | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 28.6463984693255 | NA | NA | 8.68754171674752e-08 |  | NA | 172800 |
| E4_mae | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 0.604390302614775 | NA | NA | 0.436907812627889 |  | NA | 172800 |
| F1_rn50rand0 | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 9.75575928222679 | NA | NA | 0.00178761794733718 |  | NA | 172800 |
| F2_pixels | S1_Mb_vs_Mbase | 24 | LR_chi2(1) | 0.367437945707934 | NA | NA | 0.544404130027618 |  | NA | 172800 |

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mc | 24 | zd:order3-1 | 0.0613844122807865 | 0.0531536378629684 | 1.15484875069205 | 0.248152369368123 |  | FALSE | 172800 |
| O1_dinov2b | S1_Mc | 24 | zd:order3-1 | -0.00955333140933524 | 0.0479329406109372 | -0.199306182503549 | 0.842023243254113 |  | FALSE | 172800 |
| R1_cvcl | S1_Mc | 24 | zd:order3-1 | -0.0264634636019098 | 0.0736693357188851 | -0.35921952252823 | 0.719430873116019 |  | FALSE | 172800 |
| E1_siglip | S1_Mc | 24 | zd:order3-1 | -0.177258852251061 | 0.0886080089525631 | -2.0004834139311 | 0.0454480891537207 |  | FALSE | 172800 |
| O2_openclip | S1_Mc | 24 | zd:order3-1 | 0.0176991464839675 | 0.0863975294066952 | 0.204857090306982 | 0.83768378113558 |  | FALSE | 172800 |
| E2_vitsup | S1_Mc | 24 | zd:order3-1 | 0.180872018958167 | 0.0768923879469764 | 2.35227470218369 | 0.0186589893845566 |  | FALSE | 172800 |
| E3_rn50sup | S1_Mc | 24 | zd:order3-1 | 0.117321989174975 | 0.0789590519934446 | 1.48585863448203 | 0.137316517028249 |  | FALSE | 172800 |
| E4_mae | S1_Mc | 24 | zd:order3-1 | -3.41371979760629 | 0.21523939647697 | -15.8601067159726 | 1.19690335344235e-56 |  | FALSE | 172800 |
| F1_rn50rand0 | S1_Mc | 24 | zd:order3-1 | -1.1289904635986 | 0.0782330980298859 | -14.4311102593344 | 3.29730216951523e-47 |  | FALSE | 172800 |
| F2_pixels | S1_Mc | 24 | zd:order3-1 | -0.29113540224065 | 0.0555514593837757 | -5.24082365198274 | 1.59861434586623e-07 |  | FALSE | 172800 |

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 29.1225328677465 | NA | NA | 2.11051227449856e-06 |  | NA | 172800 |
| O1_dinov2b | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 47.333589734335 | NA | NA | 2.95166474882906e-10 |  | NA | 172800 |
| R1_cvcl | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 71.77198648591 | NA | NA | 1.78138611106211e-15 |  | NA | 172800 |
| E1_siglip | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 107.801122092184 | NA | NA | 3.26219199429972e-23 |  | NA | 172800 |
| O2_openclip | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 70.9452204030749 | NA | NA | 2.67818048561108e-15 |  | NA | 172800 |
| E2_vitsup | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 44.2370469030357 | NA | NA | 1.34395511792656e-09 |  | NA | 172800 |
| E3_rn50sup | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 33.2435099029026 | NA | NA | 2.86138906818865e-07 |  | NA | 172800 |
| E4_mae | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 675.049566148256 | NA | NA | 5.39633654154895e-146 |  | NA | 172800 |
| F1_rn50rand0 | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 399.803322654378 | NA | NA | 2.44206403261213e-86 |  | NA | 172800 |
| F2_pixels | S1_Mc_vs_Mb | 24 | LR_chi2(3) | 83.5526239129686 | NA | NA | 5.30647293901934e-18 |  | NA | 172800 |

### 9.2 S2 (Mb + efectos aleatorios por experimento, 8 ítems)

Archivo: `results/S2.csv` (150 filas; modelo `S2_Mb_expRE`; `nobs` = 57600; `warn` vacío en todas las filas). `singular` es `TRUE` en las 15 filas de `F1_rn50rand0` y `FALSE` en las de los demás encoders. Se copia el término `zd:order3-1`.

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S2_Mb_expRE | 8 | zd:order3-1 | 0.0770051685845487 | 0.0834035911190682 | 0.923283608671178 | 0.355859427602914 |  | FALSE | 57600 |
| O1_dinov2b | S2_Mb_expRE | 8 | zd:order3-1 | 0.0400267757570416 | 0.0792924855326636 | 0.504799105339597 | 0.613699944260695 |  | FALSE | 57600 |
| R1_cvcl | S2_Mb_expRE | 8 | zd:order3-1 | -0.0382740569131406 | 0.0710671429255832 | -0.538561919581018 | 0.590189169300202 |  | FALSE | 57600 |
| E1_siglip | S2_Mb_expRE | 8 | zd:order3-1 | 0.00848141408208657 | 0.130692059815889 | 0.0648961696221996 | 0.948256670365851 |  | FALSE | 57600 |
| O2_openclip | S2_Mb_expRE | 8 | zd:order3-1 | 0.102788039660519 | 0.129171927509798 | 0.795745961542006 | 0.42617970649054 |  | FALSE | 57600 |
| E2_vitsup | S2_Mb_expRE | 8 | zd:order3-1 | 0.0615098024638596 | 0.10431101583079 | 0.589676957644043 | 0.555407246267056 |  | FALSE | 57600 |
| E3_rn50sup | S2_Mb_expRE | 8 | zd:order3-1 | 0.114018635177736 | 0.105468826165854 | 1.08106479727419 | 0.279668291123935 |  | FALSE | 57600 |
| E4_mae | S2_Mb_expRE | 8 | zd:order3-1 | -0.0877300350756225 | 0.110715301504704 | -0.792393046700011 | 0.428131532511114 |  | FALSE | 57600 |
| F1_rn50rand0 | S2_Mb_expRE | 8 | zd:order3-1 | -0.117976578407788 | 0.0642002038760779 | -1.83763557255226 | 0.0661161233580615 |  | TRUE | 57600 |
| F2_pixels | S2_Mb_expRE | 8 | zd:order3-1 | 0.00148327111363813 | 0.0536590485793437 | 0.027642516088314 | 0.977947271674061 |  | FALSE | 57600 |

### 9.3 S3 (Mb + `level:order`, 8 ítems)

Archivo: `results/S3.csv` (170 filas; modelo `S3_Mb_levelxorder`; `nobs` = 57600; `warn` vacío y `singular` = `FALSE` en todas las filas). Se copian `zd:order3-1`, `order3-1:levelsub` y `order3-1:levelsup`.

| encoder | model | items | term | estimate | se | z | p | warn | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.0209021282770787 | 0.0621947958195923 | -0.336075197315693 | 0.736814161407054 |  | FALSE | 57600 |
| R0_dinov2s | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.21985278368781 | 0.113067532026773 | 1.9444378040882 | 0.0518426787894246 |  | FALSE | 57600 |
| R0_dinov2s | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.624493170986697 | 0.0993693789515705 | 6.28456349003705 | 3.28776152976095e-10 |  | FALSE | 57600 |
| O1_dinov2b | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.046682156360046 | 0.0589811717642311 | -0.791475566925854 | 0.428666528471558 |  | FALSE | 57600 |
| O1_dinov2b | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.162475597442844 | 0.115792290601889 | 1.40316420547771 | 0.160567878755049 |  | FALSE | 57600 |
| O1_dinov2b | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.640663254503469 | 0.0981409073035538 | 6.52799400480242 | 6.66563642294396e-11 |  | FALSE | 57600 |
| R1_cvcl | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.0848689402246661 | 0.0640819064854369 | -1.32438226137909 | 0.185376122781388 |  | FALSE | 57600 |
| R1_cvcl | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.127624363103027 | 0.110244318356538 | 1.15765025359657 | 0.247006792498126 |  | FALSE | 57600 |
| R1_cvcl | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.811518327820756 | 0.0907276225400773 | 8.94455630050575 | 3.73446965998964e-19 |  | FALSE | 57600 |
| E1_siglip | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.100201253422961 | 0.0847290299604115 | -1.18260829222025 | 0.236964424618832 |  | FALSE | 57600 |
| E1_siglip | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.230632639117014 | 0.113892345175756 | 2.02500562053671 | 0.0428667870952184 |  | FALSE | 57600 |
| E1_siglip | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.556551309372241 | 0.104551602852845 | 5.32322120547094 | 1.01945648873391e-07 |  | FALSE | 57600 |
| O2_openclip | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.0966492201793685 | 0.0916086630401788 | -1.05502271261157 | 0.291414937231483 |  | FALSE | 57600 |
| O2_openclip | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.137250590322998 | 0.114736960850252 | 1.19621950333972 | 0.231610911845377 |  | FALSE | 57600 |
| O2_openclip | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.605250413043329 | 0.110897644822764 | 5.45773910717973 | 4.82235516917412e-08 |  | FALSE | 57600 |
| E2_vitsup | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.0376467063151102 | 0.0846610738186741 | -0.44467551162582 | 0.656554268770171 |  | FALSE | 57600 |
| E2_vitsup | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.14683904399478 | 0.126238150330343 | 1.16319071224132 | 0.244752133982356 |  | FALSE | 57600 |
| E2_vitsup | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.396230142964116 | 0.110096042123075 | 3.59894992883737 | 0.000319504693252145 |  | FALSE | 57600 |
| E3_rn50sup | S3_Mb_levelxorder | 8 | zd:order3-1 | 0.0377934128045859 | 0.0816764963420804 | 0.462720788686847 | 0.643564515217725 |  | FALSE | 57600 |
| E3_rn50sup | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.231155291607637 | 0.131297389667375 | 1.76054750359653 | 0.0783150205659408 |  | FALSE | 57600 |
| E3_rn50sup | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.599858298266286 | 0.0974013607671646 | 6.15862338617867 | 7.33800008478598e-10 |  | FALSE | 57600 |
| E4_mae | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.138447043280002 | 0.0968531937648113 | -1.42945253427774 | 0.152874208232227 |  | FALSE | 57600 |
| E4_mae | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.140446097574341 | 0.109504784744275 | 1.28255672025951 | 0.199647421245043 |  | FALSE | 57600 |
| E4_mae | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.735296446408699 | 0.0892748494965825 | 8.23632244193082 | 1.77584919090874e-16 |  | FALSE | 57600 |
| F1_rn50rand0 | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.147407174000797 | 0.0653195178224738 | -2.25670946318714 | 0.0240262351483492 |  | FALSE | 57600 |
| F1_rn50rand0 | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.106488904102555 | 0.11050679242592 | 0.963641254667147 | 0.33522581600719 |  | FALSE | 57600 |
| F1_rn50rand0 | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.802221603354119 | 0.0897140433693102 | 8.94198470190172 | 3.82242518410905e-19 |  | FALSE | 57600 |
| F2_pixels | S3_Mb_levelxorder | 8 | zd:order3-1 | -0.0379494128284485 | 0.05096571837362 | -0.744606650106422 | 0.456509546565196 |  | FALSE | 57600 |
| F2_pixels | S3_Mb_levelxorder | 8 | order3-1:levelsub | 0.117473098749957 | 0.109750672891081 | 1.07036335774032 | 0.284455785966723 |  | FALSE | 57600 |
| F2_pixels | S3_Mb_levelxorder | 8 | order3-1:levelsup | 0.775518891688937 | 0.0890214930689439 | 8.71159160505571 | 2.9963750446212e-18 |  | FALSE | 57600 |

### 9.4 S4 (dos etapas, 8 ítems, solo la primera sesión)

Salida de la etapa 2: filas con `variant = first_session` en `results/stage2_sensitivities.csv` (10 filas, una por encoder primario). En esas filas, `status = ok`, `n_exp = 12`, `suffix` vacío, `items = 8`, `ci95_contiguous = True`, `ci95_at_grid_edge = False`, `ci90_contiguous = True`, `ci90_at_grid_edge = False`. El peldaño usado en la etapa 1, según `logs/run_sunday_2026-11-21.log` y las columnas `n_L4_*` de `results/stage2_sensitivities_fit_type_by_order.csv`, es L4 en los 12 experimentos, con `n_failed_* = 0`.

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | first_session |  | 12 | ok | 0.11516494893506446 | -0.4338025364742615 | -0.5489674854093259 | 0.29004329004329005 | 0.8560606060606061 | 0.14502164502164502 | -0.147247 | 0.355753 | True | False | -0.081247 | 0.317753 | True | False | 0.7218614718614719 | 0.5844155844155844 | 0.23160173160173161 | 0.32388630948149055 | 5.223117017411622 | -0.15326072456453338 | 0.3835906224346623 |
| O1_dinov2b | 8 | first_session |  | 12 | ok | 0.0369313834979004 | -0.16603041421901926 | -0.20296179771691966 | 0.8181818181818182 | 0.591991341991342 | 0.4090909090909091 | -0.271297 | 0.333703 | True | False | -0.226297 | 0.304703 | True | False | 0.4686147186147186 | 0.3116883116883117 | 0.12337662337662338 | 0.7883835910602861 | 5.2910600168009685 | -0.29380367927804596 | 0.3676664462738468 |
| R1_cvcl | 8 | first_session |  | 12 | ok | 0.10444212925033958 | -0.20191889240306693 | -0.3063610216534065 | 0.12337662337662338 | 0.9393939393939394 | 0.06168831168831169 | -0.02671 | 0.24129 | True | False | -0.00771 | 0.21529 | True | False | 0.8181818181818182 | 0.5324675324675324 | 0.0670995670995671 | 0.12205875754337585 | 8.288067799308383 | -0.0345789973397127 | 0.24346325584039186 |
| E1_siglip | 8 | first_session |  | 12 | ok | 0.00762332692472989 | -1.3817838075104685 | -1.3894071344351984 | 0.9588744588744589 | 0.5216450216450217 | 0.47943722943722944 | -0.334071 | 0.362929 | True | False | -0.273071 | 0.281929 | True | False | 0.39285714285714285 | 0.27380952380952384 | 0.11688311688311688 | 0.9613504260482281 | 9.61896165517416 | -0.33574018412790646 | 0.35098683797736624 |
| O2_openclip | 8 | first_session |  | 12 | ok | 0.013412298259491529 | -1.3485775082541485 | -1.36198980651364 | 0.9242424242424242 | 0.538961038961039 | 0.4621212121212121 | -0.246136 | 0.305864 | True | False | -0.207136 | 0.241864 | True | False | 0.395021645021645 | 0.25757575757575757 | 0.08116883116883117 | 0.9180046145046582 | 6.6513828180162875 | -0.28638958647110707 | 0.31321418299009013 |
| E2_vitsup | 8 | first_session |  | 12 | ok | 0.11789652553187518 | -0.4095743707101492 | -0.5274708962420244 | 0.44155844155844154 | 0.7803030303030303 | 0.22077922077922077 | -0.200276 | 0.442724 | True | False | -0.143276 | 0.385724 | True | False | 0.6677489177489178 | 0.5476190476190477 | 0.3051948051948052 | 0.4389100265154747 | 6.15130706502283 | -0.22868630503965132 | 0.4644793561034017 |
| E3_rn50sup | 8 | first_session |  | 12 | ok | 0.03604655504178167 | -0.9466053048821652 | -0.9826518599239469 | 0.5952380952380952 | 0.7034632034632035 | 0.2976190476190476 | -0.108085 | 0.187915 | True | False | -0.083085 | 0.162915 | True | False | 0.4231601731601732 | 0.18181818181818182 | 0.0183982683982684 | 0.6038736815541558 | 7.9374355124555915 | -0.11803708744511648 | 0.19013019752867982 |
| E4_mae | 8 | first_session |  | 12 | ok | 0.7518491206834 | -0.752333904620205 | -1.504183025303605 | 0.11471861471861472 | 0.9437229437229437 | 0.05735930735930736 | -0.179606 | 1.787394 | True | False | -0.027606 | 1.572394 | True | False | 0.9329004329004329 | 0.9134199134199135 | 0.8668831168831169 | 0.13878267812538053 | 7.0950531223943205 | -0.3114530032236654 | 1.8151512445904654 |
| F1_rn50rand0 | 8 | first_session |  | 12 | ok | -0.01100785608289373 | -0.43459711373639154 | -0.4235892576534978 | 0.9264069264069265 | 0.46320346320346323 | 0.5378787878787878 | -0.238903 | 0.227097 | True | False | -0.203903 | 0.180097 | True | False | 0.3582251082251082 | 0.2132034632034632 | 0.05519480519480519 | 0.9191228010205756 | 9.624704736653893 | -0.247562474516618 | 0.22554676235083054 |
| F2_pixels | 8 | first_session |  | 12 | ok | 0.045499246536359506 | -0.13343500316537835 | -0.17893424970173785 | 0.3116883116883117 | 0.8452380952380952 | 0.15584415584415584 | -0.052354 | 0.142646 | True | False | -0.033354 | 0.122646 | True | False | 0.45670995670995673 | 0.10930735930735931 | 0.004329004329004329 | 0.3121784656337249 | 9.967612885153361 | -0.04977218917831516 | 0.14077068225103417 |

Tipo de ajuste × orden (`results/stage2_sensitivities_fit_type_by_order.csv`, `variant = first_session`):

| encoder | items | variant | suffix | n_L1_1-3 | n_L1_3-1 | n_L2_1-3 | n_L2_3-1 | n_L3_1-3 | n_L3_3-1 | n_L4_1-3 | n_L4_3-1 | n_failed_1-3 | n_failed_3-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| O1_dinov2b | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| R1_cvcl | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E1_siglip | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| O2_openclip | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E2_vitsup | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E3_rn50sup | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E4_mae | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| F1_rn50rand0 | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| F2_pixels | 8 | first_session |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |

### 9.5 S5 (dos etapas, 8 ítems, d en rangos)

Filas con `variant = rank` en `results/stage2_sensitivities.csv` (10 filas). Mismos valores de `status`, `n_exp`, `suffix`, `items` y banderas de IC que en S4. Tipo de ajuste: L4 en los 12 experimentos, `n_failed_* = 0`.

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | rank |  | 12 | ok | 0.031454294434006824 | -0.6069087624843368 | -0.6383630569183436 | 0.6948051948051948 | 0.6536796536796536 | 0.3474025974025974 | -0.160517 | 0.227483 | True | False | -0.143517 | 0.182483 | True | False | 0.4069264069264069 | 0.20995670995670995 | 0.04329004329004329 | 0.7163103054406186 | 5.949574986414437 | -0.17087632398522032 | 0.23378491285323397 |
| O1_dinov2b | 8 | rank |  | 12 | ok | -0.00440638968891649 | -0.5423541320300472 | -0.5379477423411307 | 0.948051948051948 | 0.474025974025974 | 0.5270562770562771 | -0.174399 | 0.162601 | True | False | -0.155399 | 0.128601 | True | False | 0.25432900432900435 | 0.09523809523809523 | 0.00974025974025974 | 0.9518311317169101 | 6.409497436730212 | -0.17344385630029385 | 0.16463107692246087 |
| R1_cvcl | 8 | rank |  | 12 | ok | 0.038345237459223885 | -0.1328067239129284 | -0.17115196137215227 | 0.354978354978355 | 0.8235930735930735 | 0.1774891774891775 | -0.05054 | 0.13046 | True | False | -0.03454 | 0.11046 | True | False | 0.3906926406926407 | 0.08008658008658008 | 0.003246753246753247 | 0.36731502503142766 | 9.037212536866017 | -0.05300374610424721 | 0.12969422102269498 |
| E1_siglip | 8 | rank |  | 12 | ok | -0.001332237376123535 | -1.32218090510408 | -1.3208486677279565 | 0.9956709956709957 | 0.49783549783549785 | 0.5032467532467533 | -0.288228 | 0.297772 | True | False | -0.250228 | 0.247772 | True | False | 0.3625541125541126 | 0.24783549783549783 | 0.08658008658008658 | 0.9922957872573186 | 9.48732476633086 | -0.302976532583113 | 0.30031205783086595 |
| O2_openclip | 8 | rank |  | 12 | ok | -0.03540586603485307 | -1.4807706082949814 | -1.4453647422601283 | 0.7857142857142857 | 0.39285714285714285 | 0.6082251082251082 | -0.271587 | 0.233413 | True | False | -0.238587 | 0.178413 | True | False | 0.4577922077922078 | 0.3268398268398268 | 0.09848484848484848 | 0.7732005904766805 | 7.150684555778514 | -0.3139107440135758 | 0.24309901194386968 |
| E2_vitsup | 8 | rank |  | 12 | ok | 0.08743941233553909 | -0.91306704762395 | -1.000506459959489 | 0.5930735930735931 | 0.7045454545454546 | 0.29653679653679654 | -0.205576 | 0.412424 | True | False | -0.159576 | 0.350424 | True | False | 0.591991341991342 | 0.4675324675324675 | 0.23376623376623376 | 0.5578954399442457 | 7.37218709760762 | -0.24607601806859014 | 0.4209548427396683 |
| E3_rn50sup | 8 | rank |  | 12 | ok | -0.013248097183233165 | -1.04852447239119 | -1.0352763752079568 | 0.8744588744588745 | 0.43722943722943725 | 0.5638528138528138 | -0.166776 | 0.160224 | True | False | -0.140776 | 0.124224 | True | False | 0.31493506493506496 | 0.13744588744588745 | 0.012987012987012988 | 0.860607709422759 | 9.581471909728734 | -0.17786372860846944 | 0.1513675342420031 |
| E4_mae | 8 | rank |  | 12 | ok | 0.060242334302165246 | -0.4092791104499107 | -0.46952144475207597 | 0.4805194805194805 | 0.7608225108225108 | 0.24025974025974026 | -0.123743 | 0.231257 | True | False | -0.085743 | 0.208257 | True | False | 0.5476190476190477 | 0.31926406926406925 | 0.06385281385281386 | 0.4744799518024086 | 9.094490907114174 | -0.12213476543660035 | 0.24261943404093084 |
| F1_rn50rand0 | 8 | rank |  | 12 | ok | -0.18276136370916146 | -0.35632237010407414 | -0.17356100639491268 | 0.032467532467532464 | 0.016233766233766232 | 0.9848484848484849 | -0.354877 | -0.012877 | True | False | -0.318877 | -0.043877 | True | False | 0.9437229437229437 | 0.8441558441558441 | 0.4090909090909091 | 0.042389376452230364 | 8.337550341685112 | -0.35758466012281875 | -0.007938067295504198 |
| F2_pixels | 8 | rank |  | 12 | ok | -0.05241176844247539 | -0.1107976750883542 | -0.05838590664587881 | 0.05844155844155844 | 0.02922077922077922 | 0.9718614718614719 | -0.104842 | 0.003158 | True | False | -0.096842 | -0.008842 | True | False | 0.5378787878787878 | 0.0367965367965368 | 0.0010822510822510823 | 0.05360440272449602 | 9.87649346749655 | -0.10581103925923634 | 0.000987502374285562 |

Tipo de ajuste × orden (`variant = rank`):

| encoder | items | variant | suffix | n_L1_1-3 | n_L1_3-1 | n_L2_1-3 | n_L2_3-1 | n_L3_1-3 | n_L3_3-1 | n_L4_1-3 | n_L4_3-1 | n_failed_1-3 | n_failed_3-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| O1_dinov2b | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| R1_cvcl | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E1_siglip | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| O2_openclip | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E2_vitsup | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E3_rn50sup | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E4_mae | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| F1_rn50rand0 | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| F2_pixels | 8 | rank |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |

### 9.6 S7 (F1 con semillas 1 y 2) y S8 (MAE con CLS, E4s)

En `results/stage2_sensitivities.csv` estas dos sensibilidades aparecen con `variant = primary` (el mismo rótulo de variante que el análisis primario, pero aquí el archivo es el de sensibilidades) y con encoders distintos de los diez primarios: `F1_rn50rand1`, `F1_rn50rand2` (S7) y `E4s_mae_cls` (S8). Son las 3 filas restantes de las 23 del archivo. `status = ok`, `n_exp = 12`, `items = 8`, `suffix` vacío; banderas de IC iguales a las de S4 y S5; tipo de ajuste L4 en los 12 experimentos, `n_failed_* = 0`.

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1_rn50rand1 | 8 | primary |  | 12 | ok | -0.09344026283504131 | -0.4778071118565108 | -0.3843668490214695 | 0.3463203463203463 | 0.17316017316017315 | 0.827922077922078 | -0.298316 | 0.096684 | True | False | -0.254316 | 0.063684 | True | False | 0.670995670995671 | 0.47294372294372294 | 0.1266233766233766 | 0.3256256351006151 | 7.317563996676598 | -0.30132822147130667 | 0.11444769580122408 |
| F1_rn50rand2 | 8 | primary |  | 12 | ok | 0.23453923209360739 | -0.5811747302429093 | -0.8157139623365167 | 0.032467532467532464 | 0.9848484848484849 | 0.016233766233766232 | 0.022162 | 0.490162 | True | False | 0.050162 | 0.440162 | True | False | 0.9502164502164502 | 0.8679653679653679 | 0.5833333333333334 | 0.06990764906475344 | 7.549341680556423 | -0.02445599514147556 | 0.49353445932869033 |
| E4s_mae_cls | 8 | primary |  | 12 | ok | -0.02827657141435158 | -0.8591952834236426 | -0.830918712009291 | 0.6493506493506493 | 0.3246753246753247 | 0.6764069264069265 | -0.161257 | 0.109743 | True | False | -0.140257 | 0.081743 | True | False | 0.36796536796536794 | 0.1396103896103896 | 0.010822510822510822 | 0.6550023308039428 | 9.632712644785373 | -0.16564219142316045 | 0.10908904859445728 |

Tipo de ajuste × orden (filas `variant = primary` de `results/stage2_sensitivities_fit_type_by_order.csv`):

| encoder | items | variant | suffix | n_L1_1-3 | n_L1_3-1 | n_L2_1-3 | n_L2_3-1 | n_L3_1-3 | n_L3_3-1 | n_L4_1-3 | n_L4_3-1 | n_failed_1-3 | n_failed_3-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1_rn50rand1 | 8 | primary |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| F1_rn50rand2 | 8 | primary |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E4s_mae_cls | 8 | primary |  | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |

## 10. S9 (ajuste mixto, descriptivo) y marcas ya calculadas por `08_regions.py`

El plan §5.4 (decisión de Patricio del 18-11 a las 12:12) define S9 como el ajuste mixto: en cada experimento, el primer peldaño de la escalera que converge (pasada 1). Las dos etapas, con 8 ítems. No requiere ajustes extra. Se reporta la tabla tipo de ajuste (L1–L4 / falló) × orden por encoder. Si S9 cambia la región de un encoder con algún δ, ese encoder se califica «sensible al tipo de ajuste» (descriptivo: no cambia regiones ni conclusiones G). El log del domingo (22-11-2026, 09:31:38, hora de Santiago) registra que la variante `S9_mixed` ya estaba en `results/stage2_primary.csv` y que no hubo ajustes extra.

`08_regions.py`, al correrse con `--stage2-s9 results/stage2_primary.csv`, filtró las filas con `variant = S9_mixed` y `suffix` vacío, y usó solo las de 8 ítems para las etiquetas `region_S9_*` (función `s9_labels` en el script). El archivo `results/stage2_primary.csv` contiene también filas `S9_mixed` con 24 ítems; se copian más abajo porque están en el archivo, con la salvedad de que el plan describe S9 con 8 ítems y de que `08_regions.py` no usó las de 24 ítems para esas etiquetas.

### 10.1 Informe de regiones con S9, transcrito de `08_regions.py`

Transcripción literal, sin cambios, de `results/regions_with_S9/regions_report.md` (marca de tiempo del archivo: 22-11-2026, 09:31, hora de Santiago):

# Regiones y conclusión global (plan §7; generado por 08_regions.py)

- Control de tubería: PASA. Perfil del agrupamiento de geometría: depende del corte.
- Referencia descriptiva, no confirmatoria (I-b): Δ en dos etapas de R0 con 8 ítems = 0.019709506140741517 [IC 95 % -0.204779; 0.274221].
- Lectura de R0 con 24 ítems (REVISION3b): inconcluso.

## Tabla única (§7.2): región con δ = 0,05 / 0,10 / 0,20 y con 24 ítems

| encoder | control_positivo | familia_holm | region_d0.05 | region_d0.10 | region_d0.20 | region_24 | tipo_de_ajuste | perfil |
|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | pasa | R0: referencia (no entra en Holm; G1) | inconcluso | inconcluso | inconcluso | inconcluso | sensible al tipo de ajuste | robusto: inconcluso; sensible al tipo de ajuste (S9, descriptivo) |
| O1_dinov2b | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | sensible al tipo de ajuste | robusto: inconcluso; sensible al tipo de ajuste (S9, descriptivo) |
| R1_cvcl | pasa | familia entrenados | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | sensible al tipo de ajuste | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20); sensible al tipo de ajuste (S9, descriptivo) |
| E1_siglip | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| O2_openclip | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| E2_vitsup | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | sensible al tipo de ajuste | robusto: inconcluso; sensible al tipo de ajuste (S9, descriptivo) |
| E3_rn50sup | pasa | familia entrenados | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | sensible al tipo de ajuste | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20); sensible al tipo de ajuste (S9, descriptivo) |
| E4_mae | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| F1_rn50rand0 | no es puerta (piso) | familia pisos | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| F2_pixels | no es puerta (piso) | familia pisos | inconcluso | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | inconcluso | estable | dependiente del margen: inconcluso (δ = 0.05); menor que δ (equivalente con margen δ) (δ = 0.10, 0.20) |

## Familias Holm (§7.3; iguales con los tres δ)

- **entrenados:** k = 7 encoders, 14 pruebas por δ; incluidos: O1_dinov2b, R1_cvcl, E1_siglip, O2_openclip, E2_vitsup, E3_rn50sup, E4_mae; excluidos: ninguno.
- **pisos:** k = 2 encoders, 4 pruebas por δ; incluidos: F1_rn50rand0, F2_pixels; excluidos: ninguno.

## Conclusión global por (δ, corte)

| delta | cut | conclusion | detalle | trained_family_size |
|---|---|---|---|---|
| 0.05 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.05 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.05 | 0.7 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.7 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.7 | G1 | R0 en 'inconcluso' | 14 |

**Perfil global:** robusta: G1

## Pisos

| delta | family_size | lectura |
|---|---|---|
| 0.05 | 4 | inconcluso |
| 0.1 | 4 | inconcluso |
| 0.2 | 4 | inconcluso |

### 10.2 Comparación de regiones primario frente a S9, copiada de `results/regions_with_S9/regions_table.csv`

La columna `tipo_de_ajuste` es la marca mecánica del script: `sensible al tipo de ajuste` si, para algún δ ∈ {0,05; 0,10; 0,20}, `region_S9_d…` no coincide con `region_d…`; en caso contrario, `estable`.

| encoder | region_d0.05 | region_S9_d0.05 | region_d0.10 | region_S9_d0.10 | region_d0.20 | region_S9_d0.20 | tipo_de_ajuste |
|---|---|---|---|---|---|---|---|
| R0_dinov2s | inconcluso | signo opuesto | inconcluso | signo opuesto | inconcluso | signo opuesto | sensible al tipo de ajuste |
| O1_dinov2b | inconcluso | signo opuesto | inconcluso | signo opuesto | inconcluso | signo opuesto | sensible al tipo de ajuste |
| R1_cvcl | inconcluso | inconcluso | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | sensible al tipo de ajuste |
| E1_siglip | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | estable |
| O2_openclip | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | estable |
| E2_vitsup | inconcluso | signo opuesto | inconcluso | signo opuesto | inconcluso | signo opuesto | sensible al tipo de ajuste |
| E3_rn50sup | inconcluso | inconcluso | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | sensible al tipo de ajuste |
| E4_mae | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | estable |
| F1_rn50rand0 | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | inconcluso | estable |
| F2_pixels | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | estable |

### 10.3 Etapa 2, variante `S9_mixed`

**8 ítems** (las filas que usó `08_regions.py` para las etiquetas del §10.2), copiadas de `results/stage2_primary.csv`:

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | S9_mixed |  | 12 | ok | -0.37015448427122033 | -1.7778896487998652 | -1.4077351645286449 | 0.049783549783549784 | 0.024891774891774892 | 0.9761904761904762 | -0.727551 | -0.006551 | True | False | -0.660551 | -0.081551 | True | False | 0.9621212121212122 | 0.9393939393939394 | 0.8506493506493507 | 0.04775780556908323 | 7.359168411448744 | -0.7355518272119643 | -0.004757141330476278 |
| O1_dinov2b | 8 | S9_mixed |  | 12 | ok | -0.5971631338869609 | -1.330825640068455 | -0.7336625061814942 | 0.021645021645021644 | 0.010822510822510822 | 0.9902597402597403 | -1.028211 | -0.164211 | True | False | -0.948211 | -0.204211 | True | False | 0.9891774891774892 | 0.9891774891774892 | 0.9534632034632035 | 0.01657227153484257 | 8.088960206306998 | -1.0536116575072134 | -0.14071461026670834 |
| R1_cvcl | 8 | S9_mixed |  | 12 | ok | -0.047263771484265416 | -0.42285782020820967 | -0.37559404872394425 | 0.658008658008658 | 0.329004329004329 | 0.672077922077922 | -0.26849 | 0.16551 | True | False | -0.22249 | 0.12751 | True | False | 0.4967532467532468 | 0.2943722943722944 | 0.06926406926406926 | 0.6376831834593485 | 6.0325738505337565 | -0.28029300438061333 | 0.18576546141208247 |
| E1_siglip | 8 | S9_mixed |  | 12 | ok | -0.2504724433360428 | -3.240688683355829 | -2.990216240019786 | 0.4393939393939394 | 0.2196969696969697 | 0.7813852813852814 | -0.906327 | 0.393673 | True | False | -0.785327 | 0.271673 | True | False | 0.7316017316017316 | 0.6872294372294372 | 0.564935064935065 | 0.41727798752624734 | 8.951794464939006 | -0.9173194327695273 | 0.41637454609744173 |
| O2_openclip | 8 | S9_mixed |  | 12 | ok | -0.13434210835302807 | -2.535923611884695 | -2.4015815035316668 | 0.551948051948052 | 0.275974025974026 | 0.7251082251082251 | -0.577975 | 0.303025 | True | False | -0.509975 | 0.229025 | True | False | 0.6461038961038961 | 0.5703463203463204 | 0.3722943722943723 | 0.5357472117544192 | 7.063185255818458 | -0.6215564413664945 | 0.3528722246604384 |
| E2_vitsup | 8 | S9_mixed |  | 12 | ok | -0.6844568289567996 | -2.04867840639652 | -1.3642215774397206 | 0.0367965367965368 | 0.0183982683982684 | 0.9826839826839827 | -1.382562 | -0.052562 | True | False | -1.233562 | -0.154562 | True | False | 0.9761904761904762 | 0.9653679653679653 | 0.9372294372294372 | 0.05011964121948533 | 8.92784099164095 | -1.3693571457175047 | 0.0004434878039054624 |
| E3_rn50sup | 8 | S9_mixed |  | 12 | ok | -0.16223138148028204 | -2.0595194207990986 | -1.8972880393188165 | 0.4675324675324675 | 0.23376623376623376 | 0.7673160173160173 | -0.65293 | 0.35107 | True | False | -0.53693 | 0.23407 | True | False | 0.6937229437229437 | 0.6147186147186147 | 0.4296536796536797 | 0.47472831390490694 | 9.224980392109883 | -0.6529052969146814 | 0.3284425339541172 |
| E4_mae | 8 | S9_mixed |  | 12 | ok | 0.1667515585259629 | -1.2849254329197484 | -1.4516769914457113 | 0.645021645021645 | 0.6785714285714286 | 0.3225108225108225 | -0.667788 | 0.977212 | True | False | -0.485788 | 0.818212 | True | False | 0.6244588744588745 | 0.5681818181818182 | 0.4621212121212121 | 0.6556550069874987 | 9.7092792853656 | -0.6442729109350557 | 0.9777760279869815 |
| F1_rn50rand0 | 8 | S9_mixed |  | 12 | ok | -0.04422590227682199 | -0.8951720258626413 | -0.8509461235858193 | 0.7878787878787878 | 0.3939393939393939 | 0.6071428571428571 | -0.397842 | 0.288158 | True | False | -0.324842 | 0.229158 | True | False | 0.48268398268398266 | 0.3593073593073593 | 0.1645021645021645 | 0.7796427431699497 | 8.937486200107719 | -0.3915375169106391 | 0.3030857123569951 |
| F2_pixels | 8 | S9_mixed |  | 12 | ok | 0.0023457336557499153 | -0.12419464082001436 | -0.12654037447576427 | 0.9632034632034632 | 0.5194805194805194 | 0.4816017316017316 | -0.094174 | 0.099826 | True | False | -0.073174 | 0.079826 | True | False | 0.1471861471861472 | 0.023809523809523808 | 0.0021645021645021645 | 0.9578112889536401 | 9.973149923685677 | -0.09404170279979022 | 0.09873317011129006 |

**24 ítems** (presentes en el mismo archivo; no usadas por `08_regions.py` para `region_S9_*`):

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 24 | S9_mixed |  | 12 | ok | 0.4110574248406449 | -3.3887226633892236 | -3.7997800882298685 | 0.21645021645021645 | 0.8928571428571429 | 0.10822510822510822 | -0.322794 | 1.126206 | True | False | -0.175794 | 1.000206 | True | False | 0.8647186147186147 | 0.8344155844155844 | 0.7402597402597403 | 0.23612978541482244 | 6.348560330290509 | -0.3476043010851103 | 1.1697191507664 |
| O1_dinov2b | 24 | S9_mixed |  | 12 | ok | 0.3746846297786952 | -3.5284526735438884 | -3.9031373033225836 | 0.2943722943722944 | 0.8538961038961039 | 0.1471861471861472 | -0.404186 | 1.136814 | True | False | -0.259186 | 1.006814 | True | False | 0.8203463203463204 | 0.7824675324675324 | 0.6904761904761905 | 0.3113786164858257 | 6.646952991106618 | -0.44282584536403047 | 1.1921951049214208 |
| R1_cvcl | 24 | S9_mixed |  | 12 | ok | 0.3076394330067451 | -3.5280620532714138 | -3.835701486278159 | 0.16017316017316016 | 0.920995670995671 | 0.08008658008658008 | -0.137223 | 0.778777 | True | False | -0.076223 | 0.691777 | True | False | 0.8874458874458875 | 0.8344155844155844 | 0.7034632034632035 | 0.1736543101479643 | 6.500703081371158 | -0.17602555387103508 | 0.7913044198845254 |
| E1_siglip | 24 | S9_mixed |  | 12 | ok | 1.0284754900542188 | -5.8251370799052475 | -6.853612569959466 | 0.25757575757575757 | 0.8722943722943723 | 0.12878787878787878 | -1.058667 | 3.075333 | True | False | -0.617667 | 2.715333 | True | False | 0.8560606060606061 | 0.841991341991342 | 0.8062770562770563 | 0.29335998702480215 | 6.253148728687513 | -1.1442023701267092 | 3.201153350235147 |
| O2_openclip | 24 | S9_mixed |  | 12 | ok | 0.9233837866514971 | -4.955694404665041 | -5.879078191316538 | 0.23160173160173161 | 0.8852813852813853 | 0.11580086580086581 | -0.81009 | 2.62091 | True | False | -0.45009 | 2.32591 | True | False | 0.8722943722943723 | 0.8560606060606061 | 0.8235930735930735 | 0.2562761138141692 | 6.1457427081002365 | -0.871654471746796 | 2.71842204504979 |
| E2_vitsup | 24 | S9_mixed |  | 12 | ok | 1.0041299507014205 | -4.655254777001935 | -5.6593847277033555 | 0.18181818181818182 | 0.9101731601731602 | 0.09090909090909091 | -0.590246 | 2.658754 | True | False | -0.326246 | 2.331754 | True | False | 0.9015151515151515 | 0.895021645021645 | 0.8712121212121212 | 0.19498306492414944 | 6.217214289610578 | -0.6736094898010772 | 2.681869391203918 |
| E3_rn50sup | 24 | S9_mixed |  | 12 | ok | 0.8723938494441636 | -4.330280885813841 | -5.202674735258005 | 0.1645021645021645 | 0.9188311688311688 | 0.08225108225108226 | -0.469141 | 2.223859 | True | False | -0.233141 | 1.978859 | True | False | 0.908008658008658 | 0.8961038961038961 | 0.8647186147186147 | 0.18226539202109826 | 6.437448550386117 | -0.5324036887303603 | 2.2771913876186876 |
| E4_mae | 24 | S9_mixed |  | 12 | ok | 0.9150911652841849 | -8.088134294581513 | -9.003225459865698 | 0.22077922077922077 | 0.8906926406926406 | 0.11038961038961038 | -0.645371 | 2.561629 | True | False | -0.359371 | 2.191629 | True | False | 0.8733766233766234 | 0.8593073593073594 | 0.8311688311688312 | 0.23476510805942094 | 6.243276420196536 | -0.7718113807698692 | 2.601993711338239 |
| F1_rn50rand0 | 24 | S9_mixed |  | 12 | ok | -0.048155454323846936 | -3.058386476892432 | -3.010231022568585 | 0.6774891774891775 | 0.33874458874458874 | 0.6623376623376623 | -0.299486 | 0.216514 | True | False | -0.247486 | 0.158514 | True | False | 0.4945887445887446 | 0.3268398268398268 | 0.11255411255411256 | 0.686635433868167 | 8.692567268424982 | -0.3106702221926697 | 0.21435931354497584 |
| F2_pixels | 24 | S9_mixed |  | 12 | ok | 0.14704825466157345 | -2.3887862333764684 | -2.535834488038042 | 0.05411255411255411 | 0.974025974025974 | 0.027056277056277056 | -0.000272 | 0.292728 | True | False | 0.027728 | 0.264728 | True | False | 0.9166666666666666 | 0.7521645021645021 | 0.21861471861471862 | 0.06083335947878094 | 6.985205068579445 | -0.008775369796101218 | 0.3028718791192481 |

### 10.4 Tipo de ajuste × orden, variante `S9_mixed`

Fuente: `results/stage2_primary_fit_type_by_order.csv`. En `results/structure_S9_mixed.csv` (240 filas) la columna `re_used` toma solo los valores L1, L2, L3 y L4; no aparece `failed`. Los recuentos por encoder, conjunto de ítems y orden están en la tabla siguiente, copiados de `results/stage2_primary_fit_type_by_order.csv`.

**8 ítems:**

| encoder | items | variant | suffix | n_L1_1-3 | n_L1_3-1 | n_L2_1-3 | n_L2_3-1 | n_L3_1-3 | n_L3_3-1 | n_L4_1-3 | n_L4_3-1 | n_failed_1-3 | n_failed_3-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | S9_mixed |  | 5 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| O1_dinov2b | 8 | S9_mixed |  | 5 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R1_cvcl | 8 | S9_mixed |  | 3 | 4 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| E1_siglip | 8 | S9_mixed |  | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| O2_openclip | 8 | S9_mixed |  | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2_vitsup | 8 | S9_mixed |  | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3_rn50sup | 8 | S9_mixed |  | 4 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E4_mae | 8 | S9_mixed |  | 4 | 5 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 |
| F1_rn50rand0 | 8 | S9_mixed |  | 4 | 3 | 2 | 1 | 0 | 2 | 0 | 0 | 0 | 0 |
| F2_pixels | 8 | S9_mixed |  | 2 | 3 | 3 | 0 | 1 | 2 | 0 | 1 | 0 | 0 |

**24 ítems:**

| encoder | items | variant | suffix | n_L1_1-3 | n_L1_3-1 | n_L2_1-3 | n_L2_3-1 | n_L3_1-3 | n_L3_3-1 | n_L4_1-3 | n_L4_3-1 | n_failed_1-3 | n_failed_3-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 24 | S9_mixed |  | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| O1_dinov2b | 24 | S9_mixed |  | 4 | 6 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| R1_cvcl | 24 | S9_mixed |  | 5 | 6 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| E1_siglip | 24 | S9_mixed |  | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| O2_openclip | 24 | S9_mixed |  | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2_vitsup | 24 | S9_mixed |  | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3_rn50sup | 24 | S9_mixed |  | 5 | 6 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| E4_mae | 24 | S9_mixed |  | 2 | 3 | 1 | 2 | 3 | 1 | 0 | 0 | 0 | 0 |
| F1_rn50rand0 | 24 | S9_mixed |  | 1 | 2 | 3 | 2 | 2 | 2 | 0 | 0 | 0 | 0 |
| F2_pixels | 24 | S9_mixed |  | 2 | 1 | 1 | 1 | 3 | 4 | 0 | 0 | 0 | 0 |

## 11. Familia A frente a F0

El plan §5.4 pide los coeficientes `condition:order3-1` de Mb por encoder, frente a F0 (`selected ~ condition*order + level + (1 | person) + (1 | image)`). Es descriptiva, sin margen fijo. El log del domingo indica que los coeficientes de Mb están en `results/S1_*.csv` y que F0 se escribió en `results/F0_24.csv` y `results/F0_8.csv`.

En `F0_8.csv` y `F0_24.csv` el encoder se llama `F0_nodistance`, el modelo es `F0`, `warn` está vacío y `singular` es `FALSE` en todas las filas (10 términos cada archivo). `nobs` es 57600 (8 ítems) y 172800 (24 ítems). No hay columna de región: `08_regions.py` no se aplicó a F0.

Se copian los tres términos `conditionsub:order3-1`, `conditionbas:order3-1` y `conditionsup:order3-1`. Primero F0 y, a continuación, el mismo término en `S1_Mb` de cada encoder.

### 11.1 Ocho ítems (`results/F0_8.csv` y `results/S1_8.csv`)

| encoder | origen | model | items | term | estimate | se | z | p | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| F0_nodistance | F0 | F0 | 8 | conditionsub:order3-1 | 1.9094512450081 | 0.0919692877296368 | 20.7618357404413 | 9.58349975784192e-96 | FALSE | 57600 |
| F0_nodistance | F0 | F0 | 8 | conditionbas:order3-1 | 1.58762171254634 | 0.0860483680467414 | 18.4503407628132 | 5.1817322947294e-76 | FALSE | 57600 |
| F0_nodistance | F0 | F0 | 8 | conditionsup:order3-1 | 1.04254443416492 | 0.086221763542358 | 12.0914301834333 | 1.17228073418233e-33 | FALSE | 57600 |
| R0_dinov2s | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 2.16992030475843 | 0.104845564253842 | 20.6963481974767 | 3.73631461369632e-95 | FALSE | 57600 |
| R0_dinov2s | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.67363197373143 | 0.101907324138577 | 16.4230783987181 | 1.30760966445124e-60 | FALSE | 57600 |
| R0_dinov2s | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.15782476597715 | 0.0997134333131409 | 11.6115224148497 | 3.60145551920752e-31 | FALSE | 57600 |
| O1_dinov2b | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 2.14500393328085 | 0.10339612846966 | 20.745495648904 | 1.34629755063517e-95 | FALSE | 57600 |
| O1_dinov2b | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.70625572729769 | 0.101997423779352 | 16.7284198372381 | 8.13685080394898e-63 | FALSE | 57600 |
| O1_dinov2b | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.23231305559539 | 0.0997490320321802 | 12.3541354787065 | 4.62703517176238e-35 | FALSE | 57600 |
| R1_cvcl | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 1.87861083419192 | 0.0982417760236814 | 19.1223215848528 | 1.64614658441216e-81 | FALSE | 57600 |
| R1_cvcl | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.54637406879031 | 0.0909275545902415 | 17.0066606955277 | 7.32983131787898e-65 | FALSE | 57600 |
| R1_cvcl | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.01325279620711 | 0.0901956757658248 | 11.2339398491544 | 2.77808502876247e-29 | FALSE | 57600 |
| E1_siglip | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 2.10552434278086 | 0.10407582898117 | 20.2306756851469 | 5.25775512960487e-91 | FALSE | 57600 |
| E1_siglip | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.62730557139946 | 0.104556471731418 | 15.5638914019558 | 1.28069876703141e-54 | FALSE | 57600 |
| E1_siglip | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.20913124004277 | 0.10104855118881 | 11.9658443967544 | 5.36490225572694e-33 | FALSE | 57600 |
| O2_openclip | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 2.16229328094943 | 0.105331010395486 | 20.5285534889552 | 1.19684908977197e-93 | FALSE | 57600 |
| O2_openclip | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.63260573194134 | 0.10863834721079 | 15.0278955254503 | 4.82079220086466e-51 | FALSE | 57600 |
| O2_openclip | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.19520537223456 | 0.10161314928273 | 11.7623101013138 | 6.10409609143766e-32 | FALSE | 57600 |
| E2_vitsup | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 2.32115489355593 | 0.10969788543401 | 21.1595226687596 | 2.25493657152055e-99 | FALSE | 57600 |
| E2_vitsup | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.80126126803006 | 0.10820814650021 | 16.646263024444 | 3.22097944451944e-62 | FALSE | 57600 |
| E2_vitsup | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.35486881415332 | 0.103281665720797 | 13.118192901884 | 2.59038318118022e-39 | FALSE | 57600 |
| E3_rn50sup | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 2.26029575001494 | 0.109970407625382 | 20.5536725635748 | 7.13553012637036e-94 | FALSE | 57600 |
| E3_rn50sup | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.79247769869231 | 0.100393824356622 | 17.854461767738 | 2.66840680711802e-71 | FALSE | 57600 |
| E3_rn50sup | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.30580106088987 | 0.0960038398631643 | 13.6015503416431 | 3.92034142936001e-42 | FALSE | 57600 |
| E4_mae | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 1.83975315172837 | 0.0940957039188318 | 19.5519356900222 | 3.97161828127904e-85 | FALSE | 57600 |
| E4_mae | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.55418863746867 | 0.0867793561075799 | 17.9096585545294 | 9.91418658017712e-72 | FALSE | 57600 |
| E4_mae | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 0.999266994778295 | 0.0890148147133256 | 11.2258504159837 | 3.04443304034576e-29 | FALSE | 57600 |
| F1_rn50rand0 | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 1.86037680956629 | 0.0996364485196294 | 18.6716491525667 | 8.42118449177662e-78 | FALSE | 57600 |
| F1_rn50rand0 | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.52908841463167 | 0.0930017875748205 | 16.4414948841872 | 9.65087237798287e-61 | FALSE | 57600 |
| F1_rn50rand0 | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 1.01670739281729 | 0.0898467506626475 | 11.3160173887064 | 1.09326645987411e-29 | FALSE | 57600 |
| F2_pixels | S1 (Mb) | S1_Mb | 8 | conditionsub:order3-1 | 1.84299529716664 | 0.0960275817487449 | 19.1923535259778 | 4.28774409271021e-82 | FALSE | 57600 |
| F2_pixels | S1 (Mb) | S1_Mb | 8 | conditionbas:order3-1 | 1.53482724882167 | 0.0865119359243593 | 17.7412195487525 | 2.01513588663215e-70 | FALSE | 57600 |
| F2_pixels | S1 (Mb) | S1_Mb | 8 | conditionsup:order3-1 | 0.970443412105546 | 0.0869738376153922 | 11.1578773423446 | 6.55376841388519e-29 | FALSE | 57600 |

### 11.2 Veinticuatro ítems (`results/F0_24.csv` y `results/S1_24.csv`)

| encoder | origen | model | items | term | estimate | se | z | p | singular | nobs |
|---|---|---|---|---|---|---|---|---|---|---|
| F0_nodistance | F0 | F0 | 24 | conditionsub:order3-1 | 0.567102355619594 | 0.0454730916016611 | 12.4711633989469 | 1.07248691664029e-35 | FALSE | 172800 |
| F0_nodistance | F0 | F0 | 24 | conditionbas:order3-1 | 0.426748138024509 | 0.0416722569900703 | 10.2405813567092 | 1.3045013568154e-24 | FALSE | 172800 |
| F0_nodistance | F0 | F0 | 24 | conditionsup:order3-1 | 0.341109480196817 | 0.0388414503089952 | 8.78209947062199 | 1.60451091478446e-18 | FALSE | 172800 |
| R0_dinov2s | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 2.00267500955405 | 0.0916585538668056 | 21.8492974748898 | 7.89378550653324e-106 | FALSE | 172800 |
| R0_dinov2s | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 1.17490893327274 | 0.0797140598885207 | 14.7390427098537 | 3.61877964765142e-49 | FALSE | 172800 |
| R0_dinov2s | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 1.01767245919144 | 0.0684289734694656 | 14.8719527357157 | 5.01295635545465e-50 | FALSE | 172800 |
| O1_dinov2b | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 1.92763877826394 | 0.0902566364469349 | 21.3573079404224 | 3.33484895636909e-101 | FALSE | 172800 |
| O1_dinov2b | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 1.20821977340847 | 0.0797954348514562 | 15.1414648677289 | 8.62704555117748e-52 | FALSE | 172800 |
| O1_dinov2b | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 1.08426224773877 | 0.0678490934743139 | 15.9804971918931 | 1.74746552626079e-57 | FALSE | 172800 |
| R1_cvcl | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 1.58560610265205 | 0.0772395551990441 | 20.5284209439838 | 1.20011780045515e-93 | FALSE | 172800 |
| R1_cvcl | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 1.21789771702288 | 0.0695130824942181 | 17.5204101634276 | 1.00101125017771e-68 | FALSE | 172800 |
| R1_cvcl | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 0.959762560655725 | 0.0617967752299775 | 15.5309489384188 | 2.14186687926221e-54 | FALSE | 172800 |
| E1_siglip | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 2.30480393244628 | 0.0953123246559427 | 24.1815939414565 | 3.47518676333246e-129 | FALSE | 172800 |
| E1_siglip | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 1.6347428721537 | 0.0961727376612004 | 16.9979862475435 | 8.49897013290348e-65 | FALSE | 172800 |
| E1_siglip | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 1.33931629711618 | 0.0796205021598493 | 16.8212490600388 | 1.7052524767759e-63 | FALSE | 172800 |
| O2_openclip | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 2.36757350423398 | 0.0957490918712543 | 24.7268507508923 | 5.50195528030304e-135 | FALSE | 172800 |
| O2_openclip | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 1.628199511776 | 0.0980732504677577 | 16.6018716011791 | 6.75513411162734e-62 | FALSE | 172800 |
| O2_openclip | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 1.24003024649817 | 0.0810182519547227 | 15.3055665430941 | 7.01881397901156e-53 | FALSE | 172800 |
| E2_vitsup | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 2.59177475991642 | 0.10416071999021 | 24.8824581873093 | 1.15216966088659e-136 | FALSE | 172800 |
| E2_vitsup | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 1.58135143998678 | 0.0965773810252793 | 16.3739316928967 | 2.93617710052078e-60 | FALSE | 172800 |
| E2_vitsup | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 1.25675744070201 | 0.0844685717270031 | 14.8784028782181 | 4.5523748012072e-50 | FALSE | 172800 |
| E3_rn50sup | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 2.49460617458897 | 0.101931587195086 | 24.473337885091 | 2.84072793951194e-132 | FALSE | 172800 |
| E3_rn50sup | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 1.70853740035052 | 0.0935464013152009 | 18.2640633560416 | 1.59933727206559e-74 | FALSE | 172800 |
| E3_rn50sup | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 1.53513009767131 | 0.0755764793109538 | 20.3122732319288 | 1.00159695396263e-91 | FALSE | 172800 |
| E4_mae | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 1.04464026490879 | 0.0643519655851992 | 16.2332301027501 | 2.93594536486605e-59 | FALSE | 172800 |
| E4_mae | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 0.768121246205017 | 0.053500994353957 | 14.3571396285312 | 9.61128201815712e-47 | FALSE | 172800 |
| E4_mae | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 0.658556539311444 | 0.0517853621772711 | 12.717040329989 | 4.75533070789772e-37 | FALSE | 172800 |
| F1_rn50rand0 | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 0.710798632011427 | 0.0565890613013318 | 12.5607072403355 | 3.47215167528572e-36 | FALSE | 172800 |
| F1_rn50rand0 | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 0.526164254394434 | 0.0481388048862174 | 10.9301478430571 | 8.27135535659107e-28 | FALSE | 172800 |
| F1_rn50rand0 | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 0.490737462725742 | 0.0431195759728265 | 11.3808508468404 | 5.20890307020375e-30 | FALSE | 172800 |
| F2_pixels | S1 (Mb) | S1_Mb | 24 | conditionsub:order3-1 | 0.747768333615192 | 0.0539154889615074 | 13.8692674038263 | 9.7265122759863e-44 | FALSE | 172800 |
| F2_pixels | S1 (Mb) | S1_Mb | 24 | conditionbas:order3-1 | 0.552605048170324 | 0.0468509283212535 | 11.7949647524837 | 4.14374087695264e-32 | FALSE | 172800 |
| F2_pixels | S1 (Mb) | S1_Mb | 24 | conditionsup:order3-1 | 0.443972363800872 | 0.043984860717664 | 10.0937539998297 | 5.88739716257353e-24 | FALSE | 172800 |

## 12. S6 (preprocesamiento común)

El plan §5.4 y §10.3 (paso 9b) sitúan S6 al final: dos etapas, 8 ítems, preprocesamiento común; si falla, no detiene nada y se registra como desviación. El log del domingo no contiene `S6 FAILED`. `results/stage2_S6.csv` tiene 10 filas (un encoder primario por fila), escritas según el log tras el paso S6.

En esas 10 filas: `variant = primary`, `items = 8`, `suffix = _s6common`, `status = ok`, `n_exp = 12`, `ci95_contiguous = True`, `ci95_at_grid_edge = False`, `ci90_contiguous = True`, `ci90_at_grid_edge = False`. El peldaño de la etapa 1, según `results/stage1_S6_g*.csv` y `results/stage2_S6_fit_type_by_order.csv`, es L4 en los 12 experimentos, con `n_failed_* = 0`. `08_regions.py` no se aplicó a S6.

Valores de la etapa 2, copiados de `results/stage2_S6.csv`:

| encoder | items | variant | suffix | n_exp | status | delta_hat | mean_31 | mean_13 | p_two_sided | p_opposite | p_positive | ci95_lo | ci95_hi | ci95_contiguous | ci95_at_grid_edge | ci90_lo | ci90_hi | ci90_contiguous | ci90_at_grid_edge | p_tost_0.05 | p_tost_0.10 | p_tost_0.20 | welch_p | welch_df | welch_ci95_lo | welch_ci95_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | primary | _s6common | 12 | ok | 0.019695760170422072 | -0.466106953075103 | -0.4858027132455251 | 0.8701298701298701 | 0.566017316017316 | 0.43506493506493504 | -0.204589 | 0.274411 | True | False | -0.188589 | 0.218411 | True | False | 0.41233766233766234 | 0.20995670995670995 | 0.06277056277056277 | 0.8529073545633398 | 5.152537404198614 | -0.23774994808082617 | 0.2771414684216703 |
| O1_dinov2b | 8 | primary | _s6common | 12 | ok | -0.06255461917835603 | -0.2031514365884227 | -0.14059681741006666 | 0.6493506493506493 | 0.3246753246753247 | 0.6764069264069265 | -0.369549 | 0.244451 | True | False | -0.345549 | 0.220451 | True | False | 0.5422077922077922 | 0.38311688311688313 | 0.14285714285714285 | 0.6543089760993708 | 5.169736391951081 | -0.39790665244810325 | 0.2727974140913912 |
| R1_cvcl | 8 | primary | _s6common | 12 | ok | 0.03417361929330476 | -0.15852334024536854 | -0.1926969595386733 | 0.48484848484848486 | 0.7586580086580087 | 0.24242424242424243 | -0.073106 | 0.141894 | True | False | -0.053106 | 0.118894 | True | False | 0.36904761904761907 | 0.09415584415584416 | 0.004329004329004329 | 0.4905664518664077 | 7.828456931325235 | -0.07520183232188937 | 0.1435490709084989 |
| E1_siglip | 8 | primary | _s6common | 12 | ok | -0.008229029595384851 | -1.2265679734996142 | -1.2183389439042294 | 0.948051948051948 | 0.474025974025974 | 0.5270562770562771 | -0.264678 | 0.249322 | True | False | -0.219678 | 0.200322 | True | False | 0.3625541125541126 | 0.22402597402597402 | 0.06168831168831169 | 0.9446384390901736 | 9.937678850951922 | -0.26590910835858167 | 0.24945104916781197 |
| O2_openclip | 8 | primary | _s6common | 12 | ok | -0.07861756752485682 | -1.6340321181162916 | -1.5554145505914347 | 0.6060606060606061 | 0.30303030303030304 | 0.698051948051948 | -0.356904 | 0.245096 | True | False | -0.319904 | 0.176096 | True | False | 0.5670995670995671 | 0.45021645021645024 | 0.23593073593073594 | 0.5959493681236333 | 6.681225233265069 | -0.4158766466384469 | 0.2586415115887333 |
| E2_vitsup | 8 | primary | _s6common | 12 | ok | -0.05444055218764399 | -0.45080604373624983 | -0.39636549154860584 | 0.6948051948051948 | 0.3474025974025974 | 0.6536796536796536 | -0.369468 | 0.277532 | True | False | -0.332468 | 0.220532 | True | False | 0.5119047619047619 | 0.3787878787878788 | 0.15692640692640691 | 0.7088614147238718 | 5.5106751494669055 | -0.4005711049188018 | 0.29169000054351385 |
| E3_rn50sup | 8 | primary | _s6common | 12 | ok | -0.02541038777745297 | -0.9873587273149379 | -0.9619483395374849 | 0.7164502164502164 | 0.3582251082251082 | 0.6428571428571429 | -0.172086 | 0.126914 | True | False | -0.148086 | 0.106914 | True | False | 0.37337662337662336 | 0.16233766233766234 | 0.008658008658008658 | 0.7228897695410806 | 7.232968546418875 | -0.18730864632443603 | 0.1364878707695301 |
| E4_mae | 8 | primary | _s6common | 12 | ok | 1.2857183103863015 | -1.1423154470261856 | -2.428033757412487 | 0.22077922077922077 | 0.8906926406926406 | 0.11038961038961038 | -0.831583 | 3.509417 | True | False | -0.437583 | 3.014417 | True | False | 0.8831168831168831 | 0.8744588744588745 | 0.8517316017316018 | 0.22246938169746763 | 7.792113888587713 | -0.9610878270402787 | 3.5325244478128814 |
| F1_rn50rand0 | 8 | primary | _s6common | 12 | ok | -0.04646227329427677 | -0.42759618431617447 | -0.3811339110218977 | 0.6103896103896104 | 0.3051948051948052 | 0.6958874458874459 | -0.243163 | 0.147837 | True | False | -0.216163 | 0.114837 | True | False | 0.48917748917748916 | 0.275974025974026 | 0.06277056277056277 | 0.6095068266426835 | 9.710117889677846 | -0.2434135980153684 | 0.15048905142681487 |
| F2_pixels | 8 | primary | _s6common | 12 | ok | 0.011831747062335474 | -0.13551391442344138 | -0.14734566148577685 | 0.7380952380952381 | 0.6320346320346321 | 0.36904761904761907 | -0.065716 | 0.088284 | True | False | -0.050716 | 0.072284 | True | False | 0.14393939393939395 | 0.016233766233766232 | 0.0010822510822510823 | 0.7386040564873767 | 9.611582238251094 | -0.06533419180200452 | 0.08899768592667547 |

Tipo de ajuste × orden, copiado de `results/stage2_S6_fit_type_by_order.csv`:

| encoder | items | variant | suffix | n_L1_1-3 | n_L1_3-1 | n_L2_1-3 | n_L2_3-1 | n_L3_1-3 | n_L3_3-1 | n_L4_1-3 | n_L4_3-1 | n_failed_1-3 | n_failed_3-1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| O1_dinov2b | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| R1_cvcl | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E1_siglip | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| O2_openclip | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E2_vitsup | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E3_rn50sup | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| E4_mae | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| F1_rn50rand0 | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |
| F2_pixels | 8 | primary | _s6common | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 6 | 0 | 0 |

`logs/01b.log` registra, para cada uno de los diez encoders primarios, una línea `S6 <encoder>` (R0_dinov2s, O1_dinov2b, R1_cvcl, E1_siglip, O2_openclip, E2_vitsup, E3_rn50sup, E4_mae, F1_rn50rand0, F2_pixels). En el tramo de E4_mae el mismo log incluye mensajes del cargador de Hugging Face con claves del decodificador marcadas como `UNEXPECTED` (el log aclara que «UNEXPECTED: can be ignored when loading from different task/architecture»). A continuación aparece `E4_mae (45, 768)` y `S6 E4_mae`. Existen los archivos `out/emb_*_s6common.npz` para los diez encoders primarios y `out/encoder_specs_s6common.json` (marca de tiempo de este último: 22-11-2026, 09:34, hora de Santiago).

## 13. Advertencia Pandas4Warning (`fillna`)

La advertencia apareció en el log del sábado (paso 7) y en el del domingo (al escribir `stage2_sensitivities.csv` y al correr `08_regions.py` con S9). El texto, en ambos días, es:

```
Pandas4Warning: 'str' is not supported as a fill value for float64 dtype. In a future version, calling fillna with an incompatible value will raise. Explicitly cast to a common dtype before filling.
```

Sitios en los scripts congelados (no se ha editado ninguno):

- `scripts/06_stage2_permutation.py`, línea 36: `s1["suffix"] = s1["suffix"].fillna("")`.
- `scripts/08_regions.py`, línea 145: `s2 = s2[(s2.variant == "primary") & (s2.suffix.fillna("") == "")]`.
- `scripts/08_regions.py`, línea 149: `s9 = s9[(s9.variant == "S9_mixed") & (s9.suffix.fillna("") == "")]`.

Entorno instalado al revisar esta advertencia: Python 3.13.5, pandas 3.1.0 (`/workspace/rv_venv`).

Hecho observado en los archivos, sin reejecutar la tubería:

- En los CSV de etapa 1 que alimentan el primario y las sensibilidades S4, S5, S7 y S8, la columna `suffix` está vacía en el texto del archivo. Al leerla con pandas 3.1.0 queda con dtype `float64` y valor `NaN` en todas las filas (pandas interpreta el campo vacío como NA numérico). En los CSV de S6, `suffix` vale `_s6common` y se lee como texto; ahí no aplica este `fillna` con cadena sobre un `float64` vacío.
- `06_stage2_permutation.py` usa `fillna("")` solo sobre la columna `suffix`, que es una clave de agrupación, no un estadístico. Después escribe `results/stage2_primary.csv` y `results/stage2_sensitivities.csv` con `suffix` de nuevo como campo vacío. Esos archivos se leen en `08_regions.py` otra vez como `float64` / `NaN`.
- En pandas 3.1.0, `fillna("")` sobre una serie `float64` de solo `NaN` sigue rellenando con la cadena vacía y pasa la serie a dtype `object`. Emite Pandas4Warning y avisa que en una versión futura la llamada levantará un error. No deja `NaN` sin rellenar.
- Con ese comportamiento, el filtro de `08_regions.py` (`variant == "primary"` y `suffix` vacío, o `variant == "S9_mixed"` y `suffix` vacío) selecciona 20 filas en cada caso (10 encoders × 2 conjuntos de ítems), que es lo que corresponde a esas variantes en `results/stage2_primary.csv`. Las tablas de regiones tienen 10 filas (una por encoder primario), porque el script agrupa después por encoder.
- Ninguna columna numérica de la etapa 2 (`delta_hat`, IC, p, TOST, Welch) pasa por ese `fillna`. La línea 46 de `06_stage2_permutation.py` (`converged.fillna(True)`) no aparece en los logs; en los CSV de etapa 1 de S4–S8 y S6, `converged` se lee como booleano y no tiene NA.

Con la versión instalada, la advertencia no cambió ningún valor numérico reportado ni añadió o quitó filas respecto del filtro previsto (suffix vacío). El riesgo que señala el aviso es de cara a una versión futura de pandas, que dejaría de aceptar una cadena como valor de `fillna` sobre `float64`.

## 14. Completitud y archivos citados

- No se modificó ningún archivo congelado ni ningún archivo de `results/`. Este documento es un archivo nuevo, `encoder_equivalence/RESULTADOS_encoders.md`, que no forma parte del manifiesto de 47 archivos.
- No existe `results/structure_DEVIATION.txt`.
- `08_regions.py` no produjo etiquetas de región para S1, S2, S3, S4, S5, S7, S8, F0 ni S6. El plan pide informar si una sensibilidad contradice la conclusión; para S9 esa marca ya está en `tipo_de_ajuste` y en las columnas `region_S9_*` (§10.2). Para las demás sensibilidades, este documento no calcula una región.
- En `results/stage2_sensitivities.csv`, S7 y S8 comparten el valor `variant = primary` con el nombre de variante del análisis primario, pero viven en otro archivo y se refieren a encoders distintos (`F1_rn50rand1`, `F1_rn50rand2`, `E4s_mae_cls`). Se deja constancia para evitar una lectura cruzada con `results/stage2_primary.csv`.
- Los embeddings de S6 (`out/emb_*_s6common.npz`) se escribieron después del congelamiento; no están en el manifiesto de 47 archivos. Tras escribir este documento, el 22-11-2026 a las 09:47:52 (hora de Santiago), se volvió a correr `scripts/09_freeze.py --verify`: `VERIFY 47 files: OK`, con código de salida 0; `files not in manifest` lista exactamente esos diez `out/emb_*_s6common.npz`. Este documento y `DESVIACIONES_encoders_v2.md` tampoco forman parte del manifiesto; el verificador no los enumera, porque solo compara el conjunto que `09_freeze.py` sabe listar.
- Archivos de resultados citados: `results/pipeline_control.json`, `results/pipeline_control.log`, `results/structure_status.json`, `results/structure_structure.csv`, `results/structure_primary.csv`, `results/structure_S9_mixed.csv`, `results/structure_per_encoder.csv`, `results/poscontrol.csv`, `results/stage2_primary.csv`, `results/stage2_primary_fit_type_by_order.csv`, `results/stage2_sensitivities.csv`, `results/stage2_sensitivities_fit_type_by_order.csv`, `results/stage2_S6.csv`, `results/stage2_S6_fit_type_by_order.csv`, `results/regions/regions_report.md`, `results/regions/regions_table.csv`, `results/regions/global_grid.csv`, `results/regions/floors.csv`, `results/regions_with_S9/regions_report.md`, `results/regions_with_S9/regions_table.csv`, `results/S1_24.csv`, `results/S1_8.csv`, `results/S2.csv`, `results/S3.csv`, `results/F0_24.csv`, `results/F0_8.csv`, `out/equivalence_grouping.json`, `DESVIACIONES_encoders_v2.md`, `FREEZE_encoders_v2.txt`, `FREEZE_encoders_v2_complemento.txt`, `logs/run_saturday_2026-11-21.log`, `logs/run_sunday_2026-11-21.log`, `logs/watch_s2_pause.log`, `logs/poscontrol.log`, `logs/01b.log`.
