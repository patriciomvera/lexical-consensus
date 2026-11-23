# Resultados del anexo de agentes (comparación de encoders v2)

Documento interno. No es una publicación. Compilación fáctica de lo que produjo la tubería congelada de agentes el 23 de noviembre de 2026 (hora de Santiago, zona America/Santiago, UTC−3). No interpreta los resultados, no propone títulos ni encuadres, y no añade conclusiones distintas de las que escribieron las reglas de decisión del plan al ser aplicadas por los scripts congelados. Don Patricio revisa e interpreta.

Los números se copian de los archivos citados, con la grafía del archivo (se mantiene el punto decimal de los CSV y de los JSON). Si un dato no está o es ambiguo, se declara en el mismo apartado. Las rutas de resultados se entienden bajo `encoder_equivalence/agents/out/results/`, salvo indicación contraria.

## Índice

1. Procedencia, congelamiento y tiempos de ejecución
2. Desviaciones registradas
3. Paridad de embeddings L&F
4. Controles de tubería y control positivo de agentes
5. Dispersión de d_perc en los 100 pares (§3.6)
6. T (descriptivo)
7. D (100 pares)
8. D-todos (4.950 pares)
9. H1, H2 y H3
10. H4 (réplica por encoder)
11. Regiones, familias Holm y conclusiones GA
12. Tabla agentes × personas y figura x1
13. Completitud, archivos citados y observaciones de forma

---

## 1. Procedencia, congelamiento y tiempos de ejecución

- **Trabajo:** anexo de agentes de la comparación de encoders v2, de arXiv 2611.14116 (Lexical Consensus, Open Mind).
- **Plan congelado:** `propuesta_agentes_multi_encoder_v2_2026-11-18.md`, sha256 `633a5e7eabf0a8cc387e4d884d8d42d68ff40289a745773cde724cf93300db25` (hash escrito en `FREEZE_agentes_v2.txt` y recomputado sobre el archivo). En el manifiesto también figura la instantánea `propuesta_agentes_bfc99fac_snapshot.md`, sha256 `bfc99face128a5b202b92c29b3c6e33b0adc8da83dcbf2539ddc930394f888e8`.
- **Aclaraciones y decisiones de (b):** `encoder_equivalence/agents/ACLARACIONES_scripts.md`, sha256 `c7d4763ae39c1ac0e07d60854dfa2dab0c88c2a0549ae44e1278334cd74420be` (listado en el manifiesto).
- **Manifiesto de congelamiento:** `encoder_equivalence/FREEZE_agentes_v2.txt`. El propio manifiesto tiene sha256 `5a33e098a899877d813e2bcae830798ad11b51f74c60b1e542cfbda26bd93673`. Cabecera: «FREEZE agentes v2 (arXiv 2611.14116, addendum to FREEZE_encoders_v2.txt) - internal freeze, NOT an external preregistration»; «time: 2026-11-22 18:00:36 America/Santiago (UTC-0300)»; «files: 99».
- **Congelamiento humano** (exigido por el de agentes): `encoder_equivalence/FREEZE_encoders_v2.txt`, sha256 `975aab4cc922a73b3b531fb22f26ce87145143c89a8b8ea6d44430a00894b606`.
- **Commit:** no se pudo leer un hash de commit desde esta copia de trabajo. El directorio `/workspace/lexical-consensus-openmind` no contiene `.git` (no es un repositorio git en esta máquina).
- **Verificación al arrancar y al reanudar** (`agents/scripts/15_freeze_agents.py --verify` y `scripts/09_freeze.py --verify`), copiada de `agents/logs/run_agents.log`:
  - 23-11-2026, 08:38:27, hora de Santiago: `VERIFY agentes 99 files: OK; files not in manifest: none`; a continuación, `VERIFY 47 files: OK; files not in manifest: ['encoder_equivalence/out/emb_E1_siglip_s6common.npz', …, 'encoder_equivalence/out/emb_R1_cvcl_s6common.npz']` (los diez `emb_*_s6common.npz`, los mismos que al congelar la parte humana); `lme4 2.0.6`.
  - 23-11-2026, 14:43:39, hora de Santiago (reanudación): las mismas dos líneas `VERIFY … OK`, más `lme4 2.0.6`.
- **Verificación al redactar este documento** (23-11-2026, 16:54 aprox., hora de Santiago), de nuevo con `--verify` y sin `--freeze`: `VERIFY agentes 99 files: OK; files not in manifest: none`; `VERIFY 47 files: OK` con la misma lista de `emb_*_s6common.npz` fuera del manifiesto. Las fechas de modificación de `FREEZE_agentes_v2.txt` y `FREEZE_encoders_v2.txt` no cambiaron (22-11-2026, 18:00:36 y 13:40:36, respectivamente).
- **Sha256 de los resultados:** los 73 archivos de `out/results/` (incluidos los de `regions/`) coinciden en nombre, tamaño y sha256 con `/workspace/agents_respaldo/results/`. La comprobación `sha256sum -c SHA256SUMS` de `/workspace/agents_respaldo/results/SHA256SUMS` (73 líneas) dio OK. Lo mismo para los 30 archivos de embeddings: `/workspace/agents_respaldo/emb/SHA256SUMS` dio OK, e idénticos a `out/emb/`. El inventario de resultados usado al cerrar la corrida está también en `/tmp/primary_done_package_2026-11-23/results_inventory.txt`.
- **Nada se publica** (última línea del log: «Nothing is published.»).

### 1.1 Tiempos de la corrida (23-11-2026, hora de Santiago)

Fuentes: `agents/logs/run_agents.log`, `agents/logs/run_start_epoch` (1795433909 = 08:38:29) y `encoder_equivalence/DESVIACIONES_encoders_v2.md` (sección del anexo de agentes).

| marca en el log | texto del log |
|---|---|
| 2026-11-23 08:38:27 -03 | 0. checks |
| 2026-11-23 08:38:29 -03 | start 2026-11-23 08:38:29 -03; cap 14 h wall |
| 2026-11-23 08:38:29 -03 | 1. parity of the encoders with the frozen human embeddings (45 L&F stimuli; &#124;diff&#124; <= 1e-5): FAIL = stop |
| 2026-11-23 08:39:08 -03 | 2. pipeline control 1 (§3.5): T with R0 from the v1 CIFAR-10 cache vs the v1 ledger |
| (misma marca, línea siguiente) | control_T ok = True &#124; rule propuesto_b1 |
| 2026-11-23 08:39:47 -03 | 3. CIFAR embeddings (c100train, c10pool, c10test; existing caches reused; E4s from the E4 pass) |
| (tras ~15 min del paso 3) | primer rate check (texto en el §2 y en DESVIACIONES) |
| ~10:24–10:43 | suspensión del computador (no es una línea del log de la corrida; ver DESVIACIONES, entrada 2 del anexo) |
| ~14:37 | reinicio del computador; la corrida se cortó (ver DESVIACIONES, entrada 3 del anexo) |
| 2026-11-23 14:43:18 -03 | RESUME after box restart (~14:37, uptime): same frozen command, outputs and E4 parts reused; cap still from original start epoch |
| 2026-11-23 14:43:18 -03 | /tmp/renv missing: rebuilding R 4.5.3 + lme4 2.0.6 (conda-forge, micromamba) |
| 2026-11-23 14:43:39 -03 | /tmp/XTMEM restored from /workspace/agents_respaldo/XTMEM |
| 2026-11-23 14:43:39 -03 | 0. checks (VERIFY agentes 99 OK; VERIFY 47 OK) |
| 2026-11-23 14:43:41 -03 | start 2026-11-23 08:38:29 -03; cap 14 h wall |
| 2026-11-23 14:43:41 -03 | 1. parity … FAIL = stop |
| 2026-11-23 14:44:26 -03 | 2. pipeline control 1 … |
| (misma marca, línea siguiente) | control_T ok = True &#124; rule propuesto_b1 |
| 2026-11-23 14:45:08 -03 | 3. CIFAR embeddings … |
| (tras ~15 min del paso 3 reanudado) | segundo rate check |
| 2026-11-23 15:21:39 -03 | backup: …/out/emb -> /workspace/agents_respaldo/emb (30 files, SHA256SUMS) |
| 2026-11-23 15:21:39 -03 | 4. §3.6 dispersion of d_perc on the 100 v1 pairs (BEFORE any accuracy) |
| 2026-11-23 15:21:51 -03 | 5. T (seed 42; R0 already done in step 2), D (100 v1 pairs) and D-todos (4,950 pairs, mandatory) |
| (durante el paso 5, sin marca de reloj propia) | control_c100: PASA |
| 2026-11-23 16:16:58 -03 | 6. H1/H2/H3 |
| 2026-11-23 16:21:38 -03 | 7. H4 per encoder (people; replica of v1 H4, 24 items) |
| 2026-11-23 16:44:32 -03 | 8. regions, GA, families, agents x humans table (14) and fig x1 (14b) |
| 2026-11-23 16:44:36 -03 | backup: …/out/results -> /workspace/agents_respaldo/results (73 files, SHA256SUMS) |
| 2026-11-23 16:44:36 -03 | primary done. Next: sensitivities (optional), then RESULTADOS_agentes.md + DESVIACIONES by hand; notify Evaristo. Nothing is published. |

Duración de calendario desde el start (08:38:29) hasta `primary done` (16:44:36): 8 h 6 min 7 s, incluido el hueco de la suspensión y el del reinicio. El tope era 22:38:29 (14 h desde 08:38:29); el margen residual fue 5 h 53 min 53 s. H4 no se recortó: hay `h4_*.csv` de los diez encoders primarios y `h4_by_encoder.csv`, y a continuación corrió el paso 8.

En el paso 1, la frase «FAIL = stop» es el texto del encabezado del log (la corrida se detendría si la paridad fallara); no es un fallo. La paridad pasó, según el §3.

---

## 2. Desviaciones registradas

Se transcribe, sin cambios, la sección «Anexo de agentes» de `encoder_equivalence/DESVIACIONES_encoders_v2.md` tal como quedó al escribir este documento. Las tres entradas de la parte humana de ese archivo no se reproducen aquí y no se modificaron.

## Anexo de agentes (plan `propuesta_agentes_multi_encoder_v2_2026-11-18.md`, congelamiento `FREEZE_agentes_v2.txt`, corrida del 23-11-2026)

Las tres entradas anteriores (parte humana) no se modifican. Horas en America/Santiago (UTC−3).

1. **Tasa del paso 3 por debajo del 50 % de la referencia (cómputo declarado, REVISION5 C5).** A los 15 minutos de los embeddings, `run_agents.sh` comprueba la tasa (img/min) y, si queda bajo el 50 % de la referencia, la registra y pide reestimar, sin detener la corrida ni cambiar scripts. Ocurrió en los dos rate checks del 23-11-2026, según `encoder_equivalence/agents/logs/run_agents.log`:

   - Arranque original (tras el inicio del paso 3 a las 08:39:47): `[rate check 15 min] img/min: {'R0_dinov2s': 1074, 'O1_dinov2b': 332, 'R1_cvcl': 466}`. Las tres tasas quedaron bajo el 50 % de su referencia (4200, 1200 y 2400 img/min, respectivamente).
   - Tras la reanudación (paso 3 reiniciado a las 14:45:08): `[rate check 15 min] img/min: {'R0_dinov2s': 1121, 'E1_siglip': 390, 'E3_rn50sup': 738, 'F2_pixels': 59965, 'O1_dinov2b': 333, 'O2_openclip': 2231, 'E4_mae': 426, 'R1_cvcl': 466, 'E2_vitsup': 431, 'F1_rn50rand0': 847}`. El log registró «< 50 %» para todos esos encoders salvo F2_pixels (que no aparece en esas líneas).

   La reestimación manual quedó en `encoder_equivalence/agents/logs/reestimacion_tasa_2026-11-23.md`: alrededor de 0,27× respecto de la referencia del 22-11. El tope de 14 h de reloj, medido desde las 08:38:29, se mantuvo (22:38:29). La corrida primaria terminó a las 16:44:36, dentro del tope, y H4 no se recortó.

2. **Suspensión del computador entre ~10:24 y ~10:43.** Registrada en la nota de las 10:45 de `encoder_equivalence/agents/logs/reestimacion_tasa_2026-11-23.md`. El computador se suspendió al quedar sin sesión activa de agente, aunque había un keep-awake por shell. Los procesos de embeddings siguieron vivos y retomaron solos. Se perdieron ~19 min de reloj; el archivo de la nota indica que los resultados no cambian (embeddings por partes, deterministas).

3. **Reinicio del computador ~14:37 y reanudación a las 14:43:18.** El reinicio cortó la corrida en el paso 3, con E4_mae aún en c100train (partes en `out/emb/.parts_c100train_E4_mae`). `/workspace` quedó intacto; `/tmp` se borró (`/tmp/renv` y `/tmp/XTMEM`). A las 14:43:18 se reanudó el mismo comando congelado (`bash scripts/run_agents.sh primary`), según `encoder_equivalence/agents/logs/run_agents.log` y `encoder_equivalence/agents/logs/resume_1437.out`. El paso 0 reconstruyó R 4.5.3 + lme4 2.0.6 y restauró XTMEM desde `/workspace/agents_respaldo/XTMEM`. Los embeddings ya terminados y las partes de E4 se reutilizaron. El tope siguió midiéndose desde el epoch original (`logs/run_start_epoch` = 1795433909 = 08:38:29). La paridad (paso 1, 14:43:41) y el control 1 (paso 2, 14:44:26; `control_T ok = True | rule propuesto_b1`) se repitieron y aprobaron. No hubo «R rebuild failed». Nota en `reestimacion_tasa_2026-11-23.md` (14:45).

4. **Sensibilidades opcionales no corridas.** No se ejecutó `bash agents/scripts/run_agents.sh sensitivities` (T-sem "episodes" ya forma parte del primario con semilla 42; el resto del bloque opcional —E4s, F1 semillas 1–2, H5, T-sem "full"— queda pendiente de la aprobación de Patricio). En `agent_pairs_dispersion.json`, F1_rn50rand1 y F1_rn50rand2 figuran como «tecnico: falta» el npz de c100train. E4s_mae_cls tiene embeddings y dispersión (salen del paso de E4 del primario), pero no hay `T_`, `D_`, `Dall_`, `H_` ni `h4_` de E4s ni de esas semillas. En `regions/agents_regions_report.md` y `regions/regions_agents.csv`, E4s_mae_cls, F1_rn50rand1 y F1_rn50rand2 quedan en «inconcluso (técnico)».

---

## 3. Paridad de embeddings L&F

El paso 1 compara, para cada encoder, los embeddings de los 45 estímulos de Lewis & Frank con los embeddings humanos congelados. La regla del log y de `10_agent_embed.py --parity` es &#124;diff&#124; ≤ 1e-5; si no se cumple, la corrida se detiene. Fuente de cada fila: `parity_<encoder>.json`. En el de E4_mae también aparece el detalle de E4s_mae_cls (el paso de E4 escribe ambos).

| archivo | encoder (campo) | ok | max_abs_diff | tol | frozen_sha256 |
| --- | --- | --- | --- | --- | --- |
| `parity_R0_dinov2s.json` | R0_dinov2s | true | 0.0 | 1e-05 | `f80e1890c6fb8babd89a0245f4d8329822c4127e07ca8bbdbbc80a38aba69176` |
| `parity_O1_dinov2b.json` | O1_dinov2b | true | 0.0 | 1e-05 | `3b2fddd24253d87d93c493da55f0ff5b155750a88eb10c628e910847515054d0` |
| `parity_R1_cvcl.json` | R1_cvcl | true | 0.0 | 1e-05 | `ae62f0c988ee2f524852a1ce96969a223ff63b6d5894ecc652bf2cc5dbf72c85` |
| `parity_E1_siglip.json` | E1_siglip | true | 8.940696716308594e-08 | 1e-05 | `2ec894ec6d762b43d15b85ed762cd3cd69056f6490af71abb67bb7a5c28fb267` |
| `parity_O2_openclip.json` | O2_openclip | true | 0.0 | 1e-05 | `cefd13c6b2b40233551480741666c4226c6f677850da3ff8a493f77a47b37b1e` |
| `parity_E2_vitsup.json` | E2_vitsup | true | 0.0 | 1e-05 | `f3cb539282613a192dda3874447380e94f9a18439d9551ecd21e4ee2326b88e6` |
| `parity_E3_rn50sup.json` | E3_rn50sup | true | 0.0 | 1e-05 | `7b35559f826208069562a967fba4581a1e234f0724e4f691debbff344955e2ad` |
| `parity_E4_mae.json` | E4_mae | true | 0.0 | 1e-05 | `9a0b58274b47305c1d684b13c282711e3faa4c682f01d4e8b588d0d11f3f7cae` |
| `parity_E4_mae.json` | E4s_mae_cls | true | 0.0 | 1e-05 | `02ea9c198c9491bafc26d92238b4ffcd79700eac4498a3bcb7281e23dfa9ffbb` |
| `parity_F1_rn50rand0.json` | F1_rn50rand0 | true | 0.0 | 1e-05 | `5c95e69f81369111871b8b845716f6e4dd61fdba69c34dd05eec5dcbeafa8402` |
| `parity_F2_pixels.json` | F2_pixels | true | 0.0 | 1e-05 | `f55b2c53f5743e6fcafec0966f4c573033b82524eeca42b6f6a477318cf6f1a9` |

Los diez archivos tienen `"ok": true`. El único `max_abs_diff` distinto de `0.0` es el de E1_siglip (`8.940696716308594e-08`), inferior a `tol` (`1e-05`). E4s_mae_cls (dentro de `parity_E4_mae.json`) también tiene `"ok": true` y `max_abs_diff` `0.0`.

---

## 4. Controles de tubería y control positivo de agentes

### 4.1 Control 1 (T con R0 frente al ledger de la v1)

La regla congelada por defecto es `propuesto_b1` (ACLARACIONES (b2), REVISION5): las mismas 5.040 filas y claves; para los cinco aprendices no aleatorios, `c1`, `c2` y `per_concept_c1` idénticos y &#124;Δ margin&#124; ≤ 1e-6; el aprendiz `random` se reporta y no se exige. El log escribe `control_T ok = True | rule propuesto_b1` a las 08:39:08 y, de nuevo, a las 14:44:26. El resultado se lee de `control_T_exact.json` (no solo del log). Contenido completo de ese JSON:

```json
{
  "n_new": 5040,
  "n_v1": 5040,
  "estricto": false,
  "propuesto_b1": true,
  "keys_identical": true,
  "nonrandom_mismatches": {},
  "random_c1_by_tier": {
    "native": {
      "mean_c1_new": 0.33688893333333336,
      "mean_c1_v1": 0.34622213333333346,
      "sd_c1_new": 0.029551410403942182,
      "sd_c1_v1": 0.03630863407945218,
      "n": 30,
      "chance": 0.3333333333333333
    },
    "near": {
      "mean_c1_new": 0.3313332999999999,
      "mean_c1_v1": 0.34399993333333334,
      "sd_c1_new": 0.032937636022388334,
      "sd_c1_v1": 0.03393248547427998,
      "n": 30,
      "chance": 0.3333333333333333
    },
    "mid": {
      "mean_c1_new": 0.3322222,
      "mean_c1_v1": 0.32888886666666667,
      "sd_c1_new": 0.02743789035670685,
      "sd_c1_v1": 0.030127330606094016,
      "n": 30,
      "chance": 0.3333333333333333
    },
    "far": {
      "mean_c1_new": 0.32422219999999996,
      "mean_c1_v1": 0.3415555666666667,
      "sd_c1_new": 0.039308170734509304,
      "sd_c1_v1": 0.03505268084819698,
      "n": 30,
      "chance": 0.3333333333333333
    }
  },
  "nonrandom_margin_max_abs_diff": 1.000000000139778e-06,
  "random_mean_c1_new": 0.3311666583333333,
  "random_mean_c1_v1": 0.340166625,
  "rule_used": "propuesto_b1",
  "ok": true,
  "rule_text": "same 5,040 rows/keys; for the 5 non-random learners c1, c2 and per_concept_c1 identical and |margin diff| <= 1e-6 (6-decimal rounding); the random learner (Gumbel noise hashed from the embedding BYTES) is reported, not required"
}
```

Campos de decisión: `"rule_used": "propuesto_b1"`, `"ok": true`, `"propuesto_b1": true`, `"keys_identical": true`, `"n_new": 5040`, `"n_v1": 5040`, `"nonrandom_mismatches": {}`. El campo `"estricto"` (regla alternativa, no usada) vale `false`. El campo `"nonrandom_margin_max_abs_diff"` vale `1.000000000139778e-06`. El texto de `"rule_text"` exige `&#124;margin diff&#124; <= 1e-6`. El script dejó `"ok": true` con la regla `propuesto_b1`. Se deja constancia de esa cifra exacta, sin reinterpretarla.

### 4.2 Control 2 (re-codificación de CIFAR-100 con R0)

Regla de §3.5, transcrita en `"thresholds"` del JSON: **PASA** si max&#124;Δ&#124; ≤ 1e-4 y Spearman ≥ 0,99999; **PASA con desviación registrada** si 1e-4 < max&#124;Δ&#124; ≤ 2e-3, Spearman ≥ 0,9999 y el β_perc de D con R0 queda dentro de ±0,005 del de la v1; **FALLA** (GA0) en cualquier otro caso. El log escribe `control_c100: PASA` durante el paso 5. Fuente: `control_c100_recode.json`. Contenido completo:

```json
{
  "verdict": "PASA",
  "ok": true,
  "max_abs_diff": 6.394441637302961e-07,
  "spearman": 0.9999999997526534,
  "n_pairs": 4950,
  "beta_perc_R0_cpu": -0.2119,
  "beta_perc_v1": -0.2119,
  "abs_beta_diff": 0.0,
  "thresholds": {
    "pass_maxabs": 0.0001,
    "pass_spearman": 0.99999,
    "dev_maxabs": 0.002,
    "dev_spearman": 0.9999,
    "dev_beta": 0.005
  },
  "versions": {
    "torch": "2.14.1+cpu",
    "transformers": "5.20.2"
  },
  "note": "beta on the v1 scale (raw d, classical OLS, regression_table.csv PS_beta_perc)"
}
```

El veredicto mecánico del script es `"verdict": "PASA"`, `"ok": true`. `max_abs_diff` = `6.394441637302961e-07` (≤ 1e-4) y `spearman` = `0.9999999997526534` (≥ 0,99999); `n_pairs` = 4950; `beta_perc_R0_cpu` = `-0.2119`, `beta_perc_v1` = `-0.2119`, `abs_beta_diff` = `0.0`.

En `regions/agents_regions_report.md` (generado por `14_agent_regions.py`) la viñeta correspondiente dice: «Control de tubería de agentes (§3.5): PASA (T exacto: True; CIFAR-100 recodificado: PASA).» `regions/agents_summary.json` tiene `controls.ok` = true. No se disparó GA0.

### 4.3 Control positivo de agentes (puerta)

Regla de §3.5, copiada en cada `T_<encoder>.json` bajo `positive_control.rule`: «prototype (centroid), seed 42, native words: 95% bootstrap CI lower bound > 1/3 (lax gate, declared)». Un encoder entrenado que no la cumple queda «no interpretable» y sale de Holm; en F1 y F2 se reporta, pero no es puerta (`control_positivo_agentes` = «no es puerta» en `regions_agents.csv`).

| archivo | encoder | status | seed | mode | ci_lo | ci_hi | pass_ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `T_R0_dinov2s.json` | R0_dinov2s | ok | 42 | episodes | 0.9217665908333333 | 0.9608887350000002 | true |
| `T_O1_dinov2b.json` | O1_dinov2b | ok | 42 | episodes | 0.9464388800000001 | 0.9773389175 | true |
| `T_R1_cvcl.json` | R1_cvcl | ok | 42 | episodes | 0.6359944141666668 | 0.7100055841666666 | true |
| `T_E1_siglip.json` | E1_siglip | ok | 42 | episodes | 0.9462222333333336 | 0.9673332350000001 | true |
| `T_O2_openclip.json` | O2_openclip | ok | 42 | episodes | 0.9619999666666668 | 0.9775555666666667 | true |
| `T_E2_vitsup.json` | E2_vitsup | ok | 42 | episodes | 0.9702165791666667 | 0.9837776675000001 | true |
| `T_E3_rn50sup.json` | E3_rn50sup | ok | 42 | episodes | 0.8433333658333334 | 0.8935610883333334 | true |
| `T_E4_mae.json` | E4_mae | ok | 42 | episodes | 0.45200009750000003 | 0.513783355 | true |
| `T_F1_rn50rand0.json` | F1_rn50rand0 | ok | 42 | episodes | 0.3835556 | 0.4340000675000001 | true |
| `T_F2_pixels.json` | F2_pixels | ok | 42 | episodes | 0.42688896666666665 | 0.4851110024999999 | true |

En los diez encoders primarios, `positive_control.pass_` es `true`. En `regions/agents_regions_report.md`, la columna `control_positivo_agentes` dice «pasa» para R0, O1, R1, E1, O2, E2, E3 y E4, y «no es puerta» para F1_rn50rand0 y F2_pixels (no hubo fallo de la puerta en los entrenados). E4s_mae_cls aparece como «no evaluable» (no hay `T_` de E4s; ver §2, entrada 4, y §13).

---

## 5. Dispersión de d_perc en los 100 pares (§3.6)

Calculada en el paso 4, antes de cualquier exactitud. No cambia ninguna regla (§3.6). Fuente: `agent_pairs_dispersion.json`. El campo `_rule` del archivo dice:

> 100 v1 pairs (phase2/sampled_pairs.csv); d_sem = exp_008 d_sem; quadrants (quadrant_counts): d_perc median of the encoder over the 4,950 pairs and d_sem median over the 4,950 (v1 rule, REVISION5 b3); quadrant_counts_100: d_perc median over the 100 pairs (descriptive). Descriptive; changes no rule.

`quadrant_counts` usa la mediana de d_perc del encoder sobre los 4.950 pares (b3). `quadrant_counts_100` es descriptivo (mediana sobre los 100 pares).

| encoder | n | sd | min | max | range | spearman_dperc_dsem | Q1–Q4 (regla 4.950) | Q1–Q4 (descriptivo, 100) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0_dinov2s | 100 | 0.13613767209690872 | 0.1414257287979126 | 0.8853307366371155 | 0.7439050078392029 | 0.01295110698443623 | 25/25/25/25 | 25/25/25/25 |
| O1_dinov2b | 100 | 0.1162828286086192 | 0.15615785121917725 | 0.9425162076950073 | 0.7863583564758301 | -0.006388492272341284 | 21/26/29/24 | 22/22/28/28 |
| R1_cvcl | 100 | 0.14632199426941384 | 0.03745764493942261 | 0.6267770528793335 | 0.5893194079399109 | 0.11970617521961292 | 29/31/21/19 | 29/29/21/21 |
| E1_siglip | 100 | 0.042696184192094026 | 0.015215158462524414 | 0.20334410667419434 | 0.18812894821166992 | 0.08024042361613619 | 28/26/22/24 | 28/28/22/22 |
| O2_openclip | 100 | 0.0888672826840226 | 0.03589349985122681 | 0.4585176110267639 | 0.4226241111755371 | 0.12873652519856152 | 32/30/18/20 | 30/30/20/20 |
| E2_vitsup | 100 | 0.12268824662406681 | 0.18996769189834595 | 0.8461833000183105 | 0.6562156081199646 | 0.051191997287576854 | 24/27/26/23 | 24/24/26/26 |
| E3_rn50sup | 100 | 0.09043117073404477 | 0.0369870662689209 | 0.4884408116340637 | 0.4514537453651428 | 0.07574326129284333 | 23/27/27/23 | 26/26/24/24 |
| E4_mae | 100 | 0.18046432665954423 | 0.011018753051757812 | 0.8762620687484741 | 0.8652433156967163 | 0.19609188827292665 | 23/32/27/18 | 27/27/23/23 |
| F1_rn50rand0 | 100 | 0.00024350480795842487 | 2.0503997802734375e-05 | 0.001495063304901123 | 0.0014745593070983887 | 0.01894932482284689 | 24/27/26/23 | 27/27/23/23 |
| F2_pixels | 100 | 0.4357685845211876 | 0.06994873285293579 | 1.8120839595794678 | 1.742135226726532 | 0.20052300415355434 | 22/32/28/18 | 28/28/22/22 |
| E4s_mae_cls | 100 | 0.013674846146170093 | 0.0016238093376159668 | 0.07278639078140259 | 0.07116258144378662 | 0.16452168867896946 | 25/31/25/19 | 30/30/20/20 |
| F1_rn50rand1 | — | — | — | — | — | — | tecnico: falta /workspace/lexical-consensus-openmind/encoder_equivalence/agents/out/emb/c100train_F1_rn50rand1.npz | — |
| F1_rn50rand2 | — | — | — | — | — | — | tecnico: falta /workspace/lexical-consensus-openmind/encoder_equivalence/agents/out/emb/c100train_F1_rn50rand2.npz | — |

F1_rn50rand1 y F1_rn50rand2 no tienen embeddings de c100train en esta corrida (sensibilidades no ejecutadas); el JSON registra el estado técnico citado en la tabla. E4s_mae_cls sí tiene n = 100 (los embeddings salen del paso de E4 del primario).

---

## 6. T (descriptivo)

T no entra en ninguna familia Holm (§3.1). Cada archivo `T_<encoder>.json` tiene `"status": "ok"`, `"seed": 42`, `"mode": "episodes"` y `"pool": "c10pool"`. El aprendiz de la puerta y de la tabla de regiones es el prototipo (`centroid`). Abajo, C1 por nivel de ese aprendiz, el orden de niveles, y la Spearman descriptiva `rho_c1_vs_dconcept` (el JSON la anota: «descriptive, n = 10, no p»).

| archivo | encoder | seconds | C1 native | C1 near | C1 mid | C1 far | orden centroid | orden random | rho (n=10) | rho ci_lo | rho ci_hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `T_R0_dinov2s.json` | R0_dinov2s | 38.2 | 0.9428888333333334 | 0.8466666999999999 | 0.6542222 | 0.5295555666666667 | native > near > mid > far | native > mid > near > far | -0.7696969696969697 | -0.9506172839506172 | -0.3071895424836601 |
| `T_O1_dinov2b.json` | O1_dinov2b | 44.8 | 0.9633333333333334 | 0.8344444 | 0.6691111000000001 | 0.5444444666666667 | native > near > mid > far | far > mid > native > near | -0.9757575757575757 | -1.0 | -0.8101265822784811 |
| `T_R1_cvcl.json` | R1_cvcl | 41.7 | 0.6753333000000001 | 0.6468889333333334 | 0.5106666333333333 | 0.42755546666666666 | native > near > mid > far | mid > near > far > native | -0.8424242424242423 | -0.9740259740259739 | -0.4102564102564103 |
| `T_E1_siglip.json` | E1_siglip | 44.3 | 0.9575555333333335 | 0.8960000333333334 | 0.6964444000000001 | 0.5342222333333333 | native > near > mid > far | near > mid > native > far | -0.8303030303030302 | -1.0 | -0.3684210526315789 |
| `T_O2_openclip.json` | O2_openclip | 38.1 | 0.9702222333333335 | 0.9144444333333333 | 0.6708889 | 0.5240000000000001 | native > near > mid > far | mid > near > native > far | -0.8181818181818182 | -0.9874999999999999 | -0.3670886075949367 |
| `T_E2_vitsup.json` | E2_vitsup | 42.1 | 0.9775555 | 0.8993333333333334 | 0.7202221999999999 | 0.518 | native > near > mid > far | near > native > far > mid | -0.9757575757575757 | -1.0 | -0.8112909004060191 |
| `T_E3_rn50sup.json` | E3_rn50sup | 63.1 | 0.8704444333333332 | 0.8575556333333332 | 0.6442222333333334 | 0.4946666 | native > near > mid > far | mid > native > far > near | -0.8666666666666665 | -1.0 | -0.46835443037974683 |
| `T_E4_mae.json` | E4_mae | 46.7 | 0.4826667 | 0.42377783333333335 | 0.4817777666666666 | 0.3882221000000002 | native > mid > near > far | near > native > far > mid | -0.41818181818181815 | -0.9622641509433963 | 0.32919254658385094 |
| `T_F1_rn50rand0.json` | F1_rn50rand0 | 69.7 | 0.4102222666666666 | 0.43111113333333334 | 0.42822206666666673 | 0.3844444333333333 | near > mid > native > far | mid > near > far > native | -0.4666666666666666 | -0.8875 | 0.2739132165605092 |
| `T_F2_pixels.json` | F2_pixels | 89.5 | 0.45777776666666664 | 0.44333330000000004 | 0.46022219999999997 | 0.3868888333333334 | mid > native > near > far | far > near > native > mid | -0.29696969696969694 | -0.9245283018867925 | 0.6153973509933773 |

C1 de todos los aprendices, leída de `c1_by_tier` del mismo archivo:

| encoder | aprendiz | native | near | mid | far |
| --- | --- | --- | --- | --- | --- |
| R0_dinov2s | centroid | 0.9428888333333334 | 0.8466666999999999 | 0.6542222 | 0.5295555666666667 |
| R0_dinov2s | exemplar_knn_k3 | 0.9315555666666668 | 0.8331111 | 0.6502221333333336 | 0.5291111333333334 |
| R0_dinov2s | kcentroid_k2 | 0.9368888666666667 | 0.8357778 | 0.6531111333333333 | 0.5253333666666666 |
| R0_dinov2s | logreg | 0.9491111666666667 | 0.8428888333333333 | 0.6411111 | 0.5348889 |
| R0_dinov2s | linsvm | 0.9497779333333334 | 0.8473333666666666 | 0.6444445333333333 | 0.5328889000000001 |
| R0_dinov2s | random | 0.33688893333333336 | 0.3313332999999999 | 0.3322222 | 0.32422219999999996 |
| O1_dinov2b | centroid | 0.9633333333333334 | 0.8344444 | 0.6691111000000001 | 0.5444444666666667 |
| O1_dinov2b | exemplar_knn_k3 | 0.9577778000000002 | 0.8284444 | 0.6731111 | 0.5417777666666667 |
| O1_dinov2b | kcentroid_k2 | 0.9613332666666669 | 0.8342221666666665 | 0.6724444666666667 | 0.5480000333333334 |
| O1_dinov2b | logreg | 0.9711111000000001 | 0.8055555666666667 | 0.6442222333333334 | 0.5653333333333332 |
| O1_dinov2b | linsvm | 0.9722221333333335 | 0.8213333666666666 | 0.6397778 | 0.5593333666666667 |
| O1_dinov2b | random | 0.3335555333333333 | 0.3306667 | 0.33422226666666666 | 0.34666670000000005 |
| R1_cvcl | centroid | 0.6753333000000001 | 0.6468889333333334 | 0.5106666333333333 | 0.42755546666666666 |
| R1_cvcl | exemplar_knn_k3 | 0.6257777333333333 | 0.6100000666666666 | 0.48577780000000004 | 0.40933326666666664 |
| R1_cvcl | kcentroid_k2 | 0.6566667 | 0.6288888666666665 | 0.508 | 0.42733326666666666 |
| R1_cvcl | logreg | 0.6782222000000001 | 0.6526666333333335 | 0.5286666666666667 | 0.4348888333333333 |
| R1_cvcl | linsvm | 0.6713333 | 0.6531111666666668 | 0.5242222333333334 | 0.4335556 |
| R1_cvcl | random | 0.32333333333333336 | 0.33377776666666675 | 0.3380000666666667 | 0.3331111666666667 |
| E1_siglip | centroid | 0.9575555333333335 | 0.8960000333333334 | 0.6964444000000001 | 0.5342222333333333 |
| E1_siglip | exemplar_knn_k3 | 0.9288889000000001 | 0.8511111333333333 | 0.6748889333333334 | 0.5251111 |
| E1_siglip | kcentroid_k2 | 0.9479999666666666 | 0.8806666666666666 | 0.6915556333333335 | 0.5373333333333333 |
| E1_siglip | logreg | 0.9620000333333335 | 0.9128889333333333 | 0.6999999999999998 | 0.5322222333333334 |
| E1_siglip | linsvm | 0.9628888333333335 | 0.9148889 | 0.6995555666666664 | 0.5322223 |
| E1_siglip | random | 0.33444436666666666 | 0.3362223333333333 | 0.33599983333333333 | 0.33399996666666676 |
| O2_openclip | centroid | 0.9702222333333335 | 0.9144444333333333 | 0.6708889 | 0.5240000000000001 |
| O2_openclip | exemplar_knn_k3 | 0.9555556333333334 | 0.8902222333333333 | 0.6684443999999999 | 0.5255555000000001 |
| O2_openclip | kcentroid_k2 | 0.9637777000000001 | 0.8995555666666666 | 0.6791111666666666 | 0.5284443666666667 |
| O2_openclip | logreg | 0.9720000333333335 | 0.9253332333333335 | 0.6762222666666667 | 0.5168888333333334 |
| O2_openclip | linsvm | 0.9731111000000001 | 0.9233333666666669 | 0.6808889333333333 | 0.5186667 |
| O2_openclip | random | 0.3299999666666667 | 0.3306668000000001 | 0.33266656666666666 | 0.3208889666666667 |
| E2_vitsup | centroid | 0.9775555 | 0.8993333333333334 | 0.7202221999999999 | 0.518 |
| E2_vitsup | exemplar_knn_k3 | 0.9715556000000002 | 0.8837778000000001 | 0.7179999666666668 | 0.5206666000000001 |
| E2_vitsup | kcentroid_k2 | 0.9724444333333335 | 0.8944445000000001 | 0.7213333666666666 | 0.5197777666666666 |
| E2_vitsup | logreg | 0.9771110333333335 | 0.9151111 | 0.6995553999999998 | 0.5206666666666667 |
| E2_vitsup | linsvm | 0.9788888333333335 | 0.9120000333333333 | 0.7004443666666667 | 0.5193333666666667 |
| E2_vitsup | random | 0.33999996666666654 | 0.3471111000000001 | 0.3275556333333333 | 0.33888883333333336 |
| E3_rn50sup | centroid | 0.8704444333333332 | 0.8575556333333332 | 0.6442222333333334 | 0.4946666 |
| E3_rn50sup | exemplar_knn_k3 | 0.8400000333333332 | 0.8284444000000001 | 0.6397777000000001 | 0.47888890000000006 |
| E3_rn50sup | kcentroid_k2 | 0.8511111 | 0.8413333999999999 | 0.6484444 | 0.4833333333333333 |
| E3_rn50sup | logreg | 0.8713333666666667 | 0.8637778999999999 | 0.6386666999999999 | 0.4871109999999999 |
| E3_rn50sup | linsvm | 0.8764444333333333 | 0.8642221666666664 | 0.6391111000000002 | 0.48533329999999997 |
| E3_rn50sup | random | 0.3311111333333333 | 0.3255555666666667 | 0.34177769999999996 | 0.33022219999999997 |
| E4_mae | centroid | 0.4826667 | 0.42377783333333335 | 0.4817777666666666 | 0.3882221000000002 |
| E4_mae | exemplar_knn_k3 | 0.4826666333333334 | 0.44466673333333334 | 0.47599993333333335 | 0.3804444333333333 |
| E4_mae | kcentroid_k2 | 0.48688890000000007 | 0.4453333 | 0.4691111333333333 | 0.3757778333333333 |
| E4_mae | logreg | 0.5166667000000001 | 0.4535554666666667 | 0.5004444333333333 | 0.39155553333333326 |
| E4_mae | linsvm | 0.5775555000000002 | 0.5162223 | 0.5166667 | 0.39844443333333324 |
| E4_mae | random | 0.33600009999999997 | 0.33866673333333336 | 0.32733340000000005 | 0.33133333333333337 |
| F1_rn50rand0 | centroid | 0.4102222666666666 | 0.43111113333333334 | 0.42822206666666673 | 0.3844444333333333 |
| F1_rn50rand0 | exemplar_knn_k3 | 0.42733340000000003 | 0.4293333666666667 | 0.4144444333333334 | 0.3724445333333333 |
| F1_rn50rand0 | kcentroid_k2 | 0.41999993333333335 | 0.41888890000000006 | 0.4160000333333333 | 0.36666669999999996 |
| F1_rn50rand0 | logreg | 0.40888890000000006 | 0.43177780000000004 | 0.4297777666666668 | 0.38177773333333326 |
| F1_rn50rand0 | linsvm | 0.4068889333333334 | 0.43355553333333335 | 0.4297777666666666 | 0.38333326666666667 |
| F1_rn50rand0 | random | 0.32666670000000003 | 0.3382222333333334 | 0.3493333 | 0.3306666666666666 |
| F2_pixels | centroid | 0.45777776666666664 | 0.44333330000000004 | 0.46022219999999997 | 0.3868888333333334 |
| F2_pixels | exemplar_knn_k3 | 0.45622233333333345 | 0.4526667333333333 | 0.4495555333333333 | 0.39244439999999997 |
| F2_pixels | kcentroid_k2 | 0.4506666333333333 | 0.4531111 | 0.4417776333333333 | 0.3866667000000001 |
| F2_pixels | logreg | 0.47311110000000006 | 0.4484444666666666 | 0.4682220999999999 | 0.3893332666666666 |
| F2_pixels | linsvm | 0.46644443333333335 | 0.45199996666666664 | 0.46688889999999994 | 0.38511110000000015 |
| F2_pixels | random | 0.3391110666666667 | 0.34022220000000003 | 0.33111113333333336 | 0.3564444333333333 |

---

## 7. D (100 pares)

Familia D (encoders nuevos entrenados) o D-pisos (F1, F2), según §3.1. La p de β_perc que entra a la región es la del OLS clásico de la v1 (`D_P="v1"`, ACLARACIONES (b6)); HC3 se guarda y, en las regiones, se informa si cambia la región (`region_hc3_*`, `hc3_cambia_region`). Fuente de cada fila: el objeto `test` de `D_<encoder>.json` (n = 100, `"status": "ok"` en los diez). `rule_centroid` es la letra de la regla de cuadrantes de la v1 (mediana de d_perc sobre 4.950 pares, b3). `rule_centroid_q100_descriptiva` es la versión descriptiva con la mediana de los 100 pares.

Regla de región, §3.2 (sin ajuste), aplicada después por `14_agent_regions.py` (el resultado mecánico está en el §11):
- **Apoyo:** β < 0 con p < 0,05, más la parte de β_sem de la regla de la v1 (p ≥ 0,10, o R² parcial menor que un tercio del de la distancia).
- **No apoyo:** el IC 95 % del r parcial excluye todo valor ≤ −SESOI, o hay un efecto opuesto con p < 0,05.
- **Inconcluso:** todo lo demás.

| archivo | encoder | rule_centroid | rule_q100_desc | n | beta_v1 | p_v1 | beta_sem_v1 | p_sem_v1 | r2p_perc_v1 | r2p_sem_v1 | vif | r_partial | r_ci_lo | r_ci_hi | p_hc3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `D_R0_dinov2s.json` | R0_dinov2s | A | A | 100 | -0.2119 | 1.8747011676629847e-07 | -0.01289 | 0.6603177348599459 | 0.2453 | 0.002 | 1.011 | -0.49524649173319324 | -0.6309721665561533 | -0.33010528314192855 | 0.00011854348217894996 |
| `D_O1_dinov2b.json` | O1_dinov2b | A | A | 100 | -0.23619 | 5.704465103265075e-10 | -0.0236 | 0.3007371043583383 | 0.3284 | 0.011 | 1.017 | -0.5730436430821244 | -0.6921569375845803 | -0.42354265499889826 | 0.03110815828441553 |
| `D_R1_cvcl.json` | R1_cvcl | C | C | 100 | 0.00875 | 0.8858778017075768 | -0.0069 | 0.8919151263159819 | 0.0002 | 0.0002 | 1.024 | 0.014609294104954597 | -0.18333482085939476 | 0.21141491174322544 | 0.8860824394729544 |
| `D_E1_siglip.json` | E1_siglip | B | B | 100 | -0.50767 | 0.002787822444480468 | -0.03052 | 0.44932349187947834 | 0.0885 | 0.0059 | 1.013 | -0.29747132285121547 | -0.46743629488113936 | -0.1062983793716469 | 0.00580200902885568 |
| `D_O2_openclip.json` | O2_openclip | B | B | 100 | -0.27331 | 0.00046804461442965866 | -0.04325 | 0.25973468602839583 | 0.1191 | 0.0131 | 1.019 | -0.34511854983315354 | -0.5079279154044917 | -0.1585022165070656 | 0.0003631758608826767 |
| `D_E2_vitsup.json` | E2_vitsup | A | A | 100 | -0.28874 | 3.442157667913604e-13 | -0.0405 | 0.09389788726335402 | 0.4222 | 0.0287 | 1.033 | -0.6497325812782205 | -0.7508398252553751 | -0.5188713988697167 | 2.255210874667606e-07 |
| `D_E3_rn50sup.json` | E3_rn50sup | B | B | 100 | -0.25937 | 0.0026504197383396557 | -0.02872 | 0.5081349346642084 | 0.0894 | 0.0045 | 1.02 | -0.2989332507381342 | -0.4686893684688827 | -0.10788459464923959 | 0.0011627543156061903 |
| `D_E4_mae.json` | E4_mae | C | C | 100 | -0.00155 | 0.9672085222534106 | -0.04292 | 0.2689419289069741 | 0.0 | 0.0126 | 1.06 | -0.004184800560452262 | -0.20143374648131895 | 0.193390326862451 | 0.9707954344711588 |
| `D_F1_rn50rand0.json` | F1_rn50rand0 | C | C | 100 | 49.18216 | 0.07025538355391509 | -0.0167 | 0.6545917230691161 | 0.0334 | 0.0021 | 1.001 | 0.18272367877707038 | -0.015241478022490033 | 0.3669039110561922 | 0.05545640000739933 |
| `D_F2_pixels.json` | F2_pixels | B | B | 100 | -0.02269 | 0.17323435614128965 | -0.00512 | 0.9008730805237729 | 0.019 | 0.0002 | 1.053 | -0.13797121523356695 | -0.32649365865525154 | 0.06110859775329919 | 0.1818201903140355 |

Los objetos `learners` del mismo JSON (centroid, exemplar_knn_k3, logreg, random) tienen todos `model_ps.n` = 100 y `model_ps.status` = `"ok"`. En `regions/regions_agents.csv`, `hc3_cambia_region` es `False` en las filas de D de los diez encoders primarios (no hay ningún `True` en el CSV).

La región por SESOI, ya aplicada por el script, se transcribe en el §11 (fuente: `regions/agents_regions_report.md` y `regions/regions_agents.csv`). En D, F1_rn50rand0 queda en «No apoyo» con los tres SESOI; E4_mae y R1_cvcl son «dependiente del margen»; F2_pixels queda en «Inconcluso» con los tres SESOI; los demás primarios, en «Apoyo» con los tres.

---

## 8. D-todos (4.950 pares)

Obligatoria (§3.6). No entra en ninguna familia ni en GA (ACLARACIONES (b14): los p son anticonservadores porque los 4.950 pares comparten clases; el uso declarado es el rótulo descriptivo «sensible al muestreo de pares»). Fuente: objeto `test` de `Dall_<encoder>.json`. En los diez, `"status": "ok"` y `n` = `n_pairs` = 4950.

| archivo | encoder | n | beta_v1 | p_v1 | beta_sem_v1 | p_sem_v1 | r2p_perc_v1 | r2p_sem_v1 | vif | r_partial | r_ci_lo | r_ci_hi | p_hc3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Dall_R0_dinov2s.json` | R0_dinov2s | 4950 | -0.23791 | 0.0 | -0.02041 | 1.405161798039775e-09 | 0.3569 | 0.0074 | 1.025 | -0.5974519397517667 | -0.6150754420775546 | -0.5792317703665314 | 0.0 |
| `Dall_O1_dinov2b.json` | O1_dinov2b | 4950 | -0.30963 | 0.0 | -0.02486 | 8.165324146074029e-20 | 0.4801 | 0.0166 | 1.024 | -0.6929203532581025 | -0.7071305283623966 | -0.6781506834619316 | 0.0 |
| `Dall_R1_cvcl.json` | R1_cvcl | 4950 | -0.0548 | 2.595314398539079e-10 | -0.025 | 0.00038935518837078775 | 0.008 | 0.0025 | 1.044 | -0.08969627131935931 | -0.11726548455092745 | -0.06198891407228948 | 4.712898312223613e-11 |
| `Dall_E1_siglip.json` | E1_siglip | 4950 | -0.74958 | 4.268038542086065e-279 | -0.02504 | 2.7251551569430314e-08 | 0.2271 | 0.0062 | 1.031 | -0.4765645202085375 | -0.49781670061473143 | -0.4547403647903061 | 1.7731357979605219e-301 |
| `Dall_O2_openclip.json` | O2_openclip | 4950 | -0.37276 | 0.0 | -0.02731 | 1.5126359520052682e-10 | 0.27 | 0.0083 | 1.041 | -0.5196431158216447 | -0.5396915011125988 | -0.4990056630613903 | 0.0 |
| `Dall_E2_vitsup.json` | E2_vitsup | 4950 | -0.3002 | 0.0 | -0.03176 | 4.970000206941465e-22 | 0.4563 | 0.0186 | 1.027 | -0.6754874733840472 | -0.6903568075317059 | -0.6600477031405929 | 0.0 |
| `Dall_E3_rn50sup.json` | E3_rn50sup | 4950 | -0.32285 | 4.351767839271942e-183 | -0.03209 | 4.233313807778927e-10 | 0.1549 | 0.0079 | 1.059 | -0.3936010935181102 | -0.41689147843795576 | -0.3697942106441827 | 3.088386184490651e-200 |
| `Dall_E4_mae.json` | E4_mae | 4950 | -0.02495 | 2.8438994618562627e-06 | -0.00915 | 0.0953902829368496 | 0.0044 | 0.0006 | 1.019 | -0.0664938317475135 | -0.09418162069420315 | -0.03870325973263522 | 2.443900983447953e-07 |
| `Dall_F1_rn50rand0.json` | F1_rn50rand0 | 4950 | 13.33424 | 0.008128643495286398 | -0.0085 | 0.14887135647291816 | 0.0014 | 0.0004 | 1.017 | 0.037618246676381584 | 0.009766206565944482 | 0.06541196280115716 | 0.004348737290802778 |
| `Dall_F2_pixels.json` | F2_pixels | 4950 | -0.03055 | 8.509237839158358e-38 | -0.00132 | 0.8197249785802388 | 0.0328 | 0.0 | 1.016 | -0.1812038225623627 | -0.20801587855550782 | -0.1541196579725958 | 6.77661712844562e-38 |

En varios archivos, `p_v1` y/o `p_hc3` aparecen como `0.0` (así está escrito en el JSON). El rótulo D (100) frente a D-todos, por SESOI, está en el §11 (tabla «D con 100 pares vs D-todos» de `agents_regions_report.md` y en `regions/sampling_sensitivity.csv`). Quedan con el rótulo «sensible al muestreo de pares»: R1_cvcl, E4_mae y F2_pixels. Los demás primarios, «estable». F1_rn50rand0 queda en «No apoyo» también en D-todos, con rótulo «estable».

---

## 9. H1, H2 y H3

### 9.1 H1 (ρ, IC bootstrap, Mantel, p unilateral frente al SESOI)

Familia C (o C-pisos). Regla de región, §3.2, con la precedencia literal de (b4): **Apoyo** si ρ > 0 y p de Mantel < 0,05 (aunque el IC quede entero bajo el SESOI; en ese caso se añade el rótulo descriptivo «Apoyo con IC bajo SESOI», que no cambia la región ni GA). **No apoyo** si el IC 95 % de ρ queda entero por debajo del SESOI. **Inconcluso:** todo lo demás. El p que entra al Holm de GA1 es el bootstrap unilateral de H0 «ρ ≥ SESOI», campo `p_below_sesoi` (ACLARACIONES (b5); 10.000 réplicas, semilla 42). Agregación de A (b8, `A_AGG="todas"`): un encoder está en Apoyo en A solo si lo está en las tres fuentes; GA1 se dispara si cualquier fuente queda en No apoyo con Holm p < 0,05.

Fuente: objeto `H1` de `H_<encoder>.json`. En los diez encoders, las tres fuentes tienen `"status": "ok"`. El n es el mismo para todos: THINGS 3400, HSJ 1390, CIFAR10H 45.

| archivo | encoder | fuente | n_pairs | rho | ci_lo | ci_hi | mantel_r | mantel_p | p_below_0.10 | p_below_0.20 | p_below_0.30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `H_R0_dinov2s.json` | R0_dinov2s | THINGS | 3400 | 0.5228020458728204 | 0.49576864056473 | 0.5489229431035256 | 0.5228020458728204 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_R0_dinov2s.json` | R0_dinov2s | HSJ | 1390 | 0.5006153155638908 | 0.4560200001699834 | 0.5427432741834424 | 0.5006153155638908 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_R0_dinov2s.json` | R0_dinov2s | CIFAR10H | 45 | 0.7948616600790513 | 0.6055972120510934 | 0.8967133496292287 | 0.7948616600790513 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_O1_dinov2b.json` | O1_dinov2b | THINGS | 3400 | 0.5206559160580527 | 0.49408651151818805 | 0.54679981870208 | 0.5206559160580527 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_O1_dinov2b.json` | O1_dinov2b | HSJ | 1390 | 0.5474917803590156 | 0.5062540095640857 | 0.5867683068292364 | 0.5474917803590156 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_O1_dinov2b.json` | O1_dinov2b | CIFAR10H | 45 | 0.6770750988142292 | 0.4400788745915541 | 0.8295248988829702 | 0.6770750988142292 | 9.999000099990002e-05 | 1.0 | 1.0 | 0.998000199980002 |
| `H_R1_cvcl.json` | R1_cvcl | THINGS | 3400 | 0.6825406514839423 | 0.6633655896400723 | 0.7004743762010189 | 0.6825406514839423 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_R1_cvcl.json` | R1_cvcl | HSJ | 1390 | 0.42936800878161613 | 0.3842239829191312 | 0.47146388403747863 | 0.42936800878161613 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_R1_cvcl.json` | R1_cvcl | CIFAR10H | 45 | 0.7861660079051382 | 0.5932970863567458 | 0.8916605005442593 | 0.7861660079051382 | 0.00019998000199980003 | 1.0 | 1.0 | 1.0 |
| `H_E1_siglip.json` | E1_siglip | THINGS | 3400 | 0.6391537674591696 | 0.6152691754150692 | 0.6617687411605735 | 0.6391537674591696 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E1_siglip.json` | E1_siglip | HSJ | 1390 | 0.5007943114147673 | 0.45663941080896986 | 0.5416608125656074 | 0.5007943114147673 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E1_siglip.json` | E1_siglip | CIFAR10H | 45 | 0.8895915678524374 | 0.7922318042460902 | 0.9369298198601859 | 0.8895915678524374 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_O2_openclip.json` | O2_openclip | THINGS | 3400 | 0.7017855040707185 | 0.6805452357664737 | 0.722267809989302 | 0.7017855040707185 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_O2_openclip.json` | O2_openclip | HSJ | 1390 | 0.5186595009462668 | 0.4755714870692667 | 0.5588474093841646 | 0.5186595009462668 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_O2_openclip.json` | O2_openclip | CIFAR10H | 45 | 0.8669301712779974 | 0.7581874464276136 | 0.9184842290151469 | 0.8669301712779974 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E2_vitsup.json` | E2_vitsup | THINGS | 3400 | 0.3933690332859021 | 0.36365952203644064 | 0.42297188412743036 | 0.3933690332859021 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E2_vitsup.json` | E2_vitsup | HSJ | 1390 | 0.5559978395825628 | 0.5168446476444081 | 0.5942418626623673 | 0.5559978395825628 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E2_vitsup.json` | E2_vitsup | CIFAR10H | 45 | 0.6611330698287219 | 0.4346470147031559 | 0.811161165005615 | 0.6611330698287219 | 0.00029997000299970003 | 0.9999000099990001 | 0.9996000399960004 | 0.9975002499750025 |
| `H_E3_rn50sup.json` | E3_rn50sup | THINGS | 3400 | 0.6180636438648961 | 0.5935837073927017 | 0.6415888285512586 | 0.6180636438648961 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E3_rn50sup.json` | E3_rn50sup | HSJ | 1390 | 0.5327702262370633 | 0.4912278827822029 | 0.5721359942624432 | 0.5327702262370633 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E3_rn50sup.json` | E3_rn50sup | CIFAR10H | 45 | 0.9119894598155466 | 0.8198132789587654 | 0.950122302573434 | 0.9119894598155466 | 9.999000099990002e-05 | 1.0 | 1.0 | 1.0 |
| `H_E4_mae.json` | E4_mae | THINGS | 3400 | 0.29181950168135423 | 0.2594545180774242 | 0.3239452169766881 | 0.29181950168135423 | 9.999000099990002e-05 | 1.0 | 1.0 | 0.30716928307169283 |
| `H_E4_mae.json` | E4_mae | HSJ | 1390 | 0.22728141284601294 | 0.1755426328595233 | 0.27684326550959537 | 0.22728141284601294 | 9.999000099990002e-05 | 1.0 | 0.8543145685431457 | 0.0025997400259974 |
| `H_E4_mae.json` | E4_mae | CIFAR10H | 45 | 0.44492753623188397 | 0.17394370554647498 | 0.6434519760218551 | 0.44492753623188397 | 0.014798520147985201 | 0.9925007499250075 | 0.962903709629037 | 0.8645135486451355 |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | THINGS | 3400 | 0.10878951313072208 | 0.075169059591897 | 0.14232888629162907 | 0.10878951313072208 | 0.037896210378962104 | 0.6968303169683032 | 9.999000099990002e-05 | 9.999000099990002e-05 |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | HSJ | 1390 | 0.1062609260219985 | 0.05234699478307431 | 0.15829234861203875 | 0.1062609260219985 | 0.013298670132986702 | 0.5913408659134086 | 0.00029997000299970003 | 9.999000099990002e-05 |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | CIFAR10H | 45 | 0.5301712779973649 | 0.24762253034479467 | 0.7126868596165644 | 0.5301712779973649 | 0.005599440055994401 | 0.9982001799820018 | 0.9877012298770123 | 0.9503049695030497 |
| `H_F2_pixels.json` | F2_pixels | THINGS | 3400 | 0.15049059973893283 | 0.11714367639518468 | 0.18480660172328392 | 0.15049059973893283 | 0.00019998000199980003 | 0.9983001699830018 | 0.0022997700229977 | 9.999000099990002e-05 |
| `H_F2_pixels.json` | F2_pixels | HSJ | 1390 | 0.19881049876695345 | 0.14661950555666792 | 0.250248663114388 | 0.19881049876695345 | 9.999000099990002e-05 | 1.0 | 0.48305169483051696 | 9.999000099990002e-05 |
| `H_F2_pixels.json` | F2_pixels | CIFAR10H | 45 | 0.29407114624505926 | 0.007379011041458002 | 0.53016736571913 | 0.29407114624505926 | 0.0504949505049495 | 0.9147085291470853 | 0.7553244675532447 | 0.47855214478552144 |

Regiones de H1 (A) según `agents_regions_report.md`: los diez primarios quedan en «Apoyo» con los tres SESOI en THINGS, en HSJ y en CIFAR10H, salvo F2_pixels en CIFAR10H, que queda en «Inconcluso» con los tres SESOI (`mantel_p` = `0.0504949505049495`, que no es < 0,05). El rótulo «Apoyo con IC bajo SESOI» aparece, en `regions_agents.csv`, en: E4_mae A/HSJ (solo SESOI 0,30); F1_rn50rand0 A/THINGS y A/HSJ (SESOI 0,20 y 0,30); F2_pixels A/THINGS (SESOI 0,20 y 0,30) y A/HSJ (SESOI 0,30). No cambia la región (sigue «Apoyo»).

El objeto `H1_desc_c10` del mismo JSON (ρ sobre 45 pares THINGS y 35 pares HSJ, descriptivo) no entra en las regiones; los valores están en esos archivos.

### 9.2 H2 (y H2b, descriptivo)

H2 entra en la familia C (o C-pisos). H2b es descriptivo y no entra en ninguna familia (§3.1). Expectativa declarada de antemano (§3.2, REVISION4 (3)): en H2 con HSJ (n = 32), «No apoyo» es casi inalcanzable incluso con SESOI = 0,30; que H2 salga inconclusa no se leerá como ausencia de efecto. GA0b en P si R0 no está en «Apoyo» en ese análisis (§3.4).

Fuente de H2: objeto `H2` de `H_<encoder>.json` (campos de primer nivel; `"status": "ok"`). n = 61 en THINGS y n = 32 en HSJ, en los diez encoders.

| archivo | encoder | fuente | n | beta | p | ci_lo | ci_hi | r_partial | r_ci_lo | r_ci_hi | r2p | p_sem | r2p_sem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `H_R0_dinov2s.json` | R0_dinov2s | THINGS | 61 | -0.007850483005505593 | 0.374153035551128 | -0.02516374975734055 | 0.009462783746329367 | -0.09953239415559487 | -0.34474832879568956 | 0.15840020599068005 | 0.00990669748634473 | 0.21087745783630463 | 0.020808517217949716 |
| `H_R0_dinov2s.json` | R0_dinov2s | HSJ | 32 | -0.03444697142116745 | 0.039344154705676065 | -0.06721210366857976 | -0.0016818391737551397 | -0.4132043453854259 | -0.6695209436333572 | -0.0689543922682849 | 0.17073783104539791 | 0.36170044159049775 | 0.020141736961238905 |
| `H_O1_dinov2b.json` | O1_dinov2b | THINGS | 61 | -0.010826421181033259 | 0.1639598321037813 | -0.026071593617635336 | 0.00441875125556882 | -0.17101727764291308 | -0.40726063085969816 | 0.08667589892344625 | 0.029246909252393507 | 0.5921702591734965 | 0.004678795101445291 |
| `H_O1_dinov2b.json` | O1_dinov2b | HSJ | 32 | -0.02728494436613215 | 0.051989891212661746 | -0.05480502886536598 | 0.00023514013310167378 | -0.4010944659671417 | -0.661433478686849 | -0.054493326437430434 | 0.16087677062946523 | 0.8580784187049088 | 0.0011126685056370292 |
| `H_R1_cvcl.json` | R1_cvcl | THINGS | 61 | 0.011364745153272526 | 0.5612717650655208 | -0.02697651550845036 | 0.049706005814995416 | 0.08337244724097778 | -0.17424561895800755 | 0.3303087463923901 | 0.006950964958949623 | 0.37259269431481123 | 0.01959113875652422 |
| `H_R1_cvcl.json` | R1_cvcl | HSJ | 32 | -0.044902234867516475 | 0.052275515242335986 | -0.09024649370437784 | 0.0004420239693448888 | -0.38570339636042456 | -0.6510665814589364 | -0.03632147033131178 | 0.14876710996396647 | 0.05411118926968829 | 0.05657853726017746 |
| `H_E1_siglip.json` | E1_siglip | THINGS | 61 | -0.01659387072119351 | 0.14397816769189573 | -0.03885284774518061 | 0.005665106302793588 | -0.16229564318691292 | -0.3997504536970672 | 0.09557220522031276 | 0.026339875797453786 | 0.5336144771039804 | 0.0062852402587922 |
| `H_E1_siglip.json` | E1_siglip | HSJ | 32 | -0.050827817205410436 | 0.01104138080431321 | -0.0900271651030059 | -0.011628469307814963 | -0.5029561891181655 | -0.7276246037704158 | -0.18083941805045123 | 0.25296492817226746 | 0.4458506869837263 | 0.019095602237487172 |
| `H_O2_openclip.json` | O2_openclip | THINGS | 61 | -0.01994885688154482 | 0.08831877899872356 | -0.04288981207688358 | 0.002992098313793941 | -0.19849501197127295 | -0.4307145804232338 | 0.05837654550405268 | 0.03940026977747568 | 0.5398416584327945 | 0.005804406865488161 |
| `H_O2_openclip.json` | O2_openclip | HSJ | 32 | -0.04302032582411904 | 0.022399944668872015 | -0.07994507688427076 | -0.006095574763967326 | -0.45434162520381194 | -0.6965475948505319 | -0.11918338354447122 | 0.20642631239284048 | 0.7730754853910484 | 0.002528127373207341 |
| `H_E2_vitsup.json` | E2_vitsup | THINGS | 61 | -0.0061192690347737135 | 0.4758800616728899 | -0.0229418209268442 | 0.010703282857296775 | -0.08408304705197751 | -0.3309461356412521 | 0.17355164320589572 | 0.007069958801545165 | 0.0322790590194068 | 0.07308245006009087 |
| `H_E2_vitsup.json` | E2_vitsup | HSJ | 32 | -0.031053495058248184 | 0.0692040478359093 | -0.06454874744003677 | 0.0024417573235404055 | -0.3852764791873573 | -0.650777606517337 | -0.03582069166873918 | 0.14843796541500573 | 0.7171675101889812 | 0.002582527688149046 |
| `H_E3_rn50sup.json` | E3_rn50sup | THINGS | 61 | -0.004229190517141182 | 0.7488290366506735 | -0.030117640356246596 | 0.021659259321964233 | -0.03833807030660297 | -0.28944940094227245 | 0.21771054055738934 | 0.0014698076348340539 | 0.1831251316192547 | 0.03298077568444878 |
| `H_E3_rn50sup.json` | E3_rn50sup | HSJ | 32 | -0.04409022030970813 | 0.05837078815780742 | -0.08974245395754751 | 0.00156201333813126 | -0.38072073893846664 | -0.6476890850463006 | -0.030487663475442836 | 0.14494828105785174 | 0.6817254060840733 | 0.0050751346786845645 |
| `H_E4_mae.json` | E4_mae | THINGS | 61 | 0.013528199504984366 | 0.2288668386896231 | -0.008507169203647772 | 0.0355635682136165 | 0.13425243002369705 | -0.12390058124129284 | 0.3753845092483107 | 0.018023714967267627 | 0.42265556584110153 | 0.008184052464249268 |
| `H_E4_mae.json` | E4_mae | HSJ | 32 | -0.023762729228589952 | 0.13084743353830844 | -0.05459080932539134 | 0.007065350868211435 | -0.26522332982431024 | -0.5663439211248606 | 0.098367392412588 | 0.07034341468309449 | 0.1128828924806442 | 0.04489717164604127 |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | THINGS | 61 | -0.006122312630492607 | 0.6554798595906768 | -0.03301724870535341 | 0.020772623444368192 | -0.05925117719451467 | -0.3085381994393243 | 0.19765177796706776 | 0.0035107019989358015 | 0.6833817766564823 | 0.002766211528768359 |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | HSJ | 32 | -0.02756895337158716 | 0.03171946515199436 | -0.05272536102539611 | -0.002412545717778214 | -0.34998649908206525 | -0.626621290930269 | 0.004976773766946199 | 0.12249054953972002 | 0.2245987322588604 | 0.027139729897031197 |
| `H_F2_pixels.json` | F2_pixels | THINGS | 61 | 0.008663890267475232 | 0.4872274272834699 | -0.015778736535820445 | 0.03310651707077091 | 0.0836043449070857 | -0.17401917452583757 | 0.3305167772285434 | 0.0069896864873429596 | 0.7886095775800752 | 0.0011453058199306725 |
| `H_F2_pixels.json` | F2_pixels | HSJ | 32 | -0.024768176820484836 | 0.14319599439330308 | -0.05792731810247043 | 0.008390964461500762 | -0.28280919577903085 | -0.5791202765248817 | 0.07950417413562293 | 0.0799810412171818 | 0.05599169989618612 | 0.056512098268864996 |

Regiones de H2 (P) según `agents_regions_report.md` (sin ajuste):

- R0_dinov2s: THINGS, «Inconcluso» con los tres SESOI; HSJ, «Apoyo» con los tres. El análisis P queda en GA0b («R0 en 'Inconcluso'») con todos los SESOI y cortes (§11).
- O1, E2: «Inconcluso» en THINGS y en HSJ, con los tres SESOI.
- E1, O2, F1_rn50rand0: THINGS «Inconcluso» (tres SESOI); HSJ «Apoyo» (tres SESOI).
- R1_cvcl, E4_mae, F2_pixels: THINGS «dependiente del margen» (Inconcluso en SESOI 0,10; No apoyo en 0,20 y 0,30); HSJ «Inconcluso» (tres SESOI).
- E3_rn50sup: THINGS «dependiente del margen» (Inconcluso en 0,10 y 0,20; No apoyo en 0,30); HSJ «Inconcluso» (tres SESOI).

H2b (descriptivo; mismos n) está en el objeto `H2b` de cada `H_<encoder>.json`. No se usa para regiones ni GA. `H2_Dall_desc` es otro objeto descriptivo del mismo archivo.

### 9.3 H3 (descriptivo)

No entra en ninguna familia (§3.1). Fuente: objeto `H3` de `H_<encoder>.json`. Campos: `tier_d_perc_e`, `tier_d_sem`, `tier_d_things`, `tier_d_hsj`, `tier_d_c10h` (n = 10, salvo `tier_d_hsj` con n = 7) y `c10h_entropy_vs_margin` (n = 10000, más `centroid_acc`).

| archivo | encoder | campo | n | rho | ci_lo | ci_hi | centroid_acc (solo entropy) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `H_R0_dinov2s.json` | R0_dinov2s | tier_d_perc_e | 10 | -0.7696969696969697 | -0.958333333333333 | -0.2724475524475564 | — |
| `H_R0_dinov2s.json` | R0_dinov2s | tier_d_sem | 10 | -0.7693218186208295 | -0.9905211130872972 | -0.12715136223332826 | — |
| `H_R0_dinov2s.json` | R0_dinov2s | tier_d_things | 10 | -0.6848484848484848 | -0.8867924528301887 | -0.08176100628930819 | — |
| `H_R0_dinov2s.json` | R0_dinov2s | tier_d_hsj | 7 | -0.7500000000000002 | -1.0 | 0.1764705882352941 | — |
| `H_R0_dinov2s.json` | R0_dinov2s | tier_d_c10h | 10 | -0.8181818181818182 | -0.9876543209876544 | -0.27382834001730055 | — |
| `H_R0_dinov2s.json` | R0_dinov2s | c10h_entropy_vs_margin | 10000 | 0.3914843200678528 | 0.3738900720134748 | 0.4090011105479213 | 0.9457 |
| `H_O1_dinov2b.json` | O1_dinov2b | tier_d_perc_e | 10 | -0.9757575757575757 | -1.0 | -0.8101265822784811 | — |
| `H_O1_dinov2b.json` | O1_dinov2b | tier_d_sem | 10 | -0.8124038404635959 | -0.990341009999871 | -0.2913707542752497 | — |
| `H_O1_dinov2b.json` | O1_dinov2b | tier_d_things | 10 | -0.7212121212121211 | -0.9876543209876544 | -0.11538461538461539 | — |
| `H_O1_dinov2b.json` | O1_dinov2b | tier_d_hsj | 7 | -1.0 | -1.0 | -0.9999999999999999 | — |
| `H_O1_dinov2b.json` | O1_dinov2b | tier_d_c10h | 10 | -0.9151515151515152 | -1.0 | -0.5569620253164557 | — |
| `H_O1_dinov2b.json` | O1_dinov2b | c10h_entropy_vs_margin | 10000 | 0.30586116566231825 | 0.287209515804072 | 0.32424044916963185 | 0.9732 |
| `H_R1_cvcl.json` | R1_cvcl | tier_d_perc_e | 10 | -0.8424242424242423 | -0.9627329192546584 | -0.425 | — |
| `H_R1_cvcl.json` | R1_cvcl | tier_d_sem | 10 | -0.8370221386594626 | -0.9969088024959091 | -0.3538459462002978 | — |
| `H_R1_cvcl.json` | R1_cvcl | tier_d_things | 10 | -0.8181818181818182 | -0.987012987012987 | -0.3766233766233766 | — |
| `H_R1_cvcl.json` | R1_cvcl | tier_d_hsj | 7 | -0.7857142857142859 | -1.0 | -0.058823529411764705 | — |
| `H_R1_cvcl.json` | R1_cvcl | tier_d_c10h | 10 | -0.8303030303030302 | -1.0 | -0.37253016591251903 | — |
| `H_R1_cvcl.json` | R1_cvcl | c10h_entropy_vs_margin | 10000 | 0.14499838843313406 | 0.12583335685637614 | 0.16394085432135672 | 0.4983 |
| `H_E1_siglip.json` | E1_siglip | tier_d_perc_e | 10 | -0.8303030303030302 | -1.0 | -0.3594771241830065 | — |
| `H_E1_siglip.json` | E1_siglip | tier_d_sem | 10 | -0.8062492659146293 | -0.9905211130872972 | -0.2400597944891279 | — |
| `H_E1_siglip.json` | E1_siglip | tier_d_things | 10 | -0.8060606060606059 | -0.9627329192546584 | -0.37254901960784303 | — |
| `H_E1_siglip.json` | E1_siglip | tier_d_hsj | 7 | -0.8928571428571429 | -1.0 | -0.411764705882353 | — |
| `H_E1_siglip.json` | E1_siglip | tier_d_c10h | 10 | -0.9393939393939393 | -1.0 | -0.6743882048943612 | — |
| `H_E1_siglip.json` | E1_siglip | c10h_entropy_vs_margin | 10000 | 0.3989289183095131 | 0.3813971660646691 | 0.4161066511120063 | 0.9325 |
| `H_O2_openclip.json` | O2_openclip | tier_d_perc_e | 10 | -0.8181818181818182 | -0.9876543209876544 | -0.3584905660377359 | — |
| `H_O2_openclip.json` | O2_openclip | tier_d_sem | 10 | -0.830867564110496 | -1.0 | -0.33508606706593635 | — |
| `H_O2_openclip.json` | O2_openclip | tier_d_things | 10 | -0.7818181818181817 | -0.9480519480519479 | -0.3544303797468355 | — |
| `H_O2_openclip.json` | O2_openclip | tier_d_hsj | 7 | -0.8928571428571429 | -1.0 | -0.411764705882353 | — |
| `H_O2_openclip.json` | O2_openclip | tier_d_c10h | 10 | -0.9272727272727272 | -1.0 | -0.6603773584905661 | — |
| `H_O2_openclip.json` | O2_openclip | c10h_entropy_vs_margin | 10000 | 0.38541266747624436 | 0.367887803164839 | 0.40298092620799636 | 0.947 |
| `H_E2_vitsup.json` | E2_vitsup | tier_d_perc_e | 10 | -0.9757575757575757 | -1.0 | -0.8101265822784811 | — |
| `H_E2_vitsup.json` | E2_vitsup | tier_d_sem | 10 | -0.8924133096001623 | -0.9999999999999999 | -0.48110725206125 | — |
| `H_E2_vitsup.json` | E2_vitsup | tier_d_things | 10 | -0.8545454545454544 | -0.9999999999999999 | -0.4339622641509434 | — |
| `H_E2_vitsup.json` | E2_vitsup | tier_d_hsj | 7 | -1.0 | -1.0 | -0.9999999999999999 | — |
| `H_E2_vitsup.json` | E2_vitsup | tier_d_c10h | 10 | -0.9757575757575757 | -1.0 | -0.8101265822784811 | — |
| `H_E2_vitsup.json` | E2_vitsup | c10h_entropy_vs_margin | 10000 | 0.2935128193801017 | 0.27466595254523957 | 0.3117615240673912 | 0.9511 |
| `H_E3_rn50sup.json` | E3_rn50sup | tier_d_perc_e | 10 | -0.8666666666666665 | -1.0 | -0.4649681528662421 | — |
| `H_E3_rn50sup.json` | E3_rn50sup | tier_d_sem | 10 | -0.793940116816696 | -0.9663202545932099 | -0.2400597944891279 | — |
| `H_E3_rn50sup.json` | E3_rn50sup | tier_d_things | 10 | -0.7575757575757575 | -0.9615384615384615 | -0.24519669503021463 | — |
| `H_E3_rn50sup.json` | E3_rn50sup | tier_d_hsj | 7 | -0.8571428571428573 | -1.0 | -0.38461538461538464 | — |
| `H_E3_rn50sup.json` | E3_rn50sup | tier_d_c10h | 10 | -0.9151515151515152 | -1.0 | -0.6470588235294118 | — |
| `H_E3_rn50sup.json` | E3_rn50sup | c10h_entropy_vs_margin | 10000 | 0.29802046678713945 | 0.28053692834268373 | 0.3159132172607989 | 0.8147 |
| `H_E4_mae.json` | E4_mae | tier_d_perc_e | 10 | -0.41818181818181815 | -0.9622641509433963 | 0.35802469135802467 | — |
| `H_E4_mae.json` | E4_mae | tier_d_sem | 10 | -0.40620192023179796 | -0.8698052229052508 | 0.32984799438208245 | — |
| `H_E4_mae.json` | E4_mae | tier_d_things | 10 | -0.3939393939393939 | -0.7735849056603774 | 0.2704402515723271 | — |
| `H_E4_mae.json` | E4_mae | tier_d_hsj | 7 | -0.42857142857142866 | -0.8867924528301889 | 0.5555555555555555 | — |
| `H_E4_mae.json` | E4_mae | tier_d_c10h | 10 | -0.406060606060606 | -0.8633540372670807 | 0.28301886792452835 | — |
| `H_E4_mae.json` | E4_mae | c10h_entropy_vs_margin | 10000 | -0.027421917134347294 | -0.04669315580277431 | -0.00862390940388564 | 0.2348 |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | tier_d_perc_e | 10 | -0.4666666666666666 | -0.8875 | 0.271523178807947 | — |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | tier_d_sem | 10 | -0.4800568148193976 | -0.9205557719776651 | 0.2672868188724552 | — |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | tier_d_things | 10 | -0.35757575757575755 | -0.7750349378881987 | 0.28664609872611435 | — |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | tier_d_hsj | 7 | -0.6071428571428572 | -1.0 | 0.06666666666666665 | — |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | tier_d_c10h | 10 | -0.406060606060606 | -0.8616460637605725 | 0.3039185810282839 | — |
| `H_F1_rn50rand0.json` | F1_rn50rand0 | c10h_entropy_vs_margin | 10000 | 0.04655211385148555 | 0.02712515864270271 | 0.06587847954314614 | 0.2479 |
| `H_F2_pixels.json` | F2_pixels | tier_d_perc_e | 10 | -0.29696969696969694 | -0.9230769230769231 | 0.6582815858609976 | — |
| `H_F2_pixels.json` | F2_pixels | tier_d_sem | 10 | -0.06154574548966636 | -0.6604923071741964 | 0.7424255638529593 | — |
| `H_F2_pixels.json` | F2_pixels | tier_d_things | 10 | 0.22424242424242422 | -0.4771387535906082 | 0.730877483443707 | — |
| `H_F2_pixels.json` | F2_pixels | tier_d_hsj | 7 | -0.14285714285714288 | -0.7647058823529412 | 0.8867924528301889 | — |
| `H_F2_pixels.json` | F2_pixels | tier_d_c10h | 10 | 0.04242424242424241 | -0.6202531645569621 | 0.8101265822784811 | — |
| `H_F2_pixels.json` | F2_pixels | c10h_entropy_vs_margin | 10000 | 0.05036090882890049 | 0.03086183409267218 | 0.06985095465063408 | 0.2757 |

---

## 10. H4 (réplica por encoder)

H4 se declara réplica de la v1: está en la familia C (o C-pisos) solo para reproducir por encoder el Holm de 6 de la v1; no tiene lectura GA propia (§3.1). El modelo es el C1 de `04c_h4_person_re.R`, sin cambios, con caída a `(1|persona)` por encoder. Regla de región, §3.2 y (b7): **Apoyo** si β < 0 con p < 0,05; **No apoyo** = signo opuesto (logOR > 0) con p < 0,05; H4 no tiene SESOI, así que las tres columnas de región coinciden.

H4 no se recortó. Fuente conjunta: `h4_by_encoder.csv` (idéntica, fila a fila, a cada `h4_<encoder>.csv`). En los diez: `RE` = slope, `status` = ok, `singular` = False, `warn` vacío, `nobs` = 172800, `persons` = 520. El campo `p` está escrito como `0` en el CSV.

| archivo por encoder | encoder | RE | logOR | se | lo | hi | p | nobs | persons | status | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `h4_R0_dinov2s.csv` | R0_dinov2s | slope | -3.22613316905941 | 0.0315887995532445 | -3.28804607849863 | -3.1642202596202 | 0 | 172800 | 520 | ok | 242.7 |
| `h4_O1_dinov2b.csv` | O1_dinov2b | slope | -2.8592995504845 | 0.0280504677782775 | -2.91427745707943 | -2.80432164388958 | 0 | 172800 | 520 | ok | 323.2 |
| `h4_R1_cvcl.csv` | R1_cvcl | slope | -3.61036360002093 | 0.0383666436178169 | -3.68556083971953 | -3.53516636032232 | 0 | 172800 | 520 | ok | 241.6 |
| `h4_E1_siglip.csv` | E1_siglip | slope | -5.48252705428954 | 0.051604932425313 | -5.58367086326778 | -5.3813832453113 | 0 | 172800 | 520 | ok | 395.1 |
| `h4_O2_openclip.csv` | O2_openclip | slope | -4.80450700970046 | 0.0502530655167071 | -4.90300120822594 | -4.70601281117498 | 0 | 172800 | 520 | ok | 283.0 |
| `h4_E2_vitsup.csv` | E2_vitsup | slope | -4.45602350099226 | 0.050354090085644 | -4.55471570403441 | -4.35733129795011 | 0 | 172800 | 520 | ok | 237.6 |
| `h4_E3_rn50sup.csv` | E3_rn50sup | slope | -4.1399066303111 | 0.0475723059913582 | -4.23314663671568 | -4.04666662390652 | 0 | 172800 | 520 | ok | 212.0 |
| `h4_E4_mae.csv` | E4_mae | slope | -7.55668466607456 | 0.0534798708613654 | -7.66150328686069 | -7.45186604528843 | 0 | 172800 | 520 | ok | 764.3 |
| `h4_F1_rn50rand0.csv` | F1_rn50rand0 | slope | -2.67817270081309 | 0.0251761627465744 | -2.7275170730653 | -2.62882832856089 | 0 | 172800 | 520 | ok | 137.7 |
| `h4_F2_pixels.csv` | F2_pixels | slope | -2.4301771910003 | 0.0185418325801086 | -2.46651851506469 | -2.39383586693592 | 0 | 172800 | 520 | ok | 272.0 |

En `agents_regions_report.md`, los diez primarios quedan en región «Apoyo» (perfil «robusto: Apoyo»). E4s_mae_cls, F1_rn50rand1 y F1_rn50rand2 figuran con `RE` vacío y `status` «falta» (sensibilidades no corridas). En `regions/agents_summary.json`, `families.*.h4_tecnico_fuera_solo_prueba` es `[]` y `n_tests` es 36 (C), 6 (D), 12 (C-pisos) y 2 (D-pisos).

`regions/replica_holm6_v1.csv` contiene la réplica del Holm de 6 por encoder (valores en ese CSV; no se reinterpreta aquí).

---

## 11. Regiones, familias Holm y conclusiones GA

Las regiones por encoder no usan Holm. Holm solo decide GA1, por SESOI y dentro de cada familia (§3.1). Precedencia de GA, §3.4: GA0 > GA0b > GA1 > GA2 > GA2b > GA3; gana la primera regla que se cumpla. Márgenes de agentes: SESOI ∈ {0,10; 0,20; 0,30} en escala de correlación; no son comparables con los δ humanos (§3.2). Constantes aplicadas, según `regions/agents_summary.json`: `D_P="v1"`, `A_AGG="todas"`, `CONCORD="particion_identica"`, `GROUPS_OVER` = los seis entrenados nuevos, `SESOIS` = [0.1, 0.2, 0.3], `CUTS` = ["0.3", "0.5", "0.7"].

Se transcribe, sin cambios, el informe generado por `14_agent_regions.py` (y la figura de `14b_fig_x1.py` que el propio informe apunta). Fuente: `regions/agents_regions_report.md`.

````markdown
# Agentes: regiones y conclusiones GA (generado por 14_agent_regions.py; sin interpretación)

- §3.7 (circularidad): frente a las personas solo se leen H1 y H2; un GA2 en D significa "suficiencia del mecanismo" (el mecanismo basta con otros encoders), no correspondencia con las personas.
- Control de tubería de agentes (§3.5): PASA (T exacto: True; CIFAR-100 recodificado: PASA).
- Márgenes de agentes (SESOI, escala de correlación; NO comparables con los δ humanos): 0.10, 0.20, 0.30.
- Geometría: grupos medidos en estímulos de Lewis & Frank; concordancia con CIFAR por corte: {'0.3': False, '0.5': False, '0.7': False}.

## Familias Holm (solo para GA1)

- **C:** incluidos ['O1_dinov2b', 'E1_siglip', 'O2_openclip', 'E2_vitsup', 'E3_rn50sup', 'E4_mae']; excluidos {}; 36 pruebas por SESOI.
- **D:** incluidos ['O1_dinov2b', 'E1_siglip', 'O2_openclip', 'E2_vitsup', 'E3_rn50sup', 'E4_mae']; excluidos {}; 6 pruebas por SESOI.
- **C-pisos:** incluidos ['F1_rn50rand0', 'F2_pixels']; excluidos {}; 12 pruebas por SESOI.
- **D-pisos:** incluidos ['F1_rn50rand0', 'F2_pixels']; excluidos {}; 2 pruebas por SESOI.

## Regiones por encoder (sin ajuste)

| encoder | analisis | fuente | control_positivo_agentes | region_0.10 | region_0.20 | region_0.30 | perfil |
|---|---|---|---|---|---|---|---|
| R0_dinov2s | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| R0_dinov2s | P | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| O1_dinov2b | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| O1_dinov2b | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | P | THINGS | pasa | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| R1_cvcl | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| R1_cvcl | D | - | pasa | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| R1_cvcl | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E1_siglip | P | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| O2_openclip | P | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E2_vitsup | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E2_vitsup | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | P | THINGS | pasa | Inconcluso | Inconcluso | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10, 0.20); No apoyo (SESOI = 0.30) |
| E3_rn50sup | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E3_rn50sup | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | P | THINGS | pasa | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| E4_mae | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E4_mae | D | - | pasa | Inconcluso | Inconcluso | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10, 0.20); No apoyo (SESOI = 0.30) |
| E4_mae | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | A | THINGS | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | A | HSJ | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | A | CIFAR10H | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | P | THINGS | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F1_rn50rand0 | P | HSJ | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | D | - | no es puerta | No apoyo | No apoyo | No apoyo | robusto: No apoyo |
| F1_rn50rand0 | Dall | - | no es puerta | No apoyo | No apoyo | No apoyo | robusto: No apoyo |
| F1_rn50rand0 | H4 | - | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | A | THINGS | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | A | HSJ | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | A | CIFAR10H | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F2_pixels | P | THINGS | no es puerta | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| F2_pixels | P | HSJ | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F2_pixels | D | - | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F2_pixels | Dall | - | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | H4 | - | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4s_mae_cls | A | THINGS | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | A | HSJ | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | A | CIFAR10H | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | P | THINGS | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | P | HSJ | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | D | - | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | Dall | - | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | H4 | - | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | A | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | A | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | A | CIFAR10H | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | P | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | P | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | D | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | Dall | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | H4 | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | A | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | A | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | A | CIFAR10H | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | P | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | P | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | D | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | Dall | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | H4 | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |

## GA por análisis (SESOI × corte)

| analisis | sesoi | corte | conclusion | detalle |
|---|---|---|---|---|
| D | 0.1 | 0.3 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.1 | 0.5 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.1 | 0.7 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.2 | 0.3 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.2 | 0.5 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.2 | 0.7 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.3 | 0.3 | GA1 | depende del encoder: ['E4_mae']; grupos: sin grupos |
| D | 0.3 | 0.5 | GA1 | depende del encoder: ['E4_mae']; grupos: sin grupos |
| D | 0.3 | 0.7 | GA1 | depende del encoder: ['E4_mae']; grupos: sin grupos |
| A | 0.1 | 0.3 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.1 | 0.5 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.1 | 0.7 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.2 | 0.3 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.2 | 0.5 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.2 | 0.7 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.3 | 0.3 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.3 | 0.5 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.3 | 0.7 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| P | 0.1 | 0.3 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.1 | 0.5 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.1 | 0.7 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.2 | 0.3 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.2 | 0.5 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.2 | 0.7 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.3 | 0.3 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.3 | 0.5 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.3 | 0.7 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |

- **Perfil D:** depende solo del margen: SESOI = 0.10: GA3; SESOI = 0.20: GA3; SESOI = 0.30: GA1
- **Perfil A:** robusta: GA2b
- **Perfil P:** robusta: GA0b

## D con 100 pares vs D-todos (rótulo descriptivo)

Nota: los p de D-todos son anticonservadores (los 4.950 pares comparten clases), así que el rótulo puede dispararse por eso; es solo descriptivo y no cambia GA.

| encoder | D100_0.10 | D100_0.20 | D100_0.30 | Dtodos_0.10 | Dtodos_0.20 | Dtodos_0.30 | rotulo | rule_centroid_D100 |
|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | A |
| O1_dinov2b | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | A |
| R1_cvcl | Inconcluso | No apoyo | No apoyo | Apoyo | Apoyo | Apoyo | sensible al muestreo de pares | C |
| E1_siglip | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | B |
| O2_openclip | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | B |
| E2_vitsup | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | A |
| E3_rn50sup | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | B |
| E4_mae | Inconcluso | Inconcluso | No apoyo | Apoyo | Apoyo | Apoyo | sensible al muestreo de pares | C |
| F1_rn50rand0 | No apoyo | No apoyo | No apoyo | No apoyo | No apoyo | No apoyo | estable | C |
| F2_pixels | Inconcluso | Inconcluso | Inconcluso | Apoyo | Apoyo | Apoyo | sensible al muestreo de pares | B |
| E4s_mae_cls | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | estable |  |
| F1_rn50rand1 | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | estable |  |
| F1_rn50rand2 | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | estable |  |

## H4: estructura de efectos aleatorios usada por encoder (caída a `noslope` por encoder, como 04c)

| encoder | RE | status |
|---|---|---|
| R0_dinov2s | slope | ok |
| O1_dinov2b | slope | ok |
| R1_cvcl | slope | ok |
| E1_siglip | slope | ok |
| O2_openclip | slope | ok |
| E2_vitsup | slope | ok |
| E3_rn50sup | slope | ok |
| E4_mae | slope | ok |
| F1_rn50rand0 | slope | ok |
| F2_pixels | slope | ok |
| E4s_mae_cls |  | falta |
| F1_rn50rand1 |  | falta |
| F1_rn50rand2 |  | falta |

- Rótulo descriptivo «Apoyo con IC bajo SESOI» (b4) y región con HC3 en D (b6): columnas `rotulo_*`, `region_hc3_*` y `hc3_cambia_region` de regions_agents.csv. ARI L&F vs CIFAR por corte (b9, descriptivo): {'0.3': 0.352, '0.5': 0.10179640718562874, '0.7': 0.10256410256410256}.

## Tabla agentes × personas (descriptiva)

| encoder | humano_beta_zd_control_positivo | humano_ic_lo | humano_ic_hi | humano_control_positivo | H4_logOR | H4_lo | H4_hi | H4_RE | humano_region_d0.05 | humano_region_d0.10 | humano_region_d0.20 | humano_region_24 | humano_perfil | C1_native | C1_near | C1_mid | C1_far | D_beta_perc_v1escala | D_beta_perc_z | D_regla_v1 | H1_rho_THINGS | H1_rho_HSJ | H1_rho_CIFAR10H | H2_beta_humana_THINGS | H2_beta_humana_HSJ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | -3.157 | -3.296 | -3.017 | True | -3.226 | -3.288 | -3.164 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9429 | 0.8467 | 0.6542 | 0.5296 | -0.2119 | -0.02885 | A | 0.5228 | 0.5006 | 0.7949 | -0.00785 | -0.03445 |
| O1_dinov2b | -2.848 | -2.976 | -2.72 | True | -2.859 | -2.914 | -2.804 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9633 | 0.8344 | 0.6691 | 0.5444 | -0.2362 | -0.02747 | A | 0.5207 | 0.5475 | 0.6771 | -0.01083 | -0.02728 |
| R1_cvcl | -0.6152 | -0.7054 | -0.525 | True | -3.61 | -3.686 | -3.535 | slope | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20) | 0.6753 | 0.6469 | 0.5107 | 0.4276 | 0.00875 | 0.001281 | C | 0.6825 | 0.4294 | 0.7862 | 0.01136 | -0.0449 |
| E1_siglip | -3.887 | -4.059 | -3.715 | True | -5.483 | -5.584 | -5.381 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9576 | 0.896 | 0.6964 | 0.5342 | -0.5077 | -0.02168 | B | 0.6392 | 0.5008 | 0.8896 | -0.01659 | -0.05083 |
| O2_openclip | -3.993 | -4.157 | -3.83 | True | -4.805 | -4.903 | -4.706 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9702 | 0.9144 | 0.6709 | 0.524 | -0.2733 | -0.02429 | B | 0.7018 | 0.5187 | 0.8669 | -0.01995 | -0.04302 |
| E2_vitsup | -4.296 | -4.462 | -4.13 | True | -4.456 | -4.555 | -4.357 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9776 | 0.8993 | 0.7202 | 0.518 | -0.2887 | -0.03542 | A | 0.3934 | 0.556 | 0.6611 | -0.006119 | -0.03105 |
| E3_rn50sup | -2.648 | -2.775 | -2.52 | True | -4.14 | -4.233 | -4.047 | slope | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20) | 0.8704 | 0.8576 | 0.6442 | 0.4947 | -0.2594 | -0.02345 | B | 0.6181 | 0.5328 | 0.912 | -0.004229 | -0.04409 |
| E4_mae | -0.1982 | -0.2971 | -0.09943 | True | -7.557 | -7.662 | -7.452 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.4827 | 0.4238 | 0.4818 | 0.3882 | -0.00155 | -0.0002797 | C | 0.2918 | 0.2273 | 0.4449 | 0.01353 | -0.02376 |
| F1_rn50rand0 | -1.102 | -1.215 | -0.9897 | True | -2.678 | -2.728 | -2.629 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.4102 | 0.4311 | 0.4282 | 0.3844 | 49.18 | 0.01197 | C | 0.1088 | 0.1063 | 0.5302 | -0.006122 | -0.02757 |
| F2_pixels | 0.116 | 0.03845 | 0.1935 | False | -2.43 | -2.467 | -2.394 | slope | inconcluso | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05); menor que δ (equivalente con margen δ) (δ = 0.10, 0.20) | 0.4578 | 0.4433 | 0.4602 | 0.3869 | -0.02269 | -0.009888 | B | 0.1505 | 0.1988 | 0.2941 | 0.008664 | -0.02477 |
| E4s_mae_cls |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F1_rn50rand1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F1_rn50rand2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Spearman entre encoders (descriptiva, sin p): {"_nota": "Spearman entre encoders (los 10 primarios), solo descriptiva: sin p ni IC inferencial", "beta_zd_humano_vs_H1_rho_THINGS": {"n": 10, "rho": -0.4545454545454545}, "beta_zd_humano_vs_H1_rho_HSJ": {"n": 10, "rho": -0.7454545454545454}, "beta_zd_humano_vs_H1_rho_CIFAR10H": {"n": 10, "rho": -0.5515151515151515}, "beta_zd_humano_vs_H2_beta_humana_THINGS": {"n": 10, "rho": 0.7454545454545454}, "beta_zd_humano_vs_H2_beta_humana_HSJ": {"n": 10, "rho": 0.4545454545454545}}

## Dispersión de d_perc en los 100 pares (§3.6)

```
{
 "_rule": "100 v1 pairs (phase2/sampled_pairs.csv); d_sem = exp_008 d_sem; quadrants (quadrant_counts): d_perc median of the encoder over the 4,950 pairs and d_sem median over the 4,950 (v1 rule, REVISION5 b3); quadrant_counts_100: d_perc median over the 100 pairs (descriptive). Descriptive; changes no rule.",
 "R0_dinov2s": {
  "n": 100,
  "sd": 0.13613767209690872,
  "min": 0.1414257287979126,
  "max": 0.8853307366371155,
  "range": 0.7439050078392029,
  "spearman_dperc_dsem": 0.01295110698443623,
  "quadrant_counts": {
   "Q1": 25,
   "Q2": 25,
   "Q3": 25,
   "Q4": 25
  },
  "quadrant_counts_100": {
   "Q1": 25,
   "Q2": 25,
   "Q3": 25,
   "Q4": 25
  }
 },
 "O1_dinov2b": {
  "n": 100,
  "sd": 0.1162828286086192,
  "min": 0.15615785121917725,
  "max": 0.9425162076950073,
  "range": 0.7863583564758301,
  "spearman_dperc_dsem": -0.006388492272341284,
  "quadrant_counts": {
   "Q1": 21,
   "Q2": 26,
   "Q3": 29,
   "Q4": 24
  },
  "quadrant_counts_100": {
   "Q1": 22,
   "Q2": 22,
   "Q3": 28,
   "Q4": 28
  }
 },
 "R1_cvcl": {
  "n": 100,
  "sd": 0.14632199426941384,
  "min": 0.03745764493942261,
  "max": 0.6267770528793335,
  "range": 0.5893194079399109,
  "spearman_dperc_dsem": 0.11970617521961292,
  "quadrant_counts": {
   "Q1": 29,
   "Q2": 31,
   "Q3": 21,
   "Q4": 19
  },
  "quadrant_counts_100": {
   "Q1": 29,
   "Q2": 29,
   "Q3": 21,
   "Q4": 21
  }
 },
 "E1_siglip": {
  "n": 100,
  "sd": 0.042696184192094026,
  "min": 0.015215158462524414,
  "max": 0.20334410667419434,
  "range": 0.18812894821166992,
  "spearman_dperc_dsem": 0.08024042361613619,
  "quadrant_counts": {
   "Q1": 28,
   "Q2": 26,
   "Q3": 22,
   "Q4": 24
  },
  "quadrant_counts_100": {
   "Q1": 28,
   "Q2": 28,
   "Q3": 22,
   "Q4": 22
  }
 },
 "O2_openclip": {
  "n": 100,
  "sd": 0.0888672826840226,
  "min": 0.03589349985122681,
  "max": 0.4585176110267639,
  "range": 0.4226241111755371,
  "spearman_dperc_dsem": 0.12873652519856152,
  "quadrant_counts": {
   "Q1": 32,
   "Q2": 30,
   "Q3": 18,
   "Q4": 20
  },
  "quadrant_counts_100": {
   "Q1": 30,
   "Q2": 30,
   "Q3": 20,
   "Q4": 20
  }
 },
 "E2_vitsup": {
  "n": 100,
  "sd": 0.12268824662406681,
  "min": 0.18996769189834595,
  "max": 0.8461833000183105,
  "range": 0.6562156081199646,
  "spearman_dperc_dsem": 0.051191997287576854,
  "quadrant_counts": {
   "Q1": 24,
   "Q2": 27,
   "Q3": 26,
   "Q4": 23
  },
  "quadrant_counts_100": {
   "Q1": 24,
   "Q2": 24,
   "Q3": 26,
   "Q4": 26
  }
 },
 "E3_rn50sup": {
  "n": 100,
  "sd": 0.09043117073404477,
  "min": 0.0369870662689209,
  "max": 0.4884408116340637,
  "range": 0.4514537453651428,
  "spearman_dperc_dsem": 0.07574326129284333,
  "quadrant_counts": {
   "Q1": 23,
   "Q2": 27,
   "Q3": 27,
   "Q4": 23
  },
  "quadrant_counts_100": {
   "Q1": 26,
   "Q2": 26,
   "Q3": 24,
   "Q4": 24
  }
 },
 "E4_mae": {
  "n": 100,
  "sd": 0.18046432665954423,
  "min": 0.011018753051757812,
  "max": 0.8762620687484741,
  "range": 0.8652433156967163,
  "spearman_dperc_dsem": 0.19609188827292665,
  "quadrant_counts": {
   "Q1": 23,
   "Q2": 32,
   "Q3": 27,
   "Q4": 18
  },
  "quadrant_counts_100": {
   "Q1": 27,
   "Q2": 27,
   "Q3": 23,
   "Q4": 23
  }
 },
 "F1_rn50rand0": {
  "n": 100,
  "sd": 0.00024350480795842487,
  "min": 2.0503997802734375e-05,
  "max": 0.001495063304901123,
  "range": 0.0014745593070983887,
  "spearman_dperc_dsem": 0.01894932482284689,
  "quadrant_counts": {
   "Q1": 24,
   "Q2": 27,
   "Q3": 26,
   "Q4": 23
  },
  "quadrant_counts_100": {
   "Q1": 27,
   "Q2": 27,
   "Q3": 23,
   "Q4": 23
  }
 },
 "F2_pixels": {
  "n": 100,
  "sd": 0.4357685845211876,
  "min": 0.06994873285293579,
  "max": 1.8120839595794678,
  "range": 1.742135226726532,
  "spearman_dperc_dsem": 0.20052300415355434,
  "quadrant_counts": {
   "Q1": 22,
   "Q2": 32,
   "Q3": 28,
   "Q4": 18
  },
  "quadrant_counts_100": {
   "Q1": 28,
   "Q2": 28,
   "Q3": 22,
   "Q4": 22
  }
 },
 "E4s_mae_cls": {
  "n": 100,
  "sd": 0.013674846146170093,
  "min": 0.0016238093376159668,
  "max": 0.07278639078140259,
  "range": 0.07116258144378662,
  "spearman_dperc_dsem": 0.16452168867896946,
  "quadrant_counts": {
   "Q1": 25,
   "Q2": 31,
   "Q3": 25,
   "Q4": 19
  },
  "quadrant_counts_100": {
   "Q1": 30,
   "Q2": 30,
   "Q3": 20,
   "Q4": 20
  }
 },
 "F1_rn50rand1": {
  "status": "tecnico: falta /workspace/lexical-consensus-openmind/encoder_equivalence/agents/out/emb/c100train_F1_rn50rand1.npz"
 },
 "F1_rn50rand2": {
  "status": "tecnico: falta /workspace/lexical-consensus-openmind/encoder_equivalence/agents/out/emb/c100train_F1_rn50rand2.npz"
 }
}
```
````

Perfiles GA (también en `regions/agents_summary.json`, campo `ga_profile`, y en `regions/ga_grid.csv`): D «depende solo del margen: SESOI = 0.10: GA3; SESOI = 0.20: GA3; SESOI = 0.30: GA1»; A «robusta: GA2b»; P «robusta: GA0b». En D con SESOI = 0,30, la celda de GA1 nombra a E4_mae y escribe «grupos: sin grupos». En A, el detalle de GA2b es «se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla)». En P, GA0b: «descriptivo, encoder por encoder: R0 en 'Inconcluso'».

Concordancia L&F frente a CIFAR (`CONCORD="particion_identica"`): `false` en los tres cortes (viñeta del informe y `agents_summary.json` → `cifar_geometry.concordance`). ARI descriptivo (b9): 0.352, 0.10179640718562874 y 0.10256410256410256 en los cortes 0.3, 0.5 y 0.7.

Un GA descriptivo por fuente está en `regions/ga_grid_por_fuente_descriptivo.csv` (b8); no decide las GA de la grilla principal.

---

## 12. Tabla agentes × personas y figura x1

La tabla descriptiva agentes × personas está en `regions/tabla_agentes_personas.csv` y transcrita en el informe del §11 (no entra en ninguna familia ni en GA; §3.1 y §3.7). La figura x1 (`14b_fig_x1.py`) es descriptiva y no entra en ninguna regla (ACLARACIONES (a12)).

No existe el directorio `agents/out/figs`. El archivo de la figura es `regions/fig_x1_por_encoder.png` (427875 bytes, sha256 `c29ea83e0231482faa346712333f7b5edab8815e92e8734afe22776e6664a267`), el mismo en `/workspace/agents_respaldo/results/regions/fig_x1_por_encoder.png`.

Las Spearman descriptivas entre columnas de esa tabla (sin p inferencial) están al pie del informe y en `regions/agents_summary.json` → `spearman_descriptiva`.

---

## 13. Completitud, archivos citados y observaciones de forma

**Completitud del primario.** Hay, para los diez encoders primarios: `parity_*.json`, `T_*.json`, `D_*.json`, `Dall_*.json`, `H_*.json`, `h4_*.csv`, más `h4_by_encoder.csv`, `agent_pairs_dispersion.json`, `control_T_exact.json`, `control_c100_recode.json` y los nueve archivos de `regions/`. Todos esos JSON se pudieron leer. En los objetos de análisis de esos diez encoders no aparecieron NaN. Los n coinciden entre encoders: T seed 42 y modo episodes; D n = 100; D-todos n = 4950; H1 n_pairs = 3400 / 1390 / 45; H2 n = 61 / 32; H4 nobs = 172800 y persons = 520.

**Sensibilidades no corridas** (ya en DESVIACIONES, entrada 4 del anexo): no hay `T_`, `D_`, `Dall_`, `H_` ni `h4_` de E4s_mae_cls, F1_rn50rand1 ni F1_rn50rand2. E4s sí tiene embeddings y una entrada de dispersión. En regiones quedan «inconcluso (técnico)». Eso no redujo las familias Holm del primario (`h4_tecnico_fuera_solo_prueba` vacío; n_tests = 36/6/12/2).

**Observaciones de forma (no son fallos de la corrida):**

1. En `control_T_exact.json`, `nonrandom_margin_max_abs_diff` vale `1.000000000139778e-06` y `estricto` vale `false`; la regla usada, `propuesto_b1`, quedó `"ok": true` (ver §4.1).
2. En varios `Dall_*.json` y en los CSV de H4, algunas p aparecen como `0.0` o `0`.
3. El informe de regiones incluye filas de E4s_mae_cls, F1_rn50rand1 y F1_rn50rand2 en «inconcluso (técnico)» porque no se corrió el bloque opcional de sensibilidades, no porque el primario se hubiera cortado.
4. No hay directorio `out/figs`; la figura quedó en `out/results/regions/`.
5. No se pudo anotar un hash de commit de esta copia (no hay `.git`).

**Archivos citados (resultados y logs).** `parity_*.json`, `control_T_exact.json`, `control_c100_recode.json`, `agent_pairs_dispersion.json`, `T_*.json`, `D_*.json`, `Dall_*.json`, `H_*.json`, `h4_*.csv`, `h4_by_encoder.csv`, `regions/agents_regions_report.md`, `regions/agents_summary.json`, `regions/regions_agents.csv`, `regions/ga_grid.csv`, `regions/ga_grid_por_fuente_descriptivo.csv`, `regions/sampling_sensitivity.csv`, `regions/replica_holm6_v1.csv`, `regions/tabla_agentes_personas.csv`, `regions/fig_x1_por_encoder.png`, `agents/logs/run_agents.log`, `agents/logs/reestimacion_tasa_2026-11-23.md`, `agents/logs/resume_1437.out`, `encoder_equivalence/FREEZE_agentes_v2.txt`, `encoder_equivalence/DESVIACIONES_encoders_v2.md`, `/workspace/agents_respaldo/results/SHA256SUMS`, `/workspace/agents_respaldo/emb/SHA256SUMS`.
