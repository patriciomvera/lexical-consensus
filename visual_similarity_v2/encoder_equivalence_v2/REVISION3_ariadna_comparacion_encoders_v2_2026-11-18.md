# Tercera revisión, antes de congelar: plan y scripts de la comparación de encoders (v2 de 2611.14116)

- **Revisora:** Ariadna (CRO). **Fecha:** 18-11-2026, ~12:20 (hora de Chile, UTC−3).
- **Plan revisado:** `propuesta_comparacion_encoders_v2_2026-11-18.md`, mtime 11:59:05, sha256 `75678c7999e1b9480aa85e01861523cfd04d0a472808f96c243abf1699234422`. Lo comparé con `propuesta_e5d2806a_snapshot_1039.md`.
- **Qué ejecuté:**
  - `tests/test_stage2.py` (5 de 5 OK) y `tests/test_regions.py` (16 de 16 OK), solo con datos sintéticos.
  - Inspeccioné las salidas sintéticas que ya estaban en `/tmp/synth_enc_test/`.
  - **No** ejecuté `test_freeze.py` (llama a `09_freeze.py`), `09_freeze.py`, `run_weekend.sh` ni `01b`.
  - Leí **solo la columna `dist`** de `h4_long.csv` (`usecols=['dist']`) para medir los casi-empates de G0. No leí `selected` ni ninguna respuesta.
- No modifiqué nada de Juanito.

## Veredicto
**¿Visto bueno para congelar? Sí, después de un arreglo (BC3) y de dos agregados pre-especificados (I1, I2).**
BC1, I-a, I-b y m1–m3 están implementados. La grilla δ × corte es excluyente y exhaustiva, y cada celda tiene una lectura declarada de antemano.
El control de rangos de G0, tal como está, casi seguro detendría el sábado sin motivo real.

---

## REVISION2: verificación
- **BC1, resuelto** (§7.1 l.43, §7.3 l.76, §7.4 G0/G3/G4). En `08_regions.py`, `region8` pone primero "no interpretable", `family_holm` excluye a esos encoders y `conclude` da G0 si falla R0. Hay pruebas unitarias.
- **I-a, resuelto** (§7.1 l.46).
- **I-b, resuelto** (§7.1 l.41–42 y l.99).
- **m1, resuelto** (registro de cambios). **m2, resuelto** (G2 "sin grupos", también en el código). **m3, resuelto** (§5.2 l.185).

## Cortes 0,3 / 0,5 / 0,7 (Patricio, 10:44)
- **Los cuatro casos de perfil (§7.4 l.93–97 y `grid_profile`) son excluyentes y exhaustivos.** Si las filas (δ) son constantes y los δ difieren, las columnas no pueden ser constantes; el caso 4 recoge el resto. Cada celda cae en G0–G4 por precedencia, así que ningún resultado de la grilla queda sin lectura.
- **Calibración por corte: correcta.** Si falla, ese corte queda "sin grupos" y G3 no es posible en él.
- **Expectativa que conviene declarar (MENOR):** con el corte 0,7 (unión con r ≥ .30), es probable que los pisos se agrupen con algún entrenado. Entonces la calibración falla y un "se mantiene en todos" saldría como "depende solo del corte" casi por construcción. Basta con decirlo en §8.

---

## BLOQUEANTE

**BC3. El control de rangos de G0 fallaría por ruido de punto flotante** (`h4_prep_encoders.py::check`; §5.1, §3.2).
En la v1, los 495 valores "únicos" de d son en realidad unos 290 si se redondea a 1e−6. **207 de las 494 brechas entre valores consecutivos son < 1e−6** (la mínima es 9,3e−10): son el mismo centroide calculado con las imágenes en otro orden. Los npz difieren 1,5e−8 de `PerceptionLayer` y las d hasta 3,6e−7, así que `rank(method="min")` estricto va a intercambiar esos casi-empates y declarar G0 = FAIL.
La prueba sintética no lo detecta porque ahí la diferencia es exactamente 0.
*Arreglo:*
- comparar rangos sobre `round(dist, 6)` en ambos archivos (la brecha real más chica es 4,6e−6) y mantener |Δ| ≤ 1e−5;
- agregar a `test_prep.py` un caso con ruido de 1e−8 inyectado que debe dar PASS, y uno con dos valores reales permutados que debe dar FAIL;
- corregir de paso §1.3 (l.50): "495 valores únicos" son ~290 a 1e−6.

---

## IMPORTANTES

**I1. Fallback de la etapa 1** (pregunta 1; `05_twostage_stage1.R`, `lib_glmm.R::fit_ok/fit_converged`).
Aceptar el fallback singular si converge respeta el plan (§5.2 solo excluye lo que no converge), y registrarlo está bien. Pero:
- el 195/240 sintético no dice nada, porque el generador no tiene varianza de pendiente por persona, así que la singularidad era segura;
- con datos reales, mezclar ajustes con y sin pendiente aleatoria hace que la etapa 2 compare estimadores distintos. Si el tipo de ajuste se reparte desigual entre "1-3" y "3-1", Δ puede moverse por eso y no por el orden.

*Arreglo mínimo, pre-declarado:*
- **S9:** etapa 2 con `(1 | person) + (1 | image)` en los 12 experimentos de cada encoder y conjunto (8 ítems), que es un estimador homogéneo. Cuesta ~1 h.
- Reportar el tipo de ajuste cruzado con el orden, por encoder (hoy `06` solo da `n_fallback` total).
- Regla: si S9 cambia la región con algún δ, el perfil de ese encoder se califica "sensible al tipo de ajuste" (descriptivo).

*Alternativa más limpia, que Patricio debe decidir:* usar como primario la estructura más rica que converja en los 12 experimentos, y la misma para los 12.

**I2. `01b_embed_common.py` (S6) no está probado** (pregunta 7).
Monkeyparchea `E.hf_vision`, depende de `/tmp/XTMEM` y de la caché de HF. Además está **antes de S7, S8 y la familia A** en `run_weekend.sh` con `set -e`: si falla, el domingo se corta ahí.
*Arreglo:*
- prueba de humo que no usa datos humanos: `01b --out-dir /tmp/s6smoke R0_dinov2s F1_rn50rand0 F2_pixels`. R0 ya usa 256 / recorte 224 / ImageNet por defecto, así que su coseno con `emb_R0_dinov2s.npz` debería ser ≥ .999 por imagen, y F1 y F2 deberían salir idénticos.
- mover S6 al final del domingo o envolverlo con `|| log` para que su fallo no bloquee lo demás.

---

## Respuestas puntuales
2. **G2 con el p de la prueba que define la región: correcto.** La región 1 exige IC 95 % < 0 y Holm sobre el p de signo opuesto. La región 2 equivale a p_TOST < .05 por el IC 90 %. Como G2 exige ambas cosas, la región y Holm, no hay incoherencia.
3. **Sacar de Holm a los "inconcluso (técnico)": OK**, porque no tienen p. **Que un control positivo sin ajuste cuente como "no pasa": OK** y conservador. En la tabla deben aparecer k y los encoders excluidos de cada familia.
4. **G0:** ver BC3.
5. **§10.3 (l.400–402) y `run_weekend.sh`:** coinciden (paso 6, control positivo; paso 7, etapa 2 y regiones). Ningún paso usa una salida antes de que exista.
6. **Hash de `01_embed.py`:** el actual es `4d41764e…`, pero **`rules_02_sha256_before_run.txt` no registra ningún hash de `01_embed.py`**: solo tiene los de `02`, y su nota dice todavía que `02` es `b815be4d`, cuando hoy es `f9d1de51`. Además, `01_embed.py` (mtime 10:18:30) es posterior a 11 de los 13 npz (10:15–10:17).
   *Arreglo (MENOR):* antes de congelar, volver a correr `01_embed.py` (segundos, sin datos humanos) y comprobar que los npz no-MAE son idénticos a los actuales. Así los npz y el script congelado quedan consistentes. Actualizar también la nota de `rules_02`.

## MENORES
- `run_weekend.sh` no implementa el tope de 32 h de CPU (§9.2). Se vigila a mano, y hay que dejarlo dicho.
- `08_regions.py` decide G0 buscando la cadena "-> PASS" en el log. Funciona, pero es frágil. Es mejor que `--check` escriba un JSON y leer `ok`.

---

## Hashes revisados (sha256, scripts/)
| Archivo | sha256 |
|---|---|
| h4_prep_encoders.py | `030cf3e18bd5cbbcdcd04a1b7b5f4dac586af0a3992786c53d3cf96eac073451` |
| 05_twostage_stage1.R | `29a344990e724d542c4593f399f6932d095127abc286cdd5b2a6187264c3bc87` |
| lib_glmm.R | `794c957fd3a49791cb590fb291b77da133b0ddf7301fe4b616250dfb77924b61` |
| 06_stage2_permutation.py | `0639059dc73f9a8089f7af19807f5515a6ba9284776968325eabb9e352735da7` |
| 07_pooled_glmm.R | `cace5c8d8c0cb29e606b1604d10c9b45d5ac59b537e93ed51d36d0127b81d7b9` |
| 08_regions.py | `ae58eb6abffdb1654c21fb09098b35f08f2bae46cbe352f361de694c6f5a7c85` |
| 02_equivalence.py | `f9d1de51fe5ff3783be7a099af0c77a8e3e18c3012d62c79e734371978bf02c8` |
| 01b_embed_common.py | `d046a96901bb948c599312ec9ae3cdf9ebb1ee0a82b3e36aa3aba610323efd8c` |
| 09_freeze.py | `7a047303ea00a7aafa053d5bf174bfe5e1d8850faed65fb1ff6d3e7520f40f4b` |
| run_weekend.sh | `23e2df64c177db1a84405388223387a7cfd1f34c9361904814dad19ab93bd916` |
| 01_embed.py | `4d41764ee8f9067e9a0ccbd96aad784901f829ec629ae5f79dfa6af0dd3f6332` |
| README_weekend.md | `7e0ed800b1d97098ed377699322eb361d25a94c2b3de00f2632e520e4a3f8325` |

Si cualquiera de estos hashes cambia por BC3, I1 o I2, basta con que yo revise el diff antes de `09_freeze.py`. No hace falta una cuarta revisión completa.
