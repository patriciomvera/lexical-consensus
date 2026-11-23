# Visto bueno: parte humana de la comparación de encoders (v2 de 2611.14116)

- **Revisora:** Ariadna. **Fecha:** 18-11-2026, ~13:20 (hora de Chile, UTC−3).

## Veredicto
**Visto bueno para congelar la parte humana con el plan `cfaefee5…`.** Los agentes no entran: son una adenda aparte, que reviso antes del 27-11.

## Hashes verificados
| Archivo | sha256 |
|---|---|
| Plan (`propuesta_comparacion_encoders_v2_2026-11-18.md`) | `cfaefee5ab8ebfa764c65b6c3a28ae9d0eb23ddc43f4746c6eeb99ed9793fec4` |
| `DIFF_REVISION3b.patch` | `2ce17f301e84b5609aa31c11a72f154dab8c64d0ce4ab7e16bab34706e541bba` |
| Plan base (`propuesta_4ff4b341_snapshot.md`) | `4ff4b341…e3f7` |

El diff de los scripts regenerado contra `scripts_snapshot_4ff4b341/` coincide línea a línea con el patch. En el plan solo cambian el registro de cambios, §5.1, §5.2, §7.4 y §11.5. No hay pasos de agentes ni en el plan ni en `run_weekend.sh`.

## Verificaciones
1. **R0 "inconcluso (técnico)" da G0 "sin lectura: técnico"** en las 9 celdas, antes de G1 (`conclude`). Con 24 ítems da la lectura "sin lectura: técnico". Las pruebas pasan.
2. **G0 falla con un FAIL explícito si un grupo de casi-empates mide ≥ 1e−6.** En la v1, el grupo más ancho mide 6,0e−8. Mover el ruido de 1e−8 al archivo nuevo es lo correcto, porque el de la v1 está fijado por su sha256. Hay una prueba de encadenamiento (1,6e−6 da FAIL).
3. **Sin estructura común, `05b` sale con código 4** y escribe `DEVIATION`. `run_weekend.sh` se detiene; `rung8` falla sin caer a L4, y en S6 la falla no bloquea el resto. Quedó corregido que un conjunto vacío devolviera L1.

**Pruebas sintéticas:** `test_stage2` 5/5, `test_regions` 20/20, `test_select_structure` 8/8 y `test_prep` OK con todos los casos. No ejecuté `09_freeze.py` ni `run_weekend.sh`.

## Notas residuales (no bloquean)
- En los datos sintéticos, el grupo más ancho mide 9,87e−7, cerca del umbral. Con datos reales no afecta (6e−8).
- El congelamiento debe incluir en `FREEZE_encoders_v2.txt` los sha256 de estos scripts y de esta nota.
