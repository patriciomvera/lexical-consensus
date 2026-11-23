# REVISION3b: revisión del diff (no es una revisión completa)

- **Revisora:** Ariadna. **Fecha:** 18-11-2026, ~12:55 (hora de Chile, UTC−3).
- **Hashes verificados:** `DIFF_REVISION3.patch` = `baf2f503…e9bd` y plan = `4ff4b341a934fcd8cacec5ae9303909781e0b7abca0e45e4a7d34b54d256e3f7`. Ambos coinciden.
- **Pruebas sintéticas en `/tmp`:** `test_stage2` 5/5, `test_regions` 18/18, `test_select_structure` 5/5 y `test_prep` OK (con los 4 casos de BC3).
- **Datos reales:** de `h4_long.csv` leí solo la columna `dist`. No ejecuté `09_freeze.py`, `test_freeze` ni `run_weekend.sh`, y no modifiqué nada de Juanito.

## Veredicto
**El diff es correcto e implementa las decisiones de Patricio de las 12:12 y 12:14. Hay un arreglo IMPORTANTE (R0 técnico), que es de una línea.**
**El visto bueno para `09_freeze.py` queda condicionado a revisar la propuesta de simulaciones de agentes que Patricio pidió a las 12:36**, que aún no existe.

## Puntos
- **BC3 (§3.2 l.162): aceptable, y mejor que mi redondeo.** Sobre `dist` de la v1:
  - 288 grupos; el **grupo más ancho mide 6,0e−8** y tiene como máximo 3 valores, así que **no hay riesgo de encadenamiento**;
  - **la brecha mínima entre grupos es 4,58e−6**, frente a diferencias esperadas de ≤ 3,6e−7, así que un intercambio real se sigue detectando (la prueba con 5e−6 falla, como debe).

  *MENOR:* agregar un `assert` de que el ancho máximo de grupo sea < 1e−6, como protección contra el encadenamiento en el futuro.
- **I2:** verificado en `run_weekend.sh`. `( s6 ) || log …` va al final del domingo y no detiene nada. Acepto la prueba de humo que informa Juanito (coseno mínimo de R0 0,9999953; F1 y F2 idénticos); no la repetí.
- **(6):** verificado.
  - los 13 npz son idénticos byte a byte a los de `npz_before_rerun/` (`cmp`);
  - el diff de `01_embed.py` solo agrega `crop_pct`;
  - `rules_02` está actualizado con `01_embed.py` = `3ed060c1…` y `02` = `f9d1de51…`.
- **I1:** implementado como lo decidió Patricio.
  - La escalera L1→L4 y el criterio de convergencia (que incluye que un ajuste singular no converge) están fijos en §5.2 (l.193–206) y en `LADDER`/`LADDER_CTL`.
  - `STRUCTURE_SCOPE = "common"` es una constante, sin opción en tiempo de ejecución.
  - La elección se hace por separado para 8 y para 24 ítems, incluye los pisos y queda fija el sábado para S4–S8.
  - `05b` elige usando solo `converged` (`richest`/`status`), nunca β, Δ ni p, y `test_selection_ignores_beta` lo comprueba. La convergencia depende inevitablemente de las respuestas, pero no del contraste por orden.
  - Excluir F0 está justificado: no tiene `zd` ni etapa 1, así que no hay nada que elegir.
  - `per_encoder_rung` y S9 quedan como descriptivos.

## IMPORTANTE
**R0 "inconcluso (técnico)" se leería como G1** (`08_regions.py::conclude`: `regs[R0] != R3_` → G1).
Si R0 queda sin peldaño completo, sale "la moderación de la v1 no se sostiene", pero en realidad sería una falla técnica.
*Arreglo:* si R0 queda en R5T, el resultado es G0 ("sin lectura: técnico"), con una prueba unitaria.

## MENOR
`rung8()` usa `L4` por defecto si no hay peldaño común. Hay que declararlo en §5.2, o mejor detenerse y registrar la desviación.

## Pendiente para el visto bueno final
La propuesta de simulaciones de agentes con cada encoder comparadas con las personas (Patricio, 12:36) cambia el alcance y entra en el mismo congelamiento. Necesito revisar su diseño, la comparación con las personas, su costo y sus scripts antes de `09_freeze.py`.
