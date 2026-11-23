# Segunda revisión: plan final de comparación de encoders (v2 de 2611.14116)

- **Revisora:** Ariadna (CRO). **Fecha:** 18-11-2026, ~10:50 (hora de Chile, UTC−3).
- **Versión revisada:** `propuesta_comparacion_encoders_v2_2026-11-18.md`, mtime 18-11 10:39:24 (UTC−3), sha256 `e5d2806aa3fede96789b28af912bf6334a138ebe2104d056004fb82711132f4c`. Ya incluye los tres márgenes. La comparé contra `encoder_equivalence/propuesta_v1004_snapshot.md`.
- **Artefactos:** `02_equivalence.py` (sha `b815be4d…`, igual a §6.1), `rules_02…txt`, `vision_params_and_checks.json`, npz y specs. No modifiqué ni ejecuté nada.

## Veredicto
**¿Listo para congelar? Sí, con un cambio de una línea (BC1) y después de revisar los scripts de análisis.**
B1, B2 y B3 quedaron resueltos. Lo nuevo no rompe la lógica: la permutación es compatible con Holm, las regiones son excluyentes dentro de cada δ y cada resultado tiene una lectura declarada de antemano.
Único hueco: el encoder que falla el control positivo.

## Estado de B1–B3
- **B1, resuelto (§5.2, §2 l.85).** El análisis es en dos etapas, a nivel experimento, con permutación exacta sobre las 924 asignaciones. El GLMM publicado queda como S1, la pendiente aleatoria por experimento como S2, y "inconcluso" se declara como el resultado más probable.
- **B2, resuelto (§7.1–§7.3).** Las regiones tienen precedencia y por eso son excluyentes. La conjunción va sin Holm, la unión con Holm por δ, y entrenados y pisos son familias separadas.
- **B3, resuelto (§6.1).** Hay una sola regla: r de zd entre encoders, residualizado por nivel, sobre las 57.600 filas de 8 ítems. El script no lee `selected` (l.60 de `02`), la calibración está pre-especificada, el script viejo nunca se corrió y no hay salidas de `02` en `out/`.

## Comprobaciones pedidas
- **p mínimo frente a Holm.** Unilateral 1/924 = .0011 y bilateral 2/924 = .0022, frente al primer umbral de Holm de .05/14 = .0036 (entrenados) y .05/4 = .0125 (pisos). El rechazo es alcanzable, aunque solo con efectos grandes (ya declarado, §8.2).
- **Exclusividad entre δ.** La precedencia la garantiza dentro de cada δ, y la regla de perfil (§7.2) cubre cualquier combinación entre δ. No queda un resultado ambiguo, salvo BC1.
- **Calibración fallida.** No hay lectura por grupos y G3 pasa a G3b (l.202, l.270). Es coherente (ver m2).
- **MAE y tubería.** Con `noise` fijo, dos pasadas idénticas (npz de las 10:18). R0 frente a la v1: 3,6e−7 en d por pares. Tolerancia de 1e−5, razonable.
- **Artefactos previos.** Coinciden con §9.3.

---

## BLOQUEANTE

**BC1. El destino de un encoder que falla el control positivo no está definido** (§5.3 l.180, §7.3 l.257, §7.4 l.268–270).
§5.3 dice "su familia B no se interpreta", pero:
- G3 y G3b exigen que "los 8 entrenados" estén en "se mantiene", sin decir qué pasa si uno quedó fuera por el control;
- no se dice si ese encoder sigue en la familia Holm de 14 pruebas;
- no se dice qué pasa si falla R0, aunque es improbable.

Así, el mismo resultado puede leerse como G3 sobre 7 encoders o como G4.
*Arreglo:*
- crear la región por encoder "no interpretable (control positivo)", que no depende de δ;
- sacar ese encoder de la familia Holm, que pasa a tener 2·(k) pruebas;
- G3 y G3b solo si **todos** los entrenados pasan el control; si alguno falla, el resultado es G4;
- si R0 falla, se detiene todo como en G0.

## IMPORTANTES

**I-a. El nombre de la región 2 induce a error** (§7.1 l.228).
Como la región 2 tiene precedencia sobre la 3 y la 4, un Δ con IC 95 % > 0 pero IC 90 % dentro de ±δ se rotula "equivalente a 0". Con δ = 0,20, eso puede pasarle a un efecto claramente positivo, incluso a R0.
*Arreglo:* rotularla "menor que δ (equivalente con margen δ)" y no "equivalente a 0". La lógica queda igual.

**I-b. El ancla de los δ está en otra escala** (§7.1 l.225 y la frase de G1 bajo §7.4).
El 0,2041 es `zd:order3-1` del Mb con 24 ítems (z sobre 24 ítems, GLMM agrupado). Los δ se aplican al Δ de dos etapas con 8 ítems (z sobre 8 ítems, contrastes de suma, promedio de pendientes por experimento). Por eso "δ = 0,20 es cerca de la magnitud completa de la v1" no está verificado. No discuto los márgenes, que ya decidió Patricio.
*Arreglo:* quitar esa equivalencia y reportar el Δ de R0 en dos etapas como referencia descriptiva (no confirmatoria).

## MENORES
- **m1. Registro de cambios.** l.19 dice "10:25–11:00", una hora futura cuando se fijó el hash. Además, l.14 dice "reemplaza a los márgenes" y debería decir "márgenes de geometría", como en l.387.
- **m2. Calibración fallida.** G2 (l.268) "describe en qué grupos" están los que fallan. Cuando la calibración falla, debería decir "sin grupos".
- **m3.** l.176: nombrar que las 56 personas que repiten rompen en parte la intercambiabilidad (S4 lo cubre).

---

## Scripts de análisis: revisar antes de congelar
`h4_prep_encoders.py`, `05_twostage_stage1.R`, `06_stage2_permutation.py`, `07_pooled_glmm.R` y `08_regions.py` todavía no existen. **El congelamiento debe incluir su hash y mi revisión de esos scripts** (§10.2, paso 3). Qué revisaré:

1. **Prep:** mismas filas y orden que `h4_long.csv`; npz leídos por nombre de imagen; G0 (|Δ| ≤ 1e−5 y rangos); z por encoder y conjunto, sin `selected`.
2. **Etapa 1:** `contr.sum` efectivo y las 4 condiciones en cada experimento; fallback y "inconcluso (técnico)" registrados; se guarda el β̂ de `zd`.
3. **Etapa 2:** enumeración exacta de las 924 asignaciones; signo 3-1 menos 1-3; p con ≥; IC por inversión con una grilla que lo cubra; TOST como máximo de los dos p desplazados. Test sintético con Δ conocido.
4. **Regiones:** precedencia 1→5 por δ; familias Holm; G0–G4 en orden; perfiles; ninguna opción para elegir δ o encoders. Tests unitarios con los dos ejemplos de §7.1 y con un encoder que falla el control.
5. **`07`:** fórmulas idénticas a §5.3, §5.4 y 04f.
6. **Congelamiento:** todos los hashes en `FREEZE_encoders_v2.txt`.
