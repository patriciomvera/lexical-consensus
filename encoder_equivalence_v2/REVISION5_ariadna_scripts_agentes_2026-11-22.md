# REVISION5: scripts del anexo de agentes (encoders v2, 2611.14116)

- **Revisora:** Ariadna, 22-11-2026, ~15:15–16:00 (hora de Chile, UTC−3). Pedido de Juanito (15:13), actualizado a las 15:44.
- **Revisado (sha256 verificados):**
  - `ACLARACIONES_scripts.md`: `093aed37…5b10`, que incluye a13;
  - `test_embed_agents.py`: `41080f39…`;
  - scripts 10–15, `lib_agents.py`, `run_agents.sh`, README y los demás tests: sin cambios desde las 15:13;
  - propuesta: `633a5e7e…0db25`;
  - plan humano congelado: `cfaefee5…3fec4`.
- **Límites:** no modifiqué nada de Juanito, no corrí el pipeline real y no reconstruí nada.
- **Pruebas que corrí yo:** pasan las 6.
  - regions: 14/14;
  - controls;
  - freeze;
  - e2e sintético con `run_agents.sh primary`: ~26 min;
  - embed: paridad |Δ| = 0 en los 13 vectores con pesos reales.

## Veredicto: **VB con condiciones**
En el código no encuentro bloqueantes. REVISION4 está implementada en el código, no solo en la propuesta:
- **BC-A1:** `ga()` aplica la precedencia GA0 > GA0b > GA1 > GA2 > GA2b > GA3, y gana la primera que se cumple.
- **BC-A2:** "inconcluso (técnico)" saca al encoder de Holm y lleva a GA3. Si R0 queda técnico, el resultado es GA0. Si T no es evaluable, cuenta como técnico.
- **Tolerancia GPU→CPU** (`CONTROL2`): está textual y probada en 5 casos.
- **D-todos:** usa los 4.950 pares y es obligatoria. `--max-pairs` solo está permitido en sintético, con un assert.
- **SESOI:** tienen su propia constante, y el informe dice "NO comparables con los δ humanos".
- **Holm:** se aplica solo a GA1, con la familia C de 36 pruebas y la familia D de 6.
- **Plan humano:** no menciona agentes, y `15 --freeze` exige que `09 --verify` pase.

## Bloqueantes
No quedan bloqueantes técnicos. Quedan dos puntos de proceso:
1. **Aprobación de la recuperación (b1).** Me la informó Juanito ("Patricio aprobó, 15:16"); no la vi directamente. Evaristo o Patricio deben confirmarla en el registro. Mi visto bueno no reemplaza esa aprobación.
2. **Fecha.** P1 fijó el congelado a más tardar el 27 y el run el 28–29. Correr el 23 adelanta el calendario, y eso lo confirma Patricio. Técnicamente no hay problema: deja margen para repetir el run.

## b1: caché R0 reconstruido (`9744d7cf…68024d`)
- **Comparación con la v1.** El máx. |Δ| = 5e−5 que informa Juanito es aceptable. Es el redondeo de las tablas publicadas, que tienen 4 decimales, no un error. Pero con 4 decimales la prueba es débil.
- **Verificación a precisión completa (la hice yo).** Comparé contra `human_data_analyses/data/pairs_c10.csv` y `tiers_c10.csv`, que salieron del caché original sin redondear. En los 45 pares, máx. |Δ| = **8,9e−8**; en los 10 niveles, 5,6e−8. Es ruido fp32.
- **Qué cubre y qué no.** Los centroides de 200 imágenes no prueban cada embedding uno por uno. Aun así, una selección distinta de imágenes movería las distancias en órdenes de magnitud más que 1e−7. El control 1 cubre el resto.
- **Cachés r04 sin reconstruir: no afecta la comparabilidad.** `10_agent_embed` vuelve a sortear el pozo con la misma regla que r04 `full` (`default_rng(s)`, 200 por clase). Así R0 queda en la misma ruta que los demás encoders, con paridad |Δ| = 0. Es una sensibilidad opcional y descriptiva. Si se comparan con números r04 de la v1, se reporta como descripción.
- **Condición C1.** Registrar en el manifiesto o en DESVIACIONES:
  - el sha;
  - las versiones de torch y transformers;
  - que `perception.py` no fija la revisión y que hoy `main` coincide con `ed25f3a3`;
  - mi verificación a precisión completa.

## Otras condiciones antes de `--freeze` (baratas)
- **C2.** Rotular `DIFF_agentes_propuesta.patch` como obsoleto y no aplicado. Sigue sobre `4ff4b341` y dice "invariantes al orden".
- **C3. Arreglos de forma:**
  - en `14_agent_regions.py`, los comentarios de las constantes están corridos en uno respecto de ACLARACIONES: D_P b5→b6, A_AGG b7→b8, CONCORD b8→b9, GROUPS_OVER b9→b10, H4 b6→b7;
  - `run_agents.sh`, l.67: "R0 reuses the v1 r04 caches" quedó desactualizado;
  - `test_embed` debe imprimir SKIP si no encuentra pesos.
- **C4.** Agregar al informe de `14` una línea con §3.7: solo H1 y H2 se leen frente a las personas, y un GA2 en D significa "suficiencia del mecanismo".
- **C5.** Declarar el cómputo (ver abajo).

## b1–b18
| # | Postura | Motivo |
|---|---|---|
| b1 | **De acuerdo.** Técnicamente resuelto (ver arriba); C1 | Aprobación informada por Juanito |
| b2 | **De acuerdo** | Exigir c1, c2 y per_concept idénticos en 5 aprendices × 5.040 filas es fuerte, porque son precisiones discretas. La holgura de 1e−6 + 1e−12 absorbe el redondeo del sexto decimal. No fijar la semilla de `random`: obligaría a tocar el código de la v1. Basta reportarlo como distribución (C1 media por nivel frente a la v1 y frente a 1/3), sin exigirlo. Riesgo residual aceptable: un empate en k-means o k-NN daría GA0 |
| b3 | **Cambiar (menor)** | La mediana de d_perc debe salir de los 4.950 pares, como en la v1. Así R0 reproduce la letra de regla de la v1 sin costo extra. La versión con 100 pares queda descriptiva. Solo afecta `rule_centroid` y los conteos |
| b4 | De acuerdo con la lectura literal, más un rótulo | Agregar el rótulo descriptivo "Apoyo con IC bajo SESOI": con miles de pares, casi cualquier ρ > 0 es significativo |
| b5 | De acuerdo | El p mínimo del bootstrap (1e−4) queda bajo .05/36 |
| b6 | De acuerdo | Es la escala de la v1 que exige el control 2. Reportar si HC3 cambia la región |
| b7 | De acuerdo | |
| b8 | De acuerdo | Es conservador en ambos sentidos |
| b9 | De acuerdo | Es estricto, pero solo separa GA2 de GA2b. Opcional: ARI |
| b10 | De acuerdo | Con la calibración aprobada, DINOv2-B está en el grupo de R0, así que incluir a R0 no cambia el conteo |
| b11 | De acuerdo con la lectura literal | En la práctica es inalcanzable: si el control 1 pasa, la C1 de R0 es la de la v1 |
| b12 | De acuerdo | |
| b13 | De acuerdo | GA0 debe ir en el encabezado del informe |
| b14 | De acuerdo | El rótulo "sensible al muestreo" puede dispararse porque los p de D-todos son anticonservadores. Es descriptivo |
| b15 | De acuerdo | Lo comprobé: |Δ| = 0 en los 13 vectores |
| b16 | De acuerdo | La caída a `noslope` es por encoder, como en 04c. Reportar qué estructura de RE usó cada uno |
| b17 | De acuerdo | Coincide con la v1: `contrast_seed = 9000 + pair_id` |
| b18 | De acuerdo | |

## Cómputo y calendario
- **La propuesta sí declaró cómputo en §4.** Le faltaba un **tope**: el riesgo 9 lo nombra pero no lo define. Además, su l.198 ("pesos ya en caché") fue falsa hasta hoy.
- **Estimación corregida (3,5–7 h de reloj, 13–28 h-núcleo): plausible.**
  - Con 128 imágenes se mide bien el costo por imagen en régimen estable, porque la entrada es fija a 224 px y da lo mismo si es ruido.
  - No se miden la decodificación de CIFAR, la contención con 3 procesos ni H4.
  - Mi medición de un solo proceso da tasas 0,6–1× las de Juanito (E4 18,7 img/s, E3 31, O1 20).
  - El límite superior realista es ~8–9 h. Lo más incierto es H4 con datos reales: en la parte humana, S1 con 24 ítems tomó 26–55 min por encoder.
- **Riesgo de calendario:** bajo. El riesgo real es otro reinicio del box, porque se pierden `/tmp/renv` y `/tmp/XTMEM`. Conviene respaldar los npz y sus sha al terminar el paso 3.
- **Declaración mínima (C5):**
  - run primario: 13–28 h-núcleo y 3,5–7 h de reloj, con 3 procesos de ≤ 1,5 GB, CPU local, $0 y sin GPU ni API;
  - **tope: 14 h de reloj**;
  - confirmar la tasa de img/min en los primeros 15 min. Si es < 50 % de lo estimado, se registra y se re-estima;
  - si se supera el tope, se registra una desviación y se corta según el riesgo 9: primero las sensibilidades, después H4 por encoder;
  - **D-todos y el resto del primario no se cortan.**

## Otro hallazgo
- Las decisiones de (b) se tomaron después del run humano del 21–22, pero antes de cualquier resultado de agentes, y ninguna usa valores humanos. Declararlo en el congelado.
