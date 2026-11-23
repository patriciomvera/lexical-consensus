# Revisión independiente: diseño de comparación de encoders (v2 de arXiv 2611.14116)

- **Revisora:** Ariadna (CRO). **Fecha:** 18-11-2026, ~10:45 (hora de Chile, UTC−3).
- **Archivo revisado:** `propuesta_comparacion_encoders_v2_2026-11-18.md`, mtime 18-11 10:04:29 (UTC−3), sha256 `afb7e139…46845c1a`.
- **Tabla de equivalencias:** **no incluida todavía.** Revisé `encoder_equivalence/`: `01_embed.py` ya generó embeddings de 12 fuentes (10:15–10:17; falló `E4s_mae_cls`); `02_equivalence.py` tiene reglas fijadas con sha256 a las 10:16, pero aún no se ha ejecutado (no hay `equivalence_*.csv` y `figs/` está vacío).
- No modifiqué nada de Juanito ni ejecuté nada.

## Veredicto
**¿Listo para congelar? Sí, con cambios.** El diseño es honesto (declara la confusión orden~blocking, separa las familias A y B y tiene controles de piso y de tubería). Pero hoy su regla de decisión no puede cambiar de verdad la lectura, por dos razones: la inferencia ignora que el orden varía entre solo 12 experimentos, y las reglas de §5.3 no son excluyentes. Además, el criterio de geometría existe en dos versiones que no coinciden. Las tres cosas se arreglan en pocas horas, antes del 21-11.

---

## BLOQUEANTES

**B1. La inferencia de la familia B se hace al nivel equivocado** (§4.2 l.117, §5.1 l.134, §5.3 l.143–144; la sensibilidad (i) en l.121 "no decide").
`order` es una variable entre experimentos y hay 6 + 6 experimentos. Mb no tiene efecto de experimento, así que el EE de 0,041 viene de 172.800 observaciones y no de 12 unidades. Con esa precisión, casi cualquier encoder correlacionado con R0 va a "mantener" B, y la TOST ±0,10 también es fácil de cumplir. El resultado lo decide la seudoreplicación, no la geometría. El propio §7.2 lo reconoce, pero deja la corrección fuera de la regla.
*Arreglo:* que el IC que decide en §5.3 salga de un análisis a nivel experimento. Hay dos opciones:
- (a) Mb con `(1 | exp) + (0 + zd | exp)`;
- (b) más barata y transparente: en dos etapas. Se estima la pendiente de zd por experimento y por encoder (12 pendientes) y se compara "1-3" contra "3-1" con una permutación exacta de etiquetas (C(12,6) = 924) o con Welch.

El GLMM publicado queda como descriptivo. Hay que declarar de antemano que el resultado más probable es "inconcluso" y que ese es un resultado válido para la v2.

**B2. Las reglas de §5.3 no son excluyentes y la multiplicidad está invertida** (l.143–153, §5.5 l.169).
- Con θB1 = 0,05 e IC [0,01; 0,09], el encoder cumple a la vez "se mantiene" (IC excluye 0) y "no se mantiene" (TOST dentro de ±0,10).
- Holm se aplica a la afirmación conjunta ("se mantiene en los 5"), que es una prueba intersección-unión y no necesita corrección. En cambio, no se aplica a "depende del encoder" (≥ 1 de 5 falla), que es una unión y sí la necesita.

*Arreglo:* definir regiones excluyentes con precedencia. Por ejemplo:
1. "no se mantiene" = TOST o signo opuesto;
2. "se mantiene" = límite inferior > 0 **y** estimación ≥ 0,10;
3. "atenuado" = IC excluye 0 pero la estimación es < 0,10;
4. el resto, inconcluso.

Además, usar la conjunción sin Holm y aplicar Holm sobre los encoders en la regla de "depende".

**B3. Hay dos criterios de geometría distintos y ninguno mide lo que entra al modelo** (§5.4 l.160–166 frente al docstring de `02_equivalence.py`).
- El diseño dice: límite superior < .90 de la RSA intra-dominio, B = 2.000, bootstrap por imágenes y cálculo *después* de congelar.
- El script dice: agrupamiento UPGMA cortado en 0,5, el .90 solo como referencia, B = 1.000, bootstrap estratificado por dominio, y está listo para correr *antes* de congelar.
- Ninguno calcula el §5.4(iii), es decir, cuánto se parecen las d de ensayo. Esa es la cantidad que entra en el GLMM; la RSA de 45 imágenes es solo un sustituto.

*Arreglo:*
- Dejar una sola regla en el diseño y citar el sha256 de `02_equivalence.py`.
- Añadir la correlación de zd(e) con zd(R0) en los 8 ítems objetivo, parcializando `level`. No usa respuestas humanas.
- Ejecutar `02` solo después de congelar el conjunto de encoders y §5.3.

---

## IMPORTANTES

**I1. `level:order` no está en el modelo** (§1.4 l.36; `04f_h4_order_moderation.R` l.13).
Dentro del dominio objetivo, d y nivel van juntos (.33 / .58 / .87; §1.5.5). Si el orden modera el efecto del nivel, `zd:order` lo absorbe. Eso es justo "viene de la estructura de los ejemplos" disfrazado de geometría, y sería invariante entre encoders por construcción.
*Arreglo:* sensibilidad pre-especificada Mb + `level:order`, al menos con 8 ítems. Si θB se desploma, se lee como estructura del diseño. Cuesta unos 12 min por encoder.

**I2. El diseño apunta al término menos importante del paper** (main.tex l.226: "we do not interpret these terms further").
Lo que sí sostiene la lectura de la v2 está en el resumen (l.65): la asociación con la distancia "persisted within the taught domain after controlling for taxonomic level and teaching condition". El control C1 con 24 ítems (−3,23, dominado por el contraste entre dominios) es casi trivial como puerta.
*Arreglo:*
- Usar como control positivo por encoder el modelo intra-dominio que ya está en el paper.
- Reportarlo como estimando descriptivo.

Cuesta unos 10 min por encoder y no cambia el alcance. Que Patricio lo decida en §10.

**I3. La escala de zd no es comparable entre encoders con 24 ítems** (§4.2, §5.2 l.139).
"Por DE de d" depende de la razón entre y dentro de dominios, que varía según el encoder (`02` ya la calcula como `between_over_within`). Por eso el SESOI de ±0,10 no significa lo mismo para todos.
*Arreglo:* aplicar el SESOI solo a la réplica de 8 ítems, o a ambas pero reportando esa razón junto a θB.

**I4. Ventaja de casa y tamaño** (§3.4 l.103; §5.2).
- La tubería (CLS de la última capa, centroide, z sobre 24 ítems) y el SESOI (la mitad de 0,2041) se diseñaron con DINOv2. Es aceptable, pero hay que declararlo.
- R0 es ViT-S y casi todos los demás son ViT-B.

*Arreglo:* pasar O1 (DINOv2-B) al núcleo, por ~1 h. Es el único control de tamaño y además calibra el criterio de geometría: si O1 sale "distinto de R0", el criterio es demasiado laxo.

**I5. El estado del archivo está desactualizado** (encabezado l.3, §8.3 l.245).
Dice que solo se hizo la prueba de tiempos, pero existen embeddings de las 12 fuentes desde las 10:15–10:17. No son d ni usan respuestas humanas, así que no contaminan nada.
*Arreglo:* registrarlo en §8.3 con horas, incluido el fallo de `E4s`, antes del congelamiento.

---

## MENORES

- **M1. MAE.** `ViTMAEModel` baraja los parches aunque `mask_ratio` sea 0. La media pasó el chequeo de determinismo porque es invariante al orden, pero el CLS falló (`log_embed.txt` l.83). La sensibilidad de §7.8 queda inviable.
  *Arreglo:* pasar `noise` fijo, que `forward` acepta (verificado en transformers de `/workspace/rv_venv`, `modeling_vit_mae.py` l.631–668).
- **M2. Preprocesamiento.** Hay recorte central en R0, O1, O2, E3, F1 y F2, y no lo hay en R1, E1, E2 ni E4. E1 y E2 normalizan con 0,5 (`encoder_specs.json`). Está cubierto por (iv); basta con una columna en la tabla.
- **M3. Tolerancia del control de tubería** (l.116). `01_embed.py` codifica las 45 imágenes en un lote. Si la v1 lo hizo de otra forma, las diferencias de fp32 pueden superar 1e−6 y detener todo en falso.
  *Arreglo:* fijar ya una tolerancia de 1e−5 en d, o exigir el mismo orden de rangos.
- **M4. Familia de Holm** (l.169). Mezcla los pisos F1 y F2 con los encoders entrenados, aunque cumplen roles distintos.
  *Arreglo:* usar dos familias.
- **M5. F1** (§8.2 l.233). Falta decidir si el primario usa la semilla 0 o las 3 semillas.
- **M6. RAM** (§7.13). El log muestra entre 6,5 y 12,7 GB disponibles, no 2,7 GB. Conviene actualizarlo.
- **M7. Datos de entrenamiento.** Las categorías de los estímulos están en ImageNet y probablemente en WebLI y LAION. No es filtración de respuestas humanas y ya está nombrado en §7.10, así que no requiere más.
- **Orden de ejecución, semillas y hardware.** No veo confusión: el cálculo es determinista, todo corre en CPU y todas las fuentes usan los mismos datos. La única alineación grave es orden~blocking~experimento, que se trata en B1.

---

## Qué debe contener la tabla de equivalencias
1. **Por encoder:**
   - checkpoint y sha;
   - objetivo y datos de entrenamiento;
   - arquitectura y parámetros *solo de visión* (hoy E1, O2 y E4 cuentan torre de texto o decoder);
   - dimensión, capa, pooling y preprocesamiento (resize, recorte, interpolación y normalización);
   - métrica;
   - resultado del chequeo de determinismo.
2. **Geometría frente a R0:**
   - RSA completa e intra-dominio con IC;
   - CKA;
   - razón entre y dentro de dominios;
   - **correlación de zd de ensayo en los 8 ítems, parcializando nivel** (B3).
3. **Fallidos y excluidos** (E4s) con su motivo, no omitidos.
4. **Calibración:** O1 debe salir "equivalente" y F1/F2 "distintos". Si no, el criterio no discrimina.
