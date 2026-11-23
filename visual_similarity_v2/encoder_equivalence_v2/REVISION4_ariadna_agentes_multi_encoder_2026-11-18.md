# REVISION4: propuesta de agentes con múltiples encoders (v2 de 2611.14116)

- **Revisora:** Ariadna. **Fecha:** 18-11-2026, ~13:15 (hora de Chile, UTC−3).
- **Revisé:** `propuesta_agentes_multi_encoder_v2_2026-11-18.md` (sha256 `bfc99face128…f888e8`, mtime 12:43) y `DIFF_agentes_propuesta.patch` (`79ed554c…12f0`). Patricio todavía no aprueba la propuesta.
- **Fuentes:** código v1 en `revision_runs/repo/`, `paper_arxiv_v_concl/main.tex` (= 2611.14116; `arxiv_src/` es *Lexical Consensus*, 2606.22207) y `perceptual_distance_matrix.csv` (solo para simular ruido). Sin datos de respuestas; no modifiqué nada de Juanito.

## Veredicto
**Se puede incorporar con cambios.** Hay **2 BLOQUEANTES**, ambos de redacción de reglas y baratos de arreglar: GA sin un equivalente de G1, y la falla técnica sin destino definido.
**Recomiendo la opción P1 de respaldo:** la parte humana se congela y corre el 21-22 sin esperar a los agentes, y los agentes van como adenda propia el 28-29. Así mi visto bueno para la parte humana deja de depender de esta propuesta.

---

## Preguntas de Juanito
**(1) No hay orden en los agentes: correcto, con una precisión.** En `run_clean.py`, l.416–422, cada aprendiz se crea de nuevo y se ajusta en lote **dentro de cada episodio**; no arrastra estado entre episodios. Por eso no existe un análogo del orden entre ensayos de 1 y 3 ejemplos, y no hay familia B.
- "Invariantes por construcción" es demasiado fuerte (`KMeans` y `LinearSVC` pueden variar numéricamente con el orden de filas); basta "sin estado entre episodios".
- Leer la comparación agentes-personas sobre distancia y alineación con distancias humanas (H1, H2) es lo correcto. (2611.14116 excluye el consenso multi-agente, main.tex l.232.)

**(2) Familias y GA0–GA3:** las veo en BC-A1, BC-A2 e I-A1.

**(3) Espectro 0,10 / 0,20 / 0,30: aceptable** (escala de ρ y r parcial, centrado en el SESOI de la v1). **No es comparable con los δ humanos** (log-odds por DE): decirlo y no tabularlos como equivalentes.
*MENOR:* con n = 32 en H2, la mitad del ancho del IC del r parcial es ~0,35, así que "No apoyo" es casi inalcanzable aun con 0,30. Hay que declararlo como expectativa.

**(4) Reusar los grupos de geometría humanos: aceptable solo como descripción.** Los 45 estímulos de Lewis & Frank son fotos de 720 px. CIFAR son imágenes de 32×32 escaladas, y la robustez a baja resolución cambia de un encoder a otro (riesgo 4 de la propuesta).
*Arreglo:* rotular "grupos medidos en estímulos L&F" y reportar al lado la ρ entre encoders en CIFAR; si discrepan, GA2/GA2b se informa "sin grupos".

**(5) Tolerancia GPU→CPU, mi propuesta.** La v1 corrió en una T4 (`modal_app.py` l.45). La T4 es arquitectura Turing y **no tiene TF32** (fuente: NVIDIA), así que el fp32 de la v1 era fp32 real. Pero las versiones de `torch` y `transformers` en Modal **no estaban fijadas** (`pip_install("torch", "transformers")`), y el procesador de imágenes pudo cambiar.
Medí: en la matriz v1, **3.301 de 4.949 brechas consecutivas son < 1e−4** (el rango exacto no sirve); con ruido de DE 1e−4 la Spearman mínima es 0,999999 y con 1e−3, 0,99994 (o sea, "ρ ≥ 0,9999" no discrimina). Referencia CPU: 1,5e−8 en embeddings y 3,6e−7 en d.

**Propuesta:**
- **PASS** si max|Δ| ≤ **1e−4** y Spearman ≥ **0,99999**.
- **PASS con desviación** si 1e−4 < max|Δ| ≤ **2e−3**, Spearman ≥ 0,9999 **y** β_perc de D con R0 a ±0,005 de la v1 (se registran versiones y causa).
- **GA0 FAIL** en cualquier otro caso.
- El control exacto de T con el cache de CIFAR-10 de la v1 sigue siendo el control principal del código.

**(6) Pares de exp_008 elegidos por cuadrantes de DINOv2: hay ventaja de casa** (ver I-A2).

---

## BLOQUEANTES

**BC-A1. GA no tiene un equivalente de G1** (§3.4).
GA2 ("no es específico de DINOv2") y GA1 ("depende del encoder") suponen que R0 muestra el efecto. En la v1, **H2 fue inconclusa con R0** (`RESULTADOS.md` l.16–17), así que, para P/H2, esas dos lecturas no tienen sentido.
*Arreglo:* agregar **GA0b**: "R0 no está en Apoyo en ese análisis (con ese SESOI): la comparación entre encoders es descriptiva", y que se aplique antes que GA1. Además, declarar la precedencia GA0 → GA0b → GA1 → GA2 → GA2b → GA3, con "la primera que se cumpla".

**BC-A2. La falla técnica no tiene destino definido** (es la misma lección que en REVISION3b).
Falta decir qué pasa con un encoder cuyos embeddings o simulaciones fallan, o cuya regresión no se puede estimar.
*Arreglo:*
- región "inconcluso (técnico)": sale de Holm, bloquea GA2 y lleva a GA3;
- **si R0 queda técnico**, el resultado es GA0 ("sin lectura"), no GA1 ni GA3.

## IMPORTANTES

**I-A1. La región "Apoyo" se define con Holm, pero GA2 dice "sin Holm"** (§3.2 l.85 frente a §3.4 GA2).
*Arreglo:* definir Apoyo, No apoyo e Inconcluso por encoder **sin ajuste**, y usar Holm solo en GA1 (la unión). Es lo mismo que en §7.3 del plan humano.
H4 está en la familia C pero no tiene una lectura GA. Si se mantiene ahí, como réplica del Holm de 6 de la v1, eso hay que decirlo; si no, la familia queda de 30.

**I-A2. D depende de la estratificación hecha con R0.**
Los 100 pares son ortogonales en d_perc × d_sem **solo para DINOv2**. Con otros encoders, d_perc puede correlacionar con d_sem en esos pares, lo que trae colinealidad, menos potencia y un sesgo hacia "depende del encoder".
*Arreglo:*
(a) reportar antes de correr la DE de d_perc y su correlación con d_sem por encoder en los 100 pares; (b) **D-todos** (4.950 pares, ~10 min por encoder) como **sensibilidad obligatoria**, no la primera en cortarse; (c) si D-todos y D-100 discrepan, perfil "sensible al muestreo de pares". No recomiendo reponderar.

**I-A3. Circularidad** (riesgo 6).
En D y T, el aprendiz y d_perc usan el mismo espacio, así que el gradiente con la distancia es casi mecánico con *cualquier* encoder. Un GA2 en D se lee como "suficiencia del mecanismo", no como evidencia sobre las personas. Solo H1 y H2 (distancias humanas) no son circulares.
El control positivo (C1 > 1/3) es laxo; aceptable como puerta, declarándolo.

## MENORES
- Reescribir §0 y §12.2 con "sin estado entre episodios" (ver la pregunta 1).
- La Spearman de n = 10 entre encoders: solo descriptiva, sin p.
- El parche se hizo sobre `4ff4b341`. Hay que revisarlo de nuevo cuando lo rebasen sobre la versión con REVISION3b.
