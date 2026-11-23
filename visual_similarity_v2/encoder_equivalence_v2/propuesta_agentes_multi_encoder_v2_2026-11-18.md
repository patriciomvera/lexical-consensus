# Propuesta: agentes y personas con múltiples encoders (v2 de arXiv 2611.14116)

- **Estado:** PROPUESTA PARA DECISIÓN DE PATRICIO. **No está congelada, no se ha agregado al plan principal y no se ha ejecutado nada** salvo una prueba de tiempos de ~1 min (§4.1). No se leyó ninguna respuesta de participantes ni se calculó ninguna precisión, distancia o correlación con encoders nuevos.
- **Origen:** aclaración de Patricio (18-11 12:36, vía Evaristo): la v2 debe **repetir lo que hizo la v1 con múltiples encoders, comparando agentes y personas**; las simulaciones de los agentes se rehacen con cada encoder y se comparan con las personas, igual que en la v1.
- **Relación con el plan principal:** `propuesta_comparacion_encoders_v2_2026-11-18.md` (sha256 de la versión leída: `4ff4b341a934fcd8cacec5ae9303909781e0b7abca0e45e4a7d34b54d256e3f7`, 18-11 12:36). Hoy cubre solo personas (12 experimentos de Lewis & Frank) y deja "repetir H6 (el aprendiz en CIFAR) con los encoders nuevos" como fuera de alcance (§8, línea 359). El cambio que implicaría está en `encoder_equivalence/DIFF_agentes_propuesta.patch`.
- **Alcance:** solo diseño. Sin título ni encuadre.
- **Versión anterior:** `propuesta_agentes_bfc99fac_snapshot.md` (sha256 `bfc99face128a5b202b92c29b3c6e33b0adc8da83dcbf2539ddc930394f888e8`, 18-11 12:43). Es la que revisó Ariadna en `REVISION4_ariadna_agentes_multi_encoder_2026-11-18.md`.
- **El patch `encoder_equivalence/DIFF_agentes_propuesta.patch` (79ed554c…) todavía refleja la versión bfc99fac** y está hecho sobre el plan `4ff4b341…`. Se rehará sobre la versión del plan con REVISION3b cuando otro trabajador termine de aplicarla. No se rehízo ahora a propósito.

## Registro de cambios (hora de Santiago, UTC−3)
- **18-11 12:43.** Primera versión (bfc99fac).
- **18-11 12:50–12:52. Respuesta a REVISION4** (Ariadna). Se cambió solo este archivo; el plan principal, los scripts y el patch no se tocaron, y no se corrió nada:
  - **BC-A1:** se agrega GA0b y la precedencia explícita GA0 > GA0b > GA1 > GA2 > GA2b > GA3, donde gana la primera regla que se cumpla (§3.4).
  - **BC-A2:** se agrega la región "inconcluso (técnico)". Un encoder ahí sale de Holm y lleva a GA3; si es R0, el resultado es GA0 "sin lectura" (§3.2, §3.4).
  - **(5):** la tolerancia GPU→CPU se adopta tal como la escribió Ariadna. El control principal sigue siendo reproducir T desde el cache de CIFAR-10 (§3.5).
  - **(1):** "sin estado entre episodios" reemplaza a "invariante al orden"; k-means y la SVM pueden variar con el orden de las filas (§0, §2).
  - **(2) / I-A1:** las regiones por encoder se definen sin ajuste y Holm se usa solo en GA1. H4 se declara como réplica de la v1 (§3.1, §3.2).
  - **(3):** los márgenes de agentes están en escala de correlación y van en una tabla separada de la humana. Se declara que con 32 pares "No apoyo" en H2 es casi inalcanzable (§3.2).
  - **(4):** la geometría es solo descriptiva, con la fuente rotulada como estímulos de Lewis & Frank. Se agrega la ρ entre encoders en CIFAR y, si discrepan, "sin grupos" (§3.3).
  - **(6) / I-A2:** antes de correr se reporta la dispersión de d_perc (y su correlación con d_sem) por encoder en los 100 pares. D-todos (4.950 pares) pasa a ser obligatoria, con el rótulo "sensible al muestreo de pares" si discrepa. No se repondera (§2, §3.6).
  - **I-A3:** se escribe que la circularidad deja solo a H1 y H2 para leer frente a las personas (§3.7).
  - **Menores:** la Spearman de n = 10 es solo descriptiva y sin p; el control positivo es laxo y se declara así.
  - **Costo** (§4.2) y **decisiones P1–P8** (§7) actualizados.

---

## 0. Hecho de partida que condiciona todo: los agentes no tienen "orden"

- Los aprendices de la v1 son **un encoder congelado + una regla de aprendizaje sin entrenamiento de pesos del encoder**: prototipo (`src/learners/centroid.py:35-47`, media normalizada de los 5 ejemplos), k-NN (k = 3), k-centroides, regresión logística y SVM lineal ajustados en lote, y un azar de referencia (`experiments/exp_007b_extension_clean/run_clean.py:73-76`). No hay modelos de lenguaje ni llamadas a API.
- La percepción es `src/agents/perception.py` (DINOv2-S, CLS de la última capa, L2; líneas 37-39 y 100-106). Es la misma función que calculó d para las personas (plan principal §1.3, línea 61).
- **Los aprendices no tienen estado entre episodios:** en cada episodio se crea un aprendiz nuevo y se ajusta en lote con los 5 ejemplos (`run_clean.py` l. 416-422). Por eso no existe un análogo del orden entre los ensayos de 1 y 3 ejemplos de las personas, y **la familia B del plan principal (orden × distancia) no tiene contraparte en los agentes**. Estudiarlo exigiría un aprendiz secuencial nuevo, que no existe en la v1 y no es "repetir la v1".
- *Precisión (REVISION4):* no se afirma que los aprendices sean invariantes al orden. k-means (k-centroides) y la SVM lineal pueden variar numéricamente con el orden de las filas dentro de un episodio. Eso es ruido numérico, no un efecto de orden.
- Lo que la v1 sí comparó entre agentes y personas es **el efecto de la distancia visual** (gradiente en agentes; asociación distancia-extensión en personas) y **la alineación del espacio del encoder con distancias humanas** (H1, H2). Eso es lo que se repite por encoder.

---

## 1. Inventario de la v1: análisis de agentes y comparaciones agentes-personas

| Id v1 | Qué es | Datos e imágenes | Resultado v1 (fuente) | Estatus v1 |
|---|---|---|---|---|
| **T (tiers)** exp_007b_extension_clean | Precisión de nombrado (C1) de palabras nativas y disyuntivas cerca/medio/lejos | CIFAR-10, pozo de 2.000 imágenes (200/clase), 3-way 5-shot, 30 episodios, semilla 42; 6 aprendices; 7 tipos de pozo (C2) | Prototipo: 0,943 → 0,847 → 0,654 → 0,530 (README del repo; `paper_arxiv_v_concl/main.tex` l. 174); mismo orden en todos los aprendices no aleatorios | Descriptivo (IC bootstrap por episodio) |
| **T-sem** semillas maestras | Replicación de T con 5 semillas (0, 1, 2, 3, 42), modos `episodes` y `full` (re-sortea el pozo y re-codifica) | idem | "replicated across five master seeds" (`main.tex` l. 174; `revision_runs/out/r04_seeds/`, `scripts/r04_run_clean_seed.py`) | Robustez |
| **D (disociación)** exp_008 | C1 de 100 pares de CIFAR-100 ~ z(d_perc) + z(d_sem) (OLS, Modelo PS) + reglas A/B/C | CIFAR-100 train 50.000 (centroides con las 500 imágenes por clase; pozo 200/clase, semilla 42), 100 pares muestreados 25 por cuadrante (mediana de d_perc DINOv2 × d_sem), 30 episodios 3-way 5-shot; aprendices centroid, k-NN3, logreg, azar | β_perc = −0,212, R² parcial 0,245, p = 1,87e−7; β_sem p = 0,660; Regla A (README del repo; `main.tex` l. 170, 174; `exp_008_preregistration.md` l. 47-62) | Confirmatorio interno (reglas previas del repo) |
| **H1** | ρ Spearman(d_perc, d_humana) + Mantel; THINGS, HSJ, CIFAR-10H | Distancias de centroides en CIFAR-100 y CIFAR-10 | ρ = .523 / .501 / .795, todos Apoyo (`RESULTADOS.md` l. 13-15; `main.tex` l. 200) | Confirmatorio (Holm de 6) |
| **H2** | acc_pair ~ z(d_humana) + z(d_sem), HC3; THINGS y HSJ | Los 100 pares de D cubiertos (n = 61 y 32) | Ambos inconclusos (`RESULTADOS.md` l. 16-17; `main.tex` l. 209) | Confirmatorio (Holm de 6) |
| **H2b** | acc_pair ~ z(d_perc) + z(d_humana) + z(d_sem) | idem | β_perc se mantiene (`RESULTADOS.md` l. 35) | Descriptivo |
| **H3** | ρ(acc_tier, d) por tier y complemento por imagen (entropía CIFAR-10H vs margen del aprendiz) | 10 conceptos de T; 10.000 imágenes de test de CIFAR-10 | ρ_tier = −.770 con d_perc; ρ imagen = .391 (`RESULTADOS.md` l. 37-47) | Descriptivo |
| **H4** (personas) | GLMM selección ~ z(dist) + level + condition + (1 + z(dist) \| sujeto) + (1 \| imagen) | Lewis & Frank, 172.800 obs. | −3,23 [−3,29; −3,17], Apoyo (`RESULTADOS.md` l. 19) | Confirmatorio (Holm de 6) |
| **H5** | Coherencia de clase vs edad de adquisición | CIFAR-100 (submuestra 100/clase) y CIFAR-10 | Sin veredicto (`human_data_analyses/RESULTADOS.md` l. 20, 83-84) | Exploratorio (BH) |
| **H6** | Repetir D (modelo PS), T, H1 con CVCL | CVCL sobre CIFAR-100 train y pozo CIFAR-10 | Tiers reaparecen; β_perc_CVCL = +0,0047, p = .688: **Inconsistente** (`human_data_analyses/RESULTADOS.md` l. 21; `out/h6_results.json`) | Exploratorio (BH) |
| **Fig. x1** | Personas (selección vs d) junto al aprendiz (C1 vs d entre categorías) | — | `main.tex` l. 157 | Descriptivo |

**Consecuencia clave:** con DINOv2-S (R0) **y con CVCL (R1)** los resultados de agentes ya se conocen (v1 y H6). Ninguno de los dos puede ser confirmatorio en la v2.

---

## 2. Los mismos análisis por encoder

**Encoders:** los mismos 10 primarios del plan principal (R0, O1, R1, E1, O2, E2, E3, E4, F1 semilla 0, F2) con las mismas representaciones y preprocesamiento (`encoder_specs.json`), más E4s y F1 semillas 1-2 como sensibilidad. F0 (sin distancia) no aplica a agentes.

**Imágenes a codificar por encoder** (ninguna es dato de participantes):
- CIFAR-100 train, 50.000 (D, H1 y H2 en CIFAR-100, H5). Para R1 ya existe (`/workspace/hda_cache/cvcl_cifar100_train.npz`).
- Pozo CIFAR-10, 2.000 (T, H1 y H3 en CIFAR-10). Para R0 y R1 ya existe (`.cache/embeddings_exp007/cifar_all10.npz`, `hda_cache/cvcl_cifar_all10.npz`).
- CIFAR-10 test, 10.000 (solo el complemento por imagen de H3).
- Pozos re-sorteados de T-sem modo `full`: 4 × 2.000 (sensibilidad).
- **Los embeddings de las 45 imágenes de `encoder_equivalence/out/emb_*.npz` no sirven para los agentes:** son estímulos de Lewis & Frank, no CIFAR.
- **R0 en CIFAR-100:** los embeddings DINOv2 de la v1 se calcularon en GPU (Modal; `exp_008_dissociation/modal_app.py`) y no están en el box; solo está la matriz de distancias (`results/exp_008_20260608_173255/phase1/perceptual_distance_matrix.csv`). Se re-codifican en CPU solo como control de tubería (§3.5).

**Cómo se cambia el encoder (sin tocar el código de la v1):** igual que H6. `human_data_analyses/scripts/h6_tiers.py` reemplaza `load_or_build_embeddings` de `run_clean.py` por un lector de npz; `h6_c100.py` importa `exp_008/src/phase3/run_episodes.py` y llama a `_run_pair` sin cambios con otra matriz de embeddings. Se generalizaría a `--encoder <ID>` y a todos los aprendices. No hay config de encoder en el repo: `perception.py` tiene el modelo fijo (l. 38), por eso se trabaja con embeddings precalculados.

| Análisis | Aprendices | Semillas y condiciones | Métricas |
|---|---|---|---|
| T | Los 6 de la v1 | Semilla 42, 30 episodios, 3-way 5-shot, todos los pozos | C1 por tier, IC bootstrap por episodio; ρ(C1, d del concepto) |
| T-sem (sensibilidad) | Los 6 | Semillas 0-3 y 42, modos `episodes` y `full` | Orden de tiers por semilla |
| D | centroid (primario), k-NN3, logreg, azar | Los **mismos 100 pares de la v1**, pozo 200/clase semilla 42, contraste 9000 + pair_id, 30 episodios | Modelo PS: β_perc, β_sem, R² parciales; reglas A/B/C con los cuadrantes recalculados por mediana del d_perc de cada encoder sobre esos 100 pares |
| **D-todos (sensibilidad obligatoria; I-A2)** | centroid | Los 4.950 pares de CIFAR-100, 30 episodios por par, misma semilla de pozo y contraste 9000 + pair_id | Modelo PS (β_perc, β_sem, r parciales); concordancia con D-100 según §3.6 (~12 s por 100 pares en la v1, ~10 min por encoder, sin medir con 4.950) |
| H1 | — | 3 fuentes, CIFAR-100 y CIFAR-10 | ρ, IC bootstrap 10.000, Mantel 10.000 (semilla 42) |
| H2, H2b | centroid | Pares de D cubiertos (61 y 32) | β_humana, r parcial, IC; H2b descriptivo. Descriptivo además: H2 sobre los pares de D-todos cubiertos por THINGS y HSJ, no decide |
| H3 | centroid | 10 conceptos; 10.000 imágenes de test | ρ descriptivos |
| H4 por encoder (personas; **réplica de la v1**) | — | 24 ítems, modelo exacto de H4 v1 | log OR de z(dist) |
| H5 (opcional) | — | Igual que v1 | Descriptivo |
| **Tabla agentes × personas** | — | Por encoder | Fila por encoder: β(zd) del control positivo humano (§5.3 del plan principal), H4, región humana (familia B), C1 por tier, β_perc de D, ρ de H1 ×3, H2 ×2. Más Spearman entre encoders (n = 10) de β(zd) humano vs ρ H1 y vs β_humana de H2: **solo descriptiva, sin p ni IC inferencial** |
| Fig. x1 por encoder | — | — | Descriptiva |

**"Órdenes":** no hay factor de orden en los agentes, porque no tienen estado entre episodios (§0). Los episodios se sortean con las semillas indicadas.

---

## 3. Encaje con el marco existente

### 3.1 Familias y qué es confirmatorio
- **R0 y R1: descriptivos.** Sus resultados de agentes se conocen por la v1 y por H6, así que no entran en ninguna familia Holm. Igual se evalúan sus regiones, porque **la región de R0 es la puerta de GA0b** (§3.4).
- **Encoders nuevos entrenados que deciden: O1, E1, O2, E2, E3 y E4** (k ≤ 6), menos los que estén en "no interpretable (control positivo de agentes)" o en "inconcluso (técnico)".
- **Familia C (agentes-personas):** por encoder, H1 × 3 fuentes, H2 × 2 y H4 × 1, en total ≤ 36 pruebas.
  - **H4 se declara réplica de la v1:** está en la familia solo para reproducir por encoder el Holm de 6 de la v1. No tiene lectura GA propia, y su resultado se reporta como réplica, encoder por encoder.
  - Sin H4, la familia sería de ≤ 30. Se fija de antemano en ≤ 36, con H4 (decisión P7).
- **Familia D (disociación de agentes):** por encoder, la prueba de β_perc < 0 del Modelo PS con los 100 pares; ≤ 6 pruebas.
- **Holm solo para GA1** (la unión "depende del encoder"), por SESOI y dentro de cada familia, igual que en §7.3 del plan humano. Las regiones por encoder no usan Holm (§3.2).
- **Pisos (F1 y F2):** familias C-pisos (12 pruebas) y D-pisos (2), cada una con Holm aparte.
- **Descriptivos:** T, T-sem, H2b, H3, H5, la tabla agentes × personas, la fig. x1, E4s y F1 semillas 1 y 2. D-todos es una sensibilidad obligatoria (§3.6) y no entra en ninguna familia.

### 3.2 Regiones por encoder (sin ajuste) y márgenes en escala de correlación
- **Tabla separada de la humana:** los márgenes de agentes son **SESOI ∈ {0,10; 0,20; 0,30}** en escala de |ρ| y |r parcial|. 0,20 es el SESOI de la v1 (`plan_analisis_datos_humanos_FROZEN_2026-10-30.md` l. 63).
  - **No son comparables** con los δ humanos (0,05, 0,10 y 0,20, en log-odds por DE). Se reportan en una tabla de agentes aparte, nunca en la misma tabla ni como equivalentes.
  - Lectura del espectro igual que §7.2 del plan principal: si la región es la misma con los tres márgenes, el perfil es "robusto"; si no, "dependiente del margen". Ningún margen se reporta sin los otros dos.
- **Regiones, sin ajuste por multiplicidad. Se aplica la primera que se cumpla:**
  0. **Inconcluso (técnico)** (BC-A2), sin SESOI: fallan los embeddings, la simulación o la estimación (regresión no estimable, matriz singular, NaN), o falta una fuente. El encoder sale de la familia Holm, bloquea GA2 y GA2b, y lleva a GA3. **Si es R0, el resultado es GA0 "sin lectura"** (§3.4).
  0b. **No interpretable (control positivo de agentes)** (§3.5): solo para los entrenados. Sale de Holm y bloquea GA2.
  1. **Apoyo** (sin ajuste):
     - H1: ρ > 0 y p de Mantel < 0,05;
     - H2 y D: β < 0 con p < 0,05, más la parte de β_sem de la regla de la v1 (p ≥ 0,10, o R² parcial menor que un tercio del de la distancia);
     - H4: β < 0 con p < 0,05.
  2. **No apoyo:**
     - H1: el IC 95 % de ρ queda entero por debajo del SESOI;
     - H2 y D: el IC 95 % del r parcial excluye todo valor ≤ −SESOI, o hay un efecto opuesto con p < 0,05.
  3. **Inconcluso:** todo lo demás.
- **Expectativa declarada de antemano (REVISION4 (3)):** en H2 con HSJ (n = 32), la mitad del ancho del IC 95 % del r parcial es ~0,35. Por eso **"No apoyo" en H2 es casi inalcanzable incluso con SESOI = 0,30**, y lo más probable es "Inconcluso" con todos los encoders. Con THINGS (n = 61) la expectativa es parecida. Que H2 salga inconclusa no se leerá como ausencia de efecto.
- **Holm** (§3.1) se usa solo para decidir GA1. Con 10.000 permutaciones, el p mínimo de Mantel es ~1e−4, menor que 0,05/36 ≈ 0,0014.

### 3.3 Geometría: solo descriptiva
- Los grupos se toman de la tabla de equivalencias del plan principal (§6.1, cortes 0,3, 0,5 y 0,7). Se rotulan siempre **"grupos medidos en estímulos de Lewis & Frank"**: son fotos de ~720 px, no CIFAR de 32 × 32 escalado.
- Junto a ellos se reporta la **ρ entre encoders en CIFAR**: Spearman de las distancias de los 4.950 pares de CIFAR-100, como H6(b).
- **Concordancia, fijada de antemano:** con cada corte, se agrupa la ρ de CIFAR con la misma regla de §6.1 del plan principal. Si los grupos de CIFAR no coinciden con los de Lewis & Frank, GA2 y GA2b se informan **"sin grupos"** con ese corte. Esto solo se describe y no cambia ninguna región.

### 3.4 Conclusiones de agentes: conjunto paralelo, con precedencia
G0–G4 se refieren al orden en personas y no cambian. Las GA se evalúan **por separado para D (disociación), A (H1) y P (H2)**, con cada SESOI y cada corte (grilla 3 × 3, misma regla de perfil que §7.4). **Precedencia: GA0 > GA0b > GA1 > GA2 > GA2b > GA3, y gana la primera regla que se cumpla.**

| # | Condición (con un SESOI dado) | Lectura |
|---|---|---|
| GA0 | Falla el control de tubería de agentes (§3.5), **o R0 está en "inconcluso (técnico)"** en ese análisis | **Sin lectura.** Se detiene solo la parte de agentes de ese análisis; la parte humana sigue. El G0 humano tampoco detiene a los agentes |
| GA0b | R0 no está en "Apoyo" en ese análisis | **Si R0 no está en Apoyo, la comparación es descriptiva:** se reporta encoder por encoder, sin GA1 ni GA2. Con P (H2) es lo esperado, porque H2 fue inconclusa con R0 en la v1 |
| GA1 | ≥ 1 encoder nuevo entrenado (que pasó el control positivo y no es técnico) está en "No apoyo" y su p de esa región es < 0,05 **con Holm** en su familia (C o D) | **Depende del encoder.** Se describe con los grupos de §3.3, o "sin grupos" |
| GA2 | Todos los nuevos entrenados pasaron el control positivo, ninguno es técnico, todos están en "Apoyo" (cada uno a su 95 %, sin Holm, como intersección-unión) y ocupan ≥ 2 grupos que concuerdan con CIFAR (§3.3) | **No es específico de DINOv2** |
| GA2b | Igual que GA2, pero hay 1 solo grupo, falla la calibración o los grupos discrepan con CIFAR | Se mantiene en todos, **sin evidencia de diferencia geométrica** |
| GA3 | Cualquier otro caso, incluido algún entrenado técnico o no interpretable | **Inconcluso o mixto**, reportado encoder por encoder |

- **Lectura de D (y de T):** ver §3.7. Un GA2 en D significa "suficiencia del mecanismo con cualquier encoder", no evidencia sobre las personas.
- **Lo que nunca se afirma:** nada sobre el orden en agentes (§0), ni que el gradiente de los agentes explique el de las personas.

### 3.5 Controles
- **Control de tubería de agentes (GA0):**
  1. **Control principal (código):** T con R0 desde el cache de CIFAR-10 de la v1 (`.cache/embeddings_exp007/cifar_all10.npz`) reproduce exactamente el ledger de la v1, como la reproducción de H6 (`out/h6_tiers_dino.json`).
  2. **Re-codificación de CIFAR-100 con R0 en CPU frente a `perceptual_distance_matrix.csv` de la v1 (GPU T4), con la tolerancia de Ariadna, adoptada tal cual:**
     - **PASA** si max|Δ| ≤ **1e−4** y Spearman ≥ **0,99999**.
     - **PASA con desviación registrada** si 1e−4 < max|Δ| ≤ **2e−3**, Spearman ≥ **0,9999** **y** el β_perc de D con R0 queda dentro de **±0,005** del de la v1. Se registran las versiones y la causa.
     - **FALLA** (GA0) en cualquier otro caso.
  - *Motivo (REVISION4 (5)):* las versiones de `torch` y `transformers` en Modal no estaban fijadas. En la matriz de la v1, 3.301 de las 4.949 brechas consecutivas son < 1e−4, así que exigir el rango exacto no sirve.
- **Control positivo de agentes por encoder (puerta):** la C1 de las palabras nativas (prototipo, semilla 42) debe tener un IC 95 % con límite inferior > 1/3, que es el azar.
  - **Es una puerta laxa, y se declara así:** solo descarta encoders con los que el aprendiz no aprende nada.
  - Un entrenado que falla queda "no interpretable" (§3.2). En F1 y F2 se reporta, pero no es puerta.
- **Control positivo humano (§5.3 del plan principal):** sin cambios. Además, entra como columna en la tabla agentes × personas.

### 3.6 Dependencia de los 100 pares elegidos con DINOv2 (I-A2)
- **Antes de correr las simulaciones** (después de los embeddings y antes de cualquier precisión), se reporta por encoder en los 100 pares de D:
  - la DE y el rango de d_perc;
  - la Spearman de d_perc con d_sem;
  - el número de pares por cuadrante recalculado.

  Se escribe en `agent_pairs_dispersion.json` y no cambia ninguna regla.
- **D-todos (4.950 pares) es obligatoria** y no es la primera sensibilidad que se corta.
- **Concordancia fijada de antemano:** se compara la región de D con 100 pares y con 4.950, con el mismo SESOI. Si difieren con algún SESOI, el encoder se rotula **"sensible al muestreo de pares"**. El rótulo es descriptivo y no cambia ni la región ni las GA.
- **No se repondera** ni se re-muestrea por encoder.
- Ventaja de casa declarada: los pares son ortogonales en d_perc × d_sem solo para DINOv2. Con otros encoders puede haber colinealidad, menos potencia y un sesgo hacia "depende del encoder".

### 3.7 Circularidad: qué se puede leer frente a las personas (I-A3)
- En **D y T**, el aprendiz y d_perc usan el mismo espacio. Por eso **el efecto de la distancia es casi mecánico con cualquier encoder**: muestra la suficiencia del mecanismo, no algo sobre las personas.
- **Solo H1 y H2** usan distancias humanas, que son independientes del encoder, y **son los únicos análisis de agentes que se leen frente a las personas**. H4 es la réplica humana. La tabla agentes × personas es descriptiva.

---

## 4. Costo

### 4.1 Medido hoy (prueba de tiempos, 18-11 12:38-12:39, ~1 min en total)
- **Montaje:** CPU, 4 hilos, lotes de 64, 128 imágenes de CIFAR-100. El box tenía carga ~2,5 por 3 procesos R de pruebas sintéticas de otro trabajador.
- **Resultados:** R0 DINOv2-S ≈ 1.000 img/min; O1 DINOv2-B ≈ 480 img/min; E1 SigLIP ≈ 590 img/min. Script en `/tmp/agentes_patch/probe*.py`.
- **Referencia de la v1:** CVCL sobre 50.000 imágenes tardó ~6 h en CPU (~140-150 img/min; `DESVIACIONES.md` puntos 20-21). Probablemente fue imagen por imagen y con carga, así que no es comparable con lotes.
- **Simulaciones:** baratas. D con el prototipo tomó ~12 s por encoder (`log_h6_c100.txt`, mtimes 18:03:50 → 18:04:02). T con 6 aprendices y todos los pozos tomó ~5-7 min por semilla, en 1 hilo (`r04_seeds/*.log`, 12:02 → 12:09).

### 4.2 Estimado por encoder (62.000 imágenes; **solo R0, O1 y E1 están medidos; el resto es estimación**)
| Encoder | Embeddings | Notas |
|---|---|---|
| R0 | ~1 h | Control de tubería |
| O1 | ~2,2 h | medido |
| E1 | ~1,8 h | medido |
| E2, E4 (ViT-B/16) | ~1,8-2 h c/u | estimado como SigLIP; E4s sale de la misma pasada |
| O2 (ViT-B/32) | ~0,7-1 h | estimado |
| E3, F1 (ResNet-50) | ~1-1,5 h c/u | estimado |
| R1 | ~0,2-1,2 h | solo falta el test de CIFAR-10 |
| F2 | minutos | |
| F1 semillas 1-2; T-sem `full` | +2-3 h; +1,5 h | sensibilidades |

- **Total de embeddings primarios:** ~13 h-proceso de 4 hilos (~40-55 h-núcleo).
- **Simulaciones y análisis:** ~4,5-8 h de CPU (T con 10 encoders: ~1 h; T-sem: ~5 h; D: < 15 min por encoder; **D-todos obligatoria: ~10 min por encoder con el prototipo, ~1,5-2 h en total, extrapolado de 12 s por 100 pares y sin medir con 4.950**; H1, H2 y H3: minutos, sin medir; dispersión de §3.6: segundos).
- **Cambio de costo por REVISION4:** solo D-todos, que pasa a obligatoria y suma **~1,5-2 h de CPU, ~30-40 min de reloj con 3 procesos**. No hace falta ninguna imagen nueva, porque usa los mismos embeddings de CIFAR-100. El resto no cambia.
- **H4 por encoder:** 9 GLMM de 172.800 obs., ~12-36 min c/u según §9.2 del plan principal; **~2-5,5 h de CPU, sin medir**.
- **Reloj con 3 procesos en paralelo en el box libre (8 núcleos, 15 GB):** ~6-9 h para los embeddings primarios, más ~2,5-3,5 h de simulaciones y análisis (D-todos incluida), más H4. Compartiendo el box con la parte humana (3 procesos R, 8-11 h de reloj según §9.2), la suma del fin de semana queda en **~16-24 h de reloj**. Incertidumbre alta: hasta ×2.
- **RAM:** 50.000 × 768 float32 ≈ 150 MB por encoder. No es el límite; el límite son los procesos R.
- **Costo monetario y API:** **ninguno**. Todo corre local en CPU, sin LLM ni Modal. Los pesos de los 8 checkpoints y CIFAR-10/100 ya están en el cache del box.

---

## 5. Calendario

**¿Cabe en el 21-22 con congelamiento el 20?** Justo, y solo si se cumplen tres condiciones:
1. Patricio aprueba el detalle el jueves 19 antes de ~12:00.
2. Los scripts nuevos se escriben y prueban con datos sintéticos el jueves 19: `10_agent_embed.py`, `11_agent_sims.py` (T, D, generalizando `h6_tiers.py` y `h6_c100.py`), `12_agent_h1h2h3.py`, `13_h4_by_encoder.R`, `14_agent_regions.py` (GA, familias C y D) y sus tests. Son ~1 jornada.
3. Ariadna revisa la sección nueva y los scripts el jueves por la tarde o el viernes por la mañana (~2-3 h).

- **Lo que más ayuda (P5):** codificar CIFAR el jueves y el viernes, antes de congelar. Solo produce npz y sha256: no se calculan centroides, distancias ni precisiones. Es análogo a §9.3 del plan principal, pero adelanta cómputo y necesita la aprobación de Patricio.
- **Opción mínima para el 21-22:**
  - solo el prototipo;
  - T con semilla 42, D con los 100 pares de la v1 y **D-todos, que ya es obligatoria**;
  - H1, H2 y H3 por tiers;
  - los 10 primarios;
  - sin T-sem, H5, el complemento por imagen de H3, E4s ni F1 semillas 1-2.

  Son ~52.000 imágenes por encoder, unas 6,5-8,5 h de reloj con D-todos. H4 por encoder iría el domingo si queda presupuesto.
- **Respaldo (recomendado por mí y por Ariadna en REVISION4):** la parte humana se congela el viernes 20 y corre el 21-22 **sin cambios**. La parte de agentes queda como adenda propia, congelada a más tardar el viernes 27 y ejecutada el sábado 28 y el domingo 29. Costo: la tabla agentes × personas se arma con resultados humanos ya vistos; es descriptiva y se declara. Con esta opción, el visto bueno de Ariadna para congelar la parte humana deja de depender de esta propuesta.

---

## 6. Riesgos y confusiones
1. **Sin estado entre episodios** (§0): la pregunta de la familia B no se puede responder con estos agentes.
2. **Resultados conocidos:** R0 (v1) y R1 (H6) no son confirmatorios. La tubería se diseñó con DINOv2 ("ventaja de casa", I4).
3. **Pares de D muestreados por cuadrantes de DINOv2:** para los otros encoders el rango de d_perc puede quedar restringido, o d_perc puede correlacionar con d_sem. Las reglas A3 y A4 se recalculan con los cuadrantes de cada encoder sobre los mismos 100 pares. La mitigación es la dispersión que se reporta antes de correr, D-todos obligatoria y el rótulo "sensible al muestreo de pares" (§3.6).
4. **Resolución:** las imágenes CIFAR son de 32 × 32 y se escalan a 224. Los encoders difieren en robustez a baja resolución, lo que puede bajar la C1 sin relación con la geometría (H6: la C1 base de CVCL fue menor).
5. **Supervisión categórica:** E2 y E3 (ImageNet) y los modelos imagen-texto probablemente vieron estas categorías, así que su C1 refleja conocimiento de clases (M7).
6. **Circularidad:** en D y T el efecto de la distancia es casi mecánico. Solo H1 y H2 se leen frente a las personas (§3.7; `main.tex` l. 174).
7. **Potencia de H2:** n = 61 y 32. "No apoyo" es casi inalcanzable y lo esperado es "Inconcluso" (§3.2). Además, GA0b la vuelve descriptiva si R0 no está en Apoyo, como en la v1.
8. **GPU→CPU:** las versiones de Modal no estaban fijadas; se cubre con la tolerancia de tres niveles (§3.5).
9. **Carga:** riesgo de no terminar. Si se pasa el tope, el orden de corte se fija de antemano: sensibilidades opcionales de agentes (T-sem `full`, H5, F1 semillas 1 y 2, E4s, T-sem `episodes`) → H4 por encoder → sensibilidades humanas. **D-todos no se corta.**
11. **Geometría con otra fuente:** los grupos vienen de los estímulos de Lewis & Frank, no de CIFAR. Por eso son solo descriptivos, con el chequeo de concordancia de §3.3.
10. **Licencia:** MAE es no comercial, igual que en el plan principal.

---

## 7. Decisiones de Patricio (con recomendación por defecto)
- **P1. Calendario:** ¿la parte humana se congela y corre el 21-22 sin cambios, y los agentes van como adenda congelada a más tardar el viernes 27 y corrida el 28-29? *Recomendado por mí y **por Ariadna (REVISION4)**: sí.* La alternativa es la opción mínima de agentes el 21-22, con aprobación antes del jueves 19 a las 12:00.
- **P2. Familias C (≤ 36, con H4) y D (≤ 6)** separadas de la B, con regiones sin ajuste, Holm solo en GA1, pisos aparte y R0 y R1 descriptivos (pero R0 como puerta de GA0b). *Recomendado: sí.*
- **P3. Margen en agentes:** SESOI 0,10, 0,20 y 0,30 como espectro, en escala de correlación y en una tabla aparte de la humana. *Recomendado: sí.*
- **P4. Geometría:** ¿los grupos de los estímulos de Lewis & Frank, solo descriptivos, con chequeo contra la ρ de CIFAR y "sin grupos" si discrepan? *Recomendado: sí.*
- **P5. Embeddings de CIFAR antes de congelar** (solo npz y sha256, sin distancias ni precisiones). *Recomendado: solo si elige la opción mínima el 21-22. Con P1 recomendado se hacen después de congelar la adenda.*
- **P6. Control de tubería de agentes:** el control principal es la reproducción exacta de T; la re-codificación de CIFAR-100 usa la tolerancia de tres niveles de Ariadna. *Recomendado: sí.*
- **P7. H4 por encoder en la familia C** como réplica declarada de la v1, sin lectura GA propia. La alternativa es dejar H4 descriptivo y la familia C de ≤ 30. *Recomendado: en la familia C.*
- **P8. D-todos ya es obligatoria.** ¿Las demás sensibilidades (T-sem, H5, F1 semillas 1 y 2, E4s) entran como opcionales, cortables en el orden de §6.9? *Recomendado: sí.*
