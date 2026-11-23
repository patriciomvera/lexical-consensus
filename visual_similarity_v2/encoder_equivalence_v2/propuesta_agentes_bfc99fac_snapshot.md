# Propuesta: agentes y personas con múltiples encoders (v2 de arXiv 2611.14116)

- **Estado:** PROPUESTA PARA DECISIÓN DE PATRICIO. **No está congelada, no se ha agregado al plan principal y no se ha ejecutado nada** salvo una prueba de tiempos de ~1 min (§4.1). No se leyó ninguna respuesta de participantes ni se calculó ninguna precisión, distancia o correlación con encoders nuevos.
- **Origen:** aclaración de Patricio (18-11 12:36, vía Evaristo): la v2 debe **repetir lo que hizo la v1 con múltiples encoders, comparando agentes y personas**; las simulaciones de los agentes se rehacen con cada encoder y se comparan con las personas, igual que en la v1.
- **Relación con el plan principal:** `propuesta_comparacion_encoders_v2_2026-11-18.md` (sha256 de la versión leída: `4ff4b341a934fcd8cacec5ae9303909781e0b7abca0e45e4a7d34b54d256e3f7`, 18-11 12:36). Hoy cubre solo personas (12 experimentos de Lewis & Frank) y deja "repetir H6 (el aprendiz en CIFAR) con los encoders nuevos" como fuera de alcance (§8, línea 359). El cambio que implicaría está en `encoder_equivalence/DIFF_agentes_propuesta.patch`.
- **Alcance:** solo diseño. Sin título ni encuadre.

---

## 0. Hecho de partida que condiciona todo: los agentes no tienen "orden"

- Los aprendices de la v1 son **un encoder congelado + una regla de aprendizaje sin entrenamiento de pesos**: prototipo (`src/learners/centroid.py:35-47`, media normalizada de los 5 ejemplos), k-NN (k = 3), k-centroides, regresión logística y SVM lineal ajustados en lote, y un azar de referencia (`experiments/exp_007b_extension_clean/run_clean.py:73-76`). No hay modelos de lenguaje ni llamadas a API.
- La percepción es `src/agents/perception.py` (DINOv2-S, CLS de la última capa, L2; líneas 37-39 y 100-106). Es la misma función que calculó d para las personas (plan principal §1.3, línea 61).
- **Todos estos aprendices son invariantes al orden de presentación por construcción** (la media, el conjunto de vecinos y un ajuste en lote no dependen del orden). Por eso **la familia B del plan principal (orden × distancia) no tiene contraparte en los agentes**: un "efecto de orden" en estos agentes es 0 mecánicamente, no un resultado. Probarlo exigiría un aprendiz secuencial nuevo, que no existe en la v1 y no es "repetir la v1".
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
| D-todos (opcional, descriptivo) | centroid | Los 4.950 pares | Igual que D (el aprendiz tarda ~12 s por 100 pares; ~10 min por encoder) |
| H1 | — | 3 fuentes, CIFAR-100 y CIFAR-10 | ρ, IC bootstrap 10.000, Mantel 10.000 (semilla 42) |
| H2, H2b | centroid | Pares de D cubiertos (61 y 32) | β_humana, r parcial, IC; H2b descriptivo |
| H3 | centroid | 10 conceptos; 10.000 imágenes de test | ρ descriptivos |
| H4 por encoder (personas) | — | 24 ítems, modelo exacto de H4 v1 | log OR de z(dist) |
| H5 (opcional) | — | Igual que v1 | Descriptivo |
| **Tabla agentes × personas** | — | Por encoder | Fila por encoder: β(zd) del control positivo humano (§5.3 del plan principal), H4, región humana (familia B), C1 por tier, β_perc de D, ρ de H1 ×3, H2 ×2. Más Spearman entre encoders (n = 10) de β(zd) humano vs β_perc agente y vs ρ H1: **solo descriptivo**, sin p |
| Fig. x1 por encoder | — | — | Descriptiva |

**"Órdenes":** no hay factor de orden en los agentes (§0). Los episodios se sortean con las semillas indicadas y la invariancia al orden se declara de antemano.

---

## 3. Encaje con el marco existente

### 3.1 Familias y qué es confirmatorio
- **R0 y R1: descriptivos** (resultados de agentes conocidos por la v1 y por H6). Se reportan, no entran en ninguna familia.
- **Encoders nuevos entrenados que deciden: O1, E1, O2, E2, E3, E4** (k ≤ 6).
- **Familia C (agentes-personas, réplica del Holm de 6 de la v1 por encoder):** por encoder, H1 × 3 fuentes, H2 × 2 y H4 × 1 = 6 pruebas; **6 × 6 = 36 pruebas, Holm** en conjunto. Es separada de la familia B (orden) del plan principal porque responde otra pregunta, y la familia B no cambia.
- **Familia D (disociación de agentes):** por encoder, la prueba de β_perc < 0 del Modelo PS; k ≤ 6 pruebas, **Holm**. β_sem y las partes A2-A4 de la regla son criterios de lectura, como en la v1, no pruebas de la familia.
- **Pisos (F1, F2):** familias C-pisos (12 pruebas) y D-pisos (2), con Holm aparte, como en §7.3 del plan principal.
- **Descriptivos:** T, T-sem, H2b, H3, H5, D-todos, la tabla agentes × personas, la fig. x1, E4s y F1 semillas 1-2.
- **Expectativa declarada de antemano:** H2 fue inconclusa en la v1 con n = 61 y 32. Con Holm sobre 36 pruebas es muy probable que siga inconclusa con todos los encoders. Con 10.000 permutaciones el p mínimo de Mantel es ~1e−4, por debajo de 0,05/36 = 0,0014, así que H1 sí puede pasar Holm.

### 3.2 Márgenes (en lugar de δ)
- Los tres δ del plan principal (0,05, 0,10 y 0,20) están en log-odds por DE del Δ humano y **no se trasladan** a ρ, r parciales ni a precisiones.
- La v1 usaba SESOI = 0,20 en |ρ| y |r parcial| (`plan_analisis_datos_humanos_FROZEN_2026-10-30.md` l. 63). Propuesta (decisión P3): evaluar las reglas de la v1 con **SESOI ∈ {0,10; 0,20; 0,30} como espectro**, con la misma lógica que los δ: misma región en los tres = "robusto"; si no, "dependiente del margen"; nunca uno sin los otros. 0,20 es el valor de la v1 y no se privilegia.
- **Regiones por prueba:** las de la v1. H1: Apoyo (ρ > 0 y p Holm < .05), No apoyo (IC < SESOI), Inconcluso. H2 y D: Apoyo (β < 0, p Holm < .05 y regla de β_sem), No apoyo (IC del r parcial excluye valores ≤ −SESOI, o efecto opuesto significativo), Inconcluso. H4: Apoyo / Inconcluso.

### 3.3 Cortes de geometría 0,3, 0,5 y 0,7
- **Propuesta por defecto (P4):** se usan **los mismos grupos** de la tabla de equivalencias del plan principal (§6.1, imágenes de Lewis & Frank) para describir en qué grupos quedan los encoders en las conclusiones de agentes. Así no se abre una segunda regla de agrupamiento.
- Además, una tabla **descriptiva** entre encoders sobre los 4.950 pares de CIFAR-100 (ρ de distancias, como H6(b)) que no agrupa ni decide.
- La alternativa, agrupar con CIFAR, exigiría una nueva calibración y otra revisión de Ariadna; no la recomiendo para el 21-22.

### 3.4 Conclusiones de agentes: un conjunto paralelo, no G0-G4
G0-G4 se refieren al orden en personas y no cambian. Propongo **GA0-GA3**, evaluadas por separado para D (disociación), A (H1) y P (H2), con cada SESOI y con cada corte (grilla 3 × 3, igual que §7.4):

| # | Condición | Lectura |
|---|---|---|
| GA0 | Falla el control de tubería de agentes (§3.5) | Se detiene **solo la parte de agentes**. La parte de personas sigue. El G0 humano tampoco detiene a los agentes, porque usan imágenes y tuberías distintas |
| GA1 | ≥ 1 encoder nuevo entrenado que pasó el control positivo cae en "No apoyo" con Holm en su familia | **Depende del encoder** (con grupos si la calibración del corte discrimina; si no, sin grupos) |
| GA2 | Todos los nuevos entrenados pasan el control y están en "Apoyo" (intersección-unión, cada uno a su 95 %, sin Holm), en ≥ 2 grupos | **No es específico de DINOv2** |
| GA2b | Igual que GA2, pero con 1 grupo o con la calibración fallida | Se mantiene en todos, **sin evidencia de diferencia geométrica** |
| GA3 | Otro caso | **Inconcluso o mixto**, reportado encoder por encoder |

- **Lo que nunca se afirma:** nada sobre el orden en agentes (§0), ni que el gradiente de los agentes "explica" el de las personas. La comparación agentes × personas es descriptiva.

### 3.5 Controles
- **Control de tubería de agentes (GA0), propuesto:**
  1. T con R0 desde el cache de la v1 reproduce el ledger de la v1 (exacto, como la reproducción de H6, `out/h6_tiers_dino.json`).
  2. La re-codificación de CIFAR-100 con R0 en CPU reproduce `perceptual_distance_matrix.csv` con |Δ| ≤ 1e−4 y Spearman ≥ 0,9999. **La tolerancia es una propuesta sin medir:** la v1 se calculó en GPU, y Ariadna debería fijarla (P6).
  3. D con R0 reproduce β_perc de la v1 dentro de 0,005.
- **Control positivo de agentes por encoder (puerta, como §5.3):** la C1 de las palabras nativas (prototipo, semilla 42) tiene un IC 95 % con límite inferior > 1/3 (azar). Si un encoder entrenado falla, queda "no interpretable (control positivo de agentes)", sale de C y de D y bloquea GA2. En F1 y F2 se reporta, pero no es puerta.
- **Control positivo humano (§5.3):** sin cambios. Además, entra como columna en la tabla agentes × personas.

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
- **Simulaciones y análisis:** ~3-6 h de CPU (T con 10 encoders: ~1 h; T-sem: ~5 h; D: < 15 min por encoder; H1, H2 y H3: minutos, no medido).
- **H4 por encoder:** 9 GLMM de 172.800 obs., ~12-36 min c/u según §9.2 del plan principal; **~2-5,5 h de CPU, sin medir**.
- **Reloj con 3 procesos en paralelo en el box libre (8 núcleos, 15 GB):** ~6-9 h para los embeddings primarios, más ~2-3 h de simulaciones y análisis, más H4. Compartiendo el box con la parte humana (3 procesos R, 8-11 h de reloj según §9.2), la suma del fin de semana queda en **~16-24 h de reloj**. Incertidumbre alta: hasta ×2.
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
  - T con semilla 42 y D con los 100 pares de la v1;
  - H1, H2 y H3 por tiers;
  - los 10 primarios;
  - sin T-sem, D-todos, H5, el complemento por imagen de H3, E4s ni F1 semillas 1-2.

  Son ~52.000 imágenes por encoder, unas 6-8 h de reloj. H4 por encoder iría el domingo si queda presupuesto.
- **Respaldo:** la parte humana se congela el viernes 20 y corre el 21-22 **sin cambios**. La parte de agentes queda como adenda propia, congelada a más tardar el viernes 27 y ejecutada el sábado 28 y el domingo 29. Costo: la tabla agentes × personas se arma con resultados humanos ya vistos; es descriptiva y se declara.

---

## 6. Riesgos y confusiones
1. **Sin orden en agentes** (§0): la pregunta de la familia B no se puede responder con estos agentes.
2. **Resultados conocidos:** R0 (v1) y R1 (H6) no son confirmatorios. La tubería se diseñó con DINOv2 ("ventaja de casa", I4).
3. **Pares de D muestreados por cuadrantes de DINOv2:** para los otros encoders el rango de d_perc puede quedar restringido o desbalanceado entre cuadrantes. Las reglas A3 y A4 se recalculan con los cuadrantes de cada encoder sobre los mismos 100 pares, pero el muestreo sigue dependiendo de R0. D-todos (4.950 pares) lo mitiga de forma descriptiva.
4. **Resolución:** las imágenes CIFAR son de 32 × 32 y se escalan a 224. Los encoders difieren en robustez a baja resolución, lo que puede bajar la C1 sin relación con la geometría (H6: la C1 base de CVCL fue menor).
5. **Supervisión categórica:** E2 y E3 (ImageNet) y los modelos imagen-texto probablemente vieron estas categorías, así que su C1 refleja conocimiento de clases (M7).
6. **Circularidad:** en agentes, d y el aprendiz usan el mismo espacio. El gradiente es en parte mecánico (suficiencia, no explicación; `main.tex` l. 174).
7. **Potencia de H2:** n = 61 y 32 con 36 pruebas en Holm.
8. **Tolerancias GPU/CPU** del control de tubería de R0 en CIFAR-100: no están medidas.
9. **Carga del fin de semana:** riesgo de no terminar. Si se pasa el tope, el orden de corte se fija de antemano: sensibilidades de agentes → H4 por encoder → sensibilidades humanas.
10. **Licencia:** MAE es no comercial, igual que en el plan principal.

---

## 7. Decisiones de Patricio (con recomendación por defecto)
- **P1. Calendario:** ¿todo el 21-22 (opción mínima) o la parte humana el 21-22 y los agentes el 28-29? *Recomendado: la parte humana el 21-22 sin cambios y los agentes el 28-29.* Si prefiere un solo fin de semana: la opción mínima, con aprobación antes del jueves 19 a las 12:00.
- **P2. Familias:** ¿la familia C (H1 ×3, H2 ×2 y H4 por encoder nuevo, Holm de 36) y la familia D (disociación, Holm de 6) separadas de la familia B, con pisos aparte y R0 y R1 descriptivos? *Recomendado: sí.*
- **P3. Margen en agentes:** ¿SESOI de la v1 (0,20) o un espectro de 0,10, 0,20 y 0,30 con la lógica de los δ? *Recomendado: el espectro.*
- **P4. Geometría:** ¿se reutilizan los grupos de las imágenes de Lewis & Frank (cortes 0,3, 0,5 y 0,7), con CIFAR solo descriptivo? *Recomendado: sí.*
- **P5. Embeddings de CIFAR antes de congelar** (solo npz, sin distancias ni precisiones). *Recomendado: sí, si se mantiene el 21-22.*
- **P6. Control de tubería de agentes:** ¿se acepta GA0, con tolerancias que fije Ariadna? *Recomendado: sí.*
- **P7. H4 por encoder** (personas, modelo exacto de la v1): ¿entra en la familia C o queda descriptivo? *Recomendado: en la familia C, porque así se replica el Holm de la v1.*
- **P8. Opcionales:** T-sem, D-todos, H5, F1 semillas 1-2 y E4s, ¿como sensibilidades? *Recomendado: sí, si cabe en el tope; si no, primero se corta D-todos.*
