# exp_007 — Concept Carving · Reporte de experimento

**Sub-experimento ejecutado:** `007a` (NATIVE)
**Fecha de ejecución:** 2026-06-05
**Estado:** PASS — H2 confirmada (sin disociación entre learners en conceptos nativos)
**Máquina:** HP i7-1165G7, 16 GB RAM, sin GPU. DINOv2-small en CPU.

---

## 0. Por qué existe este experimento

El revisor de la Ronda 1 planteó dos críticas. Una es barata (escala: 3 clases,
sin semillas, sin intervalos de confianza, sin baselines). La otra puede hundir
el paper:

> el learner es un clasificador por centroide sobre un espacio DINOv2 ya
> separable; "aprender" se reduce a promediar features de categorías que el
> encoder ya conocía → asigna un *puntero* a un cluster preexistente en vez de
> *adquirir* un concepto.

exp_007 ataca la crítica real. Pregunta organizadora:

> **¿Puede un learner adquirir un concepto cuya extensión NO coincide con un
> cluster nativo de DINOv2?**

El experimento es una grilla 2-D (plan §2):

- **Eje A — tipo de concepto:** NATIVE → DISJUNCTIVE → CROSS_CUTTING (NS y
  NMI_native decrecientes).
- **Eje B — learner:** centroid (prototipo puro) → kcentroid (multi-ancla) →
  exemplar_knn (episódico puro) + logreg / linsvm (baselines lineales del
  revisor) + random (piso de azar).

**Honestidad obligatoria (va al paper):** este experimento NO rescata la tesis
mayor de Vera et al. (adquisición > imitación como métrica fundamental de
inteligencia). Derrota la objeción "puntero, no adquisición" y mapea la frontera
prototipo/episódico. Se declara explícitamente en Limitaciones.

### Rol de 007a en la secuencia

007a es el ancla del extremo "fácil" de la curva. Es barato (reutiliza
embeddings cacheados, cero datos nuevos) y cumple tres funciones a la vez:

1. valida que los seis learners corren end-to-end sobre datos reales,
2. establece la **H2** (sin disociación en conceptos nativos),
3. produce la primera tabla con rigor estadístico completo (CIs, tests
   pareados), que ya responde la crítica "barata" del revisor.

Si algo se rompe en el pipeline estadístico o en el generador de episodios, se
descubre en el sub-experimento fácil y no en el que importa (007b/007c).

---

## 1. Diseño de 007a

| Parámetro | Valor |
|---|---|
| Tipo de concepto | NATIVE (1 concepto = 1 categoría CIFAR-10) |
| Categorías (5) | frog, horse, ship, automobile, airplane |
| Etiquetas Carroll | slithy, mimsy, vorpal, borogove, tulgey |
| Imágenes por categoría | 200 (pool; embeddings DINOv2-small, 384-d, congelados) |
| Learners (6) | centroid, kcentroid (K=2), exemplar_knn (k=3), logreg (C=1), linsvm (C=1), random |
| Ways (N) | {3, 5} |
| Shots (K) | {1, 3, 5, 10, 20} |
| Episodios por celda | 30 (pareados entre learners) |
| Query por concepto | 50 |
| Master seed | 42 |
| Filas en el ledger | 2 ways × 5 shots × 30 ep × 6 learners = **1800** |

**Hipótesis evaluada en 007a — H2 (control de no-trivialidad):** en conceptos
nativos NO hay disociación de learners (centroid ≈ exemplar ≈ baselines).
Confirma que cualquier brecha que aparezca en 007b/007c es propiedad del *tipo
de concepto*, no del learner.

---

## 2. Arquitectura construida

Todo el código nuevo es reutilizable por 007b/007c y está cubierto por tests
(89 tests pasan: 40 preexistentes + 49 nuevos).

```
src/learners/            Learner ABC + 6 learners (contrato score-es-similitud,
                         guardia pre-fit, predict = argmax(score))
src/eval/
  concepts.py            Concept / ConceptType + builder NATIVE (disjunctive y
                         cross-cutting quedan stub para 007b/007c)
  descriptors.py         NS_silhouette, compactness, centroid_lands_outside,
                         NMI_native, substrate_diagnostic
  episodes.py            generador N-way K-shot PAREADO y determinista
  metrics.py             C1, C2 (conjunto válido X_ℓ), margin, OOV-AUROC,
                         prototype_failure_gap, NMI(pred, categoría)
  stats.py               bootstrap_ci, paired_bootstrap, wilcoxon, effect_size,
                         paired_test (decisión H1/H3 DIRECCIONAL)
  falsification.py       random_label, permuted_binding, oov_probe (007b/007c)
  harness.py             puente Episode → arrays → fila de métricas por episodio
experiments/exp_007_concept_carving/
  _shared.py             grilla + cache de embeddings (build-once .npz)
  run_exp007.py          loop de condiciones → ledger.jsonl + descriptors.csv
  analyze_exp007.py      agregados, CIs, tests pareados, figura
```

### Invariante crítico — episodios pareados

Un único objeto `Episode` se genera **una vez** y se entrega **idéntico** a los
seis learners. El loop es **episodios afuera, learners adentro**, de modo que
cada learner ve exactamente el mismo split support/query. Esto es lo que vuelve
válidos el `paired_bootstrap` y el `wilcoxon` de `stats.py`: la diferencia por
episodio `d_i = Acc(exemplar)_i − Acc(centroid)_i` es un par genuino. Si se
muestreara de forma independiente por learner, todo el test H1 quedaría
invalidado en silencio. Está protegido por
`tests/eval/test_episodes.py::test_pairing_identical_across_passes` (regenera los
episodios y exige splits idénticos).

---

## 3. Ejecución

1. **Cache de embeddings (build-once).** DINOv2-small codificó 1000 imágenes
   (5 categorías × 200) en CPU a ~0.15 s/img (~2.5 min) y se guardó en
   `.cache/embeddings_exp007/cifar_native.npz`. Las corridas posteriores no
   vuelven a importar torch.
2. **Grilla.** 1800 evaluaciones learner×episodio (fit + C1 + C2 + margin + NMI).
3. **Análisis.** Agregación con bootstrap, tests pareados y figura.

Salidas en `results/exp_007/`:

| Archivo | Contenido |
|---|---|
| `ledger.jsonl` | 1800 filas — una por (sub, learner, ways, shots, episodio) |
| `descriptors.csv` | NS / compactness / centroid_lands_outside / NMI_native por concepto |
| `aggregate.csv` | media + IC95% bootstrap de c1/c2/margin/NMI por celda |
| `paired_tests.csv` | exemplar_knn vs cada learner (diff, IC, Wilcoxon p, tamaño de efecto, supported) |
| `accuracy_vs_shots.png` | C1 vs shots, una curva por learner, faceta por ways |

---

## 4. Resultados

### 4.1 Descriptores de concepto (espacio congelado, antes de aprender)

| concept_id | label | categoría | NS_silhouette | compactness | centroid_lands_outside | NMI_native |
|---|---|---|---|---|---|---|
| 0 | slithy | frog | 0.2435 | 0.678 | False | 1.0 |
| 1 | mimsy | horse | 0.3284 | 0.5972 | False | 1.0 |
| 2 | vorpal | ship | 0.2678 | 0.6368 | False | 1.0 |
| 3 | borogove | automobile | 0.2624 | 0.6401 | False | 1.0 |
| 4 | tulgey | airplane | 0.1833 | 0.7152 | False | 1.0 |

Lectura: NS_silhouette 0.18–0.33 (consistente con el 0.283 de exp_000),
`centroid_lands_outside = False` en **todos** los conceptos nativos (el
centroide cae sobre sus propios miembros, no en tierra de nadie) y NMI_native =
1.0 (la etiqueta nativa es un *rename* de la categoría). Este es el extremo de
**NS alto / NMI alto** de la curva de colapso H1 que 007b/007c extenderán hacia
la izquierda.

### 4.2 C1 — precisión de denominación (Condición 1, macro)

Media [IC95%], 30 episodios por celda. Se muestran 1, 5 y 20 shots.

**3-way (azar = 0.333):**

| learner | K=1 | K=5 | K=20 |
|---|---|---|---|
| centroid | 0.912 [0.883, 0.934] | 0.981 [0.975, 0.986] | 0.989 [0.984, 0.992] |
| kcentroid | 0.912 [0.883, 0.934] | 0.978 [0.971, 0.984] | 0.986 [0.982, 0.990] |
| exemplar_knn | 0.912 [0.883, 0.934] | 0.973 [0.962, 0.982] | 0.986 [0.982, 0.990] |
| logreg | 0.911 [0.882, 0.934] | 0.981 [0.975, 0.986] | 0.989 [0.985, 0.992] |
| linsvm | 0.911 [0.881, 0.933] | 0.983 [0.978, 0.987] | 0.989 [0.986, 0.992] |
| random | 0.326 [0.311, 0.340] | 0.332 [0.315, 0.350] | 0.335 [0.321, 0.347] |

**5-way (azar = 0.200):**

| learner | K=1 | K=5 | K=20 |
|---|---|---|---|
| centroid | 0.848 [0.824, 0.871] | 0.969 [0.964, 0.974] | 0.981 [0.977, 0.984] |
| kcentroid | 0.848 [0.824, 0.871] | 0.962 [0.955, 0.968] | 0.978 [0.974, 0.983] |
| exemplar_knn | 0.848 [0.824, 0.871] | 0.959 [0.955, 0.964] | 0.974 [0.970, 0.977] |
| logreg | 0.856 [0.832, 0.879] | 0.970 [0.964, 0.975] | 0.981 [0.978, 0.984] |
| linsvm | 0.857 [0.833, 0.879] | 0.971 [0.966, 0.976] | 0.982 [0.978, 0.985] |
| random | 0.204 [0.196, 0.212] | 0.197 [0.188, 0.206] | 0.200 [0.191, 0.209] |

Los cinco learners reales coinciden dentro del ruido; `random` se queda clavado
en azar. En K=1 los tres learners por similitud (centroid/kcentroid/exemplar)
son **idénticos** por construcción: con un solo seed por concepto, el único
sub-centroide, el único centroide y el único exemplar son el mismo vector.

### 4.3 C2 — grounding inverso (Condición 2, etiqueta → imagen)

Precisión de recuperación (media, macro). Patrón estable en todas las celdas:

- centroid / kcentroid / exemplar_knn: **0.85 → 0.99** según shots.
- logreg / linsvm: consistentemente **algo más alto** (p. ej. 5-way K=1:
  linsvm C2 = 0.909, logreg 0.902, vs 0.854 de los basados en similitud), porque
  el hiperplano de decisión ordena mejor los candidatos en este test discreto.

C2 alto en todos los learners confirma que en conceptos nativos el grounding
inverso es trivial — coherente con H2.

### 4.4 Test pareado — H2 (la pregunta central de 007a)

`paired_tests.csv`, comparación `exemplar_knn_vs_centroid` sobre C1 (la cantidad
de H1, el *prototype-failure gap*). El test es **direccional**: "supported"
exige que la brecha sea positiva (episódico > prototipo) Y significativa.

| ways | shots | diff | IC95% | Wilcoxon p | supported |
|---|---|---|---|---|---|
| 3 | 1 | 0.000 | [0.000, 0.000] | 1.000 | False |
| 3 | 3 | −0.0058 | [−0.0095, −0.0022] | 0.006 | False |
| 3 | 5 | −0.0080 | [−0.0138, −0.0027] | 0.008 | False |
| 3 | 10 | −0.0047 | [−0.0096, −0.0007] | 0.071 | False |
| 3 | 20 | −0.0024 | [−0.0065, 0.0018] | 0.441 | False |
| 5 | 1 | 0.000 | [0.000, 0.000] | 1.000 | False |
| 5 | 3 | −0.0120 | [−0.0180, −0.0069] | 0.0004 | False |
| 5 | 5 | −0.0096 | [−0.0132, −0.0057] | 0.0002 | False |
| 5 | 10 | −0.0083 | [−0.0116, −0.0049] | 0.0004 | False |
| 5 | 20 | −0.0068 | [−0.0099, −0.0040] | 0.0003 | False |

**Disociaciones prototipo→episódico: 0 / 10 celdas.** El signo de la brecha es
*negativo* (centroid marginalmente mejor) y la magnitud máxima es 1.2 puntos
porcentuales. No hay fallo del prototipo en conceptos nativos → **H2 se
sostiene**.

Sanidad: `exemplar_knn_vs_random` es supported en **10/10** celdas (diff
0.59–0.77, p ≈ 0). El test detecta brechas reales cuando existen.

---

## 5. Hallazgo metodológico (capturado al correr 007a)

Con 30 episodios fuertemente pareados, una diferencia de **0.5–1.2 pp** ya cruza
significancia. En una versión previa, `paired_test.supported` era
**agnóstico a la dirección** (`la CI excluye 0` Y `p < 0.05`), de modo que
exemplar-vs-centroid quedaba marcado "supported" en 6/10 celdas nativas — **pero
con signo negativo** (centroid algo mejor). Eso NO es un fallo del prototipo.

Corrección aplicada en `src/eval/stats.py`: `supported` ahora es **direccional**
(`ci_low > 0` Y `p < 0.05`), porque H1/H3 son hipótesis de una cola
(episódico > prototipo). Se conserva `excludes_zero` para la información de dos
colas (p. ej. detectar que un baseline es significativamente *peor*). Tras la
corrección: 0/10 falsas disociaciones, 10/10 exemplar>random. Hay un test de
regresión: `tests/eval/test_stats.py::test_paired_test_is_directional`.

**Implicación para 007b/007c:** la decisión de H1/H3 debe usar esta bandera
direccional. Una brecha significativa en la dirección equivocada no es evidencia
de disociación.

---

## 6. Interpretación

- **H2 confirmada.** En conceptos nativos los seis learners (salvo `random`)
  rinden igual dentro del ruido (C1 0.85–0.91 con 1 shot → 0.97–0.98 con 5–20
  shots). No hay disociación prototipo/episódico. Por lo tanto, cualquier brecha
  que aparezca en 007b (disjuntivos) o 007c (cross-cutting) será atribuible al
  *tipo de concepto*, no al learner — que es justo lo que H2 debía garantizar.
- **El éxito nativo es "fácil" para todos los mecanismos**, incluido el centroide
  simple. Esto es coherente con la crítica del revisor (sobre nativos, el
  centroide basta) y prepara el contraste: la pregunta es si ese éxito sobrevive
  cuando la extensión deja de coincidir con un cluster DINOv2.
- **Conexión con los hallazgos previos.** exp_005/005b/006 ya mostraron que la
  geometría compartida de DINOv2 domina la alineación entre agentes bajo toda
  perturbación probada. 007a ancla el extremo de NS alto; 007b/007c miden hasta
  dónde puede la capa léxica *tallar* regiones que la percepción congelada no
  preempaquetó.

---

## 7. Reproducibilidad

- Todas las semillas registradas: `MASTER_SEED = 42`; RNG por episodio derivado
  de `SeedSequence([master_seed, episode_id])` (idéntico sin importar cuántos
  learners lo consuman); seed por learner = `episode_id`.
- DINOv2-small congelado; embeddings cacheados y deterministas.
- Idempotencia: re-correr `--sub 007a` reescribe sólo las filas de 007a en el
  ledger.

**Comandos:**

```
python experiments/exp_007_concept_carving/run_exp007.py --sub 007a
python experiments/exp_007_concept_carving/analyze_exp007.py
python -m pytest tests/learners tests/eval -q
```

---

## 8. Próximos pasos (plan §11)

1. **007b — DISJUNCTIVE** (próximo, máximo retorno): implementar
   `concepts.build_disjunctive_concepts` usando `inter_category_distance_matrix`
   (ya cableado en `_shared`) para construir ≥5 uniones de categorías
   perceptualmente *distantes* que recorran un rango de NS. Es el test directo
   de H1 (colapso del centroide) y H3 (tallado genuino). Figura central:
   Accuracy vs NS(c), una curva por learner.
2. Controles de falsación (random-label, permuted-binding, OOV) sobre 007b.
3. **007c — CROSS_CUTTING**: generar `data/synthetic_shapes/` (formas×colores),
   correr primero el diagnóstico de sustrato (DINOv2 debe organizar por forma,
   no por color) y luego el detector de "trampa por identidad" (H4).

---

*Generado tras la ejecución de 007a el 2026-06-05. Los archivos `.csv`/`.jsonl`/
`.png` citados están en esta misma carpeta. Recordatorio del proyecto: los
resultados no se suben a GitHub.*
