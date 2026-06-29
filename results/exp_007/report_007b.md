# exp_007b — Disjunctive Concepts · Reporte de experimento

**Sub-experimento:** `007b` (DISJUNCTIVE — el payoff primario de H1/H3)
**Fecha de ejecución:** 2026-06-05
**Estado:** COMPLETO — **H1 NO soportada** (resultado de frontera, fuerte y honesto)
**Máquina:** HP i7-1165G7, 16 GB RAM, sin GPU. DINOv2-small en CPU.

> Lee primero `report.md` (007a, H2). Este documento asume ese contexto.

---

## 0. Pregunta

¿Puede algún mecanismo léxico adquirir un concepto cuya extensión NO coincide
con un cluster nativo de DINOv2 — específicamente una **unión arbitraria de
categorías perceptualmente distantes** (concepto disyuntivo)? Y si el centroide
falla, ¿lo rescata la memoria episódica (exemplar k-NN) o multi-ancla
(k-centroid con K = aridad)? Esa es la predicción central del marco
prototipo/episódico (plan §1, H1).

---

## 1. Preparación (las 6 directivas previas a 007b)

Antes de correr 007b se implementaron y verificaron seis requisitos:

1. **exemplar_knn con k ∈ {1, 3, 5}**, opción de ponderación por distancia
   (`weighted`, por defecto True: cada vecino vota con su similitud coseno).
2. **kcentroid con K ∈ {1, 2, 3}** (K = 1 colapsa al centroide — chequeo de
   sanidad; K = 2 = aridad de la disyunción es el ancla teórica).
3. **Sin comparaciones de `margin` entre learners**: las escalas de score
   difieren (coseno vs logits vs suma de similitudes), así que margin queda solo
   como diagnóstico intra-learner en `aggregate.csv`. La evidencia usa **C1, C2
   y la brecha prototipo-episódico**.
4. **Cache de las 10 categorías CIFAR-10** (`cifar_all10.npz`, 2000 imágenes)
   antes de construir conceptos disyuntivos.
5. **Descriptores primero + selección de pares**: ver §2.
6. **Lógica direccional del test pareado preservada** desde 007a
   (`supported = ci_low > 0 ∧ p < 0.05`).

Roster 007b (10 learners): centroid · kcentroid_k1/k2/k3 ·
exemplar_knn_k1/k3/k5 (ponderados) · logreg · linsvm · random.

---

## 2. Construcción y selección de conceptos disyuntivos (directiva 5)

**Regla de validez (plan §3):** cada concepto = unión de categorías
*arbitrarias y perceptualmente distantes* (no semánticamente coherentes), y los
conceptos deben ser **disjuntos por categoría** entre sí (de lo contrario una
imagen pertenecería a dos conceptos y la etiqueta sería ambigua). Con aridad 2 y
5 conceptos, un *matching* disjunto particiona las 10 categorías.

**Mecanismo de selección:** *matching* disjunto codicioso puntuado por
`distancia × min(NS de los miembros)`. El factor distancia fuerza uniones
distantes; el factor `min(NS de miembros)` empareja categorías **bien
separadas**, de modo que el NS bajo de una unión provenga de la *disyunción* y no
de un miembro ya degenerado — y, en consecuencia, aísla las categorías débiles.

**Hallazgo previo (corrección metodológica):** en DINOv2/CIFAR, **bird (NS =
0.078) y dog (NS = 0.124) son categorías nativas casi degeneradas**. Un primer
*matching* solo-por-distancia las emparejaba con categorías distantes,
contaminando la prueba (su NS bajo se heredaba del miembro débil). El criterio
`distancia × min(NS)` las aísla juntas (zorb = bird+dog) y deja 4 conceptos
"limpios" de categorías bien separadas. La referencia de "carving" se cambió de
NS **mínimo** nativo (sesgado por bird) a NS **mediano** nativo (= 0.234).

**Conjunto seleccionado (ordenado por NS ascendente = más difícil primero):**

| concepto | miembros | NS unión | NS miembros | dist. categorías | < mediana nativa |
|---|---|---|---|---|---|
| zorb | bird + dog | 0.045 | 0.078, 0.124 | 0.477 | sí |
| gleth | airplane + cat | 0.077 | 0.193, 0.195 | 0.678 | sí |
| mivor | frog + truck | 0.112 | 0.227, 0.309 | 0.734 | sí |
| plonk | automobile + deer | 0.115 | 0.261, 0.240 | 0.649 | sí |
| quax | horse + ship | 0.136 | 0.312, 0.276 | 0.714 | sí |

Los 4 conceptos limpios (gleth, mivor, plonk, quax) tienen miembros de NS alto
(0.19–0.31): su colapso de NS (0.077–0.136) se debe genuinamente a la disyunción.
zorb (bird+dog) es el concepto "heredado" de baja NS, el más difícil.

`centroid_lands_outside = False` para los 5 — incluso en uniones distantes el
centroide queda más cerca de sus propios miembros que de un intruso. Es un
hallazgo en sí: el fallo del prototipo (cuando ocurre) no se debe a que el
centroide caiga literalmente fuera, sino a que queda lejos de *ambos* modos.

---

## 3. Diseño de la corrida

| Parámetro | Valor |
|---|---|
| Tipos de concepto | NATIVE (10 categorías, ancla de NS alto) + DISJUNCTIVE (5 pares disjuntos) |
| Learners | 10 (roster 007b) |
| Ways | {3, 5} |
| Shots | {1, 3, 5, 10, 20} |
| Episodios | 30, pareados entre learners |
| Query por concepto | 50 |
| Filas en el ledger | 6000 (3000 native + 3000 disjunctive) |

Ambos tipos se corren juntos para que la curva Accuracy-vs-NS (Figura 1) salga
de una sola corrida, con granularidad por concepto (`per_concept_c1`).

---

## 4. Resultados

### 4.1 C1 — denominación (ways = 5, K = 5), media [IC95%]

| learner | NATIVE | DISJUNCTIVE |
|---|---|---|
| centroid | 0.920 [0.906, 0.935] | 0.459 [0.451, 0.468] |
| kcentroid_k2 (K=aridad) | 0.904 [0.888, 0.921] | 0.467 [0.458, 0.477] |
| exemplar_knn_k3 | 0.896 [0.879, 0.912] | 0.459 [0.450, 0.468] |
| logreg | 0.930 [0.917, 0.944] | 0.474 [0.465, 0.483] |
| linsvm | 0.931 [0.918, 0.944] | 0.469 [0.460, 0.478] |
| random | 0.195 [0.186, 0.204] | 0.196 [0.187, 0.205] |

**El centroide colapsa** de 0.92 (native) a 0.46 (disjunctive). Pero
**todos los learners por similitud colapsan igual** (~0.46). Las disyuntivas
están muy por encima del azar (0.20) pero lejos de adquisición.

### 4.2 H1 — brecha prototipo-episódico (test pareado direccional, C1)

`exemplar_knn / kcentroid` vs `centroid` en conceptos disyuntivos:

- **Episódico/multi-ancla vencen al centroide (supported) en 1 de 50 celdas.**
  La única (kcentroid_k2 @ K=5, +0.008) desaparece a K=20.
- exemplar_knn_k3 vs centroid @ K=5: diff = −0.0001, no significativa.
- Figura 2: la brecha está centrada en ~0 (ligeramente negativa) tanto en native
  como en disjunctive. Si H1 valiera, la caja disyuntiva estaría claramente
  sobre cero.

**H1 NO soportada.** La memoria episódica y la multi-ancla NO recuperan los
conceptos disyuntivos mejor que el prototipo.

### 4.3 Baselines lineales — recuperación parcial

`logreg / linsvm` vs `centroid` en disyuntivos: **supported en 9 de 20 celdas**,
y la brecha CRECE con los shots (logreg @ K=20: +0.025, p < 0.001; en zorb,
logreg = 0.604 vs centroid = 0.480). Un hiperplano global recupera algo más que
cualquier learner basado en similitud, sobre todo con más ejemplos — pero la
precisión absoluta sigue siendo ~0.47–0.60.

### 4.4 Verificación de implementación (no es un bug)

Que exemplar/kcentroid no superen al centroide es sorprendente, así que se
verificó con la geometría XOR canónica (donde los centroides de dos conceptos
coinciden por construcción): centroid = 0.495 (azar), **kcentroid_k2 = 1.000,
exemplar_k3 = 1.000**. Los learners SÍ disocian cuando la geometría lo permite.
El resultado nulo en CIFAR es real.

---

## 5. Por qué colapsan todos (interpretación)

DINOv2 organiza su espacio por **similitud perceptual/semántica** (animal vs
vehículo es el eje dominante). Los conceptos disyuntivos mezclan, a propósito,
una categoría animal con una de vehículo (gleth = avión+gato, mivor = rana+camión,
…). Entonces, para una imagen-consulta *animal*, sus vecinos más cercanos son
*otros animales* que están repartidos en TODOS los demás conceptos disyuntivos.
La memoria episódica falla por la misma razón que el prototipo: **el vecino más
cercano pertenece, con alta probabilidad, al concepto equivocado.** Ningún
mecanismo que opere sobre la percepción congelada puede recuperar una partición
que corta transversalmente la estructura de similitud de esa percepción.

Esto es coherente con exp_005 / 005b / 006: la geometría de DINOv2 domina. 007b
lo lleva a su conclusión: esa geometría no es solo un atractor de *alineación
entre agentes*, sino un **techo de adquisición** — define qué conceptos son
aprendibles y cuáles no, con independencia del mecanismo léxico.

**NS no es suficiente por sí solo.** Las categorías nativas de baja NS (bird,
dog) siguen siendo nombrables (~0.82–0.90) porque son *unimodales*; las
disyuntivas de baja NS colapsan porque son *multimodales y transversales*. El
colapso lo provoca el TIPO de concepto (disyuntivo), no la NS baja per se. (En la
Figura 1 esto se ve como los picos a ~0.85 dentro de la banda izquierda: son las
nativas bird/dog.)

---

## 6. Veredicto frente a la tabla del plan §1

| Predicción | Resultado |
|---|---|
| H1 — centroide colapsa al bajar NS | **Premisa OK** (0.92 → 0.46) |
| H1 — episódico se mantiene | **FALLA** (episódico también 0.46) |
| H1 — disociación prototipo/episódico | **NO soportada** (1/50 celdas) |
| H3 — algún learner adquiere baja-NS por sobre el azar | **Parcial** (~0.46–0.60 > 0.20 azar, pero no es adquisición; los lineales recuperan algo más) |

Fila aplicable del plan §1: **"ambos colapsan → la percepción congelada no puede
soportar conceptos que no pre-empaquetó. Frontera dura del grounding sobre
percepción congelada → argumento a favor de percepción entrenable en GLA."**

Con un matiz: los **probes lineales recuperan modestamente más** que cualquier
learner por similitud, y la ventaja crece con los shots. Sugiere que parte de la
señal disyuntiva *es* linealmente decodificable del espacio congelado, pero ni de
lejos al nivel nativo.

---

## 7. Qué responde (y qué no) a la crítica del revisor

- **Defiende el experimento por el otro flanco:** el éxito nativo del centroide
  era, en efecto, "fácil" (todos los learners lo igualan, H2). Cuando se exige
  *carving* genuino (extensión que no coincide con un cluster nativo), la
  percepción congelada **no** lo logra con NINGÚN mecanismo léxico —
  prototipo, episódico, multi-ancla o lineal.
- **Honestidad obligatoria (plan §0):** esto NO rescata la tesis mayor de Vera
  et al. Mapea la frontera del grounding sobre percepción congelada y, de paso,
  falsa la predicción prototipo→episódico EN ESTE substrato. Va en Limitaciones.

---

## 8. Reproducibilidad y archivos

```
python experiments/exp_007_concept_carving/run_exp007.py --sub 007b
python experiments/exp_007_concept_carving/analyze_exp007.py
python -m pytest -q          # 96 tests
```

| Archivo | Contenido |
|---|---|
| `ledger.jsonl` | 1800 (007a) + 6000 (007b) filas; 007b incluye `per_concept_c1` |
| `descriptors.csv` | descriptores por concepto, native(10)+disjunctive(5) para 007b |
| `aggregate.csv` | media + IC95% por celda × learner |
| `paired_tests.csv` | cada learner vs centroid, C1 y C2, brecha direccional |
| `figure1_accuracy_vs_ns.png` | FIGURA CENTRAL — C1 por concepto vs NS(c) |
| `figure2_gap_boxplots.png` | brecha prototipo-episódico por tipo de concepto |

---

## 9. Próximos pasos

1. **exp_007c — CROSS_CUTTING** (formas × colores sintéticas). 007b muestra que
   las disyunciones sobre categorías reales colapsan porque cortan el eje
   animal/vehículo. 007c pregunta lo análogo para un atributo (color)
   ortogonal a la identidad (forma), con el diagnóstico de sustrato previo
   (DINOv2 debe organizar por forma, no por color) y el detector de "trampa por
   identidad" (H4).
2. **Controles de falsación** (random-label, permuted-binding, OOV) sobre 007b,
   para confirmar que el ~0.46 disyuntivo no es un artefacto.
3. **Implicación para GLA:** 007b es el argumento empírico directo a favor de
   *percepción entrenable* — el siguiente componente de gradiente después del
   LexicalAdapter de exp_005b.

---

*Generado tras 007b el 2026-06-05. Recordatorio del proyecto: los resultados no
se suben a GitHub.*
