# Plan de análisis con datos humanos secundarios — borrador para revisión del autor

**Estado.** El plan quedó escrito el 28-10-2026 (hora de Chile, UTC−3). **No se ha corrido ningún análisis** y no se ha cruzado ningún dato humano con nuestras métricas ni con la precisión de los aprendices. Hasta ahora solo se contó la **cobertura** (qué clases y pares existen en cada fuente); los conteos están en `human_data_scouting/`. Los criterios de inclusión se fijaron conociendo esa cobertura, pero sin conocer ningún resultado. Cuando el autor apruebe este documento, se congela: se registra su hash SHA-256 y la fecha, y la decisión de preregistrarlo en OSF queda en manos de el autor. Cualquier desviación posterior se anota en un registro de desviaciones con su motivo.

**Naturaleza de los datos.** Todos los datos humanos son **secundarios**: los experimentos los diseñaron y corrieron otros equipos, con sus propias tareas, estímulos y participantes. Nosotros solo aplicamos nuestras métricas a esos datos publicados. Así se dirá en el texto, en cada tabla y en cada figura.

---

## 1. Criterios de compatibilidad e inclusión

Un dataset entra en el análisis **confirmatorio** solo si cumple los cinco criterios:

| Código | Criterio | Umbral operativo |
|---|---|---|
| **K1** | Mismas categorías o mapeo documentado | El dataset usa clases CIFAR o tiene un mapeo a ellas mediante IDs WordNet o wnids de ILSVRC. El mapeo se guarda en un archivo versionado **antes** de calcular cualquier distancia. Para CIFAR-100 se usa el mapeo corregido del Apéndice C del paper; los mapeos manuales adicionales se listan uno por uno. *Excepción en el módulo de aprendizaje (K1′):* se acepta una fuente sin categorías CIFAR si los **estímulos exactos** de cada ensayo son públicos y nuestro encoder puede procesarlos tal cual |
| **K2** | Cobertura mínima | *Módulo de disociación (CIFAR-100):* ≥30 de los 100 pares muestreados con medida humana fiable y ≥4 pares por cuadrante. *Módulo CIFAR-10:* ≥7/10 pares de tier o ≥30/45 pares. *Módulo de aprendizaje:* datos por ensayo con identificador de cada imagen |
| **K3** | Tipo de juicio comparable | La medida humana debe ser (a) similitud o confusabilidad **visual entre categorías, juzgada sobre imágenes**, que funciona como eje perceptual, o (b) generalización de una **palabra nueva** a partir de imágenes. Quedan fuera las normas solo léxicas sin imágenes, la relación o asociación semántica y la ambigüedad de anotación |
| **K4** | Licencia | Licencia explícita que permita uso en investigación, verificada en la fuente primaria. Si la licencia es NC, SA o ND, se respetan sus restricciones en lo que se redistribuya |
| **K5** | Acceso reproducible | Descarga pública sin login, de modo que un revisor pueda reproducir el análisis |
| **K6** | Controles de representación (no humanos) | Un modelo preentrenado entra solo como espacio alternativo al de DINOv2 si sus pesos son públicos, con licencia explícita verificada en el model card y aplicables a nuestras mismas imágenes. Nunca cuenta como fuente humana ni entra en la familia confirmatoria |

### Aplicación a los candidatos de `revision_datos_humanos.md`

| Fuente | K1 | K2 (cobertura medida) | K3 | K4 | K5 | Decisión |
|---|---|---|---|---|---|---|
| THINGS odd-one-out | ✔ (IDs WordNet en los metadatos; mouse, pepper y tank con mapeo manual) | ✔ 64/100 pares (Q1 14, Q2 16, Q3 19, Q4 15); CIFAR-10: 10/10 pares de tier | ✔ (a), juicios sobre imágenes de objetos | ✔ CC BY 4.0 | ✔ | **Entra (confirmatorio, CIFAR-100 y CIFAR-10)** |
| ImageNet-HSJ | ✔ (wnids; aquarium_fish→16 wnids de peces, etc., todo documentado) | ✔ con margen estrecho: 32/100 pares con ≥10 instancias (Q1 11, Q2 **4**, Q3 9, Q4 8); CIFAR-10: 7/10 pares de tier (faltan los 3 con *deer*) | ✔ (a), fotos naturales | ✔ CC BY 4.0 | ✔ (la medida por categoría no necesita las imágenes) | **Entra (confirmatorio, réplica).** Se marca como frágil porque Q2 tiene solo 4 pares |
| CIFAR-10H | ✔ (mismas imágenes) | ✔ 45/45 pares (módulo CIFAR-10); no aplica a CIFAR-100 | ✔ (a), confusión al clasificar | ✔ CC BY-NC-SA 4.0 | ✔ | **Entra (solo módulo CIFAR-10)** |
| Lewis & Frank 2018 (XTMEM) | K1′ ✔ (45 fotos públicas, IDs por ensayo) | ✔ 7.200 ensayos | ✔ (b) | ✔ CC BY 4.0 (OSF); procedencia de las fotos no documentada | ✔ | **Entra (módulo de aprendizaje)**, con una condición previa: la composición del arreglo de prueba debe poder reconstruirse desde el código del experimento (§3, H4). Si no se puede, queda fuera |
| model-vs-human | ✔ | ✘ 4/10 pares de tier y 21/45 pares; 2/100 en CIFAR-100 | ✔ (a) | ✔ CC BY 4.0 | ✔ | **Fuera** del confirmatorio. Opcional como anexo descriptivo |
| Sobreextensiones (Ferreira Pinto & Xu) | ✔ vía THINGSplus | ✔ 196/236 pares (fuera de CIFAR) | ~ (palabra→objeto en niños, sin imágenes de la situación) | ✘ sin licencia declarada | ✔ | **Fuera del confirmatorio.** Exploratorio solo si el autor decide resolver el permiso |
| Wordbank CDI | ✔ | 10/10 clases CIFAR-10; 47/100 clases CIFAR-100 (44 palabras) | ✘ normas léxicas | ✔ CC BY 4.0 (algunos BY-NC) | ✔ | **Solo exploratorio** |
| Glasgow Norms | ✔ | 10/10; 89/100 clases | ✘ normas léxicas | ✔ CC BY 4.0 | ✔ | **Solo exploratorio o covariable** |
| Peekbank | ? | no medida | ✔ (b), niños | ? no hay CC explícita | ✘ (login en Redivis) | **Fuera.** Trabajo futuro, o reevaluar si el autor aprueba crear la cuenta |
| Peterson et al. 2018 | ✘ sin etiquetas | — | ✔ (a) | ✘ datos sin licencia | ✔ | **Fuera** |
| SWOW / SimLex-999 / MEN | — | SimLex: 3/45 y 4 pares; los otros no se midieron | ✘ relación o asociación | ✘ ND / no explícita / no verificada | MEN: URL caída | **Fuera** |
| ImageNet-ReaL | ✔ | no medida | ✘ ambigüedad de anotación | ✔ Apache-2.0 | ✔ | **Fuera** |
| Kuperman 2012 / Brysbaert 2014 | — | — | ✘ | ✘ no explícita en la fuente primaria | — | **Fuera** (se reemplazan por Glasgow) |
| SAYCam / Labeled-S (Sullivan et al. 2021; Vong et al. 2024) como **dato humano** | ~ (Labeled-S: categorías de objetos en video egocéntrico; cobertura CIFAR no medida) | no medida | ✘ no es un juicio humano de similitud ni de generalización (son fotogramas y transcripciones de la experiencia de un niño) | Databrary, términos de uso propios | ✘ según el README de multimodal-baby, el acceso requiere solicitarlo en Databrary con aprobación del IRB institucional | **Fuera como dato humano** |
| Encoder visual CVCL (Vong et al. 2024; HF `wkvong/cvcl_s_dino_resnext50_embedding`) como **espacio de representación alternativo** | ✔ se aplica a nuestras mismas imágenes CIFAR | ✔ 100/100 pares y 45/45 (no requiere datos humanos) | no aplica (no es dato humano) | ✔ MIT (model card y repositorio de código) | ✔ pesos públicos sin gating | **Entra solo como control exploratorio (H6)**, no como fuente humana |

---

## 2. Variables comunes

**Del paper, ya calculadas y sin cambios:**
- `acc_pair`: precisión media de *naming* del aprendiz centroide (el mismo aprendiz y la misma variable que dan β_perc en el paper) en 30 episodios 3-way 5-shot por par. Hay una por cada uno de los 100 pares muestreados.
- `acc_tier`: precisión por concepto de tier en CIFAR-10. Hay 10.
- `d_perc`: 1 − coseno entre los centroides DINOv2 de las clases.
- `d_sem`: 1 − Wu–Palmer con el **mapeo corregido** como análisis primario; el mapeo original queda como sensibilidad.
- `quadrant`: se mantienen los cuadrantes del muestreo original.

**Humanas, una por fuente.** Todas se expresan como distancia; mayor = más distinto.
- `d_THINGS(a,b)`: 1 − similitud del embedding SPoSE de 66 dimensiones publicado. La similitud es la probabilidad de elegir el par (i,j) como el más parecido en un triplete, promediada sobre todos los terceros k, como definen Hebart et al. Si la matriz de similitud publicada está disponible, se usa esa. Si una clase CIFAR mapea a varios conceptos THINGS, se promedia.
- `d_HSJ(a,b)`: 1 − s_ab, donde s_ab es la proporción de veces que una referencia de la clase b fue elegida entre las 2 más parecidas cuando la consulta era de la clase a (y simétricamente con b como consulta). Se ponderan ambas direcciones por su número de instancias. El azar es 2/8 = 0,25. Un par requiere ≥10 instancias.
- `d_C10H(a,b)`: 1 − confusión simétrica, con c_ab = proporción de etiquetas humanas *b* en imágenes de clase *a* (de `cifar10h-counts.npy`), y la confusión simétrica = (c_ab + c_ba)/2. Los análisis usan rangos, así que la transformación no afecta.
- Para comparar entre fuentes, cada distancia se estandariza (z) dentro de su fuente sobre todos los pares que cubre.

---

## 3. Análisis

Para cada análisis se indica: hipótesis (H), predicción, variables, modelo, exclusiones y criterio de decisión. **El tamaño mínimo de interés (SESOI)** es |ρ| o |r_parcial| = 0,20. Todo se reporta, sea cual sea el resultado.

### H1. Convergencia: ¿el espacio del aprendiz coincide con la similitud humana? (confirmatorio)
- **Predicción:** ρ_Spearman(d_perc, d_humana) > 0 en cada fuente.
- **Variables:** todos los pares cubiertos. THINGS: 3.486 de los 4.950 pares de CIFAR-100. HSJ: ≈1.402 pares con ≥10 instancias. CIFAR-10H: 45 pares de CIFAR-10. THINGS y HSJ también se miden sobre los 45 pares de CIFAR-10.
- **Modelo:** Spearman con IC 95 % por bootstrap de pares (10.000 réplicas, semilla 42). Como los pares comparten clases, también se reporta una prueba de Mantel con permutación de clases (10.000 permutaciones).
- **Decisión:**
  - *Apoyo:* ρ > 0 y p (Mantel, Holm) < 0,05.
  - *No apoyo:* el IC 95 % queda completo por debajo de 0,20.
  - *Inconcluso:* cualquier otro caso.

### H2. Disociación con eje humano (confirmatorio, CIFAR-100: THINGS y HSJ)
- **Predicción:** β_humana < 0 (más distancia humana → mayor precisión del aprendiz). β_sem nulo o pequeño.
- **Modelo:** el mismo modelo de dos predictores del paper, por fuente: `acc_pair ~ z(d_humana) + z(d_sem)`, OLS con errores HC3. Usa los pares muestreados cubiertos: THINGS n = 64, HSJ n = 32.
- **Decisión:**
  - *Apoyo:* β_humana < 0 con p (Holm) < 0,05, **y** β_sem con p ≥ 0,10 o con R² parcial menor que un tercio del de d_humana. Es el mismo criterio que la regla A2 del paper.
  - *No apoyo:* el IC del r parcial de d_humana excluye valores ≤ −0,20, o β_humana > 0 con significación.
  - *Inconcluso:* el IC incluye tanto 0 como −0,20. Con n = 32–64 este desenlace es plausible y se dirá así.
- **Robustez (se reporta, no decide):** mapeo WordNet original; umbral de HSJ en ≥1 y en ≥30 instancias; la regla A3/A4 del paper (Q3 ∪ Q4) con el eje humano. Esta última es descriptiva, porque deja n pequeño.

### H2b. ¿El espacio del aprendiz aporta más allá de la similitud humana? (circularidad)
- **Modelo:** `acc_pair ~ z(d_perc) + z(d_humana) + z(d_sem)` por fuente.
- **Predicción, sin decisión confirmatoria:** β_perc sigue siendo < 0, porque el aprendiz usa su propio espacio.
- **Interpretación:**
  - Si d_humana absorbe a d_perc, la estructura perceptual humana basta para explicar la dificultad.
  - Si d_perc conserva su efecto y d_humana no lo tiene, el resultado se acota a "coherencia en el espacio del aprendiz".
  - Se reporta la diferencia β_perc − β_humana sobre el mismo subconjunto, con IC por bootstrap pareado.

### H3. Tiers CIFAR-10 (descriptivo)
- **Variables:** `acc_tier` frente a la d_humana del par que forma cada concepto. Hay n = 10 con THINGS y CIFAR-10H, y n = 7 con HSJ.
- **Análisis:** Spearman y gráfico, **sin inferencia confirmatoria**, porque n ≤ 10.
- **Complemento con CIFAR-10H a nivel imagen:** la entropía de las etiquetas humanas por imagen frente al margen del aprendiz (distancia al centroide propio menos distancia al centroide rival más cercano), en las 10.000 imágenes de prueba. Es exploratorio y se reporta como ρ con IC.

### H4. Generalización de palabras nuevas en adultos (confirmatorio, Lewis & Frank 2018)
- **Condición previa:** reconstruir el arreglo de prueba de cada ensayo desde `experiments/` del repositorio. Si no es posible, H4 se excluye y se informa.
- **Predicción:** cuanto mayor es la distancia DINOv2 de un ítem de prueba al centroide de los ejemplos de entrenamiento, menor es la probabilidad de que la persona lo elija, **controlando el nivel taxonómico** del ítem (sub, básico o superordinado).
- **Variables:**
  - `selected` (0/1) por ensayo × ítem de prueba;
  - `dist` = 1 − coseno entre el ítem y la media de los embeddings DINOv2 de las fotos de entrenamiento (`trainPics`);
  - `level` del ítem, tomado del nombre de archivo;
  - `condition` (one, three_sub, three_basic, three_sup);
  - `subid` e `image`.
- **Modelo:** GLMM logístico `selected ~ z(dist) + level + condition + (1 + z(dist) | subid) + (1 | image)`. Si no converge, se quita la pendiente aleatoria; ese reemplazo queda fijado de antemano.
- **Exclusiones:** se aplican las de los autores, tal como están implementadas en sus scripts. No se agregan exclusiones propias. Sensibilidad: excluir los experimentos 8 y 12, que no fueron preregistrados por los autores.
- **Decisión:**
  - *Apoyo:* β_dist < 0 con p (Holm) < 0,05.
  - *No apoyo:* el IC del odds ratio por DE excluye efectos con |log OR| ≥ 0,2.
  - *Inconcluso:* cualquier otro caso.
- **Exploratorio:** la interacción `dist × condition`, es decir, si la coherencia de 3 ejemplos cambia la pendiente.

### H5. Exploratorios (no confirmatorios; corrección BH dentro de este grupo)
- **AoA:** la coherencia de clase (dispersión intra-clase en DINOv2 sobre las imágenes de entrenamiento CIFAR) frente a la AoA de Wordbank (edad a la que ≥50 % produce la palabra, ajuste logístico por palabra, inglés WS) y frente a la AoA de Glasgow.
  - Modelo: Spearman parcial controlando la log-frecuencia.
  - n = 44 palabras (Wordbank) y 89 (Glasgow) en CIFAR-100; 10 en CIFAR-10.
  - Las clases que comparten palabra (las 5 de "tree") se colapsan.
- **Sobreextensiones:** solo si se resuelve la licencia. Regresión logística condicional de "par atestado" frente a 5 controles por palabra emparejados por frecuencia (semilla 42), con `d_DINO` (imágenes THINGSplus) y `d_sem` como predictores.

### H6. Control exploratorio del espacio de representación: encoder visual CVCL (no es dato humano)
- **Fuente verificada (28-oct-2026):** Vong, Wang, Orhan & Lake (2024), *Science*, doi 10.1126/science.adi1374. Código en github.com/wkvong/multimodal-baby (MIT). El README carga CVCL con `MultiModalLitModel.load_model("cvcl")`, que descarga `wkvong/cvcl_s_dino_resnext50_embedding` (model card: `license: mit`, sin gating). También existe `wkvong/cvcl_s_dino_vit_embedding` (MIT), que el README no usa por defecto. El model card no trae descripción más allá de la licencia.
- **Estatus:** robustez del espacio de representación. CVCL es un modelo entrenado con video egocéntrico de un solo niño (SAYCam-S; duración y número de categorías por verificar en el paper). **No es un dato humano** y no entra en la familia Holm.
- **Hipótesis exploratoria:** si el límite depende de la coherencia en el espacio visual del aprendiz y no de la relación semántica, el patrón del paper (β_perc < 0, β_sem ≈ 0) debería reaparecer cuando ese espacio es el de CVCL.
- **Variables:** `d_perc_CVCL` = 1 − coseno entre centroides de embeddings del encoder visual de CVCL (L2-normalizados, preprocesamiento del propio repositorio) sobre las **mismas imágenes** y los **mismos 100 pares** (semilla 42); `acc_pair_CVCL` = mismo aprendiz por centroides, 30 episodios 3-way 5-shot; `d_sem` igual que en el paper (mapeo corregido, Apéndice C).
- **Modelos:** (a) `acc_pair_CVCL ~ z(d_perc_CVCL) + z(d_sem)`, OLS con HC3; (b) Spearman `d_perc_CVCL` vs `d_perc_DINOv2` con Mantel (10.000 permutaciones) sobre los 4.950 pares de CIFAR-100 y los 45 de CIFAR-10; (c) si H1 se ejecuta, Spearman `d_perc_CVCL` vs cada d_humana en los mismos pares, en paralelo a DINOv2; (d) tiers CIFAR-10, descriptivo.
- **Corrección:** BH dentro de H6 (pruebas a, b y c).
- **Consistente:** β_perc_CVCL < 0 con IC 95 % excluyendo 0 tras BH, y β_sem con IC dentro de ±SESOI (0,20) o p sin ajustar ≥ 0,10. **Inconsistente:** β_perc_CVCL con IC que incluye 0, o β_sem significativo tras BH. Otro patrón se reporta como no concluyente. Ningún resultado de H6 sustituye ni corrige los análisis confirmatorios.
- **Riesgos:** (1) diferencia de dominio: video egocéntrico de un niño frente a fotos CIFAR de 32×32 reescaladas, con riesgo de embeddings poco discriminativos, por lo que se reporta la precisión base del aprendiz con CVCL; (2) vocabulario y categorías limitados de SAYCam: muchas clases CIFAR (animales salvajes, vehículos) no aparecen en la experiencia del niño; (3) n = 1 niño, sin generalización a niños; (4) arquitectura distinta (ResNeXt-50 frente a ViT-S) que confunde dominio y arquitectura; la variante ViT puede reportarse como sensibilidad si el autor lo aprueba; (5) no se usa el codificador de texto de CVCL ni los datos SAYCam/Labeled-S, que requieren aprobación de IRB vía Databrary.

### Comparaciones múltiples
- **Familia confirmatoria (Holm, α = 0,05):** H1 ×3 fuentes, H2 β_humana ×2 fuentes y H4 ×1, en total 6 pruebas.
- El criterio de nulidad de β_sem usa p sin ajustar ≥ 0,10, que es la versión conservadora para declarar ausencia.
- H2b, H3 y la dispersión (§4) son descriptivos, con IC. H5 usa BH. H6 usa BH dentro de su propio grupo y no es dato humano.

---

## 4. Cómo se reporta la desviación entre fuentes y frente a nuestras métricas

Como no corrimos esos experimentos, la variación entre fuentes se reporta, no se promedia sin más:

1. **Acuerdo entre fuentes:** Spearman entre pares de fuentes en los pares compartidos.
   - THINGS ∩ HSJ: 1.109 pares de CIFAR-100 con ≥10 instancias; 22 de los muestreados.
   - THINGS, HSJ y CIFAR-10H: hasta 36 pares de CIFAR-10.
   - Se da con IC por bootstrap y prueba de Mantel. Ningún umbral determina aceptación o exclusión; solo se describe.
2. **Coeficientes por fuente:** β_humana y r parcial de cada fuente con IC 95 % en un *forest plot*, junto al β_perc del paper calculado **sobre el mismo subconjunto de pares**. Se reportan el rango (mín–máx) de los coeficientes y el **I² y τ** de un metaanálisis de efectos aleatorios. I² y τ son solo descriptivos, porque las fuentes comparten pares y no son independientes.
3. **Análisis combinado:**
   - (a) d_humana compuesta = media de los z disponibles por par, sobre la unión de pares (74 de los muestreados con el umbral de HSJ);
   - (b) modelo en formato largo `acc_pair ~ z(d_hum) × fuente + z(d_sem) + (1 | par)`, cuyo término de interacción mide la heterogeneidad entre fuentes (prueba LR);
   - (c) el mismo modelo restringido a la intersección.
   - Se reportan los tres. El combinado no reemplaza a los análisis por fuente.
4. **Desviación frente a nuestra métrica:**
   - por par, Δ = z(d_humana) − z(d_perc);
   - DE de Δ y gráfico de Bland–Altman;
   - lista de los 10 pares con mayor |Δ| por fuente, que se reportan, no se excluyen.
5. **Redacción:** el texto incluirá esta frase: "Human data are secondary: they were collected by other groups, with different tasks, stimuli and participants; differences between sources are reported as such and were not adjusted away."

---

## 5. Lo que no se puede medir con datos existentes (trabajo futuro deseable, no resultado)
- Un experimento de **aprendizaje de palabras con niños sobre nuestras categorías**: pares disyuntivos near/mid/far de CIFAR y cuadrantes Q3/Q4. Es la prueba directa de la hipótesis y no existe en datos públicos.
- La **misma tarea 3-way 5-shot con adultos**, con palabras nuevas para pares CIFAR, para comparar persona y aprendiz ítem a ítem.
- **Juicios humanos de similitud sobre las imágenes CIFAR mismas**, que cubrirían *deer*, personas, árboles y flores, ausentes en HSJ.
- **Peekbank** (looking-while-listening infantil) si se decide crear la cuenta en Redivis. Antes habría que verificar la cobertura y la licencia.
- Ninguno de estos se presenta como hallazgo. Van en Limitaciones y Trabajo futuro. No hay presupuesto para ellos ahora.

---

## 6. Encuadre

El encuadre, título y venue se decidirán después de revisar los resultados de este plan.

---

### Pendientes antes de congelar
1. Que el autor apruebe los umbrales de K2, el SESOI = 0,20 y los mapeos manuales: mouse, pepper, tank en THINGS; aquarium_fish y las clases recuperables en HSJ (mushroom, television).
2. Confirmar que el arreglo de prueba de XTMEM puede reconstruirse; es una lectura de código, no de resultados.
3. Decidir si se usa el umbral de HSJ en ≥10 instancias, dado que deja Q2 con 4 pares.

---

## Decisiones fijadas 2026-10-30

Aprobadas por el autor el 30-10-2026 antes de correr cualquier análisis (hora de Chile, UTC−3).

1. **Efecto mínimo relevante (SESOI):** |ρ| = 0,20 (y |r parcial| = 0,20 donde aplica), tal como figura en §3.
2. **Clases ambiguas:** mouse, pepper, tank y cualquier otra clase con más de un sentido posible se asignan al **sentido que muestran las imágenes CIFAR** (mouse = roedor; pepper = pimiento, *sweet pepper*; tank = tanque militar; etc.). Cada mapeo se documenta uno por uno en un archivo versionado en `human_data_analyses/` antes de calcular distancias.
3. **ImageNet-HSJ:** un par entra solo con **≥10 juicios/co-ocurrencias** (instancias) por par. Los umbrales ≥1 y ≥30 quedan solo como robustez (§3, H2).
4. **Cifras de Vong et al. 2024 (H6), verificadas el 30-10-2026:**
   - **61 horas** de video: verificado en el abstract (Europe PMC / PubMed, PMID 38300999, doi 10.1126/science.adi1374).
   - **22 categorías de evaluación de Labeled-S**: verificado en el archivo de resultados del repositorio oficial (`results/alignment/joint_embeddings_with_eval_sims_seed_0.csv`, 22 columnas de categoría; "cat" y "kitty" son la misma). El abstract no da esta cifra.
