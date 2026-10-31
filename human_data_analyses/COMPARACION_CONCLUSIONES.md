# Conclusiones originales de las fuentes de datos frente a los resultados del paper principal
2026-10-30, ~18:45 hora de Santiago. Este documento es descriptivo: no reencuadra el paper ni toca el plan congelado. Todas las citas salen de los textos primarios leídos hoy, y para cada una indico dónde se leyó. Los números de página vienen de los PDF; cuando la fuente es HTML o XML de PMC, indico la sección.

## 1. Hebart et al. 2020 (Nature Human Behaviour 4:1173–1185), 49 dimensiones
Texto completo leído en PMC7666026 (HTML).
- **(1) Conclusión**:
  - Abstract: las 49 dimensiones "reflect various conceptual and perceptual properties of those objects"; los juicios de similitud "can be captured by a fairly low-dimensional, interpretable embedding".
  - Discusión: las dimensiones reflejan "basic perceptual properties of shape, color, texture as well as more high level properties such as taxonomic membership, function, or value".
  - Resultados: "dimensions that appear to reflect the semantic membership of those objects, such as … food, animals, furniture, vehicles, or tools".
- **(2) Nosotros**: no usamos el embedding de 49 dimensiones. Usamos los juicios de THINGS (2023; ver §2). Con esos datos, H1 da apoyo, porque la distancia DINOv2 correlaciona con la distancia humana THINGS. H2 THINGS es inconcluso: la distancia humana no predice la precisión del aprendiz una vez controlada d_sem (p = .374).
- **(3) Complemento/tensión**: para los autores, la similitud humana THINGS es una mezcla de propiedades perceptuales y conceptuales, incluida la pertenencia taxonómica. Por eso, que DINOv2 se alinee con ella (H1) no la valida como medida *solo perceptual*. Es una alineación con una similitud que, según la fuente, incorpora categoría semántica.

## 2. Hebart et al. 2023 (eLife 12:e82580), THINGS-data
Texto completo leído en PMC10038662 (XML de Europe PMC). El PDF de eLife no descargó.
- **(1) Conclusión**:
  - Abstract: es un recurso, "4.70 million similarity judgments", con el objetivo de "testing countless hypotheses at scale".
  - Resultados: el embedding final revela "66 interpretable dimensions underlying perceived similarity judgments".
  - Introducción: el SPoSE original dio "49 interpretable behavioral dimensions reflecting perceptual and conceptual object properties (e.g. colorful, animal-related)".
  - Es un trabajo de datos, sin una tesis teórica sobre lo perceptual frente a lo semántico.
- **(2) Nosotros**:
  - H1 THINGS: apoyo.
  - H2 THINGS: inconcluso.
  - La distancia THINGS y la de HSJ coinciden solo moderadamente (ρ = .60 en 1.042 pares).
  - Los pares de mayor discrepancia con DINOv2 incluyen boy–girl, otter–seal, dolphin–whale y oak–pine: pares que la gente juzga cercanos por categoría y que DINOv2 separa, o al revés (ver `out/dispersion_results.json`).
- **(3) Complemento**: las cifras de pares discordantes cuadran con que la fuente describe la similitud como perceptual + conceptual. No es tensión con el paper original, que no sostiene una tesis al respecto.

## 3. Roads & Love 2021 (CVPR, pp. 3547–3556), ImageNet-HSJ
PDF de CVF leído.
- **(1) Conclusión**:
  - Abstract, p. 3547: "the more complex models that perform better on task-specific benchmarks do not better conform to human semantic judgments".
  - Introducción, p. 3547: la similitud humana refleja "a rich understanding of the world". Lo ejemplifican con una botella de cerveza que se parece a los cigarrillos "because both are age-restricted".
  - p. 3553: el modelo no supervisado (DeepCluster) "was one of the worst performing models … this result highlights the challenging semantic gap".
- **(2) Nosotros**:
  - H1 HSJ: apoyo.
  - H2 HSJ: fuera de las reglas del plan (β_hum < 0, p = .039, p Holm = .079; r parcial −.41 [−.67, −.07]).
  - Varias especificaciones de robustez (≥1 o ≥30 instancias, mapeos originales) quedan por debajo de .05, pero no son decisorias.
- **(3) Tensión de interpretación**: los autores llaman *semánticos* a los juicios HSJ. Esto encaja con que HSJ sea la única fuente donde la distancia humana se asocia con la precisión más allá de d_sem (WordNet). También queda abierto qué captura exactamente HSJ. Además, el hallazgo de Roads & Love sobre el modelo no supervisado (DeepCluster, de la era de las CNN) no se puede extrapolar a DINOv2. Nuestra H1 muestra alineación para DINOv2, pero no la comparamos con otros modelos.

## 4. Peterson et al. 2019 (ICCV; arXiv 1908.07086), CIFAR-10H
PDF de arXiv leído.
- **(1) Conclusión**:
  - Abstract: "explicit training on our dataset closes this gap, supports improved generalization … and confers robustness to adversarial attacks".
  - Discusión, §8: la incertidumbre humana ofrece "a way of measuring whether our learning algorithms are inferring good similarity structure (beyond just top-1 performance)"; "the bulk of human uncertainty is concentrated in approximately 30% of the images".
  - Introducción: "we cannot directly extract ground truth perceptual similarity from the world, human categorization behavior is a natural candidate".
- **(2) Nosotros**: H1 CIFAR-10H da apoyo, con 45 pares. CIFAR-10H es la fuente con menor discrepancia frente a DINOv2 (DE Δ = .50) y la de mayor acuerdo con THINGS (.78). H2 no se aplica, porque el plan no la incluye para C10H.
- **(3) Coincidencia**: los autores tratan la confusión entre categorías como similitud *perceptual*. Es la fuente cuya construcción (confusiones al clasificar imágenes pequeñas) se acerca más a lo que mide el paper. Es también la que más coincide con DINOv2, aunque n = 45.

## 5. Xu & Tenenbaum 2007 (Psychological Review 114:245–272)
PDF leído en una copia de curso (tallinzen.net).
- **(1) Conclusión**:
  - Abstract: "Results provide strong support for the Bayesian account over competing accounts".
  - **Clave para nosotros**: el espacio de hipótesis taxonómico *se construye a partir de la similitud humana*: "a taxonomy of nested categories, which can be constructed automatically by hierarchical clustering … on human participants' similarity ratings" (p. 252), y la Fig. 7, "Hierarchical clustering of similarity judgments yields a taxonomic hypothesis space" (p. 261).
  - Sobre los niños: "It is possible that children simply use the span of perceptual similarity as a first approximation … children may have mapped the words onto regions of perceptual space" (p. 259).
- **(2) Nosotros**: el paper cita a Xu & Tenenbaum como la predicción taxonómica que *no* se observó en los aprendices artificiales (Discusión de main.tex).
- **(3) Tensión**: en el trabajo original, lo taxonómico y la similitud no son ejes opuestos. La taxonomía *es* una agrupación jerárquica de la similitud. La oposición perceptual (DINOv2) frente a semántica (WordNet) del paper no corresponde a la distinción que pone a prueba Xu & Tenenbaum, que es muestreo fuerte frente a débil y el efecto del número de ejemplos. Los propios autores dejan abierta una versión perceptual del espacio de hipótesis infantil.

## 6. Lewis & Frank 2018 (Psychological Science 29:2039–2047)
PDF de la autora leído.
- **(1) Conclusión**:
  - Abstract: "the suspicious-coincidence effect in fact is robust to presentation timing of examples but is sensitive to … trial order. Our work highlights the influence of pragmatics".
  - Las réplicas exactas dan d = 1.27 y 1.2, frente a 2.0 y 1.01 del original (Resultados).
  - El efecto se invierte con el orden tres→uno y presentación secuencial (Exps. 10–12).
  - Por lo tanto, **no concluyen que el efecto sea débil**: lo concluyen robusto pero dependiente del orden de los ensayos.
- **(2) Nosotros**: H4 da apoyo. La probabilidad de que un adulto extienda la palabra baja con la distancia DINOv2 (log OR/DE −3.23; −3.19 con solo los ítems objetivo). En el mismo GLMM, la condición "3 subordinados" reduce la selección frente a "1 ejemplo" (β = −1.49), en el sentido del efecto de coincidencia sospechosa. El plan no pedía probar ese contraste: es un coeficiente de ajuste, no un resultado. No modelamos el orden de los ensayos.
- **(3) Complemento con confusión**:
  - El paradigma está diseñado por niveles taxonómicos (sub/bas/sup), y dentro de la categoría el nivel y la distancia DINOv2 van confundidos (.33/.58/.87).
  - Lo que H4 muestra es compatible con el gradiente taxonómico que Xu & Tenenbaum explican. No discrimina entre las dos lecturas.
  - Los niveles fijos no explican el efecto de la condición (número de ejemplos), y nuestra distancia por ítem tampoco lo hace por sí sola.

## 7. Vong et al. 2024 (Science 383:504–511), CVCL
PDF leído en una copia en gwern.net; los datos del corpus se confirmaron con el material suplementario.
- **Edad y horas (confirmado)**: "one child aged 6 to 25 months … 61 hours of correlated visual-linguistic data streams" (Abstract, p. 504). Son 600.000 fotogramas y 37.500 enunciados. Según el material suplementario, el codificador visual se preentrenó además con 133 horas de video sin transcribir del mismo niño (S), y el corpus SAYCam cubre "about 1% of the child's waking hours".
- **(1) Conclusión**:
  - p. 504: "critical aspects of grounded word meaning are learnable through joint representation and associative learning from one child's input".
  - Discusión, p. 507: los sesgos, entre ellos "the shape bias", "are not essential for making genuine progress".
  - La precisión fue de 61,6 % en Labeled-S (22 conceptos) y de 34,7 % en Konkle (64 conceptos; azar 25 %).
- **(2) Nosotros**: H6 da inconsistente. La distancia CVCL no predice la precisión del aprendiz controlando d_sem (β = +.005 [−.018, .028]). Aun así, la distancia CVCL correlaciona con DINOv2 (.62 en C100, .75 en C10) y con las distancias humanas (.43–.86).
- **(3) Complemento/tensión**: en CIFAR, que queda fuera de la distribución del corpus del niño, el espacio CVCL guarda estructura parecida a la humana, pero no reproduce el gradiente de aprendibilidad del paper. Esto cuadra con la cautela de los autores sobre la generalización fuera de distribución (34,7 % en Konkle). H6 no es dato humano y el plan no permite que corrija los análisis confirmatorios.

## 8. Patrones transversales (con evidencia, sin extrapolar)
1. **Las fuentes describen la similitud humana como una mezcla perceptual + conceptual/taxonómica, no como similitud perceptual pura.**
   - Así lo dicen Hebart 2020 y 2023 ("conceptual and perceptual", "taxonomic membership") y Roads & Love ("human semantic judgments", "a rich understanding of the world").
   - En Xu & Tenenbaum, la taxonomía se *deriva* de esa misma similitud. Solo Peterson et al. la llaman perceptual, y CIFAR-10H es la fuente más cercana a DINOv2.
   - Consecuencia descriptiva: el apoyo a H1 indica que DINOv2 se alinea con una similitud que según las fuentes también contiene categoría. No aísla un componente perceptual.
2. **La oposición del paper (perceptual = DINOv2 frente a semántica = WordNet) no tiene correlato directo en los trabajos originales.**
   - Ninguno concluye que la similitud perceptual o de forma domine sobre la taxonómica en la generalización.
   - Xu & Tenenbaum sostienen un modelo taxonómico bayesiano, aunque dejan abierta una aproximación perceptual en niños (p. 259).
   - Lewis & Frank no comparan lo perceptual con lo taxonómico: estudian el muestreo y la pragmática.
3. **Lewis & Frank no encuentran un efecto débil.** Concluyen que el efecto de coincidencia sospechosa es robusto y depende del orden de los ensayos. Nuestro H4 no prueba ese efecto. El efecto de la distancia que sí prueba está confundido con el nivel taxonómico.
4. **Patrón de nuestros resultados frente a las fuentes**:
   - Donde la fuente se describe como más "semántica" (HSJ), la distancia humana se asocia con la precisión más allá de WordNet, aunque fuera de las reglas del plan.
   - Donde se describe como mixta (THINGS), el resultado es inconcluso.
   - Donde se describe como perceptual (CIFAR-10H), hay más coincidencia con DINOv2, pero no hay prueba H2.
   - Es una lectura con pocas fuentes (k = 3, no independientes); se describe, no se infiere.
5. **Vong et al.**: hay coincidencia en el marco (aprendizaje asociativo sobre representaciones visuales) y tensión en el dato. El espacio de un modelo entrenado con la experiencia de un niño no reproduce el gradiente de aprendibilidad de DINOv2 en CIFAR.

## 9. Lo que no se pudo leer o no se verificó
- El PDF de eLife (Hebart 2023) no descargó; se leyó el texto completo en PMC.
- No se leyeron los materiales suplementarios de Hebart 2020/2023, Roads & Love ni Lewis & Frank.
- Xu & Tenenbaum se leyó en una copia de curso, no en la del editor. Las páginas se asignaron según los encabezados del PDF.
- Spencer et al. 2011, citado por Lewis & Frank, no se leyó.
