# Revisión detallada de los resultados con datos humanos (2026-10-30, ~18:25 hora de Santiago)

Plan congelado `plan_analisis_datos_humanos_FROZEN_2026-10-30.md`, sha256 9b50995e…, re-verificado y sin cambios. Este documento solo describe: no saca conclusiones ni propone un encuadre. **Los datos humanos son secundarios**: vienen de experimentos de otros equipos (THINGS, ImageNet-HSJ, CIFAR-10H, Lewis & Frank 2018, normas Glasgow), que nosotros no corrimos.

## 1. Auditoría
- **Reproducibilidad**: 02_h1_h2, 05_h5, 07_dispersion, 08_holm y 06_h6 se re-corrieron y dieron JSON idénticos. Las cifras de RESULTADOS.md coinciden con `out/`, con una sola imprecisión de etiqueta (DESVIACIONES §23). Los GLMM de H4 no se re-corrieron; sus cifras coinciden con `out/h4_glmm.txt`.
- **H4, sensibilidad**: el análisis primario usa los 24 ítems, con un log OR por DE = −3.23 [−3.29, −3.17]. Con solo los 8 ítems de la categoría objetivo (57.600 obs., 600 sujetos), el log OR = −3.19 [−3.33, −3.05], con p < .001 y sin advertencias de convergencia. Sin los experimentos no prerregistrados, −3.17 [−3.24, −3.11]. Datos descriptivos: los ítems de otras categorías se eligen en el 0,46 % de los casos y los de la categoría objetivo en el 48,5 %. Dentro de la categoría objetivo, el nivel y la distancia van confundidos (distancia media sub .33, bas .58, sup .87). La regla del plan (β<0, p Holm<.05) se cumple en las tres especificaciones.
- **H2 HSJ, caso límite**: β_hum = −.034 [−.067, −.002], p = .039, p Holm = .079, r parcial −.413 [−.670, −.069].
  - *Apoyo*: no, porque p Holm ≥ .05.
  - *No apoyo*: no, porque el IC no excluye −.20 ni β>0.
  - *Inconcluso*: no, porque el IC no incluye 0.
  - Queda fuera de las tres reglas, como ya registra DESVIACIONES §7.
  - Robusteces (no deciden): p = .057 con d_sem original, .031 con ambos mapeos originales (n=22), .017 con ≥1 instancia (n=49) y .006 con ≥30 instancias (n=12).
- **Cobertura THINGS, 61 vs 64 (causa identificada, §22)**: el mapeo primario saca cattle y ray, agrega wardrobe y excluye los 3 pares entre árboles. Por eso son 3.400 pares en vez de 3.486, y se pierden los pares muestreados beaver–ray, mushroom–ray y ray–sweet_pepper.
- **Familia BH de H6**: son 9 pruebas, (a)×2, (b)×2 y (c)×5, y se reprodujeron de forma independiente. Lo ambiguo es la definición de la familia, que el plan no precisa (§21): no dice si β_sem entra ni si c10 y c100 se cuentan aparte. Esto no cambia el veredicto, porque el IC de β_perc_CVCL incluye 0 en cualquier caso.

## 2. Figuras (`figs/comparativos/`, PNG 300 dpi + PDF; script `scripts/09_figs_comparativos.py`)
- `a_h1_scatter_human_vs_dperc`: d_perc de DINOv2 frente a la distancia humana, en THINGS, HSJ y CIFAR-10H, con ρ e IC de H1.
- `b_accuracy_vs_human_and_dperc` *(descriptivo, no previsto en el plan)*: precisión del aprendiz frente a la distancia humana (izquierda) y frente a d_perc (derecha), sobre los mismos pares. CIFAR-10H usa los 10 tiers.
- `c_forest_all_H1_H6`: todos los coeficientes H1–H6 (ρ o r parcial) con IC 95 %, el SESOI ±.20 y el veredicto del plan. Tiene un panel aparte para el log OR de H4 en sus 3 especificaciones.
- `d_h4_generalization_curve` *(descriptivo)*: tasa de selección humana (Lewis & Frank) frente a la distancia DINOv2, en 12 bins por cuantiles. A la izquierda van los 24 ítems, con puntos por imagen que distinguen los ítems objetivo de los de otras categorías. A la derecha, solo los ítems objetivo.
- `e_h6_dino_vs_cvcl`: d_perc de CVCL frente a DINOv2 en C10 y C100, y la precisión por tier. El título indica la precisión media en C100.
- `f_agreement_things_vs_hsj` (§4): THINGS frente a HSJ en los 1.042 pares compartidos (ρ = .60), con los 20 pares muestreados destacados.
- `g_h5_aoa_vs_coherence`: AoA de Glasgow frente a la dispersión intra-clase, con C10 (n=10) y C100 (n=85) por separado, ρ parcial y ρ cruda.

## 3. Qué se puede afirmar y qué no (estrictamente según las reglas del plan)
| Se puede afirmar | No se puede afirmar |
|---|---|
| H1: en las 3 fuentes, el espacio DINOv2 correlaciona positivamente con la distancia humana (regla de apoyo cumplida, Holm). | Que lo haga con una magnitud dada fuera de esos pares y mapeos, ni que las fuentes humanas sean intercambiables (THINGS–HSJ ρ = .60). |
| H2 THINGS: inconcluso. | Que la similitud humana explique, o no explique, la precisión del aprendiz controlando d_sem. |
| H2 HSJ: ninguna regla del plan se cumple. | Llamarlo apoyo (p Holm = .079) o no apoyo; usar las robusteces para decidir. |
| H4: en adultos, la probabilidad de generalizar una palabra nueva baja con la distancia DINOv2, con los 24 ítems y también con solo los ítems objetivo. | Separar el efecto de la distancia del nivel taxonómico dentro de la categoría (están confundidos); afirmar nada sobre niños. |
| H3, H5 y §4: solo descripción con IC. | Inferencias confirmatorias. H5 en C10 (n=10) es exploratorio y el plan no le da regla de veredicto. |
| H6: inconsistente según la regla (IC de β_perc_CVCL incluye 0). Correlaciones CVCL–DINOv2 y CVCL–humanas positivas. | Que H6 corrija o sustituya algún análisis confirmatorio (el plan lo prohíbe); tratar CVCL como dato humano. |
| Wordbank y sobreextensiones: no corridos. | Cualquier afirmación sobre AoA infantil (Wordbank) u overextensions. |
