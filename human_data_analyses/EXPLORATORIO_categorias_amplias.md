# EXPLORATORIO post hoc: categorías amplias, humanos frente al aprendiz
**Todo este documento es exploratorio post hoc.** El plan congelado no lo prevé y nada de aquí aporta veredictos ni corrige los análisis confirmatorios.

## A. Comparaciones definidas ANTES de calcular (escrito 2026-10-30 18:32 hora de Santiago)
Fuentes del paper con categorías amplias:
- (i) Tiers native/near/mid/far de CIFAR-10. Palabras disyuntivas de 2 categorías, con su d_member (main.tex, Tabla tiers; Fig. c1 = `results/exp_007b_extension_clean/figure_c1_gradient_clean.png`). Precisión por palabra en `data/tiers_c10.csv`, de la ledger exp_007b.
- (ii) Superclases de CIFAR-100 (20×5) en exp_008 (`experiments/exp_008_dissociation/_shared.py`). Se usan en el análisis terciario same_superclass: β = +.015 [−.033, .064], p = .53 (`results/exp_008_dissociation/phase4/dissociation_analysis.json`). Solo 6 de los 100 pares muestreados son de la misma superclase.

Hay tres comparaciones fijadas. Todas las distancias DINOv2 son coseno y los IC son bootstrap percentil 95 % con B = 2000 y semilla 0.
- **C1. Gradiente por amplitud.**
  - Humanos (Lewis & Frank): tasa de selección de ítems objetivo por nivel sub/bas/sup, frente a la distancia DINOv2 media de cada nivel.
  - Aprendiz: precisión de naming de las 10 palabras disyuntivas (near/mid/far), frente a d_member.
  - Métrica: pendiente OLS de logit(y) sobre la distancia, en cada sistema. El IC se obtiene por bootstrap sobre sujetos (humanos) y sobre palabras (aprendiz).
  - También se reporta la ρ de Spearman entre la distancia y y, a nivel de unidad: ítem×condición en humanos y palabra en el aprendiz.
- **C2. Efecto dentro frente a entre superclase (CIFAR-100).**
  - Métrica: d de Cohen = (media entre − media dentro)/DE combinada, calculada para d_things, d_hsj, d_perc y para la precisión del aprendiz (acc_pair, que solo tiene 6 pares "dentro").
  - El IC se obtiene por bootstrap de pares, estratificado por dentro/entre. Los pares no son independientes, así que el IC es optimista.
- **C3. Perfil por superclase.**
  - Para cada una de las 20 superclases: media de d_perc dentro de la superclase frente a media de d_things y de d_hsj dentro de la superclase (con los pares disponibles).
  - Métrica: ρ de Spearman con IC por bootstrap sobre superclases.

## B. Resultados (EXPLORATORIO post hoc; calculado 18:34 hora de Santiago)
Script: `scripts/10_exploratorio_categorias_amplias.py`. Resultados: `out/exploratorio_categorias_amplias.json`. Se reportan las 3 comparaciones fijadas y ninguna variante.

**C1. Gradiente por amplitud**
| Sistema | Unidades | Perfil (y en la distancia media) | Pendiente de logit(y) por unidad de distancia [IC95] | ρ de unidad |
|---|---|---|---|---|
| Humanos, Lewis & Frank, ítems objetivo | 96 imagen×condición (corregido; antes 168) | sub .943 (.33) → bas .583 (.58) → sup .207 (.87) | −6.93 [−7.29, −6.56] (antes −6.81 [−7.19, −6.48]) | −.72 |
| Aprendiz centroide, tiers CIFAR-10 | 10 palabras | native .943* → near .847 (.36) → mid .654 (.56) → far .530 (.69) | −5.73 [−11.13, −2.73] | −.77 |

*La cifra native de .943 sale del paper (centroide; no hay distancia entre miembros). Coincide numéricamente con la de sub (.943), pero es una coincidencia: son tareas distintas.
- Los IC de las pendientes se solapan.
- La forma descendente es similar, pero el aprendiz no baja de su piso (azar ≈ .33), mientras que los humanos llegan a .21 en sup.

**Corrección (31-oct-2026, DESVIACIONES §27).** La versión anterior contaba por separado nombres equivalentes de condición (`3sub` y `three_subordinate`, etc.). Al unificarlos hay 96 unidades imagen×condición, no 168. La pendiente sigue negativa y casi igual. En la figura x1 los ejes verticales ahora están separados: (a) es la proporción seleccionada (extensión de la palabra) y (b) la precisión de nombrado, que no son la misma cantidad. Además, las 4 condiciones humanas se muestran por separado.

**Ítems superordinados según la condición de enseñanza** (verificado; coincide con la revisión externa):

| Condición | Proporción seleccionada | Distancia DINOv2 media |
|---|---|---|
| 1 ejemplo sub | 2,6 % | .880 |
| 3 ejemplos sub | 1,1 % | .892 |
| 3 ejemplos básicos | 1,8 % | .861 |
| 3 ejemplos superordinados | 77,5 % | .845 |

La media superordinada global de ≈21 % oculta este efecto de la condición. Con ejemplos superordinados, los humanos extienden la palabra aunque la distancia sea grande. Que la extensión baje con la distancia en las otras condiciones puede ser una inferencia adecuada y no una dificultad para aprender. Las pendientes humana y del aprendiz no son comparables como mecanismo.

**C2. Dentro frente a entre superclase de CIFAR-100** (d de Cohen = entre − dentro)
- d_things: 1.70 [1.61, 1.78], con 150 pares dentro y 3.250 entre.
- d_hsj: 1.40 [1.18, 1.64], con 103 y 1.287.
- d_perc (DINOv2): 2.17 [1.95, 2.39], con 200 y 4.750.
- Precisión del aprendiz (exp_008, palabra que une los 2 miembros del par): −1.24 [−2.15, −0.43], con **solo 6 pares dentro** y 94 entre. Media .915 dentro frente a .845 entre: los pares de la misma superclase se aprenden mejor.
- El paper ya reporta que la superclase no añade nada una vez controlada d_perc (β_same_sc = +.015 [−.033, .064], p = .53). Es decir, este efecto bruto va por la vía de la distancia.

**C3. Perfil por superclase** (ρ entre la distancia DINOv2 media dentro de la superclase y la distancia humana media dentro de la superclase)
- THINGS: ρ = .46 [−.01, .81], n = 18 superclases.
- HSJ: ρ = .08 [−.52, .60], n = 16.

Figuras (`figs/comparativos/exploratorio/`, PNG 300 dpi + PDF, marcadas EXPLORATORY):
- `x1_breadth_gradient_humans_vs_learner`: el gradiente humano por nivel y el del aprendiz por tier, en el mismo eje de distancia DINOv2 y la misma escala y.
- `x2_superclass_within_between`: dentro frente a entre superclase para THINGS, HSJ, DINOv2 y la precisión del aprendiz.
- `x3_superclass_profile`: el perfil por superclase, DINOv2 frente a THINGS y frente a HSJ.

## C. Limitaciones (EXPLORATORIO)
- **Es post hoc**: lo motiva una observación visual sobre las figuras. Las comparaciones se fijaron antes de calcular (§A), pero después de ver las figuras.
- **Pocas unidades**: el perfil humano tiene 3 niveles y el del aprendiz 3–4 tiers, así que una correlación de rangos entre perfiles es trivial. Por eso se usaron pendientes por unidad. El aprendiz tiene 10 palabras, y C2 tiene solo 6 pares de aprendiz dentro de la superclase.
- **Las distancias no son la misma cantidad**:
  - En humanos, es la distancia de la imagen de prueba a los ejemplares de entrenamiento.
  - En el aprendiz, es la distancia entre los centroides de las dos categorías que une la palabra.
  - Las tareas también difieren: selección de generalización frente a naming de 3 vías.
- **Confusión nivel/distancia**: en Lewis & Frank, el nivel taxonómico y la distancia van juntos. Además, la figura x1(a) muestra que, a una misma distancia de unos .85, la selección va de ≈0 a ≈.85 según la condición de entrenamiento (1 ejemplo, o 3 sub/bas/sup). Esa variación la explica la condición, es decir, el número y tipo de ejemplos, no la distancia.
- **C2 es en gran parte circular**: las superclases de CIFAR-100 son agrupaciones taxonómicas que también son cercanas en DINOv2. El efecto dentro/entre aparece en los tres espacios, pero no separa lo perceptual de lo taxonómico.
- **Pares no independientes**: los IC de pares en C2 son optimistas. En C3 se excluyeron las superclases sin pares humanos mapeados.
- **Sin veredicto**: nada de aquí aplica reglas del plan ni modifica H1–H6.
