# Desviaciones respecto al plan congelado

Plan: plan_analisis_datos_humanos_FROZEN_2026-10-30.md (hash en PLAN_HASH.txt). El plan no se modifica.

Cada entrada: qué cambió, por qué, y efecto conocido. Fechas en hora de Santiago (UTC-3), 2026-10-30.

## Generales
1. **Valores p bilaterales.** Las reglas piden signo; se usan p bilaterales y el signo se exige por separado (conservador).
2. **Entorno.** apt falló (repos Debian 500/sin firma); R 4.x + lme4 2.0.6 instalados vía micromamba/conda-forge en /tmp/renv. Sin efecto en cálculos.
3. **Mapeos de categorías** (map_things_v1.json, map_hsj_v1.json): se aplicaron aclaraciones de mapeo documentadas en esos JSON; se reportan robusteces con el mapeo original (out/h1_h2_results.json → H2_rob).

## H1/H2
4. **Cobertura THINGS**: 3.400 pares / n=61 pares muestreados, vs 3.486 / 64 previstos en el plan. Causa no identificada por completo (probablemente diferencias de mapeo). Se reporta tal cual.
5. **Definición de instancia HSJ**: se usó la definición implementada en 01_build_pairs.py (ver docstring); umbral principal y robusteces ≥1 y ≥30 reportadas.
6. **Acuerdo entre fuentes**: c100 THINGS–HSJ con 1.042 pares (plan: 1.109) y 20 pares muestreados en intersección (plan: 22). Consecuencia de 4–5.
7. **Vacío en la regla H2 (HSJ)**: β_hum<0 con p sin corregir .039 pero p Holm .079 (no apoyo); el IC de r parcial [−.670, −.069] excluye 0, incluye −.20 y no excluye valores ≤ −.20; no cumple ninguna de las tres reglas. Se informa como "inconcluso (fuera de reglas)"; no se reinterpretó.

## H4
8. **subid repetido entre experimentos**: se usó exp_subid (600 sujetos, 50 por experimento). Una corrida previa con subid fusionados se descartó (el estimado de zd fue prácticamente idéntico, −3.23).
9. **Ítems de test**: el plan dice nivel desde el nombre de archivo y selección por ensayo × ítem; el análisis primario usa los 24 ítems (incluye los 16 de otras categorías, casi nunca elegidos, que dominan el efecto). Sensibilidad con solo los 8 ítems de la categoría objetivo reportada.
10. Pendiente señalar si la pendiente aleatoria convergió: convergió (sin advertencias) en las tres especificaciones.

## H5 (exploratoria)
11. Embeddings DINOv2 de un subconjunto de 100 imágenes de entrenamiento por clase (semilla 42), porque la caché del repo (.cache/embeddings_exp008) no existe.
12. Frecuencia léxica: wordfreq (Zipf; datos CC BY-SA 4.0), fuente no especificada en el plan.
13. AoA Glasgow: elección de sentidos (pepper→vegetable, seal→animal, tank→army, plane→aircraft, etc.; ver 05_h5.py).
14. Wordbank no corrido: los datos solo se sirven por Redivis con login/token; wordbankr no compiló. Sobreextensiones no corridas: licencia no clara.

## H6
15. Corridas restringidas al learner centroide y al pool std_nway (verificado: reproduce exactamente los acc_tier de exp_007b con DINOv2).

## §4
16. **Modelo (b) formato largo**: acc_pair es idéntico para un par en ambas fuentes, por lo que el intercepto aleatorio por par absorbe toda la varianza (MixedLM degenerado, LR<0). Se usó OLS con LR entre modelos anidados y Wald con EE agrupados por par.
17. I²/τ con k=2 fuentes que comparten pares: descriptivo, no independiente.

## Nuevas (2026-10-30, ~11:55 hora de Santiago)
18. **H5 cobertura Glasgow**: n = 85 palabras en CIFAR-100 (plan: 89), tras colapsar las 5 clases "tree" y por las entradas de sentido disponibles en Glasgow. Se reporta tal cual.
19. **H4 interacción (exploratoria)**: el test LR (anova base vs. interacción) en 04b_h4_interaction.R falló ("object 'dd' not found") tras ajustar el modelo; se reportan solo los Wald z de los términos zd×condition. No se re-corrió.
20. **H6 c100 pendiente**: la extracción de embeddings CVCL de 50.000 imágenes CIFAR-100 (PID 1230715) va en 12.864/50.000 a las 11:53 (~150 img/min, ~4 h restantes en CPU). H6(a) y H6(b,c) en CIFAR-100 no están; la corrección BH de H6 se calculó solo sobre las pruebas c10 disponibles y deberá recalcularse con (a) incluida.

## Nuevas (2026-10-30, ~18:05 hora de Santiago)
21. **H6 completado**: la extracción CVCL de CIFAR-100 terminó (~18:03; ~6 h en CPU en vez de las ~4 h estimadas). Se corrieron h6_c100.py y 06_h6.py sin cambios. La corrección BH de H6 se aplicó a 9 pruebas: (a) con β_perc y β_sem por separado, (b) en c10 y c100, y (c) c10 ×3 + c100 ×2. El plan dice "BH dentro de H6 (pruebas a, b y c)" sin precisar si β_sem entra en la familia ni si c10 y c100 se cuentan por separado; se dejó así y no se probó otra versión. El punto 20 queda resuelto.

## Auditoría (2026-10-30, ~18:25 hora de Santiago)
22. **Causa de la cobertura THINGS 3.400/61 vs 3.486/64 (corrige §4, "causa no identificada")**: la lista del scouting usada para el plan (`cifar100_covered_by_THINGS.json`) tenía 84 clases, C(84,2) = 3.486. El mapeo primario (`map_things_v1.json`) tiene 83: saca **cattle** y **ray** (solo entran en SENS_STRICT como mapeos a hipónimos: cow/bull, stingray), agrega **wardrobe**, y excluye los 3 pares entre maple/oak/willow (todos → "tree"): C(83,2) − 3 = 3.400. Los 3 pares muestreados que se pierden son beaver–ray, mushroom–ray y ray–sweet_pepper (64 − 3 = 61). Además, la nota del JSON dice "79 classes" para el scouting, pero el archivo de cobertura tiene 84: la documentación del mapeo es inconsistente. No se re-corrió nada con otro mapeo.
23. **Etiqueta imprecisa en RESULTADOS.md (H2)**: "con el mapeo original THINGS β_hum p = .045" corresponde a `H2_rob.THINGS_origmap_dsem`, que cambia d_sem al mapeo WordNet original (d_sem_orig) y mantiene el mapeo THINGS primario. Con el mapeo THINGS del scouting, p = .542 (n=59); estricto, p = .372 (n=55). **Corregido en RESULTADOS.md el 2026-10-31 (~10:40 hora de Santiago).**
24. **H4, composición de ítems (descriptivo, ya señalado en §9)**: los ítems de otras categorías se eligen en el 0,46 % de los casos (115.200 obs.) y los de la categoría objetivo en el 48,5 % (57.600); el 98,1 % de las selecciones son de la categoría objetivo. Dentro de la categoría objetivo, el nivel y la distancia van juntos (distancia media sub .33, bas .58, sup .87). La sensibilidad con solo los ítems objetivo da log OR = −3.19 [−3.33, −3.05] (vs −3.23 primaria).
25. **Re-ejecución**: 02_h1_h2, 05_h5, 07_dispersion, 08_holm y 06_h6 se re-corrieron y dieron JSON idénticos byte a byte. La corrección BH de H6 (9 pruebas) se reprodujo de forma independiente con statsmodels (fdr_bh), con los mismos valores. No se re-corrieron los GLMM de H4 (R, largos).

## Correcciones tras revisión externa (2026-10-31, ~12:40 hora de Santiago)
26. **CORRECCIÓN DE ERROR, H4: identidad de los participantes (corrige §8; §8 se mantiene como registro)**
   - **El error.** En §8 se supuso que un subid repetido entre experimentos correspondía a personas distintas, y el GLMM se agrupó por exp_subid (600 "sujetos"). Eso es incorrecto.
   - **Verificación.** Leí `analysis/munge_anonymize_data.R` de XTMEM (commit 220ed17). Primero une con `bind_rows` los datos crudos de los 12 experimentos y después aplica `anonymize.sids(d_raw, "WorkerId")`, que asigna un subid por WorkerId único sobre el conjunto ya unido. Por lo tanto, un mismo subid es el mismo WorkerId: la misma persona.
   - **Datos.** Hay 600 participaciones de 520 personas. De ellas, 464 participan una vez; 42 participan en 2 experimentos, 10 en 3, y 1 persona en cada caso de 4, 5, 6 y 7.
   - **Re-estimación** (`scripts/04c_h4_person_re.R`, `out/h4_person_re.txt`). Mismo modelo del plan, con efectos aleatorios por persona. Todos los ajustes usan la pendiente aleatoria y convergieron sin advertencias.
     - C1, primaria, (1+zd|persona): −3.226 [−3.288, −3.164].
     - C2, añadiendo (1|participación): −3.233 [−3.295, −3.171].
     - C3, solo la primera participación: −3.183 [−3.247, −3.119]. "Primera" es el experimento de menor número, porque los datos no traen el orden temporal.
     - C4, excluyendo a quienes repiten (464 personas): −3.120 [−3.188, −3.052].
     - C5, sin los experimentos no prerregistrados: −3.168 [−3.236, −3.100].
     - C6, 8 ítems objetivo con nivel y condición: −3.157 [−3.296, −3.017].
     - C7, C6 restringido a la primera participación: −3.115 [−3.261, −2.968].
     - C8, C6 excluyendo a quienes repiten: −3.111 [−3.264, −2.958].
     - En todos los casos p < 10⁻¹⁰⁰: R devuelve 0 por underflow.
   - **Consecuencia.** La regla de apoyo de H4 (β < 0, p Holm < .05) se mantiene. Desde ahora la estimación primaria es C1. La p de H4 en la familia Holm sigue siendo < .001, así que el resultado de Holm no cambia; `08_holm.py` no se re-corrió.
   - **Pendiente.** La interacción exploratoria dist × condición (§19, `04b`) sigue agrupada por exp_subid y no se re-estimó.
27. **CORRECCIÓN DE ERROR, exploratorio C1/x1: etiquetas de condición duplicadas**
   - **El error.** `10_exploratorio_categorias_amplias.py` agrupaba por la etiqueta cruda de condición. Algunos experimentos usan `3sub`/`3bas`/`3sup` y otros `three_subordinate`/`three_basic`/`three_superordinate`, así que la misma condición contaba como dos.
   - **Corrección.** Al unificar las etiquetas, las unidades imagen×condición pasan de 168 a 96 (verificado).
     - Pendiente: de −6.81 [−7.19, −6.48] a −6.93 [−7.29, −6.56].
     - ρ por unidad: de −.716 a −.721.
     - Los perfiles por nivel no cambian, porque no dependen de la condición.
   - **Figura x1.** Se rehízo con ejes verticales separados (proporción seleccionada frente a precisión de nombrado) y con las 4 condiciones de enseñanza humanas por separado.
28. **Alcance del error de condición.** Revisé todos los scripts que usan `condition`:
   - `04_h4_glmm.R` y `04b_h4_interaction.R` ya mapeaban las 7 etiquetas a 4 niveles (one/sub/bas/sup), así que H4 no se ve afectada por este error.
   - `h4_prep.py` solo copia la etiqueta.
   - Ningún otro análisis (H1–H3, H5, H6, §4, figuras 09) usa la condición de Lewis & Frank.

## Correcciones tras segunda revisión externa (2026-10-31, ~13:45 hora de Santiago)
29. **H4, C3 y C7: "primera participación" por fecha real (corrige la definición de §26).** `all_raw_A.csv` (XTMEM 220ed17) trae AcceptTime y SubmitTime para las 600 participaciones. Verifiqué que ambas fechas eligen la misma primera sesión para las 520 personas, y que difieren del criterio "experimento de menor número" solo en una persona (subid 216: exp 7 frente a exp 5). Re-estimación (`scripts/04d_h4_firstsession_interaction.R`, `out/h4_firstsession_interaction.txt`, `data/h4_first_session.csv`): C3t −3.1671 [−3.2309, −3.1033] (antes −3.183); C7t −3.1164 [−3.2629, −2.9699] (antes −3.115). Pendiente aleatoria, sin advertencias. Solo afecta a esas dos sensibilidades.
30. **Propagación del H4 por persona.** `08_holm.py` lee `out/h4_person_re.txt` (C1 −3.2261 [−3.2880, −3.1642]); veredicto Holm sin cambios. `09_figs_comparativos.py` lee C1/C5/C6 de `out/` (con assert contra el JSON de Holm) en vez de cifras escritas a mano; `c_forest_all_H1_H6` regenerada 31-oct 13:10. README apunta a `04c_h4_person_re.R` y `04d`.
31. **Figura x1, eje horizontal.** Panel humano con xlim 0.05–1.15, cubre las 96 distancias (0.079–1.115). Regenerada 31-oct 13:12; pendiente −6.927 sin cambios.
32. **EXPLORATORIO: interacción distancia × condición con RE por persona (sustituye §19/`04b`, que agrupaba por exp_subid).** glmer `selected ~ zd*condition + level + (1+zd|persona) + (1|imagen)`, 172.800 obs., 520 personas, sin advertencias (`out/h4_interaction_person.txt`). Ref. "one": zd −2.924 [−2.996, −2.853]; zd:sub +0.100 (z = 3.44, p = .0006); zd:bas −0.862 (z = −21.88); zd:sup −1.294 (z = −36.11). LR vs modelo sin interacción: χ²(3) = 2134.2, p < 10⁻¹⁰⁰. Exploratorio, no prerregistrado como prueba confirmatoria.

## Análisis post hoc tras revisión crítica v2 (2026-10-31, ~14:25 hora de Santiago)
33. **EXPLORATORIO POST HOC: aporte incremental de la distancia en H4 (no está en el plan congelado; fuera de la familia Holm).** Motivo: la revisión crítica de la v2 (M2) señaló que el log OR del modelo de 24 ítems lo domina el contraste entre dominios. Script `scripts/04e_h4_incremental_distance.R`, salida `out/h4_incremental_distance.txt`. Solo los 8 ítems objetivo (57.600 obs., 520 personas), RE por persona (§26), mismos datos (`data/h4_long.csv`).
   - Escala de z(d): DE cruda de la distancia 0.2192 en el modelo de 24 ítems y 0.2963 en el de 8 ítems objetivo; por eso los coeficientes de ambos modelos no son directamente comparables.
   - M0 (nivel + condición, sin distancia) vs M1 (+ z(d) fijo): LR χ²(1) = 3016.96, ΔAIC = 3014.96. M0 vs M2 (+ pendiente aleatoria de z(d) por persona, = C6): χ²(3) = 3695.39, ΔAIC = 3689.39. Sin advertencias ni ajustes singulares.
   - Pendiente de z(d) dentro de cada nivel (modelo con nivel × z(d), condición, RE por persona): básico −3.21 [−3.40, −3.03], subordinado −4.22 [−4.46, −3.98], supraordinado −0.94 [−1.25, −0.62].
   - El R² marginal (Nakagawa) se calculó pero no se interpreta: baja al añadir la distancia (0.735 → 0.684) porque cambia la varianza de los efectos aleatorios; no es una medida adecuada del aporte incremental aquí.
   - No cambia ninguna decisión confirmatoria. No se re-ejecutó ningún análisis del plan.

## Análisis post hoc tras revisión automática externa (paperreview.ai, 2026-10-31; aprobados por el autor a las 17:50 hora de Santiago)
Especificaciones escritas ANTES de correr los modelos. Corrección de hora (anotada 18:45 hora de Santiago, 2026-10-31): este archivo y los scripts 04f/04g quedaron guardados juntos a las 17:52:33 (mtime); la ejecución de 04g terminó a las 17:53:51 y el proceso actual de 04f se inició a las 18:06:53 (ps lstart). La hora '~18:05' que figuraba antes aquí era errónea (posterior a la escritura real). Las especificaciones también están resumidas en la cabecera de cada script, guardada a la misma hora. Ambos son EXPLORATORIOS POST HOC, motivados por una revisión automática externa (paperreview.ai, preguntas Q4 y Q1); no están en el plan congelado, no entran en la familia Holm y no cambian ninguna decisión confirmatoria. Los resultados se reportan tal como salgan.

34. **ESPECIFICACIÓN, EXPLORATORIO POST HOC: moderación por orden de ensayos (Q4).**
   - Moderador: `order` de `data/xtmem/experiment_key.csv` (XTMEM 220ed17), variable de nivel de experimento: "3-1" = los ensayos de 3 ejemplares (3sub) preceden al de 1 ejemplar; "1-3" = al revés. Está definida para los 12 experimentos (1-3: exps 1, 2, 5, 7, 10, 12; 3-1: exps 3, 4, 6, 8, 9, 11), así que se usan todos. Antes de correr verifiqué en `all_raw_A.csv` que en los experimentos "3-1" los ensayos 1–3 son 3sub y en los "1-3" son "one". Es una variable entre experimentos: se confunde parcialmente con blocking y timing, y no se intenta separar.
   - Datos: `data/h4_long.csv`, 24 ítems, 172.800 obs., 520 personas (como §32).
   - Modelos (glmer binomial, bobyqa, RE `(1+zd|persona) + (1|imagen)`, z(d) estandarizada sobre los 24 ítems como en §32):
     - Mbase: `selected ~ zd*condition + condition*order + level`
     - Mb: Mbase + `zd:order` → LR χ²(1): ¿varía la pendiente de la distancia según el orden?
     - Mc: Mb + `zd:condition:order` → LR χ²(3): ¿varía la interacción distancia × condición según el orden?
   - Se reportan: LR, gl, p, coeficientes de los términos con orden (Wald), N obs., N personas, advertencias/singularidad. Si Mc no converge, se reporta así y se reporta solo Mb.
   - Qué cuenta como evidencia: p < .05 en el LR correspondiente (dos pruebas; se informa también Holm sobre esas 2). Con N tan grande, se reporta también el tamaño del coeficiente y no solo la p.
   - Script `scripts/04f_h4_order_moderation.R`, salida `out/h4_order_moderation.txt`.

35. **ESPECIFICACIÓN, EXPLORATORIO POST HOC: modelo solo con ítems supraordinados (Q1).**
   - Datos: `data/h4_long.csv` restringido a `in_target == 1 & level == "sup"` (los 4 ítems supraordinados de la categoría enseñada en cada ensayo; 12 imágenes en total). z(d) re-estandarizada dentro de este subconjunto.
   - Modelos (glmer binomial, bobyqa, RE `(1|persona) + (1|imagen)`):
     - Md: `selected ~ zd`; Mc: `selected ~ condition`; Mdc: `selected ~ zd + condition`.
     - LR condición | distancia: Mdc vs Md, χ²(3). LR distancia | condición: Mdc vs Mc, χ²(1).
   - Sensibilidad (también especificada ahora): lo mismo con `(1+zd|persona)` en vez de `(1|persona)`; si no converge o es singular, se reporta así.
   - Se reportan: ambos LR, coeficientes de Mdc, N obs., N personas, advertencias.
   - Qué cuenta como evidencia de aporte incremental: p < .05 en el LR respectivo, reportando también los coeficientes.
   - Script `scripts/04g_h4_superordinate_only.R`, salida `out/h4_superordinate_only.txt`.

## Licencias y redistribución (2026-10-31, ~21:00 hora de Santiago; tras la revisión final previa al envío)
36. **`data/h4_first_session.csv` deja de redistribuirse.** Es un derivado por participante (subid, primer experimento) de `all_raw_A.csv` de XTMEM. Ahora lo genera `scripts/h4_first_session.py` a partir de la descarga original (AcceptTime más temprano por subid; reproduce byte a byte el archivo usado en §29, 520 filas) y queda en `.gitignore`, igual que `data/h4_long.csv`. No cambia ningún resultado. El archivo sigue en el historial de git (commits 66ad8f4 y 1c50c3b).
   - Licencia de XTMEM verificada en fuentes primarias: el repositorio GitHub mllewis/XTMEM no tiene archivo LICENSE; el proyecto OSF de los autores (https://osf.io/yekhj/, del que se registró https://osf.io/zcbp7/ con la plantilla AsPredicted) declara CC BY 4.0 y contiene `data/anonymized_data/all_raw_A.csv`, idéntico (md5 e42712b8…) al usado aquí. La mención "CC BY 4.0 (OSF)" del plan congelado es correcta; el plan no se modifica.
   - Los agregados de CIFAR-10H (columna `d_c10h`) son CC BY-NC-SA 4.0: esos archivos se marcan como share-alike en `data/README.md` y `README.md`.
   - wordfreq (datos CC BY-SA 4.0) pide no reempaquetar sus datos como CSV: la columna `zipf` se quita de `data/h5_*_glasgow.csv` y `scripts/05_h5.py` la calcula en tiempo de ejecución con `wordfreq.zipf_frequency`. Verificado con wordfreq 3.1.1: los valores coinciden exactamente con los antes distribuidos y `out/h5_results.json` se reproduce idéntico (ρ parcial, IC, p, p_BH). No cambia ningún resultado.

