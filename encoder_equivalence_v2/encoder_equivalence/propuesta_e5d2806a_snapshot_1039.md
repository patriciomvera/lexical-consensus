# Propuesta de diseño: comparación de encoders para el efecto de orden (v2 de arXiv 2611.14116)

- **Estado:** PLAN FINAL, A LA ESPERA DE LA SEGUNDA REVISIÓN DE ARIADNA. **No está congelado** (no tiene hash de congelamiento). No se ha ajustado ningún GLMM ni se ha leído ninguna respuesta de participantes para esta v2.
- **Origen:** pedido de Patricio (18-11, 09:57, vía Evaristo), a partir de la sugerencia de @1shareShare1 en X: cambiar el backbone y ver si el efecto del orden se mantiene.
- **Alcance:** solo diseño. No se propone título ni encuadre del paper.
- **Fuentes de las cifras:** todas salen de archivos, con archivo y sección indicados: `human_data_analyses/` (commit 1e7a904 del repo `patriciomvera/lexical-consensus`), `paper_arxiv_v_concl/main.tex` y `encoder_equivalence/out/`.

## Registro de cambios (hora de Santiago, UTC−3)
- **2026-11-18 10:04.** Primera versión (sha256 `afb7e139…46845c1a`; copia en `encoder_equivalence/propuesta_v1004_snapshot.md`). Es la versión que revisó Ariadna.
- **2026-11-18 10:13.** Patricio, vía Evaristo, pide una tabla de equivalencias entre todos los encoders en vez de un umbral de .90 contra DINOv2.
- **2026-11-18 10:15–10:19.** Se calcularon embeddings de las 45 imágenes de estímulo. No son distancias de ensayo ni usan respuestas humanas.
- **2026-11-18 10:17.** Decisiones de Patricio sobre §10:
  1. entran todos los encoders, incluidos DINOv2-B y OpenCLIP;
  2. la tabla de equivalencias reemplaza a los márgenes y al umbral fijo;
  3. la réplica con 8 ítems es co-primaria;
  4. se autoriza el cómputo, pero se ejecuta solo el sábado 21 o el domingo 22 de noviembre;
  5. antes de congelar hay revisión independiente de Ariadna.
- **2026-11-18 ~10:45.** Revisión de Ariadna (`REVISION_ariadna_comparacion_encoders_v2_2026-11-18.md`): "congelable con cambios".
- **2026-11-18 10:25–11:00.** Esta versión incorpora B1, B2 y B3, los puntos I1–I5 y M1–M7, y las decisiones de Patricio. El detalle de cada cambio está en §11.
- **2026-11-18 10:37.** Decisión de Patricio, vía Evaristo, sobre δ: no hay margen único. Las regiones de B2 se evalúan y se reportan con δ = 0,05, 0,10 y 0,20 como espectro. Se reescribió §7 (nuevos §7.1–§7.4: tabla única, perfiles "robusto" o "dependiente del margen", Holm por δ y G0–G4 por δ) y se actualizaron §5.2, §9.2, §10 y §11. Este cambio no tiene impacto en el costo (§9.2).
- **2026-11-18 10:38.** Patricio confirma, vía Evaristo: se aprueban B1 (dos etapas con permutación exacta), el control positivo obligatorio por encoder y la tabla de equivalencias después de congelar. El corte de 0,5 queda por defecto como único punto pendiente. Ariadna hará una segunda revisión del plan final antes de congelar (§10.4).

---

## 1. Qué se hizo exactamente en la v1 (punto de partida)

### 1.1 Datos
- Lewis & Frank (2018), XTMEM commit 220ed17, `data/xtmem/all_raw_A.csv`. Son 600 participaciones de 520 personas en 12 experimentos, con 12 ensayos cada una y 24 ítems de test por ensayo, lo que da 172.800 observaciones (`RESULTADOS.md`, H4; `DESVIACIONES.md` §26). Licencia CC BY 4.0 en OSF (`DESVIACIONES.md` §36).
- Archivo largo: `data/h4_long.csv` (172.800 filas; sha256 `d9d30f9d…394dd1`, calculado hoy). Está en `.gitignore` y se regenera con `scripts/h4_prep.py`.

### 1.2 Estímulos que se codifican
- Son 45 imágenes `c{1,2,3}_*.jpg` de `XTMEM/experiments/image_stim/images/`, todas de 720×720 px (verificado hoy). La carpeta trae además `mrfrog.png`, que no se usa.
  - **24 ítems de test**, lista fija `TEST` en `h4_prep.py`: por dominio, sub1–2, bas1–2 y sup1–4.
  - **21 imágenes de entrenamiento** (`trainPics_T*`): por dominio, sub3–5, bas3–4 y sup5–6.
- Hay 3 dominios: c1 = vegetables, c2 = vehicles y c3 = animals (`h4_prep.py`, `cnum`).

### 1.3 Distancia d
Viene de `scripts/h4_prep.py` y `revision_runs/repo/src/agents/perception.py`.
- **Encoder:** `facebook/dinov2-small`, congelado, con el `AutoImageProcessor` por defecto. Según `main.tex` l.168, ese procesador redimensiona en bicúbico el lado corto a 256, hace un recorte central de 224 y normaliza con los valores de ImageNet.
- **Representación:** el token CLS de `last_hidden_state[:, 0, :]` (384 dimensiones), normalizado a norma L2.
- **Distancia por ensayo e ítem:** `d = 1 − cos(e_item, m)`, donde `m` es la media de los embeddings de las `trainPics` del ensayo, renormalizada a L2 (centroide).
- **Resultado:** hay 495 valores únicos de d (calculado hoy sobre `h4_long.csv`). La DE cruda de d es 0,2192 con los 24 ítems y 0,2963 con los 8 ítems objetivo (`DESVIACIONES.md` §33).
- **CVCL no se usó en H4.** Solo entró en H6, con CIFAR (`scripts/h6_embed_cvcl.py`). Su preprocesamiento fue Resize 224×224 bicúbico, sin recorte, con normalización ImageNet. La salida es la `fc` de 2048 a 512, normalizada a L2. Para la v2 habría que calcular las distancias de H4 con CVCL por primera vez.

### 1.4 Análisis del orden (exploratorio post hoc)
Fuentes: `DESVIACIONES.md` §34, `scripts/04f_h4_order_moderation.R` y `out/h4_order_moderation.txt`.
- **Moderador:** la columna `order` de `experiment_key.csv`. Vale "1-3" cuando el ensayo de 1 ejemplo va antes que los de 3, lo que ocurre en los experimentos 1, 2, 5, 7, 10 y 12. Vale "3-1" en los experimentos 3, 4, 6, 8, 9 y 11. Hay 86.400 observaciones por nivel.
  - Es una **variable entre experimentos**, y en §34 ya se declaró confundida con `blocking` y `timing`.
- **Modelos:** glmer binomial con bobyqa y efectos aleatorios `(1 + zd | person) + (1 | image)`. z(d) se estandariza sobre los 24 ítems.
  - **Mbase:** `selected ~ zd*condition + condition*order + level`.
  - **Mb:** Mbase + `zd:order`. Se compara con un LR χ²(1).
  - **Mc:** Mb + `zd:condition:order`. Se compara con un LR χ²(3). Holm sobre las 2 pruebas.
- **Resultados con DINOv2** (`out/h4_order_moderation.txt`):
  - **zd:order3-1 en Mb:** 0,2041, EE 0,0410, IC [0,1236; 0,2845]. LR χ²(1) = 24,586, p = 7,1e−07, p Holm = 1,4e−06.
  - **zd:condition:order:** LR χ²(3) = 29,12, p = 2,1e−06. En Mc, zd:order3-1 baja a 0,0614 (p = .248), mientras que zd:conditionsub:order3-1 = 0,3235 (z = 5,18) y zd:conditionsup:order3-1 = 0,1773 (z = 2,52).
  - **Términos condition × order en Mb, que no involucran a d:** sub 2,0027; bas 1,1749; sup 1,0177; order3-1 −1,2175. Todos con |z| > 14.
  - Ningún ajuste tuvo advertencias ni fue singular.
- **Cómo lo dice el paper:** `main.tex`, l.153 (resultados) y l.226 (discusión): "because order is confounded with other experiment-level differences, we do not interpret these terms further". Los análisis post hoc se especificaron después de conocer los resultados principales (l.104 y l.258).

### 1.5 Confusiones ya documentadas que heredamos
1. **El orden está confundido entre experimentos.** Según `experiment_key.csv`, 5 de los 6 experimentos "1-3" tienen `blocking = random` (la excepción es el exp. 12, blocked) y 5 de los 6 "3-1" tienen `blocking = blocked` (la excepción es el exp. 11, random). `timing` y `one_3sub_label` también varían. Los experimentos no prerregistrados son el 9 ("3-1") y el 10 ("1-3"). En la práctica, el orden queda casi alineado con `blocking`.
2. **Solo hay 12 unidades de nivel experimento** (6 por cada orden). El modelo publicado no tiene efecto aleatorio por experimento, así que los EE de los términos con `order` probablemente subestiman la incertidumbre, porque no incluyen la variación entre experimentos. Es una observación nuestra que no se probó en la v1; Ariadna la señaló como bloqueante (B1) y se corrige con el análisis en dos etapas de §5.2.
3. **Personas que repiten:** 520 personas en 600 participaciones; 56 personas participan más de una vez (§26).
4. **Dominio cruzado:** los ítems de otras categorías se eligen en el 0,46 % de los casos y los de la categoría objetivo en el 48,5 % (§24). Por eso, con 24 ítems, el log OR de d lo domina el contraste entre dominios (§33).
5. **Nivel y distancia van juntos** dentro de la categoría objetivo: la distancia media es .33 en sub, .58 en bas y .87 en sup (§24).

---


---

## 2. Pregunta, hipótesis y resultado esperado

**Pregunta.** ¿El orden modera la pendiente de la distancia a nivel de experimento, y esa moderación depende de qué encoder define d?

Separo dos familias de términos:
- **Familia A, orden × condición:** es el efecto conductual. d entra solo como covariable. Es descriptiva y se compara con F0, el modelo sin distancia.
- **Familia B, orden × distancia:** es la que puede depender de la geometría y la que decide.

**Hipótesis en competencia para la familia B:**
- **H_ej:** la moderación no es específica de DINOv2. Aparece con encoders de geometría distinta.
- **H_geo:** la moderación depende de la geometría. Desaparece o cambia de signo con encoders geométricamente distintos.
- **H_nivel-exp:** la moderación de la v1 (zd:order 0,2041, EE 0,0410) se debe a tratar 172.800 observaciones como independientes. Con inferencia a nivel experimento (12 unidades) no se sostiene, ni siquiera con DINOv2. Esta hipótesis se agrega por la revisión de Ariadna (B1).

**Resultado más probable, declarado de antemano.** El orden varía solo entre 12 experimentos (6 y 6) y está casi alineado con `blocking` (§1.5.1). Con eso, el resultado más probable es **"inconcluso"**. En la prueba exacta de permutación, el p bilateral mínimo posible es 2/924 = .0022. "Inconcluso" es un resultado válido para la v2 y se reportará igual que cualquier otro.

**Límites conceptuales (se mantienen):**
- d se calcula a partir de los ejemplos vistos, así que "viene de los ejemplos" y "viene de la geometría" no se excluyen en este diseño. Lo que se puede mostrar es especificidad o no especificidad de encoder.
- Ningún resultado separa el orden de `blocking`, `timing` u otras diferencias entre experimentos.

---

## 3. Encoders: lista final (decisión de Patricio, 18-11 10:17: entran todos)

### 3.1 Identidad, licencia y parámetros
La API de Hugging Face, los model cards y los archivos LICENSE de GitHub se verificaron el 18-11. Los parámetros *solo de visión* los conté en el modelo cargado (`encoder_equivalence/out/vision_params_and_checks.json`, script `00_vision_params_and_checks.py`).

| ID | Checkpoint @ revisión | Objetivo y datos de entrenamiento | Arquitectura | Parámetros de visión | Licencia |
|---|---|---|---|---|---|
| R0 DINOv2-S | `facebook/dinov2-small` @ `ed25f3a3…` | Auto-supervisado, autodestilación (LVD-142M según el paper; no verificado en el card) | ViT-S/14 | 22.056.576 | Apache-2.0 |
| O1 DINOv2-B | `facebook/dinov2-base` @ `f9e44c81…` | Igual a R0; control de tamaño (I4) | ViT-B/14 | 86.580.480 | Apache-2.0 |
| R1 CVCL | `wkvong/cvcl_s_dino_resnext50_embedding` @ `f50eaa0c…` | Contrastivo imagen-lenguaje con la experiencia de un niño | ResNeXt-50 32x4d + fc 2048→512 | 24.028.992 | MIT |
| E1 SigLIP | `google/siglip-base-patch16-224` @ `7fd15f06…` | Contrastivo sigmoide imagen-texto, WebLI en inglés (card) | ViT-B/16 + cabeza MAP | 92.884.224 | Apache-2.0 |
| O2 OpenCLIP | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` @ `1a25a446…` | Contrastivo imagen-texto, LAION-2B | ViT-B/32 + proyección | 87.849.216 | MIT |
| E2 ViT-sup | `google/vit-base-patch16-224` @ `3f49326e…` | Supervisado: ImageNet-21k y luego ImageNet-1k (card) | ViT-B/16 | 85.798.656 (sin cabeza) | Apache-2.0 |
| E3 RN50-sup | `microsoft/resnet-50` @ `34c2154c…` | Supervisado, ImageNet-1k (card) | ResNet-50 | 23.508.032 (sin fc) | Apache-2.0 |
| E4 MAE | `facebook/vit-mae-base` @ `25b184be…` | Auto-supervisado por reconstrucción enmascarada, ImageNet-1k (card) | ViT-B/16 (solo encoder) | 85.798.656 | Card de HF: Apache-2.0; GitHub `facebookresearch/mae`: **CC BY-NC 4.0**. Se trata como no comercial. |
| F1 RN50-aleatoria | `torchvision.models.resnet50(weights=None)`, semilla 0 (primaria); semillas 1 y 2 (sensibilidad, M5) | Ninguno | ResNet-50 | 23.508.032 | BSD-3 (torchvision) |
| F2 Píxeles | — | Ninguno | 3×32×32 | 0 | — |
| F0 Sin distancia | — (sin geometría: no entra en la tabla de equivalencias) | — | — | — | — |

Las revisiones completas están en `encoder_specs.json`.

**Ventaja de casa (I4).** La tubería (CLS de la última capa, centroide, z de d) se diseñó con DINOv2. Lo declaro abiertamente. O1 entra para separar el tamaño de la supervisión.

### 3.2 Representación, preprocesamiento y determinismo
Fuente: `encoder_equivalence/out/encoder_specs.json`, generado por `01_embed.py` desde los procesadores reales.

| ID | Capa y pooling | Dim. | Resize (interp.) | Recorte | Normalización (media / DE) | Determinismo |
|---|---|---|---|---|---|---|
| R0 | CLS, `last_hidden_state[:,0]` (post-LN), última capa | 384 | lado corto 256 (bicúbica) | central 224 | ImageNet | 2 pasadas idénticas (tolerancia 1e−6) |
| O1 | igual que R0 | 768 | lado corto 256 (bicúbica) | central 224 | ImageNet | idem |
| R1 | salida de la fc (después del GAP), como en H6 | 512 | 224×224 directo (bicúbica) | no | ImageNet | eval; sin doble pasada |
| E1 | `pooler_output` (cabeza MAP) | 768 | 224×224 directo (bicúbica) | no | 0,5 / 0,5 | 2 pasadas idénticas |
| O2 | `image_embeds` (CLS → LN → proyección) | 512 | lado corto 224 (bicúbica) | central 224 | CLIP (0,4815; 0,4578; 0,4082) / (0,2686; 0,2613; 0,2758) | idem |
| E2 | CLS, `last_hidden_state[:,0]` (post-LN) | 768 | 224×224 directo (bilineal) | no | 0,5 / 0,5 | idem |
| E3 | `pooler_output` (GAP, 2048) | 2048 | lado corto 224 con `crop_pct` 0,875 (bicúbica)* | central 224* | ImageNet | idem |
| E4 | media de los tokens de parche (post-LN); `mask_ratio = 0.0` y `noise` fijo con orden identidad (M1) | 768 | 224×224 directo (bilineal) | no | ImageNet | 2 pasadas idénticas, después del arreglo |
| E4s | CLS de MAE (sensibilidad) | 768 | idem E4 | no | ImageNet | ver nota M1 |
| F1 | GAP (2048), fc removida | 2048 | lado corto 256 (bicúbica) | central 224 | ImageNet | semilla fija; sin doble pasada |
| F2 | RGB reducido a 32×32 por área, aplanado | 3072 | lado corto 256 (bicúbica) | central 224 | ImageNet | determinista |

\* Para E3, que `ConvNextImageProcessor` con `crop_pct` 0,875 redimensione a 256 y recorte a 224 es como entiendo la implementación; no lo verifiqué en el código.

**Diferencias que quedan documentadas (M2).** Hay recorte central en R0, O1, O2, E3, F1 y F2, y no lo hay en R1, E1, E2 ni E4. E1 y E2 normalizan con 0,5. Lo cubre la sensibilidad S6, con preprocesamiento común. La métrica en todos los casos es la distancia coseno 1 − cos sobre vectores L2.

**Nota M1 (MAE).** `ViTMAEModel` baraja los parches al azar incluso con `mask_ratio = 0`. El CLS falló el chequeo de doble pasada en la corrida de ~10:16–10:17 (`out/log_embed.txt`). Diagnóstico: la diferencia máxima entre pasadas era de 4,8e−6 en el CLS y 3,8e−6 en la media; el coseno mínimo entre CLS con orden fijo y con orden aleatorio fue 0,9999995. Se corrigió pasando `noise = arange(196)`, y E4 y E4s se recalcularon a las 10:18 (mtime de los .npz) con dos pasadas idénticas (diferencia 0,0).

**Control de tubería (M3).** El embedding por lotes de `01_embed.py` frente a `PerceptionLayer.encode` de la v1, en las 45 imágenes: diferencia máxima de 1,5e−8 en el embedding y de 3,6e−7 en las distancias por pares (`vision_params_and_checks.json`). La tolerancia del control sobre d queda fijada en **1e−5**, y además se exige el mismo orden de rangos.

---

## 4. Normalización de d entre encoders (para que los coeficientes sean comparables)
1. **Primaria (igual que en la v1):** z(d) se estandariza dentro de cada encoder y cada conjunto de ítems, con todos los experimentos juntos. Se hace sobre los 24 ítems en el análisis de 24 ítems y sobre los 8 ítems objetivo en el de 8. Así, cada coeficiente está en log-odds por DE de d *de ese encoder*.
2. **Por qué el SESOI se aplica solo con 8 ítems (I3).** Con 24 ítems, una DE de d depende de cuánto separa cada encoder los dominios, es decir, de la razón entre y dentro de dominios, que cambia de un encoder a otro. Con 24 ítems, el coeficiente se reporta junto con esa razón (`distance_scale.csv`, parte A de `02`) y no se le aplica SESOI.
3. **Sensibilidad S5:** se reemplaza z(d) por el rango percentil de d dentro de cada encoder y cada conjunto de ítems. Esto vuelve la escala independiente de la forma de la distribución, por ejemplo de la anisotropía de los espacios imagen-lenguaje.
4. Ninguna d se reescala con información de las respuestas.

---

## 5. Análisis

### 5.1 Datos de entrada
`h4_prep_encoders.py`, por escribir antes del congelamiento, es idéntico a `h4_prep.py` salvo por la función `encode`, que lee `encoder_equivalence/out/emb_<ID>.npz`. Genera `data/h4_long_<ID>.csv` con las mismas filas y el mismo orden que `h4_long.csv`.
- **Control de tubería:** con R0, `dist` debe coincidir con la de `h4_long.csv` (sha256 `d9d30f9d…394dd1`) con |Δ| ≤ 1e−5 y el mismo orden de rangos. Si no coincide, se detiene todo.

### 5.2 Análisis primario de la familia B: dos etapas a nivel experimento (B1)
Ariadna propuso dos remedios: (a) pendiente aleatoria por experimento o (b) 12 pendientes más permutación exacta. **Elijo (b)** por tres razones:
1. Hace la inferencia exactamente al nivel en que se asignó el orden (12 experimentos).
2. La permutación exacta no depende de supuestos asintóticos que, con 12 grupos, son frágiles. Una varianza de pendiente estimada con 12 grupos en un GLMM tiende a quedar singular o mal estimada.
3. Es transparente y barata. Además, como cada experimento tiene su propio efecto de `level`, absorbe la posible moderación del nivel por el orden (I1).

(a) queda como sensibilidad S2.

- **Etapa 1**, por encoder e, conjunto de ítems s ∈ {8, 24} y experimento x = 1…12:
  - Modelo glmer binomial (bobyqa, `maxfun = 2e5`, `set.seed(42)`): `selected ~ zd*condition + level + (1 + zd | person) + (1 | image)`.
  - `condition` se codifica con suma a cero (`contr.sum`), así que el coeficiente de `zd` es la pendiente promedio de las 4 condiciones. Es el análogo del desplazamiento uniforme `zd:order` de Mb.
  - Con 24 ítems, `level` se usa igual que en la v1 (sub/bas/sup asignado a los 24 ítems).
  - zd es el de §4.1.
  - **Fallback pre-especificado:** si el ajuste es singular o no converge, se usa `(1 | person) + (1 | image)`, y queda registrado. Si tampoco converge, la pendiente de ese experimento falta, y el encoder queda **"inconcluso (técnico)"** en ese conjunto de ítems. No se imputa nada.
  - **Salida:** β̂(e, s, x) con su EE.
- **Etapa 2:** Δ(e, s) = media de β̂ en los experimentos "3-1" menos media en los "1-3". El signo es el mismo que `zd:order3-1` en la v1.
  - **Prueba exacta:** las 924 asignaciones 6/6 de las etiquetas de orden. p bilateral = proporción de |T*| ≥ |T_obs|; también se calculan los p unilaterales.
  - **IC 95 % y 90 %:** se obtienen invirtiendo la prueba de permutación: se desplazan las pendientes "3-1" en δ0, con una grilla de paso 0,001. Como secundario se reporta el IC de Welch.
  - **TOST:** se construye con las dos pruebas unilaterales de permutación a ±δ, para δ = 0,05, 0,10 y 0,20 (§7.1).
  - **Supuesto:** que las etiquetas de orden sean intercambiables entre experimentos bajo H0. Como el orden está alineado con `blocking`, rechazar H0 no permite atribuir el efecto al orden.

### 5.3 Control positivo por encoder (I2; obligatorio por decisión de Patricio, 18-11 10:38)
- **Modelo:** el modelo intra-dominio del paper con 8 ítems, `selected ~ zd + level + condition + (1 + zd | person) + (1 | image)` (= C6, `04c`/`04e`).
- **Criterio:** se exige β(zd) < 0 con un IC 95 % que excluya 0. Si un encoder no lo cumple, su familia B no se interpreta.
- **Además:** el coeficiente se reporta como estimando descriptivo.
- **Pisos:** el control se reporta también para F1 y F2, pero ahí no funciona como puerta, porque su papel es justamente ser el piso.

### 5.4 Descriptivos y sensibilidades (no deciden)
- **S1.** El GLMM publicado (Mb y Mc de `04f`) por encoder, con 24 y con 8 ítems: `zd:order3-1`, LR χ²(1) y χ²(3).
- **S2.** Remedio (a) de B1: Mb + `(1 | exp) + (0 + zd | exp)`, con 8 ítems.
- **S3 (I1).** Mb + `level:order`, con 8 ítems. Si θB se desploma, se lee como estructura del diseño (nivel), no como geometría.
- **S4.** Las dos etapas, con 8 ítems, solo con la primera sesión real (AcceptTime, §29).
- **S5.** Las dos etapas, con 8 ítems, con d en rangos (§4.3).
- **S6.** Las dos etapas, con 8 ítems, con preprocesamiento común: lado corto 256 bicúbico, recorte central 224 y la normalización propia de cada modelo.
- **S7.** F1 con las semillas 1 y 2.
- **S8.** MAE con CLS (E4s).
- **Familia A.** Los coeficientes `condition:order3-1` de Mb por encoder, frente a F0 (`selected ~ condition*order + level + (1 | person) + (1 | image)`). Es descriptiva, sin margen fijo, conforme a la decisión de Patricio.

---

## 6. Geometría: una sola regla, medida sobre lo que entra en el modelo (B3)

### 6.1 La regla única (script `encoder_equivalence/scripts/02_equivalence.py`, parte B; sha256 actual `b815be4d…98ad8ab2`, se re-sella al congelar)
- **Cantidad.** Para cada encoder, zd sobre las 57.600 filas de los 8 ítems objetivo, que es exactamente lo que entra en el modelo co-primario. Se residualiza sobre `level` (parcializar nivel) y se calcula r de Pearson entre encoders sobre esas filas. El script **nunca lee** `selected`: usa `usecols` sin esa columna.
- **Agrupamiento.** UPGMA (vínculo promedio) sobre 1 − r, cortado a la altura 0,5. Dentro de cada grupo, la r media entre miembros es ≥ .50. Entran los 10 encoders primarios (R0, O1, R1, E1, O2, E2, E3, E4, F1 con semilla 0 y F2).
- **Calibración pre-especificada.** El criterio *discrimina* solo si O1 queda en el grupo de R0 y si ni F1 ni F2 comparten grupo con algún encoder entrenado. Si la calibración falla, **no se hace ninguna interpretación por grupos**: solo se usan las regiones por encoder.
- **El .90 queda solo como referencia.** Se marca qué encoders tienen r ≥ .90 con R0 ("prácticamente equivalentes a R0"), pero eso no decide nada.
- **Descriptivo adicional.** La misma r con 24 ítems, residualizada sobre `level × in_target`.
- **Momento.** `02` se ejecuta **después** del congelamiento (paso 2 del sábado). Antes de congelar no se ejecutó ni la parte A ni la parte B.
  - La versión anterior del script (`02_equivalence_SUPERSEDED_never_run.py`, con reglas de UPGMA sobre RSA de 45 imágenes) **nunca se ejecutó** y queda reemplazada.
  - La parte B se probó **solo con datos sintéticos** en `/tmp`: funciona y luego se borró.

### 6.2 Contexto descriptivo (parte A de `02`; no decide)
- **Qué calcula.** La matriz encoder × encoder de RSA completa (990 pares), RSA intra-dominio (315 pares) y CKA lineal, sobre las 45 imágenes de los 13 conjuntos de embeddings: R0, O1, R1, E1, O2, E2, E3, E4, E4s, F1 con las semillas 0, 1 y 2, y F2. F0 no tiene geometría y no aparece en la tabla.
- **Bootstrap.** B = 1.000 con `default_rng(42)`, remuestreando 15 imágenes por dominio, de forma pareada entre encoders.
- **Escala de distancias.** También calcula la razón entre y dentro de dominios, `distance_scale.csv`, que se usa con I3.
- **Salidas.** `out/equivalence_descriptive_long.csv`, `out/zd_corr_*.csv`, `out/equivalence_grouping.json` y `figs/encoder_equivalence_descriptive.png`.
- **Estado.** La tabla de equivalencias **no está calculada todavía**. Se calcula el sábado, después de congelar.
  - Patricio había autorizado calcularla ya (10:13), pero la revisión de Ariadna (B3) pidió correr el script solo tras congelar, para que la regla quede fijada sin haber visto la geometría.
  - Lo único que se conoce de geometría es la RSA de la prueba de humo del 18-11 a las ~10:00 (§8.3).
- **Uso.** La tabla **no se usa para elegir encoders**. El conjunto lo fijó Patricio (10:17) y no depende de la tabla.

---

## 7. Regiones de decisión: excluyentes, con precedencia y multiplicidad bien ubicada (B2)

### 7.1 Por encoder: regiones a tres márgenes (Δ de §5.2; δ ∈ {0,05; 0,10; 0,20} log-odds por DE)
- **Decisión de Patricio (18-11 10:37):** no se elige un margen único. Las regiones se evalúan y se reportan con **tres márgenes, δ = 0,05, 0,10 y 0,20**, como espectro. La conclusión se lee a lo largo del rango. Ningún δ es el preferido ni el "por defecto", tampoco 0,10, que era el valor único de la versión anterior.
- **Dónde se aplica δ.** Solo con 8 ítems (I3). Referencia: el `zd:order3-1` de la v1 vale 0,2041, así que δ = 0,20 está cerca de la magnitud completa de la v1 y δ = 0,10 de su mitad. Esa calibración en DINOv2 queda declarada como ventaja de casa (I4).
- **Regiones con 8 ítems, para cada δ por separado.** Se aplica **la primera que se cumpla**:
  1. **Signo opuesto:** el límite superior del IC 95 % es < 0. No depende de δ.
  2. **No se mantiene (equivalente a 0):** el IC 90 % está dentro de (−δ, +δ), que es la TOST.
  3. **Se mantiene:** el límite inferior del IC 95 % es > 0 **y** Δ ≥ δ.
  4. **Atenuado:** el límite inferior del IC 95 % es > 0 **y** Δ < δ.
  5. **Inconcluso:** todo lo demás, incluido "inconcluso (técnico)", que no depende de δ.
- **Asimetría conocida y declarada de antemano.** Con un δ grande, la TOST es más fácil de pasar (región 2) y "se mantiene" es más difícil (exige Δ ≥ δ). Con un δ chico ocurre lo contrario. Por eso, que un encoder cambie de región a lo largo del espectro es esperable y no constituye una contradicción. Lo informativo es si la región es la misma en los tres márgenes.
- **Regiones con 24 ítems (sin δ; no cambian).**
  1. **Signo opuesto:** el límite superior es < 0.
  2. **Positivo:** el límite inferior es > 0. Se reporta junto con la razón entre y dentro de dominios.
  3. **Inconcluso:** lo demás.
- **Ejemplos.** Δ = 0,05 con IC [0,01; 0,09] (el caso de Ariadna) da "se mantiene" con δ = 0,05, porque Δ ≥ δ; "no se mantiene" con δ = 0,10; y "no se mantiene" con δ = 0,20. Su perfil es, por lo tanto, "dependiente del margen". Δ = 0,15 con IC [0,02; 0,28]: "se mantiene" con 0,05 y 0,10; con 0,20, "atenuado" si el IC 90 % no cae dentro de ±0,20.

### 7.2 Cómo se lee el espectro (fijado antes de ver datos, para que no se pueda elegir después)
1. **Una sola tabla obligatoria:** filas = encoders (los 10 primarios); columnas = región con δ = 0,05, 0,10 y 0,20 (8 ítems) más la región con 24 ítems. Debajo van Δ, el IC 95 %, el IC 90 % y los p (sin ajuste y con Holm por δ). Las tres columnas de δ se reportan siempre, juntas y en ese orden.
2. **Perfil por encoder:**
   - **"Robusto: <región>"** si la región es la misma con los tres δ.
   - **"Dependiente del margen"** en cualquier otro caso. Se escribe explícitamente qué rango de δ sostiene cada región, por ejemplo: "se mantiene para δ ≤ 0,10; atenuado para δ = 0,20".
3. **Lectura global:** la conclusión G0–G4 (§7.4) se evalúa por separado con cada δ, con las mismas reglas y la misma precedencia, y se reporta en una fila por δ.
   - **"Robusta: Gk"** si la conclusión es la misma con los tres δ.
   - **"Dependiente del margen"** si no lo es, indicando el rango de δ en que vale cada conclusión.
4. **Prohibiciones:**
   - En el texto de resultados no se puede reportar una conclusión de un solo δ sin las otras dos.
   - No se puede decir "se mantiene" (ni "no se mantiene") sin calificativo, salvo que el perfil sea robusto.
   - No se puede elegir δ después de ver los datos.
   - Si en algún momento se agregara otro δ, sería una desviación registrada y quedaría fuera de la lectura confirmatoria.
5. Lo que **no depende de δ** (signo opuesto, inconcluso técnico, las regiones con 24 ítems, el control positivo y los pisos con la región 1) se informa una sola vez.

### 7.3 Multiplicidad
- **Conjunción** ("se mantiene en todos"): es una prueba intersección-unión. Cada encoder se evalúa a su propio 95 %, **sin Holm**, con cada δ.
- **Unión** ("depende del encoder"), **con Holm aplicado por separado con cada δ**:
  - **Familia de entrenados:** los 7 encoders entrenados distintos de R0 (O1, R1, E1, O2, E2, E3, E4), cada uno con 2 pruebas: p_TOST(δ) (el mayor de los dos p unilaterales a ±δ) y p de signo opuesto (unilateral, el mismo con todos los δ). Son 14 pruebas por δ, con **Holm** en conjunto.
  - **Familia de pisos (M4):** F1 y F2, con las mismas 2 pruebas cada uno. Son 4 pruebas por δ, con Holm aparte.
- **No se corrige entre δ.** Los tres márgenes no son tres afirmaciones independientes que se puedan elegir. Son un mismo análisis leído a tres resoluciones, y la regla de §7.2 exige mostrar los tres. La protección frente a elegir el margen viene de esa regla, no de un ajuste.
  - **Consecuencia:** una afirmación "robusta" es más exigente que cualquier δ aislado.
- **Sensibilidades y descriptivos:** no entran en ninguna familia.

### 7.4 Conclusión global (8 ítems; se evalúa con cada δ y se aplica la primera que se cumpla)
| # | Condición (con un δ dado) | Lectura permitida |
|---|---|---|
| G0 | Falla el control de tubería (no depende de δ) | Se detiene todo; no hay lectura |
| G1 | R0 no está en la región "se mantiene" con ese δ | **La moderación de la v1 no se sostiene, con ese margen, con inferencia a nivel experimento.** Con ese δ, la comparación entre encoders pasa a ser descriptiva |
| G2 | ≥ 1 encoder entrenado distinto de R0 (que pasó el control positivo) cae en la región 1 o 2 con Holm < .05 en la familia de entrenados con ese δ | **Depende del encoder** con ese margen. Se describe en qué grupos de geometría están los encoders que fallan |
| G3 | Los 8 entrenados están en "se mantiene", la calibración discrimina y los entrenados ocupan ≥ 2 grupos de geometría | **No es específico de DINOv2 (a través de geometrías)** con ese margen. Con 24 ítems se informa si también son todos "positivos"; si no, la lectura es "solo con 8 ítems" |
| G3b | Los 8 entrenados están en "se mantiene", pero la calibración falla o hay un solo grupo | Se mantiene en todos los encoders con ese margen, **sin evidencia de que difieran en la geometría que entra en el modelo** |
| G4 | Cualquier otro caso (lo más probable) | **Inconcluso o mixto** con ese margen, reportado encoder por encoder |

- **Resultado global:** el patrón a lo largo de los tres δ (§7.2.3). Por ejemplo, "robusta: G4" o "G3 con δ = 0,05 y 0,10; G1 con δ = 0,20 (dependiente del margen)".
  - G1 también depende de δ: con δ = 0,20, R0 necesita Δ ≥ 0,20, prácticamente la magnitud completa de la v1, para quedar en "se mantiene".
- **Pisos (se leen aparte, con cada δ y con la misma regla de perfil):**
  - si F1 y F2 están en "se mantiene", la moderación no requiere una geometría aprendida;
  - si ambos caen en la región 1 o 2 con Holm de pisos, la moderación requiere alguna geometría aprendida;
  - si no, es inconcluso.
- **S1–S8 y la familia A** no cambian la región. Si contradicen la conclusión, se informa.
- **Lo que nunca se afirma:** que el orden *causa* el efecto, ni que el efecto "viene de los ejemplos" en sentido causal.

---

## 8. Riesgos y confusiones
1. **Orden ~ `blocking` ~ experimento** (§1.5.1). Es el límite estructural. Las dos etapas corrigen la *precisión*, no la *confusión*.
2. **Poca potencia con 12 unidades.** Lo más probable es G4 o G1 (§2).
3. **Primera sesión y personas que repiten:** se cubren con S4. En la etapa 1 cada persona aparece una sola vez por experimento.
4. **Dominancia del contraste entre dominios con 24 ítems:** por eso los 8 ítems son co-primarios y los márgenes δ se aplican solo con 8 ítems (I3).
5. **Nivel confundido con d y una posible interacción `level:order` no modelada (I1):** S3, y además la etapa 1 tiene un efecto de nivel propio por experimento.
6. **Preprocesamiento (M2):** S6.
7. **Escala y anisotropía:** §4 y S5.
8. **Pooling:** una sola representación primaria por encoder; solo el CLS de MAE como sensibilidad.
9. **Determinismo de MAE:** resuelto con `noise` fijo (M1).
10. **Conocimiento categórico de los encoders supervisados e imagen-lenguaje** (las categorías están en ImageNet y probablemente en WebLI y LAION; M7): no es filtración de respuestas humanas.
11. **Convergencia en la etapa 1** (experimentos de 50 personas): hay un fallback fijo y no se imputa nada.
12. **Licencias:** MAE se trata como no comercial. No se redistribuyen pesos ni derivados por ensayo de XTMEM (§36).
13. **RAM (M6):** al revisar a las ~10:00 había 2,7 GB disponibles. Durante los embeddings (10:15–10:19) el log muestra entre 6,5 y 12,1 GB. En la ejecución se usan como máximo 3 procesos de R y se vigila `MemAvailable`.
14. **Ventaja de casa (I4):** la tubería y los márgenes δ (referidos al 0,2041 de la v1) se calibraron con DINOv2.
15. **Bifurcaciones:** todo lo que decide está en §5.2, §6.1 y §7. Cualquier otro análisis es descriptivo y se rotula así.

**Fuera de alcance (trabajo futuro):**
- separar el orden de `blocking` y `timing` (haría falta un diseño factorial con participantes nuevos);
- la causalidad de "los ejemplos";
- probar capas intermedias;
- repetir H6 (el aprendiz en CIFAR) con los encoders nuevos.

---

## 9. Costo (recalculado por B1)

### 9.1 Base medida
- **GLMM de orden de la v1** (04f; 3 modelos con 172.800 observaciones): ~35,5 min. Partió a las 18:06:53 según la nota de hora en el encabezado post hoc de `DESVIACIONES.md` y la salida quedó escrita a las 18:42:24 (mtime).
- **Embeddings de las 45 imágenes:** segundos por encoder (`log_embed.txt`, 10:15–10:19). Las descargas suman ~2,65 GB según los tamaños de `model.safetensors` en la API de HF.

### 9.2 Estimación para el fin de semana (los tiempos de la etapa 1 son **estimaciones no medidas**)
Supuesto: el tiempo escala en forma aproximadamente lineal con las observaciones. Un experimento con 24 ítems tiene 14.400 observaciones y uno con 8 ítems tiene 4.800, así que estimo entre 1 y 3 min por ajuste con 24 ítems y entre 0,5 y 1 min con 8 ítems. El primer paso del sábado lo mide (§10, paso 3).

| Bloque | Ajustes | CPU estimada |
|---|---|---|
| Distancias (`h4_prep_encoders`, 13 fuentes) + `02` | — | < 15 min |
| **Dos etapas, 8 ítems** (10 fuentes × 12 experimentos) | 120 | 1–2 h |
| **Dos etapas, 24 ítems** (10 × 12) | 120 | 2–6 h |
| Control positivo, intra-dominio (9 fuentes nuevas, ~10 min) | 9 | ~1,5 h |
| S1: Mb y Mc con 24 ítems (9 × ~36 min) y con 8 ítems (10 × ~12 min) | 19 | ~7,5 h |
| S2: `(1 | exp) + (0 + zd | exp)`, 8 ítems (10 × ~15 min) | 10 | ~2,5 h |
| S3: `level:order`, 8 ítems (10 × ~12 min) | 10 | ~2 h |
| S4–S8: dos etapas con 8 ítems (4 × 10 fuentes y 3 fuentes extra) × 12 | ~520 | 4–8 h |
| Familia A: F0 | 2 | ~0,3 h |
| **Total** | | **~21–30 h de CPU**; con 3 procesos de R en paralelo, **~8–11 h de reloj** |

- **Impacto de los tres δ (10:37):** ninguno en los ajustes. Son los mismos 120 + 120 ajustes de la etapa 1; solo cambian la etapa 2 (pruebas de permutación desplazadas a ±δ, segundos) y `08_regions.py`.
- **Tope de presupuesto:** 32 h de CPU. Si se supera, las sensibilidades se cortan en orden inverso de prioridad (S8 → S1), y queda registrado. No hay re-corridas selectivas.
- **Costo monetario:** ninguno.
- **Tiempo humano:** escribir y probar los scripts antes del congelamiento, ~3–4 h. Revisión breve de Ariadna de los cambios, ~1 h. Supervisión el fin de semana, ~1 jornada. Resultados y desviaciones, ~½ jornada.

### 9.3 Qué se ejecutó antes del congelamiento (I5)
- **18-11 ~10:00:** prueba de tiempos con DINOv2-S y una ResNet-50 aleatoria sobre las 45 imágenes. Como dato colateral salió una RSA completa de ρ = .158, con una sola semilla y sin IC. No decide nada.
- **18-11 10:15–10:19:** `01_embed.py` generó los embeddings de las 13 fuentes. E4s falló el chequeo de determinismo (~10:16–10:17) y se recalculó a las 10:18, junto con E4, tras el arreglo M1.
- **18-11 ~10:24:** `00_vision_params_and_checks.py` contó los parámetros de visión y comprobó la tubería de R0 contra la v1 (§3.2).
- **Ninguno de estos pasos usó** `h4_long.csv`, `all_raw_A.csv` ni respuestas de participantes. No se calculó ninguna d de ensayo ni ninguna RSA o CKA entre los encoders nuevos.

---

## 10. Plan para congelar

### 10.1 Contenido que se congela
- **Encoders (10 primarios):** R0, O1, R1, E1, O2, E2, E3, E4, F1 (semilla 0) y F2. F0 se suma para la familia A. Sensibilidades: E4s y F1 con las semillas 1 y 2.
- **Distancias:** §5.1. Normalización: §4.
- **Familia B, primario:** las dos etapas a nivel experimento (§5.2), con 8 y 24 ítems co-primarios (decisión de Patricio a las 10:17). Regiones con δ = 0,05, 0,10 y 0,20 como espectro, solo con 8 ítems (decisión de Patricio a las 10:37; §7.1–§7.2).
- **Control positivo:** §5.3, obligatorio por encoder (aprobado a las 10:38). Sensibilidades: §5.4.
- **Corte del agrupamiento de geometría:** 0,5 por defecto (§6.1), salvo que Patricio diga otra cosa antes de congelar.
- **Geometría:** una sola regla (§6.1, `02_equivalence.py`).
- **Regiones a tres márgenes, lectura del espectro, multiplicidad y conclusión global:** §7.
- **Presupuesto y parada:** §9.2.

### 10.2 Lista antes del congelamiento (jueves 19 a viernes 20)
1. Aprobación de Patricio: hecha (10:17, 10:37 y 10:38; ver §10.4). Solo queda el corte de 0,5, que va por defecto.
2. Escribir los scripts que faltan: `h4_prep_encoders.py`, `05_twostage_stage1.R`, `06_stage2_permutation.py`, `07_pooled_glmm.R` (control positivo, S1–S3 y F0) y `08_regions.py`.
   - `08_regions.py` implementa §7 literalmente, con los tres δ y la tabla única de §7.2.
   - Se prueban **solo con datos sintéticos**.
3. Segunda revisión de Ariadna: el plan final y los scripts. Si encuentra algo de fondo, se avisa a Evaristo antes de correr.
4. Congelamiento interno:
   - `FREEZE_encoders_v2.txt` con la hora (UTC−3);
   - el sha256 de este archivo (sin el campo de hash);
   - el sha256 de todos los scripts y de `encoder_specs.json`.
   - No es un prerregistro externo.

### 10.3 Ejecución (sábado 21 y domingo 22; solo tras el paso 4)
**Sábado:**
- 0. Verificar los hashes.
- 1. `h4_prep_encoders.py` con todas las fuentes, más el control de tubería (G0).
- 2. `02_equivalence.py`: tabla descriptiva, regla de geometría y calibración.
- 3. Cronometrar un ajuste de la etapa 1 y actualizar el presupuesto (sin cambiar el análisis).
- 4. Dos etapas con 8 ítems, todas las fuentes.
- 5. Dos etapas con 24 ítems.
- 6. Etapa 2 y `08_regions.py`.
- 7. Control positivo.

**Domingo:**
- 8. S1 → S8 en ese orden, dentro del presupuesto.
- 9. Familia A.
- 10. `RESULTADOS_encoders.md`, nuevas entradas de `DESVIACIONES` y aviso a Evaristo.
- No se publica nada sin Patricio.

### 10.4 Decisiones de Patricio
**Ya decididas:**
- **10:17:** entran todos los encoders, incluidos O1 y O2; la tabla de equivalencias reemplaza a los márgenes de geometría; los 8 ítems son co-primarios; se autoriza el cómputo, a ejecutar el sábado 21 o el domingo 22; Ariadna revisa antes de congelar.
- **10:37:** **δ.** No hay margen único: las regiones se reportan con δ = 0,05, 0,10 y 0,20 como espectro (§7.1–§7.2).
- **10:38:** se aprueba **B1**, el análisis en dos etapas con permutación exacta como inferencia que decide, con el GLMM publicado como descriptivo (§5.2).
- **10:38:** se aprueba el **control positivo intra-dominio obligatorio por encoder** (§5.3).
- **10:38:** se aprueba que **la tabla de equivalencias se calcule después de congelar** (§6, B3).

**Único punto pendiente:**
1. **Corte del agrupamiento de geometría** (§6.1). No se mencionó en las decisiones. Queda **0,5 por defecto**, salvo que Patricio indique otro valor antes del congelamiento.

**Siguiente paso antes de congelar:** Ariadna hará una **segunda revisión de este plan final**.
- Si no encuentra nada de fondo, se congela con hash (§10.2, paso 4) y se ejecuta el sábado 21 y el domingo 22.
- Si encuentra algo de fondo, se avisa a Evaristo antes de correr.

---

## 11. Cómo se resolvió cada punto de la revisión de Ariadna
- **B1:** dos etapas con permutación exacta (§5.2), (a) como S2, "inconcluso" declarado como lo más probable (§2) y costo recalculado (§9).
- **B2:** regiones excluyentes con precedencia, evaluadas con tres δ como espectro (§7.1–§7.2, decisión de Patricio a las 10:37), conjunción sin Holm y unión con Holm por δ (§7.3).
- **B3:** una sola regla medida sobre zd con 8 ítems parcializando nivel (§6.1). El script viejo nunca se corrió y fue reemplazado; el nuevo corre solo tras congelar. El .90 queda como referencia.
- **I1:** S3, más el efecto de nivel por experimento en la etapa 1.
- **I2:** control positivo intra-dominio (§5.3), aprobado por Patricio como obligatorio (10:38).
- **I3:** los δ solo con 8 ítems; con 24 ítems, la razón entre y dentro de dominios (§4).
- **I4:** O1 en el núcleo y la ventaja de casa declarada.
- **I5:** §9.3.
- **M1:** `noise` fijo (§3.2).
- **M2:** tabla de preprocesamiento (§3.2) y S6.
- **M3:** tolerancia de 1e−5 más el orden de rangos (§3.2 y §5.1).
- **M4:** dos familias de Holm, por δ (§7.3).
- **M5:** F1 con semilla 0 en el primario y semillas 1 y 2 como S7.
- **M6:** RAM actualizada (§8.13).
- **M7:** sin cambios (§8.10).
- **Calibración** sugerida por Ariadna (O1 equivalente; F1 y F2 distintos): §6.1.
