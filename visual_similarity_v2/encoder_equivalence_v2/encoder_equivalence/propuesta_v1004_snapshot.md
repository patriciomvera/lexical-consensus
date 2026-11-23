# Propuesta de diseño: comparación de encoders para el efecto de orden (v2 de arXiv 2611.14116)

- **Estado:** BORRADOR DE DISEÑO, no congelado. No se ha ejecutado nada del experimento. No se ha calculado ninguna distancia con encoders nuevos sobre los estímulos de Lewis & Frank, salvo la prueba de tiempos descrita en §8.3.
- **Fecha:** 2026-11-18, ~10:00 hora de Santiago (UTC−3).
- **Origen:** pedido de Patricio (18-11, 09:57, vía Evaristo), a partir de la sugerencia de @1shareShare1 en X: cambiar el backbone y ver si el efecto del orden se mantiene. Si se mantiene con cualquier encoder, el efecto vendría de los ejemplos vistos y no de la geometría de DINOv2.
- **Alcance:** solo diseño. Para ejecutar se necesita la aprobación de Patricio (§10). No se propone título ni encuadre del paper.
- **Fuentes:** todas las cifras citadas salen de los archivos de `/workspace/lexical-consensus-openmind/human_data_analyses/` (equivale a la carpeta `human_data_analyses/` del repo `patriciomvera/lexical-consensus`, commit 1e7a904) y de `paper_arxiv_v_concl/main.tex`, con archivo y sección indicados.

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
2. **Solo hay 12 unidades de nivel experimento** (6 por cada orden). El modelo publicado no tiene efecto aleatorio por experimento, así que los EE de los términos con `order` probablemente subestiman la incertidumbre, porque no incluyen la variación entre experimentos. Es una observación nuestra que no se probó en la v1. La trato como riesgo en §7.
3. **Personas que repiten:** 520 personas en 600 participaciones; 56 personas participan más de una vez (§26).
4. **Dominio cruzado:** los ítems de otras categorías se eligen en el 0,46 % de los casos y los de la categoría objetivo en el 48,5 % (§24). Por eso, con 24 ítems, el log OR de d lo domina el contraste entre dominios (§33).
5. **Nivel y distancia van juntos** dentro de la categoría objetivo: la distancia media es .33 en sub, .58 en bas y .87 en sup (§24).

---

## 2. Pregunta e hipótesis

**Pregunta.** ¿Dependen los términos del orden de qué encoder define d?

Hay que separar dos familias de términos, porque la conjetura de X no distingue entre ellas.

- **Familia A, orden × condición** (`condition:order`): el efecto conductual del orden, como en Lewis & Frank. d entra solo como covariable. Con un encoder distinto, A solo cambia por el ajuste de la covariable.
- **Familia B, orden × distancia** (`zd:order`, `zd:condition:order`): si el orden modera la pendiente de la distancia. Depende directamente del encoder.

**Hipótesis en competencia (para la familia B, que es la que puede depender de la geometría):**
- **H_ej (la lectura de X):** la moderación de la pendiente de d por el orden no es específica de DINOv2. Aparece, con el mismo signo, con encoders de supervisión y arquitectura distintas, y también con los controles de piso que tienen geometría poco o nada aprendida.
- **H_geo:** la moderación depende de propiedades de la geometría. Aparece con DINOv2 y con algunos encoders, pero no con otros geométricamente distintos, o cambia de signo.
- **H_nulo-informativo:** con encoders cuya geometría es prácticamente la misma que la de DINOv2, la invariancia no informa nada. Por eso se exige el control de geometría (§5.4).

**Advertencia conceptual (para no sobreinterpretar).**
1. d es una función de los ejemplos vistos, porque es la distancia al centroide de los ejemplos del ensayo. "Viene de los ejemplos" y "viene de la geometría" no son alternativas excluyentes en este diseño. Lo que el experimento puede mostrar es si los términos son *específicos de un encoder*, no de dónde viene causalmente el efecto.
2. Como el orden varía solo entre experimentos (§1.5), ningún resultado de este diseño permite atribuir el efecto al orden en sí, separándolo de `blocking`, `timing` u otras diferencias entre experimentos.

---

## 3. Encoders propuestos (verificados hoy en sus fuentes primarias)

Verificación hecha el 18-11-2026 con la API de Hugging Face (`/api/models/<id>`: sha, licencia y `safetensors.total`), los model cards (README) y los LICENSE de GitHub. Se fija el `sha` de la revisión para que el análisis sea reproducible.

### 3.1 Referencias (ya en el proyecto)
| ID | Checkpoint (revisión) | Supervisión | Arquitectura | Parámetros | Licencia |
|---|---|---|---|---|---|
| R0 DINOv2-S | `facebook/dinov2-small` @ `ed25f3a31f01632728cabb09d1542f84ab7b0056` | Auto-supervisada (autodestilación) | ViT-S/14 | 22.056.576 (API HF) | Apache-2.0 (card HF y LICENSE de GitHub `facebookresearch/dinov2`) |
| R1 CVCL | `wkvong/cvcl_s_dino_resnext50_embedding` @ `f50eaa0c50a6076a5190b1dd52aeeb6c3e747045` | Contrastiva imagen-lenguaje, con datos de un niño | CNN ResNeXt-50 32x4d + fc 2048→512 | **No verificado** (la API no lo informa) | MIT (card HF; `README.md` del repo) |

### 3.2 Encoders nuevos (núcleo: 4)
| ID | Checkpoint (revisión) | Qué aporta al contraste | Arquitectura | Parámetros (API HF) | Licencia (verificada) | Representación pre-especificada |
|---|---|---|---|---|---|---|
| E1 SigLIP | `google/siglip-base-patch16-224` @ `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Contrastiva imagen-lenguaje a gran escala (WebLI, pares imagen-texto en inglés, según el card). Contrasta con R1, que también es imagen-lenguaje pero con datos de un niño. | ViT-B/16 | 203.155.970 **en total, visión + texto**. La parte de visión sola no está verificada. | Apache-2.0 (card HF) | `get_image_features` (pooling MAP de la torre de visión), L2 |
| E2 ViT-sup | `google/vit-base-patch16-224` @ `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` | Supervisada por clasificación (preentrenada en ImageNet-21k y ajustada en ImageNet-1k, según el card). Mantiene la familia ViT y el token CLS de R0 y cambia la supervisión. | ViT-B/16 | 86.567.656 (incluye la cabeza de clasificación) | Apache-2.0 (card HF; LICENSE de GitHub `google-research/vision_transformer`) | CLS de `last_hidden_state` (post-layernorm) de `ViTModel`, L2. Mismo criterio que R0. |
| E3 RN50-sup | `microsoft/resnet-50` @ `34c2154c194f829b11125337b98c8f5f9965ff19` | Supervisada en ImageNet-1k, CNN. Contrasta la arquitectura con E2 (misma clase de supervisión) y con R1 (casi la misma arquitectura, distinta supervisión y datos). | ResNet-50 | 25.610.152 (incluye fc) | Apache-2.0 (card HF) | `pooler_output` (global average pool, 2048), L2 |
| E4 MAE | `facebook/vit-mae-base` @ `25b184bea5538bf5c4c852c79d221195fdd2778d` | Auto-supervisada con un objetivo distinto (reconstrucción enmascarada en vez de autodestilación). Separa "auto-supervisado" de "DINO". | ViT-B/16 | 111.907.840 (encoder + decoder; el encoder solo no está verificado) | **Discrepancia:** el card HF dice Apache-2.0, pero el LICENSE de GitHub `facebookresearch/mae` es **CC BY-NC 4.0**. Se trata como no comercial, lo que es compatible con publicar resultados de investigación sin redistribuir los pesos. | Media de los tokens de parche de `ViTMAEModel` con `mask_ratio = 0.0` (el config trae 0.75, que enmascara al azar en la inferencia). CLS como sensibilidad. Esta elección de pooling es nuestra y la justificación no está verificada en el paper de MAE. |

### 3.3 Controles de piso (sin pesos aprendidos)
| ID | Definición | Para qué |
|---|---|---|
| F1 RN50-aleatoria | `torchvision.models.resnet50(weights=None)`, semillas 0, 1 y 2. GAP 2048, L2. Código BSD-3 (LICENSE de `pytorch/vision`). | Arquitectura sin aprendizaje. Si la familia B aparece aquí con la misma fuerza, el término no depende de una geometría aprendida. |
| F2 Píxeles | Mismo preprocesamiento que R0 (lado corto 256, recorte central 224, normalización ImageNet), reducido a 32×32 por área y aplanado (3.072), L2. | Similitud de bajo nivel. Es el piso de "cualquier distancia". |
| F0 Sin distancia | Modelo sin `zd`: `selected ~ condition*order + level + (1 | person) + (1 | image)`. | Referencia para la familia A, sin ningún encoder. |

### 3.4 Opcionales (solo si Patricio quiere más resolución)
| ID | Checkpoint | Por qué | Parámetros | Licencia |
|---|---|---|---|---|
| O1 DINOv2-B | `facebook/dinov2-base` @ `f9e44c814b77203eaa57a6bdbbd535f21ede1415` | Control de tamaño: R0 es ViT-S y E2/E4 son ViT-B. O1 separa supervisión de tamaño. | 86.580.480 | Apache-2.0 |
| O2 OpenCLIP | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` @ `1a25a446712ba5ee05982a381eed697ef9b435cf` | Segunda réplica imagen-lenguaje, de la familia CLIP. | 151.277.439 (visión + texto) | MIT (card HF) |

**Por qué no OpenAI CLIP como encoder principal.** `openai/clip-vit-base-patch32` no declara licencia en la API de HF. El repo de GitHub `openai/CLIP` es MIT, y el card declara fuera de alcance "any deployed use case". Para investigación sería usable, pero se prefiere E1/O2, que tienen la licencia declarada en el propio checkpoint.

**Preprocesamiento.** Como primario se usa el procesador por defecto de cada checkpoint, que es lo que define a cada encoder. Para CVCL se usa el de H6. Como sensibilidad se aplica un preprocesamiento común a todos (lado corto 256 bicúbico, recorte central 224, normalización propia del modelo). Los estímulos son cuadrados de 720×720, así que la diferencia entre recortar y no recortar se reduce a un borde de aproximadamente un 12,5 %.

---

## 4. Protocolo (igual al publicado, sin cambios)

1. **Distancias.** Un script nuevo, `h4_prep_encoders.py`, idéntico a `h4_prep.py` salvo por la función `encode`, genera `data/h4_long_<enc>.csv` con las mismas filas y el mismo orden.
   - d = 1 − coseno entre el embedding del ítem (L2) y el centroide de las `trainPics` (media de los embeddings L2, renormalizada).
   - **Control de tubería:** con R0, el script debe reproducir `dist` de `data/h4_long.csv` (sha256 `d9d30f9d…`) con una tolerancia de 1e−6. Si no lo hace, se detiene todo.
2. **Modelos.** Se usa `04f_h4_order_moderation.R` sin cambios, salvo la ruta del archivo de entrada: Mbase, Mb, Mc, LR y Holm sobre 2. z(d) se estandariza dentro de cada encoder sobre los 24 ítems, como en la v1.
3. **Réplica con 8 ítems objetivo.** Las mismas fórmulas, restringidas a `in_target == 1`, con z(d) re-estandarizada en ese subconjunto. Pasa a ser co-primaria por la dominancia del contraste entre dominios (§1.5.4; `DESVIACIONES.md` §33).
4. **Control positivo por encoder.** El modelo H4 primario (C1, `04c`) da β(zd) < 0. Si un encoder no lo cumple, sus términos de orden no se interpretan.
5. **Sensibilidades pre-especificadas** (no deciden):
   - (i) Añadir `(1 | exp)` a Mb y Mc.
   - (ii) Solo la primera sesión real por AcceptTime, como en `04d` (§29).
   - (iii) d por rango percentil dentro de cada encoder, en vez de z.
   - (iv) Preprocesamiento común (§3).
6. **Semillas y entorno.** `set.seed(42)` en R; `torch.manual_seed(0)` y semillas 0–2 para F1. Python con torch 2.14.1+cpu y transformers 5.20.2 (`/workspace/rv_venv`); R 4.5.3 con lme4 2.0.6 (`/tmp/renv`). Las revisiones de los checkpoints quedan fijadas por `sha`.

---

## 5. Reglas de decisión (para fijar ANTES de calcular cualquier d nueva)

Las cifras de SESOI son una **propuesta**. Patricio debe aceptarlas o cambiarlas antes de congelar el plan.

### 5.1 Estimandos
- **θB1(e):** coeficiente `zd:order3-1` en Mb, con IC Wald al 95 %, más su LR χ²(1). Es el estimando principal de la familia B.
- **θB2(e):** LR χ²(3) de `zd:condition:order` (Mc vs Mb). Como descriptivo, el coeficiente `zd:conditionsub:order3-1`.
- **θA(e):** los tres coeficientes `condition:order3-1` de Mb.

### 5.2 SESOI propuesto
- **Familia B:** ±0,10 log-odds por DE de d, la mitad del valor de R0 (0,2041). Así, "no se mantiene" significa una moderación al menos dos veces menor que con DINOv2.
- **Familia A:** ±0,25 log-odds, alrededor de una cuarta parte del menor `condition:order` de R0 (sup, 1,0177).

### 5.3 Reglas
- **B se mantiene en el encoder e:** θB1(e) > 0 y el IC 95 %, ajustado por Holm en la familia (§5.5), excluye 0.
- **B no se mantiene en e:** el IC 90 % de θB1(e) cae completo dentro de (−0,10; +0,10), según TOST; o bien θB1(e) < 0 con un IC que excluye 0 (signo opuesto).
- **Inconcluso en e:** cualquier otro caso.
- **"Se mantiene con cualquier encoder"**, que es la lectura de X, a nivel de no-especificidad. Requiere las cuatro condiciones:
  - (a) B se mantiene en los 5 encoders entrenados distintos de R0 (R1, E1–E4);
  - (b) al menos 3 de ellos pasan el control de geometría (§5.4);
  - (c) lo anterior se cumple con 24 ítems **y** con 8 ítems objetivo;
  - (d) los controles positivos pasan.
  - Si se cumple solo con 24 ítems, se informa como "no específico de DINOv2 solo en el modelo dominado por el contraste entre dominios", que es una conclusión más débil.
- **"Depende del encoder":** B no se mantiene en ≥ 1 encoder entrenado que pase el control de geometría y que haya pasado el control positivo.
- **Resultado mixto:** cualquier otra combinación. Se informa encoder por encoder, sin agregar.
- **Pisos (F1, F2):**
  - Si B se mantiene también en F1 y F2, la moderación no requiere una geometría aprendida. Es compatible con que provenga de la estructura del diseño (dominio y nivel de los ejemplos), pero no lo demuestra.
  - Si B no aparece en los pisos y sí en los encoders entrenados, el término requiere alguna geometría aprendida, aunque no la de DINOv2 en particular.
- **Familia A:** "no depende de la covariable" si, para los 3 coeficientes, |θA(e) − θA(F0)| < 0,25 en todos los encoders, con el mismo signo que F0. La comparación es descriptiva, porque las estimaciones comparten datos y no son independientes.
- **Comparaciones entre encoders:** son descriptivas. Todos los modelos usan las mismas 172.800 observaciones, así que no se hace un metaanálisis ni se calcula I² como si fueran estudios independientes. Se reporta el rango y un forest plot por encoder.

### 5.4 Control de geometría (para que una invariancia sea informativa)
- **Qué se calcula:** las matrices de distancia coseno entre las 45 imágenes (990 pares) de cada encoder, y Spearman (RSA) de cada encoder contra R0, con IC bootstrap por imágenes (B = 2.000, semilla 42).
  - (i) Matriz completa.
  - (ii) Dentro de dominio (3 × 105 pares), porque el contraste entre dominios infla cualquier correlación.
  - (iii) Correlación de los 495 valores únicos de d de ensayo con los de R0.
- **Criterio propuesto:** un encoder es "geométricamente distinto de R0" si el límite superior del IC de la RSA dentro de dominio es < .90. Si todos los encoders tienen RSA ≥ .90 con R0, la invariancia de B **no** se puede usar a favor de H_ej.
- **Momento:** este control no usa respuestas humanas. Aun así, se calcula **después** de congelar el plan, para que no influya en la elección de encoders.

### 5.5 Multiplicidad
- **Familia B confirmatoria nueva:** 2 LR por encoder × 7 fuentes de d (R1, E1–E4, F1, F2) = 14 pruebas por conjunto de ítems. Se aplica Holm dentro de cada conjunto (24 y 8). R0 no entra, porque su resultado ya se conoce.
- **Familia A:** descriptiva, sin pruebas.
- **Sensibilidades y opcionales:** no entran en ninguna familia.

### 5.6 Estado confirmatorio honesto
- Las respuestas humanas ya se analizaron con DINOv2 y los términos de orden se conocen (§1.4). Por eso el brazo R0 **no es confirmatorio**, y los demás solo lo son en el sentido de que las d nuevas aún no existen.
- El plan se congelaría internamente, con hora y sha256 en un archivo aparte. **No es un prerregistro externo.**
- Los resultados se reportan todos, incluidos nulos, mixtos e inconclusos, sin elegir encoders a posteriori.

---

## 6. Qué cambiaría nuestra lectura (resumen)
| Resultado | Lectura permitida | Lectura NO permitida |
|---|---|---|
| B se mantiene en R1 y E1–E4, con geometría distinta y con 24 y 8 ítems | La moderación del orden sobre la pendiente de d no es específica de DINOv2 | "El orden causa…"; "viene de los ejemplos" en sentido causal (orden confundido entre experimentos) |
| B se mantiene también en F1 y F2 | Basta cualquier distancia que siga la estructura dominio/nivel del diseño | Que la geometría visual no importe para H4 en general |
| B desaparece o cambia de signo en ≥ 1 encoder distinto | La moderación depende del encoder; el resultado v1 es específico de DINOv2 | Que DINOv2 sea "el correcto" |
| Todos los encoders con RSA ≥ .90 frente a R0 | Ninguna sobre especificidad (invariancia no informativa) | Apoyo a H_ej |
| B solo con 24 ítems | Dependería del contraste entre dominios | Generalizar al dominio enseñado |
| Cambia mucho con `(1 | exp)` | Los EE de la v1 para términos de orden eran optimistas | — |

---

## 7. Riesgos y confusiones conocidas
1. **Orden confundido con `blocking`, `timing` y el experimento** (§1.5.1). Es el límite principal y ningún encoder lo resuelve. Con solo 12 experimentos, la sensibilidad `(1 | exp)` puede ensanchar mucho los IC.
2. **Seudoreplicación a nivel experimento:** el modelo publicado no modela la varianza entre experimentos. Hay que reportarlo aunque haga más débiles los resultados de la v1.
3. **Primera sesión y personas que repiten** (§26, §29): se cubren con la sensibilidad (ii).
4. **Techo y dominancia entre dominios** (§24, §33): casi cualquier encoder separa vegetales, vehículos y animales, así que con 24 ítems la invariancia es casi esperable. De ahí la réplica con 8 ítems como co-primaria y el control de geometría dentro de dominio.
5. **Nivel confundido con d dentro del dominio objetivo** (§24): `level` está en el modelo, pero la colinealidad puede inflar los EE de forma distinta según el encoder.
6. **Diferencias de preprocesamiento** (recorte o no, resolución, normalización): sensibilidad (iv).
7. **Escala y anisotropía de las distancias:** los espacios imagen-lenguaje suelen tener rangos de coseno estrechos. z(d) iguala la escala lineal, pero no la forma de la distribución. Sensibilidad (iii) con rangos.
8. **Elección de pooling** (CLS, MAP, GAP, media de parches): es un camino que se bifurca. Se fija una sola representación primaria por encoder (§3) y no se prueban otras salvo el CLS de MAE como sensibilidad declarada.
9. **MAE con `mask_ratio` 0.75 por defecto:** si no se fija en 0.0, los embeddings son aleatorios. El control es que dos pasadas den embeddings idénticos.
10. **Conocimiento categórico de los encoders supervisados e imagen-lenguaje:** sus distancias pueden seguir la taxonomía, y precisamente eso es parte del contraste. No es filtración de datos, porque no ven respuestas humanas, pero hay que nombrarlo al interpretar.
11. **Convergencia:** si Mb o Mc no convergen o son singulares con un encoder, se reporta tal cual. Solo se permite el fallback ya especificado en §34 (si Mc no converge, se reporta Mb). No se re-especifica el modelo.
12. **Licencias:** la de MAE es ambigua (§3.2) y se trata como no comercial. No se redistribuyen pesos. Los derivados por ensayo de XTMEM no se redistribuyen (§36); solo se publicarían los scripts y los agregados.
13. **Recursos del computador:** hay 8 CPU y 15 GB de RAM, pero al revisar (18-11, ~10:00) había solo ~2,7 GB disponibles (`MemAvailable`). Correr GLMM en paralelo podría quedarse sin memoria.

**Fuera de alcance (trabajo futuro, no medible con estos datos):**
- Separar el orden de `blocking` y `timing`, para lo que haría falta un diseño factorial nuevo con participantes.
- Saber si el efecto viene "de los ejemplos" en sentido causal.
- Probar capas intermedias.
- Repetir H6 (el aprendiz en CIFAR) con los encoders nuevos. La extracción de CIFAR-100 con CVCL tomó ~6 h en CPU (`DESVIACIONES.md` §21), así que sería un proyecto aparte.

---

## 8. Costo estimado

### 8.1 Base medida
- **Computador:** 8 CPU (Intel Xeon), 15 GB de RAM y ~2,7 GB disponibles al revisar. Sin GPU.
- **GLMM de orden de la v1 (04f, 24 ítems, 3 modelos):** el proceso partió a las 18:06:53 (`DESVIACIONES.md`, nota de corrección de hora en el encabezado de la sección post hoc, antes de §34) y la salida quedó escrita a las 18:42:24 (mtime de `out/h4_order_moderation.txt`, 31-10). Son **~35,5 min** por encoder.
- **Embeddings (prueba de hoy, §8.3):** DINOv2-S codifica las 45 imágenes en 2,1 s; una ResNet-50 aleatoria las crea y codifica en 2,4 s (4 hilos). El costo de los embeddings es despreciable.

### 8.2 Tabla por encoder
Las descargas se estiman como parámetros × 4 bytes (fp32), sin verificar el tamaño real de los archivos. Los tiempos de los GLMM de 8 ítems y del modelo C1 son **estimaciones no medidas**: escalan con las observaciones, 57.600 frente a 172.800.

| Fuente de d | Descarga aprox. | Embeddings (45 img.) | GLMM orden 24 ítems | GLMM orden 8 ítems (est.) | Control positivo C1 (est.) | Total aprox. |
|---|---|---|---|---|---|---|
| R0 DINOv2-S | 0 (en caché) | < 5 s | ya existe | ~12 min | ya existe | ~12 min |
| R1 CVCL | 0 (en caché) | < 10 s | ~36 min | ~12 min | ~10 min | ~1 h |
| E1 SigLIP | ~0,8 GB | < 30 s | ~36 min | ~12 min | ~10 min | ~1 h |
| E2 ViT-sup | ~0,35 GB | < 15 s | ~36 min | ~12 min | ~10 min | ~1 h |
| E3 RN50-sup | ~0,1 GB | < 10 s | ~36 min | ~12 min | ~10 min | ~1 h |
| E4 MAE | ~0,45 GB | < 15 s | ~36 min | ~12 min | ~10 min | ~1 h |
| F1 RN50-aleatoria (3 semillas) | 0 | < 10 s | 3 × ~36 min (o solo la semilla 0 en el primario) | ~12 min | ~10 min | ~1–2 h |
| F2 Píxeles | 0 | < 5 s | ~36 min | ~12 min | ~10 min | ~1 h |
| F0 Sin distancia | 0 | — | ~10 min (est.) | ~5 min (est.) | — | ~15 min |
| RSA y bootstrap (§5.4) | — | — | — | — | — | < 5 min |

- **Núcleo** (R1, E1–E4, F1 con la semilla 0, F2, F0): **~7–8 h de CPU en serie**. Con 2 procesos de R en paralelo, **~4 h de reloj**, si la RAM disponible lo permite.
- **Sensibilidades** (i)–(iv): solo sobre Mb, unas 4 × 8 × ~15 min, es decir **~8 h** adicionales (estimación).
- **Opcionales** O1 y O2: ~1 h cada uno.
- **Tiempo humano:** congelar el plan y una revisión independiente ~2 h; ejecución supervisada ~1 jornada; redacción de resultados y desviaciones ~½ jornada.
- **Costo monetario:** ninguno, porque todo corre en CPU local con checkpoints públicos.

### 8.3 Prueba de tiempos hecha hoy (transparencia)
- **Qué se hizo:** el 18-11 a las ~10:00 se codificaron las 45 imágenes con DINOv2-S (en caché) y con una ResNet-50 aleatoria (semilla 0). No se usaron respuestas humanas ni se descargó nada.
- **Resultado colateral:** la RSA descriptiva de la matriz completa (990 pares) entre esas dos fue ρ = .158.
- **Alcance:** es una sola semilla, sin IC ni procedimiento congelado. No es un resultado del experimento y no se usará para decidir nada. Solo indica que el piso F1 probablemente tiene una geometría distinta de R0.

---

## 9. Plan en formato de la skill (para completar al congelar)
```text
Pregunta e hipótesis en competencia: §2 (H_ej, H_geo, H_nulo-informativo)
Unidad de observación y dependencia: decisión ítem×ensayo; anidada en persona (520) y participación (600); imagen (24); experimento (12) con order entre experimentos
Contraste primario: θB1(e) por encoder; θB2(e); 24 y 8 ítems co-primarios; confusión order~blocking/timing declarada
Baselines y controles: R0 (referencia); positivo C1 β(zd)<0; pisos F1, F2; F0 sin distancia; control de tubería (reproducir dist de R0); control de geometría §5.4
Fuentes, versiones, licencias: §1.1, §3 (sha de revisión, licencias verificadas 18-11-2026)
Métricas y criterios: §5 (SESOI ±0,10 B, ±0,25 A, propuestos)
Tamaño, parada y presupuesto: datos fijos (172.800 obs.); un ajuste por encoder; tope ~16 h CPU incluidas sensibilidades; sin re-corridas selectivas
Salvaguardas de filtración y semillas: sin respuestas humanas en la elección de encoders/pooling; semillas §4.6
Frontera exploratorio/confirmatorio: R0 ya conocido (no confirmatorio); d nuevas aún no calculadas; prueba de tiempos §8.3 declarada
Congelamiento interno: PENDIENTE (hora con zona + sha256 en archivo aparte)
Prerregistro externo: ninguno
Revisión; alcance; aprobaciones pendientes: §10
```

---

## 10. Puerta de implementación y aprobaciones pendientes
- **Dentro del alcance, si se aprueba:**
  - descargar 4 checkpoints, de ~1,7 GB en total;
  - calcular las d de 45 imágenes;
  - ajustar los GLMM de §4 y §5;
  - escribir `RESULTADOS_encoders.md` y las entradas nuevas de `DESVIACIONES.md`.
  - Todo se haría en una carpeta nueva, sin tocar el plan congelado de la v1.
- **Fuera del alcance:** cambiar la v1, hacer push o publicar en GitHub, HF, arXiv u Overleaf, nuevos datos humanos, H6 con CIFAR y propuestas de título o encuadre.
- **Decisiones de Patricio antes de congelar:**
  1. Aprobar o cambiar el conjunto de encoders: núcleo E1–E4 más pisos; O1 y O2 sí o no.
  2. Aprobar o cambiar los SESOI (±0,10 y ±0,25) y el umbral de RSA (.90).
  3. Decidir si la réplica de 8 ítems es co-primaria, como se propone, o solo sensibilidad.
  4. Autorizar el cómputo (~4 h de reloj para el núcleo y ~8 h para las sensibilidades).
  5. Decidir si se pide una revisión metodológica independiente antes de congelar.
- **Riesgos sin resolver:** la confusión del orden entre experimentos (estructural) y la RAM disponible.
