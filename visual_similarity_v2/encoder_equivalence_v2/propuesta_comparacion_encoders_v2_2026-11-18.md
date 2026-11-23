# Propuesta de diseño: comparación de encoders para el efecto de orden (v2 de arXiv 2611.14116)

- **Estado:** PLAN FINAL CON LA RESPUESTA A LA SEGUNDA REVISIÓN DE ARIADNA (REVISION2). INCLUYE EL RANGO DE CORTES DE GEOMETRÍA (PATRICIO, 10:44) Y LA RESPUESTA A REVISION3. FALTA QUE ARIADNA REVISE EL DIFF (`encoder_equivalence/DIFF_REVISION3.patch`). NO SE CONGELA ANTES DEL VIERNES 20-11. **No está congelado** (no tiene hash de congelamiento). No se ha ajustado ningún GLMM ni se ha leído ninguna respuesta de participantes para esta v2.
- **Origen:** pedido de Patricio (18-11, 09:57, vía Evaristo), a partir de la sugerencia de @1shareShare1 en X: cambiar el backbone y ver si el efecto del orden se mantiene.
- **Alcance:** solo diseño. No se propone título ni encuadre del paper.
- **Fuentes de las cifras:** todas salen de archivos, con archivo y sección indicados: `human_data_analyses/` (commit 1e7a904 del repo `patriciomvera/lexical-consensus`), `paper_arxiv_v_concl/main.tex` y `encoder_equivalence/out/`.

## Registro de cambios (hora de Santiago, UTC−3)
- **2026-11-18 10:04.** Primera versión (sha256 `afb7e139…46845c1a`; copia en `encoder_equivalence/propuesta_v1004_snapshot.md`). Es la versión que revisó Ariadna.
- **2026-11-18 10:13.** Patricio, vía Evaristo, pide una tabla de equivalencias entre todos los encoders en vez de un umbral de .90 contra DINOv2.
- **2026-11-18 10:15–10:19.** Se calcularon embeddings de las 45 imágenes de estímulo. No son distancias de ensayo ni usan respuestas humanas.
- **2026-11-18 10:17.** Decisiones de Patricio sobre §10:
  1. entran todos los encoders, incluidos DINOv2-B y OpenCLIP;
  2. la tabla de equivalencias reemplaza a los márgenes de geometría y al umbral fijo;
  3. la réplica con 8 ítems es co-primaria;
  4. se autoriza el cómputo, pero se ejecuta solo el sábado 21 o el domingo 22 de noviembre;
  5. antes de congelar hay revisión independiente de Ariadna.
- **2026-11-18 10:20** (mtime del archivo; su encabezado dice "~10:45"). Revisión de Ariadna (`REVISION_ariadna_comparacion_encoders_v2_2026-11-18.md`): "congelable con cambios".
- **2026-11-18 10:25–10:39** (mtime de la versión resultante 10:39:24, sha256 `e5d2806a…`; corrige el "10:25–11:00" anterior, m1 de REVISION2). Esta versión incorpora B1, B2 y B3, los puntos I1–I5 y M1–M7, y las decisiones de Patricio. El detalle de cada cambio está en §11.
- **2026-11-18 10:37.** Decisión de Patricio, vía Evaristo, sobre δ: no hay margen único. Las regiones de B2 se evalúan y se reportan con δ = 0,05, 0,10 y 0,20 como espectro. Se reescribió §7 (nuevos §7.1–§7.4: tabla única, perfiles "robusto" o "dependiente del margen", Holm por δ y G0–G4 por δ) y se actualizaron §5.2, §9.2, §10 y §11. Este cambio no tiene impacto en el costo (§9.2).
- **2026-11-18 10:38.** Patricio confirma, vía Evaristo: se aprueban B1 (dos etapas con permutación exacta), el control positivo obligatorio por encoder y la tabla de equivalencias después de congelar. El corte de 0,5 queda por defecto como único punto pendiente. Ariadna hará una segunda revisión del plan final antes de congelar (§10.4).
- **2026-11-18 10:41** (mtime del archivo; su encabezado dice "~10:50"). Segunda revisión de Ariadna (`REVISION2_ariadna_comparacion_encoders_v2_2026-11-18.md`) sobre `e5d2806a…`: congelable tras BC1 y tras revisar los scripts.
- **2026-11-18 10:44. Respuesta a REVISION2.** Copia previa en `encoder_equivalence/propuesta_e5d2806a_snapshot_1039.md`. Cambios (detalle en §11.2):
  - **BC1:** nueva región por encoder "no interpretable (control positivo)", que no depende de δ; esos encoders salen de la familia Holm (2·k pruebas); G3 y G3b exigen que todos los entrenados pasen el control (si no, G4); si R0 falla el control, se detiene todo como en G0 (§5.3, §7.1, §7.3, §7.4).
  - **I-a:** la región 2 pasa a llamarse "menor que δ (equivalente con margen δ)"; la lógica no cambia (§7.1, §7.2).
  - **I-b:** se quita la equivalencia "δ = 0,20 ≈ magnitud de la v1" (otra escala); el Δ en dos etapas de R0 con 8 ítems se reporta como referencia descriptiva, no confirmatoria. Los tres márgenes no cambian (§7.1, §7.4, §8.14).
  - **m1–m3:** horas del registro, "márgenes de geometría", G2 "sin grupos" si falla la calibración, e intercambiabilidad con las 56 personas que repiten (§5.2).
  - **Aclaraciones al escribir los scripts (no cambian decisiones):** nombres reales de los scripts (§10.2); un encoder "inconcluso (técnico)" tampoco entra en su familia Holm (no tiene p); en G2, el p que debe pasar Holm es el de la prueba que define la región (signo opuesto para la región 1, p_TOST(δ) para la región 2); "no interpretable" se aplica también a la región con 24 ítems; S6 necesita embeddings con preprocesamiento común, que genera `01b_embed_common.py`; en la etapa 1, el ajuste de fallback se acepta si converge aunque sea singular (queda registrado), porque §5.2 solo excluye el que "tampoco converge" (en lme4 2.0.6 la singularidad aparece entre los mensajes de convergencia y se separa explícitamente); si el control positivo de un encoder no se pudo ajustar, cuenta como "no pasa"; en §10.3, el control positivo pasa antes de `08_regions.py`.
- **2026-11-18 10:44.** Decisión de Patricio, vía Evaristo: el corte del agrupamiento de geometría es un **rango, 1 − r = 0,3, 0,5 y 0,7**, con la misma lógica que los δ (coincide en los tres = "robusto"; si no, "depende del corte"; ningún corte se reporta sin los otros dos; ninguno es preferido). La autocalibración se evalúa en cada corte por separado; un corte que falla la calibración da "sin grupos" en ese corte. Las lecturas por grupos se reportan en una grilla δ × corte con una regla de perfil declarada. Razón registrada: los encoders son distintos y no se esperan equivalencias exactas. El congelamiento no ocurre antes del viernes 20-11. Aplicado a las 10:53 en §6.1, §7.2, §7.4, §10 y §11.3, y en `02_equivalence.py` y `08_regions.py`.
- **2026-11-18 ~12:20** (encabezado; el archivo llegó antes de las 12:04). Tercera revisión de Ariadna (`REVISION3_ariadna_comparacion_encoders_v2_2026-11-18.md`) sobre `75678c79…`: congelable tras BC3, I1, I2 y el punto (6).
- **2026-11-18 12:04–12:11. Respuesta a REVISION3.** Copias previas: `encoder_equivalence/propuesta_75678c79_snapshot.md` y `encoder_equivalence/scripts_snapshot_75678c79/`; diff en `encoder_equivalence/DIFF_REVISION3.patch`.
  - **BC3:** el control de rangos de G0 ya no es estricto: los valores de d de la v1 que distan < 1e−6 se tratan como un mismo casi-empate (§3.2, §5.1). Se mantiene |Δ| ≤ 1e−5. **Desviación respecto del arreglo literal de Ariadna** (redondear a 6 decimales): el redondeo separa un casi-empate que cae justo en un borde de redondeo (p. ej. 0,3000004999 frente a 0,3000005001) y detendría G0 sin motivo; la regla por grupos de casi-empates hace lo que ella pide sin ese caso borde. Pruebas: ruido 1e−8 PASA, casi-empate invertido a 3e−7 PASA, borde de redondeo PASA (con el redondeo fallaría), intercambio real con brecha 5e−6 FALLA. §1.3: "495 valores únicos" son unos 290 a 1e−6 (medición de Ariadna en REVISION3; no la repetí, porque no abro `h4_long.csv`).
  - **I1:** nueva sensibilidad **S9** (§5.4): etapa 1 con `(1 | person) + (1 | image)` en los 12 experimentos, etapa 2 igual, más la tabla tipo de ajuste × orden por encoder. Si S9 cambia la región de un encoder con algún δ (las regiones no dependen del corte), ese encoder se rotula "sensible al tipo de ajuste" (descriptivo; no cambia regiones ni G). **No se cambia el primario a un modelo único: esa decisión es de Patricio y queda pendiente (§10.4).** *(Superado: Patricio decidió a las 12:12 y 12:14; ver abajo.)* El generador sintético ahora tiene varianza de pendiente por persona.
  - **I2:** prueba de humo de `01b_embed_common.py` (S6) hecha (§9.3). S6 pasa al final del domingo y su falla no detiene lo demás (§10.3).
  - **(2)/(3):** la tabla de regiones muestra k y los encoders excluidos de cada familia Holm (§7.3).
  - **(6):** se volvió a correr `01_embed.py`: los 13 npz salen idénticos byte a byte (§9.3). `rules_02_sha256_before_run.txt` listaba un hash viejo de `02` (`b815be4d…`) y ningún hash de `01_embed.py`; ahora tiene los actuales. `01_embed.py` cambió solo para registrar `crop_pct` en `encoder_specs.json` (la versión anterior lo omitía al reescribir; el campo se había agregado a mano), sin efecto en los embeddings.
  - **Menores:** expectativa del corte 0,7 en §8; el tope de 32 h se vigila a mano (§9.2); G0 se lee de un JSON y no buscando texto en un log.
- **2026-11-18 12:12.** Decisión de Patricio, vía Evaristo (opción B de REVISION3, I1): el primario de la etapa 1 usa el mismo modelo en los 12 experimentos, el más rico que converja en los 12; el ajuste mixto con fallback queda como sensibilidad secundaria (S9), solo para reportar. Después, Ariadna pidió (a) la escalera fijada en el plan, de más rica a más simple, con el criterio de convergencia (avisos de lme4, singularidad y tolerancias) y una elección que dependa solo de la convergencia, y (b) dejar explícito si la estructura se elige una vez para todos o por encoder.
- **2026-11-18 12:14.** Patricio confirma, vía Evaristo, la opción (1): **una sola estructura común para todos los encoders**, elegida solo por convergencia; la variante por encoder es descriptiva. La constante `STRUCTURE_SCOPE` queda fija en `"common"` (confirmado por Patricio 18-11 12:14).
- **2026-11-18 12:15–12:40. Implementación de las decisiones de 12:12 y 12:14** (incluida en `encoder_equivalence/DIFF_REVISION3.patch`; detalle en §11.4):
  - §5.2: escalera L1 `(1 + zd | person) + (1 | image)` → L2 `(1 + zd || person) + (1 | image)` → L3 `(1 | person) + (1 | image)` → L4 `(1 | person)`; criterio de convergencia con tolerancias explícitas; **un ajuste singular cuenta como no convergido**; peldaño común para los 10 primarios (pisos incluidos), por separado con 8 y 24 ítems; encoder sin ningún peldaño: fuera de la elección y evaluado con el común.
  - §5.4: S9 pasa a ser el ajuste mixto (primer peldaño que converge en cada experimento) y no cuesta ajustes; S4–S8 usan el peldaño común de 8 ítems.
  - §8.11, §9.2 (peor caso del costo; el primario no se corta), §10.2, §10.3 (paso 5b) y §10.4 (sin puntos pendientes).
  - **Más allá de la decisión (para Evaristo):** que la singularidad cuente como no convergencia; los peldaños intermedios L2 y L4; que los pisos entren en la elección; que el peldaño se elija por separado con 24 ítems; que S4–S8 hereden el peldaño de 8 ítems; la redefinición de S9; y el peor caso del costo.
- **2026-11-18 ~12:55** (encabezado del archivo). Revisión del diff por Ariadna (`REVISION3b_ariadna_diff_2026-11-18.md`) sobre `4ff4b341…`: diff correcto; un arreglo IMPORTANTE y dos menores. El visto bueno para `09_freeze.py` queda condicionado además a revisar la propuesta de simulaciones de agentes (Patricio, 12:36), que va en un documento aparte.
- **2026-11-18 12:42–13:06. Respuesta a REVISION3b.** Copias previas: `encoder_equivalence/propuesta_4ff4b341_snapshot.md` y `encoder_equivalence/scripts_snapshot_4ff4b341/`; diff en `encoder_equivalence/DIFF_REVISION3b.patch`. Detalle en §11.5.
  - **IMPORTANTE:** si R0 queda "inconcluso (técnico)" con 8 ítems, la conclusión es **G0 "sin lectura: técnico"** en las 9 celdas de la grilla, nunca G1 (§7.4). Con 24 ítems, la lectura de referencia de R0 es "sin lectura: técnico".
  - **Menor:** G0 falla también si algún grupo de casi-empates de la v1 mide ≥ 1e−6 (§5.1). La prueba de ruido 1e−8 ahora perturba el archivo nuevo y no el de la v1 (el de la v1 está fijado por su sha256); con el ruido en el archivo de la v1, los datos sintéticos (que tienen por azar dos valores a 9,9e−7) daban un grupo de 1,01e−6.
  - **Menor:** sin estructura común no hay caída a L4: `05b` termina con código 4 y `run_weekend.sh` se detiene y registra la desviación (§5.2). También se corrigió que un conjunto vacío de encoders elegibles daba L1 por vacuidad.

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
- **Resultado:** hay 495 valores únicos de d (calculado hoy sobre `h4_long.csv`), que a una resolución de 1e−6 son unos 290 (medición de Ariadna, REVISION3; muchos son el mismo centroide calculado con otro orden de imágenes). La DE cruda de d es 0,2192 con los 24 ítems y 0,2963 con los 8 ítems objetivo (`DESVIACIONES.md` §33).
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

**Control de tubería (M3).** El embedding por lotes de `01_embed.py` frente a `PerceptionLayer.encode` de la v1, en las 45 imágenes: diferencia máxima de 1,5e−8 en el embedding y de 3,6e−7 en las distancias por pares (`vision_params_and_checks.json`). La tolerancia del control sobre d queda fijada en **1e−5**, y además se exige el mismo orden de rangos, **tratando como empate los valores de la v1 que distan < 1e−6** (BC3 de REVISION3: en la v1 hay casi-empates de ~1e−9 que son el mismo centroide calculado con otro orden de imágenes; la brecha real más chica es 4,6e−6, según Ariadna). Formalmente: se agrupan los valores de d de la v1 cuya brecha consecutiva es < 1e−6, y en el archivo nuevo cada grupo debe quedar estrictamente por debajo del siguiente.

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
- **Control de tubería:** con R0, `dist` debe coincidir con la de `h4_long.csv` (sha256 `d9d30f9d…394dd1`) con |Δ| ≤ 1e−5 y el mismo orden de rangos con los casi-empates (< 1e−6) agrupados (§3.2, BC3). Además, ningún grupo de casi-empates de la v1 puede medir 1e−6 o más (de su mínimo a su máximo; protección contra el encadenamiento, REVISION3b; en la v1 el más ancho mide 6,0e−8 según Ariadna). Si algo no se cumple, se detiene todo. El resultado se escribe en `pipeline_control.json`, que es lo que lee `08_regions.py`.

### 5.2 Análisis primario de la familia B: dos etapas a nivel experimento (B1)
Ariadna propuso dos remedios: (a) pendiente aleatoria por experimento o (b) 12 pendientes más permutación exacta. **Elijo (b)** por tres razones:
1. Hace la inferencia exactamente al nivel en que se asignó el orden (12 experimentos).
2. La permutación exacta no depende de supuestos asintóticos que, con 12 grupos, son frágiles. Una varianza de pendiente estimada con 12 grupos en un GLMM tiende a quedar singular o mal estimada.
3. Es transparente y barata. Además, como cada experimento tiene su propio efecto de `level`, absorbe la posible moderación del nivel por el orden (I1).

(a) queda como sensibilidad S2.

- **Etapa 1**, por encoder e, conjunto de ítems s ∈ {8, 24} y experimento x = 1…12:
  - Modelo glmer binomial (Laplace, bobyqa, `maxfun = 2e5`, `set.seed(42)`) con efectos fijos `selected ~ zd*condition + level`, iguales en todos los peldaños de la escalera de abajo.
  - `condition` se codifica con suma a cero (`contr.sum`), así que el coeficiente de `zd` es la pendiente promedio de las 4 condiciones. Es el análogo del desplazamiento uniforme `zd:order` de Mb.
  - Con 24 ítems, `level` se usa igual que en la v1 (sub/bas/sup asignado a los 24 ítems).
  - zd es el de §4.1.
  - **Estructura de efectos aleatorios: una sola, común a todos los encoders (decisión de Patricio, 18-11 12:12, opción B de REVISION3; alcance "común" confirmado por Patricio 18-11 12:14).** Reemplaza al fallback por experimento de las versiones anteriores, que pasa a ser S9 (§5.4, solo descriptivo).
    - **Escalera fija, de más rica a más simple** (`LADDER` en `lib_glmm.R`):
      - L1 `(1 + zd | person) + (1 | image)`;
      - L2 `(1 + zd || person) + (1 | image)` (pendiente sin correlación con el intercepto);
      - L3 `(1 | person) + (1 | image)`;
      - L4 `(1 | person)`.
    - **Criterio de convergencia (fijo; tolerancias escritas explícitamente en `LADDER_CTL`, iguales a las de lme4 2.0.6 por defecto):** un ajuste converge si y solo si (i) no da error; (ii) el optimizador termina con código 0; (iii) lme4 no emite ningún aviso de convergencia (chequeo de gradiente con tol 2e−3 y `relTol` nulo; chequeo de Hessiana con tol 1e−6; p. ej. "failed to converge", "nearly unidentifiable", "unable to evaluate scaled gradient"); y (iv) **no es singular** (`isSingular`, tol 1e−4): **un ajuste singular cuenta como no convergido.** Otros avisos de R (p. ej. de `glm.fit`) se registran, pero no cuentan.
    - **Regla de elección (constante `STRUCTURE_SCOPE = "common"` en `05b_select_structure.py`; sin opción en tiempo de ejecución):** el peldaño común es el más rico que converge en los 12 experimentos **para cada uno** de los 10 encoders primarios (R0, O1, R1, E1, O2, E2, E3, E4, F1 semilla 0 y F2; **los pisos F1 y F2 entran en la elección**; F0 no tiene zd y no entra). Se elige por separado para 8 y para 24 ítems. La elección depende **solo** de la convergencia; nunca de β̂, Δ ni p (`05b` no lee esas columnas para elegir; hay una prueba que lo verifica).
    - **Cómo se busca:** pasada 1, en cada experimento se ajusta L1 → L4 hasta el primer peldaño que converge; pasada 2, se ajustan los peldaños que falten para decidir (p. ej., si un encoder necesita L2 en un experimento, se ajusta L2 en los 12 experimentos de todos).
    - **Encoder sin ningún peldaño que converja en sus 12 experimentos:** sale de la elección (para no bajar a todos a nada) y se ajusta igual con el peldaño común. Si ahí falta alguna pendiente, queda **"inconcluso (técnico)"** en ese conjunto de ítems, igual que cualquier encoder al que le falte una pendiente con el peldaño común. No se imputa nada.
    - **Sin estructura común (REVISION3b):** si para un conjunto de ítems no hay ningún peldaño común (por ejemplo, todos los encoders quedan fuera de la elección, o convergen solo en peldaños distintos), `05b_select_structure.py` termina con código 4, escribe `structure_DEVIATION.txt` y no produce tablas de la etapa 1; `run_weekend.sh` se detiene y registra la desviación. **No hay caída silenciosa a L4 ni a otro peldaño.** Lo mismo vale para el peldaño de 8 ítems que heredan S4–S8 y S6 (`rung8`): si falta, se detiene y se registra (S6 sigue sin bloquear el resto).
    - **Descriptivo (no entra en las regiones):** el peldaño que habría alcanzado cada encoder por separado (variante `per_encoder_rung`) y la tabla `structure.csv` con el primer peldaño que convergió en cada experimento.
    - **Costo que se acepta:** un encoder difícil (por ejemplo, un piso) puede bajar el peldaño de todos. Se prefiere así por comparabilidad entre encoders.
  - **Salida:** β̂(e, s, x) con su EE.
- **Etapa 2:** Δ(e, s) = media de β̂ en los experimentos "3-1" menos media en los "1-3". El signo es el mismo que `zd:order3-1` en la v1.
  - **Prueba exacta:** las 924 asignaciones 6/6 de las etiquetas de orden. p bilateral = proporción de |T*| ≥ |T_obs|; también se calculan los p unilaterales.
  - **IC 95 % y 90 %:** se obtienen invirtiendo la prueba de permutación: se desplazan las pendientes "3-1" en δ0, con una grilla de paso 0,001. Como secundario se reporta el IC de Welch.
  - **TOST:** se construye con las dos pruebas unilaterales de permutación a ±δ, para δ = 0,05, 0,10 y 0,20 (§7.1).
  - **Supuesto:** que las etiquetas de orden sean intercambiables entre experimentos bajo H0. Como el orden está alineado con `blocking`, rechazar H0 no permite atribuir el efecto al orden.
  - **Personas que repiten (m3 de REVISION2):** 56 personas participan en más de un experimento (§1), así que las 12 pendientes no son del todo independientes y la intercambiabilidad se rompe en parte. S4 (solo la primera sesión) cubre este punto.

### 5.3 Control positivo por encoder (I2; obligatorio por decisión de Patricio, 18-11 10:38)
- **Modelo:** el modelo intra-dominio del paper con 8 ítems, `selected ~ zd + level + condition + (1 + zd | person) + (1 | image)` (= C6, `04c`/`04e`).
- **Criterio:** se exige β(zd) < 0 con un IC 95 % (Wald) que excluya 0. Si un encoder no lo cumple, su familia B no se interpreta.
- **Consecuencias (BC1 de REVISION2):**
  - el encoder queda en la región **"no interpretable (control positivo)"**, con 8 y con 24 ítems y con los tres δ (§7.1);
  - sale de la familia Holm de entrenados (§7.3);
  - G3 y G3b solo son posibles si **todos** los entrenados pasan el control; si alguno falla, la conclusión es G4, salvo que antes se cumpla G1 o G2 (§7.4);
  - **si falla R0, se detiene todo, como en G0**: sin la puerta del encoder de referencia no hay lectura.
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
- **S4–S8 usan el peldaño común de 8 ítems** que se eligió el sábado para el primario (§5.2); no se vuelve a elegir. Un experimento que no converge con ese peldaño deja la sensibilidad "inconclusa (técnico)" para ese encoder.
- **S9 (I1 de REVISION3; redefinida por la decisión de Patricio de las 12:12).** El ajuste mixto: en cada experimento, el primer peldaño de la escalera que converge (pasada 1), con lo que se mezclan estructuras entre experimentos. Las dos etapas, con 8 ítems. No requiere ajustes extra. Se reporta la tabla tipo de ajuste (L1–L4 / falló) × orden por encoder. Si S9 cambia la región de un encoder con algún δ, ese encoder se califica **"sensible al tipo de ajuste"** (descriptivo: no cambia regiones ni conclusiones G). *Nota:* la S9 de REVISION3 (`(1 | person) + (1 | image)` en los 12) coincide con el primario cuando el peldaño común es L3; ya no se corre aparte.
- **Familia A.** Los coeficientes `condition:order3-1` de Mb por encoder, frente a F0 (`selected ~ condition*order + level + (1 | person) + (1 | image)`). Es descriptiva, sin margen fijo, conforme a la decisión de Patricio.

---

## 6. Geometría: una sola regla, medida sobre lo que entra en el modelo (B3)

### 6.1 La regla única (script `encoder_equivalence/scripts/02_equivalence.py`, parte B; sha256 `b815be4d…98ad8ab2` hasta el 18-11 10:39; desde la decisión de las 10:44 (tres cortes) y la parametrización para pruebas sintéticas, `f9d1de51…bf02c8`; se re-sella al congelar)
- **Cantidad.** Para cada encoder, zd sobre las 57.600 filas de los 8 ítems objetivo, que es exactamente lo que entra en el modelo co-primario. Se residualiza sobre `level` (parcializar nivel) y se calcula r de Pearson entre encoders sobre esas filas. El script **nunca lee** `selected`: usa `usecols` sin esa columna.
- **Agrupamiento (decisión de Patricio, 18-11 10:44).** UPGMA (vínculo promedio) sobre 1 − r, cortado a **tres alturas: 0,3, 0,5 y 0,7**. Se calculan y se reportan siempre las tres; ningún corte es preferido ni "por defecto", y no hay opción para elegir uno. Con UPGMA, dos subgrupos se unen solo si su r promedio entre miembros es ≥ 1 − corte (≥ .70, .50 y .30, respectivamente). Razón: los encoders son distintos y no se esperan equivalencias exactas, así que el agrupamiento se lee como espectro, igual que los δ. Entran los 10 encoders primarios (R0, O1, R1, E1, O2, E2, E3, E4, F1 con semilla 0 y F2).
- **Calibración pre-especificada, en cada corte por separado.** En un corte dado, el criterio *discrimina* solo si O1 queda en el grupo de R0 y si ni F1 ni F2 comparten grupo con algún encoder entrenado. Si la calibración falla en un corte, **ese corte queda "sin grupos"** (m2): no se hace ninguna interpretación por grupos con ese corte, y solo se usan las regiones por encoder.
- **Perfil del agrupamiento:** **"robusto"** si la partición de los 10 encoders y el resultado de la calibración son los mismos en los tres cortes; **"depende del corte"** en cualquier otro caso, indicando la partición y la calibración de cada corte.
- **El .90 queda solo como referencia.** Se marca qué encoders tienen r ≥ .90 con R0 ("prácticamente equivalentes a R0"), pero eso no decide nada.
- **Descriptivo adicional.** La misma r con 24 ítems, residualizada sobre `level × in_target`.
- **Momento.** `02` se ejecuta **después** del congelamiento (paso 2 del sábado). Antes de congelar no se ejecutó ni la parte A ni la parte B.
  - La versión anterior del script (`02_equivalence_SUPERSEDED_never_run.py`, con reglas de UPGMA sobre RSA de 45 imágenes) **nunca se ejecutó** y queda reemplazada.
  - La parte B se probó **solo con datos sintéticos** en `/tmp`: funciona y luego se borró.

### 6.2 Contexto descriptivo (parte A de `02`; no decide)
- **Qué calcula.** La matriz encoder × encoder de RSA completa (990 pares), RSA intra-dominio (315 pares) y CKA lineal, sobre las 45 imágenes de los 13 conjuntos de embeddings: R0, O1, R1, E1, O2, E2, E3, E4, E4s, F1 con las semillas 0, 1 y 2, y F2. F0 no tiene geometría y no aparece en la tabla.
- **Bootstrap.** B = 1.000 con `default_rng(42)`, remuestreando 15 imágenes por dominio, de forma pareada entre encoders.
- **Escala de distancias.** También calcula la razón entre y dentro de dominios, `distance_scale.csv`, que se usa con I3.
- **Salidas.** `out/equivalence_descriptive_long.csv`, `out/zd_corr_*.csv`, `out/equivalence_grouping.json` (con los tres cortes, la calibración por corte y el perfil) y `figs/encoder_equivalence_descriptive.png`.
- **Estado.** La tabla de equivalencias **no está calculada todavía**. Se calcula el sábado, después de congelar.
  - Patricio había autorizado calcularla ya (10:13), pero la revisión de Ariadna (B3) pidió correr el script solo tras congelar, para que la regla quede fijada sin haber visto la geometría.
  - Lo único que se conoce de geometría es la RSA de la prueba de humo del 18-11 a las ~10:00 (§8.3).
- **Uso.** La tabla **no se usa para elegir encoders**. El conjunto lo fijó Patricio (10:17) y no depende de la tabla.

---

## 7. Regiones de decisión: excluyentes, con precedencia y multiplicidad bien ubicada (B2)

### 7.1 Por encoder: regiones a tres márgenes (Δ de §5.2; δ ∈ {0,05; 0,10; 0,20} log-odds por DE)
- **Decisión de Patricio (18-11 10:37):** no se elige un margen único. Las regiones se evalúan y se reportan con **tres márgenes, δ = 0,05, 0,10 y 0,20**, como espectro. La conclusión se lee a lo largo del rango. Ningún δ es el preferido ni el "por defecto", tampoco 0,10, que era el valor único de la versión anterior.
- **Dónde se aplica δ.** Solo con 8 ítems (I3). Los tres δ los fijó Patricio y **no se anclan en el 0,2041 de la v1**: ese valor es `zd:order3-1` del Mb con 24 ítems (z sobre 24 ítems, GLMM agrupado), mientras que los δ se aplican al Δ en dos etapas con 8 ítems (z sobre 8 ítems, contrastes de suma, promedio de pendientes por experimento). No son la misma escala (I-b de REVISION2).
- **Referencia descriptiva (no confirmatoria):** junto a la tabla se reporta el Δ en dos etapas de R0 con 8 ítems, con su IC. No decide nada ni mueve los δ. La ventaja de casa de haber calibrado la tubería en DINOv2 queda declarada (I4).
- **Región previa, sin δ (BC1):** **No interpretable (control positivo).** Un encoder entrenado que no pasa el control de §5.3 queda en esta región con los tres δ y con 24 ítems, antes de mirar su Δ. No se aplica a F1 ni a F2 (para ellos el control no es puerta).
- **Regiones con 8 ítems, para cada δ por separado.** Se aplica **la primera que se cumpla**:
  1. **Signo opuesto:** el límite superior del IC 95 % es < 0. No depende de δ.
  2. **Menor que δ (equivalente con margen δ):** el IC 90 % está dentro de (−δ, +δ), que es la TOST. No significa "equivalente a 0": con δ = 0,20 puede caer aquí un efecto claramente positivo (I-a de REVISION2).
  3. **Se mantiene:** el límite inferior del IC 95 % es > 0 **y** Δ ≥ δ.
  4. **Atenuado:** el límite inferior del IC 95 % es > 0 **y** Δ < δ.
  5. **Inconcluso:** todo lo demás, incluido "inconcluso (técnico)", que no depende de δ.
- **Asimetría conocida y declarada de antemano.** Con un δ grande, la TOST es más fácil de pasar (región 2) y "se mantiene" es más difícil (exige Δ ≥ δ). Con un δ chico ocurre lo contrario. Por eso, que un encoder cambie de región a lo largo del espectro es esperable y no constituye una contradicción. Lo informativo es si la región es la misma en los tres márgenes.
- **Regiones con 24 ítems (sin δ; no cambian).**
  1. **Signo opuesto:** el límite superior es < 0.
  2. **Positivo:** el límite inferior es > 0. Se reporta junto con la razón entre y dentro de dominios.
  3. **Inconcluso:** lo demás.
- **Ejemplos.** Δ = 0,05 con IC [0,01; 0,09] (el caso de Ariadna) da "se mantiene" con δ = 0,05, porque Δ ≥ δ; "menor que δ" con δ = 0,10; y "menor que δ" con δ = 0,20. Su perfil es, por lo tanto, "dependiente del margen". Δ = 0,15 con IC [0,02; 0,28]: "se mantiene" con 0,05 y 0,10; con 0,20, "atenuado" si el IC 90 % no cae dentro de ±0,20.

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
   - No se puede decir "se mantiene" (ni "menor que δ") sin calificativo, salvo que el perfil sea robusto.
   - No se puede elegir δ después de ver los datos.
   - Si en algún momento se agregara otro δ, sería una desviación registrada y quedaría fuera de la lectura confirmatoria.
5. **Cortes de geometría (10:44).** Todo lo que usa grupos (G2 con la descripción de grupos, G3 frente a G3b) se reporta en una **grilla δ × corte** (3 × 3 = 9 celdas), siempre completa. Ningún corte se reporta sin los otros dos. La regla de perfil global está en §7.4.
6. Lo que **no depende de δ** (no interpretable por el control positivo, signo opuesto, inconcluso técnico, las regiones con 24 ítems, el control positivo y los pisos con la región 1) se informa una sola vez.

### 7.3 Multiplicidad
- **Conjunción** ("se mantiene en todos"): es una prueba intersección-unión. Cada encoder se evalúa a su propio 95 %, **sin Holm**, con cada δ.
- **Unión** ("depende del encoder"), **con Holm aplicado por separado con cada δ**:
  - **Familia de entrenados:** los k encoders entrenados distintos de R0 (de O1, R1, E1, O2, E2, E3, E4; k ≤ 7) que **pasaron el control positivo** y tienen Δ (no "inconcluso (técnico)"), cada uno con 2 pruebas: p_TOST(δ) (el mayor de los dos p unilaterales a ±δ) y p de signo opuesto (unilateral, el mismo con todos los δ). Son **2·k pruebas por δ** (14 si los 7 pasan), con **Holm** en conjunto. Los encoders en "no interpretable (control positivo)" o en "inconcluso (técnico)" no entran (BC1).
  - **Familia de pisos (M4):** F1 y F2, con las mismas 2 pruebas cada uno. Son 4 pruebas por δ (menos si uno es "inconcluso (técnico)"), con Holm aparte. El control positivo no los saca de la familia.
- **No se corrige entre δ.** Los tres márgenes no son tres afirmaciones independientes que se puedan elegir. Son un mismo análisis leído a tres resoluciones, y la regla de §7.2 exige mostrar los tres. La protección frente a elegir el margen viene de esa regla, no de un ajuste.
  - **Consecuencia:** una afirmación "robusta" es más exigente que cualquier δ aislado.
- **Tamaño de cada familia en la salida (REVISION3):** la tabla de regiones indica, por encoder, a qué familia pertenece o por qué queda excluido, y el informe da k y la lista de excluidos de cada familia.
- **Sensibilidades y descriptivos:** no entran en ninguna familia.

### 7.4 Conclusión global (8 ítems; se evalúa con cada δ y se aplica la primera que se cumpla)
| # | Condición (con un δ dado) | Lectura permitida |
|---|---|---|
| G0 | Falla el control de tubería, **o R0 falla el control positivo** (BC1), **o R0 queda "inconcluso (técnico)" con 8 ítems** (REVISION3b); no depende de δ ni del corte | Se detiene todo; no hay lectura. Con R0 técnico el texto es **"sin lectura: técnico"**: una falla técnica nunca se lee como G1 |
| G1 | R0 tiene un resultado (no técnico) y no está en la región "se mantiene" con ese δ | **La moderación de la v1 no se sostiene, con ese margen, con inferencia a nivel experimento.** Con ese δ, la comparación entre encoders pasa a ser descriptiva |
| G2 | ≥ 1 encoder entrenado distinto de R0 (que pasó el control positivo) cae en la región 1 o 2 y el p de la prueba que define esa región (signo opuesto para la 1, p_TOST(δ) para la 2) es < .05 con Holm en la familia de entrenados con ese δ | **Depende del encoder** con ese margen. Si la calibración discrimina, se describe en qué grupos de geometría están los encoders que fallan; si falla, **sin grupos** (m2) |
| G3 | **Los 8 entrenados pasaron el control positivo** y están en "se mantiene", la calibración discrimina y los entrenados ocupan ≥ 2 grupos de geometría | **No es específico de DINOv2 (a través de geometrías)** con ese margen. Con 24 ítems se informa si también son todos "positivos"; si no, la lectura es "solo con 8 ítems" |
| G3b | **Los 8 entrenados pasaron el control positivo** y están en "se mantiene", pero la calibración falla o hay un solo grupo | Se mantiene en todos los encoders con ese margen, **sin evidencia de que difieran en la geometría que entra en el modelo** |
| G4 | Cualquier otro caso (lo más probable), incluido que algún entrenado esté en "no interpretable (control positivo)" sin que se cumpla G1 o G2 | **Inconcluso o mixto** con ese margen, reportado encoder por encoder |

- **R0 con 24 ítems (REVISION3b).** No hay grilla G con 24 ítems (§7.1). Si R0 queda "inconcluso (técnico)" con 24 ítems, su lectura de referencia con 24 ítems es "sin lectura: técnico" (no "inconcluso" ni un resultado), y la frase de G3 sobre 24 ítems no se puede afirmar.
- **Grilla δ × corte (10:44).** G0, G1 y G4 por control positivo no dependen del corte. G2 tampoco, salvo la descripción de grupos, que es "sin grupos" en un corte cuya calibración falla. G3 frente a G3b sí depende del corte (calibración y número de grupos de ese corte). La conclusión se calcula en las 9 celdas.
- **Regla de perfil global (declarada de antemano; ninguna combinación queda ambigua):**
  1. **"Robusta: Gk"** si la conclusión es la misma en las 9 celdas.
  2. **"Depende solo del margen"** si, dentro de cada δ, los tres cortes coinciden, pero los δ difieren; se escribe la conclusión de cada δ.
  3. **"Depende solo del corte"** si, dentro de cada corte, los tres δ coinciden, pero los cortes difieren; se escribe la conclusión de cada corte.
  4. **"Depende del margen y del corte"** en cualquier otro caso; se escriben las 9 celdas.
- **Resultado global:** el patrón a lo largo de los tres δ (§7.2.3) y de los tres cortes, con la regla anterior. Por ejemplo, "robusta: G4" o "G3 con δ = 0,05 y 0,10; G1 con δ = 0,20 (dependiente del margen)".
  - G1 también depende de δ: con δ = 0,20, R0 necesita Δ ≥ 0,20 para quedar en "se mantiene". Cuánto es eso respecto de la v1 no se afirma, porque el 0,2041 está en otra escala (I-b); la referencia es el Δ en dos etapas de R0, descriptivo.
- **Pisos (se leen aparte, con cada δ y con la misma regla de perfil):**
  - si F1 y F2 están en "se mantiene", la moderación no requiere una geometría aprendida;
  - si ambos caen en la región 1 o 2 con Holm de pisos, la moderación requiere alguna geometría aprendida;
  - si no, es inconcluso.
- **S1–S9 y la familia A** no cambian la región. Si contradicen la conclusión, se informa.
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
11. **Convergencia en la etapa 1** (experimentos de 50 personas): una escalera fija y una estructura común elegida solo por convergencia (§5.2); no se imputa nada. El ajuste mixto queda como S9, descriptivo.
12. **Licencias:** MAE se trata como no comercial. No se redistribuyen pesos ni derivados por ensayo de XTMEM (§36).
13. **RAM (M6):** al revisar a las ~10:00 había 2,7 GB disponibles. Durante los embeddings (10:15–10:19) el log muestra entre 6,5 y 12,1 GB. En la ejecución se usan como máximo 3 procesos de R y se vigila `MemAvailable`.
14. **Ventaja de casa (I4):** la tubería se calibró con DINOv2 (los δ los fijó Patricio y no se refieren al 0,2041 de la v1, que está en otra escala; I-b).
15. **Bifurcaciones:** todo lo que decide está en §5.2, §6.1 y §7. Cualquier otro análisis es descriptivo y se rotula así.
16. **Expectativa con el corte 0,7 (REVISION3, menor):** con 0,7 se unen grupos con r promedio ≥ .30, así que es probable que algún piso quede con un entrenado. Entonces la calibración falla en ese corte y un "se mantiene en todos" saldría como "depende solo del corte" casi por construcción. Se declara de antemano para no leerlo como hallazgo.

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
| **Dos etapas, 8 ítems** (10 fuentes × 12 experimentos; escalera, §5.2) | 120 si todo converge en L1; peor caso ~600 | 1–2 h; peor caso ~5–10 h |
| **Dos etapas, 24 ítems** (10 × 12; escalera) | 120; peor caso ~600 | 2–6 h; peor caso ~10–30 h |
| Control positivo, intra-dominio (9 fuentes nuevas, ~10 min) | 9 | ~1,5 h |
| S1: Mb y Mc con 24 ítems (9 × ~36 min) y con 8 ítems (10 × ~12 min) | 19 | ~7,5 h |
| S2: `(1 | exp) + (0 + zd | exp)`, 8 ítems (10 × ~15 min) | 10 | ~2,5 h |
| S3: `level:order`, 8 ítems (10 × ~12 min) | 10 | ~2 h |
| S4–S8: dos etapas con 8 ítems (4 × 10 fuentes y 3 fuentes extra) × 12 | ~520 | 4–8 h |
| S9: ajuste mixto (sale de la pasada 1 del primario) | 0 | segundos |
| Familia A: F0 | 2 | ~0,3 h |
| **Total** | | **~22–31 h de CPU**; con 3 procesos de R en paralelo, **~8–11 h de reloj** |

- **Impacto de los tres δ (10:37):** ninguno en los ajustes. Son los mismos 120 + 120 ajustes de la etapa 1; solo cambian la etapa 2 (pruebas de permutación desplazadas a ±δ, segundos) y `08_regions.py`.
- **Escalera (12:12/12:14):** si todo converge en L1, el costo es el de antes. El peor caso (cuatro peldaños por experimento más la pasada 2) multiplica la etapa 1 por ~5; con 24 ítems eso solo ya pasaría el tope. El primario no se corta: si el tope se supera por la etapa 1, se cortan las sensibilidades y se registra como desviación. S9 ya no cuesta ajustes.
- **Tope de presupuesto:** 32 h de CPU. Si se supera, las sensibilidades se cortan en orden inverso de ejecución (S6 → S8 → … → S1), y queda registrado. No hay re-corridas selectivas. `run_weekend.sh` no aplica el tope solo: se vigila a mano con las horas del log (REVISION3, menor).
- **Costo monetario:** ninguno.
- **Tiempo humano:** escribir y probar los scripts antes del congelamiento, ~3–4 h. Revisión breve de Ariadna de los cambios, ~1 h. Supervisión el fin de semana, ~1 jornada. Resultados y desviaciones, ~½ jornada.

### 9.3 Qué se ejecutó antes del congelamiento (I5)
- **18-11 ~10:00:** prueba de tiempos con DINOv2-S y una ResNet-50 aleatoria sobre las 45 imágenes. Como dato colateral salió una RSA completa de ρ = .158, con una sola semilla y sin IC. No decide nada.
- **18-11 10:15–10:19:** `01_embed.py` generó los embeddings de las 13 fuentes. E4s falló el chequeo de determinismo (~10:16–10:17) y se recalculó a las 10:18, junto con E4, tras el arreglo M1.
- **18-11 ~10:24:** `00_vision_params_and_checks.py` contó los parámetros de visión y comprobó la tubería de R0 contra la v1 (§3.2).
- **18-11 12:05–12:08 (REVISION3, punto 6):** se volvió a correr `01_embed.py` con las 13 fuentes (solo las 45 imágenes). Los 13 npz salen idénticos byte a byte a los anteriores (diferencia máxima 0), incluidos E4 y E4s con `noise` fijo. Copias previas en `out/npz_before_rerun/`.
- **18-11 12:06 (REVISION3, I2):** prueba de humo de `01b_embed_common.py` en `/tmp/s6smoke` con R0, F1 y F2 (solo imágenes): R0 da coseno mínimo 0,9999953 por imagen frente a su npz actual; F1 y F2 salen idénticos (diferencia 0).
- **Ninguno de estos pasos usó** `h4_long.csv`, `all_raw_A.csv` ni respuestas de participantes. No se calculó ninguna d de ensayo ni ninguna RSA o CKA entre los encoders nuevos.

---

## 10. Plan para congelar

### 10.1 Contenido que se congela
- **Encoders (10 primarios):** R0, O1, R1, E1, O2, E2, E3, E4, F1 (semilla 0) y F2. F0 se suma para la familia A. Sensibilidades: E4s y F1 con las semillas 1 y 2.
- **Distancias:** §5.1. Normalización: §4.
- **Familia B, primario:** las dos etapas a nivel experimento (§5.2), con 8 y 24 ítems co-primarios (decisión de Patricio a las 10:17). Regiones con δ = 0,05, 0,10 y 0,20 como espectro, solo con 8 ítems (decisión de Patricio a las 10:37; §7.1–§7.2).
- **Control positivo:** §5.3, obligatorio por encoder (aprobado a las 10:38). Sensibilidades: §5.4.
- **Cortes del agrupamiento de geometría:** 0,3, 0,5 y 0,7 como rango (decisión de Patricio, 10:44; §6.1, §7.4).
- **Geometría:** una sola regla (§6.1, `02_equivalence.py`).
- **Regiones a tres márgenes, lectura del espectro, multiplicidad y conclusión global:** §7.
- **Presupuesto y parada:** §9.2.

### 10.2 Lista antes del congelamiento (jueves 19 a viernes 20)
1. Aprobación de Patricio: hecha (10:17, 10:37, 10:38, 10:44, 12:12 y 12:14; ver §10.4). No queda ningún punto pendiente de Patricio.
2. Escribir los scripts que faltan: `h4_prep_encoders.py`, `01b_embed_common.py` (S6), `lib_glmm.R`, `05_twostage_stage1.R`, `05b_select_structure.py`, `06_stage2_permutation.py`, `07_pooled_glmm.R` (control positivo, S1–S3 y F0), `08_regions.py`, `09_freeze.py`, `run_weekend.sh` y `README_weekend.md`, con las pruebas en `scripts/tests/`. Hecho el 18-11 (ver registro de cambios).
   - `08_regions.py` implementa §7 literalmente, con los tres δ y la tabla única de §7.2.
   - Se prueban **solo con datos sintéticos**.
3. Segunda revisión de Ariadna: el plan final (hecha, REVISION2; respondida, §11.2) y los scripts (pendiente). Si encuentra algo de fondo, se avisa a Evaristo antes de correr.
4. Congelamiento interno, **no antes del viernes 20-11** (indicación del 18-11 10:44; `09_freeze.py` lo impide antes de esa fecha):
   - `FREEZE_encoders_v2.txt` con la hora (UTC−3);
   - el sha256 de este archivo (sin el campo de hash);
   - el sha256 de todos los scripts y de `encoder_specs.json` (lo escribe `09_freeze.py`, que no se ejecuta hasta que Ariadna apruebe los scripts).
   - No es un prerregistro externo.

### 10.3 Ejecución (sábado 21 y domingo 22; solo tras el paso 4)
**Sábado:**
- 0. Verificar los hashes.
- 1. `h4_prep_encoders.py` con todas las fuentes, más el control de tubería (G0).
- 2. `02_equivalence.py`: tabla descriptiva, regla de geometría y calibración.
- 3. Cronometrar un ajuste de la etapa 1 y actualizar el presupuesto (sin cambiar el análisis).
- 4. Etapa 1 con 8 ítems, pasada 1 de la escalera, todas las fuentes.
- 5. Etapa 1 con 24 ítems, pasada 1.
- 5b. `05b_select_structure.py`: peldaño común por conjunto de ítems, solo por convergencia, con las pasadas 2 que pida (§5.2).
- 6. Control positivo (`07_pooled_glmm.R poscontrol`).
- 7. Etapa 2 y `08_regions.py`.
- *Aclaración al escribir los scripts:* el control positivo va antes de `08_regions.py`, porque las regiones lo usan (BC1). El orden anterior (6 = etapa 2 y regiones, 7 = control) no podía ejecutarse así. No cambia ningún análisis.

**Domingo:**
- 8. S1 → S5, S7 y S8 en ese orden, dentro del presupuesto (S4, S5, S7 y S8 con el peldaño común de 8 ítems), y `08_regions.py` otra vez con S9 (ajuste mixto, ya calculado el sábado; rótulo descriptivo).
- 9. Familia A.
- 9b. **S6 al final** (REVISION3, I2): si falla, no detiene nada y se registra como desviación.
- 10. `RESULTADOS_encoders.md`, nuevas entradas de `DESVIACIONES` y aviso a Evaristo.
- No se publica nada sin Patricio.

### 10.4 Decisiones de Patricio
**Ya decididas:**
- **10:17:** entran todos los encoders, incluidos O1 y O2; la tabla de equivalencias reemplaza a los márgenes de geometría; los 8 ítems son co-primarios; se autoriza el cómputo, a ejecutar el sábado 21 o el domingo 22; Ariadna revisa antes de congelar.
- **10:37:** **δ.** No hay margen único: las regiones se reportan con δ = 0,05, 0,10 y 0,20 como espectro (§7.1–§7.2).
- **10:38:** se aprueba **B1**, el análisis en dos etapas con permutación exacta como inferencia que decide, con el GLMM publicado como descriptivo (§5.2).
- **10:38:** se aprueba el **control positivo intra-dominio obligatorio por encoder** (§5.3).
- **10:38:** se aprueba que **la tabla de equivalencias se calcule después de congelar** (§6, B3).

- **10:44:** el **corte del agrupamiento de geometría es un rango (0,3, 0,5 y 0,7)**, con la misma lógica de espectro que los δ y la calibración por corte (§6.1, §7.4). El congelamiento no ocurre antes del viernes 20-11.

- **12:12 (REVISION3, I1, opción B):** el primario de la etapa 1 usa el mismo modelo en los 12 experimentos, la estructura más rica que converja en los 12; el ajuste mixto con fallback pasa a S9, secundario y solo para reportar (§5.2, §5.4).
- **12:14:** **una sola estructura común para todos los encoders**, elegida solo por convergencia (`STRUCTURE_SCOPE = "common"`: **confirmado por Patricio 18-11 12:14**); la variante por encoder queda como descriptiva (§5.2).

No queda ningún punto pendiente de Patricio.

**Siguiente paso antes de congelar:** Ariadna ya hizo la segunda revisión del plan (REVISION2, respondida en §11.2). Revisó los scripts y el plan en REVISION3 (respondida en §11.4). Falta que revise solo el diff `encoder_equivalence/DIFF_REVISION3.patch` (REVISION3 y decisiones de Patricio de las 12:12 y 12:14) y dé el visto bueno para `09_freeze.py`.
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

### 11.2 Respuesta a la segunda revisión (REVISION2)
- **BC1:** §5.3 (consecuencias), §7.1 (región previa "no interpretable (control positivo)"), §7.3 (familia de 2·k pruebas) y §7.4 (G0 con R0, G3/G3b solo si todos pasan, G4). Implementado en `08_regions.py`, con prueba unitaria.
- **I-a:** §7.1, región 2 "menor que δ (equivalente con margen δ)"; §7.2.
- **I-b:** §7.1 (sin ancla en 0,2041; Δ de R0 en dos etapas como referencia descriptiva), §7.4 (frase de G1) y §8.14. Los δ no cambian.
- **m1:** registro de cambios. **m2:** G2 en §7.4. **m3:** §5.2.
- **Lista de scripts de REVISION2:** respondida punto por punto en `encoder_equivalence/scripts/README_weekend.md`.

### 11.3 Decisión de Patricio sobre los cortes (18-11 10:44)
- §6.1 (tres cortes, calibración por corte, perfil del agrupamiento), §7.2.5 y §7.4 (grilla δ × corte y regla de perfil global), §10.1, §10.2, §10.4.
- Implementado en `02_equivalence.py` (`CUTS = (0.3, 0.5, 0.7)`, función `group_all`) y `08_regions.py` (`grid_profile`), con pruebas unitarias donde la calibración pasa en un corte y falla en otro.

### 11.4 Respuesta a la tercera revisión (REVISION3)
- **BC3:** §3.2 y §5.1 (casi-empates < 1e−6 agrupados; desviación justificada frente al redondeo a 6 decimales); §1.3 (≈290 valores a 1e−6). Implementado en `h4_prep_encoders.py::rank_order_ok`, con pruebas en `tests/test_prep.py`.
- **I1:** §5.4 (S9), §9.2 (costo), §10.3 (domingo), §10.4. Implementado en `06_stage2_permutation.py` (tipo de ajuste × orden) y `08_regions.py` (`--stage2-s9`, rótulo "sensible al tipo de ajuste").
- **I1 tras las decisiones de Patricio (12:12 y 12:14) y las precisiones (a) y (b) de Ariadna:** §5.2 (escalera L1–L4, criterio de convergencia con tolerancias, regla común, pisos en la elección, encoder sin peldaño), §5.4 (S4–S8 con el peldaño común; S9 = ajuste mixto), §8.11, §9.2 (costo y peor caso), §10.2, §10.3 (paso 5b), §10.4. Implementado en `lib_glmm.R` (`LADDER`, `LADDER_CTL`, `ladder_converged`), `05_twostage_stage1.R` (argumento `fit` = `ladder` o un peldaño), `05b_select_structure.py` (nuevo; `STRUCTURE_SCOPE = "common"`), `06` (filas sin convergencia = pendiente faltante) y `run_weekend.sh` (`select_structure`), con pruebas en `tests/test_select_structure.py`.
- **I2:** §9.3 (prueba de humo) y §10.3 (S6 al final, sin bloquear).
- **(2)/(3):** §7.3 y columna `familia_holm` más la sección "Familias Holm" de `regions_report.md`.
- **(6):** §9.3 y `out/rules_02_sha256_before_run.txt`.
- **Menores:** §8.16, §9.2 y G0 en JSON.

### 11.5 Respuesta a la revisión del diff (REVISION3b)
- **IMPORTANTE (R0 técnico):** §7.4 (G0 y G1; R0 con 24 ítems). Implementado en `08_regions.py::conclude` (`TECH0`) y `r0_reading24`, con dos pruebas en `tests/test_regions.py`.
- **Menor (ancho de los casi-empates):** §5.1. Implementado en `h4_prep_encoders.py::near_tie_max_width` (G0 falla si el grupo más ancho mide ≥ 1e−6; campo en `pipeline_control.json`), con prueba en `tests/test_prep.py`.
- **Menor (`rung8`):** §5.2. Implementado en `05b_select_structure.py` (código 4), `run_weekend.sh` (`select_structure`, `rung8`, S4–S8 y S6) y `README_weekend.md`, con tres pruebas en `tests/test_select_structure.py`.
