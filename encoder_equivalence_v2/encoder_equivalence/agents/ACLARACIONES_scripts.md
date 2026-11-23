# Aclaraciones de los scripts del anexo de agentes (v2 de arXiv 2611.14116)

Propuesta: `propuesta_agentes_multi_encoder_v2_2026-11-18.md` (sha256 `633a5e7e…0db25`, con REVISION4 incorporada).
Scripts: `encoder_equivalence/agents/scripts/`. Pruebas, solo con datos sintéticos: `encoder_equivalence/agents/tests/`.
Redactado el 22-11-2026 (hora de Santiago). Actualizado el 22-11 con REVISION5 de Ariadna (VB con condiciones; ver la sección al final). Con datos reales solo se hizo lo aprobado: la reconstrucción del caché R0 de la v1 (b1, a13) y su chequeo de distancias de centroides. No se codificaron embeddings primarios de agentes ni se calculó ninguna exactitud, distancia o correlación de agentes.

- **Sección (a):** aclaraciones que no cambian ninguna decisión de la propuesta.
- **Sección (b):** puntos que van más allá de la propuesta. Requieren decisión de Patricio y revisión de Ariadna **antes** del congelado (`15_freeze_agents.py --freeze`).

En cada punto de (b) los scripts ya traen una opción por defecto, marcada **[por defecto]**. Está en una constante con nombre, así que cambiarla es una línea.

---

## (a) Aclaraciones que no cambian decisiones

1. **d_sem de cada análisis.** D y D-todos usan el d_sem de exp_008, igual que la regla A de la v1. H2 usa el d_sem corregido de `pairs_c100.csv`, igual que `02_h1_h2.py` de la parte humana. Cada análisis conserva el d_sem que usó en la v1.
2. **Cachés que ya existían.** Se reutilizan sin volver a codificar (`lib_agents.EXISTING`):
   - R0: pozo CIFAR-10 de la v1 y test CIFAR-10 (`hda_cache/dino_c10_test.npz`);
   - R1: CIFAR-100 train y pozo (`hda_cache/cvcl_*`).
   Ver (b1) por el caché de R0 que se perdió.
3. **Pozo CIFAR-10 (2.000 imágenes).** Se reconstruye con la regla de exp_007 (`rng(42)`, 200 por clase, en el orden de `DATASET_CATEGORIES`). En el run real, `10_agent_embed.py` comprueba que el orden de hashes y categorías sea idéntico al del caché de la v1. La prueba sintética compara además, píxel a píxel, con una reimplementación independiente.
4. **IC de C1 en T.** Usa `bootstrap_ci(n_resamples=2000, seed=42)` sobre episodios, igual que `analyze_clean.py` de la v1. La puerta del control positivo de agentes aplica este IC a la C1 nativa del prototipo (límite inferior > 1/3).
5. **E4s (CLS de MAE).** Sale de la misma pasada que E4 (media de parches) y cuesta 0 imágenes extra.
6. **Control positivo no evaluable.** Si en un encoder entrenado T falta o es técnico, todas sus pruebas quedan en "inconcluso (técnico)". Así no se lee nada de un encoder cuya puerta no se pudo evaluar.
7. **Orden de ejecución.** La dispersión de d_perc en los 100 pares (§3.6) se calcula antes de cualquier exactitud (paso 4 de `run_agents.sh`). Si falla el control 1, el run se detiene antes de calcular nada con otros encoders.
8. **Hilos.**
   - Embeddings: 2 hilos de torch por proceso, 3 procesos.
   - Simulaciones: 1 hilo (OMP/BLAS = 1), 3 procesos, como r04 y el T de la v1.
   - H4 en R: BLAS con 2 hilos.
9. **Lector de embeddings de T.** Lee la matriz una sola vez y devuelve filas como vistas, igual que la rama de caché de `exp_007/_shared.py`. La primera versión copiaba el lector de `h6_tiers.py`, que relee el npz en cada fila. Eso daba ~6 GB de RSS con 384 dimensiones, y con F2 (3.072 dimensiones) habría pasado de 40 GB. La prueba e2e lo detectó: un proceso murió por falta de memoria. Los resultados de T son idénticos antes y después del arreglo (comprobado en sintético).
10. **Sensibilidad T-sem "full" de R0.** Los cachés r04 de R0 (`revision_runs/out/r04_seeds/cache_seed_{0..3}`) se perdieron en el reinicio del box del 22-11. R0 se codifica ahora con `10_agent_embed.py`, igual que los demás encoders: 4 × 2.000 imágenes, solo en `sensitivities`, que es opcional.
11. **Restauración tras el reinicio del 22-11 (~12:34–14:31).**
    - `encoder_equivalence/out/` (34 archivos de la parte humana) se restauró desde la rama `encoder-equivalence-v2` de GitHub, solo lectura. Los 16 archivos de `out/` que están en el manifiesto humano coinciden byte a byte en sha256, y `09_freeze.py --verify` da OK con los 47 archivos.
    - `human_data_analyses/out/h4_person_re.txt` se restauró de la misma rama. No hay sha previo de referencia, pero es idéntico en las ramas `main`, `encoder-equivalence-v2` y `agents-xtmem-v2` (sha256 `8e6b93d8…f6e3a`).
    - El entorno R se reconstruyó en `/tmp/renv` con micromamba (conda-forge): R 4.5.3 + lme4 2.0.6, las mismas versiones del plan, con Matrix 1.7.5. No sé qué versión de Matrix tenía el entorno anterior.
12. **Figura x1 (`14b_fig_x1.py`).** Es una aproximación descriptiva simple y no entra en ninguna regla.

13. **Recuperación del 22-11, 15:20–15:45 (hora de Santiago), aprobada por Patricio (15:16: "avance con la recuperacion de los pesos de los modelos incluidos el de CIFAR-10").**
    - **Pesos de Hugging Face.** Se descargaron los 8 checkpoints en las revisiones fijadas en `01_embed.py` y `encoder_specs.json` de la parte humana: dinov2-small `ed25f3a3`, CVCL `f50eaa0c`, siglip `7fd15f06`, vit-base `3f49326e`, resnet-50 `34c2154c`, vit-mae-base `25b184be`, dinov2-base `f9e44c81`, CLIP-B/32 `1a25a446`.
      - Las 8 revisiones resuelven al commit fijado, y hoy coinciden con `main`. Ningún modelo cambió ni desapareció. Tomó 50 s.
      - CVCL se carga sin revisión explícita (`h6_embed_cvcl.py`), pero su `main` actual es `f50eaa0c`, el mismo registrado en la parte humana.
      - Informe: `/workspace/recovery_22nov/weights_report.json`.
    - **CIFAR-10 crudo.** Se descargó del espejo de Hugging Face que indica `r03_build_cache.py` (`MIT-OL-AI-D/cifar-10-python`). El MD5 es `c58f30108f718f92721af3b95e74349a`, igual al que comprueba torchvision. Se extrajo en `revision_runs/repo/.cache/cifar10/`, y la comprobación de integridad de torchvision da OK con 50.000 imágenes.
    - **Estímulos de Lewis & Frank.** Están en `/tmp/XTMEM`: `git clone` de `mllewis/XTMEM` en el commit `220ed17e…` de `download_data.sh`. Son 45 estímulos `c*.jpg`. Están en `/tmp`, así que pueden volver a perderse.
    - **Caché de CIFAR-10 de R0 de la v1** (`revision_runs/repo/.cache/embeddings_exp007/cifar_all10.npz`). Se reconstruyó con `r03_build_cache.py` sin cambios.
      - Tardó 21 min 24 s de reloj y ~111 min de CPU. Codifica imagen por imagen con 8 hilos, y el box estaba compartido con otro ensayo sintético.
      - Nuevo sha256: `9744d7cfd6ed2c284ec4797c5d27d87bc512e4a1bba5d25d0b8994b14668024d`.
      - **No existe ningún sha ni registro previo** de ese archivo (busqué en manifiestos, logs, repositorios y resultados).
      - Comparación con métricas publicadas de la v1 que dependen de él, solo distancias entre centroides de categoría, sin exactitudes:
        - las 45 distancias de `revision_analyses/out/r05_tier_pairs.csv` (GitHub, `main`) coinciden en los 4 decimales publicados (máx. |Δ| sin redondear = 5,0e-5, por debajo del redondeo);
        - las 17 de `results/exp_007b_extension/concept_gate.csv` (valores originales de la v1) también, con máx. |Δ| = 5,0e-5.
      - El orden de hashes y categorías es el de la regla de exp_007.
      - No se comparó el ledger de T: eso es el control 1 mismo, y queda para el run.
    - **Cachés r04 de las semillas.** **No se reconstruyeron.**
      - `r04_run_clean_seed.py <s> full` sin cambios no solo codifica el pozo: corre además T, es decir, exactitudes de aprendices con R0 real. Eso queda fuera de lo autorizado antes del congelado.
      - Además ya no hacen falta: desde (a10), `10_agent_embed.py` codifica los pozos de R0 en la sensibilidad T-sem "full", y el manifiesto ya no los pide.
    - **Paridad con los pesos reales** (`test_embed_agents.py`, imágenes de ruido, 45 "estímulos" sintéticos 720×720): una copia de `01_embed.py` (solo cambian las rutas) y `10_agent_embed.py` dan **|Δ| = 0 exacto** en los 13 vectores: R0, O1, R1, E1, O2, E2, E3, E4, E4s, F1 y F2.
      - No se corrió la paridad sobre los estímulos L&F reales. Es el paso 1 de `run_agents.sh`.
    - **Verificaciones finales.**
      - `09_freeze.py --verify` humano: OK con 47 archivos.
      - `15_freeze_agents.py --list`: 97 archivos, 0 faltantes.
      - `FREEZE_agentes_v2.txt` no existe.

---

## (b) Puntos que van más allá de la propuesta (decisión de Patricio y Ariadna)

### Bloqueantes para el run real (no para la revisión)

- **b1. Se perdió el caché de CIFAR-10 de R0 de la v1** (`revision_runs/repo/.cache/embeddings_exp007/cifar_all10.npz`).
  - Se borró en el reinicio y no está en GitHub ni en Hugging Face. El control 1 (§3.5) se apoya en él, y el manifiesto del congelado lo exige.
  - Ese caché ya era una reconstrucción hecha en el box con `revision_runs/scripts/r03_build_cache.py`, en la revisión de la v1.
  - **Propuesta:**
    1. Antes del congelado, reconstruirlo con `r03_build_cache.py`, sin cambios. Eso descarga CIFAR-10 y codifica 2.000 imágenes con DINOv2-S en CPU, unos 2–5 min.
    2. Registrar su sha256 en el manifiesto.
    3. Mantener el criterio de control 1 de b2.
  - **Es codificar CIFAR, así que requiere la aprobación explícita de Patricio.**
  - **Resuelto el 22-11 (15:16, aprobación de Patricio), ver (a13):** caché reconstruido, sha256 `9744d7cf…68024d`, coincide con las distancias publicadas de la v1 en 4 decimales; pesos, CIFAR-10 y XTMEM recuperados. Ariadna verificó a precisión completa (REVISION5): máx. |Δ| 8,9e-8 en los 45 pares frente a `pairs_c10.csv` y 5,6e-8 en los 10 niveles frente a `tiers_c10.csv`. Queda registrado en el encabezado del manifiesto (C1). Falta que Patricio o Evaristo confirmen la aprobación en el registro.
  - También hay que volver a descargar, todo público:
    - los pesos de Hugging Face de los 8 checkpoints (~3 GB; el caché de HF se perdió);
    - CIFAR-10 crudo;
    - `/tmp/XTMEM`, con los estímulos de Lewis & Frank para la paridad, mediante `human_data_analyses/data/download_data.sh`.
- **b2. Criterio del control 1 (T con R0 frente al ledger de la v1).** La lectura literal, "reproduce exactamente", **no se cumple en el box**: el re-run r04 de la semilla 42 ya difería del ledger de la v1.
  - El aprendiz `random` usa ruido Gumbel con semilla tomada de un hash de los **bytes** del embedding, y los márgenes difieren en ~1e-6, que es el redondeo a 6 decimales.
  - **[por defecto] `propuesto_b1`:**
    - las mismas 5.040 filas y claves;
    - para los 5 aprendices no aleatorios, `c1`, `c2` y `per_concept_c1` idénticos y |Δ margin| ≤ 1e-6;
    - `random` se reporta pero no se exige. **REVISION5:** se reporta como distribución: la C1 media (y su DE) por nivel, frente a la v1 y frente a 1/3 (`random_c1_by_tier` en `control_T_exact.json`). La semilla de `random` no se fija, porque eso tocaría código de la v1.
  - La alternativa `estricto` queda disponible (`AGENTS_CONTROL_T_RULE=estricto`). Con un caché reconstruido (b1), es muy probable que `estricto` falle y deje todo en GA0.

### Decisiones de lectura (cambian qué se concluye)

- **b3. Cuadrantes de D.** La propuesta dice "mediana del encoder sobre los 100 pares".
  - **[por defecto, cambiado por REVISION5]** El lado d_perc usa la mediana del encoder sobre los **4.950 pares**, como en la v1. El lado d_sem conserva la mediana de la v1 sobre los 4.950 pares. Así R0 reproduce la letra de regla de la v1.
  - La versión con la mediana de los 100 pares queda descriptiva: `rule_centroid_q100_descriptiva` y `quadrant_counts_100_descriptiva`. Solo afecta `rule_centroid` y los conteos por cuadrante.
- **b4. Un ρ significativo pero diminuto cuenta como Apoyo en H1.** Con la precedencia literal (Apoyo > No apoyo), un ρ > 0 con Mantel p < .05 es Apoyo aunque su IC quede entero bajo el SESOI.
  - **[por defecto] literal.**
  - **REVISION5:** se agrega el rótulo descriptivo **"Apoyo con IC bajo SESOI"** (columna `rotulo_<SESOI>` de `regions_agents.csv`). En A se dispara si ci_hi < SESOI; en P y D, si el límite inferior del IC de r parcial es > −SESOI. No cambia la región ni GA.
  - Alternativa no adoptada: exigir además que ci_hi ≥ SESOI.
- **b5. Qué p entra al Holm de GA1 ("No apoyo").** La propuesta fija la familia pero no el p.
  - **[por defecto]**
    - H1: p bootstrap unilateral de H0 "ρ ≥ SESOI", con las mismas réplicas del IC (10.000, semilla 42).
    - P y D: el menor entre (i) p unilateral Fisher z de "r parcial ≤ −SESOI" y (ii) el p del β_perc si β > 0 (efecto opuesto).
    - H4: p bilateral si logOR > 0, y 1 en otro caso.
- **b6. p de β_perc en D y D-todos.** **[por defecto] `D_P="v1"`:** OLS clásico con d sin estandarizar, el `fit_models` de la v1 (regla A). La alternativa `hc3` usa `stats_lib.ols_hc3`. Ambos se calculan y guardan siempre. **REVISION5:** se reporta si HC3 cambia la región (columnas `region_hc3_<SESOI>` y `hc3_cambia_region`, solo D).
- **b7. Región "No apoyo" en H4**, que no tiene SESOI. **[por defecto]** No apoyo = signo opuesto (logOR > 0) con p < .05.
- **b8. Agregar las 3 fuentes de H1 (THINGS, HSJ, CIFAR-10H).** **[por defecto] `A_AGG="todas"`:**
  - un encoder está en Apoyo en A solo si lo está en las 3 fuentes;
  - GA1 se dispara si cualquier fuente queda en No apoyo con Holm p < .05.
  - Además se reporta un GA descriptivo por fuente.
- **b9. "Los grupos L&F y CIFAR coinciden."** **[por defecto] `CONCORD="particion_identica"`:** la misma partición de los 10 primarios en ese corte. **REVISION5 (opcional):** se reporta además el ARI (`ari_descriptivo`), solo descriptivo.
- **b10. "≥ 2 grupos" para GA2.** **[por defecto]** Se cuentan entre los 6 entrenados nuevos (`GROUPS_OVER`).
- **b11. R0 que falla el control positivo de agentes.** En lectura literal queda en GA0b (descriptivo). El análogo humano (G0) sugeriría GA0.
  - **[por defecto] literal (GA0b).**
- **b12. Exclusión de una familia Holm.** **[por defecto]** Un encoder sale de la familia si es técnico (o "no interpretable") en **cualquier** prueba de esa familia. La familia se achica en vez de imputar.
  - **Excepción para H4 en la familia C (REVISION5b, 22-11, antes del congelado):** si el H4 de un encoder falta o queda técnico (por el corte del tope de 14 h o porque falló el ajuste), sale de la familia C **solo esa prueba H4**, no el encoder. Sus H1 y H2 siguen en la familia y pueden dar GA1. Con un H4 técnico, C queda con 35 pruebas por SESOI; si faltan los 6 H4, con 30 (3 H1 + 2 H2 por encoder).
  - Motivo: si no, un corte de cómputo cambiaría la familia confirmatoria de H1/H2.
  - Las demás exclusiones de b12 no cambian. Un encoder técnico en H1 o H2, o que falla el control positivo, sale entero. Las familias D, C-pisos y D-pisos siguen como antes; en C-pisos un H4 técnico de un piso todavía saca al piso entero, porque la indicación se limitó a C.
  - El informe y `agents_summary.json` listan qué H4 salieron solo como prueba (`h4_tecnico_fuera_solo_prueba`), y `n_tests` es el número real de pruebas.
- **b13. Controles de tubería.**
  - **[por defecto]** Cualquier control fallido deja GA0 en todos los análisis.
  - El control 1 **detiene** el run. El control 2 solo marca GA0, y el run sigue para dejar los números descriptivos.
  - **REVISION5:** si hay GA0, va en el encabezado del informe de `14`.
- **b14. Valores p de D-todos.** Los 4.950 pares no son independientes (comparten clases), así que el p de OLS es anticonservador.
  - **[por defecto]** D-todos solo da el rótulo descriptivo "sensible al muestreo de pares" y no entra en ninguna familia ni en GA.
  - **REVISION5:** el informe advierte que ese rótulo puede dispararse justamente porque esos p son anticonservadores; es descriptivo.

### Añadidos menores (no cambian conclusiones, pero no están en la propuesta)

- **b15. Paridad en los 45 estímulos L&F.** Antes de codificar CIFAR, cada encoder debe reproducir `out/emb_<ID>.npz` de la parte humana con |Δ| ≤ 1e-5. Si falla, el run se detiene.
  - En sintético, la paridad pasó en los 9 checkpoints: los 8 de HF más E4s (Δ ≤ 3e-8), F1 y F2. Las imágenes eran ruido, con los pesos reales.
- **b16. H4 también para R0**, como reproducción descriptiva del H4 de la v1. La caída a `noslope` es por encoder, como en 04c. **REVISION5:** el informe lista qué estructura de efectos aleatorios usó cada encoder (columna `H4_RE` y tabla en `RESULTADOS_agentes_auto.md`).
- **b17. Semillas de contraste de D-todos.** Se usa 9000 + k, con k el índice del par en el orden triangular superior de la matriz de la v1. D-100 usa `sample_id`, como el código de la v1.
- **b18. Réplica descriptiva del Holm-de-6 de la v1 por encoder**, y GA por fuente. Ambas son descriptivas.

---

## Costo: nota sobre la parte humana y estimación realista para agentes

**Lección de la parte humana.** El plan estimó la CPU de S1 en ~7,5 h-CPU y gastó ~43 h-CPU, unas 6 veces más. En cambio, el **reloj sí coincidió**.
- La causa es que cada ajuste de R/lme4 usa varios hilos (BLAS/OpenMP): el estimado contó h-proceso como si fueran h-núcleo.
- Por eso, abajo se dan h-núcleo y reloj por separado, y H4 se corre con BLAS fijado en 2 hilos.

**Tiempos medidos hoy en sintético** (box de 8 núcleos y 15 GB; mismo código, 384 dimensiones; el sintético es más fácil que el real):

| Etapa | Medido en sintético | Comentario |
|---|---|---|
| T, 1 encoder, semilla 42, 1 hilo | 35–44 s | La v1 real tardó 5–7 min por semilla (r04): se usa la v1. |
| D, 100 pares × 4 aprendices, 1 hilo | 97–102 s por encoder | Con 768–3.072 dimensiones, logreg sube: 3–8 min estimado. |
| D-todos, 300 pares (prototipo) | 32–34 s por encoder | Equivale a ~9 min por encoder con 4.950 pares; 10–20 min con dimensiones reales. |
| H1/H2/H3, 1 encoder | ~70 s | 10.000 bootstrap + 10.000 Mantel. |
| H4, GLMM con 172.800 filas, BLAS 2 | 67–106 s por encoder (9 de 10 cayeron a `noslope`) | Con datos reales puede ser mucho más lento: en la parte humana, S1 con 24 ítems tardó 1.550–3.280 s por encoder. |
| Ensayo e2e completo (`run_agents.sh primary`, sintético) | 22 min de reloj, 62 min de CPU | Sin embeddings, porque ya estaban hechos. |
| Throughput de embeddings, 2 hilos, pesos reales sobre 128 imágenes de ruido 32×32 (22-11, ~15:43) | img/s: R0 70, O1 20, R1 41, E1 26, O2 154, E2 30, E3 52, E4(+E4s) 30, F1 51, F2 1.626 | Un proceso a la vez; con 3 en paralelo, contar ~0,8×. Antes de la recuperación medí F1 = 14 img/s con el box cargado. |

**Estimación para el run primario** (3 procesos en paralelo, box libre):

| Bloque | h-núcleo | Reloj |
|---|---|---|
| Re-descargas (pesos HF, CIFAR-10, XTMEM) | — | **hechas el 22-11** (~1 min) |
| Embeddings primarios (62.000 img × 8 encoders nuevos + R0 CIFAR-100 + R1 test) | 8–12 | 1,5–2,5 h |
| T (10 × 5–7 min) | ~1 | ~25 min |
| D (10 × 3–8 min) | 0,5–1,5 | 15–30 min |
| D-todos (10 × 10–20 min) | 2–3,5 | 40–70 min |
| H1/H2/H3 | 0,3–0,5 | ~10 min |
| H4 (10 GLMM, BLAS 2; 1,5–30 min c/u × ~1,5–2 núcleos) | 0,5–10 | 10 min – 2 h |
| **Total primario** | **~13–28** | **~3,5–7 h** (hasta ~14 h si todo sale ×2) |

Ariadna (REVISION5) midió tasas de un proceso de 0,6–1× las mías (E4 18,7 img/s, E3 31, O1 20). Su límite superior realista es ~8–9 h. Lo más incierto es H4 con datos reales.

- **Embeddings, desglose (corregido el 22-11 con los throughputs medidos con pesos reales).** En h-proceso con 2 hilos: R0 0,2; O1 0,85; R1 0,07; E1 0,66; O2 0,11; E2 0,57; E3 0,33; E4 0,58; F1 0,34; F2 ≈ 0. Son ~3,7 h-proceso, ~7,5 h-núcleo, y ~9–12 con contención y decodificación.
  - El grupo más lento de `par3` (O1, O2, E4) suma ~1,5 h.
  - **Mi estimación anterior (9–13 h de reloj, 45–60 h-núcleo) era muy pesimista:** usaba los throughputs del 18-11, medidos con el box cargado.
  - La propuesta (~6–9 h de reloj y ~40–55 h-núcleo de embeddings) también sobreestima, unas 4–5 veces.
  - No modifiqué la propuesta: el error va a favor del calendario y no cambia ninguna regla. Queda para que Patricio y Ariadna decidan si se corrige el texto.
  - El throughput salió de 128 imágenes; hay que confirmarlo en los primeros minutos del run (log de img/min de `10_agent_embed.py`).
- **Sensibilidades** (opcionales, en orden inverso de corte):
  - T-sem "episodes" (40 T): ~4 h-núcleo, ~1,5 h de reloj;
  - E4s: ~30 min;
  - F1 semillas 1–2: ~3 h-núcleo;
  - H5: minutos;
  - T-sem "full" (32 pozos de 2.000 + 40 T): ~6–8 h-núcleo, ~3 h de reloj.
- **RAM.** Con el lector arreglado (a9), T usa ~230 MB. D y D-todos con F2 (50.000 × 3.072 float32 ≈ 0,6 GB, ×2 al normalizar) quedan en ~1,5 GB por proceso: con 3 procesos cabe holgado. H4 en R: ~0,4–1 GB por ajuste.
- **Para el calendario (corregido el 22-11).** El run primario debería tomar ~3,5–7 h de reloj (hasta ~14 h si todo sale ×2), sin las sensibilidades (+~6 h de reloj, ~14–16 h-núcleo).
  - Lo más incierto ahora es H4 con datos reales (10 min – 2 h) y D/D-todos con dimensiones reales.
  - Partiendo en la mañana, el primario termina en el día con margen.
  - Lo de la parte humana (CPU subestimada ~6× por los hilos de R) se refleja contando h-núcleo y fijando BLAS = 2 en H4.

---

## REVISION5 (Ariadna, 22-11-2026): qué se aplicó

Revisión: `REVISION5_ariadna_scripts_agentes_2026-11-22.md` (sha256 `21ba87e1…`). Resultado: VB con condiciones, sin bloqueantes técnicos. Se aplicó todo menos el congelado.

**Cambios de defaults:** b2 (random como distribución), b3 (mediana de d_perc sobre 4.950 pares), b4 (rótulo "Apoyo con IC bajo SESOI"). Ver cada punto arriba.

**Notas menores aplicadas:** b6 (¿HC3 cambia la región?), b9 (ARI opcional), b13 (GA0 en el encabezado), b14 (advertencia de p anticonservadores), b16 (estructura RE por encoder).

**Condiciones:**
- **C1.** El encabezado del manifiesto (`15_freeze_agents.py --header`, que `--freeze` escribe arriba de los sha) registra:
  - el sha del caché R0 de la v1;
  - las versiones de torch, transformers y otras librerías;
  - que `perception.py` no fija la revisión de dinov2-small y que hoy `main` = `ed25f3a3`;
  - la verificación a precisión completa de Ariadna.
  - `perception.py` y REVISION5 se agregaron a la lista del manifiesto.
  - `--verify` ahora solo lee líneas `<sha256 de 64 hex>  <ruta>`, así que el encabezado no puede confundirse con un sha.
- **C2.** `DIFF_agentes_propuesta.patch` lleva arriba la nota "OBSOLETO / NO APLICADO".
- **C3.**
  - Se corrigieron los comentarios corridos en `14_agent_regions.py`: D_P b6, A_AGG b8, CONCORD b9, GROUPS_OVER b10, H4 b7 (además p_region b5 en 14 y b5 en 12).
  - En `run_agents.sh`, la línea de T-sem full ya no dice que R0 reusa los cachés r04.
  - `test_embed_agents.py` imprime SKIP por cada encoder sin pesos en el caché.
- **C4.** El informe de `14` incluye la línea §3.7: solo H1 y H2 se leen frente a las personas, y un GA2 en D significa "suficiencia del mecanismo".
- **C5.** El cómputo está declarado en el manifiesto y en `run_agents.sh`:
  - run primario: 13–28 h-núcleo y 3,5–7 h de reloj, con 3 procesos de ≤ 1,5 GB, CPU local, $0 y sin GPU ni API;
  - tope de 14 h de reloj desde el inicio de `primary` (`logs/run_start_epoch`);
  - chequeo automático de img/min a los 15 min del paso 3. Si alguna tasa es < 50 % de la medida, se registra en el log y se re-estima a mano;
  - si se supera el tope:
    - las sensibilidades se cortan enteras;
    - después, cada H4 restante se salta y queda técnico;
    - todo con línea "LOG A DEVIATION";
    - D-todos y el resto del primario no se cortan.
  - Respaldos fuera de /tmp y fuera de `out/`, en `/workspace/agents_respaldo/`:
    - `emb/` + `SHA256SUMS` al terminar el paso 3;
    - `results/` al terminar;
    - `XTMEM/`, ya copiado el 22-11 (commit `220ed17e`).
  - Paso 0: si faltan `/tmp/XTMEM` o `/tmp/renv`, los restaura (desde el respaldo, o con un clon fijado al commit) o los reconstruye (micromamba, R 4.5.3 + lme4 2.0.6).
  - El log de embeddings ahora antepone `[ID set]` a cada línea, para que el chequeo de tasa sepa qué encoder mide.

**Declaración (para el congelado):** las decisiones de (b) se tomaron después del run humano del 21–22, pero antes de cualquier resultado de agentes, y ninguna usa valores humanos. Va en el encabezado del manifiesto.

**REVISION5b (Ariadna, 22-11, `REVISION5b_ariadna_verificacion_2026-11-22.md`, sha256 `d0d2a306…`):** OK para congelar. Se agregó a la lista del manifiesto y es el archivo que se pasa a `--scripts-review`. REVISION5 no se tocó. Se aplicó su arreglo de b12: H4 técnico o faltante sale de C solo como prueba (ver b12), con dos escenarios nuevos en `test_regions_agents.py`.

**Confirmaciones de Patricio (hechas, registradas en el encabezado del manifiesto):**
1. b1 (reconstruir el caché R0 y recuperar pesos/CIFAR-10): aprobado el 22-11 a las 15:16 (t218u).
2. Congelar hoy y correr el 23-11 a las 08:30, adelantando P1: aprobado el 22-11 a las 17:59 ("Sí, congela hoy cuando Ariadna dé el OK y parte mañana a las 08:30").
