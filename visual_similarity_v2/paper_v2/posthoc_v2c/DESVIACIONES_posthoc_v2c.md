# Análisis post hoc v2c (fuera de todo plan congelado)

Registro de los análisis agregados después de las dos revisiones externas (paperreview.ai y DeepReview), con autorización de Patricio
(24-11-2026: «Solo A y B, los experimentos nuevos pueden ser declarados como trabajo futuro»). Todos son **exploratorios**:
no reemplazan ni modifican ningún resultado confirmatorio y se informan aunque sean desfavorables. Hora de Santiago (UTC−3).

| Fecha y hora | Análisis | Script | Datos de entrada | Estado |
|---|---|---|---|---|
| 24-11-2026 | P01: distancias a ejemplares (mínima y media) y dispersión del conjunto de enseñanza, por encoder | `scripts/p01_prep_exemplar.py` | `h4_long_<ID>.csv` congelados (a945ed2), embeddings congelados, `all_raw_A.csv` (md5 e42712b8…) | Verificado: `dist` reproduce la congelada (max \|Δ\| = 0) |
| 24-11-2026 | B1: pendientes aleatorias de distancia por imagen en el control positivo (8 ítems), escalera I1–I4 declarada antes de ajustar; variante con intercepto por celda imagen×condición; para R0, modelo solo-supraordinado con pendiente por imagen | `scripts/p02_image_slopes.R` | `data/ex_long_<ID>.csv` | ver `out/p02_<ID>.csv` |
| 24-11-2026 | B2: modelos de ejemplares (distancia media y mínima) frente al centroide, por AIC y pendiente; mismos efectos aleatorios que el modelo del centroide | `scripts/p03_exemplar.R` | `data/ex_long_<ID>.csv` | ver `out/p03_<ID>.csv` |
| 24-11-2026 | B3: distribución de distancias de los ítems supraordinados por condición (figura) | `scripts/p05_sup_distances.py` | `data/ex_long_<ID>.csv` | `out/fig_sup_distances.pdf` |
| 24-11-2026 | B4: H2 con estimaciones e IC por encoder, y efecto mínimo detectable (80 %, α = .05 bilateral; y α/36) para n = 61 y 32 | `scripts/p04_h2_power.py` | `agents/out/results/H_<ID>.json` congelados | `out/p04_h2_*.csv` |

Reglas: semilla 42 en todo; nada se ajustó después de ver resultados; un modelo que no converge (criterio `fit_ok` de `lib_glmm.R`:
sin error, código 0, sin avisos de convergencia, no singular) se informa como "no convergió". Tope de cómputo: ~30 min por análisis, sin GPU.
Los archivos de `data/` son derivados por ensayo para uso local; no se redistribuyen.

**Nota (24-11-2026).** Con la aprobación de Patricio, las tablas por ensayo `data/ex_long_<ID>.csv` (y las tablas congeladas `h4_long_<ID>.csv`) se publicaron en Hugging Face, en `MrPatoVera/lexical-consensus-data`, carpeta `v2_encoders/` (Parquet, CC BY 4.0, con atribución a Lewis & Frank, 2018), configuraciones `v2_trials_exemplar` y `v2_trials_centroid`. En git siguen sin guardarse; `scripts/run_posthoc.sh` las regenera.
