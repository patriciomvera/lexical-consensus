# Comparación de encoders v2 (parte humana)

Paquete de la comparación de encoders v2, parte humana, asociada a [arXiv:2611.14116](https://arxiv.org/abs/2611.14116). El congelamiento interno se escribió el 20-11-2026; la corrida se ejecutó el 21–22-11-2026 (hora de Chile, America/Santiago).

La subcarpeta se llama `encoder_equivalence/` para que las rutas del manifiesto de congelamiento coincidan con las que usa `scripts/09_freeze.py`.

## Procedencia

- Manifiesto de congelamiento: `encoder_equivalence/FREEZE_encoders_v2.txt` (sha256 `975aab4cc922a73b3b531fb22f26ce87145143c89a8b8ea6d44430a00894b606`), 47 archivos, escrito el 2026-11-20 a las 09:15:04 (America/Santiago).
- Complemento (archivos de revisión que el script de congelamiento no lista): `encoder_equivalence/FREEZE_encoders_v2_complemento.txt`.
- Plan congelado: `propuesta_comparacion_encoders_v2_2026-11-18.md` (sha256 `cfaefee5ab8ebfa764c65b6c3a28ae9d0eb23ddc43f4746c6eeb99ed9793fec4`).
- Visto bueno de Ariadna (parte humana) y revisiones: `VISTOBUENO_ariadna_parte_humana_2026-11-18.md`, `REVISION_ariadna_comparacion_encoders_v2_2026-11-18.md`, `REVISION2_ariadna_comparacion_encoders_v2_2026-11-18.md`, `REVISION3_ariadna_comparacion_encoders_v2_2026-11-18.md`, `REVISION3b_ariadna_diff_2026-11-18.md` y `REVISION4_ariadna_agentes_multi_encoder_2026-11-18.md` (estos dos últimos figuran en el complemento).

## Contenido

- `encoder_equivalence/scripts/`: scripts del análisis (Python y R), incluidos los tests y `09_freeze.py`.
- `encoder_equivalence/results/`: salidas de la corrida. Las pendientes por experimento y encoder están en `stage1_*.csv`. Ahí también están la selección de estructura, el análisis primario (`stage2_primary.csv`, `results/regions/`), las sensibilidades S1–S8 y S6, S9 (`results/regions_with_S9/`), F0 y el control positivo.
- `encoder_equivalence/out/`: embeddings de los 45 estímulos por encoder (`emb_*.npz`). Las distancias coseno del análisis se obtienen de estos archivos; no hay una matriz de distancias como archivo aparte. Los `emb_*_s6common.npz` corresponden a la sensibilidad S6 (preprocesamiento común) y son posteriores al congelamiento.
- `encoder_equivalence/data/`: datos por ensayo (`h4_long_*.csv`), uno por encoder, más las versiones `*_s6common.csv` de S6. Cada fila es una respuesta con la distancia del estímulo. Estos archivos se derivan de Lewis & Frank (2018), XTMEM (licencia CC BY 4.0 en el proyecto OSF de los autores: https://osf.io/yekhj/).
- `encoder_equivalence/logs/`: registros de la corrida del 21–22-11-2026.
- `encoder_equivalence/figs/`: figuras generadas por la corrida.
- `encoder_equivalence/RESULTADOS_encoders.md`: cifras de la corrida, cada una con su archivo de origen.
- `encoder_equivalence/DESVIACIONES_encoders_v2.md`: desviaciones registradas respecto del plan.
- Instantáneas del plan (`propuesta_*snapshot*.md`) y parches de revisión (`DIFF_*.patch`) en `encoder_equivalence/`.

## Verificar el congelamiento

Desde esta carpeta (`encoder_equivalence_v2/`):

```bash
python encoder_equivalence/scripts/09_freeze.py --verify
```

El verificador compara los 47 archivos del manifiesto. Los embeddings `emb_*_s6common.npz` y otros archivos posteriores al congelamiento aparecen como «files not in manifest»; eso es esperado y no indica un desajuste de los archivos congelados.

## Tamaños

Ningún archivo de este paquete supera 50 MB. No se comprimió ninguno y no se usó Git LFS.
