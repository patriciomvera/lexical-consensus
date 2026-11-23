# Comparación de encoders v2 (parte humana y anexo de agentes)

Paquete de la comparación de encoders v2, asociada a [arXiv:2611.14116](https://arxiv.org/abs/2611.14116). El congelamiento interno de la parte humana se escribió el 20-11-2026; esa corrida se ejecutó el 21–22-11-2026 (hora de Chile, America/Santiago). El congelamiento del anexo de agentes se escribió el 22-11-2026; esa corrida primaria se ejecutó el 23-11-2026 (hora de Chile, America/Santiago).

La subcarpeta se llama `encoder_equivalence/` para que las rutas del manifiesto de congelamiento coincidan con las que usa `scripts/09_freeze.py`. Los scripts y salidas de agentes están en `encoder_equivalence/agents/`.

## Procedencia

- Manifiesto de congelamiento (parte humana): `encoder_equivalence/FREEZE_encoders_v2.txt` (sha256 `975aab4cc922a73b3b531fb22f26ce87145143c89a8b8ea6d44430a00894b606`), 47 archivos, escrito el 2026-11-20 a las 09:15:04 (America/Santiago).
- Complemento (archivos de revisión que el script de congelamiento no lista): `encoder_equivalence/FREEZE_encoders_v2_complemento.txt`.
- Plan congelado (parte humana): `propuesta_comparacion_encoders_v2_2026-11-18.md` (sha256 `cfaefee5ab8ebfa764c65b6c3a28ae9d0eb23ddc43f4746c6eeb99ed9793fec4`).
- Visto bueno de Ariadna (parte humana) y revisiones: `VISTOBUENO_ariadna_parte_humana_2026-11-18.md`, `REVISION_ariadna_comparacion_encoders_v2_2026-11-18.md`, `REVISION2_ariadna_comparacion_encoders_v2_2026-11-18.md`, `REVISION3_ariadna_comparacion_encoders_v2_2026-11-18.md`, `REVISION3b_ariadna_diff_2026-11-18.md` y `REVISION4_ariadna_agentes_multi_encoder_2026-11-18.md` (estos dos últimos figuran en el complemento).
- Manifiesto de congelamiento (anexo de agentes): `encoder_equivalence/FREEZE_agentes_v2.txt` (sha256 `5a33e098a899877d813e2bcae830798ad11b51f74c60b1e542cfbda26bd93673`), 99 archivos, escrito el 2026-11-22 a las 18:00:36 (America/Santiago).
- Plan congelado (anexo de agentes): `propuesta_agentes_multi_encoder_v2_2026-11-18.md` (sha256 `633a5e7eabf0a8cc387e4d884d8d42d68ff40289a745773cde724cf93300db25`), con instantánea `propuesta_agentes_bfc99fac_snapshot.md`.
- Revisiones de scripts de agentes: `REVISION5_ariadna_scripts_agentes_2026-11-22.md` y `REVISION5b_ariadna_verificacion_2026-11-22.md`.

## Contenido

- `encoder_equivalence/scripts/`: scripts del análisis de la parte humana (Python y R), incluidos los tests y `09_freeze.py`.
- `encoder_equivalence/results/`: salidas de la corrida humana. Las pendientes por experimento y encoder están en `stage1_*.csv`. Ahí también están la selección de estructura, el análisis primario (`stage2_primary.csv`, `results/regions/`), las sensibilidades S1–S8 y S6, S9 (`results/regions_with_S9/`), F0 y el control positivo.
- `encoder_equivalence/out/`: embeddings de los 45 estímulos por encoder (`emb_*.npz`). Las distancias coseno del análisis se obtienen de estos archivos; no hay una matriz de distancias como archivo aparte. Los `emb_*_s6common.npz` corresponden a la sensibilidad S6 (preprocesamiento común) y son posteriores al congelamiento.
- `encoder_equivalence/data/`: datos por ensayo (`h4_long_*.csv`), uno por encoder, más las versiones `*_s6common.csv` de S6. Cada fila es una respuesta con la distancia del estímulo. Estos archivos se derivan de Lewis & Frank (2018), XTMEM (licencia CC BY 4.0 en el proyecto OSF de los autores: https://osf.io/yekhj/).
- `encoder_equivalence/logs/`: registros de la corrida del 21–22-11-2026.
- `encoder_equivalence/figs/`: figuras generadas por la corrida humana.
- `encoder_equivalence/RESULTADOS_encoders.md`: cifras de la corrida humana, cada una con su archivo de origen.
- `encoder_equivalence/DESVIACIONES_encoders_v2.md`: desviaciones registradas respecto del plan (parte humana y, en un anexo, la corrida de agentes).
- Instantáneas del plan (`propuesta_*snapshot*.md`) y parches de revisión (`DIFF_*.patch`) en `encoder_equivalence/`.
- `encoder_equivalence/agents/scripts/` y `encoder_equivalence/agents/tests/`: scripts del anexo de agentes (Python y R), el orquestador `run_agents.sh` y `15_freeze_agents.py`.
- `encoder_equivalence/agents/out/results/`: salidas de la corrida primaria de agentes del 23-11-2026 (paridad, controles, T, D, D-todos, H1–H3, H4, dispersión y `regions/`).
- `encoder_equivalence/agents/out/sims/`: ledgers por episodio de T, tablas de episodios de D y archivos `dperc_c100_*.npz` de la dispersión (todos por debajo de 50 MB).
- `encoder_equivalence/agents/out/emb/embed_manifest.jsonl`: manifiesto de embeddings de CIFAR generados o reutilizados en la corrida. Los archivos `.npz` de embeddings de CIFAR no forman parte de este paquete (superan 50 MB; véase «Tamaños»).
- `encoder_equivalence/agents/logs/`: registros de la corrida primaria del 23-11-2026, incluida la reestimación de tasa y la nota de reanudación.
- `encoder_equivalence/agents/RESULTADOS_agentes.md`: cifras de la corrida de agentes, cada una con su archivo de origen.
- `encoder_equivalence/agents/README_agentes.md` y `encoder_equivalence/agents/ACLARACIONES_scripts.md`.

## Verificar el congelamiento

Desde esta carpeta (`encoder_equivalence_v2/`):

```bash
python encoder_equivalence/scripts/09_freeze.py --verify
```

El verificador compara los 47 archivos del manifiesto de la parte humana. Los embeddings `emb_*_s6common.npz` y otros archivos posteriores al congelamiento aparecen como «files not in manifest»; eso es esperado y no indica un desajuste de los archivos congelados.

El manifiesto de agentes, `encoder_equivalence/FREEZE_agentes_v2.txt`, lista 99 archivos según las rutas de la máquina de la corrida (incluye caches y rutas fuera de este paquete). La verificación `python encoder_equivalence/agents/scripts/15_freeze_agents.py --verify` se ejecutó ahí (OK). En este repositorio no se esperan los 99 archivos en esas mismas rutas.

## Tamaños

Ningún archivo de este paquete supera 50 MB. No se comprimió ninguno y no se usó Git LFS. No se incluyen los embeddings de CIFAR de `agents/out/emb/*.npz` (varios superan 50 MB y el límite de 100 MB de GitHub).
