# Anexo de agentes (v2 de arXiv 2611.14116): scripts, pruebas y orden de ejecución

**Estado (22-11-2026):** escrito y probado solo con datos sintéticos. REVISION5 de Ariadna (VB con condiciones) aplicada; ver el final de `ACLARACIONES_scripts.md`. **No se ha corrido el run de agentes.** Con datos reales solo se reconstruyó el caché R0 de la v1 (b1, aprobado).

Antes de correr hay que completar los requisitos de las secciones siguientes.

## Requisitos previos y congelado
1. **Revisión de Ariadna**, con un archivo de revisión escrito: `REVISION5_ariadna_scripts_agentes_2026-11-22.md`, VB con condiciones C1–C5, aplicadas.
2. **Aprobación de Patricio**, incluida la confirmación en el registro de b1 y de correr el 23-11, adelantando P1.
3. **Puntos de `ACLARACIONES_scripts.md` (b) decididos** y llevados a la propuesta o a las constantes de los scripts.
4. **Restituciones tras el reinicio del 22-11**, ver ACLARACIONES (b1). La reconstrucción del caché v1 de CIFAR-10 requiere la aprobación de Patricio:
   - `r03_build_cache.py`;
   - pesos de Hugging Face;
   - CIFAR-10;
   - `/tmp/XTMEM`.
5. **Congelado interno:**
   `python agents/scripts/15_freeze_agents.py --freeze --scripts-review /workspace/lexical-consensus-openmind/REVISION5b_ariadna_verificacion_2026-11-22.md`
   Escribe `encoder_equivalence/FREEZE_agentes_v2.txt`, con un encabezado de declaraciones (C1, C5 y la declaración de (b)) que se ve antes con `--header`. Nunca sobrescribe, y exige que el congelado humano verifique.

## Ejecución (después del congelado)
- `bash agents/scripts/run_agents.sh primary`. Corre en orden:
  1. verificación (`15 --verify`, `09 --verify`, R, XTMEM);
  2. paridad L&F;
  3. control 1;
  4. embeddings;
  5. dispersión §3.6;
  6. T, D y D-todos;
  7. H1/H2/H3;
  8. H4;
  9. regiones y GA (`14`) y figura (`14b`).

  Se detiene en el primer error.
- **Cómputo declarado (REVISION5 C5):**
  - 13–28 h-núcleo, 3,5–7 h de reloj, 3 procesos de ≤ 1,5 GB, CPU local, $0, sin GPU ni API;
  - **tope de 14 h de reloj**;
  - tasa de img/min chequeada a los 15 min de los embeddings; < 50 % se registra y se re-estima;
  - si se supera el tope, se cortan primero las sensibilidades y después H4 por encoder. D-todos y el resto del primario nunca se cortan;
  - respaldos en `/workspace/agents_respaldo/` (emb + SHA256SUMS tras el paso 3, results al final, XTMEM).
- `bash agents/scripts/run_agents.sh sensitivities` es opcional. Corre en orden inverso de corte (§6.9): T-sem "episodes" → E4s → F1 semillas 1–2 → H5 → T-sem "full".
- Python: `/workspace/rv_venv/bin/python`. R: `/tmp/renv/bin/Rscript` (R 4.5.3 + lme4 2.0.6).

## Scripts (`agents/scripts/`)
| Script | Qué hace |
|---|---|
| `lib_agents.py` | Constantes, rutas (modo sintético con `AGENTS_SYNTH_DIR=/tmp/synth_agents*`) y carga sin cambios del código de la v1. |
| `10_agent_embed.py` | Embeddings de CIFAR por encoder (mismas ramas y revisiones que `01_embed.py`), por bloques y reanudable, con manifiesto sha256. `--parity` sobre los 45 estímulos L&F. |
| `11_agent_sims.py` | Dispersión §3.6; T (`run_clean.py` de la v1 sin cambios); D (100 pares, `_run_pair` de exp_008); D-todos (4.950 pares); controles 1 y 2 de §3.5. |
| `12_agent_h1h2h3.py` | H1 (ρ + IC bootstrap + Mantel + p unilateral frente al SESOI), H2/H2b (HC3), H3 descriptivo, H5 opcional. |
| `13_h4_by_encoder.R` | H4 por encoder: el modelo C1 de `04c_h4_person_re.R`, sin cambios, con su caída a (1\|persona). |
| `14_agent_regions.py` | Regiones por encoder sin ajuste, familias Holm (C, D, C-pisos, D-pisos; solo para GA1), GA0–GA3 en la grilla 3 SESOI × 3 cortes, concordancia CIFAR, rótulo "sensible al muestreo de pares", tabla agentes × personas e informe md. |
| `14b_fig_x1.py` | Figura descriptiva. |
| `15_freeze_agents.py` | `--list`, `--verify`, `--freeze`. |
| `run_agents.sh` | Orquestador (3 procesos en paralelo; para al primer fallo). |

## Pruebas (`agents/tests/`, solo datos sintéticos)
| Prueba | Cubre |
|---|---|
| `make_synth_agents.py` | Mundo sintético con efectos plantados (encoders buenos, pisos nulos y un encoder "muerto"). Los resultados "v1" salen del código v1 sin cambios. |
| `test_regions_agents.py` | 16 escenarios: precedencia de regiones y de GA, Holm, concordancia, puerta del control positivo, rótulo de muestreo, y H4 técnico o faltante que sale de la familia C solo como prueba (b12, REVISION5b). |
| `test_controls_agents.py` | Control 1 (estricto / b1, ledgers perturbados) y control 2 (PASA / PASA con desviación / FALLA). |
| `test_embed_agents.py` | Paridad con una copia de `01_embed.py` (solo cambian las rutas) sobre imágenes de ruido, selección del pozo exp_007, bloques y reanudación, cachés existentes y throughput. |
| `test_freeze_agents.py` | `--freeze` y `--verify` sobre una copia con el manifiesto en `/tmp`, manipulación, no sobrescritura y guardia del modo sintético. |
| `test_e2e_agents.py` | Verifica los efectos plantados después de `run_agents.sh primary` en `/tmp/synth_agents_e2e`. |

Ensayo completo:
```
AGENTS_SYNTH_DIR=/tmp/synth_agents_e2e python tests/make_synth_agents.py /tmp/synth_agents_e2e --h4-full
AGENTS_SYNTH_DIR=/tmp/synth_agents_e2e AGENTS_SYNTH_SKIP_PARITY=1 AGENTS_SYNTH_DALL_MAX=300 bash scripts/run_agents.sh primary
python tests/test_e2e_agents.py
```
