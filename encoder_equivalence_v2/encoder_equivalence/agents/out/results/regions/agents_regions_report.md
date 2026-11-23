# Agentes: regiones y conclusiones GA (generado por 14_agent_regions.py; sin interpretación)

- §3.7 (circularidad): frente a las personas solo se leen H1 y H2; un GA2 en D significa "suficiencia del mecanismo" (el mecanismo basta con otros encoders), no correspondencia con las personas.
- Control de tubería de agentes (§3.5): PASA (T exacto: True; CIFAR-100 recodificado: PASA).
- Márgenes de agentes (SESOI, escala de correlación; NO comparables con los δ humanos): 0.10, 0.20, 0.30.
- Geometría: grupos medidos en estímulos de Lewis & Frank; concordancia con CIFAR por corte: {'0.3': False, '0.5': False, '0.7': False}.

## Familias Holm (solo para GA1)

- **C:** incluidos ['O1_dinov2b', 'E1_siglip', 'O2_openclip', 'E2_vitsup', 'E3_rn50sup', 'E4_mae']; excluidos {}; 36 pruebas por SESOI.
- **D:** incluidos ['O1_dinov2b', 'E1_siglip', 'O2_openclip', 'E2_vitsup', 'E3_rn50sup', 'E4_mae']; excluidos {}; 6 pruebas por SESOI.
- **C-pisos:** incluidos ['F1_rn50rand0', 'F2_pixels']; excluidos {}; 12 pruebas por SESOI.
- **D-pisos:** incluidos ['F1_rn50rand0', 'F2_pixels']; excluidos {}; 2 pruebas por SESOI.

## Regiones por encoder (sin ajuste)

| encoder | analisis | fuente | control_positivo_agentes | region_0.10 | region_0.20 | region_0.30 | perfil |
|---|---|---|---|---|---|---|---|
| R0_dinov2s | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| R0_dinov2s | P | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R0_dinov2s | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| O1_dinov2b | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| O1_dinov2b | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O1_dinov2b | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | P | THINGS | pasa | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| R1_cvcl | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| R1_cvcl | D | - | pasa | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| R1_cvcl | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| R1_cvcl | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E1_siglip | P | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E1_siglip | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| O2_openclip | P | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| O2_openclip | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | P | THINGS | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E2_vitsup | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E2_vitsup | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E2_vitsup | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | P | THINGS | pasa | Inconcluso | Inconcluso | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10, 0.20); No apoyo (SESOI = 0.30) |
| E3_rn50sup | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E3_rn50sup | D | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E3_rn50sup | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | A | THINGS | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | A | HSJ | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | A | CIFAR10H | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | P | THINGS | pasa | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| E4_mae | P | HSJ | pasa | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| E4_mae | D | - | pasa | Inconcluso | Inconcluso | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10, 0.20); No apoyo (SESOI = 0.30) |
| E4_mae | Dall | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4_mae | H4 | - | pasa | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | A | THINGS | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | A | HSJ | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | A | CIFAR10H | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | P | THINGS | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F1_rn50rand0 | P | HSJ | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F1_rn50rand0 | D | - | no es puerta | No apoyo | No apoyo | No apoyo | robusto: No apoyo |
| F1_rn50rand0 | Dall | - | no es puerta | No apoyo | No apoyo | No apoyo | robusto: No apoyo |
| F1_rn50rand0 | H4 | - | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | A | THINGS | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | A | HSJ | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | A | CIFAR10H | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F2_pixels | P | THINGS | no es puerta | Inconcluso | No apoyo | No apoyo | dependiente del margen: Inconcluso (SESOI = 0.10); No apoyo (SESOI = 0.20, 0.30) |
| F2_pixels | P | HSJ | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F2_pixels | D | - | no es puerta | Inconcluso | Inconcluso | Inconcluso | robusto: Inconcluso |
| F2_pixels | Dall | - | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| F2_pixels | H4 | - | no es puerta | Apoyo | Apoyo | Apoyo | robusto: Apoyo |
| E4s_mae_cls | A | THINGS | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | A | HSJ | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | A | CIFAR10H | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | P | THINGS | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | P | HSJ | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | D | - | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | Dall | - | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| E4s_mae_cls | H4 | - | no evaluable | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | A | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | A | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | A | CIFAR10H | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | P | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | P | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | D | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | Dall | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand1 | H4 | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | A | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | A | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | A | CIFAR10H | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | P | THINGS | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | P | HSJ | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | D | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | Dall | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |
| F1_rn50rand2 | H4 | - | no es puerta | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | robusto: inconcluso (técnico) |

## GA por análisis (SESOI × corte)

| analisis | sesoi | corte | conclusion | detalle |
|---|---|---|---|---|
| D | 0.1 | 0.3 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.1 | 0.5 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.1 | 0.7 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.2 | 0.3 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.2 | 0.5 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.2 | 0.7 | GA3 | inconcluso o mixto: {'E4_mae': 'Inconcluso'} |
| D | 0.3 | 0.3 | GA1 | depende del encoder: ['E4_mae']; grupos: sin grupos |
| D | 0.3 | 0.5 | GA1 | depende del encoder: ['E4_mae']; grupos: sin grupos |
| D | 0.3 | 0.7 | GA1 | depende del encoder: ['E4_mae']; grupos: sin grupos |
| A | 0.1 | 0.3 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.1 | 0.5 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.1 | 0.7 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.2 | 0.3 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.2 | 0.5 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.2 | 0.7 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.3 | 0.3 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.3 | 0.5 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| A | 0.3 | 0.7 | GA2b | se mantiene en todos, sin evidencia de diferencia geométrica (calibración L&F falla) |
| P | 0.1 | 0.3 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.1 | 0.5 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.1 | 0.7 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.2 | 0.3 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.2 | 0.5 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.2 | 0.7 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.3 | 0.3 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.3 | 0.5 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |
| P | 0.3 | 0.7 | GA0b | descriptivo, encoder por encoder: R0 en 'Inconcluso' |

- **Perfil D:** depende solo del margen: SESOI = 0.10: GA3; SESOI = 0.20: GA3; SESOI = 0.30: GA1
- **Perfil A:** robusta: GA2b
- **Perfil P:** robusta: GA0b

## D con 100 pares vs D-todos (rótulo descriptivo)

Nota: los p de D-todos son anticonservadores (los 4.950 pares comparten clases), así que el rótulo puede dispararse por eso; es solo descriptivo y no cambia GA.

| encoder | D100_0.10 | D100_0.20 | D100_0.30 | Dtodos_0.10 | Dtodos_0.20 | Dtodos_0.30 | rotulo | rule_centroid_D100 |
|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | A |
| O1_dinov2b | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | A |
| R1_cvcl | Inconcluso | No apoyo | No apoyo | Apoyo | Apoyo | Apoyo | sensible al muestreo de pares | C |
| E1_siglip | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | B |
| O2_openclip | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | B |
| E2_vitsup | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | A |
| E3_rn50sup | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | Apoyo | estable | B |
| E4_mae | Inconcluso | Inconcluso | No apoyo | Apoyo | Apoyo | Apoyo | sensible al muestreo de pares | C |
| F1_rn50rand0 | No apoyo | No apoyo | No apoyo | No apoyo | No apoyo | No apoyo | estable | C |
| F2_pixels | Inconcluso | Inconcluso | Inconcluso | Apoyo | Apoyo | Apoyo | sensible al muestreo de pares | B |
| E4s_mae_cls | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | estable |  |
| F1_rn50rand1 | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | estable |  |
| F1_rn50rand2 | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | inconcluso (técnico) | estable |  |

## H4: estructura de efectos aleatorios usada por encoder (caída a `noslope` por encoder, como 04c)

| encoder | RE | status |
|---|---|---|
| R0_dinov2s | slope | ok |
| O1_dinov2b | slope | ok |
| R1_cvcl | slope | ok |
| E1_siglip | slope | ok |
| O2_openclip | slope | ok |
| E2_vitsup | slope | ok |
| E3_rn50sup | slope | ok |
| E4_mae | slope | ok |
| F1_rn50rand0 | slope | ok |
| F2_pixels | slope | ok |
| E4s_mae_cls |  | falta |
| F1_rn50rand1 |  | falta |
| F1_rn50rand2 |  | falta |

- Rótulo descriptivo «Apoyo con IC bajo SESOI» (b4) y región con HC3 en D (b6): columnas `rotulo_*`, `region_hc3_*` y `hc3_cambia_region` de regions_agents.csv. ARI L&F vs CIFAR por corte (b9, descriptivo): {'0.3': 0.352, '0.5': 0.10179640718562874, '0.7': 0.10256410256410256}.

## Tabla agentes × personas (descriptiva)

| encoder | humano_beta_zd_control_positivo | humano_ic_lo | humano_ic_hi | humano_control_positivo | H4_logOR | H4_lo | H4_hi | H4_RE | humano_region_d0.05 | humano_region_d0.10 | humano_region_d0.20 | humano_region_24 | humano_perfil | C1_native | C1_near | C1_mid | C1_far | D_beta_perc_v1escala | D_beta_perc_z | D_regla_v1 | H1_rho_THINGS | H1_rho_HSJ | H1_rho_CIFAR10H | H2_beta_humana_THINGS | H2_beta_humana_HSJ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | -3.157 | -3.296 | -3.017 | True | -3.226 | -3.288 | -3.164 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9429 | 0.8467 | 0.6542 | 0.5296 | -0.2119 | -0.02885 | A | 0.5228 | 0.5006 | 0.7949 | -0.00785 | -0.03445 |
| O1_dinov2b | -2.848 | -2.976 | -2.72 | True | -2.859 | -2.914 | -2.804 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9633 | 0.8344 | 0.6691 | 0.5444 | -0.2362 | -0.02747 | A | 0.5207 | 0.5475 | 0.6771 | -0.01083 | -0.02728 |
| R1_cvcl | -0.6152 | -0.7054 | -0.525 | True | -3.61 | -3.686 | -3.535 | slope | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20) | 0.6753 | 0.6469 | 0.5107 | 0.4276 | 0.00875 | 0.001281 | C | 0.6825 | 0.4294 | 0.7862 | 0.01136 | -0.0449 |
| E1_siglip | -3.887 | -4.059 | -3.715 | True | -5.483 | -5.584 | -5.381 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9576 | 0.896 | 0.6964 | 0.5342 | -0.5077 | -0.02168 | B | 0.6392 | 0.5008 | 0.8896 | -0.01659 | -0.05083 |
| O2_openclip | -3.993 | -4.157 | -3.83 | True | -4.805 | -4.903 | -4.706 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9702 | 0.9144 | 0.6709 | 0.524 | -0.2733 | -0.02429 | B | 0.7018 | 0.5187 | 0.8669 | -0.01995 | -0.04302 |
| E2_vitsup | -4.296 | -4.462 | -4.13 | True | -4.456 | -4.555 | -4.357 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.9776 | 0.8993 | 0.7202 | 0.518 | -0.2887 | -0.03542 | A | 0.3934 | 0.556 | 0.6611 | -0.006119 | -0.03105 |
| E3_rn50sup | -2.648 | -2.775 | -2.52 | True | -4.14 | -4.233 | -4.047 | slope | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20) | 0.8704 | 0.8576 | 0.6442 | 0.4947 | -0.2594 | -0.02345 | B | 0.6181 | 0.5328 | 0.912 | -0.004229 | -0.04409 |
| E4_mae | -0.1982 | -0.2971 | -0.09943 | True | -7.557 | -7.662 | -7.452 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.4827 | 0.4238 | 0.4818 | 0.3882 | -0.00155 | -0.0002797 | C | 0.2918 | 0.2273 | 0.4449 | 0.01353 | -0.02376 |
| F1_rn50rand0 | -1.102 | -1.215 | -0.9897 | True | -2.678 | -2.728 | -2.629 | slope | inconcluso | inconcluso | inconcluso | inconcluso | robusto: inconcluso | 0.4102 | 0.4311 | 0.4282 | 0.3844 | 49.18 | 0.01197 | C | 0.1088 | 0.1063 | 0.5302 | -0.006122 | -0.02757 |
| F2_pixels | 0.116 | 0.03845 | 0.1935 | False | -2.43 | -2.467 | -2.394 | slope | inconcluso | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | inconcluso | dependiente del margen: inconcluso (δ = 0.05); menor que δ (equivalente con margen δ) (δ = 0.10, 0.20) | 0.4578 | 0.4433 | 0.4602 | 0.3869 | -0.02269 | -0.009888 | B | 0.1505 | 0.1988 | 0.2941 | 0.008664 | -0.02477 |
| E4s_mae_cls |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F1_rn50rand1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| F1_rn50rand2 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

Spearman entre encoders (descriptiva, sin p): {"_nota": "Spearman entre encoders (los 10 primarios), solo descriptiva: sin p ni IC inferencial", "beta_zd_humano_vs_H1_rho_THINGS": {"n": 10, "rho": -0.4545454545454545}, "beta_zd_humano_vs_H1_rho_HSJ": {"n": 10, "rho": -0.7454545454545454}, "beta_zd_humano_vs_H1_rho_CIFAR10H": {"n": 10, "rho": -0.5515151515151515}, "beta_zd_humano_vs_H2_beta_humana_THINGS": {"n": 10, "rho": 0.7454545454545454}, "beta_zd_humano_vs_H2_beta_humana_HSJ": {"n": 10, "rho": 0.4545454545454545}}

## Dispersión de d_perc en los 100 pares (§3.6)

```
{
 "_rule": "100 v1 pairs (phase2/sampled_pairs.csv); d_sem = exp_008 d_sem; quadrants (quadrant_counts): d_perc median of the encoder over the 4,950 pairs and d_sem median over the 4,950 (v1 rule, REVISION5 b3); quadrant_counts_100: d_perc median over the 100 pairs (descriptive). Descriptive; changes no rule.",
 "R0_dinov2s": {
  "n": 100,
  "sd": 0.13613767209690872,
  "min": 0.1414257287979126,
  "max": 0.8853307366371155,
  "range": 0.7439050078392029,
  "spearman_dperc_dsem": 0.01295110698443623,
  "quadrant_counts": {
   "Q1": 25,
   "Q2": 25,
   "Q3": 25,
   "Q4": 25
  },
  "quadrant_counts_100": {
   "Q1": 25,
   "Q2": 25,
   "Q3": 25,
   "Q4": 25
  }
 },
 "O1_dinov2b": {
  "n": 100,
  "sd": 0.1162828286086192,
  "min": 0.15615785121917725,
  "max": 0.9425162076950073,
  "range": 0.7863583564758301,
  "spearman_dperc_dsem": -0.006388492272341284,
  "quadrant_counts": {
   "Q1": 21,
   "Q2": 26,
   "Q3": 29,
   "Q4": 24
  },
  "quadrant_counts_100": {
   "Q1": 22,
   "Q2": 22,
   "Q3": 28,
   "Q4": 28
  }
 },
 "R1_cvcl": {
  "n": 100,
  "sd": 0.14632199426941384,
  "min": 0.03745764493942261,
  "max": 0.6267770528793335,
  "range": 0.5893194079399109,
  "spearman_dperc_dsem": 0.11970617521961292,
  "quadrant_counts": {
   "Q1": 29,
   "Q2": 31,
   "Q3": 21,
   "Q4": 19
  },
  "quadrant_counts_100": {
   "Q1": 29,
   "Q2": 29,
   "Q3": 21,
   "Q4": 21
  }
 },
 "E1_siglip": {
  "n": 100,
  "sd": 0.042696184192094026,
  "min": 0.015215158462524414,
  "max": 0.20334410667419434,
  "range": 0.18812894821166992,
  "spearman_dperc_dsem": 0.08024042361613619,
  "quadrant_counts": {
   "Q1": 28,
   "Q2": 26,
   "Q3": 22,
   "Q4": 24
  },
  "quadrant_counts_100": {
   "Q1": 28,
   "Q2": 28,
   "Q3": 22,
   "Q4": 22
  }
 },
 "O2_openclip": {
  "n": 100,
  "sd": 0.0888672826840226,
  "min": 0.03589349985122681,
  "max": 0.4585176110267639,
  "range": 0.4226241111755371,
  "spearman_dperc_dsem": 0.12873652519856152,
  "quadrant_counts": {
   "Q1": 32,
   "Q2": 30,
   "Q3": 18,
   "Q4": 20
  },
  "quadrant_counts_100": {
   "Q1": 30,
   "Q2": 30,
   "Q3": 20,
   "Q4": 20
  }
 },
 "E2_vitsup": {
  "n": 100,
  "sd": 0.12268824662406681,
  "min": 0.18996769189834595,
  "max": 0.8461833000183105,
  "range": 0.6562156081199646,
  "spearman_dperc_dsem": 0.051191997287576854,
  "quadrant_counts": {
   "Q1": 24,
   "Q2": 27,
   "Q3": 26,
   "Q4": 23
  },
  "quadrant_counts_100": {
   "Q1": 24,
   "Q2": 24,
   "Q3": 26,
   "Q4": 26
  }
 },
 "E3_rn50sup": {
  "n": 100,
  "sd": 0.09043117073404477,
  "min": 0.0369870662689209,
  "max": 0.4884408116340637,
  "range": 0.4514537453651428,
  "spearman_dperc_dsem": 0.07574326129284333,
  "quadrant_counts": {
   "Q1": 23,
   "Q2": 27,
   "Q3": 27,
   "Q4": 23
  },
  "quadrant_counts_100": {
   "Q1": 26,
   "Q2": 26,
   "Q3": 24,
   "Q4": 24
  }
 },
 "E4_mae": {
  "n": 100,
  "sd": 0.18046432665954423,
  "min": 0.011018753051757812,
  "max": 0.8762620687484741,
  "range": 0.8652433156967163,
  "spearman_dperc_dsem": 0.19609188827292665,
  "quadrant_counts": {
   "Q1": 23,
   "Q2": 32,
   "Q3": 27,
   "Q4": 18
  },
  "quadrant_counts_100": {
   "Q1": 27,
   "Q2": 27,
   "Q3": 23,
   "Q4": 23
  }
 },
 "F1_rn50rand0": {
  "n": 100,
  "sd": 0.00024350480795842487,
  "min": 2.0503997802734375e-05,
  "max": 0.001495063304901123,
  "range": 0.0014745593070983887,
  "spearman_dperc_dsem": 0.01894932482284689,
  "quadrant_counts": {
   "Q1": 24,
   "Q2": 27,
   "Q3": 26,
   "Q4": 23
  },
  "quadrant_counts_100": {
   "Q1": 27,
   "Q2": 27,
   "Q3": 23,
   "Q4": 23
  }
 },
 "F2_pixels": {
  "n": 100,
  "sd": 0.4357685845211876,
  "min": 0.06994873285293579,
  "max": 1.8120839595794678,
  "range": 1.742135226726532,
  "spearman_dperc_dsem": 0.20052300415355434,
  "quadrant_counts": {
   "Q1": 22,
   "Q2": 32,
   "Q3": 28,
   "Q4": 18
  },
  "quadrant_counts_100": {
   "Q1": 28,
   "Q2": 28,
   "Q3": 22,
   "Q4": 22
  }
 },
 "E4s_mae_cls": {
  "n": 100,
  "sd": 0.013674846146170093,
  "min": 0.0016238093376159668,
  "max": 0.07278639078140259,
  "range": 0.07116258144378662,
  "spearman_dperc_dsem": 0.16452168867896946,
  "quadrant_counts": {
   "Q1": 25,
   "Q2": 31,
   "Q3": 25,
   "Q4": 19
  },
  "quadrant_counts_100": {
   "Q1": 30,
   "Q2": 30,
   "Q3": 20,
   "Q4": 20
  }
 },
 "F1_rn50rand1": {
  "status": "tecnico: falta /workspace/lexical-consensus-openmind/encoder_equivalence/agents/out/emb/c100train_F1_rn50rand1.npz"
 },
 "F1_rn50rand2": {
  "status": "tecnico: falta /workspace/lexical-consensus-openmind/encoder_equivalence/agents/out/emb/c100train_F1_rn50rand2.npz"
 }
}
```