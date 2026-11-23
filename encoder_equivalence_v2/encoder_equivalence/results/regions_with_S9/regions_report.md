# Regiones y conclusión global (plan §7; generado por 08_regions.py)

- Control de tubería: PASA. Perfil del agrupamiento de geometría: depende del corte.
- Referencia descriptiva, no confirmatoria (I-b): Δ en dos etapas de R0 con 8 ítems = 0.019709506140741517 [IC 95 % -0.204779; 0.274221].
- Lectura de R0 con 24 ítems (REVISION3b): inconcluso.

## Tabla única (§7.2): región con δ = 0,05 / 0,10 / 0,20 y con 24 ítems

| encoder | control_positivo | familia_holm | region_d0.05 | region_d0.10 | region_d0.20 | region_24 | tipo_de_ajuste | perfil |
|---|---|---|---|---|---|---|---|---|
| R0_dinov2s | pasa | R0: referencia (no entra en Holm; G1) | inconcluso | inconcluso | inconcluso | inconcluso | sensible al tipo de ajuste | robusto: inconcluso; sensible al tipo de ajuste (S9, descriptivo) |
| O1_dinov2b | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | sensible al tipo de ajuste | robusto: inconcluso; sensible al tipo de ajuste (S9, descriptivo) |
| R1_cvcl | pasa | familia entrenados | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | sensible al tipo de ajuste | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20); sensible al tipo de ajuste (S9, descriptivo) |
| E1_siglip | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| O2_openclip | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| E2_vitsup | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | sensible al tipo de ajuste | robusto: inconcluso; sensible al tipo de ajuste (S9, descriptivo) |
| E3_rn50sup | pasa | familia entrenados | inconcluso | inconcluso | menor que δ (equivalente con margen δ) | inconcluso | sensible al tipo de ajuste | dependiente del margen: inconcluso (δ = 0.05, 0.10); menor que δ (equivalente con margen δ) (δ = 0.20); sensible al tipo de ajuste (S9, descriptivo) |
| E4_mae | pasa | familia entrenados | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| F1_rn50rand0 | no es puerta (piso) | familia pisos | inconcluso | inconcluso | inconcluso | inconcluso | estable | robusto: inconcluso |
| F2_pixels | no es puerta (piso) | familia pisos | inconcluso | menor que δ (equivalente con margen δ) | menor que δ (equivalente con margen δ) | inconcluso | estable | dependiente del margen: inconcluso (δ = 0.05); menor que δ (equivalente con margen δ) (δ = 0.10, 0.20) |

## Familias Holm (§7.3; iguales con los tres δ)

- **entrenados:** k = 7 encoders, 14 pruebas por δ; incluidos: O1_dinov2b, R1_cvcl, E1_siglip, O2_openclip, E2_vitsup, E3_rn50sup, E4_mae; excluidos: ninguno.
- **pisos:** k = 2 encoders, 4 pruebas por δ; incluidos: F1_rn50rand0, F2_pixels; excluidos: ninguno.

## Conclusión global por (δ, corte)

| delta | cut | conclusion | detalle | trained_family_size |
|---|---|---|---|---|
| 0.05 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.05 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.05 | 0.7 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.1 | 0.7 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.3 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.5 | G1 | R0 en 'inconcluso' | 14 |
| 0.2 | 0.7 | G1 | R0 en 'inconcluso' | 14 |

**Perfil global:** robusta: G1

## Pisos

| delta | family_size | lectura |
|---|---|---|
| 0.05 | 4 | inconcluso |
| 0.1 | 4 | inconcluso |
| 0.2 | 4 | inconcluso |