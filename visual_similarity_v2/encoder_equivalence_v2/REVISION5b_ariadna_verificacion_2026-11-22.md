# REVISION5b: verificación rápida de REVISION5 aplicada (anexo de agentes)

**Contexto.** Ariadna, 22-11-2026, ~16:50–17:05 (hora de Chile, UTC−3). Responde al aviso de Juanito de las 16:46.
- No congelé ni corrí nada real. No volví a correr el e2e.
- Solo corrí comandos de lectura: `15 --header`, `15 --list` y `09 --verify`.
- Este archivo va aparte para que no cambie el sha de REVISION5, que figura en la lista del manifiesto.

## Veredicto: **OK para congelar**, con una observación menor que conviene resolver antes
Para el congelado, usar `--scripts-review /workspace/lexical-consensus-openmind/REVISION5b_ariadna_verificacion_2026-11-22.md`.

## 1. Hashes
Los 10 archivos nuevos coinciden con los prefijos que informó Juanito: 10, 11, 12, 14, 15, `run_agents.sh`, `test_embed`, ACLARACIONES, README y el parche. Los que debían quedar igual, quedaron igual: 13, 14b, lib y los 5 tests. Ninguno cambió aparte de esos.

## 2. Implementación
Comparé con la versión que revisé: diff completo de `14` y búsqueda de los cambios en los demás.

| Punto | Estado |
|---|---|
| b2 | `random_c1_by_tier` se reporta y no se exige |
| b3 | Cuadrantes con la mediana de los 4.950 pares; `quadrant_e100` y su letra quedan descriptivos |
| b4 | Columna `rotulo_*` con "Apoyo con IC bajo SESOI"; la región no cambia |
| b6 | `region_hc3_*` y `hc3_cambia_region` en D |
| b9 | ARI descriptivo por corte |
| b13 | Encabezado GA0 en el informe cuando fallan los controles |
| b14 | Nota de p anticonservadores en D-todos |
| b16 | Tabla con el RE de H4 por encoder |
| C1 | `--header` incluye el sha `9744d7cf…`, torch 2.14.1 y transformers 5.20.2, la revisión `ed25f3a3` (con la aclaración de que `perception.py` no la fija) y mi verificación con 8,9e−8 y 5,6e−8 |
| C2 | El parche está marcado "OBSOLETO / NO APLICADO" |
| C3 | Comentarios de las constantes corregidos (b5–b10); la línea de r04 en `run_agents.sh` corregida; `test_embed` imprime SKIP |
| C4 | Línea de §3.7 en el informe |
| C5 | Declaración en `--header` y en `run_agents.sh`, con tope de 14 h, control de tasa a los 15 min y respaldos fuera de `/tmp`. La cota de ~8–9 h está en ACLARACIONES |
| (b) | Declarado en `--header`: las decisiones se tomaron después del run humano y antes de cualquier resultado de agentes |

**Observación menor:** el corte de H4 por tope.
- Si se supera el tope de 14 h, `run_agents.sh` no corre H4 para ese encoder. Como falta su `h4_*.csv`, el encoder queda en "tecnico: falta H4".
- Por la regla b12, eso lo saca **entero** de la familia C. Así, sus H1 y H2 dejan de poder disparar GA1.
- El resultado es que un corte por cómputo cambia quién entra al Holm confirmatorio. Lo mismo pasa si un GLMM de H4 falla.
- **Arreglo de una línea:** si H4 falta o es técnico, sacar solo la clave H4 de la familia C, que queda de 30, en vez de excluir al encoder. Como alternativa, declararlo así en ACLARACIONES.
- La probabilidad es baja (lo esperado son 3,5–7 h). No es bloqueante.

## 3. Logs y verificaciones
- `/workspace/recovery_22nov/rev5/e2e.log`: "ALL E2E CHECKS PASSED" (16:40). `embed.log`: "ALL EMBED TESTS PASSED".
- `09_freeze.py --verify` (lo corrí yo): "VERIFY 47 files: OK". Los únicos archivos fuera del manifiesto son los `*_s6common.npz` de S6, como se esperaba.
- `15 --list` (lo corrí yo): 99 OK y 0 MISSING, igual que `list.txt`.
- `FREEZE_agentes_v2.txt` no existe.
