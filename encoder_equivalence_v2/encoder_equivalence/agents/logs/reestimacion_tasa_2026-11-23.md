# Re-estimación a mano tras el rate check de 15 min (REVISION5 C5), 23-11-2026 09:35 America/Santiago

Rate check (log, 08:55 aprox.): R0 1074, O1 332, R1 466 img/min, todos < 50 % de lo medido el 22-11.
Referencia 22-11 (ACLARACIONES, 128 imágenes de ruido 32x32, 1 proceso, 2 hilos): img/s R0 70, O1 20, R1 41, E1 26, O2 154, E2 30, E3 52, E4 30, F1 51, F2 1626.
Tasas observadas a las 09:30 (3 procesos en paralelo): R0 ~1074, O1 ~343, E1 ~424, E2 ~478 img/min, es decir ~26-29 % de la referencia (~0,27x).

Re-estimación del paso 3, asumiendo ~0,27x para los encoders que faltan (c100train 50.000 + c10pool 2.000 + c10test 10.000; cachés existentes reutilizadas):
- g0: R0 listo (09:24). E1 ~146 min, E3 ~75 min, F2 pocos min -> fin ~13:15
- g1: O1 en 16.384/50.000, ~134 min restantes; O2 ~25 min; E4 ~127 min -> fin ~14:15-14:30
- g2: R1 listo. E2 ~105 min, F1 ~75 min -> fin ~12:35
Fin estimado de embeddings: ~14:30. Pasos 4-8 después; tope de 14 h a las 22:38:29.
Si pasos 4-8 también corren ~0,27x respecto del estimado, H4 podría quedar cerca del tope; se aplica el recorte pre-declarado (riesgo 9) sin cambios.
Nada del congelado se modifica. Desviación a registrar en DESVIACIONES de agentes: tasa < 50 % de la referencia.

## Nota 10:45: pausa del computador
Entre ~10:24 y ~10:43 los procesos no avanzaron y el heartbeat no escribió: el computador se suspendió al quedar sin sesión activa, aunque el keep-awake por shell estaba corriendo. Los procesos siguieron vivos y retomaron solos. Se pierden ~19 min de reloj; los resultados no cambian (embeddings por partes, deterministas). Se mantiene un monitor activo para evitarlo.

## Nota 14:45: reinicio del computador
Hacia las 14:37 el computador se reinició; la corrida se cortó (último log 14:33:29). /workspace quedó intacto (out/emb con 9 encoders completos y 23 partes de E4 c100train), /tmp se borró (renv, XTMEM). A las 14:43:18 se reanudó el mismo comando congelado (`run_agents.sh primary`): salidas existentes y partes se reutilizan, renv se reconstruye y XTMEM se restaura del respaldo en el paso 0; el tope sigue midiéndose desde 08:38:29. Pasos 1-2 (paridad, control 1) se repiten de forma determinista.
