# Nota: checksum de `DIFF_agentes_propuesta.patch` frente al manifiesto de agentes

Fecha de la nota: 23-11-2026. El manifiesto `FREEZE_agentes_v2.txt` no se modifica.

- `FREEZE_agentes_v2.txt` (congelado el 22-11-2026 a las 18:00:36, hora de Santiago), línea 31, espera:
  `21125d6a91ed5f8a3d28a8fbba56455eddb222a7075462a1f90182470a3cf87d  encoder_equivalence/DIFF_agentes_propuesta.patch`
- El archivo publicado en este repositorio tiene sha256
  `79ed554c5e3df16f1ab5513a1f8f22981d8d0115ee84cd1a8f0039a91de612f0`.
  Es la versión subida en el commit `23e4082` (22-11-2026, 10:29:33 −03), antes del congelamiento, y se mantuvo igual en la consolidación (`2f447a1`) y en `a945ed2`.
- La copia que corresponde al manifiesto (sha256 `21125d6a…f87d`, 14.008 bytes) es la versión local del 22-11-2026, 15:58 (hora de Santiago). Difiere de la publicada solo en una cabecera de 5 líneas agregada al inicio; sin esas 5 líneas (`tail -n +6`) su sha256 es exactamente `79ed554c…12f0`. La diferencia es anterior a la consolidación en `main`.

Las 5 líneas de cabecera, textuales:

```
OBSOLETO / NO APLICADO (REVISION5 C2, 22-11-2026). Este parche se generó sobre propuesta_comparacion_encoders_v2_2026-11-18.md
en 4ff4b341 y NO se aplicó a ningún archivo.
Varias frases quedaron superadas: p. ej. "invariantes al orden" (la propuesta de agentes vigente dice lo contrario: "no se afirma que los aprendices sean invariantes al orden"). El texto vigente es
propuesta_agentes_multi_encoder_v2_2026-11-18.md más agents/ACLARACIONES_scripts.md. Se conserva solo como registro histórico.
-----------------------------------------------------------------------------------------------------------------------------
```

El parche nunca se aplicó a ningún archivo (lo dice la propia cabecera), y ningún script de análisis lo usa: solo `agents/scripts/15_freeze_agents.py` lo incluye en la lista de archivos del manifiesto. La diferencia no afecta ningún resultado. La cabecera corresponde a la corrección C2 registrada en `agents/ACLARACIONES_scripts.md` (línea 207). Se deja registrada para que la verificación del manifiesto pueda explicarse. El texto de esta diferencia también está en la tabla de desviaciones de la versión 2 del preprint.
