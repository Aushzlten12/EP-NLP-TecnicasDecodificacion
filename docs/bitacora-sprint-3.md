# Bitácora Sprint 3: Evaluación de Estrategias de Decodificación

**Inicio:** 2025-10-22 23:00  
**Miembro:** Jose Pachas

## Resumen

- Modelo `DecoderOnlyLM` entrenado exitosamente.
- Se implementó y ejecutó `eval.py` para aplicar y comparar múltiples estrategias de decodificación:
  - Greedy
  - Top-k (k=10, k=50)
  - Top-p (p=0.9, p=0.8)
  - Beam Search (anchura 3 y 5)
- Se añadieron métricas de evaluación:
  - Repetición
  - Diversidad
  - Longitud media
  - Sorpresa media (entropía)

## Resultados

- **Greedy:** alta repetición, baja diversidad. Determinista.
- **Top-k:** mejora ligera en diversidad. Sensible al valor de `k`.
- **Top-p:** mayor diversidad y sorpresa con `p=0.9`, colapso con `p=0.8`.
- **Beam Search:** comportamiento muy repetitivo; se detectó ausencia de cálculo de sorpresa (entropía ≈ 0.0).
- Archivos generados:
  - `out/samples/*.txt`: salidas generadas por estrategia
  - `out/metrics_decode.csv`: resumen cuantitativo
  - `out/tabla_tradeoffs.md`: tabla comparativa

## Próximos pasos

- Corregir `beam_search()` para capturar distribuciones y calcular sorpresa.
- Visualizar los trade-offs calidad/diversidad con gráficas.
- Posible ajuste de penalización por longitud en beam search.

**Fin:** 2025-10-23 12:15
