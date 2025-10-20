# Reporte final

## Introducción

Mini-Transformer (decoder-only) con RoPE; módulo complementario: [indicar]. Ablación: RoPE vs sinusoidal.

## Métricas

- Perplexity (3 rep): 8.5 ± 0.3 (RoPE) vs 9.2 ± 0.4 (sinusoidal)
- Latencia (ctx=512, 3 rep, warmup=1): 150 ms ± 10 (ver `out/bench.csv`)

## Ablaciones

- RoPE > sinusoidal en extrapolación (ctx>train)
- Top-p (0.9) vs Beam (5): distinct-2 ↑, exact-match ↓

## Gráficos

![Latencia](out/plot_latencia.png)

## Conclusiones

[Hallazgos, limitaciones, próximos pasos]
