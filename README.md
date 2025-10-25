# Proyecto CC0c2: Mini-Transformer + módulo complementario

## Descripción

Este proyecto implementa un modelo **Mini-Transformer decoder-only** autoregresivo, basado en la arquitectura Transformer. Está diseñado para la **generación de texto**, evaluando diferentes **estrategias de decodificación** (Greedy, Beam Search, Top-k, Top-p), con un enfoque en la comparación entre **calidad** y **diversidad** de las secuencias generadas.

## Dependencias

- Permitido: Python stdlib, numpy, torch (si preinstalado).
- Fallback: Usar NumPy para atención/MLP si torch no está disponible; justificar en `docs/reporte.md`.

## Uso rápido

```bash
make deps && make build && make data && make verify-corpus && make train && make eval
```

## Técnicas de Decodificación Implementadas

Las siguientes estrategias de decodificación se implementan y evalúan en este modelo:

1. **Greedy**:

   - Selección del token más probable en cada paso.
   - Determinista y rápida, pero con baja diversidad.

2. **Top-k Sampling**:

   - Selección entre los **k tokens más probables**.
   - Controla la aleatoriedad en la generación del texto.

3. **Top-p (Nucleus) Sampling**:

   - Selección de tokens cuyos **probabilidades acumuladas sumen ≥ p**.
   - Más flexible que top-k y adecuado para explorar diversidad.

4. **Beam Search**:
   - Mantiene las mejores **k secuencias** (beams) en cada paso de la generación.
   - Puede colapsar a secuencias muy repetitivas sin penalización por longitud.

### Parámetros Ajustables

Cada técnica de decodificación tiene parámetros ajustables que permiten experimentar con el comportamiento de la generación:

- **Greedy**: No requiere parámetros adicionales.
- **Top-k**: `top_k` (número de tokens a considerar) y `temperature` (ajuste de la distribución de probabilidad).
- **Top-p**: `top_p` (umbral de probabilidad acumulada) y `temperature`.
- **Beam Search**: `beam_width` (ancho del beam) y `length_penalty` (penalización por longitud de la secuencia).

## Entrenamiento del Modelo

El modelo se entrena sobre un corpus tokenizado con **tokenización determinista**. Se utilizan los siguientes pasos:

1. **Generación del Corpus**: Se genera un corpus de ejemplo con palabras `word1`, `word2`, ..., para evaluar el rendimiento del modelo.
2. **Entrenamiento**: El modelo es entrenado utilizando el optimizador **AdamW** y la función de pérdida **Cross-Entropy Loss**.
3. **Guardado del Modelo**: El modelo entrenado se guarda en un archivo `.tar.gz`, que incluye los pesos y los metadatos necesarios para la evaluación.

## Métricas de Evaluación

Durante la evaluación, se calculan las siguientes métricas para cada estrategia de decodificación:

- **Repetición**: Proporción de n-gramas (1-gramas, 2-gramas y 3-gramas) repetidos en la secuencia generada.
- **Diversidad**: Proporción de n-gramas únicos en la secuencia generada.
- **Longitud Promedio**: Número promedio de tokens generados por muestra.
- **Sorpresa Promedio**: Promedio de la entropía de las distribuciones de probabilidad a lo largo de los pasos de generación.

## Resultados Esperados

1. **Greedy**:
   - Alta repetición, baja diversidad.
   - Generación determinista, rápida.
2. **Top-k Sampling**:

   - Mejor balance entre calidad y diversidad.
   - Depende del valor de `k`; valores pequeños producen texto más coherente, pero con menor diversidad.

3. **Top-p (Nucleus) Sampling**:

   - Mayor diversidad en las secuencias generadas.
   - Dependencia del parámetro `top_p`.

4. **Beam Search**:
   - Generación estable, pero con riesgo de generar secuencias repetitivas sin penalización por longitud.
   - El valor de `beam_width` afecta la precisión de la generación.

## Visualizaciones

Se generan las siguientes visualizaciones para comparar las estrategias:

- **Gráfico de Repetición vs Diversidad**: Compara cómo cambian la repetición y la diversidad con distintas estrategias y parámetros.
- **Tabla de Calidad-Diversidad**: Resumen en formato de tabla de las métricas de repetición, diversidad, longitud y sorpresa para cada estrategia de decodificación.
