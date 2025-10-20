# Bitácora Sprint 1: Setup y Mini-Transformer

**Inicio:** [YYYY-MM-DD HH:MM] **Equipo/Miembro:** [Nombres]

## Comandos

- [YYYY-MM-DD HH:MM] `make data` -> Corpus (SHA256 en `out/corpus_sha256.txt`)
- [YYYY-MM-DD HH:MM] `pytest tests/test_transformer.py` -> 5/5 (cov=75%)

## AAA/RGR (ejemplo)

- **Arrange**: secuencia=128, máscara causal ON
- **Act**: aplicar atención
- **Assert**: logits futuras anuladas => (falló) índice corregido => (verde)

## Métricas

- Perplexity baseline: 10.3 (RoPE)

**Fin:** [YYYY-MM-DD HH:MM]
