# Bitácora Sprint 1: Setup y Mini-Transformer

**Inicio:** 2025-10-20 22:25 **Equipo/Miembro:** Jose Pachas

## Comandos

- [2025-10-20 22:25] `make data` -> Corpus generado correctamente (`out/corpus.txt`, `out/seed.txt`, `out/corpus_sha256.txt`)
- [2025-10-20 22:27] `make verify-corpus` -> Hash verificado ✅ (`HGEN == HSAVED`)
- [2025-10-20 22:30] `make tokenize` -> Tokenización estable (`out/tokens.jsonl`, `out/vocab.txt`, `out/tokenizer_meta.json`)

## AAA/RGR (ejemplo)

- **Arrange**: secuencia=64, máscara causal pendiente de prueba
- **Act**: aplicar tokenización determinista
- **Assert**: hashes idénticos tras segunda ejecución (`make test-idem`) => (verde)

## Métricas

- Corpus reproducible (hash estable)
- Tokenizer determinista y cache por timestamps funcional
- Próximo paso: integrar bloque Mini-Transformer

**Fin:** 2025-10-20 22:45
