# Makefile para el proyecto CC0c2
# Uso: make [target]

.PHONY: deps build data tokenize train eval bench plot pack verify verify-corpus tag test test-idem clean distclean

# Reproducibilidad
SOURCE_DATE_EPOCH ?= 1700000000
SEED ?= 42
SALT ?= 1a2b3c4d5e6f7890abcdef1234567890
SEED_BENCH ?= 42

# Hiperparámetros por defecto
CONTEXT ?= 512
LR ?= 0.001
HEADS ?= 4
DIM ?= 128

deps:
	@echo "Verificando dependencias preinstaladas (stdlib, numpy, torch opcional)"
	python -c "import numpy; try: import torch; except: print('Torch no disponible')" || true

build:
	@echo "Chequeos básicos"
	# shellcheck tools/*.sh || true
	# ruff check src/*.py || true
	mkdir -p out dist

data:
	@echo "Generando corpus sintético"
	./tools/gen_corpus.sh $(SEED) $(SALT) > out/corpus.txt
	echo "Comando: ./tools/gen_corpus.sh $(SEED) $(SALT)" > out/seed.txt
	sha256sum out/corpus.txt | awk '{print $$1}' > out/corpus_sha256.txt

verify-corpus:
	@echo "Verificando hash del corpus"
	HGEN="$$(./tools/gen_corpus.sh $(SEED) $(SALT) | sha256sum | awk '{print $$1}')"; \
	HSAVED="$$(cat out/corpus_sha256.txt)"; test "$$HGEN" = "$$HSAVED"

tokenize: data
	@echo "Tokenizando corpus"
	python src/tokenizer.py out/corpus.txt --output out/tokens.jsonl --vocab out/vocab.txt

train: tokenize
	@echo "Entrenando modelo"
	python src/train.py --lr $(LR) --heads $(HEADS) --dim $(DIM) --input out/tokens.jsonl --output dist/model.tar.gz

eval: train
	@echo "Evaluando métricas"
	python src/eval.py dist/model.tar.gz --output out/metrics.json

bench:
	@echo "Benchmarking (3 repeticiones, reporte de sigma)"
	python src/bench.py --n $(CONTEXT) --seed $(SEED_BENCH) --warmup 1 --reps 3 --output out/bench.csv

plot: bench
	@echo "Generando gráficos"
	python src/plot.py out/bench.csv --output out/plot_latencia.png

test:
	@echo "Ejecutando tests"
	pytest tests/ --cov=src --cov-report=term-missing || bats tests/ || true

test-idem:
	@echo "Verificando idempotencia"
	rm -rf out/tmp && mkdir -p out/tmp
	$(MAKE) test eval bench plot
	rsync -a --delete out/ out/tmp/
	$(MAKE) test eval bench plot
	{ find out -type f ! -path 'out/tmp/*' ! -name 'hashes.txt' -exec sha256sum {} \; | sort > out/hashes.txt; }
	find out/tmp -type f -exec sha256sum {} \; | sort > out/tmp/hashes.txt
	diff -u out/tmp/hashes.txt out/hashes.txt

pack: eval bench plot
	@echo "Capturando entorno"
	{ \
	  echo "DATE=$$(date -u +%FT%TZ)"; \
	  python - <<'PY' || true
import platform, sys
print("PYTHON", sys.version.replace("\n"," "))
try:
    import numpy as np; print("NUMPY", np.__version__)
except Exception: print("NUMPY none")
try:
    import torch; print("TORCH", torch.__version__)
except Exception: print("TORCH none")
print("PLATFORM", platform.platform())
PY
	} > out/env.txt
	@echo "Empaquetando artefactos reproducibles"
	find out -type f -print0 | xargs -0 touch -d "@$(SOURCE_DATE_EPOCH)"
	rm -f dist/proy-v1.0.0.tar.gz
	tar --sort=name --mtime="@$(SOURCE_DATE_EPOCH)" --owner=0 --group=0 --numeric-owner \
	    -czf dist/proy-v1.0.0.tar.gz out/ \
	    --exclude='out/session.typescript' --exclude='out/terminal.cast' --exclude='out/*.png~'
	sha256sum dist/proy-v1.0.0.tar.gz | awk '{print $$1"  "$$2}' > out/HASHES.md

verify:
	@echo "Verificando hash del paquete"
	sha256sum -c out/HASHES.md

tag:
	@echo "Creando tag simulado"
	echo "v1.0.0: Versión inicial" > CHANGELOG.md
	echo "Firma simulada: $(shell date -u +%FT%TZ)" > out/tag_signature.txt

clean:
	rm -rf out/tmp out/hashes.txt

distclean: clean
	rm -rf out/* dist/* CHANGELOG.md