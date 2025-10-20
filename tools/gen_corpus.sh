#!/bin/bash
set -euo pipefail
SEED=${1:-}; SALT=${2:-}
if [ -z "${SEED}" ] || [ -z "${SALT}" ]; then
  echo "Uso: $0 <SEED-decimal> <SALT-hex>" >&2; exit 1
fi

python - <<'PY' "$SEED" "$SALT"
import hashlib, sys, random
seed, salt = sys.argv[1], sys.argv[2]
h = hashlib.sha256(f"{seed}-{salt}".encode()).hexdigest()
random.seed(int(h[:16],16))  # 64 bits
N=50000
print(' '.join(f"word{random.randint(1,1000)}" for _ in range(N)))
PY