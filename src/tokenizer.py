import argparse, json, re, sys, hashlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", required=True)
    ap.add_argument("--vocab", required=True)
    args = ap.parse_args()

    txt = open(args.input, "r", encoding="utf-8").read()
    # tokenizer simple por espacios (stub); determinista
    toks = txt.strip().split()

    # vocabulario mínimo (top-N palabras)
    vocab = sorted(set(toks))
    with open(args.vocab, "w", encoding="utf-8") as f:
        for v in vocab:
            f.write(v + "\n")

    # emite JSONL de tokens (uno por "línea" lógica)
    with open(args.output, "w", encoding="utf-8") as f:
        # corta en bloques de 64 para simular "secuencias"
        for i in range(0, len(toks), 64):
            seq = toks[i : i + 64]
            f.write(json.dumps({"tokens": seq}) + "\n")


if __name__ == "__main__":
    main()
