import argparse, json, sys, hashlib, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", required=True)  # out/tokens.jsonl
    ap.add_argument("--vocab", required=True)  # out/vocab.txt
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    toks = txt.split()

    # Vocab
    vocab = sorted(set(toks))
    specials = ["<pad>", "<bos>", "<eos>"]
    vocab = specials + vocab

    # Vocab
    with open(args.vocab, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(vocab) + "\n")

    # Mapa a IDs
    stoi = {w: i for i, w in enumerate(vocab)}

    # JSON determinista
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for i in range(0, len(toks), args.seq_len):
            seq = toks[i : i + args.seq_len]
            ids = [stoi[t] for t in seq]
            rec = {"tokens": seq, "ids": ids}
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    # Metadatos
    meta = {
        "input_sha256": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
        "seq_len": args.seq_len,
        "ntokens": len(toks),
        "nvocab": len(vocab),
    }
    os.makedirs("out", exist_ok=True)
    with open("out/tokenizer_meta.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
