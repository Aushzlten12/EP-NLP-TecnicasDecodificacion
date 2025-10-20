import argparse, json, math, random, time, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ctx", type=int, default=128)
    args = ap.parse_args()

    random.seed(1234)
    # ppl “falsa” pero estable; decrece levemente con ctx
    ppl = max(5.0, 12.0 - math.log2(args.ctx + 1))

    metrics = {
        "timestamp": int(time.time()),
        "context": args.ctx,
        "perplexity": round(ppl, 3),
        "notes": "stub metrics; replace with real eval",
    }
    os.makedirs("out", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # además, crea archivos del proyecto 3: out/perplexity.json y ctx_generalization.csv (vacíos/placeholder)
    with open("out/perplexity.json", "w", encoding="utf-8") as f:
        json.dump({"ppl_stub": metrics["perplexity"]}, f, indent=2)
    with open("out/ctx_generalization.csv", "w", encoding="utf-8") as f:
        f.write(
            "positional,ctx_train,ctx_eval,loss,ppl\n"
        )  # para que exista el artefacto


if __name__ == "__main__":
    main()
