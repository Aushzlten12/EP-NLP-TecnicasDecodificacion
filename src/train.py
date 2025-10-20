import argparse, tarfile, io, json, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    meta = {
        "lr": args.lr,
        "heads": args.heads,
        "dim": args.dim,
        "timestamp": int(time.time()),
        "notes": "stub model; replace with real training later",
    }

    # empaqueta un checkpoint “vacío” + metadatos
    with tarfile.open(args.output, "w:gz") as tar:
        info = tarfile.TarInfo("checkpoint/meta.json")
        data = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


if __name__ == "__main__":
    main()
