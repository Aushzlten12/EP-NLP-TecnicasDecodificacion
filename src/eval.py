import argparse, json, tarfile, os
import torch
import torch.nn.functional as F
from decoder import build_decoder_only, generate, beam_search
import matplotlib.pyplot as plt
import pandas as pd


def load_model_from_tar(path):
    with tarfile.open(path, "r:gz") as tar:
        meta = json.load(tar.extractfile("checkpoint/meta.json"))
        state_dict = torch.load(
            tar.extractfile("checkpoint/model.pt"), map_location="cpu"
        )

    model = build_decoder_only(
        vocab_size=meta["vocab_size"],
        max_len=meta["seq_len"],
        d_model=meta["dim"],
        n_heads=meta["heads"],
        n_layers=2,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta


def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()


def compute_metrics(samples, logits_per_step=None):
    ngrams_total = 0
    ngrams_unique = set()
    repeated = 0
    lengths = []
    surprises = []

    for i, seq in enumerate(samples):
        tokens = tuple(seq)
        lengths.append(len(tokens))
        seen = set()
        for n in range(1, 4):  # Para 1-gramas, 2-gramas y 3-gramas
            for i in range(len(tokens) - n + 1):
                ngram = tokens[i : i + n]
                ngrams_total += 1
                if ngram in seen:
                    repeated += 1
                seen.add(ngram)
                ngrams_unique.add(ngram)

    if logits_per_step:
        for probs in logits_per_step:
            surprises.append(entropy(probs))  # Calcula la entropía

    return {
        "repetition": repeated / ngrams_total if ngrams_total else 0.0,
        "diversity": len(ngrams_unique) / ngrams_total if ngrams_total else 0.0,
        "length_avg": sum(lengths) / len(lengths),
        "surprise_avg": sum(surprises) / len(surprises) if surprises else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Ruta al modelo .tar.gz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs("out/samples", exist_ok=True)

    model, meta = load_model_from_tar(args.model)

    with open("out/vocab.txt", encoding="utf-8") as f:
        itos = [line.strip() for line in f]
    stoi = {w: i for i, w in enumerate(itos)}

    strategies = {
        "greedy": {"greedy": True},
        "topk_10": {"top_k": 10, "temperature": 1.0},
        "topk_50": {"top_k": 50, "temperature": 0.8},
        "topp_0.9": {"top_p": 0.9, "temperature": 1.0},
        "topp_0.8": {"top_p": 0.8, "temperature": 0.7},
    }

    all_metrics = {}

    for name, config in strategies.items():
        samples = []
        for _ in range(args.samples):
            prompt = torch.tensor([[stoi["<bos>"]]], dtype=torch.long)
            out, step_probs = generate(
                model,
                prompt,
                max_new_tokens=args.max_tokens,
                return_probs=True,
                **config,
            )
            samples.append(out[0].tolist()[1:])

        decoded = [" ".join(itos[t] for t in seq) for seq in samples]
        with open(f"out/samples/{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(decoded))

        metrics = compute_metrics(samples, logits_per_step=step_probs)
        metrics["strategy"] = name
        all_metrics[name] = metrics

    for beam_width in [3, 5]:
        name = f"beam{beam_width}"
        samples = []
        for _ in range(args.samples):
            prompt = torch.tensor([[stoi["<bos>"]]], dtype=torch.long)
            out = beam_search(
                model, prompt, max_new_tokens=args.max_tokens, beam_width=beam_width
            )
            samples.append(out[0].tolist()[1:])

        decoded = [" ".join(itos[t] for t in seq) for seq in samples]
        with open(f"out/samples/{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(decoded))

        metrics = compute_metrics(samples)
        metrics["strategy"] = name
        all_metrics[name] = metrics

    # Guardar JSON requerido por el Makefile
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # CSV
    with open("out/metrics_decode.csv", "w", encoding="utf-8") as f:
        f.write("strategy,repetition,diversity,length_avg,surprise_avg\n")
        for m in all_metrics.values():
            f.write(
                f"{m['strategy']},{m['repetition']:.4f},{m['diversity']:.4f},{m['length_avg']:.1f},{m['surprise_avg']:.4f}\n"
            )

    # Markdown
    with open("out/tabla_tradeoffs.md", "w", encoding="utf-8") as f:
        f.write("| Estrategia | Repetición | Diversidad | Longitud | Sorpresa |\n")
        f.write("|------------|------------|------------|----------|----------|\n")
        for m in all_metrics.values():
            f.write(
                f"| {m['strategy']} | {m['repetition']:.2f} | {m['diversity']:.2f} | {m['length_avg']:.1f} | {m['surprise_avg']:.2f} |\n"
            )

    metrics_df = pd.read_csv("out/metrics_decode.csv")

    # Colores personalizados para las estrategias
    colors = {
        "greedy": "blue",
        "topk_10": "orange",
        "topk_50": "green",
        "topp_0.9": "red",
        "topp_0.8": "purple",
        "beam3": "brown",
        "beam5": "pink",
    }

    # Crear figura
    plt.figure(figsize=(8, 6))

    # Crear puntos en el gráfico con mayor tamaño y borde
    for strategy in metrics_df["strategy"].unique():
        subset = metrics_df[metrics_df["strategy"] == strategy]
        plt.scatter(
            subset["diversity"],
            subset["repetition"],
            label=strategy,
            s=150,  # Aumentar tamaño de los puntos
            c=colors[strategy],  # Asignar color según la estrategia
            edgecolor="black",  # Bordes para mayor contraste
            linewidth=1,  # Borde más grueso
        )

    # Etiquetas y título
    plt.xlabel("Diversidad", fontsize=12)
    plt.ylabel("Repetición", fontsize=12)
    plt.title("Repetición vs Diversidad por Estrategia", fontsize=14)

    # Agregar la leyenda
    plt.legend(fontsize=10, loc="upper left")

    # Mostrar la cuadrícula
    plt.grid(True, linestyle="--", alpha=0.5)

    # Guardar y mostrar gráfico
    plt.savefig("out/repetition_vs_diversity.png", dpi=300)
    plt.show()

    # Gráfico de Repetición vs Longitud Media
    plt.figure(figsize=(8, 6))

    for strategy in metrics_df["strategy"].unique():
        subset = metrics_df[metrics_df["strategy"] == strategy]
        plt.scatter(
            subset["length_avg"],
            subset["repetition"],
            label=strategy,
            s=150,
            c=colors[strategy],
            edgecolor="black",
            linewidth=1,
        )

    plt.xlabel("Longitud Promedio", fontsize=12)
    plt.ylabel("Repetición", fontsize=12)
    plt.title("Repetición vs Longitud Media por Estrategia", fontsize=14)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("out/repetition_vs_length_avg.png", dpi=300)
    plt.show()

    # Gráfico de Diversidad vs Sorpresa Media
    plt.figure(figsize=(8, 6))

    for strategy in metrics_df["strategy"].unique():
        subset = metrics_df[metrics_df["strategy"] == strategy]
        plt.scatter(
            subset["diversity"],
            subset["surprise_avg"],
            label=strategy,
            s=150,
            c=colors[strategy],
            edgecolor="black",
            linewidth=1,
        )

    plt.xlabel("Diversidad", fontsize=12)
    plt.ylabel("Sorpresa Media", fontsize=12)
    plt.title("Diversidad vs Sorpresa Media por Estrategia", fontsize=14)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("out/diversity_vs_surprise_avg.png", dpi=300)
    plt.show()

    # Gráfico de Longitud Media vs Sorpresa Media
    plt.figure(figsize=(8, 6))

    for strategy in metrics_df["strategy"].unique():
        subset = metrics_df[metrics_df["strategy"] == strategy]
        plt.scatter(
            subset["length_avg"],
            subset["surprise_avg"],
            label=strategy,
            s=150,
            c=colors[strategy],
            edgecolor="black",
            linewidth=1,
        )

    plt.xlabel("Longitud Promedio", fontsize=12)
    plt.ylabel("Sorpresa Media", fontsize=12)
    plt.title("Longitud Media vs Sorpresa Media por Estrategia", fontsize=14)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("out/length_avg_vs_surprise_avg.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
