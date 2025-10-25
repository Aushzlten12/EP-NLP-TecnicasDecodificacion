import argparse
import time
import torch
import json
import tarfile
from decoder import build_decoder_only, generate, beam_search


def load_model_from_tar(path, vocab_file):
    with tarfile.open(path, "r:gz") as tar:
        meta = json.load(tar.extractfile("checkpoint/meta.json"))

        state_dict = torch.load(
            tar.extractfile("checkpoint/model.pt"), map_location="cpu"
        )

    # cargar vocab
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]

    # Construir el modelo
    model = build_decoder_only(
        vocab_size=meta["vocab_size"],
        max_len=meta["seq_len"],
        d_model=meta["dim"],
        n_heads=meta["heads"],
        n_layers=2,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab


# Función para calcular el tiempo promedio para cada estrategia
def benchmark(model, prompt, max_tokens, strategy, n_reps=3):
    times = []
    for _ in range(n_reps):
        start_time = time.time()
        if strategy == "greedy":
            generate(model, prompt, max_new_tokens=max_tokens, greedy=True)
        elif strategy == "topk_10":
            generate(
                model, prompt, max_new_tokens=max_tokens, top_k=10, temperature=1.0
            )
        elif strategy == "topk_50":
            generate(
                model, prompt, max_new_tokens=max_tokens, top_k=50, temperature=0.8
            )
        elif strategy == "topp_0.9":
            generate(
                model, prompt, max_new_tokens=max_tokens, top_p=0.9, temperature=1.0
            )
        elif strategy == "topp_0.8":
            generate(
                model, prompt, max_new_tokens=max_tokens, top_p=0.8, temperature=0.7
            )
        elif strategy.startswith("beam"):
            beam_width = int(strategy[4:])  # beam3 o beam5
            beam_search(model, prompt, max_new_tokens=max_tokens, beam_width=beam_width)
        end_time = time.time()
        times.append(end_time - start_time)
    return sum(times) / len(times)


def main():
    parser = argparse.ArgumentParser()
    # El modelo
    model_path = "dist/model.tar.gz"
    vocab_file = "out/vocab.txt"

    parser.add_argument(
        "--n", type=int, default=10, help="Número de repeticiones para el benchmark"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=30,
        help="Máximo de tokens por secuencia generada",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Semilla para la aleatoriedad"
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="Número de iteraciones para el warmup"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Número de repeticiones para cada estrategia",
    )
    parser.add_argument(
        "--output", required=True, help="Ruta de salida para los resultados"
    )
    args = parser.parse_args()

    # Cargar el modelo usando la ruta estática y vocabulario
    model, vocab = load_model_from_tar(model_path, vocab_file)

    # Definir el token de inicio <bos>
    stoi = {word: idx for idx, word in enumerate(vocab)}
    prompt = torch.tensor([[stoi["<bos>"]]], dtype=torch.long)

    # Estrategias
    strategies = [
        "greedy",
        "topk_10",
        "topk_50",
        "topp_0.9",
        "topp_0.8",
        "beam3",
        "beam5",
    ]

    # Ejecutar benchmark
    benchmark_results = {}
    for strategy in strategies:
        avg_time = benchmark(model, prompt, args.max_tokens, strategy, n_reps=args.reps)
        benchmark_results[strategy] = avg_time

    # Guardar
    with open(args.output, "w") as f:
        f.write("strategy,avg_time\n")
        for strategy, avg_time in benchmark_results.items():
            f.write(f"{strategy},{avg_time:.4f}\n")

    print(f"Benchmarking completado. Resultados guardados en {args.output}.")


if __name__ == "__main__":
    main()
