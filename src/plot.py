import matplotlib.pyplot as plt
import pandas as pd


def generate_plots():
    # Cargar
    metrics_df = pd.read_csv("out/bench.csv")

    # Colores
    colors = {
        "greedy": "blue",
        "topk_10": "orange",
        "topk_50": "green",
        "topp_0.9": "red",
        "topp_0.8": "purple",
        "beam3": "brown",
        "beam5": "pink",
    }

    # Crear gráfico de Repetición vs Diversidad
    plt.figure(figsize=(8, 6))
    for strategy in metrics_df["strategy"].unique():
        subset = metrics_df[metrics_df["strategy"] == strategy]
        plt.scatter(
            subset[
                "strategy"
            ],  # En el benchmark solo tenemos 'strategy' y 'avg_time', por lo que la diversificación es distinta
            subset["avg_time"],  # Mostrar 'avg_time' como ejemplo
            label=strategy,
            s=150,
            c=colors.get(
                strategy, "black"
            ),  # Si no hay color asignado, usar negro por defecto
            edgecolor="black",
            linewidth=1,
        )

    plt.xlabel("Estrategias", fontsize=12)
    plt.ylabel("Tiempo Promedio (segundos)", fontsize=12)
    plt.title("Benchmarking de Estrategias de Decodificación", fontsize=14)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("out/benchmarking_comparison.png", dpi=300)
    plt.show()

    # Gráfico de Repetición vs Longitud Media
    plt.figure(figsize=(8, 6))
    for strategy in metrics_df["strategy"].unique():
        subset = metrics_df[metrics_df["strategy"] == strategy]
        plt.scatter(
            subset["strategy"],  # Nuevamente usamos 'strategy'
            subset["avg_time"],  # Mostrar 'avg_time' para cada estrategia
            label=strategy,
            s=150,
            c=colors.get(strategy, "black"),
            edgecolor="black",
            linewidth=1,
        )

    plt.xlabel("Estrategias", fontsize=12)
    plt.ylabel("Tiempo Promedio (segundos)", fontsize=12)
    plt.title("Comparación de Estrategias vs Tiempo Promedio", fontsize=14)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("out/strategies_vs_time.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    generate_plots()
