from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _save(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def plot_runtime_by_method(input_path=RESULTS_DIR / "baseline_summary.csv"):
    df = _read_csv(input_path)
    methods = df["method"].drop_duplicates().tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    for method in methods:
        subset = df[df["method"] == method].sort_values("n")
        ax.plot(subset["n"], subset["avg_time"], marker="o", label=method)

    ax.set_title("Runtime Scaling by Method")
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Average runtime (seconds)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    return fig


def plot_gap_by_method(input_path=RESULTS_DIR / "tsplib_summary.csv"):
    df = _read_csv(input_path)
    df = df[df["avg_gap_pct"].notna()].copy()
    pivot = df.pivot(index="method", columns="n", values="avg_gap_pct")

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("TSPLIB Optimality Gap by Method")
    ax.set_xlabel("Method")
    ax.set_ylabel("Average gap (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=35)
    return fig


def plot_scaling(input_path=RESULTS_DIR / "baseline_summary.csv"):
    df = _read_csv(input_path)
    selected = [
        "Random + 2-opt",
        "Nearest Neighbor + 2-opt fast",
        "Greedy Insertion + 2-opt fast",
        "MultiStart 2-opt fast (10)",
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in selected:
        subset = df[df["method"] == method].sort_values("n")
        if subset.empty:
            continue
        ax.plot(subset["n"], subset["avg_time"], marker="o", linewidth=2, label=method)

    ax.set_title("Runtime vs Problem Size")
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Average runtime (seconds)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    return fig


def plot_runtime_vs_gap(input_path=RESULTS_DIR / "tsplib_summary.csv"):
    df = _read_csv(input_path)
    df = df[df["avg_gap_pct"].notna()].copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    for method, subset in df.groupby("method"):
        ax.scatter(subset["avg_time"], subset["avg_gap_pct"], label=method, s=55, alpha=0.8)

    ax.set_title("Runtime vs Gap Tradeoff")
    ax.set_xlabel("Average runtime (seconds)")
    ax.set_ylabel("Average gap (%)")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return fig


def plot_cp_nodes(input_path=RESULTS_DIR / "cp_results.csv"):
    df = _read_csv(input_path)
    grouped = (
        df.groupby(["n", "lower_bound"], as_index=False)["nodes_explored"]
        .mean()
        .sort_values("n")
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for lower_bound, subset in grouped.groupby("lower_bound"):
        ax.plot(
            subset["n"],
            subset["nodes_explored"],
            marker="o",
            linewidth=2,
            label=lower_bound,
        )

    ax.set_title("Branch-and-Bound Nodes Explored")
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Average nodes explored")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Lower bound")
    return fig


def plot_cp_pruning(input_path=RESULTS_DIR / "cp_results.csv"):
    df = _read_csv(input_path)
    grouped = (
        df.groupby(["branch_policy", "lower_bound"], as_index=False)["pruning_rate"]
        .mean()
        .sort_values("pruning_rate")
    )
    labels = [
        f"{row.branch_policy}\n{row.lower_bound}"
        for row in grouped.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, grouped["pruning_rate"], color="#4C78A8")
    ax.set_title("Branch-and-Bound Pruning Effectiveness")
    ax.set_xlabel("Branch policy and lower bound")
    ax.set_ylabel("Average pruned states / nodes explored")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=35)
    return fig


def plot_learned_policy_metric(
    metric,
    ylabel,
    input_path=RESULTS_DIR / "learned_cp_summary.csv",
):
    df = _read_csv(input_path)
    grouped = df.groupby("branch_policy", as_index=False)[metric].mean()
    grouped = grouped.sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grouped["branch_policy"], grouped[metric], color="#59A14F")
    ax.set_title(f"Learned vs Heuristic Branching: {ylabel}")
    ax.set_xlabel("Branch policy")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    return fig


def generate_all_plots():
    _save(plot_runtime_by_method(), RESULTS_DIR / "runtime_plot.png")
    _save(plot_gap_by_method(), RESULTS_DIR / "gap_plot.png")
    _save(plot_scaling(), RESULTS_DIR / "scaling_plot.png")
    _save(plot_runtime_vs_gap(), RESULTS_DIR / "runtime_vs_gap.png")

    cp_results = RESULTS_DIR / "cp_results.csv"
    if cp_results.exists():
        _save(plot_cp_nodes(cp_results), RESULTS_DIR / "cp_nodes_plot.png")
        _save(plot_cp_pruning(cp_results), RESULTS_DIR / "cp_pruning_plot.png")

    learned_summary = RESULTS_DIR / "learned_cp_summary.csv"
    if learned_summary.exists():
        _save(
            plot_learned_policy_metric(
                "avg_runtime",
                "Average runtime (seconds)",
                learned_summary,
            ),
            RESULTS_DIR / "learned_runtime_plot.png",
        )
        _save(
            plot_learned_policy_metric(
                "avg_nodes_explored",
                "Average nodes explored",
                learned_summary,
            ),
            RESULTS_DIR / "learned_nodes_plot.png",
        )
        _save(
            plot_learned_policy_metric(
                "avg_pruning_rate",
                "Average pruning rate",
                learned_summary,
            ),
            RESULTS_DIR / "learned_pruning_plot.png",
        )


if __name__ == "__main__":
    generate_all_plots()
