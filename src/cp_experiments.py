import csv
import pickle
from pathlib import Path

from exact_solver import solve_tsp_branch_and_bound
from tsp_utils import compute_distance_matrix, generate_euclidean_instance


def make_row(n, seed, branch_policy, lower_bound, incumbent, cost, stats):
    return {
        "n": n,
        "seed": seed,
        "branch_policy": branch_policy,
        "lower_bound": lower_bound,
        "incumbent_heuristic": incumbent,
        "optimal_cost": cost,
        "runtime": stats["runtime"],
        "nodes_explored": stats["nodes_explored"],
        "pruned_states": stats["pruned_states"],
        "pruning_rate": stats["pruning_rate"],
        "proof_time": stats["proof_time"],
        "optimality_proven": stats["optimality_proven"],
        "time_limit_hit": stats["time_limit_hit"],
        "initial_incumbent": stats["initial_incumbent"],
        "incumbent_updates": stats["incumbent_updates"],
    }


def save_rows(rows, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "n",
        "seed",
        "branch_policy",
        "lower_bound",
        "incumbent_heuristic",
        "optimal_cost",
        "runtime",
        "nodes_explored",
        "pruned_states",
        "pruning_rate",
        "proof_time",
        "optimality_proven",
        "time_limit_hit",
        "initial_incumbent",
        "incumbent_updates",
    ]

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(rows, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    grouped = {}

    for row in rows:
        key = (
            row["n"],
            row["branch_policy"],
            row["lower_bound"],
            row["incumbent_heuristic"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []

    for (n, branch_policy, lower_bound, incumbent), group in sorted(grouped.items()):
        count = len(group)
        avg_runtime = sum(r["runtime"] for r in group) / count
        avg_nodes = sum(r["nodes_explored"] for r in group) / count
        avg_pruned = sum(r["pruned_states"] for r in group) / count
        avg_pruning_rate = sum(r["pruning_rate"] for r in group) / count
        proof_rate = sum(1 for r in group if r["optimality_proven"]) / count
        time_limit_rate = sum(1 for r in group if r["time_limit_hit"]) / count
        avg_incumbent_updates = sum(r["incumbent_updates"] for r in group) / count

        summary_rows.append({
            "n": n,
            "branch_policy": branch_policy,
            "lower_bound": lower_bound,
            "incumbent_heuristic": incumbent,
            "num_runs": count,
            "avg_runtime": avg_runtime,
            "avg_nodes_explored": avg_nodes,
            "avg_pruned_states": avg_pruned,
            "avg_pruning_rate": avg_pruning_rate,
            "proof_rate": proof_rate,
            "time_limit_rate": time_limit_rate,
            "avg_incumbent_updates": avg_incumbent_updates,
        })

    fieldnames = [
        "n",
        "branch_policy",
        "lower_bound",
        "incumbent_heuristic",
        "num_runs",
        "avg_runtime",
        "avg_nodes_explored",
        "avg_pruned_states",
        "avg_pruning_rate",
        "proof_rate",
        "time_limit_rate",
        "avg_incumbent_updates",
    ]

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved CP summary to: {filepath}")


def run_cp_experiments(
    sizes=(8, 10, 12),
    seeds=(0, 1, 2),
    branch_policies=("lexicographic", "nearest", "greedy", "random"),
    lower_bounds=("trivial", "mst"),
    incumbents=("nn_2opt_fast", "greedy_2opt_fast", "multistart_2opt_fast"),
    time_limit=3.0,
    output_path="results/cp_results.csv",
    summary_path="results/cp_summary.csv",
    learned_model_path=None,
):
    rows = []
    learned_model = None
    if learned_model_path is not None:
        with Path(learned_model_path).open("rb") as f:
            learned_model = pickle.load(f)

    for n in sizes:
        for seed in seeds:
            points = generate_euclidean_instance(n=n, seed=seed)
            dist = compute_distance_matrix(points)

            for lower_bound in lower_bounds:
                for branch_policy in branch_policies:
                    for incumbent in incumbents:
                        print(
                            f"CP n={n} seed={seed} "
                            f"bound={lower_bound} branch={branch_policy} "
                            f"incumbent={incumbent}"
                        )
                        _, cost, stats = solve_tsp_branch_and_bound(
                            dist,
                            branch_policy=branch_policy,
                            lower_bound=lower_bound,
                            incumbent_heuristic=incumbent,
                            seed=seed,
                            time_limit=time_limit,
                            model=learned_model if branch_policy == "learned" else None,
                        )
                        rows.append(make_row(
                            n,
                            seed,
                            branch_policy,
                            lower_bound,
                            incumbent,
                            cost,
                            stats,
                        ))
                        save_rows(rows, output_path)

    save_rows(rows, output_path)
    save_summary(rows, summary_path)
    print(f"Saved CP results to: {output_path}")
    return rows


def run_learned_policy_experiments(
    model_path="models/logistic_branch.pkl",
    output_path="results/learned_cp_results.csv",
    summary_path="results/learned_cp_summary.csv",
):
    return run_cp_experiments(
        sizes=(8, 10, 12),
        seeds=(100, 101, 102),
        branch_policies=("lexicographic", "random", "nearest", "greedy", "learned"),
        lower_bounds=("mst",),
        incumbents=("nn_2opt_fast",),
        time_limit=5.0,
        output_path=output_path,
        summary_path=summary_path,
        learned_model_path=model_path,
    )


if __name__ == "__main__":
    run_cp_experiments(
        sizes=(8, 10, 12),
        seeds=(0, 1, 2),
        branch_policies=("lexicographic", "nearest", "greedy", "random"),
        lower_bounds=("trivial", "mst"),
        incumbents=("nn_2opt_fast", "greedy_2opt_fast", "multistart_2opt_fast"),
        time_limit=3.0,
        output_path="results/cp_results.csv",
        summary_path="results/cp_summary.csv",
    )
