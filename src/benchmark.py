import time
import csv
from pathlib import Path
from collections import defaultdict

from tsp_utils import generate_euclidean_instance, compute_distance_matrix, load_tsplib_coordinates
from baselines import (
    random_tour,
    nearest_neighbor,
    greedy_insertion,
    multi_start_greedy_insertion,
    multi_start_nearest_neighbor,
    two_opt,
    two_opt_fast,
    # multi_start_two_opt,
    multi_start_two_opt_fast,
    simulated_annealing,
    tour_length,
)

TSPLIB_OPTIMAL_LENGTHS = {
    "eil51": 426,
    "berlin52": 7542,
    "kroA100": 21282,
}

def run_tsplib_benchmark(filepaths, num_starts=10):
    all_results = []

    for filepath in filepaths:
        instance_name = Path(filepath).stem
        optimal = TSPLIB_OPTIMAL_LENGTHS.get(instance_name)
        print(f"\n========== TSPLIB: {instance_name} ==========")

        points = load_tsplib_coordinates(filepath)
        dist = compute_distance_matrix(points)
        n = len(points)
        seed = 0

        rand, rand_time = time_solver(lambda: random_tour(n, seed=seed))
        all_results.append(make_row(instance_name, seed, "Random", tour_length(rand, dist), rand_time, optimal=optimal))

        rand_2opt_fast, rand_2opt_fast_time = time_solver(lambda: two_opt_fast(rand, dist))
        all_results.append(make_row(instance_name, seed, "Random + 2-opt fast", tour_length(rand_2opt_fast, dist), rand_2opt_fast_time, optimal=optimal))

        nn, nn_time = time_solver(lambda: nearest_neighbor(dist))
        all_results.append(make_row(instance_name, seed, "Nearest Neighbor", tour_length(nn, dist), nn_time, optimal=optimal))

        nn_2opt_fast, nn_2opt_fast_time = time_solver(lambda: two_opt_fast(nn, dist))
        all_results.append(make_row(instance_name, seed, "Nearest Neighbor + 2-opt fast", tour_length(nn_2opt_fast, dist), nn_2opt_fast_time, optimal=optimal))

        (_, ms_nn_len), ms_nn_time = time_solver(lambda: multi_start_nearest_neighbor(dist))
        all_results.append(make_row(instance_name, seed, "MultiStart Nearest Neighbor", ms_nn_len, ms_nn_time, optimal=optimal))

        greedy, greedy_time = time_solver(lambda: greedy_insertion(dist))
        all_results.append(make_row(instance_name, seed, "Greedy Insertion", tour_length(greedy, dist), greedy_time, optimal=optimal))

        greedy_2opt_fast, greedy_2opt_fast_time = time_solver(lambda: two_opt_fast(greedy, dist))
        all_results.append(make_row(instance_name, seed, "Greedy Insertion + 2-opt fast", tour_length(greedy_2opt_fast, dist), greedy_2opt_fast_time, optimal=optimal))

        (_, ms_greedy_len), ms_greedy_time = time_solver(lambda: multi_start_greedy_insertion(dist))
        all_results.append(make_row(instance_name, seed, "MultiStart Greedy Insertion", ms_greedy_len, ms_greedy_time, optimal=optimal))

        (_, sa_len), sa_time = time_solver(lambda: simulated_annealing(dist, initial_tour=greedy_2opt_fast, seed=seed, iterations=5000))
        all_results.append(make_row(instance_name, seed, "Simulated Annealing", sa_len, sa_time, optimal=optimal))

        (_, ms_fast_len), ms_fast_time = time_solver(lambda: multi_start_two_opt_fast(dist, num_starts=num_starts))
        all_results.append(make_row(instance_name, seed, f"MultiStart 2-opt fast ({num_starts})", ms_fast_len, ms_fast_time, optimal=optimal))

        for row in all_results[-10:]:
            gap_text = ""
            if row["gap_pct"] != "":
                gap_text = f" gap={row['gap_pct']:.2f}%"

            print(
                f"  {row['method']:<35} "
                f"length={row['length']:.2f} "
                f"time={row['time']:.4f}s"
                f"{gap_text}"
            )

    summarize_results(all_results)
    save_results_csv(all_results, filepath="results/tsplib_results.csv")
    save_summary_csv(all_results, filepath="results/tsplib_summary.csv")

    return all_results


def time_solver(solver_fn):
    start = time.perf_counter()
    result = solver_fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_one_instance(n, seed, num_starts=10):
    points = generate_euclidean_instance(n=n, seed=seed)
    dist = compute_distance_matrix(points)

    results = []

    rand, rand_time = time_solver(lambda: random_tour(n, seed=seed))
    results.append(make_row(n, seed, "Random", tour_length(rand, dist), rand_time))

    rand_2opt, rand_2opt_time = time_solver(lambda: two_opt(rand, dist))
    results.append(make_row(n, seed, "Random + 2-opt", tour_length(rand_2opt, dist), rand_2opt_time))

    rand_2opt_fast, rand_2opt_fast_time = time_solver(lambda: two_opt_fast(rand, dist))
    results.append(make_row(n, seed, "Random + 2-opt fast", tour_length(rand_2opt_fast, dist), rand_2opt_fast_time))

    nn, nn_time = time_solver(lambda: nearest_neighbor(dist))
    results.append(make_row(n, seed, "Nearest Neighbor", tour_length(nn, dist), nn_time))

    nn_2opt, nn_2opt_time = time_solver(lambda: two_opt(nn, dist))
    results.append(make_row(n, seed, "Nearest Neighbor + 2-opt", tour_length(nn_2opt, dist), nn_2opt_time))

    nn_2opt_fast, nn_2opt_fast_time = time_solver(lambda: two_opt_fast(nn, dist))
    results.append(make_row(n, seed, "Nearest Neighbor + 2-opt fast", tour_length(nn_2opt_fast, dist), nn_2opt_fast_time))

    (_, ms_nn_len), ms_nn_time = time_solver(lambda: multi_start_nearest_neighbor(dist))
    results.append(make_row(n, seed, "MultiStart Nearest Neighbor", ms_nn_len, ms_nn_time))

    greedy, greedy_time = time_solver(lambda: greedy_insertion(dist))
    results.append(make_row(n, seed, "Greedy Insertion", tour_length(greedy, dist), greedy_time))

    greedy_2opt_fast, greedy_2opt_fast_time = time_solver(lambda: two_opt_fast(greedy, dist))
    results.append(make_row(n, seed, "Greedy Insertion + 2-opt fast", tour_length(greedy_2opt_fast, dist), greedy_2opt_fast_time))

    (_, ms_greedy_len), ms_greedy_time = time_solver(lambda: multi_start_greedy_insertion(dist))
    results.append(make_row(n, seed, "MultiStart Greedy Insertion", ms_greedy_len, ms_greedy_time))

    (_, sa_len), sa_time = time_solver(lambda: simulated_annealing(dist, initial_tour=greedy_2opt_fast, seed=seed, iterations=5000))
    results.append(make_row(n, seed, "Simulated Annealing", sa_len, sa_time))

    # (_, ms_len), ms_time = time_solver(lambda: multi_start_two_opt(dist, num_starts=num_starts))
    # results.append(make_row(n, seed, f"MultiStart 2-opt ({num_starts})", ms_len, ms_time))

    (_, ms_fast_len), ms_fast_time = time_solver(lambda: multi_start_two_opt_fast(dist, num_starts=num_starts))
    results.append(make_row(n, seed, f"MultiStart 2-opt fast ({num_starts})", ms_fast_len, ms_fast_time))

    return results


def make_row(n, seed, method, length, runtime, optimal=None):
    row = {
        "n": n,
        "seed": seed,
        "method": method,
        "length": length,
        "time": runtime,
        "gap_pct": "",
    }

    if optimal is not None:
        row["gap_pct"] = 100 * (length - optimal) / optimal

    return row


def save_results_csv(results, filepath="results/baseline_results.csv"):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "seed", "method", "length", "time", "gap_pct"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved raw results to: {filepath}")


def save_summary_csv(results, filepath="results/baseline_summary.csv"):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    for row in results:
        grouped[(row["n"], row["method"])].append(row)

    summary_rows = []

    for (n, method), rows in sorted(grouped.items()):
        avg_len = sum(row["length"] for row in rows) / len(rows)
        avg_time = sum(row["time"] for row in rows) / len(rows)
        numeric_gaps = [row["gap_pct"] for row in rows if row["gap_pct"] != ""]
        avg_gap = ""
        if numeric_gaps:
            avg_gap = sum(numeric_gaps) / len(numeric_gaps)

        summary_rows.append({
            "n": n,
            "method": method,
            "avg_length": avg_len,
            "avg_time": avg_time,
            "avg_gap_pct": avg_gap,
        })

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "method", "avg_length", "avg_time", "avg_gap_pct"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary results to: {filepath}")


def summarize_results(results):
    grouped = defaultdict(list)

    for row in results:
        grouped[(row["n"], row["method"])].append(row)

    has_gap = any(row["gap_pct"] != "" for row in results)

    print("\n================ SUMMARY ================")

    if has_gap:
        print(f"{'n':>10} | {'Method':<35} | {'Avg Length':>12} | {'Avg Time (s)':>12} | {'Avg Gap %':>10}")
        print("-" * 92)
    else:
        print(f"{'n':>10} | {'Method':<35} | {'Avg Length':>12} | {'Avg Time (s)':>12}")
        print("-" * 80)

    for (n, method), rows in sorted(grouped.items()):
        avg_len = sum(row["length"] for row in rows) / len(rows)
        avg_time = sum(row["time"] for row in rows) / len(rows)
        numeric_gaps = [row["gap_pct"] for row in rows if row["gap_pct"] != ""]

        if has_gap:
            avg_gap = ""
            if numeric_gaps:
                avg_gap = sum(numeric_gaps) / len(numeric_gaps)
                avg_gap = f"{avg_gap:.2f}"

            print(f"{str(n):>10} | {method:<35} | {avg_len:>12.2f} | {avg_time:>12.4f} | {avg_gap:>10}")
        else:
            print(f"{str(n):>10} | {method:<35} | {avg_len:>12.2f} | {avg_time:>12.4f}")


def run_benchmark(sizes=(20, 50, 100), num_seeds=5, num_starts=10):
    all_results = []

    for n in sizes:
        print(f"\n========== Synthetic TSP: n={n} ==========")

        for seed in range(num_seeds):
            print(f"Running seed {seed}...")
            results = run_one_instance(n=n, seed=seed, num_starts=num_starts)
            all_results.extend(results)

            for row in results:
                print(
                    f"  {row['method']:<35} "
                    f"length={row['length']:.2f} "
                    f"time={row['time']:.4f}s"
                )

    summarize_results(all_results)
    save_results_csv(all_results)
    save_summary_csv(all_results)

    return all_results


if __name__ == "__main__":
    run_benchmark(sizes=(20, 50, 100), num_seeds=5)

    run_tsplib_benchmark([
        "data/eil51.tsp",
        "data/berlin52.tsp",
        "data/kroA100.tsp",
    ], num_starts=10)
