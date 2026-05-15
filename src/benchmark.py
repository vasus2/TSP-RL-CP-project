import time
import csv
from pathlib import Path
from collections import defaultdict

from tsp_utils import generate_euclidean_instance, compute_distance_matrix, load_tsplib_coordinates
from baselines import (
    random_tour,
    nearest_neighbor,
    greedy_insertion,
    two_opt,
    two_opt_fast,
    # multi_start_two_opt,
    multi_start_two_opt_fast,
    tour_length,
)
def run_tsplib_benchmark(filepaths, num_starts=10):
    """
    Run baseline comparison on TSPLIB coordinate-based .tsp instances.
    Results are saved separately from the synthetic benchmark results.
    """
    all_results = []

    for filepath in filepaths:
        instance_name = Path(filepath).stem
        print(f"\n========== TSPLIB: {instance_name} ==========")

        points = load_tsplib_coordinates(filepath)
        dist = compute_distance_matrix(points)
        n = len(points)
        seed = 0

        rand, rand_time = time_solver(lambda: random_tour(n, seed=seed))
        all_results.append(make_row(instance_name, seed, "Random", tour_length(rand, dist), rand_time))

        rand_2opt_fast, rand_2opt_fast_time = time_solver(lambda: two_opt_fast(rand, dist))
        all_results.append(make_row(instance_name, seed, "Random + 2-opt fast", tour_length(rand_2opt_fast, dist), rand_2opt_fast_time))

        nn, nn_time = time_solver(lambda: nearest_neighbor(dist))
        all_results.append(make_row(instance_name, seed, "Nearest Neighbor", tour_length(nn, dist), nn_time))

        nn_2opt_fast, nn_2opt_fast_time = time_solver(lambda: two_opt_fast(nn, dist))
        all_results.append(make_row(instance_name, seed, "Nearest Neighbor + 2-opt fast", tour_length(nn_2opt_fast, dist), nn_2opt_fast_time))

        greedy, greedy_time = time_solver(lambda: greedy_insertion(dist))
        all_results.append(make_row(instance_name, seed, "Greedy Insertion", tour_length(greedy, dist), greedy_time))

        greedy_2opt_fast, greedy_2opt_fast_time = time_solver(lambda: two_opt_fast(greedy, dist))
        all_results.append(make_row(instance_name, seed, "Greedy Insertion + 2-opt fast", tour_length(greedy_2opt_fast, dist), greedy_2opt_fast_time))

        (_, ms_fast_len), ms_fast_time = time_solver(lambda: multi_start_two_opt_fast(dist, num_starts=num_starts))
        all_results.append(make_row(instance_name, seed, f"MultiStart 2-opt fast ({num_starts})", ms_fast_len, ms_fast_time))

        for row in all_results[-7:]:
            print(
                f"  {row['method']:<35} "
                f"length={row['length']:.2f} "
                f"time={row['time']:.4f}s"
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

    greedy, greedy_time = time_solver(lambda: greedy_insertion(dist))
    results.append(make_row(n, seed, "Greedy Insertion", tour_length(greedy, dist), greedy_time))

    greedy_2opt_fast, greedy_2opt_fast_time = time_solver(lambda: two_opt_fast(greedy, dist))
    results.append(make_row(n, seed, "Greedy Insertion + 2-opt fast", tour_length(greedy_2opt_fast, dist), greedy_2opt_fast_time))

    # (_, ms_len), ms_time = time_solver(lambda: multi_start_two_opt(dist, num_starts=num_starts))
    # results.append(make_row(n, seed, f"MultiStart 2-opt ({num_starts})", ms_len, ms_time))

    (_, ms_fast_len), ms_fast_time = time_solver(lambda: multi_start_two_opt_fast(dist, num_starts=num_starts))
    results.append(make_row(n, seed, f"MultiStart 2-opt fast ({num_starts})", ms_fast_len, ms_fast_time))

    return results


def make_row(n, seed, method, length, runtime):
    return {
        "n": n,
        "seed": seed,
        "method": method,
        "length": length,
        "time": runtime,
    }


def save_results_csv(results, filepath="results/baseline_results.csv"):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "seed", "method", "length", "time"])
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

        summary_rows.append({
            "n": n,
            "method": method,
            "avg_length": avg_len,
            "avg_time": avg_time,
        })

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "method", "avg_length", "avg_time"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary results to: {filepath}")


def summarize_results(results):
    grouped = defaultdict(list)

    for row in results:
        grouped[(row["n"], row["method"])].append(row)

    print("\n================ SUMMARY ================")
    print(f"{'n':>5} | {'Method':<35} | {'Avg Length':>12} | {'Avg Time (s)':>12}")
    print("-" * 78)

    for (n, method), rows in sorted(grouped.items()):
        avg_len = sum(row["length"] for row in rows) / len(rows)
        avg_time = sum(row["time"] for row in rows) / len(rows)
        print(f"{n:>5} | {method:<35} | {avg_len:>12.2f} | {avg_time:>12.4f}")


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