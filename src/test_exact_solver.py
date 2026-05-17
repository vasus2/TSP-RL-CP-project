import itertools

from tsp_utils import generate_euclidean_instance, compute_distance_matrix, tour_length
from exact_solver import solve_tsp_branch_and_bound


def brute_force_tsp(dist, start=0):
    n = len(dist)
    cities = [i for i in range(n) if i != start]

    best_tour = None
    best_cost = float("inf")

    for perm in itertools.permutations(cities):
        tour = [start] + list(perm)
        cost = tour_length(tour, dist)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost


def test_branch_and_bound_against_brute_force():
    for n in [6, 7, 8]:
        for seed in range(5):
            points = generate_euclidean_instance(n=n, seed=seed)
            dist = compute_distance_matrix(points)

            _, brute_cost = brute_force_tsp(dist)

            _, bb_cost, stats = solve_tsp_branch_and_bound(
                dist,
                branch_policy="nearest",
                lower_bound="mst",
                incumbent_heuristic="nn_2opt_fast",
                seed=seed,
                time_limit=10.0,
            )

            print(f"n={n}, seed={seed}, brute={brute_cost:.4f}, bb={bb_cost:.4f}")

            assert stats["optimality_proven"] is True
            assert abs(brute_cost - bb_cost) < 1e-6


class NearestEdgeModel:
    def predict_proba(self, rows):
        probs = []
        for row in rows:
            score = 1.0 / (1.0 + row[7])
            probs.append([1.0 - score, score])
        return probs


def test_learned_branch_policy_against_brute_force():
    points = generate_euclidean_instance(n=8, seed=0)
    dist = compute_distance_matrix(points)
    _, brute_cost = brute_force_tsp(dist)

    _, bb_cost, stats = solve_tsp_branch_and_bound(
        dist,
        branch_policy="learned",
        lower_bound="mst",
        incumbent_heuristic="nn_2opt_fast",
        seed=0,
        time_limit=10.0,
        model=NearestEdgeModel(),
    )

    assert stats["optimality_proven"] is True
    assert abs(brute_cost - bb_cost) < 1e-6


if __name__ == "__main__":
    test_branch_and_bound_against_brute_force()
    test_learned_branch_policy_against_brute_force()
    print("All exact solver tests passed.")
