import csv
import random
from pathlib import Path

from branch_ordering import greedy_order, nearest_neighbor_order
from exact_solver import SearchState
from learning_features import FEATURE_NAMES, extract_candidate_features
from tsp_utils import compute_distance_matrix, generate_euclidean_instance


def teacher_preferred_city(state, dist, teacher):
    if teacher == "nearest":
        return nearest_neighbor_order(state, dist)[0]
    if teacher == "greedy":
        return greedy_order(state, dist)[0]
    raise ValueError(f"Unknown teacher policy: {teacher}")


def next_rollout_city(state, dist, teacher, rng, exploration_rate):
    if rng.random() < exploration_rate:
        return rng.choice(sorted(state.remaining))
    return teacher_preferred_city(state, dist, teacher)


def rows_for_state(instance_id, seed, teacher, state, dist):
    preferred = teacher_preferred_city(state, dist, teacher)
    rows = []

    for candidate in sorted(state.remaining):
        features = extract_candidate_features(state, candidate, dist)
        row = {
            "instance_id": instance_id,
            "seed": seed,
            "teacher": teacher,
            "raw_current_city": state.partial_tour[-1],
            "raw_candidate_city": candidate,
            "preferred_city": preferred,
            "label": int(candidate == preferred),
        }
        row.update(dict(zip(FEATURE_NAMES, features)))
        rows.append(row)

    return rows


def generate_branch_dataset(
    sizes=(8, 10, 12),
    seeds=range(100),
    teachers=("nearest", "greedy"),
    rollouts_per_instance=3,
    exploration_rate=0.35,
    output_path="results/branch_data.csv",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for n in sizes:
        for seed in seeds:
            points = generate_euclidean_instance(n=n, seed=seed)
            dist = compute_distance_matrix(points)
            instance_id = f"synthetic_n{n}_seed{seed}"

            for teacher in teachers:
                for rollout in range(rollouts_per_instance):
                    rng = random.Random((seed + 1) * 1009 + rollout * 9176 + n)
                    state = SearchState(
                        partial_tour=(0,),
                        visited=frozenset({0}),
                        remaining=frozenset(set(range(n)) - {0}),
                        current_cost=0.0,
                        start=0,
                    )

                    while state.remaining:
                        rows.extend(rows_for_state(instance_id, seed, teacher, state, dist))
                        city = next_rollout_city(state, dist, teacher, rng, exploration_rate)
                        current = state.partial_tour[-1]
                        state = SearchState(
                            partial_tour=state.partial_tour + (city,),
                            visited=state.visited | {city},
                            remaining=state.remaining - {city},
                            current_cost=state.current_cost + float(dist[current][city]),
                            start=0,
                        )

    fieldnames = [
        "instance_id",
        "seed",
        "teacher",
        "raw_current_city",
        "raw_candidate_city",
        "preferred_city",
        "label",
    ] + FEATURE_NAMES

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} branch training rows to: {output_path}")
    return rows


if __name__ == "__main__":
    generate_branch_dataset()
