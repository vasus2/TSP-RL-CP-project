import numpy as np


FEATURE_NAMES = [
    "n",
    "current_city",
    "candidate_city",
    "depth",
    "remaining_count",
    "current_cost",
    "current_cost_per_edge",
    "candidate_edge_distance",
    "candidate_to_start_distance",
    "min_candidate_to_remaining",
    "avg_candidate_to_remaining",
    "max_candidate_to_remaining",
    "min_current_to_remaining",
    "avg_current_to_remaining",
    "candidate_edge_rank",
    "candidate_return_rank",
]


def _rank(value, values):
    ordered = sorted(values)
    return ordered.index(value) / max(len(ordered) - 1, 1)


def extract_candidate_features(state, candidate_city, dist):
    n = len(dist)
    current = state.partial_tour[-1]
    start = state.start
    remaining = sorted(state.remaining)
    others = [city for city in remaining if city != candidate_city]

    edge_distances = [float(dist[current][city]) for city in remaining]
    return_distances = [float(dist[city][start]) for city in remaining]
    candidate_to_others = [float(dist[candidate_city][city]) for city in others]
    current_to_remaining = [float(dist[current][city]) for city in remaining]

    avg_edge = float(np.mean(dist[dist > 0])) if np.any(dist > 0) else 1.0
    candidate_edge = float(dist[current][candidate_city])
    candidate_return = float(dist[candidate_city][start])

    if candidate_to_others:
        min_candidate_remaining = min(candidate_to_others)
        avg_candidate_remaining = sum(candidate_to_others) / len(candidate_to_others)
        max_candidate_remaining = max(candidate_to_others)
    else:
        min_candidate_remaining = 0.0
        avg_candidate_remaining = 0.0
        max_candidate_remaining = 0.0

    return [
        float(n),
        float(current) / max(n - 1, 1),
        float(candidate_city) / max(n - 1, 1),
        float(len(state.partial_tour)) / max(n, 1),
        float(len(remaining)) / max(n, 1),
        float(state.current_cost) / max(n * avg_edge, 1e-12),
        float(state.current_cost) / max(len(state.partial_tour), 1),
        candidate_edge / max(avg_edge, 1e-12),
        candidate_return / max(avg_edge, 1e-12),
        min_candidate_remaining / max(avg_edge, 1e-12),
        avg_candidate_remaining / max(avg_edge, 1e-12),
        max_candidate_remaining / max(avg_edge, 1e-12),
        min(current_to_remaining) / max(avg_edge, 1e-12),
        (sum(current_to_remaining) / len(current_to_remaining)) / max(avg_edge, 1e-12),
        _rank(candidate_edge, edge_distances),
        _rank(candidate_return, return_distances),
    ]
