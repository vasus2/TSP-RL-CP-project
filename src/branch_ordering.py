import random

import numpy as np

from learning_features import extract_candidate_features


def lexicographic_order(state, dist, rng=None, model=None):
    return sorted(state.remaining)


def nearest_neighbor_order(state, dist, rng=None, model=None):
    current = state.partial_tour[-1]
    return sorted(state.remaining, key=lambda city: (float(dist[current][city]), city))


def greedy_order(state, dist, rng=None, model=None):
    current = state.partial_tour[-1]
    start = state.start
    return sorted(
        state.remaining,
        key=lambda city: (float(dist[current][city]) + float(dist[city][start]), city),
    )


def random_order(state, dist, rng=None, model=None):
    cities = list(state.remaining)
    if rng is None:
        rng = random.Random(0)
    rng.shuffle(cities)
    return cities


def learned_ordering(state, dist, rng=None, model=None):
    if model is None:
        raise ValueError("branch_policy='learned' requires a trained model.")

    candidates = sorted(state.remaining)
    feature_rows = [
        extract_candidate_features(state, candidate, dist)
        for candidate in candidates
    ]
    scores = np.asarray(model.predict_proba(feature_rows))[:, 1]
    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return [candidate for candidate, _ in ranked]


BRANCH_POLICIES = {
    "lexicographic": lexicographic_order,
    "nearest": nearest_neighbor_order,
    "greedy": greedy_order,
    "random": random_order,
    "learned": learned_ordering,
}
