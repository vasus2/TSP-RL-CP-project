import numpy as np


def random_tour(n, seed=None):
    rng = np.random.default_rng(seed)
    tour = list(range(n))
    rng.shuffle(tour)
    return tour


def nearest_neighbor(dist, start=0):
    n = len(dist)
    visited = {start}
    tour = [start]
    curr = start

    for _ in range(n - 1):
        next_city = min(
            (j for j in range(n) if j not in visited),
            key=lambda j: dist[curr][j]
        )
        tour.append(next_city)
        visited.add(next_city)
        curr = next_city

    return tour


def tour_length(tour, dist):
    total = 0.0
    n = len(tour)

    for i in range(n):
        total += dist[tour[i]][tour[(i + 1) % n]]

    return total


def two_opt(tour, dist):
    best = tour[:]
    improved = True

    while improved:
        improved = False
        best_length = tour_length(best, dist)

        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue

                new_tour = best[:]
                new_tour[i:j] = reversed(best[i:j])
                new_length = tour_length(new_tour, dist)

                if new_length < best_length:
                    best = new_tour
                    improved = True
                    break

            if improved:
                break

    return best

def two_opt_fast(tour, dist):
    best = tour[:]
    n = len(best)
    improved = True

    while improved:
        improved = False

        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue

                a, b = best[i - 1], best[i]
                c, d = best[j - 1], best[j % n]

                old_cost = dist[a][b] + dist[c][d]
                new_cost = dist[a][c] + dist[b][d]

                if new_cost < old_cost:
                    best[i:j] = reversed(best[i:j])
                    improved = True
                    break

            if improved:
                break

    return best

def multi_start_two_opt(dist, num_starts=10):
    n = len(dist)
    best_tour = None
    best_len = float("inf")

    for seed in range(num_starts):
        tour = random_tour(n, seed=seed)
        improved = two_opt(tour, dist)
        length = tour_length(improved, dist)

        if length < best_len:
            best_len = length
            best_tour = improved

    return best_tour, best_len


def multi_start_two_opt_fast(dist, num_starts=10):
    n = len(dist)
    best_tour = None
    best_len = float("inf")

    for seed in range(num_starts):
        tour = random_tour(n, seed=seed)
        improved = two_opt_fast(tour, dist)
        length = tour_length(improved, dist)

        if length < best_len:
            best_len = length
            best_tour = improved

    return best_tour, best_len

def greedy_insertion(dist, start=0):
    n = len(dist)

    nearest = min(
        (j for j in range(n) if j != start),
        key=lambda j: dist[start][j]
    )

    tour = [start, nearest]
    unvisited = set(range(n)) - set(tour)

    while unvisited:
        best_city = -1
        best_pos = -1
        best_increase = float("inf")

        for city in unvisited:
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]

                increase = dist[a][city] + dist[city][b] - dist[a][b]

                if increase < best_increase:
                    best_increase = increase
                    best_city = city
                    best_pos = i + 1

        if best_city == -1 or best_pos == -1:
            raise RuntimeError("Greedy insertion failed to select a city.")

        tour.insert(best_pos, best_city)
        unvisited.remove(best_city)

    return tour


def multi_start_nearest_neighbor(dist):
    n = len(dist)
    best_tour = None
    best_len = float("inf")

    for start in range(n):
        tour = nearest_neighbor(dist, start=start)
        length = tour_length(tour, dist)
        if length < best_len:
            best_tour = tour
            best_len = length

    return best_tour, best_len


def multi_start_greedy_insertion(dist):
    n = len(dist)
    best_tour = None
    best_len = float("inf")

    for start in range(n):
        tour = greedy_insertion(dist, start=start)
        length = tour_length(tour, dist)
        if length < best_len:
            best_tour = tour
            best_len = length

    return best_tour, best_len


def simulated_annealing(
    dist,
    initial_tour=None,
    seed=0,
    iterations=5000,
    initial_temp=None,
    cooling_rate=0.995,
):
    rng = np.random.default_rng(seed)
    n = len(dist)

    if initial_tour is None:
        current = nearest_neighbor(dist, start=seed % n)
    else:
        current = initial_tour[:]

    current_len = tour_length(current, dist)
    best = current[:]
    best_len = current_len

    if initial_temp is None:
        nonzero = dist[dist > 0]
        initial_temp = float(np.mean(nonzero)) if len(nonzero) else 1.0

    temp = max(float(initial_temp), 1e-12)

    for _ in range(iterations):
        i, j = sorted(rng.choice(np.arange(1, n), size=2, replace=False))
        if j - i == 1:
            continue

        a, b = current[i - 1], current[i]
        c, d = current[j - 1], current[j % n]
        delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]

        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current[i:j] = reversed(current[i:j])
            current_len += delta

            if current_len < best_len:
                best = current[:]
                best_len = current_len

        temp = max(temp * cooling_rate, 1e-12)

    return best, best_len
