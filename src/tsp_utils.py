import numpy as np
from pathlib import Path


def generate_euclidean_instance(n, seed: int = 0, scale: int = 1000):
    rng = np.random.default_rng(seed)
    points = rng.random((n, 2)) * scale
    return points


def compute_distance_matrix(points):
    n = len(points)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist[i, j] = np.linalg.norm(points[i] - points[j])

    return dist


def load_tsplib_coordinates(filepath):
    filepath = Path(filepath)
    coords = []
    reading_coords = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue

            if line == "EOF":
                break

            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    _, x, y = parts[:3]
                    coords.append((float(x), float(y)))

    return np.array(coords)


def tour_length(tour, dist):
    total = 0.0
    n = len(tour)

    for i in range(n):
        total += dist[tour[i]][tour[(i + 1) % n]]

    return total