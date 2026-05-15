from tsp_utils import generate_euclidean_instance, compute_distance_matrix, tour_length
from baselines import random_tour, nearest_neighbor, two_opt


def main():
    points = generate_euclidean_instance(n=50, seed=1)
    dist = compute_distance_matrix(points)

    tour = random_tour(len(points), seed=42)
    length = tour_length(tour, dist)

    print("Tour:", tour)
    print("Tour length:", length)

    nn_tour = nearest_neighbor(dist)
    nn_length = tour_length(nn_tour, dist)

    print("NN length:", nn_length)

    two_opt_tour = two_opt(nn_tour, dist)
    two_opt_len = tour_length(two_opt_tour, dist)

    print("2-opt length:", two_opt_len)


if __name__ == "__main__":
    main()