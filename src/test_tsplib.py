from tsp_utils import *
from baselines import *

points = load_tsplib_coordinates("data/berlin52.tsp")

print("Loaded points shape:", points.shape)

dist = compute_distance_matrix(points)

nn = nearest_neighbor(dist)
nn_len = tour_length(nn, dist)

nn_2opt = two_opt_fast(nn, dist)
nn_2opt_len = tour_length(nn_2opt, dist)

greedy = greedy_insertion(dist)
greedy_len = tour_length(greedy, dist)

greedy_2opt = two_opt_fast(greedy, dist)
greedy_2opt_len = tour_length(greedy_2opt, dist)

print("NN:", nn_len)
print("NN + 2-opt fast:", nn_2opt_len)

print("Greedy:", greedy_len)
print("Greedy + 2-opt fast:", greedy_2opt_len)