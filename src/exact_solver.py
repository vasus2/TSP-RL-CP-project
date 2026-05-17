from dataclasses import dataclass
import random
import time

from baselines import (
    greedy_insertion,
    multi_start_two_opt_fast,
    nearest_neighbor,
    tour_length,
    two_opt_fast,
)
from branch_ordering import BRANCH_POLICIES
from lower_bounds import LOWER_BOUNDS


@dataclass(frozen=True)
class SearchState:
    partial_tour: tuple
    visited: frozenset
    remaining: frozenset
    current_cost: float
    start: int = 0


def initial_incumbent(dist, heuristic="nn_2opt_fast", num_starts=10):
    if heuristic == "none":
        return None, float("inf")
    if heuristic == "nearest_neighbor":
        tour = nearest_neighbor(dist, start=0)
        return tour, tour_length(tour, dist)
    if heuristic == "greedy_2opt_fast":
        tour = two_opt_fast(greedy_insertion(dist, start=0), dist)
        return tour, tour_length(tour, dist)
    if heuristic == "multistart_2opt_fast":
        return multi_start_two_opt_fast(dist, num_starts=num_starts)

    tour = two_opt_fast(nearest_neighbor(dist, start=0), dist)
    return tour, tour_length(tour, dist)


def branch(partial_state, dist=None, policy="lexicographic", rng=None, model=None):
    if dist is None:
        return sorted(partial_state.remaining)

    policy_fn = BRANCH_POLICIES[policy]
    return policy_fn(partial_state, dist, rng=rng, model=model)


def compute_lower_bound(state, dist, bound="mst"):
    return LOWER_BOUNDS[bound](state, dist)


def solve_tsp_branch_and_bound(
    dist,
    start=0,
    branch_policy="nearest",
    lower_bound="mst",
    incumbent_heuristic="nn_2opt_fast",
    num_starts=10,
    seed=0,
    time_limit=None,
    model=None,
):
    started = time.perf_counter()
    n = len(dist)
    rng = random.Random(seed)
    best_tour, best_cost = initial_incumbent(
        dist,
        heuristic=incumbent_heuristic,
        num_starts=num_starts,
    )

    stats = {
        "nodes_explored": 0,
        "pruned_states": 0,
        "leaf_nodes": 0,
        "incumbent_updates": 0,
        "initial_incumbent": best_cost,
        "lower_bound": lower_bound,
        "branch_policy": branch_policy,
        "incumbent_heuristic": incumbent_heuristic,
        "time_limit_hit": False,
        "optimality_proven": False,
        "proof_time": None,
        "runtime": 0.0,
    }

    root = SearchState(
        partial_tour=(start,),
        visited=frozenset({start}),
        remaining=frozenset(set(range(n)) - {start}),
        current_cost=0.0,
        start=start,
    )
    stack = [root]

    while stack:
        if time_limit is not None and time.perf_counter() - started >= time_limit:
            stats["time_limit_hit"] = True
            break

        state = stack.pop()
        stats["nodes_explored"] += 1

        bound = compute_lower_bound(state, dist, bound=lower_bound)
        if bound >= best_cost:
            stats["pruned_states"] += 1
            continue

        if not state.remaining:
            stats["leaf_nodes"] += 1
            total_cost = state.current_cost + float(dist[state.partial_tour[-1]][start])
            if total_cost < best_cost:
                best_cost = total_cost
                best_tour = list(state.partial_tour)
                stats["incumbent_updates"] += 1
            continue

        ordered = branch(state, dist=dist, policy=branch_policy, rng=rng, model=model)
        for city in reversed(ordered):
            current = state.partial_tour[-1]
            stack.append(SearchState(
                partial_tour=state.partial_tour + (city,),
                visited=state.visited | {city},
                remaining=state.remaining - {city},
                current_cost=state.current_cost + float(dist[current][city]),
                start=start,
            ))

    stats["runtime"] = time.perf_counter() - started
    stats["optimality_proven"] = not stats["time_limit_hit"]
    stats["proof_time"] = stats["runtime"] if stats["optimality_proven"] else ""
    stats["pruning_rate"] = (
        stats["pruned_states"] / stats["nodes_explored"]
        if stats["nodes_explored"]
        else 0.0
    )

    return best_tour, best_cost, stats
