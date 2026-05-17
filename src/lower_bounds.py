import heapq


def trivial_lower_bound(state, dist):
    return state.current_cost


def _mst_cost(nodes, dist):
    if len(nodes) <= 1:
        return 0.0

    start = nodes[0]
    visited = {start}
    heap = [(float(dist[start][node]), node) for node in nodes[1:]]
    heapq.heapify(heap)
    total = 0.0

    while heap and len(visited) < len(nodes):
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue

        visited.add(node)
        total += cost

        for nxt in nodes:
            if nxt not in visited:
                heapq.heappush(heap, (float(dist[node][nxt]), nxt))

    return total


def mst_lower_bound(state, dist):
    """
    Bound a partial tour by adding an MST over unvisited nodes plus two links.

    The two links connect the current path endpoint to the unvisited set and the
    unvisited set back to the fixed start city. Runtime is O(r^2 log r), where r
    is the number of remaining cities.
    """
    remaining = list(state.remaining)
    if not remaining:
        return state.current_cost + float(dist[state.partial_tour[-1]][state.start])

    current = state.partial_tour[-1]
    start = state.start

    mst = _mst_cost(remaining, dist)
    from_current = min(float(dist[current][city]) for city in remaining)
    to_start = min(float(dist[city][start]) for city in remaining)

    return state.current_cost + mst + from_current + to_start


LOWER_BOUNDS = {
    "trivial": trivial_lower_bound,
    "mst": mst_lower_bound,
}
