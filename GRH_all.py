#!/usr/bin/env python3
"""
Python translation of GRH_all.for~.

Calculates hyperbolic greedy routing statistics between sampled pairs of nodes
in an undirected network.

Default input/output paths reproduce the original Fortran program:
    ./RG_steps/network_0.dat
    ./RG_steps/hyperbolic_coords_0.dat
    stat.dat

The original Fortran loops over target nodes as j = 1, 101, 201, ...;
this is controlled by --pair-step and defaults to 100.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

Position = Tuple[float, float]
Adjacency = List[List[int]]

PI = 3.1415926535897932


def dhyp(x1: float, y1: float, x2: float, y2: float) -> float:
    """Hyperbolic distance used by the original Fortran dhyp function.

    y1 and y2 are angular coordinates in degrees.
    """
    angular = min(abs(y1 - y2), 360.0 - abs(y1 - y2)) * PI / 180.0
    arg = 0.5 * (
        (1.0 - math.cos(angular)) * math.cosh(x1 + x2)
        + (1.0 + math.cos(angular)) * math.cosh(x1 - x2)
    )

    # math.acosh is undefined for values < 1.0. The mathematical expression
    # should be >= 1.0, but roundoff can produce 0.9999999999999999.
    if arg < 1.0 and arg > 1.0 - 1e-12:
        arg = 1.0
    return math.acosh(arg)


def read_positions(path: str) -> Dict[int, Position]:
    """Read node positions from a whitespace-separated file: i x y."""
    positions: Dict[int, Position] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number}: expected 'node x y'")
            node = int(parts[0])
            positions[node] = (float(parts[1]), float(parts[2]))
    return positions


def read_network(path: str) -> Tuple[Adjacency, int, int]:
    """Read an undirected edge list and return adjacency, number of nodes, edges."""
    edges: List[Tuple[int, int]] = []
    max_node = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number}: expected 'i j'")
            i, j = int(parts[0]), int(parts[1])
            edges.append((i, j))
            if i > max_node:
                max_node = i
            if j > max_node:
                max_node = j

    adjacency: Adjacency = [[] for _ in range(max_node + 1)]  # node labels are 1-based
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    return adjacency, max_node, len(edges)


def get_position(positions: Dict[int, Position], node: int) -> Position:
    """Return a node position, using (0, 0) for missing labels as Fortran did."""
    return positions.get(node, (0.0, 0.0))


def greedy_route(
    source: int,
    target: int,
    adjacency: Adjacency,
    positions: Dict[int, Position],
    diam: int = 1000,
    initial_min_distance: float = 100.0,
) -> Tuple[bool, int, float]:
    """Run hyperbolic greedy routing from source to target.

    Returns:
        success: True if the route reaches target, False if a cycle is detected
        hops: number of greedy hops accumulated before success/failure
        distance_sum: sum of hyperbolic edge lengths along the greedy route

    This mirrors the original Fortran logic, including the initial xdmin=100.0.
    """
    current = source
    hops = 0
    distance_sum = 0.0
    visited_route_nodes: List[int] = []

    while current != target:
        previous = current
        best_distance = initial_min_distance

        for neighbor in adjacency[current] if current < len(adjacency) else []:
            if neighbor == target:
                current = target
                break

            target_x, target_y = get_position(positions, target)
            neigh_x, neigh_y = get_position(positions, neighbor)
            candidate_distance = dhyp(target_x, target_y, neigh_x, neigh_y)
            if candidate_distance < best_distance:
                current = neighbor
                best_distance = candidate_distance

        if current in visited_route_nodes:
            return False, hops, distance_sum

        hops += 1
        visited_route_nodes.append(current)
        prev_x, prev_y = get_position(positions, previous)
        curr_x, curr_y = get_position(positions, current)
        distance_sum += dhyp(prev_x, prev_y, curr_x, curr_y)

        if hops > diam:
            # The Fortran code used a fixed naux(1:diam) array. Instead of writing
            # past that array, treat an overlong greedy route as a failure.
            return False, hops, distance_sum

    return True, hops, distance_sum


def shortest_path_length(
    source: int,
    target: int,
    adjacency: Adjacency,
    diam: int = 1000,
) -> Optional[int]:
    """Breadth-first shortest path length, capped at diam as in the Fortran code."""
    if source == target:
        return 0
    if source >= len(adjacency) or target >= len(adjacency):
        return None

    visited = {source}
    queue = deque([(source, 0)])

    while queue:
        node, distance = queue.popleft()
        if distance >= diam:
            continue

        for neighbor in adjacency[node]:
            if neighbor == target:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return None


def sampled_targets(number_of_nodes: int, pair_step: int) -> Iterable[int]:
    """Fortran equivalent of: do j=1,NODOS,100."""
    return range(1, number_of_nodes + 1, pair_step)


def run(
    network_path: str,
    positions_path: str,
    output_path: str,
    diam: int = 1000,
    pair_step: int = 100,
    report_every: int = 10000,
) -> None:
    positions = read_positions(positions_path)
    adjacency, number_of_nodes, number_of_edges = read_network(network_path)

    print("Number edges ", number_of_edges)
    print("Number nodes ", number_of_nodes)
    print("Number pairs ", number_of_nodes * (number_of_nodes - 1))

    failures = 0
    report_counter = 0
    total_pairs = 0

    stretch_sum = 0.0      # xatp: topological stretch sum
    distance_sum = 0.0     # xadp: hyperbolic-distance stretch sum
    stretch_sq_sum = 0.0   # x2tp
    distance_sq_sum = 0.0  # x2dp
    identical_positions = 0

    with open(output_path, "w", encoding="utf-8") as output:
        for source in range(1, number_of_nodes + 1):
            for target in sampled_targets(number_of_nodes, pair_step):
                if source == target:
                    continue

                success, greedy_hops, greedy_distance_sum = greedy_route(
                    source, target, adjacency, positions, diam=diam
                )

                if not success:
                    failures += 1
                else:
                    shortest = shortest_path_length(source, target, adjacency, diam=diam)
                    if shortest is None or shortest == 0:
                        # This should not happen after a successful greedy route unless the
                        # graph data are inconsistent or diam is too small.
                        failures += 1
                    else:
                        topo_stretch = greedy_hops / float(shortest)
                        stretch_sum += topo_stretch
                        stretch_sq_sum += topo_stretch ** 2

                        source_pos = get_position(positions, source)
                        target_pos = get_position(positions, target)
                        if source_pos[0] - target_pos[0] == 0.0 and source_pos[1] - target_pos[1] == 0.0:
                            identical_positions += 1
                        else:
                            direct_distance = dhyp(source_pos[0], source_pos[1], target_pos[0], target_pos[1])
                            dist_stretch = greedy_distance_sum / float(direct_distance)
                            distance_sum += dist_stretch
                            distance_sq_sum += dist_stretch ** 2

                report_counter += 1
                total_pairs += 1
                successes = total_pairs - failures
                success_ratio = successes / float(total_pairs)

                if report_counter == report_every:
                    if successes > 0:
                        average_topological = stretch_sum / float(successes)
                        average_distance = distance_sum / float(successes)
                        std_topological = math.sqrt(max(stretch_sq_sum / float(successes) - average_topological ** 2, 0.0))
                        std_distance = math.sqrt(max(distance_sq_sum / float(successes) - average_distance ** 2, 0.0))
                    else:
                        average_topological = float("nan")
                        average_distance = float("nan")
                        std_topological = float("nan")
                        std_distance = float("nan")

                    line = (
                        f"{total_pairs} {success_ratio} {average_topological} "
                        f"{std_topological} {average_distance} {std_distance} {identical_positions}"
                    )
                    print(line)
                    output.write(line + "\n")
                    output.flush()
                    report_counter = 0

    print(success_ratio if total_pairs > 0 else float("nan"))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperbolic greedy routing statistics translated from GRH_all.for~."
    )
    parser.add_argument(
        "--network",
        default="./RG_steps/network_0.dat",
        help="Path to edge-list file with rows: i j",
    )
    parser.add_argument(
        "--positions",
        default="./RG_steps/hyperbolic_coords_0.dat",
        help="Path to coordinates file with rows: i x y",
    )
    parser.add_argument(
        "--output",
        default="stat.dat",
        help="Output statistics file",
    )
    parser.add_argument("--diam", type=int, default=1000, help="Maximum diameter/search depth")
    parser.add_argument(
        "--pair-step",
        type=int,
        default=100,
        help="Target-node step. Fortran default was do j=1,NODOS,100.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=10000,
        help="Write progress/statistics every N evaluated pairs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run(
        network_path=args.network,
        positions_path=args.positions,
        output_path=args.output,
        diam=args.diam,
        pair_step=args.pair_step,
        report_every=args.report_every,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
