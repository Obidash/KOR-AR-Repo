"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name:
Email address:
*******************
"""

import heapq
from typing import Dict, Optional
from ar_hackathon.models.graph_state import GraphState


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    This is the function that students will implement. The game engine will
    call this function for each idle drive unit at each time step to
    determine where it should go next.

    Pickups and deliveries are automatic: a drive unit with free capacity
    that stops at (or passes through) a node with a waiting pod picks it up,
    and a drive unit that reaches a carried pod's destination station drops
    it off.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    drive_unit = state.get_drive_unit(drive_unit_id)
    if drive_unit is None or drive_unit.in_transit:
        return None

    target_node = None
    if drive_unit.carrying:
        pod = state.get_pod(drive_unit.carrying[0])
        if pod is not None:
            target_node = pod.destination_station
    else:
        waiting_pods = [
            pod for pod in state.active_pods
            if pod.carried_by is None and pod.current_node is not None
        ]
        waiting_pods.sort(key=lambda pod: (pod.entry_time, pod.id))
        if waiting_pods:
            target_node = waiting_pods[0].current_node

    if target_node is None or target_node == drive_unit.current_node:
        return None

    distances: Dict[int, float] = {drive_unit.current_node: 0}
    first_hops: Dict[int, int] = {}
    queue = [(0, drive_unit.current_node)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == target_node:
            return first_hops[node]

        for neighbor in state.neighbors(node):
            edge = state.get_edge(node, neighbor)
            if edge is None:
                continue
            next_distance = distance + edge.weight
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                first_hops[neighbor] = first_hops.get(node, neighbor)
                heapq.heappush(queue, (next_distance, neighbor))

    return None
