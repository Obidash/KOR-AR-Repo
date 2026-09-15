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


_assigned_pods: Dict[int, str] = {}
_last_time_step: Optional[int] = None


def _shortest_path(state: GraphState, start: int, target: int):
    distances: Dict[int, float] = {start: 0}
    first_hops: Dict[int, int] = {}
    queue = [(0, start)]

    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == target:
            return first_hops.get(node), distance

        for neighbor in state.neighbors(node):
            edge = state.get_edge(node, neighbor)
            if edge is None:
                continue
            if (edge.capacity is not None
                    and state.edge_occupancy(node, neighbor) >= edge.capacity):
                continue
            destination = state.get_node(neighbor)
            if (destination is not None
                    and destination.capacity is not None
                    and state.node_occupancy(neighbor) >= destination.capacity):
                continue

            next_distance = distance + edge.weight
            if next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                first_hops[neighbor] = first_hops.get(node, neighbor)
                heapq.heappush(queue, (next_distance, neighbor))

    return None, float("inf")


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
    global _last_time_step

    if _last_time_step is not None and state.current_time_step < _last_time_step:
        _assigned_pods.clear()
    _last_time_step = state.current_time_step

    drive_unit = state.get_drive_unit(drive_unit_id)
    if drive_unit is None or drive_unit.in_transit:
        return None

    target_node = None
    if drive_unit.carrying:
        _assigned_pods.pop(drive_unit_id, None)
        pod = state.get_pod(drive_unit.carrying[0])
        if pod is not None:
            target_node = pod.destination_station
    else:
        assigned_pod_id = _assigned_pods.get(drive_unit_id)
        assigned_pod = state.get_pod(assigned_pod_id) if assigned_pod_id else None
        if (assigned_pod is None or assigned_pod.carried_by not in (None, drive_unit_id)
                or assigned_pod.current_node is None):
            _assigned_pods.pop(drive_unit_id, None)

        waiting_pods = [
            pod for pod in state.active_pods
            if (pod.carried_by is None and pod.current_node is not None
                    and pod.id not in _assigned_pods.values())
        ]
        if not waiting_pods:
            waiting_pods = [
                pod for pod in state.active_pods
                if pod.carried_by is None and pod.current_node is not None
            ]
        if assigned_pod is None and waiting_pods:
            candidates = []
            for pod in waiting_pods:
                _, distance = _shortest_path(
                    state, drive_unit.current_node, pod.current_node)
                candidates.append((distance, pod.entry_time, pod.id, pod))
            candidates.sort(key=lambda item: item[:3])
            assigned_pod = candidates[0][3]
            _assigned_pods[drive_unit_id] = assigned_pod.id
        if assigned_pod is not None:
            target_node = assigned_pod.current_node

    if target_node is None:
        current_node = state.get_node(drive_unit.current_node)
        if current_node is not None and current_node.node_type == "station":
            for neighbor in state.neighbors(drive_unit.current_node):
                destination = state.get_node(neighbor)
                if (destination is None or destination.capacity is None
                        or state.node_occupancy(neighbor) < destination.capacity):
                    return neighbor
        return None

    if target_node == drive_unit.current_node:
        return None

    next_node, _ = _shortest_path(state, drive_unit.current_node, target_node)
    return next_node
