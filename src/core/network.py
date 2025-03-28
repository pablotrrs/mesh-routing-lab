import math
import threading
import logging as log
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from core.clock import clock
from core.node import Node
from core.packet_registry import registry


class Network:
    """Represents a network of nodes with dynamic changes and connections.

    Attributes:
        nodes (Dict[int, Node]): Dictionary of nodes in the network, keyed by node ID.
        connections (Dict[int, List[int]]): Dictionary of node connections, keyed by node ID.
        active_nodes (Set[int]): Set of currently active nodes.
        dynamic_change_events (List[int]): List of times when dynamic changes occurred.
        running (bool): Indicates whether the network is running.
        lock (threading.Lock): Lock for thread-safe operations.
        mean_disconnection_interval_ms (Optional[float]): Mean interval between dynamic changes.
        mean_reconnection_interval_ms (Optional[float]): Interval for node reconnection.
        disconnection_probability (Optional[float]): Probability of node disconnection.
    """

    def __init__(self) -> None:
        """Initializes the network with default values."""
        self.nodes: Dict[int, Node] = {}
        self.connections: Dict[int, List[int]] = {}
        self.active_nodes: Set[int] = set()
        self.dynamic_change_events: List[int] = []
        self.running: bool = True
        self.lock: threading.Lock = threading.Lock()
        self.mean_disconnection_interval_ms: Optional[float] = None
        self.mean_reconnection_interval_ms: Optional[float] = None
        self.disconnection_probability: Optional[float] = None
        log.debug("Network initialized.")

    def set_mean_disconnection_interval_ms(self, mean_disconnection_interval_ms: float) -> None:
        """Sets the mean interval between dynamic changes.

        Args:
            mean_disconnection_interval_ms (float): Mean interval in milliseconds.
        """
        self.mean_disconnection_interval_ms = mean_disconnection_interval_ms

    def set_mean_reconnection_interval_ms(self, mean_reconnection_interval_ms: float) -> None:
        """Sets a mean interval for node reconnection when disconnected.

        Args:
            mean_reconnection_interval_ms (float): Mean reconnection interval in milliseconds.
        """
        self.mean_reconnection_interval_ms = mean_reconnection_interval_ms

    def set_disconnection_interval_ms(self, disconnection_interval_ms: float) -> None:
        """Sets a fixed interval between dynamic changes.

        Args:
            disconnection_interval_ms (float): Fixed interval in milliseconds.
        """
        self.disconnection_interval_ms = disconnection_interval_ms

    def set_reconnection_interval_ms(self, reconnection_interval_ms: float) -> None:
        """Sets the fixed interval for node reconnection when disconnected.

        Args:
            reconnection_interval_ms (float): Fixed reconnection interval in milliseconds.
        """
        self.reconnection_interval_ms = reconnection_interval_ms

    def set_disconnection_probability(self, disconnection_probability: float) -> None:
        """Sets the probability of node disconnection.

        Args:
            disconnection_probability (float): Probability of disconnection (0.0 to 1.0).
        """
        self.disconnection_probability = disconnection_probability

    def generate_next_dynamic_change(self) -> int:
        """Generates the next dynamic change time based on configured intervals.

        Returns:
            int: Time (ms) for the next dynamic change.
        """
        if hasattr(self, "disconnection_interval_ms") and self.disconnection_interval_ms is not None:
            return self.disconnection_interval_ms
        elif self.mean_disconnection_interval_ms == float("inf"):
            return int(1e12)  # A very large number to simulate no changes
        else:
            return int(np.random.exponential(self.mean_disconnection_interval_ms))

    def start_dynamic_changes(self) -> None:
        """Starts a thread to apply dynamic changes based on the central clock."""
        current_time = clock.get_current_time()
        log.debug(f"Network clock starts: {current_time}")
        threading.Thread(target=self._monitor_dynamic_changes, daemon=True).start()

    def stop_dynamic_changes(self) -> None:
        """Stops the dynamic changes thread."""
        self.running = False
        log.debug("Dynamic changes stopped.")

    def _monitor_dynamic_changes(self) -> None:
        """Monitors the central clock and applies dynamic changes automatically."""
        next_event_time = clock.get_current_time() + self.generate_next_dynamic_change()
        while self.running:
            current_time = clock.get_current_time()
            with self.lock:
                if current_time >= next_event_time:
                    log.debug("⚡ZZZAP⚡")  # Dynamic change event
                    self.trigger_dynamic_change()
                    self.dynamic_change_events.append(current_time)
                    next_event_time = current_time + self.generate_next_dynamic_change()

                self._handle_reconnections()

    def trigger_dynamic_change(self) -> None:
        """Applies the logic for dynamic changes in the network (e.g., random node disconnections)."""
        current_time = clock.get_current_time()

        for node_id in list(self.active_nodes):
            if np.random.rand() < self.disconnection_probability:
                node = self.nodes[node_id]
                node.status = False

                # fixed reconnection mode
                if self.reconnection_interval_ms is not None:
                    node.disconnected_at = current_time
                # mean reconnection mode
                elif self.mean_reconnection_interval_ms is not None:
                    node.reconnect_time = current_time + np.random.exponential(
                        scale=self.mean_reconnection_interval_ms
                    )

                log.debug(f"Node {node_id} disconnected at {current_time:.2f}.")

    def _handle_reconnections(self) -> None:
        """Handles reconnections of previously disconnected nodes."""
        current_time = clock.get_current_time()

        for node_id, node in self.nodes.items():
            if not node.status:
                # Fixed reconnection mode
                if self.reconnection_interval_ms is not None:
                    if not hasattr(node, "disconnected_at") or node.disconnected_at is None:
                        continue
                    if current_time >= node.disconnected_at + self.reconnection_interval_ms:
                        node.status = True
                        delattr(node, "disconnected_at")
                        log.debug(f"Node {node_id} reconnected at {current_time:.2f}.")
                # Mean reconnection mode
                elif self.mean_reconnection_interval_ms is not None:
                    if not hasattr(node, "reconnect_time"):
                        node.reconnect_time = current_time + np.random.exponential(self.mean_reconnection_interval_ms)
                    if current_time >= node.reconnect_time:
                        node.status = True
                        delattr(node, "reconnect_time")
                        log.debug(f"Node {node_id} reconnected at {current_time:.2f}.")

    def validate_connection(self, from_node_id: int, to_node_id: int) -> bool:
        """Validates if a connection between two nodes is valid.

        Args:
            from_node_id (int): ID of the source node.
            to_node_id (int): ID of the destination node.

        Returns:
            bool: True if the connection is valid, False otherwise.
        """
        with self.lock:
            return (
                to_node_id in self.connections.get(from_node_id, [])
                and from_node_id in self.active_nodes
                and to_node_id in self.active_nodes
            )

    def add_node(self, node: Node) -> None:
        """Adds a node to the network.

        Args:
            node (Node): The node to add.
        """
        self.nodes[node.node_id] = node
        self.connections[node.node_id] = []
        self.active_nodes.add(node.node_id)
        log.debug(f"Node {node.node_id} added to the network.")

    def connect_nodes(self, node1_id: int, node2_id: int) -> None:
        """Connects two nodes in the network without duplicating connections.

        Args:
            node1_id (int): ID of the first node.
            node2_id (int): ID of the second node.
        """
        if node2_id not in self.connections.get(node1_id, []):
            self.connections.setdefault(node1_id, []).append(node2_id)
        if node1_id not in self.connections.get(node2_id, []):
            self.connections.setdefault(node2_id, []).append(node1_id)
        log.debug(f"Nodes {node1_id} and {node2_id} connected.")

    def get_neighbors(self, node_id: int) -> List[int]:
        """Returns a node's neighbors, excluding itself.

        Args:
            node_id (int): ID of the node.

        Returns:
            List[int]: List of neighbor node IDs.
        """
        neighbors = self.connections.get(node_id, [])
        return list(set(neighbor for neighbor in neighbors if neighbor != node_id))

    def get_nodes(self) -> List[int]:
        """Returns a list of all node IDs in the network.

        Returns:
            List[int]: List of node IDs.
        """
        return list(self.nodes.keys())

    def is_node_reachable(self, from_node_id: int, to_node_id: int) -> bool:
        """Checks if a node is reachable from another node.

        Args:
            from_node_id (int): ID of the source node.
            to_node_id (int): ID of the destination node.

        Returns:
            bool: True if the destination node is reachable, False otherwise.
        """
        return (
            to_node_id in self.connections.get(from_node_id, [])
            and from_node_id in self.active_nodes
            and to_node_id in self.active_nodes
        )

    def send(self, from_node_id: int, to_node_id: int, packet: Dict) -> None:
        """Sends a packet in the network.

        Args:
            from_node_id (int): ID of the source node.
            to_node_id (int): ID of the destination node.
            packet (Dict): The packet to send.
        """
        episode_number = packet.get("episode_number")
        registry.initialize_episode(episode_number)

        if not self.is_node_reachable(from_node_id, to_node_id):
            registry.mark_packet_lost(
                episode_number, from_node_id, to_node_id, packet["type"].value
            )
            return

        if to_node_id is None:
            registry.mark_episode_complete(episode_number, True)
            return

        latency = (
            self.get_latency(from_node_id, to_node_id) if to_node_id != "N/A" else 0
        )

        registry.log_packet_hop(
            episode_number,
            from_node_id,
            to_node_id,
            self.nodes[from_node_id].get_assigned_function()
            if self.nodes[from_node_id].get_assigned_function()
            else "N/A",
            "active" if from_node_id in self.active_nodes else "inactive",
            latency,
            packet["type"].value,
        )

        if self.is_node_reachable(from_node_id, to_node_id):
            log.debug(
                f"Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds"
            )
            time.sleep(latency)
            self.nodes[to_node_id].application.receive_packet(packet)
        else:
            log.debug(
                f"Failed to send packet: Node {to_node_id} is not reachable from Node {from_node_id}"
            )

    @classmethod
    def from_yaml(cls, file_path: str) -> Tuple["Network", Node]:
        """Creates a Network instance from a YAML file and returns the network and sender node.

        Args:
            file_path (str): Path to the YAML file.

        Returns:
            Tuple[Network, Node]: The network and the sender node.

        Raises:
            ValueError: If no sender node is found in the YAML file.
        """
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        network = cls()
        sender_node = None

        for node_id, node_info in data["nodes"].items():
            node_id = int(node_id)
            position = tuple(node_info["position"]) if "position" in node_info else None

            node = Node(node_id, network, position=position)
            network.add_node(node)

            if "type" in node_info and node_info["type"] == "sender":
                node.is_sender = True
                sender_node = node

        if sender_node is None:
            raise ValueError("No sender node found in the YAML file.")

        for node_id, node_info in data["nodes"].items():
            node_id = int(node_id)
            neighbors = node_info["neighbors"]
            for neighbor_id in neighbors:
                network.connect_nodes(node_id, neighbor_id)

        log.debug(f"Network loaded from {file_path}.")
        return network, sender_node

    def get_distance(self, node_id1: int, node_id2: int) -> float:
        """Calculates the Euclidean distance between two nodes.

        Args:
            node_id1 (int): ID of the first node.
            node_id2 (int): ID of the second node.

        Returns:
            float: Distance between the nodes.
        """
        pos1 = self.nodes[node_id1].position
        pos2 = self.nodes[node_id2].position
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2
            + (pos1[1] - pos2[1]) ** 2
            + (pos1[2] - pos2[2]) ** 2
        )

    def get_latency(
        self, node_id1: int, node_id2: int, propagation_speed: float = 3e8
    ) -> float:
        """Calculates the propagation latency between two nodes.

        Args:
            node_id1 (int): ID of the first node.
            node_id2 (int): ID of the second node.
            propagation_speed (float): Speed of signal propagation. Defaults to 3e8 m/s.

        Returns:
            float: Latency in seconds.
        """
        distance = self.get_distance(node_id1, node_id2)
        return distance / propagation_speed

    def __str__(self) -> str:
        """Returns a string representation of the network topology.

        Returns:
            str: String representation of the network.
        """
        result = ["\nNetwork Topology:"]
        for node_id, neighbors in self.connections.items():
            result.append(f"Node {node_id} -> Neighbors: {neighbors}")
        return "\n".join(result)

    def get_dynamic_changes_by_episode(
        self, start_time: int, end_time: int
    ) -> List[int]:
        """Filters dynamic changes that occurred within a specific time range.

        Args:
            start_time (int): Start time of the interval.
            end_time (int): End time of the interval.

        Returns:
            List[int]: List of times when dynamic changes occurred within the given interval.
        """
        return [
            change
            for change in self.dynamic_change_events
            if start_time <= change <= end_time
        ]
