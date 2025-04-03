import datetime
import logging as log
from typing import Dict, List, Optional
from core.metrics_manager import MetricsManager


class PacketRegistry:
    """Global registry to manage the packet log of the network.

    Attributes:
        packet_log (Dict[int, Dict]): Dictionary to store packet logs for each episode.
    """

    def __init__(self) -> None:
        """Initializes the PacketRegistry with an empty packet log."""
        self.packet_log: Dict[int, Dict] = {}
        self.metrics_manager = MetricsManager()
        log.debug("PacketRegistry initialized.")

    def initialize(
        self,
        max_hops: int,
        topology_file: str,
        functions_sequence: List[str],
        mean_disconnection_interval_ms: float,
        mean_reconnection_interval_ms: float,
        disconnection_probability: float,
        algorithms: List[str],
        penalty: float,
    ) -> None:
        """Initializes the metrics for a new simulation with multiple algorithms.

        Args:
            max_hops (int): Maximum number of hops allowed.
            topology_file (str): Path to the topology file.
            functions_sequence (List[str]): Sequence of node functions.
            mean_disconnection_interval_ms (float): Mean interval for dynamic changes.
            mean_reconnection_interval_ms (float): Interval for node reconnection.
            disconnection_probability (float): Probability of node disconnection.
            algorithms (List[str]): List of algorithms used in the simulation.
            penalty (float): Penalty for Q-Routing.
        """
        self.metrics = {
            "simulation_id": 1,
            "parameters": {
                "max_hops": max_hops,
                "algorithms": algorithms,
                "mean_disconnection_interval_ms": mean_disconnection_interval_ms,
                "mean_reconnection_interval_ms": mean_reconnection_interval_ms,
                "topology_file": topology_file,
                "functions_sequence": [func.value for func in functions_sequence],
                "disconnection_probability": disconnection_probability,
            },
            "total_time": None,
            "runned_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        for algorithm in algorithms:
            self.metrics[algorithm] = {"success_rate": 0.0, "episodes": []}

            if algorithm == "Q_ROUTING":
                self.metrics[algorithm]["penalty"] = penalty

        log.debug("Metrics initialized for simulation.")

    def log_episode(
        self,
        algorithm: str,
        episode_number: int,
        start_time: float,
        end_time: float,
        episode_success: bool,
        route: List[str],
        total_hops: int,
        dynamic_changes: List[Dict],
    ) -> None:
        """Logs the metrics of an episode for a specific algorithm.

        Args:
            algorithm (str): The algorithm used in the episode.
            episode_number (int): The number of the episode.
            start_time (float): Start time of the episode.
            end_time (float): End time of the episode.
            episode_success (bool): Whether the episode was successful.
            route (List[str]): The route taken in the episode.
            total_hops (int): Total number of hops in the episode.
            dynamic_changes (List[Dict]): List of dynamic changes during the episode.
        """
        if algorithm not in self.metrics:
            self.metrics[algorithm] = {"success_rate": 0.0, "episodes": []}

        self.metrics[algorithm]["episodes"].append(
            {
                "episode_number": episode_number,
                "start_time": start_time,
                "end_time": end_time,
                "episode_duration": end_time - start_time,
                "episode_success": episode_success,
                "route": route,
                "total_hops": total_hops,
                "dynamic_changes": dynamic_changes,
                "dynamic_changes_count": len(dynamic_changes),
            }
        )

        log.debug(f"Logged episode {episode_number} for algorithm {algorithm}.")

    def finalize_simulation_for_algorithm(
        self, total_time: float, successful_episodes: int, episodes: int
    ) -> None:
        """Finalizes the simulation and saves the results.

        Args:
            total_time (float): Total time taken for the simulation.
            successful_episodes (int): Number of successful episodes.
            episodes (int): Total number of episodes.
        """
        algorithm = self.metrics["parameters"]["algorithms"][-1]
        self.metrics[algorithm]["success_rate"] = (
            successful_episodes / episodes if episodes > 0 else 0.0
        )
        self.metrics["total_time"] = total_time

        self.metrics_manager.save_metrics_to_file()
        self.metrics_manager.save_results_to_excel()
        log.debug("Simulation finalized and results saved.")

    def initialize_episode(self, episode_number: int) -> None:
        """Initializes the packet log for a new episode.

        Args:
            episode_number (int): The number of the episode to initialize.
        """
        if episode_number not in self.packet_log:
            self.packet_log[episode_number] = {
                "episode_success": False,
                "episode_duration": None,
                "route": [],
            }
            log.debug(f"Initialized packet log for episode {episode_number}.")

    def restart_packet_log(self) -> None:
        """Clears the packet log."""
        self.packet_log = {}
        log.debug("Packet log restarted.")

    def mark_packet_lost(
        self,
        episode_number: int,
        from_node_id: int,
        to_node_id: Optional[int],
        packet_type: str,
    ) -> None:
        """Marks a packet as lost in the packet log.

        Args:
            episode_number (int): The number of the episode.
            from_node_id (int): ID of the source node.
            to_node_id (Optional[int]): ID of the destination node (or None if not applicable).
            packet_type (str): Type of the packet.
        """
        log.debug(f"Packet from {from_node_id} to {to_node_id} lost.")
        self.packet_log[episode_number]["route"].append(
            {
                "from": from_node_id,
                "to": to_node_id if to_node_id is not None else "N/A",
                "function": "N/A",
                "node_status": "inactive",
                "latency": 0,
                "packet_type": packet_type,
            }
        )

    def mark_episode_complete(self, episode_number: int, episode_success: bool) -> None:
        """Marks an episode as complete in the packet log.

        Args:
            episode_number (int): The number of the episode.
            episode_success (bool): Whether the episode was successful.
        """
        log.debug(
            f"Marking episode {episode_number} as {'successful' if episode_success else 'failed'}."
        )
        self.packet_log[episode_number]["episode_success"] = episode_success

    def log_packet_hop(
        self,
        episode_number: int,
        from_node_id: int,
        to_node_id: int,
        function: str,
        node_status: str,
        latency: float,
        packet_type: str,
    ) -> None:
        """Logs a packet hop in the packet log.

        Args:
            episode_number (int): The number of the episode.
            from_node_id (int): ID of the source node.
            to_node_id (int): ID of the destination node.
            function (str): Function assigned to the source node.
            node_status (str): Status of the source node.
            latency (float): Latency of the packet hop.
            packet_type (str): Type of the packet.
        """
        self.packet_log[episode_number]["route"].append(
            {
                "from": from_node_id,
                "to": to_node_id,
                "function": function,
                "node_status": node_status,
                "latency": latency,
                "packet_type": packet_type,
            }
        )
        log.debug(
            f"Logged packet hop from {from_node_id} to {to_node_id} in episode {episode_number}."
        )

# Global instance of the PacketRegistry
registry: PacketRegistry = PacketRegistry()
