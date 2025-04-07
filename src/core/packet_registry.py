import datetime
import logging as log
from typing import Dict, Optional

from core.base import SimulationConfig
from core.clock import clock
from core.reports_manager import ReportsManager
from core.network import Network


class PacketRegistry:
    """Global registry to manage the packet log of the network.

    Attributes:
        packet_log (Dict[int, Dict]): Dictionary to store packet logs for each episode.
    """

    def __init__(self) -> None:
        """Initializes the PacketRegistry with an empty packet log."""
        self.packet_log: Dict[int, Dict] = {}
        self.reports_manager: ReportsManager = ReportsManager()
        self.network: Network = None
        self.config: SimulationConfig = None
        self._current_algorithm: str = None
        self._current_episode_number: int = 0
        self._current_episode_start_time: int = 0
        log.debug("PacketRegistry initialized.")

    def log_simulation_start(self, config: SimulationConfig, network: Network) -> None:
        """Inicializa la métrica global de la simulación a partir de SimulationConfig."""
        self.network = network
        self.config = config

        self.metrics = {
            "simulation_id": 1,
            "parameters": {
                "episode_number": config.episodes,
                "max_hops": config.max_hops,
                "algorithms": config.algorithms,
                "mean_disconnection_interval_ms": config.mean_disconnection_interval_ms
                if config.mean_disconnection_interval_ms is not None
                else float("inf"),
                "mean_reconnection_interval_ms": config.mean_reconnection_interval_ms
                if config.mean_reconnection_interval_ms is not None
                else float("inf"),
                "disconnection_interval_ms": config.disconnection_interval_ms
                if config.disconnection_interval_ms is not None
                else float("inf"),
                "reconnection_interval_ms": config.reconnection_interval_ms
                if config.reconnection_interval_ms is not None
                else float("inf"),
                "topology_file": config.topology_file,
                "functions_sequence": [f.value for f in config.functions_sequence],
                "disconnection_probability": config.disconnection_probability,
                "episode_timeout_ms": config.episode_timeout_ms,
            },
            "total_time": None,
            "runned_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self._successful_episodes = {}
        for algorithm in config.algorithms:
            self.metrics[algorithm] = {"success_rate": 0.0, "episodes": []}
            self._successful_episodes[algorithm] = 0

            if algorithm == "Q_ROUTING":
                self.metrics[algorithm]["penalty"] = config.penalty

        log.debug("Initialized simulation metrics with config.")

    def log_algorithm_start(self, algorithm: str) -> None:
        """Prepara el contador e inicializa estructuras internas."""
        self._current_algorithm = algorithm
        self._successful_episodes[algorithm] = 0
        self.packet_log.clear()
        log.debug(f"Started logging for algorithm {algorithm}.")

    def log_episode_start(self, episode_number: int) -> None:
        """Initializes the packet log for a new episode.

        Args:
            episode_number (int): The number of the episode to initialize.
        """
        self._current_episode_number = episode_number
        self._current_episode_start_time = clock.get_current_time()

        if episode_number not in self.packet_log:
            self.packet_log[episode_number] = {
                "episode_success": False,
                "episode_duration": None,
                "route": [],
            }
            log.debug(f"Initialized packet log for episode {episode_number}.")

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

    def log_lost_packet(
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

    def log_complete_episode(self, episode_number: int, episode_success: bool) -> None:
        """Marks an episode as complete in the packet log.

        Args:
            episode_number (int): The number of the episode.
            episode_success (bool): Whether the episode was successful.
        """
        log.debug(
            f"Marking episode {episode_number} as {'successful' if episode_success else 'failed'}."
        )
        self.packet_log[episode_number]["episode_success"] = episode_success

    def log_episode_end(self) -> None:
        episode_number = self._current_episode_number
        algorithm = self._current_algorithm
        start_time = self._current_episode_start_time
        end_time = clock.get_current_time()

        episode_data = self.packet_log.get(episode_number, {})
        episode_success = episode_data.get("episode_success", False)
        route = episode_data.get("route", [])
        total_hops = len(route)

        # TODO: refactorizar esto para usar el success del EpisodeEnded
        if episode_success:
            self._successful_episodes[algorithm] = +1

        dynamic_changes = self.network.get_dynamic_changes_by_episode(
            start_time, end_time
        )

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

        self._current_episode_number = None
        self._current_episode_start_time = None
        log.debug(f"Logged episode {episode_number} for algorithm {algorithm}.")

    def log_algorithm_end(self) -> None:
        """Calcula el success_rate y persiste los resultados para el algoritmo actual."""
        algorithm = self._current_algorithm
        total_episodes = self.config.episodes
        successful = self._successful_episodes[algorithm]

        self.metrics[algorithm]["success_rate"] = (
            successful / total_episodes if total_episodes > 0 else 0.0
        )
        self.metrics[algorithm]["successful_episodes"] = successful

        self._current_algorithm = None

    def log_simulation_end(self) -> None:
        """Marca el final de la simulación y guarda la duración total."""
        self.metrics["total_time"] = clock.get_current_time()
        self.reports_manager.save_metrics_to_file()
        self.reports_manager.save_results_to_excel()
        self.reports_manager.generar_comparative_graphs_from_excel()

        # q_tables = []
        # if algorithm == "Q_ROUTING":
        #     for node in self.network.nodes.values():
        #         q_tables.append(node.application.q_table)
        # self.generate_heat_map(q_tables)

registry: PacketRegistry = PacketRegistry()
