import logging as log

from core.clock import clock
from core.base import Algorithm, EpisodeEnded, EpisodeTimeout, SimulationConfig
from core.node import Node
from core.metrics_manager import MetricsManager
from core.packet_registry import registry


class Simulation:
    """Manages the execution of the simulation.

    Attributes:
        config (SimulationConfig): The simulation configuration.
        network (Network): The network to simulate.
        sender_node (Node): The sender node in the network.
    """

    def __init__(self) -> None:
        """Initializes an empty Simulation."""
        self.config: SimulationConfig = None
        self.network: Network = None
        self.sender_node: Node = None
        self.metrics_manager: MetricsManager = None

    def initialize(
        self, config: SimulationConfig, network: "Network", sender_node: "Node"
    ) -> None:
        """Initializes the simulation with configuration and network.

        Args:
            config (SimulationConfig): Simulation parameters.
            network (Network): The network topology.
            sender_node (Node): The sender node that initiates the episodes.
        """
        self.config = config
        self.network = network
        self.sender_node = sender_node
        self.metrics_manager = MetricsManager()
        self.metrics_manager.initialize(
            max_hops=config.max_hops,
            topology_file=config.topology_file,
            functions_sequence=config.functions_sequence,
            mean_disconnection_interval_ms=config.mean_disconnection_interval_ms,
            mean_reconnection_interval_ms=config.mean_reconnection_interval_ms,
            disconnection_probability=config.disconnection_probability,
            algorithms=[algo.name for algo in config.algorithms],
            penalty=config.penalty,
        )
        log.debug("Simulation initialized with the given configuration.")

    def run(self) -> None:
        """Runs the simulation for all selected algorithms."""
        clock.start()
        self.network.start_dynamic_changes()

        for algorithm in self.config.algorithms:
            successful_episodes = 0

            self._run_algorithm(algorithm, successful_episodes)

            registry.restart_packet_log()

            self.metrics_manager.finalize_simulation(
                clock.get_current_time(), successful_episodes, self.config.episodes
            )

        self._finalize_simulation()

    def _run_algorithm(self, algorithm: Algorithm, successful_episodes: int) -> None:
        """Sets up and runs a specific algorithm.

        Args:
            algorithm (Algorithm): The algorithm to run.
        """
        log.info(f"[{algorithm.name}] Running {self.config.episodes} episodes")
        self._setup_algorithm(algorithm)
        for episode_number in range(1, self.config.episodes + 1):
            self._run_episode(episode_number, successful_episodes, algorithm)

    def _setup_algorithm(self, algorithm: Algorithm) -> None:
        """Installs the appropriate application on nodes based on the algorithm.

        Args:
            algorithm (Algorithm): The algorithm to configure.
        """
        match algorithm:
            case Algorithm.Q_ROUTING:
                from algorithms.q_routing import (
                    IntermediateQRoutingApplication,
                    QRoutingApplication,
                    SenderQRoutingApplication,
                )

                self.sender_node.install_application(SenderQRoutingApplication)
                self.sender_node.application.set_params(
                    self.config.max_hops,
                    self.config.functions_sequence,
                    self.config.episode_timeout_ms,
                )

                for node_id, node in self.network.nodes.items():
                    if node_id != self.sender_node.node_id:
                        node.install_application(IntermediateQRoutingApplication)
                        node.application.set_params(
                            self.config.max_hops,
                            self.config.functions_sequence,
                            self.config.episode_timeout_ms,
                        )

            case Algorithm.DIJKSTRA:
                from algorithms.dijkstra import (
                    IntermediateDijkstraApplication,
                    SenderDijkstraApplication,
                )

                self.sender_node.install_application(SenderDijkstraApplication)
                self.sender_node.application.set_params(
                    self.config.max_hops,
                    self.config.functions_sequence,
                    self.config.episode_timeout_ms,
                )

                for node_id, node in self.network.nodes.items():
                    if node_id != self.sender_node.node_id:
                        node.install_application(IntermediateDijkstraApplication)
                        node.application.set_params(
                            self.config.max_hops,
                            self.config.functions_sequence,
                            self.config.episode_timeout_ms,
                        )

            case Algorithm.BELLMAN_FORD:
                from algorithms.bellman_ford import (
                    IntermediateBellmanFordApplication,
                    SenderBellmanFordApplication,
                )

                self.sender_node.install_application(SenderBellmanFordApplication)
                self.sender_node.application.set_params(
                    self.config.max_hops,
                    self.config.functions_sequence,
                    self.config.episode_timeout_ms,
                )

                for node_id, node in self.network.nodes.items():
                    if node_id != self.sender_node.node_id:
                        node.install_application(IntermediateBellmanFordApplication)
                        node.application.set_params(
                            self.config.max_hops,
                            self.config.functions_sequence,
                            self.config.episode_timeout_ms,
                        )

    def _run_episode(self, episode_number: int, successful_episodes: int, algorithm: Algorithm) -> None:
        """Runs a single episode for the selected algorithm.

        Args:
            episode_number (int): The number of the episode to run.
        """
        log.info(f"[{algorithm}] Starting Episode #{episode_number}")

        start_time = clock.get_current_time()
        registry.initialize_episode(episode_number)

        try:
            self.sender_node.start_episode(episode_number)
        except EpisodeEnded as e:
            log.info(f"[{algorithm}] Episode #{episode_number} ended successfully")
        except EpisodeTimeout:
            log.warning(f"[{algorithm}] Episode #{episode_number} timed out")
        # else: #TODO: dijkstra y bellman ford fallan esta validación, no debería pasar!
        #     error_msg = f"[{algorithm}] Episode #{episode_number} finished without an EpisodeEnded or EpisodeTimeout exception"
        #     log.error(error_msg)
        #     raise ValueError(error_msg)

        end_time = clock.get_current_time()

        episode_data = registry.packet_log.get(episode_number, {})
        episode_success = episode_data.get("episode_success", False)
        route = episode_data.get("route", [])
        total_hops = len(route)
        dynamic_changes = self.network.get_dynamic_changes_by_episode(
            start_time, end_time
        )

        self.metrics_manager.log_episode(
            algorithm,
            episode_number,
            start_time,
            end_time,
            episode_success,
            route,
            total_hops,
            dynamic_changes,
        )

        if episode_success:
            successful_episodes += 1

    def _finalize_simulation(self) -> None:
        """Finalizes the simulation and generates reports."""
        log.debug("Finalizing simulation.")
        clock.stop()
        self.network.stop_dynamic_changes()
