import logging as log

from core.base import Algorithm, EpisodeEnded, EpisodeTimeout, SimulationConfig
from core.clock import clock
from core.node import Node
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

    def run(self) -> None:
        """Runs the simulation for all selected algorithms."""
        log.info("Simulation started")
        registry.log_simulation_start(self.config, self.network)
        clock.start()
        self.network.start_dynamic_changes()

        for algorithm in self.config.algorithms:
            registry.log_algorithm_start(algorithm)
            self._run_algorithm(algorithm)
            registry.log_algorithm_end()

        clock.stop()
        self.network.stop_dynamic_changes()
        registry.log_simulation_end()
        log.info("Simulation finished")

    def _run_algorithm(self, algorithm: Algorithm) -> None:
        """Sets up and runs a specific algorithm.

        Args:
            algorithm (Algorithm): The algorithm to run.
        """
        log.info(f"[{algorithm.name}] Running {self.config.episodes} episodes")
        self._setup_algorithm(algorithm)
        for episode_number in range(1, self.config.episodes + 1):
            self._run_episode(episode_number, algorithm)

        log.info(
            f"[{algorithm}] Finished running {self.config.episodes} episodes"
        )

    def _run_episode(self, episode_number: int, algorithm: Algorithm) -> None:
        """Runs a single episode for the selected algorithm.

        Args:
            episode_number (int): The number of the episode to run.
        """
        log.info(f"[{algorithm}] Starting Episode #{episode_number}")

        registry.log_episode_start(episode_number)

        try:
            self.sender_node.start_episode(episode_number)
        except EpisodeEnded as e:
            log.info(f"[{algorithm}] Episode #{episode_number} ended successfully")
        except EpisodeTimeout:
            log.warning(f"[{algorithm}] Episode #{episode_number} timed out")
        # else:
        #     error_msg = f"[{algorithm}] Episode #{episode_number} finished without an EpisodeEnded or EpisodeTimeout exception"
        #     log.error(error_msg)
        #     raise ValueError(error_msg)

        registry.log_episode_end()

    def _setup_algorithm(self, algorithm: Algorithm) -> None:
        """Installs the appropriate application on nodes based on the algorithm.

        Args:
            algorithm (Algorithm): The algorithm to configure.
        """
        match algorithm:
            case Algorithm.Q_ROUTING:
                from algorithms.q_routing import (
                    IntermediateQRoutingApplication,
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
