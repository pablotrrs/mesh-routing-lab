import logging as log

from core.base import Algorithm, EpisodeEnded, EpisodeTimeout, SimulationConfig
from core.packet_registry import registry
from core.clock import clock


class Simulation:
    """Manages the execution of the simulation.

    Attributes:
        network (Network): The network to simulate.
        sender_node (Node): The sender node in the network.
    """

    def __init__(self) -> None:
        self.network: Network = None
        self.sender_node: Node = None
        self.config: SimulationConfig = None

    def initialize(self, config, network, sender_node):
        self.network: Network = network
        self.sender_node: Node = sender_node
        self.config: SimulationConfig = config

        registry.initialize_simulation(config)
        log.debug("Simulation initialized.")

    def run(self):
        for algorithm in self.config.algorithms:
            self._run_algorithm(algorithm)

        self._finalize_simulation()

    def _run_algorithm(self, algorithm: Algorithm) -> None:
        """Método privado para configurar y ejecutar un algoritmo"""
        registry.set_current_algorithm(algorithm.name)

        self._setup_algorithm(algorithm)

        clock.start()
        self.network.start_dynamic_changes()

        for episode_number in range(1, self.config.episodes + 1):
            self._run_episode(episode_number)

        clock.stop()
        self.network.stop_dynamic_changes()
        registry.finalize_algorithm()

    def _setup_algorithm(self, algorithm: Algorithm) -> None:
        """Configura las aplicaciones según el algoritmo"""
        match algorithm:
            case Algorithm.Q_ROUTING:
                from algorithms.q_routing import (
                    IntermediateQRoutingApplication,
                    QRoutingApplication,
                    SenderQRoutingApplication,
                )

                self.sender_node.install_application(SenderQRoutingApplication)
                self.sender_node.application.set_params(self.config.max_hops, self.config.functions_sequence)

                if isinstance(self.sender_node.application, QRoutingApplication):
                    self.sender_node.application.set_penalty(self.config.penalty)

                for node_id, node in self.network.nodes.items():
                    if node_id != self.sender_node.node_id:
                        node.install_application(IntermediateQRoutingApplication)
                        node.application.set_params(self.config.max_hops, self.config.functions_sequence)

            case Algorithm.DIJKSTRA:
                from algorithms.dijkstra import (
                    IntermediateDijkstraApplication,
                    SenderDijkstraApplication,
                )

                self.sender_node.install_application(SenderDijkstraApplication)
                self.sender_node.application.set_params(self.config.max_hops, self.config.functions_sequence)

                for node_id, node in self.network.nodes.items():
                    if node_id != self.sender_node.node_id:
                        node.install_application(IntermediateDijkstraApplication)
                        node.application.set_params(self.config.max_hops, self.config.functions_sequence)

            case Algorithm.BELLMAN_FORD:
                from algorithms.bellman_ford import (
                    IntermediateBellmanFordApplication,
                    SenderBellmanFordApplication,
                )

                self.sender_node.install_application(SenderBellmanFordApplication)
                self.sender_node.application.set_params(self.config.max_hops, self.config.functions_sequence)

                for node_id, node in self.network.nodes.items():
                    if node_id != self.sender_node.node_id:
                        node.install_application(IntermediateBellmanFordApplication)
                        node.application.set_params(self.config.max_hops, self.config.functions_sequence)

    def _run_episode(self, episode_number: int) -> None:
        """Ejecuta un episodio individual"""
        registry.initialize_episode(episode_number)

        try:
            self.sender_node.start_episode(episode_number)
            registry.mark_episode_complete(episode_number, success=False)

        except EpisodeEnded as e:
            registry.mark_episode_complete(episode_number, success=e.success)

        except EpisodeTimeout:
            registry.mark_episode_complete(episode_number, success=False)

    def _finalize_simulation(self) -> None:
        """Finaliza toda la simulación"""
        registry.finalize_simulation()
        log.info("Simulation completed")
