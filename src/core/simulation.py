import logging

from core.base import EpisodeEnded
from core.clock import clock
from core.packet_registry import registry


class Simulation:
    """Manages the execution of the simulation.

    Attributes:
        network (Network): The network to simulate.
        sender_node (Node): The sender node in the network.
        metrics_manager (MetricsManager): The metrics manager to log simulation results.
    """

    def __init__(
        self, network: "Network", sender_node: "Node", metrics_manager: "MetricsManager"
    ) -> None:
        """Initializes the Simulation with the network, sender node, and metrics manager.

        Args:
            network (Network): The network to simulate.
            sender_node (Node): The sender node in the network.
            metrics_manager (MetricsManager): The metrics manager to log simulation results.
        """
        self.network = network
        self.sender_node = sender_node
        self.metrics_manager = metrics_manager
        logging.info("Simulation initialized.")

    def start(self, algorithm_enum: "Algorithm", episodes: int) -> None:
        """Starts the simulation for a specified number of episodes.

        Args:
            algorithm_enum (Algorithm): The algorithm to use for routing.
            episodes (int): The number of episodes to run.
        """
        clock.start()
        self.network.start_dynamic_changes()
        successful_episodes = 0

        algorithm = algorithm_enum.name
        logging.info(f"Starting simulation with {episodes} episodes using {algorithm}.")

        for episode_number in range(1, episodes + 1):
            logging.info(f"=== Starting Episode #{episode_number} ({algorithm}) ===")

            start_time = clock.get_current_time()
            registry.initialize_episode(episode_number)

            try:
                self.sender_node.start_episode(episode_number)
            except EpisodeEnded:
                logging.info(f"=== Episode #{episode_number} ended ===")

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

        clock.stop()
        self.network.stop_dynamic_changes()
        registry.restart_packet_log()

        self.metrics_manager.finalize_simulation(
            clock.get_current_time(), successful_episodes, episodes
        )

        logging.info("Simulation finished and clock stopped.")
