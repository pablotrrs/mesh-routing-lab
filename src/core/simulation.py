from core.base import EpisodeEnded
from core.clock import clock
from core.packet_registry import registry


class Simulation:
    def __init__(self, network, sender_node, metrics_manager):
        self.network = network
        self.sender_node = sender_node
        self.metrics_manager = metrics_manager

    def start(self, algorithm_enum, episodes):
        clock.start()
        self.network.start_dynamic_changes()
        successful_episodes = 0

        algorithm = algorithm_enum.name
        for episode_number in range(1, episodes + 1):
            print(f"\n\n=== Starting Episode #{episode_number} ({algorithm}) ===\n")

            start_time = clock.get_current_time()
            registry.initialize_episode(episode_number)

            try:
                self.sender_node.start_episode(episode_number)
            except EpisodeEnded:
                print(f"\n\n=== Episode #{episode_number} ended ===\n")

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

        print("\n[Simulation] Simulation finished and clock stopped.")
