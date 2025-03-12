import time
import threading
from classes.base import EpisodeEnded
from classes.packet_registry import packet_registry as registry

class Simulation:
    def __init__(self, network, sender_node, metrics_manager):
        self.network = network
        self.clock = 0  # Tiempo en milisegundos
        self.running = False  # Control del reloj
        self.lock = threading.Lock()  # Para sincronizar accesos al reloj
        self.sender_node = sender_node
        self.max_hops = None
        self.metrics_manager = metrics_manager

    def set_max_hops(self, max_hops):
        self.max_hops = max_hops

    def set_functions_sequence(self, functions_sequence):
        self.functions_sequence = functions_sequence

    def start_clock(self):
        """Inicia un hilo dedicado al reloj central."""
        self.running = True
        threading.Thread(target=self._run_clock, daemon=True).start()

    def stop_clock(self):
        """Detiene el hilo del reloj."""
        self.running = False

    def _run_clock(self):
        """Incrementa el reloj centralizado continuamente en milisegundos."""
        while self.running:
            with self.lock:
                self.clock += 1
            time.sleep(0.001)  # 1 ms en tiempo real

    def get_current_time(self):
        """Obtiene el tiempo actual del reloj central."""
        with self.lock:
            return self.clock

    def tick(self):
        """Avanza el reloj centralizado."""
        self.clock += self.time_increment
        return self.clock

    def get_current_time(self):
        """Devuelve el tiempo actual del reloj central."""
        return self.clock

    def start(self, algorithm_enum, episodes, penalty=None):
        algorithm = algorithm_enum.name
        self.network.set_simulation_clock(self)
        self.start_clock()
        self.network.start_dynamic_changes()

        successful_episodes = 0

        for episode_number in range(1, episodes + 1):
            print(f'\n\n=== Starting Episode #{episode_number} ({algorithm}) ===\n')

            start_time = self.get_current_time()
            registry.initialize_episode(episode_number)

            try:
                self.sender_node.start_episode(episode_number, self.max_hops, self.functions_sequence, penalty if algorithm == "Q_ROUTING" else None)
            except EpisodeEnded:
                print(f'\n\n=== Episode #{episode_number} ended ===\n')

            end_time = self.get_current_time()

            episode_data = registry.packet_log.get(episode_number, {})
            episode_success = episode_data.get("episode_success", False)
            route = episode_data.get("route", [])
            total_hops = len(route)
            dynamic_changes = self.network.get_dynamic_changes_by_episode(start_time, end_time)

            self.metrics_manager.log_episode(
                algorithm,
                episode_number,
                start_time,
                end_time,
                episode_success,
                route,
                total_hops,
                dynamic_changes
            )

            if episode_success:
                successful_episodes += 1

        self.stop_clock()
        self.network.stop_dynamic_changes()
        registry.restart_packet_log()

        self.metrics_manager.finalize_simulation(self.get_current_time(), successful_episodes, episodes)

        print("\n[Simulation] Simulation finished and clock stopped.")
