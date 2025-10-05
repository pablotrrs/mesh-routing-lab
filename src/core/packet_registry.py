import datetime
import logging as log
from typing import Dict, Optional

from core.base import SimulationConfig
from core.clock import clock
from core.reports_manager import reports_manager
from core.network import Network


class PacketRegistry:
    """Global registry to manage the packet log of the network.

    Attributes:
        packet_log (Dict[int, Dict]): Dictionary to store packet logs for each episode.
    """

    def __init__(self) -> None:
        """Initializes the PacketRegistry with an empty packet log."""
        self.packet_log: Dict[int, Dict] = {}
        self.network: Network = None
        self.config: SimulationConfig = None
        self._current_algorithm: str = None
        self._current_episode_number: int = 0
        self._current_episode_start_time: int = 0
        self._episode_failure_reason = None
        self._pending_policy_decision = None
        self._q_value = None
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
        hop_data = {
            "from": from_node_id,
            "to": to_node_id,
            "function": function,
            "node_status": node_status,
            "latency": latency,
            "packet_type": packet_type,
        }

        if self._pending_policy_decision is not None:
            hop_data.update(self._pending_policy_decision)
            self._pending_policy_decision = None

        if self._q_value is not None:
            hop_data.update(self._q_value)
            self._q_value = None

        self.packet_log[episode_number]["route"].append(hop_data)
        log.debug(
            f"Logged packet hop from {from_node_id} to {to_node_id} in episode {episode_number}."
        )

    def log_policy_decision(self, decision: str, epsilon: float) -> None:
        """Registra si el hop fue por exploración o explotación y el valor de epsilon usado.

        Args:
            decision (str): 'EXPLORATION' o 'EXPLOITATION'
            epsilon (float): Valor de epsilon en ese momento
        """
        if decision not in {"EXPLORATION", "EXPLOITATION"}:
            log.warning(f"Invalid policy decision: {decision}")
            return
        self._pending_policy_decision = {"policy_decision": decision, "epsilon": epsilon}
        log.debug(
            f"[Episode {self._current_episode_number}] Set decision: {decision}, epsilon: {epsilon:.4f}"
        )

    def log_q_table_value_update(
        self,
        node_id: int,
        next_node: int,
        old_q: float,
        new_q: float,
        actual_time: float,
    ) -> None:
        """Registra el nuevo q value obtenido para el nodo.

        Args:
            node_id (int): ID del nodo actual.
            next_node (int): ID del siguiente nodo.
            old_q (float): Valor Q anterior.
            new_q (float): Nuevo valor Q.
            estimated_time (float): Tiempo estimado.
            actual_time (float): Tiempo real.
        """
        self._q_value = {"q_value": new_q}

        log.debug(
            f"[Node_ID={node_id}] Updated Q-Value for state {node_id} -> action {next_node} "
            f"from {old_q:.4f} to {new_q:.4f} (actual time {actual_time})"
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

    def log_episode_failure_reason(self, reason: str) -> None:
        """Guarda el motivo del fallo del episodio actual (solo si falla).

        Args:
            reason (str): Razón específica del fallo que será mapeada a categorías estándar.
        """
        # Mapeo de razones específicas a categorías estándar para mejor análisis
        reason_mapping = {
            "ROUTING_LOOP": "MAX_HOPS",  # Los bucles son conceptualmente similares a exceso de hops
            "TOO_MANY_BOUNCES": "MAX_HOPS",  # Demasiados rebotes también
            "INVALID_NEXT_NODE": "NO_ROUTE",  # Nodo inválido significa no hay ruta válida
            "INVALID_PREVIOUS_HOP": "NO_ROUTE",  # Hop anterior inválido también
            "NETWORK_DISCONNECTED": "NO_ROUTE",  # Red desconectada = sin ruta
            "CALLBACK_STACK_EMPTY": "NO_ROUTE",  # Stack vacío = problema de ruta
            "UNKNOWN_FAILURE": "NO_ROUTE"  # Fallos desconocidos como problema de ruta
        }
        
        # Aplicar mapeo si la razón específica tiene una categoría estándar
        mapped_reason = reason_mapping.get(reason, reason)
        
        # Razones válidas estándar
        valid_reasons = {"MAX_HOPS", "TIMEOUT", "RETRY_EXHAUSTED", "NETWORK_PARTITIONED", "NO_ROUTE"}
        
        # Si después del mapeo sigue siendo inválida, registrar como NO_ROUTE por defecto
        if mapped_reason not in valid_reasons:
            log.warning(f"Invalid failure reason: {reason}, mapping to NO_ROUTE")
            mapped_reason = "NO_ROUTE"
            
        self._episode_failure_reason = mapped_reason
        log.debug(f"[Episode {self._current_episode_number}] Registered failure reason: {reason} -> {mapped_reason}")

    def log_retry_attempt(self, node_id: int, operation: str, attempt: int, max_attempts: int) -> None:
        """Registra intentos de reintento para análisis de vulnerabilidad."""
        if self._current_episode_number not in self.packet_log:
            return
        
        if "retry_attempts" not in self.packet_log[self._current_episode_number]:
            self.packet_log[self._current_episode_number]["retry_attempts"] = []
        
        self.packet_log[self._current_episode_number]["retry_attempts"].append({
            "node_id": node_id,
            "operation": operation,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "timestamp": clock.get_current_time()
        })
        
        log.debug(f"[Node {node_id}] Retry attempt {attempt}/{max_attempts} for {operation}")

    def log_timeout_cause(self, cause: str) -> None:
        """Registra la causa específica del timeout."""
        if self._current_episode_number not in self.packet_log:
            return
        
        self.packet_log[self._current_episode_number]["timeout_cause"] = cause
        log.debug(f"[Episode {self._current_episode_number}] Timeout cause: {cause}")

    def log_episode_end(self) -> None:
        episode_number = self._current_episode_number
        algorithm = self._current_algorithm
        start_time = self._current_episode_start_time
        current_time = clock.get_current_time()
        
        # Obtener datos del episodio
        episode_data = self.packet_log.get(episode_number, {})
        
        # Si hay failure_reason de TIMEOUT, limitar end_time al timeout configurado
        if self._episode_failure_reason == "TIMEOUT":
            # Para timeouts, el tiempo de fin debe ser exactamente start_time + timeout
            end_time = start_time + self.config.episode_timeout_ms
        else:
            end_time = current_time

        episode_success = episode_data.get("episode_success", False)
        route = episode_data.get("route", [])
        total_hops = len(route)

        # TODO: refactorizar esto para usar el success del EpisodeEnded
        if episode_success:
            self._successful_episodes[algorithm] += 1

        dynamic_changes = self.network.get_dynamic_changes_by_episode(
            start_time, end_time
        )

        episode_entry = {
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

        # Agregar retry_attempts si existen
        if "retry_attempts" in episode_data:
            episode_entry["retry_attempts"] = episode_data["retry_attempts"]

        if not episode_success and self._episode_failure_reason:
            episode_entry["failure_reason"] = self._episode_failure_reason

        self.metrics[algorithm]["episodes"].append(episode_entry)

        self._current_episode_number = None
        self._current_episode_start_time = None
        self._episode_failure_reason = None
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
        reports_manager.config = self.config
        reports_manager.generate_reports()

    @property
    def current_episode(self) -> int:
        """Retorna el número del episodio actual para estabilización Q-learning."""
        return self._current_episode_number

    @current_episode.setter
    def current_episode(self, value: int) -> None:
        """Establece el número del episodio actual."""
        self._current_episode_number = value

registry: PacketRegistry = PacketRegistry()
