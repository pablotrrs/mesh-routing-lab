import random
import time
import logging as log
from dataclasses import dataclass
from collections import deque
from enum import Enum
import threading
from tabulate import tabulate
from utils.thread_killer import kill_thread
from typing import Optional
import math
import numpy as np

from core.clock import clock
from core.base import Application, EpisodeEnded, EpisodeTimeout
from core.packet_registry import registry
from utils.visualization import print_q_table
from utils.custom_excep_hook import custom_thread_excepthook

EPISODE_TIMEOUT_TRIGGERED = False

SMALL_BONUS = -0.1
BIG_BONUS = -50

DEFAULT_ESTIMATE = 20000.0

# More aggressive learning
ALPHA = 0.15  # Increase learning rate
EPSILON_DECAY_RATE = 0.995  # Faster epsilon decay
MIN_EPISODES_FOR_CONVERGENCE = 80  # Earlier convergence check

GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.999976975
EPSILON_MIN = 0.1

# Enhanced convergence parameters
EPSILON_START = 0.9
EPSILON_END = 0.01
CONVERGENCE_THRESHOLD = 0.02

CURRENT_HOP_COUNT = 0

RETRY_BASE_DELAY_MS = 50

EPISODE_COMPLETED = False

CALLBACK_STACK = deque()

# Ecuación de Bellman:
# ΔQ_x(d, y) = α * (s + t - Q_x(d, y))
# Donde:
# - s: Tiempo de transmisión, calculado como la diferencia entre el timestamp guardado y el tiempo actual.
# - t: Tiempo estimado restante que el nodo calculó cuando eligió el vecino al que enviaría el paquete.
# - Q_x(d, y): Valor Q actual para el nodo actual (x) hacia el vecino (y).
# - α (alpha): Tasa de aprendizaje.
BELLMAN_EQ = lambda s, t, q_current: q_current + ALPHA * (s + t - q_current)


class PacketType(str, Enum):
    PACKET_HOP = "PACKET_HOP"
    CALLBACK = "CALLBACK"
    MAX_HOPS_REACHED = "MAX_HOPS_REACHED"


@dataclass
class CallbackChainStep:
    previous_hop_node: int
    current_node: int
    next_hop_node: int
    send_timestamp: float
    estimated_time: float
    episode_number: int


class QRoutingApplication(Application):
    def __init__(self, node):
        self.node = node
        self.q_table = {}
        self.assigned_function = None
        self.callback_stack = deque()
        self.performance_history = []
        self.converged = False
        self.convergence_episode = None
        self._last_selections = [] 

        # Default convergence parameters (will be overridden by config)
        self.convergence_success_rate = 0.7
        self.convergence_epsilon_threshold = 0.5
        self.convergence_min_episodes = 80
        self.convergence_min_history = 10
        self.epsilon_start = 0.9
        self.epsilon_end = 0.01
        self.epsilon_decay_rate = 0.995
        
        # Initialize epsilon with default (will be updated by config)
        self.epsilon = self.epsilon_start
        self.performance_history = []
        self.converged = False
        self.convergence_episode = None
        self._last_selections = []

    def set_convergence_params(self, config):
        """Set convergence parameters from simulation config"""
        self.convergence_success_rate = config.convergence_success_rate
        self.convergence_epsilon_threshold = config.convergence_epsilon_threshold
        self.convergence_min_episodes = config.convergence_min_episodes
        self.convergence_min_history = config.convergence_min_history
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_rate = config.epsilon_decay_rate
        
        # Reset epsilon with new start value
        self.epsilon = self.epsilon_start
        
        log.info(f"[Node_ID={self.node.node_id}] Convergence parameters set:")
        log.info(f"  - Success rate threshold: {self.convergence_success_rate}")
        log.info(f"  - Epsilon threshold: {self.convergence_epsilon_threshold}")
        log.info(f"  - Min episodes: {self.convergence_min_episodes}")
        log.info(f"  - Min history: {self.convergence_min_history}")
        log.info(f"  - Epsilon range: {self.epsilon_start} → {self.epsilon_end}")
        log.info(f"  - Decay rate: {self.epsilon_decay_rate}")

    def ensure_not_timeout(self):
        global EPISODE_TIMEOUT_TRIGGERED
        if EPISODE_TIMEOUT_TRIGGERED:
            log.warning(f"[Node_ID={self.node.node_id}] Episode aborted due to timeout.")
            raise EpisodeTimeout()

    def receive_packet(self, packet):
        self.ensure_not_timeout()
        log.debug(f"[Node_ID={self.node.node_id}] Received packet {packet}")

        if packet["type"] == PacketType.PACKET_HOP:
            self.handle_packet_hop(packet)
            return
        elif packet["type"] == PacketType.CALLBACK:
            self.handle_echo_callback(packet)
            return
        elif packet["type"] == PacketType.MAX_HOPS_REACHED:
            self.handle_lost_packet(packet)
            return

    def handle_packet_hop(self, packet):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def handle_echo_callback(self, packet):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def handle_lost_packet(self, packet):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_epsilon(self, episode_number, episode_success):
        """Enhanced epsilon update with convergence tracking"""
        
        log.info(f"[Node_ID={self.node.node_id}] DEBUG: update_epsilon called with episode={episode_number}, success={episode_success}")
        
        # Try to get episode success if not provided
        if episode_success is None and episode_number is not None and episode_number > 1:
            from core.packet_registry import registry
            log.info(f"[Node_ID={self.node.node_id}] DEBUG: Looking for episode {episode_number-1} success")
            
            # Try to find previous episode data
            algorithm_data = registry.results.get("Q_ROUTING", {})
            episodes_data = algorithm_data.get("episodes", [])
            log.info(f"[Node_ID={self.node.node_id}] DEBUG: Found {len(episodes_data)} episodes in registry")
            
            previous_episode = episode_number - 1
            for episode_data in episodes_data:
                if episode_data.get("episode_number") == previous_episode:
                    episode_success = episode_data.get("episode_success", None)
                    log.info(f"[Node_ID={self.node.node_id}] DEBUG: Found episode {previous_episode} success: {episode_success}")
                    break
        
        # Track performance
        if episode_success is not None:
            self.performance_history.append(1 if episode_success else 0)
            log.info(f"[Node_ID={self.node.node_id}] DEBUG: Added to history. Total entries: {len(self.performance_history)}")
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history.pop(0)
        
        if (episode_number is not None and 
            episode_number > self.convergence_min_episodes and 
            len(self.performance_history) >= self.convergence_min_history):
            
            recent_performance = self.performance_history[-self.convergence_min_history:]
            success_rate = sum(recent_performance) / len(recent_performance)
            performance_variance = np.var(recent_performance)
            
            log.info(f"[Node_ID={self.node.node_id}] CONVERGENCE CHECK:")
            log.info(f"  - Episode: {episode_number}")
            log.info(f"  - Success rate: {success_rate:.3f} (threshold: {self.convergence_success_rate})")
            log.info(f"  - Epsilon: {self.epsilon:.3f} (threshold: {self.convergence_epsilon_threshold})")
            
            # Use configurable thresholds
            if (success_rate > self.convergence_success_rate and 
                self.epsilon < self.convergence_epsilon_threshold and 
                episode_number > self.convergence_min_episodes):
                
                if not self.converged:
                    self.converged = True
                    self.convergence_episode = episode_number
                    log.info(f"[Node_ID={self.node.node_id}] ✅ CONVERGED at episode {episode_number}")
                    
                    # Track convergence in reports manager
                    from core.reports_manager import reports_manager
                    reports_manager.track_convergence_metrics(
                        episode_number, 
                        self.node.node_id, 
                        self.epsilon, 
                        self.converged, 
                        success_rate
                    )
                return
        
        # Normal decay if not converged using configurable rate
        if not self.converged:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay_rate)
            log.debug(f"[Node_ID={self.node.node_id}] Epsilon updated to {self.epsilon:.4f}")

    def check_network_convergence(self, episode_number):
        """Check if entire network has converged"""
        if episode_number % 50 == 0:  # Check every 50 episodes
            from core.reports_manager import reports_manager
            
            if hasattr(reports_manager, 'convergence_data'):
                converged_nodes = sum(1 for data in reports_manager.convergence_data.values() 
                                    if data.get('convergence_episode') is not None)
                total_nodes = len(reports_manager.convergence_data)
                
                if converged_nodes >= total_nodes * 0.8:  # 80% of nodes converged
                    log.info(f"[Episode {episode_number}] Network-wide convergence achieved!")
                    log.info(f"Converged nodes: {converged_nodes}/{total_nodes}")
                    
                    # Optional: Further reduce exploration across network
                    if not self.converged:
                        self.epsilon = EPSILON_END
                        log.info(f"[Node_ID={self.node.node_id}] Network convergence triggered final epsilon reduction")

    def update_q_value(self, next_node, s, t, function_id: str):
        """
        Actualiza el valor Q para el nodo actual y la acción (saltar al vecino `next_node`)
        usando la ecuación de Bellman.
        """
        self.ensure_not_timeout()

        current_node_id = self.node.node_id
        
        # Initialize if needed
        if current_node_id not in self.q_table:
            self.q_table[current_node_id] = {}
        if function_id not in self.q_table[current_node_id]:
            self.q_table[current_node_id][function_id] = {}
        
        # Get current Q-value
        old_q = self.q_table[current_node_id][function_id].get(next_node, DEFAULT_ESTIMATE)
        
        # Bellman update: Q ← Q + α[s + t - Q]
        new_q = old_q + ALPHA * (s + t - old_q)
        
        # Store updated value
        self.q_table[current_node_id][function_id][next_node] = new_q
        
        # Log the update
        registry.log_q_table_value_update(current_node_id, next_node, old_q, new_q, t, s)

        return

    def select_next_node(self, function_id: str, episode_number=None, episode_success=None) -> int:
        """Enhanced node selection with convergence tracking and loop detection"""
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()
        
        from core.packet_registry import registry
        from core.reports_manager import reports_manager
        
        # Update epsilon based on performance if data is available
        if episode_number is not None:
            self.update_epsilon(episode_number, episode_success)
        
        # Track convergence metrics every 20 episodes
        if episode_number is not None and episode_number % 20 == 0:
            # Calculate success rate from recent episodes
            success_count = 0
            total_episodes = min(20, episode_number)
            
            for ep in range(max(1, episode_number - 19), episode_number + 1):
                episode_data = registry.packet_log.get(ep, {})
                if episode_data.get("episode_success", False):
                    success_count += 1
            
            success_rate = success_count / total_episodes if total_episodes > 0 else 0.0
            
            # Track convergence metrics
            reports_manager.track_convergence_metrics(
                episode_number, 
                self.node.node_id, 
                self.epsilon, 
                self.converged, 
                success_rate
            )

            self.check_network_convergence(episode_number)

        retry_count = 0
        current_node_id = self.node.node_id

        while True:
            self.ensure_not_timeout()
            next_node = None

            all_neighbors = self.node.network.get_neighbors(current_node_id)
            log.debug(f"[Node_ID={current_node_id}] All neighbors: {all_neighbors}")

            active_neighbors = [
                neighbor for neighbor in all_neighbors
                if self.node.network.get_node(neighbor).status and neighbor != current_node_id
            ]
            log.debug(f"[Node_ID={current_node_id}] Active neighbors: {active_neighbors}")


            self._last_selections.append(current_node_id)
            if len(self._last_selections) > 8:
                self._last_selections.pop(0)

            if len(self._last_selections) >= 8:
                recent_selections = self._last_selections[-6:]
                unique_recent = set(recent_selections)
                
                if len(unique_recent) <= 2:
                    pattern_detected = False
                    
                    if len(unique_recent) == 2:
                        nodes = list(unique_recent)
                        if (recent_selections[0] == recent_selections[2] == recent_selections[4] and
                            recent_selections[1] == recent_selections[3] == recent_selections[5]):
                            pattern_detected = True
                    
                    elif len(unique_recent) == 1:
                        if all(node == recent_selections[0] for node in recent_selections):
                            pattern_detected = True
                    
                    if pattern_detected:
                        log.warning(f"[Node_ID={current_node_id}] Loop pattern detected: {recent_selections}")
                        log.warning(f"[Node_ID={current_node_id}] Forcing exploration to break loop")
                        
                        if active_neighbors:
                            available_neighbors = [n for n in active_neighbors if n not in unique_recent]
                            if available_neighbors:
                                next_node = random.choice(available_neighbors)
                            else:
                                next_node = random.choice(active_neighbors)
                            
                            log.info(f"[Node_ID={current_node_id}] Loop-breaking exploration selected Node {next_node}")
                            self._last_selections = []
                            
                            # Update epsilon only if not converged
                            if not self.converged:
                                self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_end)
                            return next_node

            if random.random() < self.epsilon:
                log.debug(f"[Node_ID={current_node_id}] Performing exploration with epsilon={self.epsilon:.4f}")
                registry.log_policy_decision("EXPLORATION", self.epsilon)
                if active_neighbors:
                    next_node = random.choice(active_neighbors)
                    log.debug(f"[Node_ID={current_node_id}] Exploration selected Node {next_node}")
                else:
                    log.debug(f"[Node_ID={current_node_id}] No active neighbors available for exploration.")
            else:
                log.debug(f"[Node_ID={current_node_id}] Performing exploitation with epsilon={self.epsilon:.4f}")
                registry.log_policy_decision("EXPLOITATION", self.epsilon)
                next_node = self.choose_best_action(function_id)
                log.debug(f"[Node_ID={current_node_id}] Exploitation chose {next_node}")

                if next_node is not None and next_node == current_node_id:
                    log.debug(f"[Node_ID={current_node_id}] Exploitation selected self node. Invalid.")
                    next_node = None
                elif next_node is not None and not self.node.network.get_node(next_node).status:
                    log.debug(f"[Node_ID={current_node_id}] Exploitation selected inactive node {next_node}.")
                    next_node = None

                if next_node is None and active_neighbors:
                    next_node = random.choice(active_neighbors)
                    log.debug(f"[Node_ID={current_node_id}] Fallback to exploration selected Node {next_node}")
                elif next_node is None:
                    log.debug(f"[Node_ID={current_node_id}] Fallback to exploration found no valid neighbors.")

            if next_node is not None:
                # Update epsilon only if not converged
                if not self.converged:
                    self.epsilon = max(self.epsilon * self.epsilon_decay_rate, self.epsilon_end)
                log.debug(f"[Node_ID={current_node_id}] Returning next node: {next_node}")
                return next_node

            self.ensure_not_timeout()
            delay_ms = RETRY_BASE_DELAY_MS * (2 ** retry_count)
            delay_ms = min(delay_ms, 10000)
            log.debug(f"[Node_ID={current_node_id}] No valid next node found. Retrying in {delay_ms}ms...")
            time.sleep(delay_ms / 1000)
            retry_count += 1

    def choose_best_action(self, function_id: str) -> Optional[int]:
        """
        Selecciona el mejor vecino para alcanzar algún nodo que pueda ejecutar la función dada,
        utilizando Q-routing adaptado a entornos orientados a funciones.
        """
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()
        current_node_id = self.node.node_id

        if current_node_id not in self.q_table:
            return None
        if function_id not in self.q_table[current_node_id]:
            return None
        
        # Find neighbor with minimum Q-value for this function
        best_neighbor = min(
            self.q_table[current_node_id][function_id].items(),
            key=lambda x: x[1]
        )[0]

        return best_neighbor

    def initialize_or_update_q_table(self) -> None:
        self.ensure_not_timeout()

        current_node_id = self.node.node_id
        if current_node_id not in self.q_table:
            self.q_table[current_node_id] = {}

        for neighbor_id in self.node.network.get_neighbors(current_node_id):
            neighbor_node = self.node.network.get_node(neighbor_id)
            raw_function = neighbor_node.get_assigned_function()

            # Filtramos nodos sin función asignada
            if not raw_function or raw_function == "N/A":
                continue

            # Si es un enum NodeFunction, convertimos a string
            function_id = raw_function.value if hasattr(raw_function, "value") else raw_function

            if function_id not in self.q_table[current_node_id]:
                self.q_table[current_node_id][function_id] = {}
                
            if neighbor_id not in self.q_table[current_node_id][function_id]:
                self.q_table[current_node_id][function_id][neighbor_id] = DEFAULT_ESTIMATE
                log.debug(f"[Q-Table Init] ({current_node_id} | {function_id} → {neighbor_id}) = {DEFAULT_ESTIMATE}")

        log.info("self.q_table")
        log.info(self.q_table)

    def estimate_remaining_time(self, next_node, function_id) -> float:
        """
        Estima el tiempo restante para alcanzar un nodo que procese la función `function_id`
        a través del vecino `next_node`. Si no hay valores Q asociados, retorna infinito.
        """
        self.ensure_not_timeout()

        log.info(f'self.q_table {self.q_table}')

        current_node_id = self.node.node_id
        
        # CORRECTED STRUCTURE ACCESS:
        if (current_node_id not in self.q_table or 
            function_id not in self.q_table[current_node_id] or
            next_node not in self.q_table[current_node_id][function_id]):
            return DEFAULT_ESTIMATE

        return self.q_table[current_node_id][function_id][next_node]

    def update_q_table_with_incomplete_info(
        self, next_node: int, function_id: str, estimated_time_remaining: float
    ) -> None:
        """
        Actualiza la Q-table para el salto (self.node -> next_node) y la función objetivo `function_id`,
        usando información incompleta sobre el tiempo restante estimado.
        """
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()

        current_node_id = self.node.node_id
        
        # CORRECTED STRUCTURE ACCESS:
        current_q = DEFAULT_ESTIMATE
        if (current_node_id in self.q_table and 
            function_id in self.q_table[current_node_id] and
            next_node in self.q_table[current_node_id][function_id]):
            current_q = self.q_table[current_node_id][function_id][next_node]

        # Validaciones
        def is_invalid(value):
            return (
                value is None
                or isinstance(value, dict)
                or isinstance(value, str)
                or (isinstance(value, float) and math.isnan(value))
            )

        if is_invalid(estimated_time_remaining):
            log.warning(f"[WARNING] estimated_time_remaining tiene valor inválido: {estimated_time_remaining} (tipo: {type(estimated_time_remaining)})")
            raise ValueError("estimated_time_remaining no es un float válido")

        if is_invalid(current_q):
            log.warning(f"[WARNING] current_q tiene valor inválido: {current_q} (tipo: {type(current_q)})")
            raise ValueError("current_q no es un float válido")

        log.info("\n" + tabulate([
            ["Current node", current_node_id],
            ["Next node", next_node],
            ["Function ID", function_id],
            ["Estimated time remaining", estimated_time_remaining],
            ["Current Q-value", current_q],
        ], headers=["Q-routing update (incomplete info)", "Valor"], tablefmt="fancy_grid"))

        # Ecuación de actualización
        updated_q = current_q + ALPHA * (estimated_time_remaining - current_q)

        # Inicializar subtablas si hiciera falta
        if current_node_id not in self.q_table:
            self.q_table[current_node_id] = {}
        if function_id not in self.q_table[current_node_id]:
            self.q_table[current_node_id][function_id] = {}

        # para ser compliants con si es el NodeFunction o un str crudo
        raw_function = function_id.value if hasattr(function_id, "value") else function_id

        # Guardar nuevo valor
        self.q_table[current_node_id][function_id][next_node] = updated_q

        log.info(f"✅ Updated Q-value: {updated_q}")

        # if hop_processes_correct_function:
        #
        #     global SMALL_BONUS
        #     log.debug(
        #         f"[Node_ID={self.node.node_id}] Applying bonus {SMALL_BONUS} for hop to Node {next_node} (processed correct function)"
        #     )
        #     updated_q += SMALL_BONUS
        #
        # self.q_table[self.node.node_id][next_node] = updated_q

        registry.log_q_table_value_update(
            current_node_id, next_node, current_q, updated_q, estimated_time_remaining, None
        )

    def is_invalid(value):
        return (
            value is None
            or isinstance(value, dict)
            or isinstance(value, str)
            or (isinstance(value, float) and (math.isnan(value) or math.isinf(value)))
        )

    def initiate_max_hops_callback(self, packet):
        self.ensure_not_timeout()
        global CALLBACK_STACK

        # si no tiene callback stack porque puede pasar
        # porque nunca salió del 0 -> n hop

        episode_data = registry.packet_log.get(packet["episode_number"], {})
        route = episode_data.get("route", [])

        if not CALLBACK_STACK:
            if self.didnt_make_it_further_than_first_hop(route):
                self.send_packet(packet["from_node_id"], callback_packet)

        callback_packet = {
            "type": PacketType.MAX_HOPS_REACHED,
            "episode_number": packet["episode_number"],
            "from_node_id": self.node.node_id,
            "hops": packet["hops"],
        }

        callback_data = CALLBACK_STACK.pop()
        log.debug(
            f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m"
        )

        # self.penalize_q_value(
        #     next_node=callback_data.next_hop_node, penalty=packet["penalty"]
        # )

        # movement: backward
        self.send_packet(callback_data.previous_hop_node, callback_packet)
        return

    def didnt_make_it_further_than_first_hop(self, route):
        self.ensure_not_timeout()
        from_values = [step["from"] for step in route]

        from_counts = {}
        for value in from_values:
            from_counts[value] = from_counts.get(value, 0) + 1

        if len(from_counts) == 2:
            if (
                0 in from_counts and from_counts[0] == 1
            ):
                other_value = [value for value in from_counts if value != 0][
                    0
                ]
                if (
                    from_counts[other_value] == len(route) - 1
                ):
                    return True
        return False

    def penalize_q_value(self, next_node, penalty):
        """
        Penaliza el valor Q de la acción (ir al vecino `next_node`) con un aumento fuerte.
        """
        self.ensure_not_timeout()
        old_q = self.q_table[self.node.node_id].get(next_node, 0.0)
        new_q = old_q + penalty

        self.q_table[self.node.node_id][next_node] = max(
            new_q, 0
        )

        log.debug(
            f"[Node_ID={self.node.node_id}] Penalized by {penalty} Q-Value for state {self.node.node_id} -> action {next_node} "
            f"from {old_q:.4f} to {new_q:.4f} (hard penalty applied)"
        )
        return

    def get_assigned_function(self):
        """Returns the function assigned to this node."""
        self.ensure_not_timeout()
        assigned_function = self.assigned_function

        return assigned_function.value if assigned_function is not None else "N/A"

    def __str__(self) -> str:
        return f"Node(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"

    def __repr__(self) -> str:
        return self.__str__()

def log_nodos_y_vecinos(network, function_sequence=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
    from collections import defaultdict

    rows = []

    for node_id, node in network.nodes.items():
        function = node.get_assigned_function() or "N/A"
        vecinos = [
            str(other_id)
            for other_id in network.nodes
            if other_id != node_id and network.is_node_reachable(node_id, other_id)
        ]
        vecinos_str = ", ".join(vecinos)
        rows.append([node_id, function, vecinos_str])

    table = tabulate(rows, headers=["Nodo", "Función", "Vecinos alcanzables"], tablefmt="fancy_grid")
    log.info("\n\n===== Estado de la Red al inicio del episodio =====")
    log.info(table)
    log.info(f"\nSecuencia objetivo de funciones: {' -> '.join(function_sequence)}\n")

    # Merge de todas las Q-tables en una global
    q_table_global = defaultdict(lambda: defaultdict(dict))

    for node_id, node in network.nodes.items():
        # if hasattr(node, "q_table"):
        local_q_table = node.application.q_table
        for src, neighbors in local_q_table.items():
            for neighbor, function_map in neighbors.items():
                for function, q_value in function_map.items():
                    q_table_global[src][neighbor][function] = q_value

    # Mostrar Q-table global
    log.info("===== Q-Table Global (Origen -> Vecino → Función) =====")
    for src, neighbors in q_table_global.items():
        log.info(f"\n[Desde Nodo {src}]")
        for neighbor, functions in neighbors.items():
            for function, q_value in functions.items():
                log.info(f"  → Hacia {neighbor} | Función '{function}' | Q = {q_value:.2f}")

class SenderQRoutingApplication(QRoutingApplication):
    def __init__(self, node):
        super().__init__(node)
        self.max_hops = None
        self.functions_sequence = None
        self.penalty = 0.0

    def set_penalty(self, penalty):
        self.penalty = penalty

    def _get_previous_episode_success(self, current_episode):
        """Get success status of previous episode for convergence tracking"""
        if current_episode <= 1:
            return None
        
        from core.packet_registry import registry
        previous_episode = current_episode - 1
        episode_data = registry.packet_log.get(previous_episode, {})
        return episode_data.get("episode_success", None)

    def start_episode(self, episode_number: int) -> None:
        """Initiates an episode by creating a packet and sending it asynchronously."""

        global EPISODE_COMPLETED
        EPISODE_COMPLETED = False

        global EPISODE_TIMEOUT_TRIGGERED
        EPISODE_TIMEOUT_TRIGGERED = False

        log_nodos_y_vecinos(self.node.network)

        self.episode_start_time = clock.get_current_time()

        episode_thread = threading.Thread(target=self._process_episode, args=(episode_number, 0))
        timeout_watcher_thread = threading.Thread(target=self._monitor_timeout, args=(episode_thread, episode_number))

        threading.excepthook = custom_thread_excepthook

        log.debug(f"[Episode #{episode_number}] Starting episode thread.")
        episode_thread.start()

        log.debug(f"[Episode #{episode_number}] Starting timeout watcher thread.")
        timeout_watcher_thread.start()

        # ✅ Esperar que termine el episodio
        episode_thread.join()

        if timeout_watcher_thread.is_alive():
            timeout_watcher_thread.join()

        log.debug(f"[Episode #{episode_number}] Episode fully handled (thread joined and timeout watcher done).")

        EPISODE_TIMEOUT_TRIGGERED = False

    def _process_episode(self, episode_number: int, current_hop_count: int) -> None:
        """Core logic for processing an episode, runs asynchronously."""
        try:
            global EPISODE_COMPLETED
            EPISODE_COMPLETED = False

            log.debug(f"\n\033[93mClearing callback stacks for Episode {episode_number}\033[0m")
            global CALLBACK_STACK
            CALLBACK_STACK.clear()

            packet = {
                "type": PacketType.PACKET_HOP,
                "episode_number": episode_number,
                "from_node_id": self.node.node_id,
                "functions_sequence": self.functions_sequence.copy(),
                "function_counters": {func: 0 for func in self.functions_sequence},
                "hops": current_hop_count,
                "max_hops": self.max_hops,
                "is_delivered": False,
                "penalty": self.penalty,
            }

            self.initialize_or_update_q_table()

            episode_success = self._get_previous_episode_success(episode_number)
            next_node = self.select_next_node(
                function_id=packet["functions_sequence"][0].value,
                episode_number=episode_number, 
                episode_success=episode_success
            )

            if next_node is None:
                log.debug(
                    f"[Node_ID={self.node.node_id}] No valid next node found. Can't initiate episode!."
                )
                # movement: none
                packet["hops"] += 1
                registry.log_lost_packet(
                    packet["episode_number"], packet["from_node_id"], None, packet["type"]
                )
                log.debug(f'[Node_ID={self.node.node_id}] Packet hop count {packet["hops"]}')

                if packet["hops"] > self.max_hops:
                    registry.log_episode_failure_reason("MAX_HOPS")
                    self.mark_episode_result(packet, success=False)

                # si no se puede empezar el episodio, se sigue intentando hasta que se pueda
                self._process_episode(episode_number, packet["hops"])
                return
            else:
                # estimated_time_remaining = self.estimate_remaining_time(next_node)
                # estimated_time_remaining = list(self.estimate_remaining_time(next_node, packet["functions_sequence"][0]).values())[0]
                estimated_time_remaining = self.estimate_remaining_time(next_node, packet["functions_sequence"][0])

                self.update_q_table_with_incomplete_info(
                    next_node=next_node, function_id=packet["functions_sequence"][0].value ,estimated_time_remaining=estimated_time_remaining
                )

                # movement: forward
                self.send_packet(next_node, packet)
                return

        except EpisodeEnded as e:
            log.debug(f"[Sender Node] Episode ended with success={e.success}")
            raise e

        except EpisodeTimeout as e:
            log.warning(f"[Sender Node] Episode timed out!")
            raise e

    def _monitor_timeout(self, episode_thread: threading.Thread, episode_number: int) -> None:
        """Continuously monitors the timeout and kills the episode thread if exceeded."""
        if self.episode_timeout_ms is None or self.episode_start_time is None:
            return

        global EPISODE_TIMEOUT_TRIGGERED
        EPISODE_TIMEOUT_TRIGGERED = False

        while episode_thread.is_alive():
            current_time = clock.get_current_time()
            elapsed_time = current_time - self.episode_start_time

            if elapsed_time >= self.episode_timeout_ms:
                log.debug(f"[Sender Node] Timeout reached after {elapsed_time} ms. Terminating episode...")
                registry.log_episode_failure_reason("TIMEOUT")

                EPISODE_TIMEOUT_TRIGGERED = True
                # kill_thread(episode_thread)
                # log.info(f"[Episode #{episode_number}] Episode forcefully terminated due to timeout.")
                return

            import time
            time.sleep(0.001)

    def handle_packet_hop(self, packet) -> None:
        self.ensure_not_timeout()
        episode_number = packet["episode_number"]
        episode_success = self._get_previous_episode_success(episode_number)
        next_node = self.select_next_node(
            function_id=packet["functions_sequence"][0],
            episode_number=episode_number, 
            episode_success=episode_success
        )

        if next_node is None:
            log.debug(
                f"[Node_ID={self.node.node_id}] No valid next node found. Stopping packet hop."
            )
            # movement: none
            packet["hops"] += 1
            registry.log_lost_packet(
                packet["episode_number"], packet["from_node_id"], None, packet["type"]
            )
            if packet["hops"] > packet["max_hops"]:
                # max hops reached
                registry.log_episode_failure_reason("MAX_HOPS")
                self.mark_episode_result(packet, success=False)
            else:
                # retry until there is a valid next node
                self.handle_packet_hop(packet)
                return

        # estimated_time_remaining = self.estimate_remaining_time(next_node)
        # estimated_time_remaining = list(self.estimate_remaining_time(next_node).values())[0]
        estimated_time_remaining = self.estimate_remaining_time(next_node, packet["functions_sequence"][0])

        self.update_q_table_with_incomplete_info(
            next_node=next_node, function_id=packet["functions_sequence"][0] ,estimated_time_remaining=estimated_time_remaining
        )

        callback_chain_step = CallbackChainStep(
            previous_hop_node=packet["from_node_id"],
            current_node=self.node.node_id,
            next_hop_node=next_node,
            send_timestamp=clock.get_current_time(),
            estimated_time=estimated_time_remaining,
            episode_number=packet["episode_number"],
        )

        global CALLBACK_STACK
        CALLBACK_STACK.append(callback_chain_step)
        log.debug(
            f"\033[92m[CALLBACK_STACK] Encolando {callback_chain_step}, callback_stack: {CALLBACK_STACK}\033[0m"
        )

        log.debug(
            f"[Node_ID={self.node.node_id}] Adding step to callback chain stack: {callback_chain_step}"
        )

        # movement: forward
        if not self.send_packet(next_node, packet):
            # max hops reached
            self.initiate_max_hops_callback(packet)
            return
        return

    def handle_echo_callback(self, packet) -> None:
        self.ensure_not_timeout()
        global CALLBACK_STACK
        if len(CALLBACK_STACK) > 0:
            callback_data = CALLBACK_STACK.pop()
            log.debug(
                f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m"
            )

            log.info(f"clock.get_current_time() = {clock.get_current_time()}")
            log.info(f"callback_data.send_timestamp = {callback_data.send_timestamp}")
            log.info(f"clock.get_current_time() - callback_data.send_timestamp = {clock.get_current_time() - callback_data.send_timestamp}")
            log.info(f"callback_data.estimated_time = {callback_data.estimated_time}")

            # TODO: podríamos guardar esto en el callback_data directamente
            function_id = self.node.network.get_node(callback_data.next_hop_node).get_assigned_function()
            log.info(f"function_id = {function_id}")

            self.update_q_value(
                next_node=callback_data.next_hop_node,
                s=clock.get_current_time() - callback_data.send_timestamp,
                t=callback_data.estimated_time,
                function_id=function_id
            )

            # movement: backward
            self.send_packet(callback_data.previous_hop_node, packet)
            return
        else:
            print_q_table(self)
            log.debug(
                f'\n[Node_ID={self.node.node_id}] Episode {packet["episode_number"]} finished.'
            )

            self.mark_episode_result(packet, success=True)

    def handle_max_hops_reached(self, packet) -> None:
        self.ensure_not_timeout()
        episode_number = packet["episode_number"]
        log.debug(f"\n[Node_ID={self.node.node_id}] Episode {episode_number} failed.")

        registry.log_episode_failure_reason("MAX_HOPS")
        self.mark_episode_result(packet, success=False)

    def mark_episode_result(self, packet, success=True):
        """
        Marca un episodio como exitoso o fallido y lo registra en el registry global.

        Args:
            packet (dict): El paquete asociado al episodio.
            success (bool): `True` si el episodio fue exitoso, `False` si falló.
        """
        self.ensure_not_timeout()
        global EPISODE_COMPLETED
        EPISODE_COMPLETED = True

        status_text = "SUCCESS" if success else "FAILURE"
        episode_number = packet["episode_number"]
        log.debug(
            f"\n[Node_ID={self.node.node_id}] Marking Episode {episode_number} as {status_text}."
        )

        registry.log_complete_episode(episode_number, success)

        global CURRENT_HOP_COUNT
        CURRENT_HOP_COUNT = 0
        raise EpisodeEnded(success)

    def __str__(self) -> str:
        return f"SenderNode(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"


class IntermediateQRoutingApplication(QRoutingApplication):
    def __init__(self, node):
        super().__init__(node)

    def start_episode(self, episode_number):
        raise NotImplementedError(
            "Intermediate node is not supposed to start an episode"
        )

    def _get_previous_episode_success(self, current_episode):
        """Get success status of previous episode for convergence tracking"""
        if current_episode <= 1:
            return None
        
        from core.packet_registry import registry
        previous_episode = current_episode - 1
        episode_data = registry.packet_log.get(previous_episode, {})
        return episode_data.get("episode_success", None)

    def handle_packet_hop(self, packet):
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()

        if packet["hops"] > packet["max_hops"]:
            global EPISODE_COMPLETED

            log.debug(f"episode completion {EPISODE_COMPLETED}")
            if EPISODE_COMPLETED:
                episode_number = packet["episode_number"]
                log.debug(
                    f"\033[91m[Node_ID={self.node.node_id}] Episode {episode_number} already ended or not found. Ignoring packet.\033[0m"
                )
                return
            log.debug(
                f"[Node_ID={self.node.node_id}] Max hops reached. Initiating full echo callback"
            )
            self.initiate_max_hops_callback(packet)
            return

        if self.assigned_function is None:
            self.assign_function(packet)

        hop_processes_correct_function = False

        # if function to process is function assigned to this node
        if (
            self.assigned_function == packet["functions_sequence"][0]
            if packet["functions_sequence"]
            else None
        ):
            log.debug(
                f"[Node_ID={self.node.node_id}] Removing function {self.assigned_function} from functions to process"
            )
            hop_processes_correct_function = True
            if packet["functions_sequence"]:
                packet["functions_sequence"].pop(0)

        # if all functions have been processed
        if len(packet["functions_sequence"]) == 0:
            log.debug(
                f"[Node_ID={self.node.node_id}] Function sequence is complete! Initiating full echo callback"
            )

            # from_node = packet["from_node_id"]
            # to_node = self.node.node_id
            #
            # if from_node not in self.q_table:
            #     self.q_table[from_node] = {}
            #
            # if to_node not in self.q_table[from_node]:
            #     self.q_table[from_node][to_node] = 0.0
            #
            # self.q_table[from_node][to_node] = self.q_table[from_node][to_node] - 50

            self.initiate_full_echo_callback(packet)
            return

        episode_number = packet["episode_number"]
        episode_success = self._get_previous_episode_success(episode_number)
        next_node = self.select_next_node(
            function_id=packet["functions_sequence"][0],
            episode_number=episode_number, 
            episode_success=episode_success
        )

        log.debug(f"[Node_ID={self.node.node_id}] Next node is {next_node}")
        if next_node is not None:
            # estimated_time_remaining = self.estimate_remaining_time(next_node)
            estimated_time_remaining = self.estimate_remaining_time(next_node, packet["functions_sequence"][0])

            callback_chain_step = CallbackChainStep(
                previous_hop_node=packet["from_node_id"],
                current_node=self.node.node_id,
                next_hop_node=next_node,
                send_timestamp=clock.get_current_time(),
                estimated_time=estimated_time_remaining,
                episode_number=packet["episode_number"],
            )
            global CALLBACK_STACK
            CALLBACK_STACK.append(callback_chain_step)
            log.debug(
                f"\033[92m[CALLBACK_STACK] Encolando {callback_chain_step}, callback_stack: {CALLBACK_STACK}\033[0m"
            )

            log.debug(
                f"[Node_ID={self.node.node_id}] Adding step to callback chain stack: {callback_chain_step}"
            )

            # estimated_time_remaining = list(self.estimate_remaining_time(next_node).values())[0]
            estimated_time_remaining = self.estimate_remaining_time(next_node, packet["functions_sequence"][0])

            self.update_q_table_with_incomplete_info(
                next_node=next_node,
                function_id=packet["functions_sequence"][0],
                estimated_time_remaining=estimated_time_remaining,
                # hop_processes_correct_function=hop_processes_correct_function
            )

            # FIXME: está habiendo algo acá que está haciendo que corte el episodio sin que tenga que cortar
            # movement: forward
            self.send_packet(next_node, packet)
        else:
            # movement: none
            packet["hops"] += 1
            registry.log_lost_packet(
                packet["episode_number"], packet["from_node_id"], None, packet["type"]
            )
            still_hops_remaining = packet["hops"] < packet["max_hops"]

            if not still_hops_remaining:
                # max hops reached
                self.initiate_max_hops_callback(packet)
                return
            else:
                # retry until there is a valid next node
                self.handle_packet_hop(packet)
                return

    def handle_echo_callback(self, packet):
        """Maneja el callback cuando regresa el paquete."""
        self.ensure_not_timeout()
        global CALLBACK_STACK

        while CALLBACK_STACK:
            callback_data = CALLBACK_STACK.pop()
            log.debug(
                f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m"
            )

            if self.node.node_id == callback_data.next_hop_node:
                log.warning(f"[Node_ID={self.node.node_id}] Ignoring invalid callback step with same current and next hop: {callback_data}")
                continue  # Seguimos buscando en el stack

            log.info(f"callback_data = {callback_data}")
            log.info(f"clock.get_current_time() = {clock.get_current_time()}")
            log.info(f"callback_data.send_timestamp = {callback_data.send_timestamp}")
            log.info(f"clock.get_current_time() - callback_data.send_timestamp = {clock.get_current_time() - callback_data.send_timestamp}")
            log.info(f"callback_data.estimated_time = {callback_data.estimated_time}")

            function_id = self.node.network.get_node(callback_data.next_hop_node).get_assigned_function()
            log.info(f"function_id = {function_id}")

            # Si encontramos uno válido:
            self.update_q_value(
                next_node=callback_data.next_hop_node,
                s=clock.get_current_time() - callback_data.send_timestamp,
                t=callback_data.estimated_time,
                function_id=function_id
            )

            # movement: backward
            self.send_packet(callback_data.previous_hop_node, packet)
            return

        # Si no quedó nada válido en el stack:
        log.error(f"[Node_ID={self.node.node_id}] Callback stack exhausted without valid steps. Episode will be aborted.")
        self.mark_episode_result(packet, success=False)

    def handle_lost_packet(self, packet) -> None:
        self.ensure_not_timeout()
        global CALLBACK_STACK
        callback_data = CALLBACK_STACK.pop()
        log.debug(
            f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m"
        )

        # self.penalize_q_value(
        #     next_node=callback_data.next_hop_node, penalty=packet["penalty"]
        # )

        # movement: backward
        self.send_packet(callback_data.previous_hop_node, packet)
        return

    def initiate_full_echo_callback(self, packet):
        """Inicia el proceso de full echo callback hacia el nodo anterior."""
        self.ensure_not_timeout()

        callback_packet = {
            "type": PacketType.CALLBACK,
            "episode_number": packet["episode_number"],
            "from_node_id": self.node.node_id,
        }

        # movement: backward
        self.send_packet(packet["from_node_id"], callback_packet)
        return

    def assign_function(self, packet):
        """Asigna la función menos utilizada basada en los contadores del paquete."""
        self.ensure_not_timeout()
        min_assignments = min(packet["function_counters"].values())

        least_assigned_functions = [
            func
            for func, count in packet["function_counters"].items()
            if count == min_assignments
        ]

        if len(least_assigned_functions) == 1:
            function_to_assign = least_assigned_functions[0]
        else:
            function_to_assign = random.choice(least_assigned_functions)

        log.debug(
            f"[Node_ID={self.node.node_id}] Node has no function, assigning function {function_to_assign}"
        )
        self.assigned_function = function_to_assign
        packet["function_counters"][function_to_assign] += 1
        return

    def update_q_value_with_reward(self, from_node, to_node, reward):
        """Actualiza el valor Q entre dos nodos usando una recompensa directa."""
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()

        old_q = self.q_table[from_node].get(to_node, 0.0)
        new_q = old_q + ALPHA * (reward - old_q)

        self.q_table[from_node][to_node] = new_q
        return
        #
        # registry.log_q_table_value_update(
        #     from_node,
        #     to_node,
        #     old_q,
        #     new_q,
        #     reward,
        #     None  # no hay "estimated time" acá, es reward puro
        # )

    def __str__(self):
        return f"IntermediateNode(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"
