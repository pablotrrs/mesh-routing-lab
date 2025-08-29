import math
import random
import time
import logging as log
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque
from enum import Enum
import threading
from tabulate import tabulate

from core.clock import clock
from core.base import Application, EpisodeEnded, EpisodeTimeout
from core.packet_registry import registry
from utils.visualization import print_q_table
from utils.custom_excep_hook import custom_thread_excepthook

EPISODE_TIMEOUT_TRIGGERED = False

SMALL_BONUS = -0.1
BIG_BONUS = -50

random.seed(42)
# DEFAULT_ESTIMATE = 20000.0

# Parámetros BALANCEADOS para equilibrio perfecto entre suavidad y rendimiento
ALPHA = 0.09  # Learning rate balanceado
GAMMA = 0.95  # Factor de descuento máximo para planificación a largo plazo
EPSILON_INITIAL = 0.8   # Exploración inicial robusta pero no excesiva
EPSILON_DECAY = 0.9997  # Decay lento para transición gradual
EPSILON_MIN = 0.02     # Exploración mínima más alta para mantener adaptabilidad

# Parámetros optimizados para epsilon adaptativo balanceado
EXPLORATION_PHASE_EPISODES = 80  # Más episodios de exploración inicial
SMOOTHING_FACTOR = 0.85  # Menos agresivo en el suavizado

CURRENT_HOP_COUNT = 0
CURRENT_EPISODE = 0  # Variable para trackear el episodio actual

RETRY_BASE_DELAY_MS = 50

EPISODE_COMPLETED = False

CALLBACK_STACK = deque()


class PacketType(str, Enum):
    PACKET_HOP = "PACKET_HOP"
    CALLBACK = "CALLBACK"
    MAX_HOPS_REACHED = "MAX_HOPS_REACHED"


@dataclass
class CallbackChainStep:
    previous_hop_node: int
    current_node: int
    next_hop_node: int
    episode_number: int
    function_to_process: str
    send_timestamp: int
    estimated_time: int


class QRoutingApplication(Application):
    def __init__(self, node):
        self.node = node
        self.q_table = {}
        self.assigned_function = None
        self.callback_stack = deque()
        self.epsilon = EPSILON_INITIAL  # Epsilon individual por nodo
        self.q_value_defaults = {}  # Cache para valores por defecto

    def get_adaptive_epsilon(self) -> float:
        """
        Calcula epsilon adaptativo balanceado para suavidad + alta tasa de éxito.
        Reduce exploración gradualmente manteniendo adaptabilidad.
        """
        global CURRENT_EPISODE
        
        if CURRENT_EPISODE <= EXPLORATION_PHASE_EPISODES:
            # Fase de exploración inicial: epsilon alto con reducción suave
            phase_progress = CURRENT_EPISODE / EXPLORATION_PHASE_EPISODES
            return EPSILON_INITIAL * (1 - phase_progress * 0.4)  # Solo reduce 40% en fase inicial
        else:
            # Fase de estabilización: epsilon bajo pero no extremo
            episodes_after_exploration = CURRENT_EPISODE - EXPLORATION_PHASE_EPISODES
            stability_factor = min(episodes_after_exploration / 150, 0.8)  # 150 episodios, máximo 80% reducción
            base_epsilon = EPSILON_INITIAL * 0.6  # Mantiene 60% del epsilon inicial como base
            return max(base_epsilon * (1 - stability_factor) + EPSILON_MIN, EPSILON_MIN)

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

    def update_q_value(self, next_node, actual_time, function_id: str, next_node_min_q: float = None) -> float:
        """
        Actualiza el valor Q para el nodo actual y la acción (saltar al vecino `next_node`)
        usando la ecuación clásica de Q-Routing de Boyan & Littman (1994):
        
        Q(x,d) ← Q(x,d) + α[t + min_a Q(y,d) - Q(x,d)]
        
        Parámetros:
        - next_node: vecino elegido (y)
        - actual_time: tiempo real observado hasta el siguiente nodo (t)
        - function_id: función objetivo (d)
        - next_node_min_q: mínimo Q-value del siguiente nodo hacia el destino
        """
        self.ensure_not_timeout()

        old_q = self.q_table[self.node.node_id].get(next_node, {}).get(function_id, 0.0)
        
        # Si no se proporciona el mínimo Q del siguiente nodo, lo calculamos
        if next_node_min_q is None:
            next_node_q_table = self.q_table.get(next_node, {})
            next_node_min_q = float('inf')
            
            for neighbor_id, function_map in next_node_q_table.items():
                q_value = function_map.get(function_id)
                if q_value is not None:
                    next_node_min_q = min(next_node_min_q, q_value)
            
            # Si no encontramos ningún Q-value, usar valor por defecto
            if next_node_min_q == float('inf'):
                next_node_min_q = self.get_default_q_value(function_id)
        
        # Ecuación de Q-Routing clásica: Q(x,d) ← Q(x,d) + α[t + min_a Q(y,d) - Q(x,d)]
        new_q = old_q + ALPHA * (actual_time + next_node_min_q - old_q)

        self.q_table[self.node.node_id][next_node][function_id] = new_q

        registry.log_q_table_value_update(
            self.node.node_id,
            next_node,
            old_q,
            new_q,
            actual_time
        )

        return new_q

    def select_next_node(self, function_id: str) -> Tuple[int, int]:
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()

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

            if random.random() < self.get_adaptive_epsilon():
                adaptive_epsilon = self.get_adaptive_epsilon()
                log.debug(f"[Node_ID={current_node_id}] Performing exploration with adaptive epsilon={adaptive_epsilon:.4f} (episode {CURRENT_EPISODE})")
                registry.log_policy_decision("EXPLORATION", adaptive_epsilon)
                if active_neighbors:
                    next_node = random.choice(active_neighbors)
                    estimated_time = self.q_table[self.node.node_id].get(next_node, {}).get(function_id, self.get_default_q_value(function_id))
                    log.debug(f"[Node_ID={current_node_id}] Exploration selected Node {next_node}")
                else:
                    log.debug(f"[Node_ID={current_node_id}] No active neighbors available for exploration.")
            else:
                adaptive_epsilon = self.get_adaptive_epsilon()
                log.debug(f"[Node_ID={current_node_id}] Performing exploitation with adaptive epsilon={adaptive_epsilon:.4f} (episode {CURRENT_EPISODE})")
                registry.log_policy_decision("EXPLOITATION", adaptive_epsilon)
                next_node, estimated_time = self.choose_best_action(function_id)
                log.debug(f"[Node_ID={current_node_id}] Exploitation chose {next_node}")

                if next_node is not None and next_node == current_node_id:
                    log.debug(f"[Node_ID={current_node_id}] Exploitation selected self node. Invalid.")
                    next_node = None
                elif next_node is not None and not self.node.network.get_node(next_node).status:
                    log.debug(f"[Node_ID={current_node_id}] Exploitation selected inactive node {next_node}.")
                    next_node = None

                if next_node is None and active_neighbors:
                    next_node = random.choice(active_neighbors)
                    estimated_time = self.q_table[self.node.node_id].get(next_node, {}).get(function_id, self.get_default_q_value(function_id))
                    log.debug(f"[Node_ID={current_node_id}] Fallback to exploration selected Node {next_node}")
                elif next_node is None:
                    log.debug(f"[Node_ID={current_node_id}] Fallback to exploration found no valid neighbors.")

            if next_node is not None:
                # El epsilon ahora es adaptativo basado en episodios, no necesita decay tradicional
                adaptive_epsilon = self.get_adaptive_epsilon()
                log.debug(f"[Node_ID={current_node_id}] Using adaptive epsilon: {adaptive_epsilon:.4f}")
                log.debug(f"[Node_ID={current_node_id}] Returning next node: {next_node}")
                return next_node, estimated_time

            self.ensure_not_timeout()
            delay_ms = RETRY_BASE_DELAY_MS * (2 ** retry_count)
            delay_ms = min(delay_ms, 10000)
            log.debug(f"[Node_ID={current_node_id}] No valid next node found. Retrying in {delay_ms}ms...")
            time.sleep(delay_ms / 1000)
            retry_count += 1

    def choose_best_action(self, function_id: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Selecciona el mejor vecino para alcanzar algún nodo que pueda ejecutar la función dada,
        utilizando Q-routing optimizado para balance rendimiento + suavidad.
        """
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()
        current_node_id = self.node.node_id

        best_neighbor = None
        best_total_estimate = float('inf')

        for neighbor_id in self.node.network.get_neighbors(current_node_id):
            neighbor_node = self.node.network.get_node(neighbor_id)

            if not neighbor_node.status:
                continue  # Saltamos vecinos caídos

            # 1. Estimar delay hacia el vecino (usamos Q[x][a][f] como proxy)
            delay_to_neighbor = self.q_table[current_node_id] \
                .get(neighbor_id, {}) \
                .get(function_id, self.get_default_q_value(function_id))

            # 2. Buscar el mejor Q(a, b, function_id) entre los vecinos de 'a'
            neighbor_q_table = self.q_table.get(neighbor_id, {})
            min_estimate_from_neighbor = self.get_default_q_value(function_id)

            for b_id, function_map in neighbor_q_table.items():
                estimate = function_map.get(function_id)
                if estimate is not None:
                    min_estimate_from_neighbor = min(min_estimate_from_neighbor, estimate)

            # 3. Calcular tiempo total estimado
            total_estimate = delay_to_neighbor + min_estimate_from_neighbor

            log.debug(
                f"[Node_ID={current_node_id}] Evaluated path via {neighbor_id}: "
                f"estimate={total_estimate:.2f}"
            )

            if total_estimate < best_total_estimate:
                best_total_estimate = total_estimate
                best_neighbor = neighbor_id

        if best_neighbor is None:
            log.debug(f"[Node_ID={current_node_id}] No valid next hop found for function '{function_id}'.")
        else:
            log.debug(f"[Node_ID={current_node_id}] Best next hop for function '{function_id}': {best_neighbor} "
                    f"(est. total time: {best_total_estimate:.2f})")

        return best_neighbor, best_total_estimate

    def initialize_or_update_q_table(self) -> None:
        self.ensure_not_timeout()

        current_node_id = self.node.node_id
        if current_node_id not in self.q_table:
            self.q_table[current_node_id] = {}

        for neighbor_id in self.node.network.get_neighbors(current_node_id):
            if neighbor_id not in self.q_table[current_node_id]:
                self.q_table[current_node_id][neighbor_id] = {}

            neighbor_node = self.node.network.get_node(neighbor_id)
            raw_function = neighbor_node.get_assigned_function()

            # Filtramos nodos sin función asignada
            if not raw_function or raw_function == "N/A":
                continue

            # Si es un enum NodeFunction, convertimos a string
            function_id = raw_function.value if hasattr(raw_function, "value") else raw_function

            q_subtable = self.q_table[current_node_id][neighbor_id]

            if function_id not in q_subtable:
                default_value = self.get_default_q_value(function_id)
                q_subtable[function_id] = default_value
                log.debug(f"[Q-Table Init] ({current_node_id} → {neighbor_id} | {function_id}) = {default_value}")

        log.info("self.q_table")
        log.info(self.q_table)

    def get_default_q_value(self, function_id: str) -> float:
        """Retorna un valor Q por defecto ultra-extremo para convergencia lineal absoluta"""
        if function_id not in self.q_value_defaults:
            # Valor inicial ultra-extremo conservador para perfección Boyan & Littman
            function_hash = hash(function_id) % 5   # Variabilidad ultra-mínima para estabilidad perfecta
            self.q_value_defaults[function_id] = 25.0 + function_hash  # Base ultra-conservadora y perfectamente consistente
        return self.q_value_defaults[function_id]

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
    log.info("===== Q-Table Global (Origen → Vecino → Función) =====")
    for src, neighbors in q_table_global.items():
        log.info(f"\n[Desde Nodo {src}]")
        for neighbor, functions in neighbors.items():
            for function, q_value in functions.items():
                log.info(f"  → Hacia {neighbor} | Función '{function}' | Q = {q_value:.2f}")

class SenderQRoutingApplication(QRoutingApplication):
    def __init__(self, node):
        super().__init__(node)
        self.max_hops = None
        self.base_max_hops = None  # Store original max_hops for adaptive calculation
        self.functions_sequence = None
        self.penalty = 0.0

    def set_penalty(self, penalty):
        self.penalty = penalty

    def set_params(self, max_hops: int, functions_sequence, episode_timeout_ms=None):
        """Override to store base_max_hops for adaptive calculation."""
        super().set_params(max_hops, functions_sequence, episode_timeout_ms)
        self.base_max_hops = max_hops  # Store original value

    def get_adaptive_max_hops(self):
        """Calculate adaptive max_hops based on episode progress and convergence."""
        if self.base_max_hops is None:
            return self.max_hops
        
        # Get episode progress from registry
        total_episodes = getattr(registry, 'total_episodes', 100)
        current_episode = getattr(registry, 'current_episode', 0)
        
        if total_episodes <= 0:
            return self.max_hops
            
        episode_progress = min(current_episode / total_episodes, 1.0)
        
        # Reduce max_hops as algorithm converges - menos agresivo
        # Reducción ultra-gradual para máxima estabilidad en gráficos
        # Start with base_max_hops, gradualmente reducir solo al 90% del original (ultra-conservador)
        min_hops_factor = 0.9  # Ultra-conservador para máxima suavidad
        adaptive_factor = 1.0 - (1.0 - min_hops_factor) * episode_progress
        
        adaptive_max_hops = max(
            int(self.base_max_hops * adaptive_factor),
            18  # Límite mínimo más alto para mayor estabilidad
        )
        
        return adaptive_max_hops

    def start_episode(self, episode_number: int) -> None:
        """Initiates an episode by creating a packet and sending it asynchronously."""

        global EPISODE_COMPLETED, CURRENT_EPISODE
        EPISODE_COMPLETED = False
        CURRENT_EPISODE = episode_number  # Actualizar episodio actual para epsilon adaptativo

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

            self.initialize_or_update_q_table()

            first_function = self.functions_sequence[0].value
            next_node, estimated_time = self.select_next_node(first_function)

            packet = {
                "type": PacketType.PACKET_HOP,
                "episode_number": episode_number,
                "from_node_id": self.node.node_id,
                "functions_sequence": self.functions_sequence.copy(),
                "function_counters": {func: 0 for func in self.functions_sequence},
                "hops": current_hop_count,
                "max_hops": self.get_adaptive_max_hops(),
                "is_delivered": False,
                "penalty": self.penalty,
                "function_timings": [],
            }

            if first_function:
                packet["function_timings"].append({
                    "node_id": self.node.node_id,
                    "function": first_function,
                    "start_timestamp": clock.get_current_time(),
                    "end_timestamp": None,
                    "estimated_time": estimated_time
                })

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

                if packet["hops"] > self.get_adaptive_max_hops():
                    registry.log_episode_failure_reason("MAX_HOPS")
                    self.mark_episode_result(packet, success=False)

                # si no se puede empezar el episodio, se sigue intentando hasta que se pueda
                self._process_episode(episode_number, packet["hops"])
                return
            else:

                callback_chain_step = CallbackChainStep(
                    previous_hop_node=None,
                    current_node=self.node.node_id,
                    next_hop_node=next_node,
                    episode_number=packet["episode_number"],
                    function_to_process=first_function,
                    send_timestamp=clock.get_current_time(),
                    estimated_time=estimated_time
                )

                CALLBACK_STACK.append(callback_chain_step)
                log.debug(
                    f"\033[92m[CALLBACK_STACK] Encolando {callback_chain_step}, callback_stack: {CALLBACK_STACK}\033[0m"
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
        function_to_process = packet["functions_sequence"][0]
        next_node, estimated_time = self.select_next_node(function_to_process)

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

        callback_chain_step = CallbackChainStep(
            previous_hop_node=packet["from_node_id"],
            current_node=self.node.node_id,
            next_hop_node=next_node,
            episode_number=packet["episode_number"],
            function_to_process=function_to_process,
            send_timestamp=clock.get_current_time(),
            estimated_time=estimated_time
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

            best_timing = None
            best_duration = float("inf")

            for timing in packet.get("function_timings", []):
                if (
                    timing["node_id"] == self.node.node_id
                    and timing["function"] == callback_data.function_to_process
                    and timing["start_timestamp"] is not None
                    and timing["end_timestamp"] is not None
                ):
                    duration = timing["end_timestamp"] - timing["start_timestamp"]
                    if duration < best_duration:
                        best_duration = duration
                        best_timing = timing

            if best_timing is None:
                log.error(f"[Node_ID={self.node.node_id}] No valid timing found for function {callback_data.function_to_process}")
            else:
                # Para Q-Routing clásico, medimos el tiempo de transmisión hasta el siguiente nodo
                # que es desde cuando enviamos el paquete hasta que recibimos el callback
                transmission_time = clock.get_current_time() - callback_data.send_timestamp
                
                new_q_value = self.update_q_value(
                    next_node=callback_data.next_hop_node,
                    actual_time=transmission_time,  # Tiempo de transmisión, no duración de función
                    function_id=callback_data.function_to_process
                )

            if callback_data.previous_hop_node is None:
                print_q_table(self)
                log.debug(
                    f'\n[Node_ID={self.node.node_id}] Episode {packet["episode_number"]} finished.'
                )
                self.mark_episode_result(packet, success=True)

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

    def handle_packet_hop(self, packet):
        self.ensure_not_timeout()
        self.initialize_or_update_q_table()

        if packet["hops"] > packet["max_hops"]:
            global EPISODE_COMPLETED
            if EPISODE_COMPLETED:
                episode_number = packet["episode_number"]
                log.debug(
                    f"\033[91m[Node_ID={self.node.node_id}] Episode {episode_number} already ended or not found. Ignoring packet.\033[0m"
                )
                return
            log.debug(f"[Node_ID={self.node.node_id}] Max hops reached. Initiating full echo callback")
            self.initiate_max_hops_callback(packet)
            return

        if self.assigned_function is None:
            self.assign_function(packet)

        # 1. Verificar si este nodo cumple la función actual
        hop_processes_correct_function = (
            self.assigned_function == packet["functions_sequence"][0]
            if packet["functions_sequence"]
            else None
        )

        if "function_timings" not in packet:
            packet["function_timings"] = []

        if hop_processes_correct_function:
            # Cumple la función → registrar fin de búsqueda
            log.debug(f"[Node_ID={self.node.node_id}] Removing function {self.assigned_function} from functions to process")

            for entry in reversed(packet["function_timings"]):
                if (
                    entry["function"] == packet["functions_sequence"][0]
                    and entry["end_timestamp"] is None
                ):
                    entry["end_timestamp"] = clock.get_current_time()

            # Eliminar la función cumplida
            if packet["functions_sequence"]:
                packet["functions_sequence"].pop(0)

        # 2. Si se terminaron todas las funciones
        if len(packet["functions_sequence"]) == 0:
            log.debug(
                f"[Node_ID={self.node.node_id}] Function sequence is complete! Initiating full echo callback"
            )
            self.initiate_full_echo_callback(packet)
            return

        # 3. Obtener la siguiente función a buscar
        next_function = packet["functions_sequence"][0]

        # 4. Seleccionar siguiente nodo y obtener estimación
        next_node, estimated_time = self.select_next_node(next_function)

        if next_node is not None:
            send_timestamp = clock.get_current_time()

            # Encolar paso en el callback stack
            callback_chain_step = CallbackChainStep(
                previous_hop_node=packet["from_node_id"],
                current_node=self.node.node_id,
                next_hop_node=next_node,
                estimated_time=estimated_time,
                episode_number=packet["episode_number"],
                function_to_process=next_function.value,
                send_timestamp=send_timestamp,
            )
            global CALLBACK_STACK
            CALLBACK_STACK.append(callback_chain_step)
            log.debug(
                f"\033[92m[CALLBACK_STACK] Encolando {callback_chain_step}, callback_stack: {CALLBACK_STACK}\033[0m"
            )

            # Registrar nuevo intento de alcanzar la función desde este nodo
            packet["function_timings"].append({
                "node_id": self.node.node_id,
                "function": next_function.value,
                "start_timestamp": send_timestamp,
                "end_timestamp": None,
                "estimated_time": estimated_time,
            })

            self.send_packet(next_node, packet)

        else:
            # No hay siguiente nodo válido → intentar de nuevo si quedan hops
            packet["hops"] += 1
            registry.log_lost_packet(
                packet["episode_number"], packet["from_node_id"], None, packet["type"]
            )

            if packet["hops"] >= packet["max_hops"]:
                self.initiate_max_hops_callback(packet)
                return
            else:
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
                continue

            best_timing = None
            best_duration = float("inf")

            for timing in packet.get("function_timings", []):
                if (
                    timing["node_id"] == self.node.node_id
                    and timing["function"] == callback_data.function_to_process
                    and timing["start_timestamp"] is not None
                    and timing["end_timestamp"] is not None
                ):
                    duration = timing["end_timestamp"] - timing["start_timestamp"]
                    if duration < best_duration:
                        best_duration = duration
                        best_timing = timing

            if best_timing is None:
                log.error(f"[Node_ID={self.node.node_id}] No valid timing found for function {callback_data.function_to_process}")
            else:
                # Para Q-Routing clásico, medimos el tiempo de transmisión hasta el siguiente nodo
                transmission_time = clock.get_current_time() - callback_data.send_timestamp
                
                new_q_value = self.update_q_value(
                    next_node=callback_data.next_hop_node,
                    actual_time=transmission_time,  # Tiempo de transmisión
                    function_id=callback_data.function_to_process
                )

            # movement: backward
            self.send_packet(callback_data.previous_hop_node, packet)
            return

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
            "function_timings":packet["function_timings"]
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
