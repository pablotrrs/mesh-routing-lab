from collections import deque
from enum import Enum
from dataclasses import dataclass
import random
import time
from classes.base import Application, EpisodeEnded
from visualization import print_q_table

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1

PENALTY = 0.0

MAX_HOPS = None

CURRENT_HOP_COUNT = 0

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

class PacketType(Enum):
    PACKET_HOP = "PACKET_HOP"
    CALLBACK = "CALLBACK"
    MAX_HOPS_REACHED = "MAX_HOPS_REACHED"

FUNCTION_SEQ = None

@dataclass
class CallbackChainStep:
    previous_hop_node: int  # Next node in the callback chain
    current_node: int  # Current node in the callback chain
    next_hop_node: int  # Node which we have to calculate the Q-Value for
    send_timestamp: float  # Timestamp del momento en que se realizó el salto to next hop
    estimated_time: float  # Tiempo estimado para completar la secuencia según la Q-table
    episode_number: int  # Número de episodio asociadoMax hops reached. Initiating full echo callback

class QRoutingApplication(Application):
    def __init__(self, node):
        self.node = node
        self.q_table = {}
        self.assigned_function = None
        self.callback_stack = deque()
        self.max_hops = None
        self.functions_sequence = None

    def receive_packet(self, packet):
        print(f'[Node_ID={self.node.node_id}] Received packet {packet}')

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

    def update_q_value(self, next_node, s, t):
        """
        Actualiza el valor Q para el nodo actual y la acción (saltar al vecino `next_node`) 
        usando la ecuación de Bellman.
        """
        old_q = self.q_table[self.node.node_id].get(next_node, 0.0)
        new_q = BELLMAN_EQ(s, t, old_q)

        self.q_table[self.node.node_id][next_node] = new_q

        print(
            f"[Node_ID={self.node.node_id}] Updated Q-Value for state {self.node.node_id} -> action {next_node} "
            f"from {old_q:.4f} to {new_q:.4f} (estimated time {t}, actual time {s})"
        )
        return

    def select_next_node(self) -> int:
        """
        Selecciona el siguiente nodo basado en la política ε-greedy con epsilon decay.
        """
        self.initialize_or_update_q_table()
        global EPSILON

        next_node = None
        current_node_id = self.node.node_id  # ID del nodo actual

        if random.random() < EPSILON:
            # Exploración: elegir un vecino aleatorio
            print(f'[Node_ID={current_node_id}] Performing exploration with epsilon={EPSILON:.4f}')
            valid_neighbors = [
                neighbor for neighbor in self.node.network.get_neighbors(current_node_id)
                if self.node.network.nodes[neighbor].status and neighbor != current_node_id 
                and self.node.network.validate_connection(current_node_id, neighbor)  # Validar conexión
            ]
            if valid_neighbors:
                next_node = random.choice(valid_neighbors)
        else:
            # Explotación: elegir la mejor acción
            print(f'[Node_ID={current_node_id}] Exploitation with epsilon={EPSILON:.4f}')
            next_node = self.choose_best_action()

            # Validar si la acción elegida es alcanzable
            if next_node == current_node_id or not self.node.network.validate_connection(current_node_id, next_node):
                print(f'[Node_ID={current_node_id}] Exploitation selected an unreachable or self node. Falling back to exploration.')
                next_node = None  # Forzar exploración

        # Si la explotación falló o se seleccionó a sí mismo, usar exploración como fallback
        if next_node is None:
            print(f'[Node_ID={current_node_id}] Exploitation failed, falling back to exploration')

            valid_neighbors = [
                neighbor for neighbor in self.node.network.get_neighbors(current_node_id)
                if self.node.network.nodes[neighbor].status and neighbor != current_node_id
                and self.node.network.validate_connection(current_node_id, neighbor)  # Validar conexión
            ]

            if valid_neighbors:
                next_node = random.choice(valid_neighbors)
                print(f'[Node_ID={current_node_id}] Exploration selected Node {next_node}')
            else:
                # Si no hay vecinos conectados, no se puede enviar el paquete
                print(f'[Node_ID={current_node_id}] No available neighbors to send the packet. Dropping packet.')
                return None

        # Update epsilon after each decision
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        return next_node

    def choose_best_action(self) -> int:
        """
        Encuentra la mejor acción según los valores Q actuales en la Q-table del nodo.
        """
        self.initialize_or_update_q_table()

        # Filter neighbors based on their status
        neighbors_q_values = {
            neighbor: q_value for neighbor, q_value in self.q_table[self.node.node_id].items()
            if self.node.network.nodes[neighbor].status
        }

        if not neighbors_q_values:
            print(f"[Node_ID={self.node.node_id}] No available neighbors with status True")

            return None

        return min(neighbors_q_values, key=neighbors_q_values.get)

    def initialize_or_update_q_table(self) -> None:
        """
        Asegura que la Q-table del nodo esté inicializada o actualizada como una matriz,
        donde cada vecino tiene un valor Q asociado. Si un vecino no tiene un valor Q,
        se inicializa con un valor por defecto.
        """
        if self.node.node_id not in self.q_table:
            self.q_table[self.node.node_id] = {}

        for neighbor in self.node.network.get_neighbors(self.node.node_id):
            if neighbor not in self.q_table[self.node.node_id]:
                self.q_table[self.node.node_id][neighbor] = 0.0
        return

    def estimate_remaining_time(self, next_node) -> float:
        """
        Estima el tiempo restante a partir del valor Q del nodo siguiente.
        Si no hay valores Q asociados, retorna infinito.
        """
        if self.node.node_id not in self.q_table or next_node not in self.q_table[self.node.node_id]:
            return float('inf')

        return self.q_table[self.node.node_id][next_node]

    def update_q_table_with_incomplete_info(self, next_node, estimated_time_remaining) -> None:
        """Actualiza la Q-table para el estado-acción actual usando información incompleta."""

        self.initialize_or_update_q_table()

        current_q = self.q_table[self.node.node_id].get(next_node, 0.0)  # Valor Q actual para el nodo
        updated_q = current_q + ALPHA * (estimated_time_remaining - current_q)

        self.q_table[self.node.node_id][next_node] = updated_q
        return

    def send_packet(self, to_node_id, packet, lost_packet=False) -> None:
        if packet.get("hops") is not None:
            packet["hops"] += 1

        if lost_packet:
            self.node.network.send(self.node.node_id, None, packet, lost_packet=True)
        else:
            packet["from_node_id"] = self.node.node_id

            print(f'[Node_ID={self.node.node_id}] Sending packet to Node {to_node_id}\n')
            self.node.network.send(self.node.node_id, to_node_id, packet)

        if packet["hops"] > MAX_HOPS:
            print(f'[Node_ID={self.node.node_id}] Max hops reached. Initiating full echo callback')
            return False
        else:
            return True

    def initiate_max_hops_callback(self, packet):
        global CALLBACK_STACK

        # si no tiene callback stack porque puede pasar
        # porque nunca salió del 0 -> n hop

        episode_data = self.node.network.packet_log.get(packet["episode_number"], {})
        route = episode_data.get("route", [])

        if not CALLBACK_STACK:
            if self.didnt_make_it_further_than_first_hop(route):
                self.send_packet(packet["from_node_id"], callback_packet)

        callback_packet = {
            "type": PacketType.MAX_HOPS_REACHED,
            "episode_number": packet["episode_number"],
            "from_node_id": self.node.node_id,
            "max_hops": self.max_hops
        }

        callback_data = CALLBACK_STACK.pop()
        print(f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m")

        self.penalize_q_value(
            next_node=callback_data.next_hop_node
        )

        # movement: backward
        self.send_packet(callback_data.previous_hop_node, callback_packet)
        return

    def didnt_make_it_further_than_first_hop(self, route):
        # Extraer todos los valores de "from"
        from_values = [step["from"] for step in route]

        # Contar la frecuencia de cada valor
        from_counts = {}
        for value in from_values:
            from_counts[value] = from_counts.get(value, 0) + 1

        # Verificar las condiciones
        if len(from_counts) == 2:  # Solo debe haber dos valores distintos
            if 0 in from_counts and from_counts[0] == 1:  # El 0 debe aparecer exactamente una vez
                other_value = [value for value in from_counts if value != 0][0]  # El otro valor
                if from_counts[other_value] == len(route) - 1:  # El otro valor debe aparecer en todos los demás pasos
                    return True
        return False

    def penalize_q_value(self, next_node):
        """
        Penaliza el valor Q de la acción (ir al vecino `next_node`) con una reducción fuerte.
        """
        global PENALTY
        old_q = self.q_table[self.node.node_id].get(next_node, 0.0)
        new_q = old_q - PENALTY

        self.q_table[self.node.node_id][next_node] = max(new_q, 0)  # Evita valores negativos extremos

        print(
            f"[Node_ID={self.node.node_id}] Penalized by {PENALTY} Q-Value for state {self.node.node_id} -> action {next_node} "
            f"from {old_q:.4f} to {new_q:.4f} (hard penalty applied)"
        )
        return

    def get_assigned_function(self):
        """Returns the function assigned to this node."""
        return self.assigned_function

    def __str__(self) -> str:
        return f"Node(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"

    def __repr__(self) -> str:
        return self.__str__()

class SenderQRoutingApplication(QRoutingApplication):
    def __init__(self, node):
        super().__init__(node)

    def set_penalty(self, penalty):
        print(f'setting penalty to {penalty}')
        global PENALTY
        PENALTY = penalty
        print(f'penalty setted to {PENALTY}')

    def start_episode(self, episode_number, max_hops=None, functions_sequence=None, penalty=0.0, current_hop_count=0) -> None:
        """Initiates an episode by creating a packet and sending it to chosen node."""
        self.max_hops=max_hops

        global EPISODE_COMPLETED
        EPISODE_COMPLETED = False

        print(f"\n\033[93mClearing callback stacks for Episode {episode_number}\033[0m")
        global CALLBACK_STACK
        CALLBACK_STACK.clear()

        global MAX_HOPS
        MAX_HOPS=max_hops

        global PENALTY
        PENALTY = penalty

        global FUNCTION_SEQ
        FUNCTION_SEQ=functions_sequence

        packet = {
            "type": PacketType.PACKET_HOP,
            "episode_number": episode_number,
            "from_node_id": self.node.node_id,
            "functions_sequence": FUNCTION_SEQ.copy(),
            "function_counters": {func: 0 for func in FUNCTION_SEQ},
            "hops": current_hop_count,
            "time": 0,
            "max_hops": max_hops,
            "is_delivered": False
        }

        self.initialize_or_update_q_table()

        next_node = self.select_next_node()

        if next_node is None:
            print(f'[Node_ID={self.node.node_id}] No valid next node found. Can\'t initiate episode!.')
            # movement: none
            self.send_packet(None, packet, True)
            packet["hops"] += 1
            print(f'[Node_ID={self.node.node_id}] Packet hop count {packet["hops"]}')

            if packet["hops"] > MAX_HOPS:
                self.mark_episode_result(packet, success=False)

            # si no se puede empezar el episodio, se sigue intentando hasta que se pueda
            self.start_episode(episode_number, max_hops, functions_sequence, penalty, packet["hops"])
            return
        else:
            estimated_time_remaining = self.estimate_remaining_time(next_node)

            self.update_q_table_with_incomplete_info(
                next_node=next_node,
                estimated_time_remaining=estimated_time_remaining
            )

            # movement: forward
            self.send_packet(next_node, packet)
            return

    def handle_packet_hop(self, packet) -> None:
        next_node = self.select_next_node()

        # Base case to stop recursion if no valid next node is found
        if next_node is None:
            print(f'[Node_ID={self.node.node_id}] No valid next node found. Stopping packet hop.')
            # movement: none
            if not self.send_packet(None, packet, True):
                # max hops reached
                self.mark_episode_result(packet, success=False)
            else:
                # retry until there is a valid next node
                self.handle_packet_hop(packet)
                return

        estimated_time_remaining = self.estimate_remaining_time(next_node)

        self.update_q_table_with_incomplete_info(
            next_node=next_node,
            estimated_time_remaining=estimated_time_remaining
        )

        callback_chain_step = CallbackChainStep(previous_hop_node=packet["from_node_id"],
                                                current_node=self.node.node_id,
                                                next_hop_node=next_node,
                                                send_timestamp=get_current_time(),
                                                estimated_time=estimated_time_remaining,
                                                episode_number=packet["episode_number"])


        global CALLBACK_STACK
        CALLBACK_STACK.append(callback_chain_step)
        print(f"\033[92m[CALLBACK_STACK] Encolando {callback_chain_step}, callback_stack: {CALLBACK_STACK}\033[0m")

        print(f'[Node_ID={self.node.node_id}] Adding step to callback chain stack: {callback_chain_step}')

        # movement: forward
        if not self.send_packet(next_node, packet):
            # max hops reached
            self.initiate_max_hops_callback(packet)
            return
        return

    def handle_echo_callback(self, packet) -> None:
        global CALLBACK_STACK
        if len(CALLBACK_STACK) > 0:
            callback_data = CALLBACK_STACK.pop()
            print(f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m")

            self.update_q_value(
                next_node=callback_data.next_hop_node,
                s=get_current_time() - callback_data.send_timestamp,
                t=callback_data.estimated_time,
            )

            # movement: backward
            self.send_packet(callback_data.previous_hop_node, packet)
            return
        else:
            print_q_table(self)
            print(f'\n[Node_ID={self.node.node_id}] Episode {packet["episode_number"]} finished.')

            self.mark_episode_result(packet, success=True)
            return

    def handle_lost_packet(self, packet) -> None:
        episode_number = packet["episode_number"]
        print(f"\n[Node_ID={self.node.node_id}] Episode {episode_number} failed.")

        self.mark_episode_result(packet, success=False)

    def mark_episode_result(self, packet, success=True):
        """
        Marca un episodio como exitoso o fallido y lo notifica a la red.

        Args:
            packet (Packet): El paquete asociado al episodio.
            success (bool): `True` si el episodio fue exitoso, `False` si falló.
        """

        global EPISODE_COMPLETED
        EPISODE_COMPLETED = True

        print(f'marking episode as mf completed {EPISODE_COMPLETED}')

        status_text = "SUCCESS" if success else "FAILURE"
        episode_number = packet["episode_number"]
        print(f"\n[Node_ID={self.node.node_id}] Marking Episode {episode_number} as {status_text}.")

        # Llamar a la red para registrar el estado del episodio
        self.node.network.send(
            from_node_id=self.node.node_id, 
            to_node_id=None,  # No es necesario enviar el paquete a otro nodo en este caso
            packet=packet, 
            episode_success=success
        )

        global CURRENT_HOP_COUNT
        CURRENT_HOP_COUNT = 0
        raise EpisodeEnded()

    def __str__(self) -> str:
        return f"SenderNode(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"

class IntermediateQRoutingApplication(QRoutingApplication):
    def __init__(self, node):
        super().__init__(node)

    def start_episode(self, episode_number):
        raise NotImplementedError("Intermediate node is not supposed to start an episode")

    def handle_packet_hop(self, packet):

        self.initialize_or_update_q_table()

        if packet["hops"] > packet["max_hops"]:
            global EPISODE_COMPLETED

            print(f'episode completion {EPISODE_COMPLETED}')
            if EPISODE_COMPLETED:
                episode_number = packet["episode_number"]
                print(f"\033[91m[Node_ID={self.node.node_id}] Episode {episode_number} already ended or not found. Ignoring packet.\033[0m")
                return
            print(f'[Node_ID={self.node.node_id}] Max hops reached. Initiating full echo callback')
            self.initiate_max_hops_callback(packet)
            return

        if self.assigned_function is None:
            self.assign_function(packet)

        # if function to process is function assigned to this node
        if self.assigned_function == packet["functions_sequence"][0] if packet["functions_sequence"] else None:
            print(f'[Node_ID={self.node.node_id}] Removing function {self.assigned_function} from functions to process')
            if packet["functions_sequence"]:
                packet["functions_sequence"].pop(0)

        # if all functions have been processed
        if len(packet["functions_sequence"]) == 0:
            print(f'[Node_ID={self.node.node_id}] Function sequence is complete! Initiating full echo callback')
            self.initiate_full_echo_callback(packet)
            return

        next_node = self.select_next_node()

        print(f'[Node_ID={self.node.node_id}] Next node is {next_node}')
        if next_node is not None:
            estimated_time_remaining = self.estimate_remaining_time(next_node)

            callback_chain_step = CallbackChainStep(previous_hop_node=packet["from_node_id"],
                                                    current_node=self.node.node_id,
                                                    next_hop_node=next_node,
                                                    send_timestamp=get_current_time(),
                                                    estimated_time=estimated_time_remaining,
                                                    episode_number=packet["episode_number"])
            global CALLBACK_STACK
            CALLBACK_STACK.append(callback_chain_step)
            print(f"\033[92m[CALLBACK_STACK] Encolando {callback_chain_step}, callback_stack: {CALLBACK_STACK}\033[0m")

            print(f'[Node_ID={self.node.node_id}] Adding step to callback chain stack: {callback_chain_step}')

            self.update_q_table_with_incomplete_info(
                next_node=next_node,
                estimated_time_remaining=estimated_time_remaining
            )

            # movement: forward
            self.send_packet(next_node, packet)
            return
        else:
            # movement: none
            still_hops_remaining = self.send_packet(None, packet, lost_packet=True)
            print(f'mf still_hops_remaining {still_hops_remaining}')

            if not still_hops_remaining:
                # max hops reached
                print('mf initiate_max_hops_callback')
                # initiate_max_hops_callback
                self.initiate_max_hops_callback(packet)
                return
            else:
                # retry until there is a valid next node
                self.handle_packet_hop(packet)
                return

    def handle_echo_callback(self, packet):
        """Maneja el callback cuando regresa el paquete."""
        global CALLBACK_STACK
        callback_data = CALLBACK_STACK.pop()
        print(f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m")

        self.update_q_value(
            next_node=callback_data.next_hop_node,
            s=get_current_time() - callback_data.send_timestamp,
            t=callback_data.estimated_time,
        )

        # movement: backward
        self.send_packet(callback_data.previous_hop_node, packet)
        return

    def handle_lost_packet(self, packet) -> None:
        global CALLBACK_STACK
        callback_data = CALLBACK_STACK.pop()
        print(f"\033[91m[CALLBACK_STACK] Desencolando {callback_data}, callback_stack: {CALLBACK_STACK}\033[0m")

        self.penalize_q_value(
            next_node=callback_data.next_hop_node
        )

        # movement: backward
        self.send_packet(callback_data.previous_hop_node, packet)
        return

    def initiate_full_echo_callback(self, packet):
        """Inicia el proceso de full echo callback hacia el nodo anterior."""

        callback_packet = {
            "type": PacketType.CALLBACK,
            "episode_number": packet["episode_number"],
            "from_node_id": self.node.node_id,
            "max_hops": self.max_hops
        }

        # movement: backward
        self.send_packet(packet["from_node_id"], callback_packet)
        return

    def assign_function(self, packet):
        """Asigna la función menos utilizada basada en los contadores del paquete."""
        min_assignments = min(packet["function_counters"].values())

        least_assigned_functions = [
            func for func, count in packet["function_counters"].items() if count == min_assignments
        ]

        if len(least_assigned_functions) == 1:
            function_to_assign = least_assigned_functions[0]
        else:
            function_to_assign = random.choice(least_assigned_functions)

        print(f'[Node_ID={self.node.node_id}] Node has no function, assigning function {function_to_assign}')
        self.assigned_function = function_to_assign
        packet["function_counters"][function_to_assign] += 1
        return

    def __str__(self):
        return f"IntermediateNode(id={self.node.node_id}, neighbors={self.node.network.get_neighbors(self.node.node_id)})"

def get_current_time():
    return time.time()
