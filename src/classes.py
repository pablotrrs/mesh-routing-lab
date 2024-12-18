# Copyright (c) 2024 Franco Brégoli, Pablo Torres,
# Universidad Nacional de General Sarmiento (UNGS), Buenos Aires, Argentina.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation;
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
# Authors: Franco Brégoli <bregolif.fb@gmail.com>,
#          Pablo Torres <ptorres@campus.ungs.edu.ar>
#
# This project is part of our thesis at Universidad Nacional de General
# Sarmiento (UNGS), and is part of a research initiative to apply reinforcement
# learning for optimized packet routing in ESP-based mesh networks.
#
# The source code for this project is available at:
# https://github.com/pablotrrs/py-q-mesh-routing
#
from enum import Enum
from dataclasses import dataclass
import random
import time
import yaml
import os

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.1

# Ecuación de Bellman:
# ΔQ_x(d, y) = α * (s + t - Q_x(d, y))
# Donde:
# - s: Tiempo de transmisión, calculado como la diferencia entre el timestamp guardado y el tiempo actual.
# - t: Tiempo estimado restante que el nodo calculó cuando eligió el vecino al que enviaría el paquete.
# - Q_x(d, y): Valor Q actual para el nodo actual (x) hacia el vecino (y).
# - α (alpha): Tasa de aprendizaje.
BELLMAN_EQ = lambda s, t, q_current: q_current + ALPHA * (s + t - q_current)

class NodeFunction(Enum):
    A = "A"
    B = "B"
    C = "C"

FUNCTION_SEQ = [NodeFunction.A, NodeFunction.B, NodeFunction.C]

@dataclass
class CallbackChainStep:
    previous_hop_node: int  # Next node in the callback chain
    next_hop_node: int  # Node which we have to calculate the Q-Value for
    send_timestamp: float  # Timestamp del momento en que se realizó el salto to next hop
    estimated_time: float  # Tiempo estimado para completar la secuencia según la Q-table

class Node:
    def __init__(self, node_id, network, epsilon=1.0, epsilon_decay=0.99, min_epsilon=0.1):
        self.node_id = node_id
        self.network = network
        self.q_table = {}
        self.assigned_function = None
        self.callback_stack = []
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def receive_packet(self, packet):
        print(f'[Node_ID={self.node_id}] Received packet {packet}')

        if packet.type == PacketType.PACKET_HOP:
            self.handle_packet_hop(packet)
        elif packet.type == PacketType.CALLBACK:
            self.handle_echo_callback(packet)

    def handle_packet_hop(self, packet):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def handle_echo_callback(self, packet):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_q_value(self, next_node, s, t):
        """
        Actualiza el valor Q para el nodo actual y la acción (saltar al vecino `next_node`) 
        usando la ecuación de Bellman.
        """
        old_q = self.q_table[self.node_id].get(next_node, 0.0)
        new_q = BELLMAN_EQ(s, t, old_q)

        self.q_table[self.node_id][next_node] = new_q

        print(
            f"[Node_ID={self.node_id}] Updated Q-Value for state {self.node_id} -> action {next_node} "
            f"from {old_q:.4f} to {new_q:.4f} (estimated time {t}, actual time {s})"
        )

    def select_next_node(self) -> int:
        """
        Selecciona el siguiente nodo basado en la política ε-greedy con epsilon decay.
        """
        self.initialize_or_update_q_table()
        global EPSILON

        if random.random() < EPSILON:
            # Exploración: elegir un vecino aleatorio
            print(f'[Node_ID={self.node_id}] Performing exploration with epsilon={EPSILON:.4f}')
            next_node = random.choice(self.network.get_neighbors(self.node_id))
        else:
            # Explotación: elegir la mejor acción
            print(f'[Node_ID={self.node_id}] Exploitation with epsilon={EPSILON:.4f}')
            next_node = self.choose_best_action()

        # Update epsilon after each decision
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

        return next_node

    def choose_best_action(self) -> int:
        """
        Encuentra la mejor acción según los valores Q actuales en la Q-table del nodo.
        """
        self.initialize_or_update_q_table()

        neighbors_q_values = self.q_table[self.node_id]
        return min(neighbors_q_values, key=neighbors_q_values.get)

    def initialize_or_update_q_table(self) -> None:
        """
        Asegura que la Q-table del nodo esté inicializada o actualizada como una matriz,
        donde cada vecino tiene un valor Q asociado. Si un vecino no tiene un valor Q,
        se inicializa con un valor por defecto.
        """
        if self.node_id not in self.q_table:
            self.q_table[self.node_id] = {}

        for neighbor in self.network.get_neighbors(self.node_id):
            if neighbor not in self.q_table[self.node_id]:
                self.q_table[self.node_id][neighbor] = 0.0

    def estimate_remaining_time(self, next_node) -> float:
        """
        Estima el tiempo restante a partir del valor Q del nodo siguiente.
        Si no hay valores Q asociados, retorna infinito.
        """
        if self.node_id not in self.q_table or next_node not in self.q_table[self.node_id]:
            return float('inf')

        return self.q_table[self.node_id][next_node]

    def update_q_table_with_incomplete_info(self, next_node, estimated_time_remaining) -> None:
        """Actualiza la Q-table para el estado-acción actual usando información incompleta."""

        self.initialize_or_update_q_table()

        current_q = self.q_table[self.node_id].get(next_node, 0.0)  # Valor Q actual para el nodo
        updated_q = current_q + ALPHA * (estimated_time_remaining - current_q)

        self.q_table[self.node_id][next_node] = updated_q

    def send_packet(self, to_node_id, packet) -> None:
        """
        Envia un paquete al nodo destino utilizando la red.
        """
        packet.hops += 1
        packet.time += 1
        packet.from_node_id = self.node_id

        print(f'[Node_ID={self.node_id}] Sending packet to Node {to_node_id}\n')
        self.network.send(self.node_id, to_node_id, packet)

    def __str__(self) -> str:
        return f"Node(id={self.node_id}, neighbors={self.network.get_neighbors(self.node_id)})"

    def __repr__(self) -> str:
        return self.__str__()

class SenderNode(Node):
    def __init__(self, node_id, neighbors):
        super().__init__(node_id, neighbors)

    def start_episode(self, episode_number) -> None:
        """Initiates an episode by creating a packet and sending it to chosen node."""

        packet = Packet(
            episode_number=episode_number,
            type=PacketType.PACKET_HOP,
            from_node_id=self.node_id
        )

        self.initialize_or_update_q_table()

        next_node = self.select_next_node()

        estimated_time_remaining = self.estimate_remaining_time(next_node)

        callback_chain_step = CallbackChainStep(previous_hop_node=None,
                                                next_hop_node=next_node,
                                                send_timestamp=get_current_time(),
                                                estimated_time=estimated_time_remaining)

        self.callback_stack.append(callback_chain_step)

        self.send_packet(next_node, packet)

    def handle_packet_hop(self, packet) -> None:
        next_node = self.select_next_node()

        estimated_time_remaining = self.estimate_remaining_time(next_node)

        self.update_q_table_with_incomplete_info(
            next_node=next_node,
            estimated_time_remaining=estimated_time_remaining
        )

        callback_chain_step = CallbackChainStep(previous_hop_node=packet.from_node_id,
                                                next_hop_node=next_node,
                                                send_timestamp=get_current_time(),
                                                estimated_time=estimated_time_remaining)

        self.callback_stack.append(callback_chain_step)

        print(f'[Node_ID={self.node_id}] Adding step to callback chain stack: {callback_chain_step}')

        self.send_packet(next_node, packet)

    def handle_echo_callback(self, packet) -> None:
        callback_data = self.callback_stack.pop()

        self.update_q_value(
            next_node=callback_data.next_hop_node,
            s=get_current_time() - callback_data.send_timestamp,
            t=callback_data.estimated_time,
        )

        if self.callback_stack:
            self.send_packet(callback_data.previous_hop_node, packet)
            return

        print(f'\n[Node_ID={self.node_id}] Episode {packet.episode_number} finished. Q-Table: {self.q_table}')

    def __str__(self) -> str:
        return f"SenderNode(id={self.node_id}, neighbors={self.network.get_neighbors(self.node_id)})"

class IntermediateNode(Node):
    def __init__(self, node_id, neighbors):
        super().__init__(node_id, neighbors)

    def handle_packet_hop(self, packet):
        self.initialize_or_update_q_table()

        if self.assigned_function is None:
            self.assign_function(packet)

        if self.assigned_function == packet.next_function():
            print(f'[Node_ID={self.node_id}] Removing function {self.assigned_function} from functions to process')
            packet.remove_next_function()

        if packet.is_sequence_completed():
            print(f'[Node_ID={self.node_id}] Function sequence is complete! Initiating full echo callback')
            self.initiate_full_echo_callback(packet)
            return

        next_node = self.select_next_node()
        print(f'[Node_ID={self.node_id}] Next node is {next_node}')

        if next_node is not None:
            estimated_time_remaining = self.estimate_remaining_time(next_node)

            self.update_q_table_with_incomplete_info(
                next_node=next_node,
                estimated_time_remaining=estimated_time_remaining
            )

            callback_chain_step = CallbackChainStep(previous_hop_node=packet.from_node_id,
                                                    next_hop_node=next_node,
                                                    send_timestamp=get_current_time(),
                                                    estimated_time=estimated_time_remaining)

            self.callback_stack.append(callback_chain_step)

            print(f'[Node_ID={self.node_id}] Adding step to callback chain stack: {callback_chain_step}')

            self.send_packet(next_node, packet)

    def handle_echo_callback(self, packet):
        """Maneja el callback cuando regresa el paquete."""
        callback_data = self.callback_stack.pop()

        self.update_q_value(
            next_node=callback_data.next_hop_node,
            s=get_current_time() - callback_data.send_timestamp,
            t=callback_data.estimated_time,
        )

        self.send_packet(callback_data.previous_hop_node, packet)

    def initiate_full_echo_callback(self, packet):
        """Inicia el proceso de full echo callback hacia el nodo anterior."""

        callback_packet = Packet(
            episode_number=packet.episode_number,
            from_node_id=self.node_id,
            type=PacketType.CALLBACK
        )

        self.send_packet(packet.from_node_id, callback_packet)

    def assign_function(self, packet):
        """Asigna la función menos utilizada basada en los contadores del paquete."""
        min_assignments = min(packet.function_counters.values())

        least_assigned_functions = [
            func for func, count in packet.function_counters.items() if count == min_assignments
        ]

        if len(least_assigned_functions) == 1:
            function_to_assign = least_assigned_functions[0]
        else:
            function_to_assign = random.choice(least_assigned_functions)

        print(f'[Node_ID={self.node_id}] Node has no function, assigning function {function_to_assign}')
        self.assigned_function = function_to_assign
        packet.increment_function_counter(function_to_assign)

    def __str__(self):
        return f"IntermediateNode(id={self.node_id}, neighbors={self.network.get_neighbors(self.node_id)})"

class PacketType(Enum):
    PACKET_HOP = "PACKET_HOP"
    CALLBACK = "CALLBACK"

class Packet:
    def __init__(self, episode_number, from_node_id, type):
        self.type = type
        self.episode_number = episode_number  # Número de episodio al que pertenece este paquete
        self.from_node_id = from_node_id  # Nodo anterior por el que pasó el paquete
        self.functions_sequence = FUNCTION_SEQ.copy()  # Secuencia de funciones a procesar
        self.function_counters = {func: 0 for func in FUNCTION_SEQ}  # Contadores de funciones asignadas
        self.hops = 0  # Contador de saltos
        self.time = 0  # Tiempo total acumulado del paquete

    def increment_function_counter(self, function):
        """
        Incrementa el contador de asignaciones para una función específica.
        """
        if function not in self.function_counters:
            raise ValueError(f"La función {function} no es válida.")
        self.function_counters[function] += 1

    def is_sequence_completed(self):
        """Revisa si la secuencia de funciones ha sido completamente procesada."""
        return len(self.functions_sequence) == 0

    def next_function(self):
        """Obtiene la próxima función en la secuencia, si existe."""
        return self.functions_sequence[0] if self.functions_sequence else None

    def remove_next_function(self):
        """Elimina la función actual de la secuencia, marcándola como procesada."""
        if self.functions_sequence:
            self.functions_sequence.pop(0)

    def __str__(self):
        functions_sequence_str = [func.value for func in self.functions_sequence]
        function_counters_str = {func.value: count for func, count in self.function_counters.items()}
        return (
            f"Packet("
            f"type={self.type.value}, "
            f"episode_number={self.episode_number}, "
            f"from_node_id={self.from_node_id}, "
            f"functions_sequence={functions_sequence_str}, "
            f"function_counters={function_counters_str}, "
            f"hops={self.hops}, "
            f"time={self.time})"
        )

class Network:
    def __init__(self):
        self.nodes = {}  # {node_id: Node}
        self.connections = {}  # {node_id: [neighbors]}
        self.active_nodes = set()

    def add_node(self, node):
        self.nodes[node.node_id] = node
        self.connections[node.node_id] = []
        self.active_nodes.add(node.node_id)

    def connect_nodes(self, node_id1, node_id2):
        if node_id1 in self.nodes and node_id2 in self.nodes:
            self.connections[node_id1].append(node_id2)
            self.connections[node_id2].append(node_id1)
        else:
            raise ValueError("One or both nodes don't exist in the network.")

    def get_neighbors(self, node_id):
        """Returns a node's neighbors, excluding itself."""
        neighbors = self.connections.get(node_id, [])
        return list(set(neighbor for neighbor in neighbors if neighbor != node_id))

    def send(self, from_node_id, to_node_id, packet):
        """
        Envía un paquete entre dos nodos dentro de la red.
        """
        if to_node_id in self.connections.get(from_node_id, []) and to_node_id in self.active_nodes:
            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id}")
            self.nodes[to_node_id].receive_packet(packet)
        else:
            print(f"[Network] Failed to send packet: Node {to_node_id} is not reachable from Node {from_node_id}")

    @classmethod
    def from_yaml(cls, file_path):
        """
        Crea una instancia de Network a partir de un archivo YAML y devuelve la red y el nodo emisor.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        from classes import SenderNode, IntermediateNode

        network = cls()
        sender_node = None

        for node_id, node_info in data['nodes'].items():
            node_id = int(node_id)

            if 'type' in node_info and node_info['type'] == 'sender':
                node = SenderNode(node_id, network)
                sender_node = node
            else:
                node = IntermediateNode(node_id, network)

            network.add_node(node)

        if sender_node is None:
            raise ValueError("No se encontró un nodo de tipo 'sender' en el archivo YAML.")

        for node_id, node_info in data['nodes'].items():
            node_id = int(node_id)
            neighbors = node_info['neighbors']
            for neighbor_id in neighbors:
                network.connect_nodes(node_id, neighbor_id)

        return network, sender_node

    def __str__(self) -> str:
        result = ["\nNetwork Topology:"]
        for node_id, neighbors in self.connections.items():
            result.append(f"Node {node_id} -> Neighbors: {neighbors}")
        return "\n".join(result)

# TODO: en esta clase es que hay que manejar el tema de que los
# nodos se desconecten y se reconecten
class Simulation:
    def __init__(self, network, sender_node):
        self.network = network
        self.sender_node = sender_node

    def run_episode(self, episode_number):
        # TODO: dynamic_network_change() que cambie los nodos de network
        print(f'\n\n=== Starting episode #{episode_number} ===\n\n')
        self.sender_node.start_episode(episode_number)

def get_current_time():
    return time.time()

def main():
    topology_file_path = os.path.join(os.path.dirname(__file__), "../resources/dummy_topology.yaml")
    network, sender_node = Network.from_yaml(topology_file_path)

    print(network)

    simulation = Simulation(network, sender_node)

    # TODO:
    #  1. hacer que se conecten y desconecten los nodos con distribución exponencial
    #  2. hacer que se de por perdido el paquete después de 100 hops
    #  3. relevar resultados (integrar con lo que había antes para visualizar, y exportar a un csv como en modelado, 
    #     para comparar y hacer gráficos de cómo cambian los parámetros Latencia Promedio, Consistencia en la Latencia,
    #     Tasa de Éxito, Balanceo de Carga, Overhead de Comunicación, Tiempo de Cómputo, Adaptabilidad a Cambios en la Red
    #     con respecto a los pasos tiempo)
    #  4. la q-table es local para cada nodo (solo tiene los state-values correspondientes a los vecinos de ese nodo). 
    #     tendríamos que hacer algo para tener una q-table global (parecido al punto 3.)
    #  4. hacer que se pueda elegir cambiar por los métodos de ruteo dijkstra y bellman ford

    # for episode in range(1, 10):
        # simulation.run_episode(episode)
    simulation.run_episode(1)


if __name__ == "__main__":
    main()
