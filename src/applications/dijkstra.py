from enum import Enum
from classes import Application
from queue import PriorityQueue
from tabulate import tabulate
import time
from dataclasses import dataclass, field
from typing import Set
import random

class NodeFunction(Enum):
    A = "A"
    B = "B"
    C = "C"

FUNCTION_SEQ = [NodeFunction.A, NodeFunction.B, NodeFunction.C]

class BroadcastState:
    def __init__(self):
        self.message_id = None            # Identificador único del mensaje de broadcast
        self.expected_acks = 0                  # Número de ACKs esperados
        self.acks_received = 0                  # Contador de ACKs recibidos
        self.parent_node = None                 # Nodo del que se recibió el BROADCAST originalmente
        self.received_messages = set()          # Conjunto de mensajes ya recibidos
        self.completed = False                  # Indica si el broadcast se completó
        self.received_acks = set()              # Conjunto de nodos desde los que se recibieron ACKs
        self.node_function_map = {}

    def set_parent_node(self, parent_node: int):
        """Establece el nodo del que se recibió el BROADCAST."""
        self.parent_node = parent_node

    def increment_acks_received(self, from_node_id: int):
        """Incrementa el contador de ACKs recibidos y registra el nodo que envió el ACK."""
        self.acks_received += 1
        self.received_acks.add(from_node_id)

    def mark_completed(self):
        """Marca el broadcast como completado."""
        self.completed = True

    def add_received_message(self, message_id: str):
        """Agrega un mensaje al conjunto de mensajes ya recibidos."""
        self.received_messages.add(message_id)

    def has_received_message(self, message_id: str) -> bool:
        """Verifica si un mensaje ya fue recibido."""
        return message_id in self.received_messages

class PacketType(Enum):
    BROADCAST = "BROADCAST"
    ACK = "ACK"
    PACKET_HOP = "PACKET_HOP"
    BROKEN_PATH = "BROKEN_PATH"
    SUCCESS = "SUCCESS"
    PROBE = "PROBE"

class Packet:
    def __init__(self, episode_number, from_node_id, type):
        self.type = type
        self.episode_number = episode_number  # Número de episodio al que pertenece este paquete
        self.from_node_id = from_node_id  # Nodo anterior por el que pasó el paquete
        self.functions_sequence = FUNCTION_SEQ.copy()  # Secuencia de funciones a procesar
        self.function_counters = {func: 0 for func in FUNCTION_SEQ}  # Contadores de funciones asignadas
        self.processed_functions = []  # Funciones ya procesadas
        self.hops = 0  # Contador de saltos
        self.time = 0  # Tiempo total acumulado del paquete
        self.max_hops = 250  # Número máximo de saltos permitidos

    def __init__(self, episode_number, from_node_id, type, message_id):
        self.type = type
        self.episode_number = episode_number  # Número de episodio al que pertenece este paquete
        self.from_node_id = from_node_id  # Nodo anterior por el que pasó el paquete
        self.functions_sequence = FUNCTION_SEQ.copy()  # Secuencia de funciones a procesar
        self.function_counters = {func: 0 for func in FUNCTION_SEQ}  # Contadores de funciones asignadas
        self.processed_functions = []  # Funciones ya procesadas
        self.hops = 0  # Contador de saltos
        self.time = 0  # Tiempo total acumulado del paquete
        self.max_hops = 250  # Número máximo de saltos permitidos
        self.message_id = message_id

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

class DijkstraApplication(Application):
    def __init__(self, node):
        super().__init__(node)
        self.routes = {}  # Almacena las rutas más cortas calculadas

    def select_next_function_node(self, packet):
        """
        Selecciona el siguiente nodo que debe procesar la próxima función faltante.
        Si todos los nodos vecinos no tienen función asignada, elige el nodo con menor peso en la arista.
        """
        next_function = packet["functions_sequence"][0]  # Obtener la próxima función de la secuencia
        neighbors = self.node.network.get_neighbors(self.node.node_id)

        # Filtrar nodos que puedan procesar la próxima función
        valid_neighbors = [neighbor for neighbor in neighbors if self.node.network.nodes[neighbor].get_assigned_function() == next_function]

        # Si hay nodos que pueden procesar la próxima función, elegir el más cercano
        if valid_neighbors:
            return min(valid_neighbors, key=lambda n: self.node.network.get_latency(self.node.node_id, n))

        # Si no hay nodos que puedan procesar la función, elegir el más cercano sin función asignada
        nodes_without_function = [neighbor for neighbor in neighbors if self.node.network.nodes[neighbor].get_assigned_function() is None]
        if nodes_without_function:
            return min(nodes_without_function, key=lambda n: self.node.network.get_latency(self.node.node_id, n))

        # Si todos los nodos tienen función asignada pero ninguna coincide, elegir el más cercano
        return min(neighbors, key=lambda n: self.node.network.get_latency(self.node.node_id, n))

    def send_packet(self, to_node_id, packet):

        if "hops" in packet:
            packet["hops"] += 1
        else:
            packet["hops"] = 0

        if "max_hops" not in packet:
            packet["max_hops"] = 500 # TODO: es una cagada hacer esto así,
                                     # hacer que el método send tenga varias implementaciones,
                                     # tomando un packet que sea un dict y el resto parámetros (max hops etc)

        if "time" in packet:
            packet["time"] += 1

        if "from_node_id" in packet:
            packet["from_node_id"] = self.node.node_id

        print(f'\n[Node_ID={self.node.node_id}] Sending packet to Node {to_node_id}\n')
        self.node.network.send_dict(self.node.node_id, to_node_id, packet)

class SenderDijkstraApplication(DijkstraApplication):
    def __init__(self, node):
        super().__init__(node)
        self.routes = {}  # Almacena las rutas más cortas calculadas
        self.previous_node_id = None

    def start_episode(self, episode_number) -> None:
        """
        Inicia un episodio calculando las rutas más cortas y enviando el paquete.
        """
        print(f"[Node_ID={self.node.node_id}] Starting broadcast for episode {episode_number}")

        # Iniciar el broadcast para recopilar latencias y asignar funciones
        message_id = f"broadcast_{self.node.node_id}_{episode_number}"

        self.broadcast_state = BroadcastState()
        self.start_broadcast(message_id)

        # Esperar hasta que se complete el broadcast (ACKs recibidos)
        while self.broadcast_state.acks_received < self.broadcast_state.expected_acks:
            pass  # Espera activa; se puede mejorar con eventos o temporizadores

        print(f"[Node_ID={self.node.node_id}] Broadcast completed. Computing shortest paths...")
        self.compute_shortest_paths()

        # Esperar hasta que la bandera paths_computed sea True
        while not self.paths_computed:
            pass

        print(f"[Node_ID={self.node.node_id}] Starting episode {episode_number}")
        packet = {
            "type": PacketType.PACKET_HOP,
            "episode_number": episode_number,
            "from_node_id": self.node.node_id,
            "functions_sequence": FUNCTION_SEQ.copy(),  # Copia de la secuencia de funciones a procesar
            "function_counters": {func: 0 for func in FUNCTION_SEQ},  # Contadores de funciones asignadas
            "hops": 0,  # Contador de saltos
            "time": 0,  # Tiempo total acumulado del paquete
            "max_hops": 250  # Número máximo de saltos permitidos
        }
        next_node = self.select_next_function_node(packet)

        if next_node is None:
            print("No suitable next node found.")
            return

        self.send_packet(next_node, packet)

    def start_broadcast(self, message_id):
        """
        Inicia el proceso de broadcast desde el nodo sender.
        """
        neighbors = self.node.network.get_neighbors(self.node.node_id)
        broadcast_packet = {
            "type": PacketType.BROADCAST,
            "message_id": message_id,
            "from_node_id": self.node.node_id,
            "episode_number": 0,
            "visited_nodes": {self.node.node_id},
            "functions_sequence": FUNCTION_SEQ.copy(),
            "function_counters": {func: 0 for func in FUNCTION_SEQ},
            "node_function_map": {}
        }

        # Inicializar el estado de broadcast
        self.broadcast_state = BroadcastState()
        self.broadcast_state.expected_acks = len(neighbors)
        print(f"[Node_ID={self.node.node_id}] Expected ACKs: {self.broadcast_state.expected_acks}")

        for neighbor in neighbors:
            self.send_packet(neighbor, broadcast_packet)

    def compute_shortest_paths(self):
        """
        Calcula las rutas más cortas desde el nodo de origen a todos los demás nodos,
        utilizando las latencias definidas en la clase Network.
        """
        self.paths_computed = False  # Inicializar la bandera en False

        # Inicializar distancias y nodos previos
        distances = {node_id: float('inf') for node_id in self.node.network.get_nodes()}
        distances[self.node.node_id] = 0
        previous_nodes = {node_id: None for node_id in self.node.network.get_nodes()}

        priority_queue = PriorityQueue()
        priority_queue.put((0, self.node.node_id))

        visited = set()  # Conjunto de nodos visitados

        while not priority_queue.empty():
            current_distance, current_node = priority_queue.get()

            if current_node in visited:  # Si el nodo ya fue visitado, saltarlo
                continue

            visited.add(current_node)  # Marcar el nodo como visitado

            # Iterar sobre los vecinos del nodo actual
            for neighbor in self.node.network.get_neighbors(current_node):
                distance = current_distance + self.node.network.get_latency(current_node, neighbor)
                if distance < distances[neighbor]:  # Si se encuentra una ruta más corta
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    priority_queue.put((distance, neighbor))

        # Reconstruir rutas usando los nodos previos
        self.routes = self._reconstruct_paths(self.node.node_id, previous_nodes)
        self._log_routes()

        self.paths_computed = True  # Marcar la bandera como True al finalizar

    def _reconstruct_paths(self, sender_node_id, previous_nodes):
        """
        Reconstruye las rutas más cortas desde el diccionario de nodos previos,
        incluyendo las funciones que se procesan en cada nodo.
        """
        routes = {}
        for node_id in self.node.network.get_nodes():
            path = []
            functions = []  # Lista para almacenar las funciones procesadas en el camino
            current = node_id
            while current is not None:
                path.insert(0, current)
                assigned_function = self.node.network.nodes[current].get_assigned_function()
                functions.insert(0, assigned_function if assigned_function else None)
                current = previous_nodes[current]
            if path[0] == sender_node_id:  # Solo guarda rutas alcanzables
                routes[node_id] = {"path": path, "functions": functions}
        return routes

    def get_route_to(self, destination_node_id):
        """
        Devuelve la ruta más corta hacia un nodo destino,
        incluyendo las funciones que se procesan en el camino.
        """
        route_info = self.routes.get(destination_node_id, {})
        return route_info.get("path", []), route_info.get("functions", [])

    def receive_packet(self, packet):
        """
        Maneja los paquetes recibidos según su tipo.
        Finaliza el episodio cuando el paquete regresa al nodo sender.
        """
        print(f"[Node_ID={self.node.node_id}] Received packet {packet}")

        match packet["type"]:
            case PacketType.SUCCESS:
                if packet.is_sequence_completed():
                    print(f"[Node_ID={self.node.node_id}] Episode {packet.episode_number} completed with total time {packet.time}")

            case PacketType.BROADCAST:
                print(f"[Node_ID={self.node.node_id}] Received BROADCAST packet with ID {packet.message_id}")

                # Verificar si el mensaje ya fue recibido
                if packet.message_id in self.broadcast_state.received_messages:
                    print(f"[Node_ID={self.node.node_id}] Ignoring duplicate BROADCAST packet.")
                    return  # Ignorar si ya se procesó este mensaje

                # Registrar el mensaje recibido
                self.broadcast_state.received_messages.add(packet.message_id)
                self.broadcast_state.parent_node = packet.from_node_id

                # Propagar el paquete BROADCAST a los vecinos
                neighbors = self.node.network.get_neighbors(self.node.node_id)
                for neighbor in neighbors:
                    if neighbor != packet.from_node_id:  # Evitar reenviar al nodo del que se recibió el paquete
                        broadcast_packet = {
                            "type": PacketType.BROADCAST,
                            "message_id": message_id,
                            "from_node_id": self.node.node_id,
                            "episode_number": packet.episode_number,
                            "visited_nodes": {self.node.node_id},
                        }
                        self.send_packet(neighbor, broadcast_packet)

                # Inicializar contador de ACKs esperados si es necesario
                if self.broadcast_state.expected_acks == 0:
                    self.broadcast_state.expected_acks = len(neighbors) - 1

            case PacketType.ACK:
                print(f"[Node_ID={self.node.node_id}] Received ACK for message ID {packet['message_id']}")

                if "node_function_map" in packet:
                    combined_node_function_map = {**self.broadcast_state.node_function_map, **packet["node_function_map"]}
                    self.broadcast_state.node_function_map = combined_node_function_map
                    print(self.broadcast_state.node_function_map)

                if packet["from_node_id"] not in self.broadcast_state.received_acks:
                    self.broadcast_state.increment_acks_received(packet["from_node_id"])
                    acks_left = self.broadcast_state.expected_acks - self.broadcast_state.acks_received
                    print(f"[Node_ID={self.node.node_id}] {acks_left} ACKs left")
                else:
                    print(f"[Node_ID={self.node.node_id}] Duplicate ACK received from Node {packet['from_node_id']}. Ignoring.")

                # Verificar si se recibieron todos los ACKs esperados
                if self.broadcast_state.acks_received == self.broadcast_state.expected_acks:
                    print(f"[Node_ID={self.node.node_id}] Broadcast completed successfully.")
                    print(f"[Node_ID={self.node.node_id}] Final node-function map: {self.broadcast_state.node_function_map}")
                    self.broadcast_state.mark_completed()
                    if self.broadcast_state.parent_node is not None:
                        ack_packet = {
                            "type": PacketType.ACK,
                            "message_id": message_id,
                            "from_node_id": self.node.node_id,
                            "node_function_map": {}
                        }
                        # Solo agregar al mapa si el nodo tiene una función asignada
                        if self.assigned_function is not None:
                            ack_packet["node_function_map"][self.node.node_id] = self.assigned_function
                            self.broadcast_state.node_function_map[self.node.node_id] = self.assigned_function

                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

            case PacketType.BROKEN_PATH:
                print(f"[Node_ID={self.node.node_id}] Restarting episode {packet.episode_number} because pre calculated shortest path is broken. Packet={packet}")
                self.start_episode(packet.episode_number)

            case _:
                print(f"[Node_ID={self.node.node_id}] Received unknown packet type: {packet.type}")

        # else:
        #     self.previous_node_id = packet.from_node_id
        #     next_node = self.select_next_function_node(packet)
        #
        #     if next_node == None or self.node.network.nodes[next_node].status: # todo: crear un método auxiliar isup o algo así
        #         print(f"[Node_ID={self.node.node_id}] Restarting episode {packet.episode_number} because pre calculated shortest path is broken. Packet={packet}")
        #         self.start_episode(packet.episode_number)
        #
        #     self.send_packet(next_node, packet)

    def _log_routes(self):
        """
        Genera un log gráfico de las rutas calculadas usando tabulate, incluyendo los pesos y las funciones procesadas.
        """
        from tabulate import tabulate
        table = []
        node_function_map = self.broadcast_state.node_function_map  # Funciones asignadas desde el broadcast_state

        for destination, route_info in self.routes.items():
            path = route_info["path"]

            # Obtener las funciones asignadas de cada nodo en el path
            functions = [
                node_function_map.get(node, "None") for node in path
            ]

            path_str = " -> ".join(map(str, path))
            functions_str = " -> ".join(map(str, functions))
            total_latency = sum(
                self.node.network.get_latency(path[i], path[i + 1]) 
                for i in range(len(path) - 1)
            )

            table.append([
                f"{self.node.node_id} to {destination}",
                path_str,
                functions_str,
                f"{total_latency:.6f} s"
            ])

        print("Routes calculated:")
        print(tabulate(table, headers=["Route", "Path", "Functions", "Total Latency"], tablefmt="grid"))

class IntermediateDijkstraApplication(DijkstraApplication):
    def __init__(self, node):
        super().__init__(node)
        self.assigned_function = None
        self.previous_node_id = None
        self.broadcast_state = None

    def start_episode(self, episode_number):
        pass

    def receive_packet(self, packet):
        packet_type = packet["type"]
        print(f"[Node_ID={self.node.node_id}] Received {packet_type} packet.")
        match packet_type:
            case PacketType.BROADCAST:
                print(packet)
                message_id = packet["message_id"]

                # Verificar si el mensaje ya fue recibido
                if self.broadcast_state and message_id in self.broadcast_state.received_messages:
                    print(f"[Node_ID={self.node.node_id}] Received duplicate BROADCAST packet. Sending ACK back.")
                    ack_packet = {
                        "type": PacketType.ACK,
                        "message_id": message_id,
                        "from_node_id": self.node.node_id,
                        "node_function_map": {}
                    }
                    # Solo agregar al mapa si el nodo tiene una función asignada
                    if self.assigned_function is not None:
                        ack_packet["node_function_map"][self.node.node_id] = self.assigned_function
                        self.broadcast_state.node_function_map[self.node.node_id] = self.assigned_function
                    self.send_packet(packet["from_node_id"], ack_packet)
                    return

                # Registrar el mensaje recibido
                if not self.broadcast_state:
                    self.broadcast_state = BroadcastState()
                self.broadcast_state.received_messages.add(message_id)
                self.broadcast_state.parent_node = packet["from_node_id"]

                # Asignarse una función si no tiene una ya asignada
                if not self.assigned_function:
                    # Buscar la función menos asignada globalmente
                    if packet["function_counters"]:
                        available_functions = list(packet["function_counters"].items())

                        # Filtrar las funciones menos asignadas
                        min_count = min(count for _, count in available_functions)
                        least_assigned_functions = [func for func, count in available_functions if count == min_count]

                        # Seleccionar una función aleatoriamente de las menos asignadas
                        function_to_assign = random.choice(least_assigned_functions)

                        # Asignar la función al nodo
                        self.assigned_function = function_to_assign

                        # Incrementar el contador de asignaciones para la función
                        packet["function_counters"][function_to_assign] += 1

                        print(f"[Node_ID={self.node.node_id}] Assigned function: {self.assigned_function}")

                        # Actualizar el diccionario de mapeo de funciones en el broadcast state
                        self.broadcast_state.node_function_map[self.node.node_id] = self.assigned_function
                        packet["node_function_map"][self.node.node_id] = self.assigned_function

                        print(f"[Node_ID={self.node.node_id}] Added function to node function dict: {self.broadcast_state.node_function_map}")
                            # Propagar el paquete a vecinos no visitados
                        neighbors = self.node.network.get_neighbors(self.node.node_id)
                        neighbors_to_broadcast = [
                            n for n in neighbors if n not in packet["visited_nodes"]
                        ]

                        # Inicializar contador de ACKs esperados
                        self.broadcast_state.expected_acks = len(neighbors_to_broadcast)
                        print(f"[Node_ID={self.node.node_id}] {self.broadcast_state.expected_acks} expected ACKs from nodes {neighbors_to_broadcast}")

                        for neighbor in neighbors_to_broadcast:
                            broadcast_packet = {
                                "type": PacketType.BROADCAST,
                                "message_id": message_id,
                                "from_node_id": self.node.node_id,
                                "episode_number": packet["episode_number"],
                                "visited_nodes": packet["visited_nodes"].copy(),
                                "functions_sequence": FUNCTION_SEQ.copy(),
                                "function_counters": {func: 0 for func in FUNCTION_SEQ},
                                "node_function_map": packet["node_function_map"]
                            }
                            broadcast_packet["visited_nodes"].add(self.node.node_id)
                            self.send_packet(neighbor, broadcast_packet)

                        # Enviar ACK al nodo padre si no hay vecinos a los que propagar
                        if not neighbors_to_broadcast:
                            ack_packet = {
                                "type": PacketType.ACK,
                                "message_id": message_id,
                                "from_node_id": self.node.node_id,
                                "node_function_map": {}
                            }
                            # Solo agregar al mapa si el nodo tiene una función asignada
                            if self.assigned_function is not None:
                                ack_packet["node_function_map"][self.node.node_id] = self.assigned_function
                                self.broadcast_state.node_function_map[self.node.node_id] = self.assigned_function

                            self.send_packet(self.broadcast_state.parent_node, ack_packet)

            case PacketType.ACK:
                print(f"[Node_ID={self.node.node_id}] Received ACK for message ID {packet['message_id']}")
                print("printing mf packet to see if has map of mf nodes and mf functions")
                print(packet)

                if packet["from_node_id"] not in self.broadcast_state.received_acks:
                    self.broadcast_state.increment_acks_received(packet["from_node_id"])
                    acks_left = self.broadcast_state.expected_acks - self.broadcast_state.acks_received
                    print(f"[Node_ID={self.node.node_id}] {acks_left} ACKs left")

                if "node_function_map" not in packet:
                    packet["node_function_map"] = self.broadcast_state.node_function_map
                combined_node_function_map = {**self.broadcast_state.node_function_map, **packet["node_function_map"]}
                if self.assigned_function is not None:
                    combined_node_function_map[self.node.node_id] = self.assigned_function

                self.broadcast_state.node_function_map = combined_node_function_map

                print(f"[Node_ID={self.node.node_id}] add node function to node function map {self.broadcast_state.node_function_map}")

                # Verificar si se recibieron todos los ACKs esperados
                if self.broadcast_state.acks_received == self.broadcast_state.expected_acks:
                    print(f"[Node_ID={self.node.node_id}] All ACKs received. Sending ACK to parent node {self.broadcast_state.parent_node}.")

                    message_id = packet["message_id"]
                    # Enviar ACK al nodo padre
                    if self.broadcast_state.parent_node is not None:

                        combined_node_function_map = {**self.broadcast_state.node_function_map, **packet["node_function_map"]}
                        self.broadcast_state.node_function_map = combined_node_function_map

                        ack_packet = {
                            "type": PacketType.ACK,
                            "message_id": message_id,
                            "from_node_id": self.node.node_id,
                            "node_function_map": self.broadcast_state.node_function_map
                        }

                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

                    # Marcar el broadcast como completado
                    self.broadcast_state.mark_completed()

            case PacketType.PACKET_HOP:
                if not self.assigned_function:
                    function_to_assign = packet.next_function()
                    self.assigned_function = function_to_assign
                    packet.increment_function_counter(function_to_assign)
                    print(f"[Node_ID={self.node.node_id}] Assigned function: {self.assigned_function}")

                print(f"[Node_ID={self.node.node_id}] Processing packet at node {self.node}: {packet}")
                # lógica para reenviar el paquete al siguiente nodo si faltan funciones
                if not packet.is_sequence_completed():
                    self.previous_node_id = packet.from_node_id
                    next_node = self.select_next_function_node(packet)

                    if next_node == None or self.node.network.nodes[next_node].status: # todo: crear un método auxiliar isup o algo así
                        print(f"[Node_ID={self.node.node_id}] Broken path at node {self.node}: {packet}")
                        broken_path_packet = PACKET(episode_number, self.node.node_id, PacketType.BROKEN_PATH)
                        self.send_packet(previous_node_id, broken_path_packet)

                    self.send_packet(next_node, packet)
                else:
                    success_packet = PACKET(episode_number, self.node.node_id, PacketType.SUCCESS)
                    self.send_packet(previous_node_id, success_packet)

            case PacketType.BROKEN_PATH:
                self.send_packet(previous_node_id, packet)

            case PacketType.SUCCESS:
                self.send_packet(previous_node_id, packet)

    def get_assigned_functions(self):
        """
        Devuelve la función asignada a este nodo.
        """
        return [self.assigned_function] if self.assigned_function else []

# Diccionario global para llevar el conteo de funciones asignadas
GLOBAL_FUNCTION_COUNTER = {func: 0 for func in FUNCTION_SEQ}

def assign_function_to_node(node):
    """
    Asigna una función a un nodo basándose en el contador global.
    """
    global GLOBAL_FUNCTION_COUNTER

    # Obtener la función menos asignada
    min_count = min(GLOBAL_FUNCTION_COUNTER.values())
    least_assigned_functions = [func for func, count in GLOBAL_FUNCTION_COUNTER.items() if count == min_count]

    # Seleccionar una función aleatoriamente de las menos asignadas
    function_to_assign = random.choice(least_assigned_functions)

    # Asignar la función al nodo y actualizar el contador global
    node.assigned_function = function_to_assign
    GLOBAL_FUNCTION_COUNTER[function_to_assign] += 1

    print(f"[Node_ID={node.node_id}] Assigned global function: {function_to_assign}")

    # Actualizar el mapeo en el broadcast state (si aplica)
    if hasattr(node, 'broadcast_state') and node.broadcast_state:
        node.broadcast_state.node_function_map[node.node_id] = function_to_assign
