from enum import Enum
from classes.base import Application
import random

class NodeFunction(Enum):
    A = "A"
    B = "B"
    C = "C"

FUNCTION_SEQ = [NodeFunction.A, NodeFunction.B, NodeFunction.C]

global_function_counters = {func: 0 for func in FUNCTION_SEQ}

broken_path = False

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

class BellmanFordApplication(Application):
    def __init__(self, node):
        super().__init__(node)
        self.routes = {}  # Almacena las rutas más cortas calculadas
        self.callback_stack = []

    def select_next_function_node(self, packet):
        """
        Selecciona el siguiente nodo que debe procesar la próxima función faltante.
        Si todos los nodos vecinos no tienen la función correspondiente en el mapa, elige el nodo con menor peso en la arista.
        """
        next_function = packet["functions_sequence"][0]  # Obtener la próxima función de la secuencia
        neighbors = self.node.network.get_neighbors(self.node.node_id)

        print(f"[Node_ID={self.node.node_id}] Selecting next node to process function: {next_function}")
        print(f"[Node_ID={self.node.node_id}] Neighbors: {neighbors}")
        print(f"[Node_ID={self.node.node_id}] Functions to node map: {packet['node_function_map']}")

        # Filtrar nodos que puedan procesar la próxima función según el mapa functions_to_node_map
        valid_neighbors = [
            neighbor for neighbor in neighbors
            if packet["node_function_map"].get(neighbor) == next_function
        ]

        print(f"[Node_ID={self.node.node_id}] Valid neighbors for function {next_function}: {valid_neighbors}")

        # Si hay nodos que pueden procesar la próxima función, elegir el más cercano
        if valid_neighbors:
            selected_node = min(valid_neighbors, key=lambda n: self.node.network.get_latency(self.node.node_id, n))
            print(f"[Node_ID={self.node.node_id}] Selected node {selected_node} to process function {next_function}")
            return selected_node

        # Si no hay vecinos válidos para procesar la función requerida
        # Considerar solo nodos vecinos que no tienen función asignada, excluyendo explícitamente el nodo 0 (sender)
        neighbors_without_function = [
            neighbor for neighbor in neighbors
            if neighbor not in packet["node_function_map"] and neighbor != 0
        ]

        print(f"[Node_ID={self.node.node_id}] Neighbors without assigned function: {neighbors_without_function}")

        if neighbors_without_function:
            selected_node = min(neighbors_without_function, key=lambda n: self.node.network.get_latency(self.node.node_id, n))
            print(f"[Node_ID={self.node.node_id}] Selected node {selected_node} without assigned function")
            return selected_node

        # Si todos los nodos tienen función asignada pero ninguna coincide, elegir el más cercano que no sea el nodo 0
        valid_closest_neighbors = [n for n in neighbors if n != 0]

        if valid_closest_neighbors:
            selected_node = min(valid_closest_neighbors, key=lambda n: self.node.network.get_latency(self.node.node_id, n))
            print(f"[Node_ID={self.node.node_id}] Selected closest node {selected_node} (excluding 0)")
            return selected_node

        # En caso extremo, si solo queda el nodo 0 como opción (por diseño, no debería ocurrir)
        print(f"[Node_ID={self.node.node_id}] No other nodes available. Defaulting to node 0.")
        return 0

    def send_packet(self, to_node_id, packet, lost_packet=False):

        if lost_packet:
            self.node.network.send_dict(None, None, None, True)
            return

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

class SenderBellmanFordApplication(BellmanFordApplication):
    def __init__(self, node):
        super().__init__(node)
        self.routes = {}  # Almacena las rutas más cortas calculadas
        self.previous_node_id = None

    def start_episode(self, episode_number) -> None:
        global broken_path
        if broken_path or episode_number == 1:
            broken_path = False
            print(f"[Node_ID={self.node.node_id}] Starting broadcast for episode {episode_number}")

            # Iniciar el broadcast para recopilar latencias y asignar funciones
            message_id = f"broadcast_{self.node.node_id}_{episode_number}"

            self.broadcast_state = BroadcastState()
            self.start_broadcast(message_id, episode_number)

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
            "max_hops": 250,  # Número máximo de saltos permitidos
            "node_function_map": self.broadcast_state.node_function_map
        }
        next_node = self.select_next_function_node(packet)

        if next_node is None:
            print("No suitable next node found.")
            return

        self.send_packet(next_node, packet)
        return

    def start_broadcast(self, message_id, episode_number):
        """
        Inicia el proceso de broadcast desde el nodo sender.
        """
        neighbors = self.node.network.get_neighbors(self.node.node_id)
        broadcast_packet = {
            "type": PacketType.BROADCAST,
            "message_id": message_id,
            "from_node_id": self.node.node_id,
            "episode_number": episode_number,
            "visited_nodes": {self.node.node_id},
            "functions_sequence": FUNCTION_SEQ.copy(),
            "function_counters": {func: 0 for func in FUNCTION_SEQ},
            "node_function_map": {},
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
        utilizando el algoritmo de Bellman-Ford.
        """
        self.paths_computed = False  # Inicializar la bandera en False

        # Inicializar distancias y nodos previos
        distances = {node_id: float('inf') for node_id in self.node.network.get_nodes()}
        distances[self.node.node_id] = 0
        previous_nodes = {node_id: None for node_id in self.node.network.get_nodes()}

        # Recopilar todas las aristas del grafo
        edges = []
        for node_id in self.node.network.get_nodes():
            for neighbor in self.node.network.get_neighbors(node_id):
                weight = self.node.network.get_latency(node_id, neighbor)
                edges.append((node_id, neighbor, weight))  # (nodo_origen, nodo_destino, peso)

        # Relajar todas las aristas |V| - 1 veces
        for _ in range(len(self.node.network.get_nodes()) - 1):
            for u, v, weight in edges:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous_nodes[v] = u

        # Detección de ciclos negativos
        for u, v, weight in edges:
            if distances[u] + weight < distances[v]:
                raise ValueError("El grafo contiene un ciclo negativo")

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
            case PacketType.PACKET_HOP:
                print(f"[Node_ID={self.node.node_id}] Processing packet at node {self.node}: {packet}")

                # lógica para reenviar el paquete al siguiente nodo si faltan funciones
                if packet["functions_sequence"]:
                    print(f"[Node_ID={self.node.node_id}] Remaining functions: {packet['functions_sequence']}")
                    self.previous_node_id = packet["from_node_id"]

                    # Seleccionar el siguiente nodo
                    next_node = self.select_next_function_node(packet)

                    # Verificar si el siguiente nodo es válido
                    if next_node is None or self.node.network.nodes[next_node].status:  # TODO: Crear un método auxiliar is_up o algo similar
                        episode_number = packet["episode_number"]
                        print(f"[Node_ID={self.node.node_id}] Restarting episode {episode_number} because pre calculated shortest path is broken. Packet={packet}")
                        self.start_episode(episode_number, True)
                    else:
                        # Reenviar el paquete al siguiente nodo
                        self.callback_stack.append(packet["from_node_id"])
                        self.send_packet(next_node, packet)
                else:
                    # Si la secuencia de funciones está completa
                    print(f"[Node_ID={self.node.node_id}] Function sequence completed.")
                    # print(f"[Node_ID={self.node.node_id}] Episode {packet.episode_number} completed with total time {packet.time}")
                    episode_number = packet["episode_number"]
                    print(f"[Node_ID={self.node.node_id}] Episode {episode_number} completed")

            case PacketType.SUCCESS:
                # print(f"[Node_ID={self.node.node_id}] Episode {packet.episode_number} completed with total time {packet.time}")
                episode_number = packet["episode_number"]
                print(f"[Node_ID={self.node.node_id}] Episode {episode_number} completed")

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
                            "node_function_map": {},
                            "episode_number": packet.episode_number
                        }
                        # Solo agregar al mapa si el nodo tiene una función asignada
                        if self.assigned_function is not None:
                            ack_packet["node_function_map"][self.node.node_id] = self.assigned_function
                            self.broadcast_state.node_function_map[self.node.node_id] = self.assigned_function

                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

            case PacketType.BROKEN_PATH:
                episode_number = packet["episode_number"]
                print(f"[Node_ID={self.node.node_id}] Episode {episode_number} detected a broken path. Packet={packet}")
                global broken_path
                broken_path = True
                self.send_packet(None, None, True)
                # TODO: acá habría que revisar que el paquete quede como que no fue entregado
                # self.start_episode(episode_number, True)

            case _:
                packet_type = packet["type"]
                print(f"[Node_ID={self.node.node_id}] Received unknown packet type: {packet_type}")

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

class IntermediateBellmanFordApplication(BellmanFordApplication):
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
                        "node_function_map": {},
                        "episode_number": packet["episode_number"]
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

                #
                # Asignarse una función si no tiene una ya asignada
                #
                if not self.assigned_function:
                    # Buscar la función menos asignada globalmente
                    min_count = min(global_function_counters.values())
                    least_assigned_functions = [
                        func for func, count in global_function_counters.items() if count == min_count
                    ]

                    # Seleccionar una función aleatoriamente de las menos asignadas
                    function_to_assign = random.choice(least_assigned_functions)

                    # Asignar la función al nodo
                    self.assigned_function = function_to_assign

                    # Incrementar el contador global de asignaciones para la función
                    global_function_counters[function_to_assign] += 1

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
                        "node_function_map": {},
                        "episode_number": packet["episode_number"]
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
                            "node_function_map": self.broadcast_state.node_function_map,
                            "episode_number": packet["episode_number"]
                        }

                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

                    # Marcar el broadcast como completado
                    self.broadcast_state.mark_completed()

            case PacketType.PACKET_HOP:

                # Si el nodo tiene una función asignada, procesarla
                if self.assigned_function:
                    if packet["functions_sequence"] and packet["functions_sequence"][0] == self.assigned_function:
                        # Procesar la función asignada
                        print(f"[Node_ID={self.node.node_id}] Processing assigned function: {self.assigned_function}")

                        # Eliminar la función procesada de la secuencia
                        packet["functions_sequence"].pop(0)
                        print(f"[Node_ID={self.node.node_id}] Function {self.assigned_function} removed from sequence. Remaining: {packet['functions_sequence']}")
                else:
                    function_to_assign = packet.next_function()
                    self.assigned_function = function_to_assign
                    packet.increment_function_counter(function_to_assign)
                    print(f"[Node_ID={self.node.node_id}] Assigned function: {self.assigned_function}")

                    print(f"[Node_ID={self.node.node_id}] Processing assigned function: {self.assigned_function}")

                    # Eliminar la función procesada de la secuencia
                    packet["functions_sequence"].pop(0)
                    print(f"[Node_ID={self.node.node_id}] Function {self.assigned_function} removed from sequence. Remaining: {packet['functions_sequence']}")

                # lógica para reenviar el paquete al siguiente nodo si faltan funciones
                if packet["functions_sequence"]:
                    self.previous_node_id = packet["from_node_id"]

                    # Seleccionar el siguiente nodo
                    next_node = self.select_next_function_node(packet)

                    print(next_node is None or not self.node.network.nodes[next_node].status)

                    # Verificar si el siguiente nodo es válido
                    if next_node is None or not self.node.network.nodes[next_node].status:  # TODO: Crear un método auxiliar is_up o algo similar
                        print(f"[Node_ID={self.node.node_id}] Broken path at node {self.node.node_id}: {packet}")

                        # Crear un paquete BROKEN_PATH como dict
                        broken_path_packet = {
                            "type": PacketType.BROKEN_PATH,
                            "episode_number": packet["episode_number"],
                            "from_node_id": self.node.node_id,
                            "hops": packet["hops"] + 1
                        }

                        # Enviar el paquete BROKEN_PATH al nodo anterior
                        self.send_packet(packet["from_node_id"], broken_path_packet)
                    else:
                        # Reenviar el paquete al siguiente nodo
                        self.callback_stack.append(packet["from_node_id"])
                        self.send_packet(next_node, packet)
                else:
                    # Si la secuencia de funciones está completa
                    print(f"[Node_ID={self.node.node_id}] Function sequence completed.")

                    # Crear un paquete SUCCESS como dict
                    success_packet = {
                        "type": PacketType.SUCCESS,
                        "episode_number": packet["episode_number"],
                        "from_node_id": self.node.node_id,
                        "hops": packet["hops"] + 1
                    }

                    # Enviar el paquete SUCCESS al nodo anterior
                    from_node_id = packet["from_node_id"]
                    print(f"[Node_ID={self.node.node_id}] Sending SUCCESS packet back to node {from_node_id}.")
                    self.send_packet(from_node_id, success_packet)

            case PacketType.BROKEN_PATH:
                previous_node = self.callback_stack.pop()
                self.send_packet(previous_node, packet)

            case PacketType.SUCCESS:
                previous_node = self.callback_stack.pop()
                self.send_packet(previous_node, packet)

    def get_assigned_functions(self):
        """
        Devuelve la función asignada a este nodo.
        """
        return [self.assigned_function] if self.assigned_function else []
