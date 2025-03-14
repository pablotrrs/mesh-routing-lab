import time
from enum import Enum
from classes.base import Application, EpisodeEnded
import random
import threading
from classes.clock import clock
from classes.packet_registry import packet_registry as registry
from tabulate import tabulate

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
        self.latency_map = {}

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
    MAX_HOPS = "MAX_HOPS"

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

    def send_packet(self, to_node_id, packet):

        if "hops" in packet:
            packet["hops"] += 1
        else:
            packet["hops"] = 0

        if "time" in packet:
            packet["time"] += 1

        if "from_node_id" in packet:
            packet["from_node_id"] = self.node.node_id

        print(f'\n[Node_ID={self.node.node_id}] Sending packet to Node {to_node_id}\n')
        self.node.network.send(self.node.node_id, to_node_id, packet)

    def get_assigned_function(self) -> str:
        """Returns the function assigned to this node or 'N/A' if None."""
        func = self.broadcast_state.node_function_map.get(self.node.node_id)

        return func.value if func is not None else "N/A"

class SenderBellmanFordApplication(BellmanFordApplication):
    def __init__(self, node):
        super().__init__(node)
        self.routes = {}  # Almacena las rutas más cortas calculadas
        self.previous_node_id = None
        self.last_route_update = 0  # Última vez que se ejecutó el recalculo de rutas
        self.running = True  # Control del hilo de monitoreo
        self.max_hops = None
        self.functions_sequence = None

    def start_route_monitoring(self):
        """Inicia un hilo que verifica periódicamente si se debe ejecutar Bellman-Ford."""
        threading.Thread(target=self._monitor_route_updates, daemon=True).start()

    def stop_route_monitoring(self):
        """Detiene el monitoreo de actualización de rutas."""
        self.running = False

    def _monitor_route_updates(self):
        """Monitorea el reloj central y ejecuta Bellman-Ford cada 30 segundos."""
        while self.running:
            current_time = self.node.network.simulation_clock.get_current_time()

            if current_time - self.last_route_update >= 30000:  # 30 segundos en ms
                print(f"[Node {self.node.node_id}] Recalculando rutas con Bellman-Ford en {current_time} ms")
                self.compute_shortest_paths()
                self.last_route_update = current_time

            time.sleep(0.1)  # Evita sobrecarga de CPU

    def start_episode(self, episode_number):
        self.start_route_monitoring()

        if episode_number == 1:
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
            "functions_sequence": self.functions_sequence.copy(),  # Copia de la secuencia de funciones a procesar
            "function_counters": {func: 0 for func in self.functions_sequence},  # Contadores de funciones asignadas
            "hops": 0,  # Contador de saltos
            "max_hops": self.max_hops,  # Número máximo de saltos permitidos
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
            "functions_sequence": self.functions_sequence.copy(),
            "function_counters": {func: 0 for func in self.functions_sequence},
            "node_function_map": {},
            "latency_map": {},
        }

        # Inicializar el estado de broadcast
        self.broadcast_state = BroadcastState()
        self.broadcast_state.expected_acks = len(neighbors)
        print(f"[Node_ID={self.node.node_id}] Expected ACKs: {self.broadcast_state.expected_acks}")

        for neighbor in neighbors:
            start_time = clock.get_current_time()
            self.send_packet(neighbor, broadcast_packet)
            broadcast_packet["latency_map"][(self.node.node_id, neighbor)] = start_time  # 🔥 Guardamos la latencia


        # dejar solo las latencias mínimas
        self.broadcast_state.latency_map = {
            (min(a, b), max(a, b)): min(latency for (x, y), latency in self.broadcast_state.latency_map.items() 
                                    if {x, y} == {a, b})
            for a, b in self.broadcast_state.latency_map
        }

        latency_table = [
            [src, dst, latency] for (src, dst), latency in self.broadcast_state.latency_map.items()
        ]
        print(f"\n[Node_ID={self.node.node_id}] Latency Map After Broadcast:\n")
        print(tabulate(latency_table, headers=["Source Node", "Destination Node", "Latency (ms)"], tablefmt="grid"))

    def compute_shortest_paths(self):
        """
        Calcula las rutas más cortas desde el nodo de origen a todos los demás nodos,
        utilizando Bellman-Ford con las latencias medidas en el broadcast.
        """
        self.paths_computed = False

        distances = {node_id: float('inf') for node_id in self.node.network.get_nodes()}
        distances[self.node.node_id] = 0
        previous_nodes = {node_id: None for node_id in self.node.network.get_nodes()}

        edges = []
        for node_id in self.node.network.get_nodes():
            for neighbor in self.node.network.get_neighbors(node_id):
                latency = self.node.application.broadcast_state.latency_map.get(
                    (node_id, neighbor), float('inf')
                )
                edges.append((node_id, neighbor, latency))

        for _ in range(len(self.node.network.get_nodes()) - 1):
            for u, v, weight in edges:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous_nodes[v] = u

        for u, v, weight in edges:
            if distances[u] + weight < distances[v]:
                raise ValueError("El grafo contiene un ciclo negativo")

        self.routes = self._reconstruct_paths(self.node.node_id, previous_nodes)
        self._log_routes()
        self.paths_computed = True

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

            case PacketType.MAX_HOPS:
                episode_number = packet["episode_number"]
                print(f"\n[Node_ID={self.node.node_id}] Episode {episode_number} failed.")

                self.mark_episode_result(packet, success=False)

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
                    episode_number = packet["episode_number"]
                    print(f"[Node_ID={self.node.node_id}] Episode {episode_number} completed")

            case PacketType.SUCCESS:
                episode_number = packet["episode_number"]
                print(f"[Node_ID={self.node.node_id}] Episode {episode_number} completed")
                self.mark_episode_result(packet, success=True)

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
                            # "latency_map": self.broadcast_state.latency_map,
                            "latency_map": packet["latency_map"],
                        }
                        self.send_packet(neighbor, broadcast_packet)

                # Inicializar contador de ACKs esperados si es necesario
                if self.broadcast_state.expected_acks == 0:
                    self.broadcast_state.expected_acks = len(neighbors) - 1

            case PacketType.ACK:
                print(f"[Node_ID={self.node.node_id}] Received ACK for message ID {packet['message_id']}")

                # 📌 Guardamos el tiempo actual al recibir el ACK
                end_time = clock.get_current_time()

                # 📌 Fusionar `node_function_map`
                if "node_function_map" in packet:
                    self.broadcast_state.node_function_map.update(packet["node_function_map"])
                    print(f"[Node_ID={self.node.node_id}] Updated node-function map: {self.broadcast_state.node_function_map}")

                # 📌 Fusionar `latency_map` (si existe en el paquete)
                if "latency_map" in packet:
                    for (src, dst), latency in packet["latency_map"].items():
                        # 🔥 Solo agregamos si no está o si encontramos una latencia menor (mejor ruta)
                        if (src, dst) not in self.broadcast_state.latency_map or latency < self.broadcast_state.latency_map[(src, dst)]:
                            self.broadcast_state.latency_map[(src, dst)] = latency
                            print(f"[Node_ID={self.node.node_id}] Added latency {latency} ms for route {src} -> {dst}")

                # 📌 Verificar si es un nuevo ACK (evitar duplicados)
                if packet["from_node_id"] not in self.broadcast_state.received_acks:
                    self.broadcast_state.increment_acks_received(packet["from_node_id"])
                    acks_left = self.broadcast_state.expected_acks - self.broadcast_state.acks_received
                    print(f"[Node_ID={self.node.node_id}] {acks_left} ACKs left")

                    # 📌 Calcular latencia solo si tenemos el tiempo de envío guardado
                    if (packet["from_node_id"], self.node.node_id) in self.broadcast_state.latency_map:
                        start_time = self.broadcast_state.latency_map[(packet["from_node_id"], self.node.node_id)]
                        latency = end_time - start_time
                        self.broadcast_state.latency_map[(packet["from_node_id"], self.node.node_id)] = latency
                        print(f"[Node_ID={self.node.node_id}] Measured latency from {packet['from_node_id']}: {latency} ms")

                else:
                    print(f"[Node_ID={self.node.node_id}] Duplicate ACK received from Node {packet['from_node_id']}. Ignoring.")

                # 📌 Verificar si se recibieron todos los ACKs esperados
                if self.broadcast_state.acks_received == self.broadcast_state.expected_acks:
                    print(f"[Node_ID={self.node.node_id}] Broadcast completed successfully.")
                    print(f"[Node_ID={self.node.node_id}] Final node-function map: {self.broadcast_state.node_function_map}")
                    print(f"[Node_ID={self.node.node_id}] Final latency map: {self.broadcast_state.latency_map}")
                    self.broadcast_state.mark_completed()

                    if self.broadcast_state.parent_node is not None:
                        ack_packet = {
                            "type": PacketType.ACK,
                            "message_id": packet["message_id"],
                            "from_node_id": self.node.node_id,
                            "node_function_map": self.broadcast_state.node_function_map,
                            # "latency_map": self.broadcast_state.latency_map,  # 🔥 Enviar el latency_map completo
                            "latency_map": self.broadcast_state.latency_map,  # 🔥 Enviar el latency_map completo
                            "episode_number": packet["episode_number"]
                        }
                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

            case PacketType.BROKEN_PATH:
                packet["hops"] += 1
                registry.mark_packet_lost(packet["episode_number"], packet["from_node_id"], None, packet["type"].value)

                episode_number = packet["episode_number"]
                print(f"[Node_ID={self.node.node_id}] Episode {episode_number} detected a broken path. Packet={packet}")

            case _:
                packet_type = packet["type"]
                print(f"[Node_ID={self.node.node_id}] Received unknown packet type: {packet_type}")

    def mark_episode_result(self, packet, success=True):
        """
        Marca un episodio como exitoso o fallido y lo notifica a la red.

        Args:
            packet (Packet): El paquete asociado al episodio.
            success (bool): `True` si el episodio fue exitoso, `False` si falló.
        """
        status_text = "SUCCESS" if success else "FAILURE"
        episode_number = packet["episode_number"]
        print(f"\n[Node_ID={self.node.node_id}] Marking Episode {episode_number} as {status_text}.")

        registry.mark_episode_complete(episode_number, success)

        self.stop_route_monitoring()
        raise EpisodeEnded()

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

                # 📌 Verificar si el mensaje ya fue recibido (para evitar loops)
                if self.broadcast_state and message_id in self.broadcast_state.received_messages:
                    print(f"[Node_ID={self.node.node_id}] Received duplicate BROADCAST packet. Sending ACK back.")

                    # 📌 Enviar ACK de vuelta con `latency_map` fusionado
                    ack_packet = {
                        "type": PacketType.ACK,
                        "message_id": message_id,
                        "from_node_id": self.node.node_id,
                        "node_function_map": {},
                        "episode_number": packet["episode_number"],
                        "latency_map": packet["latency_map"],  # 🔥 Asegurar que el mapa se propague correctamente
                    }

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
                    min_count = min(packet["function_counters"].values())
                    least_assigned_functions = [
                        func for func, count in packet["function_counters"].items() if count == min_count
                    ]

                    # Seleccionar una función aleatoriamente de las menos asignadas
                    function_to_assign = random.choice(least_assigned_functions)

                    # Asignar la función al nodo
                    self.assigned_function = function_to_assign

                    # Incrementar el contador global de asignaciones para la función
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

                # 📌 Guardamos el `start_time` en el `latency_map` dentro del paquete
                updated_latency_map = packet["latency_map"].copy()
                for neighbor in neighbors_to_broadcast:
                    updated_latency_map[(self.node.node_id, neighbor)] = clock.get_current_time()

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
                        "functions_sequence": packet["functions_sequence"].copy(),
                        "function_counters": {func: 0 for func in packet["functions_sequence"]},
                        "node_function_map": packet["node_function_map"],
                        "latency_map": updated_latency_map  # 🔥 Ahora el `latency_map` viaja en el paquete
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
                        "episode_number": packet["episode_number"],
                        "latency_map": packet["latency_map"],
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

                ack_from = packet["from_node_id"]
                end_time = clock.get_current_time()

                # 📌 Calcular la latencia real en el `latency_map` del paquete
                if (self.node.node_id, ack_from) in packet["latency_map"]:
                    start_time = packet["latency_map"][(self.node.node_id, ack_from)]
                    latency = end_time - start_time
                    packet["latency_map"][(self.node.node_id, ack_from)] = latency
                    print(f"[Node_ID={self.node.node_id}] Measured latency from {ack_from}: {latency} ms")

                # 📌 Fusionamos `latency_map` del paquete con los datos recibidos
                combined_latency_map = {**self.broadcast_state.latency_map, **packet["latency_map"]}
                self.broadcast_state.latency_map = combined_latency_map

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

                        combined_latency_map = {**self.broadcast_state.latency_map, **packet["latency_map"]}
                        self.broadcast_state.latency_map = combined_latency_map

                        ack_packet = {
                            "type": PacketType.ACK,
                            "message_id": message_id,
                            "from_node_id": self.node.node_id,
                            "node_function_map": self.broadcast_state.node_function_map,
                            "episode_number": packet["episode_number"],
                            "latency_map": combined_latency_map,
                        }

                        self.send_packet(self.broadcast_state.parent_node, ack_packet)

                    # Marcar el broadcast como completado
                    self.broadcast_state.mark_completed()

            case PacketType.MAX_HOPS:
                previous_node = self.callback_stack.pop()
                self.send_packet(previous_node, packet)

            case PacketType.PACKET_HOP:

                global MAX_HOPS
                if packet["hops"] > packet["max_hops"]:
                    print(f'[Node_ID={self.node.node_id}] Max hops reached. Initiating callback')
                    # print(f"*******callback_stack: {self.callback_stack}")

                    failure_packet = {
                        "type": PacketType.MAX_HOPS,
                        "episode_number": packet["episode_number"],
                        "from_node_id": self.node.node_id,
                        "hops": packet["hops"] + 1,
                    }

                    from_node_id = packet["from_node_id"]
                    print(f"[Node_ID={self.node.node_id}] Sending MAX_HOPS packet back to node {from_node_id}.")
                    self.send_packet(from_node_id, failure_packet)
                    return

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
