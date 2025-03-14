import numpy as np
import time
import yaml
import math
import threading
from classes.clock import clock
from classes.packet_registry import packet_registry as registry

class Network:
    def __init__(self):
        self.nodes = {}  # {node_id: Node}
        self.connections = {}  # {node_id: [neighbors]}
        self.active_nodes = set()
        self.dynamic_change_events = []  # Tiempos (ms) de cambios dinámicos
        self.running = False  # Control del hilo
        self.lock = threading.Lock()  # Para acceso seguro al reloj y nodos
        self.mean_interval_ms = None
        self.reconnect_interval_ms = None

    def set_mean_interval_ms(self, mean_interval_ms):
        self.mean_interval_ms = mean_interval_ms

    def set_reconnect_interval_ms(self, reconnect_interval_ms):
        self.reconnect_interval_ms = reconnect_interval_ms

    def set_disconnect_probability(self, disconnect_probability):
        self.disconnect_probability = disconnect_probability

    def generate_next_dynamic_change(self):
        """
        Genera el próximo cambio dinámico en función de una distribución exponencial.

        Args:
            mean_interval_ms (float): Intervalo promedio entre eventos dinámicos.

        Returns:
            int: Tiempo (ms) para el próximo cambio dinámico.
        """
        if self.mean_interval_ms == float('inf'):
            # No generar cambios dinámicos; retorna un tiempo muy alto
            return int(1e12)  # Un valor grande que nunca será alcanzado en la simulación
        else:
            # Generar el próximo cambio dinámico normalmente
            return int(np.random.exponential(self.mean_interval_ms))

    def start_dynamic_changes(self):
        """Inicia un hilo que aplica los cambios dinámicos en función del reloj central."""
        self.running = True
        current_time = clock.get_current_time()
        print(f"[Network] clock starts: {current_time}")
        threading.Thread(target=self._monitor_dynamic_changes, daemon=True).start()

    def stop_dynamic_changes(self):
        """Detiene el hilo de cambios dinámicos."""
        self.running = False

    def _monitor_dynamic_changes(self):
        """Monitorea el reloj central y aplica cambios dinámicos automáticamente."""
        next_event_time = clock.get_current_time() + self.generate_next_dynamic_change()
        while self.running:
            current_time = clock.get_current_time()
            # print(f"[Network] clock tics: {current_time}")
            with self.lock:
                if current_time >= next_event_time:
                    print("\033[93m\n⚡ZZZAP⚡\n\033[0m")
                    # print(f"[Network] Dynamic Change triggered at {current_time} ms")
                    self.trigger_dynamic_change()
                    self.dynamic_change_events.append(current_time)
                    next_event_time = current_time + self.generate_next_dynamic_change()

                self._handle_reconnections(current_time)

    def trigger_dynamic_change(self):
        """
        Aplica la lógica de cambios dinámicos en la red.
        Ejemplo: desconexión aleatoria de nodos.
        """
        for node_id in list(self.active_nodes):
            if np.random.rand() < self.disconnect_probability:
                self.nodes[node_id].status = False
                self.nodes[node_id].reconnect_time = np.random.exponential(scale=self.reconnect_interval_ms)

                current_time = clock.get_current_time()
                print(f"\033[91m\n⚡[Network] Node {node_id} disconnected at {current_time:.2f}. ⚡\n\033[0m")

        return

    def _handle_reconnections(self, current_time):
        """
        Verifica qué nodos desconectados pueden reconectarse y los reactiva.
        """
        for node_id, node in self.nodes.items():
            if not node.status and node.reconnect_time is not None and current_time >= node.reconnect_time:
                node.status = True
                node.reconnect_time = None

                current_time = clock.get_current_time()
                print(f"\033[92m\n⚡ [Network] Node {node_id} reconnected at {current_time:.2f}. ⚡\n\033[0m")
                return

    def validate_connection(self, from_node_id, to_node_id):
        """
        Verifica si una conexión entre dos nodos es válida.

        Args:
            from_node_id (int): ID del nodo de origen.
            to_node_id (int): ID del nodo de destino.

        Returns:
            bool: True si la conexión es válida, False de lo contrario.
        """
        with self.lock:
            return (
                to_node_id in self.connections.get(from_node_id, []) and
                from_node_id in self.active_nodes and
                to_node_id in self.active_nodes
            )

    def add_node(self, node):
        self.nodes[node.node_id] = node
        self.connections[node.node_id] = []
        self.active_nodes.add(node.node_id)

    def connect_nodes(self, node1_id, node2_id):
        """Conecta dos nodos en la red sin duplicar conexiones."""
        if node2_id not in self.connections.get(node1_id, []):
            self.connections.setdefault(node1_id, []).append(node2_id)
        if node1_id not in self.connections.get(node2_id, []):
            self.connections.setdefault(node2_id, []).append(node1_id)

    def get_neighbors(self, node_id):
        """Returns a node's neighbors, excluding itself."""
        neighbors = self.connections.get(node_id, [])
        return list(set(neighbor for neighbor in neighbors if neighbor != node_id))

    def get_nodes(self):
        """
        Returns a list of all node IDs in the network.
        """
        return list(self.nodes.keys())

    def is_node_reachable(self, from_node_id, to_node_id):
        return to_node_id in self.connections.get(from_node_id, []) and \
                from_node_id in self.active_nodes and \
                to_node_id in self.active_nodes

    def send(self, from_node_id, to_node_id, packet):
        """
        Envía un paquete en la red sin necesidad de parámetros extra para marcar episodios o pérdidas.
        """
        episode_number = packet.get("episode_number")
        registry.initialize_episode(episode_number)

        # **Caso: Paquete perdido**
        if not self.is_node_reachable(from_node_id, to_node_id):
            registry.mark_packet_lost(episode_number, from_node_id, to_node_id, packet["type"].value)
            return  # No seguir procesando

        # **Caso: Fin del episodio (llegó al nodo original)**
        if to_node_id is None:
            registry.mark_episode_complete(episode_number, True)
            return  # No seguir procesando

        # **Calcular latencia**
        latency = self.get_latency(from_node_id, to_node_id) if to_node_id != "N/A" else 0

        # **Registrar información del hop en packet_log**
        registry.log_packet_hop(
            episode_number,
            from_node_id,
            to_node_id,
            self.nodes[from_node_id].get_assigned_function() if self.nodes[from_node_id].get_assigned_function() else "N/A",
            "active" if from_node_id in self.active_nodes else "inactive",
            latency,
            packet["type"].value
        )

        # **Validar si el nodo destino es alcanzable**
        if self.is_node_reachable(from_node_id, to_node_id):
            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")

            time.sleep(latency)  # Simulación de la latencia
            self.nodes[to_node_id].application.receive_packet(packet)
        else:
            print(f"[Network] Failed to send packet: Node {to_node_id} is not reachable from Node {from_node_id}")

    @classmethod
    def from_yaml(cls, file_path):
        """
        Crea una instancia de Network a partir de un archivo YAML y devuelve la red y el nodo emisor.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Crear la red
        network = cls()
        sender_node = None

        from classes.base import Node

        # Crear y añadir nodos
        for node_id, node_info in data['nodes'].items():
            node_id = int(node_id)
            position = tuple(node_info['position']) if 'position' in node_info else None

            # Crear instancia de Node con posición
            node = Node(node_id, network, position=position)
            network.add_node(node)

            # Identificar el nodo sender para devolverlo después
            if 'type' in node_info and node_info['type'] == 'sender':
                node.is_sender = True
                sender_node = node

        if sender_node is None:
            raise ValueError("No se encontró un nodo de tipo 'sender' en el archivo YAML.")

        # Conectar los nodos
        for node_id, node_info in data['nodes'].items():
            node_id = int(node_id)
            neighbors = node_info['neighbors']
            for neighbor_id in neighbors:
                network.connect_nodes(node_id, neighbor_id)

        return network, sender_node

    # util methods to calculate latency on send considering positions
    def get_distance(self, node_id1, node_id2):
        """Calcula la distancia euclidiana entre dos nodos."""
        pos1 = self.nodes[node_id1].position
        pos2 = self.nodes[node_id2].position
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

    def get_latency(self, node_id1, node_id2, propagation_speed=3e8):
        """Calcula la latencia de propagación entre dos nodos."""
        distance = self.get_distance(node_id1, node_id2)
        return distance / propagation_speed  # Latencia en segundos

    def __str__(self) -> str:
        result = ["\nNetwork Topology:"]
        for node_id, neighbors in self.connections.items():
            result.append(f"Node {node_id} -> Neighbors: {neighbors}")
        return "\n".join(result)

    def get_dynamic_changes_by_episode(self, start_time, end_time):
        """
        Filtra los cambios dinámicos que ocurrieron dentro de un rango de tiempo específico.

        Args:
            start_time (int): Tiempo de inicio del intervalo.
            end_time (int): Tiempo de fin del intervalo.

        Returns:
            list: Lista de tiempos en los que ocurrieron cambios dinámicos dentro del intervalo dado.
        """
        return [cambio for cambio in self.dynamic_change_events if start_time <= cambio <= end_time]
