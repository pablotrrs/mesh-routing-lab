import numpy as np
import time
import yaml
import math
import threading
import sys
sys.path.insert(1, 'src/classes')

class Network:
    def __init__(self):
        self.nodes = {}  # {node_id: Node}
        self.connections = {}  # {node_id: [neighbors]}
        self.active_nodes = set()
        self.packet_log = {}  # List to store packet logs
        self.dynamic_change_events = []  # Tiempos (ms) de cambios dinámicos
        self.simulation_clock = 0  # Reloj central en ms
        self.running = False  # Control del hilo
        self.lock = threading.Lock()  # Para acceso seguro al reloj y nodos
        self.mean_interval_ms = None
        self.reconnect_interval_ms = None
        self.max_hops = None

    def set_max_hops(self, max_hops):
        self.max_hops = max_hops

    def set_mean_interval_ms(self, mean_interval_ms):
        self.mean_interval_ms = mean_interval_ms

    def set_reconnect_interval_ms(self, reconnect_interval_ms):
        self.reconnect_interval_ms = reconnect_interval_ms

    def set_simulation_clock(self, clock_reference):
        """Sincroniza el reloj central con el de la simulación."""
        self.simulation_clock = clock_reference

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
        current_time = self.simulation_clock.get_current_time()
        print(f"[Network] clock starts: {current_time}")
        threading.Thread(target=self._monitor_dynamic_changes, daemon=True).start()

    def stop_dynamic_changes(self):
        """Detiene el hilo de cambios dinámicos."""
        self.running = False

    def _monitor_dynamic_changes(self):
        """Monitorea el reloj central y aplica cambios dinámicos automáticamente."""
        next_event_time = self.simulation_clock.get_current_time() + self.generate_next_dynamic_change()
        while self.running:
            current_time = self.simulation_clock.get_current_time()
            # print(f"[Network] clock tics: {current_time}")
            with self.lock:
                if current_time >= next_event_time:
                    print("\n⚡ZZZAP⚡")
                    # print(f"[Network] Dynamic Change triggered at {current_time} ms")
                    self.trigger_dynamic_change()
                    self.dynamic_change_events.append(current_time)
                    next_event_time = current_time + self.generate_next_dynamic_change()
            time.sleep(0.01)  # Chequeos periódicos (10 ms)

    def trigger_dynamic_change(self):
        """
        Aplica la lógica de cambios dinámicos en la red.
        Ejemplo: desconexión aleatoria de nodos.
        """
        for node_id in list(self.active_nodes):
            if np.random.rand() < 0.3:  # 30% de probabilidad de desconexión
                self.nodes[node_id].status = False
                self.nodes[node_id].reconnect_time = np.random.exponential(scale=self.reconnect_interval_ms)
                print(f"[Network] Node {node_id} disconnected.")

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

    def get_nodes(self):
        """
        Returns a list of all node IDs in the network.
        """
        return list(self.nodes.keys())

    def send_dict(self, from_node_id, to_node_id, packet, lost_packet=False):
        # Si el paquete está marcado como perdido, registrarlo y salir
        if lost_packet:
            print(f"[Network] Packet marked as lost on demand.")
            if packet and "episode_number" in packet:
                if packet["episode_number"] not in self.packet_log:
                    self.packet_log[packet["episode_number"]] = []

                # Registrar el paquete perdido
                self.packet_log[packet["episode_number"]].append({
                    'from': from_node_id,
                    'to': to_node_id,
                    'packet': packet,
                    'is_delivered': False,  # Marcado como no entregado
                    'latency': None  # Latencia indefinida para paquetes perdidos
                })
            return

        # Validar y calcular la latencia solo si no se marca como perdido
        latency = self.get_latency(from_node_id, to_node_id)

        if "episode_number" in packet:
            if packet["episode_number"] not in self.packet_log:
                self.packet_log[packet["episode_number"]] = []

            # Registrar el paquete con estado por defecto
            self.packet_log[packet["episode_number"]].append({
                'from': from_node_id,
                'to': to_node_id,
                'packet': packet,
                'is_delivered': False,  # Estado inicial
                'latency': latency
            })

        # Validar y enviar
        hops = packet["hops"]

        if to_node_id in self.connections.get(from_node_id, []) and \
            from_node_id in self.active_nodes and \
            to_node_id in self.active_nodes and \
            hops < self.max_hops:

            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {hops} of {self.max_hops} max hops")

            time.sleep(latency)

            # Actualizar el estado de entrega directamente en el registro
            episode_number = packet["episode_number"]
            self.packet_log[episode_number][-1]['is_delivered'] = True

            self.nodes[to_node_id].application.receive_packet(packet)
        elif hops >= self.max_hops:
            print(f"[Network] Packet from Node {from_node_id} was lost. Max hops reached.")
        else:
            print(f"[Network] Failed to send packet: Node {to_node_id} is not reachable from Node {from_node_id}")

    def send(self, from_node_id, to_node_id, packet, lost_packet=False):
        # Si el paquete está marcado como perdido, registrarlo y salir
        if lost_packet:
            print(f"[Network] Packet marked as lost on demand.")
            if packet and "episode_number" in packet:
                if packet["episode_number"] not in self.packet_log:
                    self.packet_log[packet["episode_number"]] = []

                # Registrar el paquete perdido
                self.packet_log[packet["episode_number"]].append({
                    'from': from_node_id,
                    'to': to_node_id,
                    'packet': packet,
                    'is_delivered': False,  # Marcado como no entregado
                    'latency': None  # Latencia indefinida para paquetes perdidos
                })
            return

        # Validar y calcular la latencia solo si no se marca como perdido
        latency = self.get_latency(from_node_id, to_node_id)

        if packet.episode_number not in self.packet_log:
            self.packet_log[packet.episode_number] = []

        # Registrar el paquete con estado por defecto
        self.packet_log[packet.episode_number].append({
            'from': from_node_id,
            'to': to_node_id,
            'packet': packet,
            'is_delivered': False,  # Estado inicial
            'latency': latency
        })

        if to_node_id in self.connections.get(from_node_id, []) and \
            from_node_id in self.active_nodes and \
            to_node_id in self.active_nodes and \
            packet.hops < self.max_hops:

            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {packet.hops} of {self.max_hops} max hops")

            time.sleep(latency)

            # Actualizar el estado de entrega directamente en el registro
            self.packet_log[packet.episode_number][-1]['is_delivered'] = True

            self.nodes[to_node_id].application.receive_packet(packet)
        elif packet.hops >= self.max_hops:
            print(f"[Network] Packet from Node {from_node_id} was lost. Max hops reached.")
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

        from base import Node

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

    def get_dynamic_changes_by_episode(self, episode_times):
        """
        Asocia los cambios dinámicos en la red con los episodios en los que ocurrieron.

        Args:
            episode_times (dict): Diccionario con el tiempo de inicio y fin de cada episodio.

        Returns:
            dict: Un diccionario donde las claves son los números de episodio y los valores son listas
                  de tiempos en los que ocurrieron cambios dinámicos dentro de ese episodio.
        """
        cambios_por_episodio = {episode: [] for episode in episode_times}

        for cambio in self.dynamic_change_events:
            for episode, times in episode_times.items():
                start_time = times["start_time"]
                end_time = times["end_time"]

                # Si el cambio ocurrió dentro del intervalo del episodio, lo agregamos
                if start_time <= cambio <= end_time:
                    cambios_por_episodio[episode].append(cambio)

        return cambios_por_episodio
