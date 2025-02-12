import numpy as np
import time
import yaml
import math
import threading

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
        self.mean_interval_ms = mean_interval_ms * 1000

    def set_reconnect_interval_ms(self, reconnect_interval_ms):
        self.reconnect_interval_ms = reconnect_interval_ms

    def set_disconnect_probability(self, disconnect_probability):
        self.disconnect_probability = disconnect_probability

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

                current_time = self.simulation_clock.get_current_time()
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

                current_time = self.simulation_clock.get_current_time()
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

    def send_dict(self, from_node_id, to_node_id, packet_dict, lost_packet=False, episode_success=False):
        """
        Registra el envío de un paquete en el packet_log y maneja la lógica de transmisión 
        utilizando un diccionario en lugar de una instancia de la clase Packet.

        Args:
            from_node_id (int): Nodo de origen.
            to_node_id (int): Nodo de destino (puede ser None si el paquete ha llegado al nodo original).
            packet_dict (dict): Paquete representado como un diccionario.
            lost_packet (bool): Indica si el paquete debe marcarse como perdido.
            episode_success (bool): Indica si el episodio fue exitoso.
        """
        episode_number = packet_dict.get("episode_number")

        # **Siempre inicializar packet_log**
        if episode_number not in self.packet_log:
            self.packet_log[episode_number] = {
                "episode_success": None,  # Se actualizará al final del episodio
                "episode_duration": None,
                "route": []
            }

        # **Caso: Paquete perdido**
        if lost_packet:
            print(f"[Network] Packet from {from_node_id} to {to_node_id} lost.")
            self.packet_log[episode_number]["route"].append({
                "from": from_node_id,
                "to": to_node_id if to_node_id is not None else "N/A",
                "function": "N/A",
                "node_status": "inactive",
                "latency": 0,
                "packet_type": packet_dict["type"].value
            })
            return  # No seguir procesando

        # **Caso: Fin del episodio (llegó al nodo original)**
        if to_node_id is None:
            print(f"[Network] Packet reached final destination at Node {from_node_id}. Marking episode completion.")
            to_node_id = "N/A"  # Marcar destino como N/A
            self.packet_log[episode_number]["episode_success"] = episode_success
            return  # No seguir procesando

        # **Calcular latencia**
        latency = self.get_latency(from_node_id, to_node_id) if to_node_id != "N/A" else 0

        # **Registrar información del hop en packet_log**
        self.packet_log[episode_number]["route"].append({
            "from": from_node_id,
            "to": to_node_id,
            "function": self.nodes[from_node_id].get_assigned_function().value if self.nodes[from_node_id].get_assigned_function() else None,
            "node_status": "active" if from_node_id in self.active_nodes else "inactive",
            "latency": latency,
            "packet_type": packet_dict["type"].value
        })

        # **Validar si el nodo destino es alcanzable**
        if to_node_id in self.connections.get(from_node_id, []) and \
                from_node_id in self.active_nodes and \
                to_node_id in self.active_nodes:

            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {packet_dict.get('hops', 0)} of {self.max_hops} max hops")

            time.sleep(latency)  # Simulación de la latencia
            self.nodes[to_node_id].application.receive_packet(packet_dict)
        else:
            print(f"[Network] Failed to send packet: Node {to_node_id} is not reachable from Node {from_node_id}")

    def send(self, from_node_id, to_node_id, packet, lost_packet=False, episode_success=False):
        """
        Envía un paquete entre nodos de la red, registrando su trazabilidad y manejando errores de conectividad.

        Args:
            from_node_id (int): Nodo de origen.
            to_node_id (int): Nodo de destino (puede ser None si el paquete ha llegado al nodo original).
            packet (Packet): Instancia de la clase Packet.
            lost_packet (bool): Indica si el paquete se debe marcar como perdido.
            episode_success (bool): Indica si el episodio fue exitoso.
        """
        episode_number = packet.episode_number

        # **Siempre inicializar packet_log**
        if episode_number not in self.packet_log:
            self.packet_log[episode_number] = {
                "episode_success": None,
                "episode_duration": None,
                "route": []
            }

        # **Caso: Paquete perdido**
        if lost_packet:
            print(f"[Network] Packet from {from_node_id} to {to_node_id} lost.")
            self.packet_log[episode_number]["route"].append({
                "from": from_node_id,
                "to": to_node_id if to_node_id is not None else "N/A",
                "function": "N/A",
                "node_status": "inactive",
                "latency": 0,
                "packet_type": packet.type.value
            })
            return  # No seguir procesando

        # **Caso: Fin del episodio (llegó al nodo original)**
        if to_node_id is None:
            print(f"[Network] Packet reached final destination at Node {from_node_id}. Marking episode completion.")
            to_node_id = "N/A"  # Marcar destino como N/A
            self.packet_log[episode_number]["episode_success"] = episode_success
            return  # No seguir procesando

        # **Calcular latencia**
        latency = self.get_latency(from_node_id, to_node_id) if to_node_id != "N/A" else 0

        # **Registrar información del hop en packet_log**
        self.packet_log[episode_number]["route"].append({
            "from": from_node_id,
            "to": to_node_id,
            "function": self.nodes[from_node_id].get_assigned_function().value if self.nodes[from_node_id].get_assigned_function() else None,
            "node_status": "active" if from_node_id in self.active_nodes else "inactive",
            "latency": latency,
            "packet_type": packet.type.value
        })

        # **Validar si el nodo destino es alcanzable**
        if to_node_id in self.connections.get(from_node_id, []) and \
                from_node_id in self.active_nodes and \
                to_node_id in self.active_nodes:

            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {packet.hops} of {self.max_hops} max hops")

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
