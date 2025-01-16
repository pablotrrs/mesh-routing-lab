import time
import numpy as np
import yaml
from abc import ABC, abstractmethod
from visualization import animate_network, generate_heat_map, print_q_table
from tabulate import tabulate
import math

class Application(ABC):
    def __init__(self, node):
        self.node = node

    @abstractmethod
    def start_episode(self, episode_number):
        pass

    @abstractmethod
    def receive_packet(self, packet):
        pass

class Node:
    def __init__(self, node_id, network, position):
        self.node_id = node_id
        self.network = network
        self.application = None
        self.is_sender = False
        self.lifetime = None
        self.reconnect_time = None
        self.status = None
        self.position = position

    def install_application(self, application_class):
        self.application = application_class(self)
        print(f"[Node_ID={self.node_id}] Installed {self.application.__class__.__name__}")

        # Set attributes if the application is not SenderQRoutingApplication
        if not self.is_sender:
            self.lifetime = np.random.exponential(scale=2)
            self.reconnect_time = np.random.exponential(scale=2)
            self.status = True
        else:
            self.lifetime = None
            self.reconnect_time = None
            self.status = None

    def start_episode(self, episode_number):
        if self.application:
            self.application.start_episode(episode_number)
        else:
            raise RuntimeError(f"[Node_ID={self.node_id}] No application installed")

    def get_assigned_function(self):
        if self.application and hasattr(self.application, 'get_assigned_function'):
            return self.application.get_assigned_function()
        return None

    def update_status(self):
        if not self.is_sender:
            if self.status:
                self.lifetime -= 1

                if self.lifetime <= 0:
                    self.status = False
                    self.reconnect_time = np.random.exponential(scale=2)
            else:
                self.reconnect_time -= 1

                if self.reconnect_time <= 0:
                    self.status = True
                    self.lifetime = np.random.exponential(scale=2)

        return self.status

    def __str__(self) -> str:
        return f"Node(id={self.node_id}, neighbors={self.network.get_neighbors(self.node_id)})"

    def __repr__(self) -> str:
        return self.__str__()

class Network:
    def __init__(self):
        self.nodes = {}  # {node_id: Node}
        self.connections = {}  # {node_id: [neighbors]}
        self.active_nodes = set()
        self.packet_log = {}  # List to store packet logs

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

    def send_dict(self, from_node_id, to_node_id, packet):

        # Initialize the packet log for the episode if it doesn't exist
        if "episode_number" in packet:
            if packet["episode_number"] not in self.packet_log:
                self.packet_log[packet["episode_number"]] = []

            # Log the packet
            self.packet_log[packet["episode_number"]].append({
                'from': from_node_id,
                'to': to_node_id,
                'packet': packet
            })

        hops = packet["hops"]
        max_hops = packet["max_hops"]

        # validate and send
        if to_node_id in self.connections.get(from_node_id, []) and \
            from_node_id in self.active_nodes and \
            to_node_id in self.active_nodes and \
            packet["hops"] < packet["max_hops"]:

            latency = self.get_latency(from_node_id, to_node_id)
            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {hops} of {max_hops} max hops")

            time.sleep(latency)

            self.nodes[to_node_id].application.receive_packet(packet)
        elif hops >= max_hops:
            print(f"[Network] Packet from Node {from_node_id} was lost. Max hops reached.")
        else:
            print(f"[Network] Failed to send packet: Node {to_node_id} is not reachable from Node {from_node_id}")

    def send(self, from_node_id, to_node_id, packet):

        # Initialize the packet log for the episode if it doesn't exist
        if packet.episode_number not in self.packet_log:
            self.packet_log[packet.episode_number] = []

        # Log the packet
        self.packet_log[packet.episode_number].append({
            'from': from_node_id,
            'to': to_node_id,
            'packet': packet
        })

        # validate and send
        if to_node_id in self.connections.get(from_node_id, []) and \
            from_node_id in self.active_nodes and \
            to_node_id in self.active_nodes and \
            packet.hops < packet.max_hops:

            latency = self.get_latency(from_node_id, to_node_id)
            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {packet.hops} of {packet.max_hops} max hops")

            time.sleep(latency)

            self.nodes[to_node_id].application.receive_packet(packet)
        elif packet.hops >= packet.max_hops:
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

class Simulation:
    def __init__(self, network, sender_node):
        self.network = network
        self.sender_node = sender_node

    def start(self, episodes_number):
        # TODO: dynamic_network_change() que cambie los nodos de network

        for episode_number in range(1, episodes_number + 1):
            print(f'\n\n=== Starting episode #{episode_number} ===\n')

            # print info for logging
            node_info = []
            for node in self.network.nodes.values():
                if not node.is_sender:
                    node_info.append([
                        node.node_id,
                        node.status,
                        node.lifetime,
                        node.reconnect_time
                    ])

            headers = ["Node ID", "Connected", "Lifetime", "Reconnect Time"]
            print(tabulate(node_info, headers=headers, tablefmt="grid"))
            print("\n")

            # perform the simulation
            self.sender_node.start_episode(episode_number)

            # animate_network(
            #     episode_number, self.network.packet_log[episode_number], list(self.network.nodes.keys()),
            #     self.network.connections, self.network
            # )

            # print episode results

            # print q table if algorithm is q routing
            from applications.q_routing import QRoutingApplication
            if isinstance(self.sender_node, QRoutingApplication):
                q_tables = []

                for node in self.network.nodes.values():
                    print_q_table(node.application)
                    q_tables.append(node.application.q_table)
                    node.update_status()

                generate_heat_map(q_tables, episode_number)

            # TODO: generar csv con métricas de la red
