import numpy as np
from os import pardir
import os
import time
import numpy as np
import yaml
from abc import ABC, abstractmethod
from visualization import animate_network, generate_heat_map, print_q_table
from tabulate import tabulate
import math
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

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
        # Calculate initial latency for this hop
        latency = self.get_latency(from_node_id, to_node_id)

        # Initialize the packet log for the episode if it doesn't exist
        if "episode_number" in packet:
            if packet["episode_number"] not in self.packet_log:
                self.packet_log[packet["episode_number"]] = []

            # Log the packet with latency and default delivery status
            self.packet_log[packet["episode_number"]].append({
                'from': from_node_id,
                'to': to_node_id,
                'packet': packet,
                'is_delivered': False,  # Default to False
                'latency': latency
            })

        # Validate and send
        hops = packet["hops"]
        max_hops = packet["max_hops"]

        if to_node_id in self.connections.get(from_node_id, []) and \
            from_node_id in self.active_nodes and \
            to_node_id in self.active_nodes and \
            hops < max_hops:

            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {hops} of {max_hops} max hops")

            time.sleep(latency)

            # Update delivery status directly in the log
            episode_number = packet["episode_number"]
            self.packet_log[episode_number][-1]['is_delivered'] = True

            self.nodes[to_node_id].application.receive_packet(packet)
        elif hops >= max_hops:
            print(f"[Network] Packet from Node {from_node_id} was lost. Max hops reached.")
        else:
            print(f"[Network] Failed to send packet: Node {to_node_id} is not reachable from Node {from_node_id}")

    def send(self, from_node_id, to_node_id, packet):
        # Initialize the packet log for the episode if it doesn't exist
        if packet.episode_number not in self.packet_log:
            self.packet_log[packet.episode_number] = []

        # Calculate initial latency for this hop
        latency = self.get_latency(from_node_id, to_node_id)

        # Log the packet with latency and default delivery status
        self.packet_log[packet.episode_number].append({
            'from': from_node_id,
            'to': to_node_id,
            'packet': packet,
            'is_delivered': False,  # Default to False
            'latency': latency
        })

        # Validate and send
        if to_node_id in self.connections.get(from_node_id, []) and \
            from_node_id in self.active_nodes and \
            to_node_id in self.active_nodes and \
            packet.hops < packet.max_hops:

            print(f"[Network] Sending packet from Node {from_node_id} to Node {to_node_id} with latency {latency:.6f} seconds")
            print(f"[Network] Packet hops: {packet.hops} of {packet.max_hops} max hops")

            time.sleep(latency)

            # Update delivery status directly in the log
            self.packet_log[packet.episode_number][-1]['is_delivered'] = True

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
    def __init__(self, network, sender_node, episodes_number):
        self.sender_node = sender_node
        self.network = network
        self.episodes_number = episodes_number
        self.dynamic_change_episodes = self.generate_dynamic_change_episodes(episodes_number, 5) # episodios en los que va a haber cambios
        self.metrics = {
            "Q_ROUTING": {
                "latencia_promedio": [],
                "consistencia_latencia": [],
                "tasa_exito": [],
                "latencia_pre_cambio": [],
                "latencia_post_cambio": [],
                "tasa_exito_pre_cambio": [],
                "tasa_exito_post_cambio": [],
            },
            "DIJKSTRA": {
                "latencia_promedio": [],
                "consistencia_latencia": [],
                "tasa_exito": [],
                "latencia_pre_cambio": [],
                "latencia_post_cambio": [],
                "tasa_exito_pre_cambio": [],
                "tasa_exito_post_cambio": [],
            },
            "BELLMAN_FORD": {
                "latencia_promedio": [],
                "consistencia_latencia": [],
                "tasa_exito": [],
                "latencia_pre_cambio": [],
                "latencia_post_cambio": [],
                "tasa_exito_pre_cambio": [],
                "tasa_exito_post_cambio": [],
            },
        }

    def start(self, algorithm_enum):
        algorithm = algorithm_enum.name
        print(f"Algorithm is: {algorithm}")

        # Generar episodios con cambios dinámicos antes de iniciar la simulación
        self.dynamic_change_episodes = self.generate_dynamic_change_episodes(self.episodes_number, mean_interval=5)
        print(f"[DEBUG] Episodes with Dynamic Changes: {self.dynamic_change_episodes}")

        for episode_number in range(1, self.episodes_number + 1):
            print(f'\n\n=== Starting episode #{episode_number} ===\n')

            # Aplicar cambios dinámicos si este episodio está marcado
            if episode_number in self.dynamic_change_episodes:
                print(f"\n--- Dynamic Network Change in episode {episode_number} ---\n")

                # Registrar métricas pre-cambio
                print(f"[DEBUG] Registering pre-change metrics at episode #{episode_number}")
                pre_latencia = self.metrics[algorithm]["latencia_promedio"][-1] if self.metrics[algorithm]["latencia_promedio"] else None
                pre_tasa_exito = self.metrics[algorithm]["tasa_exito"][-1] if self.metrics[algorithm]["tasa_exito"] else None

                self.metrics[algorithm].setdefault("latencia_pre_cambio", []).append(pre_latencia)
                self.metrics[algorithm].setdefault("tasa_exito_pre_cambio", []).append(pre_tasa_exito)

                print(f"[DEBUG] Saved pre-change Latency: {self.metrics[algorithm]['latencia_pre_cambio']}")
                print(f"[DEBUG] Saved pre-change Success Rate: {self.metrics[algorithm]['tasa_exito_pre_cambio']}")

                # Aplicar cambio dinámico
                self.dynamic_network_change()
                print(f"[DEBUG] Network change applied at episode {episode_number}")

            # Inicializar métricas del episodio
            latencias = []
            entregados = 0
            total_paquetes = 0

            # Print info for logging
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

            # Ejecutar la simulación del episodio
            self.sender_node.start_episode(episode_number)

            # Analizar resultados de Q-Routing si corresponde
            from applications.q_routing import QRoutingApplication
            if isinstance(self.sender_node, QRoutingApplication):
                q_tables = []
                for node in self.network.nodes.values():
                    print_q_table(node.application)
                    q_tables.append(node.application.q_table)
                generate_heat_map(q_tables, episode_number)

            # Recolectar datos del episodio
            print(f"Packet Log for Episode #{episode_number}: {self.network.packet_log.get(episode_number, [])}")
            for log in self.network.packet_log.get(episode_number, []):
                if log['is_delivered']:
                    latencias.append(log['latency'])
                    entregados += 1
                total_paquetes += 1

            # Calcular métricas
            latencia_promedio = sum(latencias) / len(latencias) if latencias else None
            consistencia_latencia = np.std(latencias) if len(latencias) > 1 else None
            tasa_exito = (entregados / total_paquetes * 100) if total_paquetes > 0 else 0

            print(f"\nEpisode #{episode_number} Metrics:")
            print(f"  Latencia Promedio: {latencia_promedio}")
            print(f"  Consistencia Latencia: {consistencia_latencia}")
            print(f"  Tasa de Éxito: {tasa_exito}%")

            # Guardar métricas en el algoritmo correspondiente
            self.metrics[algorithm]["latencia_promedio"].append(latencia_promedio)
            self.metrics[algorithm]["consistencia_latencia"].append(consistencia_latencia)
            self.metrics[algorithm]["tasa_exito"].append(tasa_exito)

            # Guardar métricas post-cambio en el siguiente episodio
            if episode_number - 1 in self.dynamic_change_episodes:
                print(f"\n--- Recording Post-Change Metrics for episode {episode_number} ---\n")
                self.metrics[algorithm].setdefault("latencia_post_cambio", []).append(latencia_promedio)
                self.metrics[algorithm].setdefault("tasa_exito_post_cambio", []).append(tasa_exito)
                print(f"[DEBUG] Saved post-change Latency: {self.metrics[algorithm]['latencia_post_cambio']}")
                print(f"[DEBUG] Saved post-change Success Rate: {self.metrics[algorithm]['tasa_exito_post_cambio']}")

        # Mostrar resultados finales de métricas
        print("\nFinal Metrics:")
        print(self.metrics[algorithm])
        self.save_results_to_excel()
        self.generar_graficos_desde_excel()

    def generate_dynamic_change_episodes(self, total_episodes, mean_interval):
        """
        Genera episodios con cambios dinámicos basados en una distribución exponencial.
        """
        current_episode = 0
        change_episodes = []
        
        print(f"[DEBUG] Generating dynamic changes for {total_episodes} episodes with mean interval {mean_interval}")

        while current_episode < total_episodes:
            interval = np.random.exponential(mean_interval)
            interval_int = int(interval)
            
            print(f"[DEBUG] Generated interval: {interval} -> Rounded: {interval_int}")

            if interval_int == 0:  
                interval_int = 1  # Asegurar que no haya intervalos de 0
            
            current_episode += interval_int

            if current_episode <= total_episodes:
                change_episodes.append(current_episode)
                print(f"[DEBUG] Added episode {current_episode} to change_episodes")

        print(f"[DEBUG] Final Episodes with Dynamic Changes: {change_episodes}")
        return change_episodes

    def dynamic_network_change(self):
        print("ZZZZAP")
        for node in self.network.nodes.values():
            node.update_status()

    def save_results_to_excel(self, filename="../results/resultados_simulacion.xlsx"):
        """
        Guarda los datos de la simulación en un archivo Excel, con una hoja por algoritmo.
        """
        with pd.ExcelWriter(filename) as writer:
            for algorithm, metrics in self.metrics.items():
                num_episodes = self.episodes_number  # Número total de episodios

                def pad_list(lst, length):
                    """Asegura que la lista tenga el mismo tamaño que el número de episodios, rellenando con None."""
                    return lst + [None] * (length - len(lst))

                data = {
                    "Episodio": list(range(1, num_episodes + 1)),
                    "Latencia Promedio": pad_list(metrics["latencia_promedio"], num_episodes),
                    "Consistencia Latencia": pad_list(metrics["consistencia_latencia"], num_episodes),
                    "Tasa de Éxito": pad_list(metrics["tasa_exito"], num_episodes),
                    "Cambio Dinámico": ["Sí" if ep in self.dynamic_change_episodes else "No" for ep in range(1, num_episodes + 1)],
                    "Latencia Pre-Cambio": pad_list(metrics.get("latencia_pre_cambio", []), num_episodes),
                    "Latencia Post-Cambio": pad_list(metrics.get("latencia_post_cambio", []), num_episodes),
                    "Tasa de Éxito Pre-Cambio": pad_list(metrics.get("tasa_exito_pre_cambio", []), num_episodes),
                    "Tasa de Éxito Post-Cambio": pad_list(metrics.get("tasa_exito_post_cambio", []), num_episodes),
                }

                df = pd.DataFrame(data)
                sheet_name = algorithm[:31]  # Excel limita los nombres de hojas a 31 caracteres
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"\n📁 Resultados guardados en {filename}.")

    def generar_graficos_desde_excel(self, filename="../results/resultados_simulacion.xlsx"):
        """
        Genera gráficos para cada algoritmo usando matplotlib y guarda las imágenes en ../results/.
        """
        os.makedirs("../results", exist_ok=True)  # Crear la carpeta si no existe

        # Cargar el archivo Excel
        xls = pd.ExcelFile(filename)

        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # Crear figura para gráficos
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))  # 2 filas, 3 columnas de gráficos
            fig.suptitle(f"Análisis del Algoritmo: {sheet_name}\nTotal Episodios: {len(df)} - Cambios Dinámicos: {df['Cambio Dinámico'].tolist().count('Sí')}", fontsize=14, fontweight="bold")

            # Identificar episodios con cambios dinámicos
            cambios = df[df["Cambio Dinámico"] == "Sí"]["Episodio"]

            # 📊 Gráfico 1: Latencia Promedio vs Episodio
            axs[0, 0].plot(df["Episodio"], df["Latencia Promedio"], label="Latencia Promedio", marker="o", color="blue")
            for cambio in cambios:
                axs[0, 0].axvline(x=cambio, color="red", linestyle="--", alpha=0.5)
            axs[0, 0].set_title("Latencia Promedio vs Episodio")
            axs[0, 0].set_xlabel("Episodio")
            axs[0, 0].set_ylabel("Latencia")
            axs[0, 0].grid()
            axs[0, 0].legend()

            # 📊 Gráfico 2: Tasa de Éxito vs Episodio
            axs[0, 1].plot(df["Episodio"], df["Tasa de Éxito"], label="Tasa de Éxito (%)", marker="s", color="green")
            axs[0, 1].set_title("Tasa de Éxito vs Episodio")
            axs[0, 1].set_xlabel("Episodio")
            axs[0, 1].set_ylabel("Tasa de Éxito (%)")
            axs[0, 1].grid()
            axs[0, 1].legend()

            # 📊 Gráfico 3: Consistencia en la Latencia vs Episodio
            axs[0, 2].plot(df["Episodio"], df["Consistencia Latencia"], label="Consistencia Latencia", marker="^", color="purple")
            axs[0, 2].set_title("Consistencia en la Latencia vs Episodio")
            axs[0, 2].set_xlabel("Episodio")
            axs[0, 2].set_ylabel("Desviación Estándar")
            axs[0, 2].grid()
            axs[0, 2].legend()

            # 📊 Guardar la imagen en ../results/
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"../results/{sheet_name}.png")
            plt.close()

        print("\n📊 Gráficos generados en '../results/'.")
