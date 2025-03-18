import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import time
from matplotlib.animation import FuncAnimation

# Cargar datos de la simulación
def load_simulation_data(simulation_file):
    with open(simulation_file, 'r') as file:
        return json.load(file)

# Cargar topología de la red desde el archivo de simulación
def load_topology_from_simulation(simulation_data):
    topology_file = simulation_data['parameters']['topology_file']
    with open(topology_file, 'r') as file:
        return yaml.safe_load(file)

# Inicializar gráfico 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Función para inicializar la animación
def init():
    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Network Simulation Visualization")

# Función para actualizar la animación
def update(frame, nodes, edges, routes, node_functions, algorithm_text):
    ax.clear()
    init()
    
    # Dibujar nodos con IDs y funciones asignadas
    for node_id, data in nodes.items():
        x, y, z = data['position']
        color = 'gray' if data.get('inactive', False) else 'blue'
        ax.scatter(x, y, z, color=color, s=100)
        ax.text(x, y, z, f"{node_id}\n{node_functions.get(node_id, 'N/A')}", color='black', fontsize=8, ha='center')

    # Dibujar conexiones
    for (node1, node2) in edges:
        x1, y1, z1 = nodes[node1]['position']
        x2, y2, z2 = nodes[node2]['position']
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', linestyle='dashed')

    # Dibujar la ruta del paquete actual
    if frame < len(routes):
        from_node, to_node, function, status, algorithm = routes[frame]
        x1, y1, z1 = nodes[from_node]['position']
        x2, y2, z2 = nodes[to_node]['position']
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='red', linewidth=2)

        # Mostrar información de la simulación en pantalla
        text_str = (f"Routing Packet {frame+1}/{len(routes)}\n"
                    f"Algorithm: {algorithm}\n"
                    f"Total Episodes: {len(routes)}\n"
                    f"{algorithm_text}")

        ax.text2D(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10, ha='left', va='top')

# Función principal para visualizar la simulación
def visualize_simulation(simulation_file):
    data = load_simulation_data(simulation_file)
    topology = load_topology_from_simulation(data)
    
    # Extraer nodos y conexiones de la topología
    nodes = {}
    edges = []
    for node_id, info in topology['nodes'].items():
        nodes[int(node_id)] = {
            'position': tuple(info['position']),
            'neighbors': info.get('neighbors', []),
            'inactive': False
        }
        for neighbor in info.get('neighbors', []):
            edges.append((int(node_id), neighbor))
    
    # Obtener rutas de los paquetes en la simulación y funciones de los nodos
    routes = []
    node_functions = {}
    all_algorithms = data['parameters']['algorithms']
    algorithm_text = ""

    for algorithm in all_algorithms:
        episodes = data[algorithm]['episodes']
        for episode in episodes:
            for hop in episode['route']:
                from_node = hop['from']
                to_node = hop['to']
                function = hop.get('function', 'N/A')
                status = hop.get('node_status', 'active')
                routes.append((from_node, to_node, function, status, algorithm))
                node_functions[to_node] = function
    
    algorithm_text += (f"Max Hops: {data['parameters']['max_hops']}\n"
                        f"Mean Interval: {data['parameters']['mean_interval_ms']} ms\n"
                        f"Reconnect Interval: {data['parameters']['reconnect_interval_ms']} ms\n"
                        f"Topology File: {data['parameters']['topology_file']}\n"
                        f"Functions Sequence: {', '.join(data['parameters']['functions_sequence'])}\n"
                        f"Disconnect Probability: {data['parameters']['disconnect_probability']}\n\n")

    # Acelerar la animación
    ani = FuncAnimation(fig, update, frames=len(routes), fargs=(nodes, edges, routes, node_functions, algorithm_text), interval=100)
    plt.show()

# Ejecutar visualización con archivo de prueba
visualize_simulation('../../resources/results/single-run/simulation_1.json')
