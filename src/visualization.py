import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import os
from matplotlib.animation import FuncAnimation

output_folder = 'simulation_images'
os.makedirs(output_folder, exist_ok=True)

# Define positions for the nodes in the network
# positions = {
#     'tx': (0 * 2, 3 * 2),
#     'rx': (7 * 2, 3 * 2)
# }

# for i in range(36):
#     row, col = divmod(i, 6)
#     positions[f'i{i}'] = ((col + 1) * 2, row * 2)

# def animate_network(path, processed_functions, functions_sequence, episode, nodes, neighbors, node_status, node_functions):
#     """Plots the network graph showing the path, applied functions, and missing functions, with progressive colors and active node highlight."""
#     plt.clf()

#     G = nx.DiGraph()
#     G.add_nodes_from(nodes)
#     G.add_edges_from([(node1, node2) for node1 in neighbors for node2 in neighbors[node1]])

#     # Base color for nodes (green if alive, gray if dead)
#     node_colors = ['green' if node_status.get(node, True) else 'gray' for node in nodes]
#     node_labels = {node: f'{node}\nFunc: {node_functions.get(node, "")}' for node in nodes}

#     nx.draw(G, pos=positions, labels=node_labels, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False)

#     # Create a colormap to vary the color of edges as the agent moves
#     cmap = cm.get_cmap('plasma')

#     total_hops_in_path = len(path) - 1
#     min_color = 0.4
#     max_color = 1.0

#     applied_functions = processed_functions
#     missing_functions = [func for func in functions_sequence if func not in processed_functions]

#     applied_text = "Applied functions: " + ", ".join(applied_functions) if applied_functions else "No function applied"
#     missing_text = "Missing functions: " + ", ".join(missing_functions) if missing_functions else "All functions applied"

#     # Create a legend with the applied and missing functions
#     handles = [
#         plt.Line2D([0], [0], color='white', label=applied_text),
#         plt.Line2D([0], [0], color='white', label=missing_text)
#     ]
#     plt.legend(handles=handles, loc='upper right', fontsize=8)

#     # Draw each hop with progressively lighter colors
#     for i in range(total_hops_in_path):
#         edges_in_path = [(path[i], path[i + 1])]

#         # Compute the color for the current hop
#         color_index = min_color + (i / total_hops_in_path) * (max_color - min_color)
#         color = cmap(color_index)

#         # Highlight the destination node of the hop (next_node)
#         destination_node = path[i + 1]  # Destination of the hop
#         node_colors = ['red' if node == destination_node else 'green' if node_status.get(node, True) else 'gray' for node in nodes]

#         nx.draw(G, pos=positions, labels=node_labels, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False)

#         nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color=[color], width=3, arrows=True)

#         plt.title(f'Episode {episode} - Path: {" -> ".join(path[:i + 2])}')
#         plt.pause(0.001)

#     plt.pause(0.0000001)

def animate_network(
    episode_number, packet_logs, processed_functions,
    functions_sequence, nodes, connections, active_nodes, positions,
    network
):
    """Plots the network graph showing the path, applied functions, and missing functions, with progressive colors and active node highlight."""
    G = nx.DiGraph()
    
    # Add nodes and edges to the graph
    for node in nodes:
        G.add_node(node)
    for node, neighbors in connections.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    
    # Calculate positions for the nodes using spring layout
    positions = nx.spring_layout(G)
    
    node_labels = {}
    node_colors = {}
    for node in nodes:
        assigned_function = network.nodes[node].get_assigned_function()
        if assigned_function is not None:
            node_labels[node] = f"{node} : {assigned_function.value}"
        else:
            node_labels[node] = f"{node}"
        
        # Set node color based on status
        if network.nodes[node].status:
            node_colors[node] = 'green'
        elif network.nodes[node].is_sender:
            node_colors[node] = 'blue'
        else:
            node_colors[node] = 'gray'

    dpi = 100  # Dots per inch
    fig, ax = plt.subplots(figsize=(800 / dpi, 800 / dpi), dpi=dpi)

    def update(frame):
        ax.clear()
        packet_log = packet_logs[frame]
        from_node = packet_log['from']
        to_node = packet_log['to']
        packet_type = packet_log['packet'].type.value
        path_color = "red"

        # Highlight the path
        edge_colors = []
        for edge in G.edges():
            if edge == (from_node, to_node):
                edge_colors.append(path_color)
            else:
                edge_colors.append('black')

        nx.draw_networkx_nodes(G, pos=positions, node_color=[node_colors[node] for node in G.nodes()], node_size=750, ax=ax)
        nx.draw_networkx_labels(G, pos=positions, labels=node_labels, font_size=8, font_weight='bold', ax=ax, font_color='white')
        nx.draw_networkx_edges(G, pos=positions, edge_color=edge_colors, width=4, arrows=True, ax=ax, arrowsize=20, connectionstyle='arc3,rad=0.1')

        ax.set_title(f"Network Visualization - Episode {episode_number}")

        # Add legend
        legend_text = f" Packet Path: {from_node} -> {to_node}\n Type: {packet_type}"
        ax.legend([legend_text], loc='upper right')

    ani = FuncAnimation(fig, update, frames=len(packet_logs), interval=1000, repeat=False)
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+0+0")  # Center the window

    plt.show()

def plot_q_tables(q_table, episode):
    cell_size = 30
    pixels_per_inch = 96

    for i in range(0, 36):
        node = f'i{i}'
        q_values = np.array(list(q_table[node].values()))
        N = q_values.shape[0] 
        figure_size_in_pixels = N * cell_size
        figure_size_in_inches = figure_size_in_pixels / pixels_per_inch
        fig, ax = plt.subplots(figsize=(figure_size_in_inches, figure_size_in_inches))
        cax = ax.matshow(q_values, cmap="RdYlGn", vmin=-100, vmax=100)
        fig.colorbar(cax, ax=ax)

        for (row, col), val in np.ndenumerate(q_values):
            ax.text(col, row, f'{val:.1f}', ha='center', va='center', fontsize=8, color='black')

        ax.set_title(f"Q-table: {node} - Episode {episode}", fontsize=10)
        ax.set_xticks(range(N))
        ax.set_xticklabels([f'{n}' for n in range(N)], rotation=90, fontsize=8)
        ax.set_yticks(range(N))
        ax.set_yticklabels([f'{n}' for n in range(N)], fontsize=8)
        ax.tick_params(axis='x', which='major', pad=10)
        ax.tick_params(axis='y', which='major', pad=10)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'{output_folder}/q_table_{node}_episode_{episode}.png')
        plt.close(fig)

def plot_network(path, nodes, neighbors, node_status, episode):
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(node1, node2) for node1 in neighbors for node2 in neighbors[node1]])

    node_colors = []
    for node in nodes:
        if node == 'tx' or node == 'rx':
            node_colors.append('green')
        elif node_status.get(node, True):
            node_colors.append('green')
        else:
            node_colors.append('gray')

    nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False)

    edges_in_path = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color='blue', width=3, arrows=True)

    plt.title(f'Episode {episode} - Path: {" -> ".join(path)}')

    plt.savefig(f'{output_folder}/network_episode_{episode}.png')