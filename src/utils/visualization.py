from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tabulate import tabulate
import networkx as nx
import os

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

def animate_network(episode_number, packet_logs, nodes, connections, network):
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

import matplotlib.pyplot as plt
import numpy as np
import os

output_folder = 'simulation_images'
os.makedirs(output_folder, exist_ok=True)

def generate_heat_map(q_tables, episode_number):
    q_table_data = []
    for q_table in q_tables:
        for state, actions in q_table.items():
            for action, q_value in actions.items():
                q_table_data.append((state, action, q_value))

    if not q_table_data:
        print(f"No Q-table data available for episode {episode_number}")
        return

    # Extract unique states and actions
    states = sorted(set(state for state, _, _ in q_table_data))
    actions = sorted(set(action for _, action, _ in q_table_data))

    # Create a matrix to hold Q-values
    q_matrix = np.zeros((len(states), len(actions)))

    for state, action, q_value in q_table_data:
        state_index = states.index(state)
        action_index = actions.index(action)
        q_matrix[state_index, action_index] = q_value

    fig, ax = plt.subplots()
    cax = ax.imshow(q_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add color bar
    fig.colorbar(cax)

    # Set axis labels
    ax.set_xticks(np.arange(len(actions)))
    ax.set_yticks(np.arange(len(states)))
    ax.set_xticklabels(actions)
    ax.set_yticklabels(states)

    # Annotate each cell with its value
    for i in range(len(states)):
        for j in range(len(actions)):
            ax.text(j, i, f'{q_matrix[i, j]:.2f}', ha='center', va='center', color='black')

    plt.xlabel('Actions (Next Node ID)')
    plt.ylabel('States (Node ID)')
    plt.title(f'Q-Table Heat Map - Episode {episode_number}')

    # Save the heat map as a .png file
    filename = os.path.join(output_folder, f'q_table_heat_map_episode_{episode_number}.png')
    plt.savefig(filename)
    plt.close()

    print(f'Heat map saved to {filename}')

def print_q_table(application):
    q_table_data = []
    for state, actions in application.q_table.items():
        for action, q_value in actions.items():
            q_table_data.append([state, action, q_value])

    headers = ["State (Node ID)", "Action (Next Node ID)", "Q-Value"]
    print(f'\n[Node_ID={application.node.node_id}] Q-Table:')
    print(tabulate(q_table_data, headers=headers, tablefmt="grid"))