import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

# Define positions for the nodes in the network
positions = {
    'tx': (0 * 2, 3 * 2),
    'rx': (7 * 2, 3 * 2)
}

for i in range(36):
    row, col = divmod(i, 6)
    positions[f'i{i}'] = ((col + 1) * 2, row * 2)

def plot_network(path, processed_functions, functions_sequence, episode, nodes, neighbors, node_status, node_functions):
    """Plots the network graph showing the path, applied functions, and missing functions, with progressive colors and active node highlight."""
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(node1, node2) for node1 in neighbors for node2 in neighbors[node1]])

    # Base color for nodes (green if alive, gray if dead)
    node_colors = ['green' if node_status.get(node, True) else 'gray' for node in nodes]
    node_labels = {node: f'{node}\nFunc: {node_functions.get(node, "")}' for node in nodes}

    nx.draw(G, pos=positions, labels=node_labels, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False)

    # Create a colormap to vary the color of edges as the agent moves
    cmap = cm.get_cmap('plasma')

    total_hops_in_path = len(path) - 1
    min_color = 0.4
    max_color = 1.0

    applied_functions = processed_functions
    missing_functions = [func for func in functions_sequence if func not in processed_functions]

    applied_text = "Applied functions: " + ", ".join(applied_functions) if applied_functions else "No function applied"
    missing_text = "Missing functions: " + ", ".join(missing_functions) if missing_functions else "All functions applied"

    # Create a legend with the applied and missing functions
    handles = [
        plt.Line2D([0], [0], color='white', label=applied_text),
        plt.Line2D([0], [0], color='white', label=missing_text)
    ]
    plt.legend(handles=handles, loc='upper right', fontsize=8)

    # Draw each hop with progressively lighter colors
    for i in range(total_hops_in_path):
        edges_in_path = [(path[i], path[i + 1])]

        # Compute the color for the current hop
        color_index = min_color + (i / total_hops_in_path) * (max_color - min_color)
        color = cmap(color_index)

        # Highlight the destination node of the hop (next_node)
        destination_node = path[i + 1]  # Destination of the hop
        node_colors = ['red' if node == destination_node else 'green' if node_status.get(node, True) else 'gray' for node in nodes]

        nx.draw(G, pos=positions, labels=node_labels, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False)

        nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color=[color], width=3, arrows=True)

        plt.title(f'Episode {episode} - Path: {" -> ".join(path[:i + 2])}')
        plt.pause(0.001)

    plt.pause(0.001)
